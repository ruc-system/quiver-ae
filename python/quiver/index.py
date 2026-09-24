from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt


class QuiverError(RuntimeError):
    """Raised when Quiver cannot execute or return valid search results."""


_DATA_TYPES = {
    "float": (np.dtype("<f4"), ".fbin"),
    "float32": (np.dtype("<f4"), ".fbin"),
    "uint8": (np.dtype("u1"), ".u8bin"),
    "int8": (np.dtype("i1"), ".i8bin"),
}


class IndexQuiver:
    """Search a Quiver index with NumPy arrays.

    The Python API invokes the compiled ``quiver_search`` engine. Passing more
    than one PCI address enables Quiver's striped multi-SSD SPDK backend.
    Without ``ssds`` or ``ssd_list_file``, Quiver uses its memory backend.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str],
        *,
        data_type: str = "float32",
        ssds: Sequence[str] | None = None,
        ssd_list_file: str | os.PathLike[str] | None = None,
        spdk_base_lba: int | None = None,
        binary: str | os.PathLike[str] | None = None,
        memory_backend: str = "heap",
        num_blocks: int = 756,
        queries_per_block: int = 1,
        pipe_width: int = 2,
        poll_threads: int = 6,
        runners_per_ssd: int = 1,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        if not self.index_dir.is_dir():
            raise ValueError(f"index_dir is not a directory: {self.index_dir}")

        normalized_type = data_type.lower()
        if normalized_type not in _DATA_TYPES:
            raise ValueError("data_type must be float32, float, uint8, or int8")
        self.data_type = "float" if normalized_type == "float32" else normalized_type
        self._numpy_dtype, self._query_suffix = _DATA_TYPES[normalized_type]

        if ssds and ssd_list_file:
            raise ValueError("pass either ssds or ssd_list_file, not both")
        self.ssds = tuple(str(ssd).strip() for ssd in (ssds or ()))
        if any(not ssd for ssd in self.ssds):
            raise ValueError("SSD PCI addresses must not be empty")
        self.ssd_list_file = (
            Path(ssd_list_file).expanduser().resolve() if ssd_list_file else None
        )
        if self.ssd_list_file and not self.ssd_list_file.is_file():
            raise ValueError(f"ssd_list_file does not exist: {self.ssd_list_file}")
        if spdk_base_lba is not None and spdk_base_lba < 0:
            raise ValueError("spdk_base_lba must be non-negative")
        self.spdk_base_lba = spdk_base_lba

        if memory_backend not in {"heap", "mmap"}:
            raise ValueError("memory_backend must be heap or mmap")
        self.memory_backend = memory_backend
        self.binary = self._find_binary(binary)
        self.num_blocks = self._positive(num_blocks, "num_blocks")
        self.queries_per_block = self._positive(
            queries_per_block, "queries_per_block"
        )
        self.pipe_width = self._positive(pipe_width, "pipe_width")
        self.poll_threads = self._positive(poll_threads, "poll_threads")
        self.runners_per_ssd = self._positive(runners_per_ssd, "runners_per_ssd")
        self.env = dict(env or {})

    @property
    def num_ssds(self) -> int:
        if self.ssds:
            return len(self.ssds)
        if not self.ssd_list_file:
            return 0
        return len(
            [
                line
                for line in self.ssd_list_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        )

    def search(
        self,
        queries: npt.ArrayLike,
        topk: int = 10,
        *,
        ef_search: int = 50,
        early_exit_policy: str = "gpruning",
        timeout: float | None = None,
        extra_args: Sequence[str] = (),
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """Return ``(ids, distances)`` with shape ``(len(queries), topk)``."""
        topk = self._positive(topk, "topk")
        ef_search = self._positive(ef_search, "ef_search")
        if early_exit_policy not in {"gpruning", "none"}:
            raise ValueError("early_exit_policy must be gpruning or none")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")

        array = np.asarray(queries)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("queries must be a non-empty 1D or 2D array")
        array = np.ascontiguousarray(array, dtype=self._numpy_dtype)

        with tempfile.TemporaryDirectory(prefix="quiver-python-") as temp_dir:
            work = Path(temp_dir)
            query_path = work / f"queries{self._query_suffix}"
            result_prefix = work / "result"
            self._write_bin(query_path, array)
            ssd_list = self._materialize_ssd_list(work)

            command = [
                str(self.binary),
                "--index-dir",
                str(self.index_dir),
                "--query",
                str(query_path),
                "--data-type",
                self.data_type,
                "--topk",
                str(topk),
                "--ef-search",
                str(ef_search),
                "--repeat",
                "1",
                "--num-blocks",
                str(self.num_blocks),
                "--queries-per-block",
                str(self.queries_per_block),
                "--pipe-width",
                str(self.pipe_width),
                "--poll-threads",
                str(self.poll_threads),
                "--runners-per-ssd",
                str(self.runners_per_ssd),
                "--memory-backend",
                self.memory_backend,
                "--early-exit-policy",
                early_exit_policy,
                "--result-prefix",
                str(result_prefix),
            ]
            if ssd_list is not None:
                command.extend(["--ssd-list-file", str(ssd_list)])
            command.extend(str(arg) for arg in extra_args)

            process_env = os.environ.copy()
            process_env.update(self.env)
            if self.spdk_base_lba is not None:
                process_env["SPDK_BASE_LBA"] = str(self.spdk_base_lba)
            try:
                completed = subprocess.run(
                    command,
                    env=process_env,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise QuiverError(
                    f"quiver_search exceeded the {timeout:g} second timeout"
                ) from error
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise QuiverError(
                    f"quiver_search exited with code {completed.returncode}: {detail}"
                )

            ids = self._read_bin(result_prefix.with_name("result_ids.bin"), np.int32)
            distances = self._read_bin(
                result_prefix.with_name("result_distances.bin"), np.float32
            )
            expected = (array.shape[0], topk)
            if ids.shape != expected or distances.shape != expected:
                raise QuiverError(
                    f"unexpected result shape: ids={ids.shape}, "
                    f"distances={distances.shape}, expected={expected}"
                )
            return ids, distances

    def _materialize_ssd_list(self, work: Path) -> Path | None:
        if self.ssd_list_file:
            return self.ssd_list_file
        if not self.ssds:
            return None
        path = work / "ssds.txt"
        path.write_text("".join(f"{ssd}\n" for ssd in self.ssds), encoding="ascii")
        return path

    @staticmethod
    def _write_bin(path: Path, values: npt.NDArray[np.generic]) -> None:
        with path.open("wb") as output:
            output.write(struct.pack("<ii", values.shape[0], values.shape[1]))
            output.write(values.tobytes(order="C"))

    @staticmethod
    def _read_bin(path: Path, dtype: npt.DTypeLike) -> npt.NDArray[np.generic]:
        if not path.is_file():
            raise QuiverError(f"quiver_search did not create {path.name}")
        with path.open("rb") as source:
            header = source.read(8)
            if len(header) != 8:
                raise QuiverError(f"invalid result header in {path}")
            count, dims = struct.unpack("<ii", header)
            if count < 0 or dims < 0:
                raise QuiverError(f"invalid result dimensions in {path}")
            values = np.fromfile(source, dtype=np.dtype(dtype).newbyteorder("<"))
        if values.size != count * dims:
            raise QuiverError(f"truncated result data in {path}")
        return values.reshape(count, dims)

    @staticmethod
    def _positive(value: int, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _find_binary(binary: str | os.PathLike[str] | None) -> Path:
        candidates: list[Path] = []
        if binary:
            candidates.append(Path(binary).expanduser())
        elif os.environ.get("QUIVER_BINARY"):
            candidates.append(Path(os.environ["QUIVER_BINARY"]).expanduser())
        else:
            executable = shutil.which("quiver_search")
            if executable:
                candidates.append(Path(executable))
            root = Path(__file__).resolve().parents[2]
            candidates.extend(
                [root / "build" / "bin" / "quiver_search", root / "bin" / "quiver_search"]
            )

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return resolved
        searched = ", ".join(str(path) for path in candidates) or "PATH"
        raise QuiverError(f"quiver_search executable not found (searched: {searched})")
