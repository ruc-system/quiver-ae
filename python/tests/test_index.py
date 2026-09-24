from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest

from quiver import IndexQuiver, QuiverError


FAKE_BINARY = r'''#!/usr/bin/env python3
import numpy as np
import struct
import sys
from pathlib import Path

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

query = Path(value("--query"))
prefix = Path(value("--result-prefix"))
topk = int(value("--topk"))
with query.open("rb") as source:
    count, dims = struct.unpack("<ii", source.read(8))
    source.read()

ssd_flag = "--ssd-list-file" in args
if ssd_flag:
    ssds = [line for line in Path(value("--ssd-list-file")).read_text().splitlines() if line]
    assert ssds == ["0000:01:00.0", "0000:02:00.0"]

ids = np.arange(count * topk, dtype="<i4")
distances = np.arange(count * topk, dtype="<f4") / 10
for suffix, array in (("_ids.bin", ids), ("_distances.bin", distances)):
    with Path(str(prefix) + suffix).open("wb") as output:
        output.write(struct.pack("<ii", count, topk))
        output.write(array.tobytes())
'''


@pytest.fixture()
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "quiver_search"
    path.write_text(FAKE_BINARY, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_numpy_search_and_multi_ssd(tmp_path: Path, fake_binary: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index = IndexQuiver(
        index_dir,
        data_type="float32",
        ssds=["0000:01:00.0", "0000:02:00.0"],
        binary=fake_binary,
    )

    ids, distances = index.search(np.ones((3, 4), dtype=np.float64), topk=2)

    assert index.num_ssds == 2
    assert ids.dtype == np.dtype("<i4")
    assert distances.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(ids, [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_allclose(distances, [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])


def test_rejects_conflicting_ssd_configuration(
    tmp_path: Path, fake_binary: Path
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    ssd_list = tmp_path / "ssds.txt"
    ssd_list.write_text("0000:01:00.0\n", encoding="ascii")

    with pytest.raises(ValueError, match="either ssds or ssd_list_file"):
        IndexQuiver(
            index_dir,
            ssds=["0000:01:00.0"],
            ssd_list_file=ssd_list,
            binary=fake_binary,
        )


def test_reports_engine_failure(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    binary = tmp_path / "quiver_search"
    binary.write_text(
        "#!/usr/bin/env sh\necho 'index metadata is invalid' >&2\nexit 7\n",
        encoding="ascii",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    index = IndexQuiver(index_dir, binary=binary)

    with pytest.raises(QuiverError, match="code 7: index metadata is invalid"):
        index.search(np.ones((1, 4), dtype=np.float32))


def test_reports_engine_timeout(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    binary = tmp_path / "quiver_search"
    binary.write_text(
        "#!/usr/bin/env python3\nwhile True:\n    pass\n", encoding="ascii"
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    index = IndexQuiver(index_dir, binary=binary)

    with pytest.raises(QuiverError, match="exceeded the 0.05 second timeout"):
        index.search(np.ones((1, 4), dtype=np.float32), timeout=0.05)
