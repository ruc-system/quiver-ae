# Repository Guidelines

## Project Structure & Module Organization

FusionAnns splits CLI entry points in `app/`, core libraries under `src/`, and build outputs in `bin/` and `indices/`. `src/common/` hosts shared types and I/O helpers, `src/index/` implements the offline builder pipeline, and `src/online/` holds CUDA kernels plus the future query runtime. Sample datasets sit in `data/sift/`; keep generated indices inside `indices/` so existing paths resolve during development.

## Build, Test, and Development Commands

- `xmake`: Configure toolchains, resolve `faiss`/`openblas`, and build all targets in debug mode.
- `xmake f -m release`: Reconfigure for a release build; adjust CUDA `gencodes` in `xmake.lua` to match your GPU.
- `xmake run build_index`: Execute the builder from the repo root; artifacts land in `indices/`.
- `xmake run query_server`: Execute the online query server; stats land in `indices/query_stats.csv`.

## Coding Style & Naming Conventions

Stick to C++17 with CUDA extensions and project-relative includes. Use two-space indentation, same-line braces, PascalCase for classes (`FusionAnnsBuilder`), trailing underscores for members, and snake_case for free functions (`create_optimized_layout`). Keep filenames lowercase with underscores. Mirror the bilingual commenting style already present, and run `clang-format -style=LLVM` (or your local equivalent) before sending changes.

## Testing Guidelines

Automated suites are not yet defined. When adding functionality, supply focused repro drivers under `app/` or add GoogleTest suites in a `tests/` folder that mirrors module names (e.g., `common_io_test.cpp`). Validate builds with `xmake run build_index` on a small slice of `data/sift` and confirm new artifacts in `indices/`. Document any datasets or parameters needed to reproduce results in your PR notes.

## Commit & Pull Request Guidelines

This snapshot ships without Git history, so follow Conventional Commits (`feat:`, `fix:`, `docs:`) with subjects under 72 characters and imperative voice. Squash WIP commits before opening a PR. Provide context in the description, link tracking issues, call out affected modules, and list verification commands/logs—especially for GPU or IO changes.

## Data & Configuration Notes

Keep large raw datasets and generated binaries out of version control; rely on the existing ignore rules for `indices/`. Update `xmake.lua` if you require different CUDA architectures or dependencies, and note external system packages (`faiss`, `openblas`, CUDA toolkit) so teammates can reproduce your environment.
