# Third-Party Licenses

This C++ project (`code/routing_cpp/`) vendors several third-party libraries
that ship inside the `external/` and `include/` subtrees. This file enumerates
each one with its license, copyright holder(s), and the location of the
upstream license text(s) that travel with this repository.

The full license texts are preserved in their original locations so that the
redistribution requirements (notably the common BSD/MIT obligation to retain
the copyright notice and disclaimer) are satisfied automatically as long as
this directory is distributed in source form. If a binary derived from this
code is ever distributed externally (for example, a compiled `Autowiring_CPP`
executable bundled with a release), the relevant license texts must be
reproduced in that distribution per the **BSD-3 source-distribution clause
(clause 1)** and **binary-distribution clause (clause 2)** and the
equivalent obligations in the other licenses below.

**License-mix summary.** Everything **compiled into and linked by the
`Autowiring_CPP` binary** is under permissive licenses (BSD 3-Clause / MIT /
Apache 2.0 OR MIT / Boost 1.0). The vendored source trees additionally
include a handful of utility files (build-system helpers, documentation
generators, sample programs) under different licenses — most notably:
- one **GPL-3.0** Catch2 documentation utility script (see Catch2 section),
- a few **CC-BY-SA-3.0** licensed sample/example files (see pcg-cpp),

neither of which is part of the runtime build.

## Direct dependency

### dual-annealing
- **Upstream**: <https://github.com/twesterhout/dual-annealing>
- **License**: BSD 3-Clause
- **Copyright**: 2018–2019, Tom Westerhout
  (the bundled `LICENSE` text states 2019; per-file headers in
  `external/dual-annealing/include/config.hpp` and
  `external/dual-annealing/examples/rastrigin.cpp` include 2018).
- **License file in repo**: `external/dual-annealing/LICENSE`
- **Role**: provides the generalized simulated annealing (GSA) optimizer
  invoked from `src/main.cpp`. This is the implementation cited in the paper
  for the routing-optimization computation.

## Vendored inside `external/dual-annealing/third_party/`

These are dependencies of `dual-annealing` itself, kept unmodified inside its
`third_party/` directory tree.

### Catch2
- **Upstream**: <https://github.com/catchorg/Catch2>
- **Framework license**: Boost Software License 1.0 (`LICENSE.txt`).
- **Framework copyright**: Two Blue Cubes Ltd (`single_include/catch2/catch.hpp`),
  with per-file additional copyrights for individual reporters
  (Martin Moene in `catch_reporter_tap.hpp`,
  Justin R. Wilson in `catch_reporter_automake.hpp`, etc.).
- **Bundled CMake helpers** under `contrib/`:
  - `Catch.cmake`, `CatchAddTests.cmake`, `ParseAndAddCatchTests.cmake` —
    **BSD 3-Clause**, originating from upstream CMake/Kitware
    (`# Distributed under the OSI-approved BSD 3-Clause License`).
  - `gdbinit`, `lldbinit` — debugger configuration files, no explicit
    license header.
- **Bundled CMake-codecov helpers** under `CMake/`:
  - `FindGcov.cmake`, `Findcodecov.cmake`, `llvm-cov-wrapper` — distributed
    by RWTH Aachen with separate copyright. The license text these helpers
    point to is **not present** in this vendored snapshot; if you redistribute
    a binary derived from this tree, either include the full upstream license
    or strip these helpers (they are not used by `Autowiring_CPP`).
- **Bundled documentation utility** under `scripts/`:
  - `updateDocumentToC.py` — **GPL-3.0** (based on Sebastian Raschka's
    *markdown-toclify*). This script is a developer-only utility for
    regenerating the markdown table of contents in Catch2's docs; it is not
    invoked by any build target. **If a downstream redistribution must avoid
    GPL'd files, delete this single script before packaging** — its absence
    has no effect on the build or the binary.
- **Embedded library**: a copy of **Clara v1.1.5** (Boost Software License 1.0,
  © 2017 Two Blue Cubes Ltd) ships at `third_party/clara.hpp` and
  `include/external/clara.hpp`.
- **License file**: `external/dual-annealing/third_party/Catch2/LICENSE.txt`
- **Role**: unit-test framework used by `dual-annealing`'s own tests **and
  by the M5-introduced `sinic_tests` target** under `tests/` (Catch2 v2
  single-header from `single_include/catch2/catch.hpp`). The `Autowiring_CPP`
  release binary itself does NOT link Catch2 — the test target is gated
  by the `SINIC_BUILD_TESTS` CMake option.
- **M5 vendored patch (2026-05-14)**: `single_include/catch2/catch.hpp` line
  7628-7647 was patched so that `CATCH_TRAP()` falls back to
  `__builtin_trap()` on Apple Silicon (`defined(__aarch64__) ||
  defined(__arm64__)`) instead of `__asm__("int $3\n")`. The x86 `int $3`
  inline assembly fails to assemble on arm64 (clang errors
  "unrecognized instruction mnemonic"). Same class of platform-compat
  patch as the existing `lbfgs-cpp` arm64 adjustments documented in the
  parent README. Linux / x86_64 paths are untouched. The Boost Software
  License 1.0 permits derivative works as long as the license stays
  intact — the patch preserves the framework copyright header and adds
  only the conditional fallback.

### gsl-lite
- **Upstream**: <https://github.com/gsl-lite/gsl-lite>
- **Headers license**: MIT for the `gsl-lite` headers themselves
  (`include/gsl/gsl-lite.hpp` etc.).
- **Headers copyright**: 2015–2018 Martin Moene; 2015–2018 Microsoft Corporation
  (per per-file headers; the bundled `LICENSE` text file itself only states
  2015).
- **Helper scripts under `script/`** (5 files; mixed licenses):
  - `upload-conan.py`, `create-cov-rpt.py`, `create-vcpkg.py` —
    **Boost Software License 1.0** (© 2019 Martin Moene).
  - `install-gsl-pkg.py`, `update-version.py` — **MIT** (© 2015–2018
    Martin Moene).
- **Embedded test framework**: `test/lest_cpp03.hpp` — **Boost Software
  License 1.0** (the *lest* C++ test framework, vendored for gsl-lite's own
  test suite).
- **License file**: `external/dual-annealing/third_party/gsl-lite/LICENSE`
- **Role**: header-only Microsoft *Guidelines Support Library*; used inside
  `dual-annealing` for `gsl::span`, `Expects` / `Ensures`, etc.

### lbfgs-cpp
- **Upstream**: <https://github.com/twesterhout/lbfgs-cpp>
- **License**: BSD 3-Clause
- **Copyright**: 2019, Tom Westerhout
- **License file**: `external/dual-annealing/third_party/lbfgs-cpp/LICENSE`
- **Role**: provides the L-BFGS local-search refinement step invoked via the
  5-argument form of `dual_annealing::minimize` in `src/main.cpp`.
  `tcm::lbfgs::lbfgs_param_t` is constructed at `src/main.cpp:555`, and the
  GSA chain at `external/dual-annealing/include/chain.hpp` includes
  `<lbfgs/lbfgs.hpp>` (line 9) and calls `tcm::lbfgs::minimize` (line 378).
  Transitively linked through `dual_annealing`. *Note*: the manuscript's
  supplemental document states this gradient-based refinement is
  algorithmically disabled in the published results; that statement concerns
  runtime behaviour, not the dependency relationship documented here.

### pcg-cpp
- **Upstream**: <https://github.com/imneme/pcg-cpp>
- **Library license**: Apache License 2.0 **OR** MIT (dual-licensed; either
  may be chosen by the downstream user).
- **Library copyright**: 2014–2019 Melissa O'Neill and PCG Project
  contributors (per per-file headers in `include/pcg_random.hpp`; the
  bundled `LICENSE-MIT.txt` text itself states 2014–2017).
- **License files**:
  - `external/dual-annealing/third_party/pcg-cpp/LICENSE-APACHE.txt`
  - `external/dual-annealing/third_party/pcg-cpp/LICENSE-MIT.txt`
  - `external/dual-annealing/third_party/pcg-cpp/LICENSE.spdx`
- **Sample program with separate license**:
  - `sample/cppref-sample.cpp` — **CC-BY-SA-3.0 OR Apache-2.0 OR MIT**
    (© 2012–2014 cppreference.com Contributors; © 2014–2017 Melissa O'Neill
    and PCG Project contributors). Copyleft-style CC-BY-SA-3.0 is one of
    three options; the file itself does not require Apache or MIT alone.
    This file is a documentation example; not built by the Autowiring_CPP
    target. Strip from packaging if CC-BY-SA-3.0 is undesirable.
- **Role**: PCG random number generator (`pcg_random.hpp`). Included
  directly by `src/main.cpp:22`, and additionally pulled in transitively
  through `external/dual-annealing/include/chain.hpp`.

## Vendored inside `external/dual-annealing/third_party/lbfgs-cpp/third_party/`

These are second-level transitive dependencies, included by `lbfgs-cpp`.

### Catch2 (second copy)
- Same license stack and same embedded files as the first Catch2 copy above:
  Boost-1.0 framework, BSD-3 `contrib/` CMake helpers
  (`Catch.cmake`, `CatchAddTests.cmake`, `ParseAndAddCatchTests.cmake`),
  RWTH-Aachen `CMake/` codecov helpers, GPL-3.0
  `scripts/updateDocumentToC.py` developer utility, embedded
  Boost-1.0 Clara v1.1.5.
- **License file**: `external/dual-annealing/third_party/lbfgs-cpp/third_party/Catch2/LICENSE.txt`

### gsl-lite (second copy)
- Same license stack as the first `gsl-lite` copy above
  (MIT for the headers, mixed Boost-1.0 / MIT for the scripts,
  Boost-1.0 for the embedded `test/lest_cpp03.hpp`).
- **License file**: `external/dual-annealing/third_party/lbfgs-cpp/third_party/gsl-lite/LICENSE`

### SG14
- **Upstream**: <https://github.com/WG21-SG14/SG14>
- **License**: declared **per file**, predominantly Boost Software License 1.0
  (see the headers of `SG14/flat_map.h`, `SG14/inplace_function.h`,
  `SG14/slot_map.h`, `SG14/flat_set.h`, etc.).
  - One bundled file, `SG14/plf_colony.h`, is independently licensed under
    **zlib** by Matthew Bentley (`mattreecebentley@gmail.com`).
  - Two files — `SG14/algorithm_ext.h` and `SG14/ring.h` — carry **no
    license header at all** in this snapshot. Consult the upstream SG14
    repository before reusing those two files; the rest of SG14 is Boost-1.0.
- **License file**: *no top-level LICENSE file is shipped with SG14*; license
  declarations live in the per-file headers.
- **Location**: `external/dual-annealing/third_party/lbfgs-cpp/third_party/SG14/`
- **Role**: vendored under `lbfgs-cpp/third_party/`. SG14's own
  `CMakeLists.txt` and tests under `SG14_test/` reference it internally,
  but **none of the `Autowiring_CPP`, `dual_annealing`, or `lbfgs-cpp`
  build targets** add it as a subdirectory or link against it
  (`lbfgs-cpp/CMakeLists.txt` adds `gsl-lite` and `Catch2` only). Not
  linked into `Autowiring_CPP`.

## Header-only vendor under `include/`

### nlohmann/json
- **Upstream**: <https://github.com/nlohmann/json> (version 3.11.3)
- **License**: MIT overall (declared via `SPDX-License-Identifier: MIT` at
  the top of the header). The header bundles small amounts of code derived
  from third-party sources, identified inline by SPDX-style tags:
  - **Apache License 2.0** for embedded Google Abseil-derived code
    (declared near `json.hpp:3190`).
  - Additional inline attributions for code derived from Hedley
    (Evan Nemerson, near `json.hpp:341`), the UTF-8 decoder by
    Bjorn Hoehrmann (near `json.hpp:17600`), and the **Grisu2** float-to-string
    routines derived from Florian Loitsch's algorithm.
- **Copyright**: 2013–2025 Niels Lohmann; with additional embedded-code
  copyrights to **Evan Nemerson**, **The Abseil Authors**, **Bjorn Hoehrmann**,
  and **Florian Loitsch** declared via SPDX `FileCopyrightText` headers
  inside `json.hpp`.
- **Location**: `include/nlohmann/json.hpp` (single-header build)
- **Role**: vendored locally for JSON serialization in `src/main.cpp`.
  *Caveat*: at the time of this writing, `src/main.cpp:27` includes
  nlohmann/json via an absolute external path (a leftover from a WSL
  build environment) rather than the vendored copy here. The vendored copy
  is still distributed with this directory, so the MIT/Apache attribution
  obligations apply regardless. See the project README at
  `sinic_main/README.md` ("Known issues / debt" section) for the cleanup
  TODO.

## Notes on compliance

- The license texts cited above are present in this repository under their
  upstream locations. **Do not remove them** when distributing this directory.
- **Do not remove the per-file copyright headers** (where present) in any of
  the vendored source files — `tsallis_distribution.hpp`, `pcg_random.hpp`,
  the SG14 headers (those that have headers), `json.hpp`, etc.
  Some files use only `#pragma once` without a header (e.g. `chain.hpp`); for
  those, the controlling license is the one stated in the parent project's
  top-level `LICENSE` file.
- The **BSD 3-Clause** license — applicable to **dual-annealing** and
  **lbfgs-cpp** (both © Tom Westerhout) — additionally forbids using the
  copyright holder's name to endorse or promote products derived from this
  software without prior written permission.
- The **Apache License 2.0** — applicable to **pcg-cpp** (when chosen out of
  its dual-license offer), to the embedded Abseil-derived code in
  `nlohmann/json.hpp`, and as one option for `pcg-cpp/sample/cppref-sample.cpp`
  — imposes its own trademark and patent-grant clauses (Sections 3 and 6 of
  the license).
- **Copyleft caveats** for downstream packaging:
  - `external/dual-annealing/third_party/Catch2/scripts/updateDocumentToC.py`
    (and its sibling in the second Catch2 copy) is **GPL-3.0**. It is a
    developer-only doc utility, never executed by the build, and can be
    safely deleted before redistribution if GPL'd files are unwanted.
  - `external/dual-annealing/third_party/pcg-cpp/sample/cppref-sample.cpp`
    can be redistributed under Apache-2.0 or MIT; only the **CC-BY-SA-3.0**
    third option is share-alike.
- For binary distributions, reproduce the relevant license notices in
  accompanying documentation or release notes, and verify that the
  RWTH-Aachen CMake-codecov helpers in `Catch2/CMake/` are accompanied by
  their full upstream license (or stripped from the package — they are not
  used by `Autowiring_CPP`).
