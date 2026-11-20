# Native C++ validation — November 19, 2026

The C++ companion was built and run against the released OpenCV **5.1.0** and
ONNX Runtime **1.24.3**, using native CPU inference. Normal builds and runs do
not require Python. The [validation manifest](validation.json) records the
environment, scope, and outcomes; the [release check](release-verification.json)
records the latest stable OpenCV release at validation time.

| Platform | Compiler | CTest | Benchmark rows | Synthetic rows |
| --- | --- | --- | --- | --- |
| macOS 26.6.2, Apple M5 Pro, arm64 | Apple Clang 21.0.0 | 10/10 | 180 | 216 |
| Ubuntu 24.04, Core i7-6850K, x86-64 | GCC 13.3.0 | 10/10 | 180 | 216 |

Both Release builds use C++17 and `-Wall -Wextra -Wpedantic -Werror`.
CTest includes 13 geometry regressions, all six matcher/estimator routes,
the notebook demonstrations, a synthetic study, and CLI error/output-reuse
checks. Tests run outside the source directory. Seven additional macOS CLI
checks cover empty neural features, help, missing models, and saved ECC candidates.

Each benchmark has ten image pairs, three timing trials, and six
matcher/estimator combinations. Two pairs are validation data; eight are
evaluation data. With MAGSAC++, SIFT succeeds on 6/8 evaluation pairs, ALIKED-NN
on 6/8, and LightGlue on 7/8, on both platforms. Success requires acceptance
and independent median overlap-grid error below three pixels. Repeated trials
are not additional independent image pairs. The synthetic study has 12 seeds,
three noise levels, three thresholds, and two estimators.

The [Python CPU comparison](python_cpu_parity.json) passes five shared geometry
cases and twelve real pipelines: three methods on box, Graf6, Leuven6, and form
images. Shared-point homographies, inlier masks, projected corners, warped PNGs,
and validity masks agree exactly; median residual differences are below
`2e-14`. All real-pair acceptance decisions agree. Match precision is at least
99.898% and recall is 100%, using the explicit tolerances retained in that file.
The [graph export checks](../../cpp/models/export_validation.json) contain 132
comparisons against the original PyTorch layers; worst absolute error is
0.0001345 against a 0.0002 limit.

On Graf6, Python CPU and C++ CPU both accept an estimate with approximately
11.97 px independent error. It fails the strict accuracy criterion. The
article's stored Python GPU run has 1.76 px error on this pair. LightGlue's
upstream CPU/GPU pruning rules differ, and floating-point differences can also
change sparse decisions. Demo validation follows the notebook's acceptance
check while reporting independent accuracy separately. The benchmark keeps
its stricter criterion and retains failed estimates.

OpenCV is requested to use one thread and ONNX Runtime uses four intra-op
threads and one inter-op thread. macOS GCD's `getNumThreads()` reports pool
capacity even when OpenCV bypasses parallel execution after `setNumThreads(1)`;
reports retain both requested and reported values. CPU timing distributions
are separate from the article's GPU measurements. See the raw reports,
CSV files, build information, and logs under [macos-arm64](macos-arm64/) and
[linux-x64](linux-x64/).

The CMake downloader verified all 35 sample files in a fresh asset directory.
All eleven bundled ONNX files passed SHA-256 verification. The native synthetic
generator uses `std::mt19937`; Python uses NumPy's generator, so individual
random draws differ. Shared point fixtures establish geometric equivalence
without depending on the two generators.
