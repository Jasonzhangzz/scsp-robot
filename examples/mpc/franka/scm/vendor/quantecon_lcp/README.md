# QuantEcon Lemke LCP solver (vendored subset)

The three Python implementation modules are unmodified copies from the
official QuantEcon.py `v0.11.2` release, with its MIT license retained in
`LICENSE`. Only the package initializer is local; this is a minimal solver
subset, not a replacement for the complete QuantEcon package.

Official sources:

- https://github.com/QuantEcon/QuantEcon.py/blob/v0.11.2/quantecon/optimize/lcp_lemke.py
- https://github.com/QuantEcon/QuantEcon.py/blob/v0.11.2/quantecon/optimize/pivoting.py
- https://github.com/QuantEcon/QuantEcon.py/blob/v0.11.2/quantecon/optimize/linprog_simplex.py
- https://github.com/QuantEcon/QuantEcon.py/blob/v0.11.2/LICENSE

The solver uses Lemke's algorithm with lexicographic pivoting. It requires
NumPy and Numba (already present in the benchmark's pi0 environment).
The benchmark wrapper certifies nonnegative impulses, nonnegative slacks
and complementarity in the original, unscaled LCP independently of the
library's termination flag. It never falls back silently to the local solver.

Direct package/archive downloads failed in this environment; the exact
release sources were obtained through the web connector from the official
raw GitHub URLs, with all source lines retained.
