# DART contact equation port

`../../contact_dynamics.py` ports the deterministic, zero-start boxed PGS and
zero-restitution point-contact equations. No DART collision detector is used.

Sources checked:

- https://github.com/dartsim/dart/blob/v6.16.0/dart/constraint/ContactConstraint.cpp
- https://github.com/dartsim/dart/blob/v6.16.0/dart/constraint/PgsBoxedLcpSolver.cpp
- https://github.com/dartsim/dart/blob/v6.16.0/dart/constraint/PgsBoxedLcpSolver.hpp

The included BSD notice applies to the ported DART portions.

`pgs_reference.json` contains outputs from the official `dartpy==6.16.7` wheel's
C++ PgsBoxedLcpSolver. Its Python binding lacks a constructor; validation used a
small C++ wrapper compiled against the release headers and linked to the wheel.
Each case starts from zero with the default solver options, including 30 sweeps
and deterministic ordering. The test regenerates matrices from the recorded
NumPy RNG seed. A separate 300-case check matched native impulses to 1.56e-15;
4 native cases reached the iteration limit, so convergence is recorded rather
than inferred from a finite result.
