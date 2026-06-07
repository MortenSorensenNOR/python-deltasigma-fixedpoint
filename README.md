python-deltasigma (fixed-point fork)
====================================

A fork of [`python-deltasigma`](https://github.com/ggventurini/python-deltasigma)
focused on **evaluating fixed-point hardware implementations** of delta-sigma
modulators.

The upstream package is a Python port of Richard Schreier's
[MATLAB Delta Sigma Toolbox](http://www.mathworks.com/matlabcentral/fileexchange/19-delta-sigma-toolbox)
and remains the place to look for the synthesis / mapping / scaling tooling.
This fork keeps that surface intact and adds the things needed to ask *"what
does the SNR look like once I pin every state, coefficient, and accumulator to
a given Q-format?"*

* **Origin (this fork):** [MortenSorensenNOR/python-deltasigma-fixedpoint](https://github.com/MortenSorensenNOR/python-deltasigma-fixedpoint)
* **Direct upstream:** [hpretl/python-deltasigma](https://github.com/hpretl/python-deltasigma)
* **Original project:** [ggventurini/python-deltasigma](https://github.com/ggventurini/python-deltasigma)

[![BSD 2 clause license](http://img.shields.io/badge/license-BSD-brightgreen.png)](LICENSE)

---

## What this fork adds

* **Fixed-point simulation backend** for `simulateDSM` — pass a
  `FixedPointConfig` (one `QFormat` each for state, coefficients, input, and
  the quantizer-input accumulator `y`) and the modulator inner loop runs in
  the requested Q-format instead of `float64`. Backed by the pure-Python
  [`fixedpoint`](https://pypi.org/project/fixedpoint/) library, so it is
  slower than the float Cython/BLAS backends — useful for targeted
  wordlength sweeps, not bulk Monte Carlo.
* **Hardware-form coefficient constraints** — snap coefficients to
  power-of-two or CSD forms with configurable margins, optionally per-group.
* **`constrain_with_compensation`** for CRFB and CRFF — snap the interstage
  gains `c_i` to a hardware-friendly form (po2 / CSD) while absorbing the
  resulting state-scaling into `a`, `b`, `c`, and resonator `g` coefficients.
  Because the snap is a similarity transform on the loop filter, the NTF
  and STF are preserved exactly. Supports a `zero_circle_margin` to keep
  resonator zeros away from the unit circle after compensation.
* **Modernization** — Python 3.10+, current numpy / scipy / matplotlib,
  `collections.abc` fixes, and removal of `np.float` / `np.int` aliases. The
  pre-fork codebase targeted a much older stack.

### Minimal fixed-point example

```python
import numpy as np
from deltasigma import synthesizeNTF, realizeNTF, mapABCD, simulateDSM
from deltasigma import FixedPointConfig, QFormat

ntf = synthesizeNTF(order=3, osr=64, opt=1)
a, g, b, c = realizeNTF(ntf, form="CRFB")
ABCD = mapABCD(a, g, b, c, form="CRFB")

# Q1.15 state and coefficients; Q3.13 on the accumulator before the quantizer.
cfg = FixedPointConfig(
    state=QFormat(signed=True, m=1, n=15, overflow="clamp", rounding="convergent"),
    coeff=QFormat(signed=True, m=1, n=15, overflow="clamp", rounding="convergent"),
    input=QFormat(signed=True, m=1, n=15, overflow="clamp", rounding="convergent"),
    y=QFormat(signed=True, m=3, n=13, overflow="clamp", rounding="convergent"),
)

N = 8192
u = 0.5 * np.sin(2 * np.pi * 5 / N * np.arange(N))
v, xn, xmax, y = simulateDSM(u, ABCD, fixedpoint=cfg)
```

Omitting `fixedpoint=` falls back to the unmodified float backends, so any
existing upstream script keeps working unchanged.

## Status

Upstream's fundamental functionality is intact (synthesis, realization,
scaling, simulation). The fixed-point backend and coefficient-constraint
utilities added in this fork are usable but newer — see the test suite
(`deltasigma/tests/test_simulateDSM_fixedpoint.py`,
`test_constrain_with_compensation.py`) for what is currently covered.

Upstream's secondary features (native quadrature modulator support, PIS, ESL)
are still incomplete; this fork does not address them. The upstream
[ROADMAP](https://github.com/ggventurini/python-deltasigma/blob/master/ROADMAP.md)
and [files.csv](files.csv) still describe their state.

## Install

Runs on Linux, macOS, and Windows. Requires Python 3.10+, recent numpy,
scipy, and matplotlib; the fixed-point backend additionally requires
[`fixedpoint`](https://pypi.org/project/fixedpoint/). **Cython** is strongly
recommended for the float backends — it gives ~100x faster simulations.

The supported install path is [uv](https://docs.astral.sh/uv/):

    uv pip install -e .

or, inside a checkout, drop into a uv-managed venv:

    uv venv
    uv pip install -e .

Plain `pip` works too if your environment is not PEP-668-managed.

The unmodified upstream package is still available from PyPI as
`deltasigma` (`pip install deltasigma`) but does not include the fixed-point
additions in this fork.

### Testing

The test suite uses `pytest`:

    uv run pytest deltasigma

## Documentation

* Upstream [package documentation](http://python-deltasigma.readthedocs.org/en/latest/)
  covers the synthesis / mapping / scaling APIs.
* The original MATLAB toolbox documentation is the deepest reference —
  see [DSToolbox.pdf](delsig/DSToolbox.pdf) and
  [OnePageStory.pdf](delsig/OnePageStory.pdf).
* Richard Schreier and Gabor C. Temes, *Understanding Delta-Sigma Data
  Converters*, Wiley-IEEE Press, 2004 (ISBN 978-0-471-46585-0) — chapters
  8–9 walk through the MATLAB toolbox, and the same observations apply
  here.
* The new fixed-point and coefficient-constraint modules document their
  semantics in module-level docstrings:
  `deltasigma/_fixedpoint_config.py`,
  `deltasigma/_simulateDSM_fixedpoint.py`,
  `deltasigma/_constrain_compensation.py`,
  `deltasigma/_fixedpoint_constraints.py`.

## Licensing and copyright notice

All original MATLAB code is Copyright (c) 2009, Richard Schreier. The Python
port is a derivative work distributed under the same license terms (BSD
2-Clause); see the `LICENSE` file.

This package contains some source code from `pydsm`, also based on the same
MATLAB toolbox. The `pydsm` package is copyright (c) 2012, Sergio Callegari.

When not otherwise specified, the upstream Python code is Copyright 2013,
Giuseppe Venturini and the python-deltasigma contributors.

The fixed-point simulation backend and coefficient-constraint utilities
added in this fork are Copyright (c) 2026, Morten Sørensen, and are
distributed under the same BSD 2-Clause terms as the rest of the project.

MATLAB is a registered trademark of The MathWorks, Inc.
