
# hydropack

hydropack is a library for modeling subglacial hydrology beneath glaciers and ice sheets using the finite element method.
The physics includes both distributed linked-cavity drainage and discrete Röthlisberger channels, following the GlaDS formulation (Werder et al., 2013).
Each constitutive relation (closure laws, channel growth, cavity opening, etc.) is exposed as a swappable function, so you can test alternative physics without modifying the solver.

hydropack is built on [Firedrake](https://www.firedrakeproject.org) and designed to couple with [icepack](https://github.com/icepack/icepack) for ice-flow simulations.

### Getting started

Once you have a working Firedrake installation:

```bash
git clone https://github.com/hoffmaao/hydropack.git
cd hydropack
pip install -e .
```

The directory `notebooks/` contains tutorial notebooks that walk through the main features:

**01 — Greenland outlet glacier**: distributed sheet drainage, steady-state effective pressure

**02 — Channels**: how Röthlisberger channels emerge at higher melt rates

**03 — Moulins**: concentrated moulin input and channel network structure

**04 — Mountain glacier**: valley geometry, overdeepenings, and glaciohydraulic supercooling

### Validation

hydropack has been validated against the [SHMIP](https://shmip.bitbucket.io/) subglacial hydrology model intercomparison (Suites A–F), comparing with the Werder GlaDS MATLAB reference.
The SHMIP test scripts are maintained in a [separate repository](https://github.com/hoffmaao/shmip).

