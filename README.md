# Hydropack

**Hydropack** is a Firedrake-based finite-element model for simulating
subglacial hydrology and basal processes beneath glaciers and ice sheets.
It provides a modular solver infrastructure for both distributed and
channelized drainage systems and is designed for seamless coupling with
the [Icepack](https://github.com/icepack/icepack) ice-flow model.

---

## Overview

Hydropack solves the coupled sheet–channel hydrology equations using
Firedrake’s automated finite element framework.  The current release
implements:

- **Distributed sheet** evolution (`h`) following the GlaDS model  
- **Channelized system** evolution (`S`) for Röthlisberger channels  
- **Potential solver** (`φ`) using nonlinear Newton–Krylov methods  
- **Mesh tools** for synthetic geometries (e.g., SHMIP Suite E)  
- **Robust PETSc nonlinear solvers** with LU/GMRES back-ends  
- **Checkpointing & post-processing utilities**

Hydropack mirrors Icepack’s philosophy: equations are described in UFL,
discretized through Firedrake, and exposed to users via a high-level
Python interface for composing experiments.

---

## Relationship to Icepack

Hydropack is developed to provide the subglacial hydrology component of
future coupled ice–hydrology simulations with Icepack.  The two codes
share:

- the Firedrake infrastructure and data structures,  
- consistent mesh handling and checkpoint I/O, and  
- similar solver configuration interfaces.

Where Icepack advances ice velocity and geometry, Hydropack evolves the
subglacial water system that modulates basal friction and effective
pressure.

---

## Installation

Hydropack requires a working [Firedrake](https://www.firedrakeproject.org)
installation (PETSc ≥ 3.18).  Once Firedrake is installed:

```bash
git clone https://github.com/hoffmaao/hydropack.git
cd hydropack
pip install -e .
