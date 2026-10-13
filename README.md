
# hydropack

hydropack is a library for modeling subglacial hydrology beneath glaciers and ice sheets using the finite element method.
The physics includes both distributed linked-cavity drainage and discrete Röthlisberger channels, following the formulation described by (Werder et al., 2013).
Each constitutive relation (closure laws, channel growth, cavity opening, etc.) is expressed as a swappable function, so you can test alternative physics without modifying the solver.

hydropack is built on [Firedrake](https://www.firedrakeproject.org) and designed to couple with [icepack](https://github.com/icepack/icepack) for coupled ice-flow-subglacial hydrology simulations.

### Getting started

Once you have a working Firedrake installation:

```bash
git clone https://github.com/hoffmaao/hydropack.git
cd hydropack
pip install -e .
```

The directory `notebooks/` contains tutorial notebooks that walk through the main features:

**01 Greenland outlet glacier**: distributed sheet drainage, steady-state effective pressure

**02 Channels**: how Röthlisberger channels emerge at higher melt rates

**03 Moulins**: concentrated moulin input and channel network structure

**04 Mountain glacier**: valley geometry, overdeepenings, and glaciohydraulic supercooling

### Validation

hydropack has been validated against the [SHMIP](https://shmip.bitbucket.io/) subglacial hydrology model intercomparison (Suites A–F), comparing with the Werder GlaDS MATLAB reference.
The SHMIP test scripts are maintained in a [separate repository](https://github.com/hoffmaao/shmip).


### GlaDS-2: ice uplift and free-surface flow

`hydropack.glads2` implements the bounded-pressure formulation of GlaDS-2 ([Wells et al., 2026](https://doi.org/10.5194/egusphere-2026-4310)), which extends GlaDS so that water pressure stays between atmospheric and ice overburden.
The model solves mass conservation for the total water thickness `h`. A piecewise closure relation (Eq. 4) then gives the water pressure and selects one of three regimes:

- **ice uplift**: `h > h_g + Δh`, at overburden pressure
- **pressurised flow**: `h_g ≤ h ≤ h_g + Δh`, with englacial storage
- **free-surface flow**: `h < h_g`, at atmospheric pressure; this regime extends to the proglacial zone

Channels can also be partially water-filled.
Following the paper, it is a finite-difference model on a structured staggered grid with explicit time stepping, and it needs only NumPy (no Firedrake).
Each constitutive relation lives in `hydropack/glads2/physics.py` and can be swapped by keyword:

```python
from hydropack.glads2 import BoundedHydrologyModel, StaggeredGrid, PointSource
from hydropack.glads2.forcing import parabolic_pulse

grid = StaggeredGrid(nx=100, ny=30, dx=500.0, dy=500.0)
model = BoundedHydrologyModel(
    grid, bed=B, thickness=H, melt_rate=1e-3 / 86400,
    point_sources=[PointSource(10e3, 7.5e3, parabolic_pulse(44e6, 1.4 * 3600))],
    boundary_conditions={"west": ("flux", 1e-6), "east": "atmospheric"},
    e_v=1e-4,
)
model.run(300 * 3600)
model.flotation_fraction(), model.sheet_regime(), model.blister_volume()
```

`hydropack.glads2.experiments` builds the paper's set-ups: the stepped glacier (Sect. 3) and the synthetic lake drainage (Sect. 4.1).
`hydropack.glads2.north_lake` builds the North Lake run (Sect. 4.2) from a BedMachine Greenland file, which you must supply.
The scripts in `examples/glads2/` regenerate Figs. 3–6:

```bash
python examples/glads2/fig3_stepped_glacier.py
python examples/glads2/fig4_6_synthetic_lake_drainage.py --figure 4   # also 5, 6
```

The paper does not give formulas for its synthetic geometries, so they were digitised from Figs. 3a and 4b (see the docstrings).

### Tests

```bash
pip install numpy matplotlib pytest
python -m pytest
```

The GlaDS-2 tests check:

- each constitutive relation against its equation in the paper
- steady states against analytic solutions in each of the three sheet regimes
- exact mass conservation, symmetry and boundary conditions
- coarse-grid versions of the lake-drainage experiment

They run in CI on every push.

#### Reproduction status (Wells et al., 2026)

Figures produced by the example scripts are in `docs/glads2/`.
They use the Table 2 parameters and the digitised geometries.

- **Fig. 4 (synthetic lake drainage): close match for the transient.**
  - Blister height and position at 2, 10, 40 and 90 h, its drainage at the terminus, and channel sizes all agree with the paper.
  - Before drainage, though, the water pressure inland is about 0.5 of overburden, against about 0.75 in Fig. 4a. An independent 1D shooting solution of the same steady equations also gives 0.5, so the difference comes from the paper's set-up rather than from the solver.
- **Fig. 3 (stepped glacier): partial.**
  - The sheet-only steady state at 20 mm/d follows the digitised reference for φ and much of h.
  - The two cases with channels do not match. The reference cavity height (about 2 cm at the divide) is below the minimum that Table 2 values of A, u_b and h_r allow under about 200 m of ice. This suggests that the Schoof/Hewitt reference used different parameters, which the paper does not list.
  - At 20 mm/d with channels, the model does not settle to a steady state. Water repeatedly builds up under uplifted ice near the terminus and drains through a growing channel.
- **Figs. 5 and 6:** scripts provided, not yet run at full resolution. The e_v = 1e-6 case is very expensive with explicit time stepping, as the paper also notes.
- **Figs. 7 and 8 (North Lake):** set-up provided; it needs BedMachine Greenland v6.
