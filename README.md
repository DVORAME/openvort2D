-- This branch includes theoretical derivations of some formulae used in the simulation. Primarly use the `main` branch for practical uses. --

This code simulates the motion of vortex points in superfluid 4He using the Taichi 
language (https://www.taichi-lang.org/) for parallelization of the calculation.

Run `uv sync` to install necessary dependencies (you need the `uv` package management
tool, https://docs.astral.sh/uv/getting-started/installation/)

Various models of pinning, dissipation and externally-applied superflow are available. Run
`python src/main.py --help` for a list of options.

Example:
```
#!/bin/bash
source .venv/bin/activate
alpha=0.034
alphap=0.001383
D=0.1
dt=1e-5

N=20000
v_pin=10
v_probe=15
freq=2000

python src/main.py --N $N --D $D --dt=$dt --alpha $alpha --alphap $alphap --walls \
    --save --save-every 20 --no-plot --output output_directory \
    --pinning-v $v_pin --probe-v $v_probe --probe-v-freq $freq --pin-type drag\
	--gpu
```
save as `run.sh`, make executable `chmod +x run.sh` and run `./run.sh` which will run the
simulation with `N` random vortex points in domain size `D = 0.1` cm, with time step `dt`, mutual friction coefficients `alpha` and `alphap`, with solid walls in the y-direction and periodic conditions in the x-direction (`--walls`). The output will be saved in the `output_directory` every 20 time steps. Plotting will only save a snapshot, not open an interactive window (`--no-plot`). The vortices will be subject to pinning with depinning velocity `v_pin` and will be driven by a probe velocity `v_probe` along the x-direction oscillating at frequency `freq`. The simulation will be GPU accelerated, if CUDA is available (`--gpu`).
