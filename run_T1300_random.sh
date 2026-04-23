#!/bin/bash
source .venv/bin/activate
alpha=0.034
alphap=0.001383
D=0.1
dt=1e-5

N=20000
v_pin=0
v_probe=0
freq=2000

python src/main.py --D $D --dt=$dt --alpha $alpha --alphap $alphap --walls --save --save-every 20 \
	--pinning-v $v_pin --probe-v $v_probe --probe-v-freq $freq --polarization-type none --gridx 2 --gridy 2 --grid-sigma-div 5 \
	--N $N --output random_N$N --no-plot --variable-save-rate 0.001 --pin-type none --gpu 
