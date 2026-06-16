#! /bin/bash

alpha=0.061
alphap=0.01746
D=0.1
dt=1e-5
tmax=60

N=20000
v_pin=15
v_probe=25
freq=2000

python src/main.py --D $D --dt=$dt --alpha $alpha --alphap $alphap --walls --save --save-every 20 \
	--tmax $tmax --pinning-v $v_pin --probe-v $v_probe --probe-v-freq $freq --N $N --no-plot --gpu 