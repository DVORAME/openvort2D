#!/usr/bin/env python
# This file is part of the openvort2D project.
#
# Copyright (C) 2024 Emil Varga and superfluid lab, MFF CUNI
#
# openvort2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# openvort2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with openvort2D. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.random import randn
import time
import argparse

import os.path as path
from glob import glob

import taichi as ti
import os

kappa = 9.96e-4

from VortexPoints import VortexPoints
"""
Main runner for the openvort2D point-vortex simulation.

This script provides a command-line interface to initialize and run a
collection of 2D quantum point vortices using Taichi accelerated kernels
defined in `VortexPoints.py`. It supports periodic or wall boundary
conditions, probe flows, injection of vortex pairs, restart from saved
state, and saving of frames and restart snapshots.

Typical usage (from project root):
    python -m src.main --N 1000 --D 1e-2 --save --output output

The script is lightweight: plotting is optional and uses `matplotlib`.
"""
   
if __name__ == '__main__':
    # Command-line arguments overview:
    # - `--D`: domain size (cm). Simulation runs in a square [0,D] x [0,D].
    # - `--N`: initial number of vortices. Half positive, half negative by default.
    # - `--dt`: time-step for Euler integration (seconds).
    # - `--alpha`, `--alphap`: mutual-friction/dissipation parameters.
    # - `--vpin`, `--pin-type`: pinning model parameters (threshold/drag/none).
    # - `--probe-*`: define an externally applied probe flow (uniform/grid/combined).
    # - `--walls`: enable reflecting horizontal walls instead of periodic BCs.
    # - `--inject`: periodically inject vortex-antivortex pairs.
    # - `--save`, `--output`, `--save-every`: control saving of frames and restart files.
    # - `--restart`: load last saved state from the output directory and continue.
    # - `--gpu`: run Taichi on GPU when available; otherwise CPU.
    # - `--tmax`: maximum simulation time (stop when reached).
    # Use `python -m src.main -h` to see per-argument help.
    parser = argparse.ArgumentParser()
    parser.add_argument('--D', type=float, default=1e-2, help='Domain size (cm) of the square simulation box.')
    parser.add_argument('--alpha', type=float, default=0.03, help='Mutual friction coefficient alpha.')
    parser.add_argument('--alphap', type=float, default=1.76e-2, help="Mutual friction coefficient alpha' (prime).")
    parser.add_argument('--output', type=str, default='output', help='Directory to write frames, logs, and restart files.')
    parser.add_argument('--save', action='store_true', help='Enable saving of frames and restart snapshots.')
    parser.add_argument('--inject', action='store_true', help='Enable periodic injection of vortex-antivortex pairs.')
    parser.add_argument('--save-every', type=int, default=1, help='Save every N iterations (initial). Can be increased by --variable-save-rate.')
    parser.add_argument('--polarization', type=float, default=0, help='Fraction or magnitude controlling polarized initial conditions.')
    parser.add_argument('--polarization-type', type=str, default='none', help="Initial pattern type: 'none','jet','dipole','grid','pairs'.")
    parser.add_argument('--gridx', type=int, help='Grid x-dimension for grid initializations.')
    parser.add_argument('--gridy', type=int, help='Grid y-dimension for grid initializations.')
    parser.add_argument('--grid-sigma-div', type=float, help='Standard deviation divisor for grid cluster spread.')
    parser.add_argument('--gpu', action='store_true', help='Attempt to run Taichi on GPU instead of CPU.')
    parser.add_argument('--N', type=int, default=1000, help='Initial number of vortices (total).')
    parser.add_argument('--dt', type=float, default=1e-9, help='Time step used in the Euler integrator (s).')
    parser.add_argument('--restart', action='store_true', help='Load the most recent restart (.npz) from output and continue.')
    parser.add_argument('--walls', action='store_true', help='Enable reflecting horizontal walls (non-periodic in y).')
    parser.add_argument('--pinning-v', type=float, default=0, help='Characteristic pinning velocity vpin used for depinning threshold models.')
    parser.add_argument('--probe-v', type=float, default=0, help='Amplitude of uniform probe flow.')
    parser.add_argument('--probe-v-freq', type=float, default=0, help='Frequency (Hz) of time-oscillation for probe flows.')
    parser.add_argument('--probe-grid', type=int, nargs=2, default=[0,0], help='Integer wave numbers (n,k) used by grid probe flow.')
    parser.add_argument('--probe-grid-v', type=float, default=0, help='Amplitude for spatial grid probe flow.')
    parser.add_argument('--probe-type', type=str, default='uniform', 
                        help="Probe flow type. Options: 'uniform' (constant across space), 'grid' (spatially varying), 'combined'.")
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting (recommended for headless runs).')
    parser.add_argument('--pin-type', type=str, default='threshold', 
                        help="Pinning/dissipation model: 'threshold' (no motion below vpin), 'drag' (continuous drag), 'none'.")
    parser.add_argument('--variable-save-rate', type=float, default=0,
                        help="Fractional increase of save interval between frames. Positive => sparser saves over time.")
    parser.add_argument('--tmax', type=float, default=None,
                        help="Maximum simulation time (seconds). If unspecified run until vortices annihilate.")
    args = parser.parse_args()
    D = args.D
    alpha = args.alpha
    alphap = args.alphap
    output = args.output
    save = args.save

    if args.tmax is not None:
        if args.tmax <= 0:
            raise ValueError("tmax must be a positive number.")
        if args.tmax < args.dt:
            raise ValueError("tmax must be greater than dt.")
        tmax = args.tmax
    else:
        tmax = np.inf

    if args.gpu:
        ti.init(ti.gpu)
    else:
        ti.init(ti.cpu)
    # Initialize Taichi on the selected device. Taichi kernels are used
    # inside `VortexPoints` for performance-critical computations.
    
    base_output = output
    suffix_k = 1 
    if os.path.exists(output) and not args.restart:
        while True:
            output = base_output + f"_{suffix_k}"
            if not os.path.exists(output):
                break
            suffix_k += 1
    
    if args.restart:
        vp_files = glob(path.join(output, '*.npz'))
        vp_files.sort()
        print(len(vp_files))
        restart_file = np.load(vp_files[-1], allow_pickle=True)
        vp = restart_file['arr_0'].item()
        if not hasattr(vp, 'step_n'):
            vp.step_n = 0
        Lfile_mode = 'a'
        frame = len(vp_files)
    else:
        # Create output directory for frames and restart files.
        os.makedirs(output)    
        vp = VortexPoints(args.N, D, polarization=args.polarization, polarization_type=args.polarization_type,
                          walls=args.walls, vpin=args.pinning_v,
                          probe_v=args.probe_v, probe_v_freq=args.probe_v_freq,
                          gridx=args.gridx, gridy=args.gridy, grid_div=args.grid_sigma_div,
                          probe_type=args.probe_type, probe_grid=args.probe_grid, probe_grid_v=args.probe_grid_v)
        Lfile_mode='w'
        frame = 0
    fig, ax = plt.subplots()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal')
    pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r', ms=2)
    neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b', ms=2)
    dt = args.dt
    last_inject = 0
    it = 0
    save_rate = args.save_every
    save_countdown = 0
    with open(f'{output}/L_t.txt', Lfile_mode) as Lfile:
        while True:
            if args.inject and vp.t - last_inject > 0.0005:
                vp.inject(5)
                vp.annihilate()
                vp.check()
                last_inject = vp.t
                # print("injecting")
            vp.update_velocity()
            vp.check()
            vp.dissipation(alpha, alphap)
            vp.step(dt)
            vp.annihilate()
            vp.check()
            vp.coerce()
            pos.set_xdata(vp.xs[vp.signs > 0])
            pos.set_ydata(vp.ys[vp.signs > 0])
            neg.set_xdata(vp.xs[vp.signs < 0])
            neg.set_ydata(vp.ys[vp.signs < 0])
            if not args.no_plot:
                plt.pause(0.001)
            N = abs(vp.signs).sum()
            print(it, vp.t, N, vp.N, sum(vp.signs))
            # Save condition: write L(t), dump fig image and save a restart npz
            if save_countdown == 0 and save:
                Lfile.write(f"{it}\t{vp.t}\t{N}\n")
                Lfile.flush()
                fig.savefig(f'{output}/frame{frame:08d}.png')
                frame += 1
                np.savez(f'{output}/vp_{frame:08d}.npz', vp)
                save_rate = save_rate*(1 + args.variable_save_rate)
                save_countdown = int(save_rate)
            if N == 0:
                break
            if vp.t >= tmax:
                print(f"Reached tmax = {tmax}. Stopping simulation.")
                break
            it += 1
            save_countdown -= 1
