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
import matplotlib.pyplot as plt
from numpy.random import randn
import time
import argparse

import os.path as path
from glob import glob

import taichi as ti
import os

KAPPA = 9.96e-4

from VortexPoints import VortexPoints
"""
Main runner for the openvort2D point-vortex simulation.

This script provides a command-line interface to initialize and run a
collection of 2D quantum point vortices using Taichi accelerated kernels
defined in `VortexPoints.py`. It supports periodic or wall boundary
conditions, probe flows, injection of vortex pairs, restart from saved
state, and saving of frames and restart snapshots.

Typical usage (from project root):
	python src.main --N 1000 --D 1e-2 --save --output output

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

	parser.add_argument('--N', type=int, default=1000, help='Initial number of vortices (total).')
	parser.add_argument('--dt', type=float, default=1e-9, help='Time step used in the Euler integrator (s).')
	parser.add_argument('--tmax', type=float, default=None,
						help="Maximum simulation time (seconds). If unspecified run until vortices annihilate.")
	
	parser.add_argument('--alpha', type=float, default=0.03, help='Mutual friction coefficient alpha.')
	parser.add_argument('--alphap', type=float, default=1.76e-2, help="Mutual friction coefficient alpha' (prime).")
	parser.add_argument('--pinning-v', type=float, default=0, help='Characteristic pinning velocity vpin used for depinning threshold models.')
	parser.add_argument('--pin-type', type=str, default='threshold', 
						help="Pinning/dissipation model: 'threshold' (no motion below vpin), 'drag' (continuous drag), 'none'.")
	
	parser.add_argument('--D', type=float, default=1e-2, help='Domain size (cm) of the square simulation box.')
	parser.add_argument('--walls', action='store_true', help='Enable reflecting horizontal walls (non-periodic in y).')
	parser.add_argument('--circle', action='store_true', help='Enable reflecting circular boundary (non-periodic). Takes precedence over walls if both are specified.')
	parser.add_argument('--omega', type=float, default=0, help='Angular velocity (rad/s) for rotating resonator.')
	parser.add_argument('--omega-lambda', type=str, default='', help="Angular velocity expressed in terms of time t, ie. '0.1*t' or 'np.sin(t)**2'. Overrides --omega if specified.")
	
	parser.add_argument('--polarization', type=float, default=0, help='Fraction or magnitude controlling polarized initial conditions.')
	parser.add_argument('--polarization-type', type=str, default='none', help="Initial pattern type: 'none','jet','dipole','grid','pairs'.")
	parser.add_argument('--gridx', type=int, help='Grid x-dimension for grid initializations.')
	parser.add_argument('--gridy', type=int, help='Grid y-dimension for grid initializations.')
	parser.add_argument('--grid-sigma-div', type=float, help='Standard deviation divisor for grid cluster spread.')
	
	parser.add_argument('--probe-v', type=float, default=0, help='Amplitude of uniform probe flow.')
	parser.add_argument('--probe-v-freq', type=float, default=0, help='Frequency (Hz) of time-oscillation for probe flows.')
	parser.add_argument('--probe-type', type=str, default='uniform', 
						help="Probe flow type. Options: 'uniform' (constant across space), 'grid' (spatially varying), 'combined'.")
	parser.add_argument('--probe-grid', type=int, nargs=2, default=[0,0], help='Integer wave numbers (n,k) used by grid probe flow.')
	parser.add_argument('--probe-grid-v', type=float, default=0, help='Amplitude for spatial grid probe flow.')
	
	parser.add_argument('--inject', action='store_true', help='Enable periodic injection of vortex-antivortex pairs.')
	
	parser.add_argument('--save', action='store_true', help='Enable saving of frames and restart snapshots.')
	parser.add_argument('--save-every', type=int, default=1, help='Save every N iterations (initial). Can be increased by --variable-save-rate.')
	parser.add_argument('--variable-save-rate', type=float, default=0,
						help="Fractional increase of save interval between frames. Positive => sparser saves over time.")
	parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting (recommended for headless runs).')
	parser.add_argument('--no-plot-save', action='store_true', help='Disable saving of plot frames (only save restart files).')
	parser.add_argument('--plot-info', action='store_true', help='Enable overlay of simulation info on plot frames (t, N, L, phi, omega).')
	parser.add_argument('--output', type=str, default='output', help='Directory to write frames, logs, and restart files.')
	
	parser.add_argument('--gpu', action='store_true', help='Attempt to run Taichi on GPU instead of CPU.')
	parser.add_argument('--restart', action='store_true', help='Load the most recent restart (.npz) from output and continue.')
	
	args = parser.parse_args()
	D = args.D
	alpha = args.alpha
	alphap = args.alphap
	calculate_omega = args.omega_lambda != ''
	if calculate_omega:
		# If omega is specified as a function of time, define a lambda function to evaluate it.
		omega_func = eval(f"lambda t: {args.omega_lambda}")
		omega = omega_func(0)
	else:
		omega = args.omega
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

	# Initialize Taichi on the selected device. Taichi kernels are used
	# inside `VortexPoints` for performance-critical computations.
	if args.gpu:
		ti.init(ti.gpu)
	else:
		ti.init(ti.cpu)
	
	if args.restart:
		vp_files = glob(path.join(output, '*.npz'))
		vp_files.sort()
		print(len(vp_files))
		restart_file = np.load(vp_files[-1], allow_pickle=True)
		vp = restart_file['arr_0'].item()
		if not hasattr(vp, 'step_n'):
			vp.step_n = 0
		file_mode = 'a'
		frame = len(vp_files)
	else:
		base_output = output
		suffix_k = 1
		if os.path.exists(output):
			while True:
				output = base_output + f"_{suffix_k}"
				if not os.path.exists(output):
					break
				suffix_k += 1
		# Create output directory for frames and restart files.
		os.makedirs(output)    
		vp = VortexPoints(N=args.N, D=D, polarization=args.polarization, polarization_type=args.polarization_type,
						  walls=args.walls, circle=args.circle, vpin=args.pinning_v,
						  probe_v=args.probe_v, probe_v_freq=args.probe_v_freq,
						  gridx=args.gridx, gridy=args.gridy, grid_div=args.grid_sigma_div,
						  probe_type=args.probe_type, probe_grid=args.probe_grid, probe_grid_v=args.probe_grid_v)
		file_mode='w'
		frame = 0
	
	draw = not args.no_plot or not args.no_plot_save
	if draw:
		# Initialize matplotlib figure for plotting vortex positions.
		fig, ax = plt.subplots()
		if args.circle:
			ax.set_xlim(-D/2, D/2)
			ax.set_ylim(-D/2, D/2)
			circle = plt.Circle((0, 0), D/2, color='k', fill=False)
			ax.add_artist(circle)
			handle, = plt.plot(D/2, 0, marker='o', color='k', ms=5)
		else:
			ax.set_xlim(0, D)
			ax.set_ylim(0, D)
		ax.set_aspect('equal')
		pos, = ax.plot(vp.xs[vp.signs > 0], vp.ys[vp.signs > 0], 'o', color='r', ms=2)
		neg, = ax.plot(vp.xs[vp.signs < 0], vp.ys[vp.signs < 0], 'o', color='b', ms=2)

		if args.plot_info:
			info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
								verticalalignment='top', horizontalalignment='left',
								bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

		if not args.no_plot:
			plt.ion()
			plt.show()
		elif not args.no_plot_save:
			plt.ioff()
	
	phi = 0
	dt = args.dt
	last_inject = 0
	it = 0
	save_rate = args.save_every
	save_countdown = 0
	if not args.restart:
		with open(os.path.join(output, 'info.txt'), 'w') as file:
			file.write(f"{{N:{args.N},dt:{args.dt},tmax:{args.tmax},alpha:{args.alpha},alphap:{args.alphap},pinning_v:{args.pinning_v},pin_type:'{args.pin_type}',D:{args.D},walls:{args.walls},circle:{args.circle},omega:{args.omega},omega_lambda:{args.omega_lambda},polarization:{args.polarization},polarization_type:'{args.polarization_type}',gridx:{args.gridx},gridy:{args.gridy},grid_sigma_div:{args.grid_sigma_div},probe_v:{args.probe_v},probe_v_freq:{args.probe_v_freq},probe_type:'{args.probe_type}',probe_grid:({args.probe_grid[0]},{args.probe_grid[1]}),probe_grid_v:{args.probe_grid_v},inject:{args.inject}}}\n")
	with open(os.path.join(output, 'out.csv'), file_mode) as file:
		if not args.restart:
			file.write("it,t,N,L,phi,omega\n")
		while True:
			if args.inject and vp.t - last_inject > 0.0005:
				vp.inject(5)
				vp.annihilate()
				vp.check()
				last_inject = vp.t
				# print("injecting")
			vp.update_velocity()
			vp.check()
			vp.dissipation(alpha, alphap, omega)
			vp.step(dt)
			if calculate_omega:
				omega = omega_func(vp.t)
			phi += omega*dt
			# TODO: Maybe switch order to eliminate one call of anihilate()?
			vp.annihilate()
			vp.check()
			vp.coerce()

			if draw:
				pos.set_xdata(vp.xs[vp.signs > 0])
				pos.set_ydata(vp.ys[vp.signs > 0])
				neg.set_xdata(vp.xs[vp.signs < 0])
				neg.set_ydata(vp.ys[vp.signs < 0])
				if args.circle:
					handle.set_xdata([np.cos(phi)*D/2])
					handle.set_ydata([np.sin(phi)*D/2])
				if args.plot_info:
					info_text.set_text(f"t = {vp.t:.6e} s\nN = {abs(vp.signs).sum()}\nL = {sum(vp.signs) * KAPPA:.6e} cm^2/s\nphi = {phi:.6e} rad\nomega = {omega:.6e} rad/s")
			if not args.no_plot:
				fig.canvas.draw()
				fig.canvas.flush_events()
				plt.pause(0.001)
			N = abs(vp.signs).sum()
			save_string = "{it:d},{t:.6e},{N:d},{L:.6e},{phi:.6e},{omega:.6e}\n".format(it=it, t=vp.t, N=N, L=sum(vp.signs) * KAPPA, phi=phi, omega=omega)
			print(save_string, end='')
			# Save condition: write output, dump fig image and save a restart npz
			if save_countdown == 0 and save:
				file.write(save_string)
				file.flush()
				if not args.no_plot_save:
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
