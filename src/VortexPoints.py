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
from numpy.random import rand, randn

import taichi as ti

# Quantum of circulation (cm^2/s) used in the point-vortex velocity formula
# Physical meaning: circulation quantum that sets the strength of the induced
# velocity field of each point vortex.
KAPPA = 9.96e-4
PI = 3.14159

# Cache Taichi types as constants because Python doesn't like calls in type expressions.
TI_ARRAY = ti.types.ndarray()
TI_INT_ARRAY = ti.types.ndarray(dtype=ti.int64)
# Float cannot be cached for some reason, using pyright ignore to silence the type checker.


@ti.kernel
def update_velocity_ti(xs: TI_ARRAY, ys: TI_ARRAY, signs: TI_ARRAY,
					   vx: TI_ARRAY, vy: TI_ARRAY, shifts: TI_ARRAY):
	"""Compute velocities on each vortex point using periodic image shifts.

	This Taichi kernel evaluates the 2D point-vortex Biot-Savart induced
	velocity for every active vortex j by summing contributions from all
	vortices k including periodic images specified by `shifts`.

	Args:
		xs, ys: arrays of vortex coordinates.
		signs: array of vortex circulations (+1, -1, 0 for removed).
		vx, vy: output arrays to write computed velocities.
		shifts: 1D array of shifts (e.g., [-D, 0, D]) applied to both axes
				to account for periodic images.
	"""
	# Number of vortex points and number of image shifts to consider.
	# Complexity: for N vortices and S image shifts per axis this kernel
	# performs O(N^2 * S^2) pairwise operations. Keep S small (commonly 3)
	# to account for nearest periodic images only.

	N = xs.shape[0]
	S = shifts.shape[0]
	for j in range(N):
		if signs[j] == 0:
			continue
		vx[j] = 0
		vy[j] = 0
		for k in range(N):
			for xshift in range(S):
				for yshift in range(S):
					# skip self-interaction for the zero-image
					# XXX Why include images? Won't their effect be cancelled by symmetry??
					if k == j and shifts[xshift] == 0 and shifts[yshift] == 0:
						continue
					x_jk = xs[j] - xs[k] + shifts[xshift]
					y_jk = ys[j] - ys[k] + shifts[yshift]
					r2_jk = x_jk**2 + y_jk**2
					# Biot-Savart for a point vortex in 2D: v = (kappa/2pi) * (z_hat x r) / r^2
					vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]
					vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]
	# Notes:
	#  - `signs` carries +/-1 for vortex circulation and 0 for removed vortices.
	#  - We explicitly skip the self-interaction term for the zero-image only;
	#    other images of the same vortex are included as they represent
	#    periodic copies.
	#  - Taichi kernels operate on Taichi-managed ndarrays. Careful with
	#    Python-side resizing: kernels expect consistent array sizes.

@ti.kernel
def update_velocity_walls_ti(xs: TI_ARRAY, ys: TI_ARRAY, signs: TI_ARRAY,
							 vx: TI_ARRAY, vy: TI_ARRAY, D: ti.types.float64): # pyright: ignore[reportInvalidTypeForm]
	"""Compute velocities with hard-wall boundary conditions using mirror images.

	Instead of full periodic tiling, this kernel enforces no-penetration at
	the horizontal walls (y=0 and y=D) by adding mirror vortices with flipped
	circulation (image method). Periodicity in x is handled by a few x-shifts.
	"""

	N = xs.shape[0]
	for j in range(N):
		if signs[j] == 0:
			continue
		vx[j] = 0
		vy[j] = 0
		for k in range(N):
			for xshift in range(-1, 2):
				x_jk = xs[j] - xs[k] + xshift*D

				# no shift (regular contribution)
				# XXX Same question as before.
				if k!=j or xshift!=0:
					y_jk = ys[j] - ys[k]
					r2_jk = x_jk**2 + y_jk**2
					vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]
					vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]

				# mirror across top wall (image position y -> 2D - y). The
				# image vortex has opposite circulation for an impermeable
				# wall (no normal flow) boundary condition; hence the
				# `mirror_flip = -1` factor.
				mirror_flip = -1
				y_jk = ys[j] - (2*D - ys[k])
				r2_jk = x_jk**2 + y_jk**2
				vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]*mirror_flip
				vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]*mirror_flip
				
				# mirror across bottom wall (image position y -> -y).
				# Again the image circulations are flipped to enforce the
				# physical boundary condition.
				mirror_flip = -1
				y_jk = ys[j] + ys[k]
				r2_jk = x_jk**2 + y_jk**2
				vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]*mirror_flip
				vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]*mirror_flip

@ti.kernel
def update_velocity_circle_ti(xs: TI_ARRAY, ys: TI_ARRAY, signs: TI_ARRAY,
							 vx: TI_ARRAY, vy: TI_ARRAY, D: ti.types.float64): # pyright: ignore[reportInvalidTypeForm]
	"""Compute velocities with circular boundary conditions using mirror images.

	This kernel enforces no-penetration at a circular boundary of radius D/2
	by adding mirror vortices with flipped circulation (image method). The
	mirror position is computed as the inversion of the original vortex
	position with respect to the circle.
	"""

	N = xs.shape[0]
	for j in range(N):
		if signs[j] == 0:
			continue
		vx[j] = 0
		vy[j] = 0
		for k in range(N):
			if signs[k] == 0 or j == k:
				continue
			x_jk = xs[j] - xs[k]
			y_jk = ys[j] - ys[k]
			r2_jk = x_jk**2 + y_jk**2
			vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]
			vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]

			# Mirror across circular boundary using inversion
			r2_k = xs[k]**2 + ys[k]**2
			if r2_k < (D/2)**2:
				mirror_x = (D/2)**2 * xs[k] / r2_k
				mirror_y = (D/2)**2 * ys[k] / r2_k
				x_mirror_jk = xs[j] - mirror_x
				y_mirror_jk = ys[j] - mirror_y
				r2_mirror_jk = x_mirror_jk**2 + y_mirror_jk**2
				vx[j] += -KAPPA/2/PI/r2_mirror_jk*y_mirror_jk*(-signs[k])
				vy[j] += KAPPA/2/PI/r2_mirror_jk*x_mirror_jk*(-signs[k])

@ti.kernel
def calculate_velocity_walls_ti(vort_xs: TI_ARRAY, vort_ys: TI_ARRAY, signs: TI_ARRAY,
								xs: TI_ARRAY, ys: TI_ARRAY,
								vx: TI_ARRAY, vy: TI_ARRAY, D: ti.types.float64): # pyright: ignore[reportInvalidTypeForm]
	"""Compute velocities at arbitrary probe positions `xs,ys` due to vortices with walls.

	This kernel mirrors the logic of `update_velocity_walls_ti` but evaluates the
	induced velocity at a set of probe points (e.g., grid points) instead of at
	vortex locations.
	"""
	# N: number of source vortices; M: number of sample/probe points

	N = vort_xs.shape[0]
	M = xs.shape[0]
	for j in range(M):
		vx[j] = 0
		vy[j] = 0
		for k in range(N):
			if signs[k] == 0:
				continue
			for xshift in range(-1, 2):
				x_jk = xs[j] - vort_xs[k] + xshift*D

				# no shift
				y_jk = ys[j] - vort_ys[k]
				r2_jk = x_jk**2 + y_jk**2
				vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]
				vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]

				# mirror across top
				mirror_flip = -1
				y_jk = ys[j] - (2*D - vort_ys[k])
				r2_jk = x_jk**2 + y_jk**2
				vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]*mirror_flip
				vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]*mirror_flip
				
				# mirror across bottom
				mirror_flip = -1
				y_jk = ys[j] + vort_ys[k]
				r2_jk = x_jk**2 + y_jk**2
				vx[j] += -KAPPA/2/PI/r2_jk*y_jk*signs[k]*mirror_flip
				vy[j] += KAPPA/2/PI/r2_jk*x_jk*signs[k]*mirror_flip

@ti.kernel
def annihilate_ti(xs: TI_ARRAY, ys: TI_ARRAY, 
				  signs: TI_INT_ARRAY, shifts: TI_ARRAY,
				  a0: float, to_annihilate: TI_ARRAY) -> int:
	"""Detect and remove close vortex-antivortex pairs using periodic shifts.

	The kernel first marks pairs closer than `a0` for annihilation into
	the `to_annihilate` array. If any vortex would be involved in multiple
	annihilations (conflict), the kernel falls back to a serialized loop to
	safely clear pairs one-by-one.

	Returns:
		int: flag indicating whether the optimistic marking was OK (1) or not (0).
	"""

	N = xs.shape[0]
	S = shifts.shape[0]

	# Reset the marker array used to indicate which vortices will be removed.
	for j in range(N):
		to_annihilate[j] = 0

	# optimistic parallel marking: check unique pairs j<k
	for j in range(N):
		if signs[j] == 0:
			continue
		move_on = False
		for k in range(j+1, N):
			if signs[k] == 0 or signs[j] == signs[k]:
				continue
			for xshift in range(S):
				for yshift in range(S):
					x_jk = xs[k] - xs[j] + shifts[xshift]
					y_jk = ys[k] - ys[j] + shifts[yshift]
					r_jk = ti.sqrt(x_jk**2 + y_jk**2)
					if r_jk < a0:
						# When two opposite-signed vortices are closer than a0
						# we mark both for annihilation. We increment the
						# counters in `to_annihilate` so that we can detect
						# conflicting multiple annihilation claims.
						to_annihilate[j] += 1
						to_annihilate[k] += 1
						move_on = True
						break
				if move_on:
					break
			if move_on:
				break
	
	# If any entry in to_annihilate > 1 it means a vortex would be removed
	# in more than one pair simultaneously; this is a conflict that we
	# resolve by falling back to a serialized safe removal loop.
	OK = True
	for j in range(N):
		if to_annihilate[j] > 1:
			OK = False

	if OK:
		# Safe to remove all marked vortices in parallel
		for j in range(N):
			if to_annihilate[j] > 0:
				signs[j] = 0
	else:
		# serialize removal to avoid conflicts when a vortex is close to >1 partner
		# XXX I don't quite understand this, why not do that in the first loop? Why iterate over whole N? The pair would be removed when the first vortex comes up anyway.
		# OK, I get the difference between the loops now, butthe rest still stands.
		ti.loop_config(serialize=True)
		for j in range(N):
			if signs[j] == 0:
				continue
			move_on = False
			for k in range(N):
				if k == j:
					continue
				if signs[k] == 0 or signs[j] == signs[k]:
					continue
				for xshift in range(S):
					for yshift in range(S):
						x_jk = xs[k] - xs[j] + shifts[xshift]
						y_jk = ys[k] - ys[j] + shifts[yshift]
						r_jk = ti.sqrt(x_jk**2 + y_jk**2)
						if r_jk < a0:
							# perform immediate pair removal in serialized
							# fashion to avoid double-deletion and ensure
							# determinism in the presence of conflicts.
							signs[j] = 0
							signs[k] = 0
							move_on = True
							break
					if move_on:
						break
				if move_on:
					break
	return OK

@ti.kernel
def annihilate_walls_ti(xs: TI_ARRAY, ys: TI_ARRAY, 
						signs: TI_INT_ARRAY, shifts: TI_ARRAY,
						a0: float, to_annihilate: TI_ARRAY) -> int:
	"""Annihilate vortex-antivortex pairs considering wall reflections.

	Similar to `annihilate_ti`, but distance checks only consider x-shifts
	(periodicity in x) while y-reflections are handled explicitly. Finally,
	vortices too close to the physical walls (y < a0 or D-y < a0) are removed.
	"""

	N = xs.shape[0]
	S = shifts.shape[0]

	for j in range(N):
		to_annihilate[j] = 0

	for j in range(N):
		if signs[j] == 0:
			continue
		move_on = False
		for k in range(j+1, N):
			if signs[k] == 0 or signs[j] == signs[k]:
				continue
			for xshift in range(S):
				x_jk = xs[k] - xs[j] + shifts[xshift]
				y_jk = ys[k] - ys[j]
				r_jk = ti.sqrt(x_jk**2 + y_jk**2)
				if r_jk < a0:
					to_annihilate[j] += 1
					to_annihilate[k] += 1
					move_on = True
					break
			if move_on:
				break
	
	OK = True
	for j in range(N):
		if to_annihilate[j] > 1:
			OK = False

	if OK:
		for j in range(N):
			if to_annihilate[j] > 0:
				signs[j] = 0
	else:
		ti.loop_config(serialize=True)
		for j in range(N):
			if signs[j] == 0:
				continue
			move_on = False
			for k in range(N):
				if k == j:
					continue
				if signs[k] == 0 or signs[j] == signs[k]:
					continue
				for xshift in range(S):
					x_jk = xs[k] - xs[j] + shifts[xshift]
					y_jk = ys[k] - ys[j]
					r_jk = ti.sqrt(x_jk**2 + y_jk**2)
					if r_jk < a0:
						signs[j] = 0
						signs[k] = 0
						move_on = True
						break
				if move_on:
					break
	
	# now annihilate vortices that hit the physical horizontal walls
	for k in range(N):
		if signs[k] == 0:
			continue
		if ys[k] < a0 or shifts[2] - ys[k] < a0:
			signs[k] = 0
	return OK
		
@ti.kernel
def annihilate_circle_ti(xs: TI_ARRAY, ys: TI_ARRAY, 
						signs: TI_INT_ARRAY, D: float,
						a0: float, to_annihilate: TI_ARRAY) -> int:
	"""Annihilate vortex-antivortex pairs considering circular boundary.

	Similar to `annihilate_ti`, but 
	"""

	N = xs.shape[0]

	for j in range(N):
		to_annihilate[j] = 0

	for j in range(N):
		if signs[j] == 0:
			continue
		for k in range(j+1, N):
			if signs[k] == 0 or signs[j] == signs[k]:
				continue
			x_jk = xs[k] - xs[j]
			y_jk = ys[k] - ys[j]
			r2_jk = x_jk**2 + y_jk**2
			if r2_jk < a0**2:
				to_annihilate[j] += 1
				to_annihilate[k] += 1
				break
	
	#XXX Couldn't this be done with np.max or similar? How much does taichi hate that?
	OK = True
	for j in range(N):
		if to_annihilate[j] > 1:
			OK = False

	if OK:
		for j in range(N):
			if to_annihilate[j] > 0:
				signs[j] = 0
	else:
		ti.loop_config(serialize=True)
		for j in range(N):
			if signs[j] == 0:
				continue
			for k in range(N):
				if k == j:
					continue
				if signs[k] == 0 or signs[j] == signs[k]:
					continue
				x_jk = xs[k] - xs[j]
				y_jk = ys[k] - ys[j]
				r2_jk = x_jk**2 + y_jk**2
				if r2_jk < a0**2:
					signs[j] = 0
					signs[k] = 0
					break
	
	# Now annihilate vortices that hit the physical circular boundary
	for k in range(N):
		if signs[k] == 0:
			continue
		r2 = xs[k]**2 + ys[k]**2
		if r2 > (D/2 - a0)**2:
			signs[k] = 0
	
	return OK

def init_circle_positions(N, D):
	"""Initialize N vortices randomly inside a circle of radius D/2.

	Positions are uniformly distributed within the circular area.
	"""
	radius = D / 2
	xs = np.zeros(N)
	ys = np.zeros(N)

	x = rand(int(N * 4 / 3)) * D - radius
	y = rand(int(N * 4 / 3)) * D - radius
	xcache = x[x**2 + y**2 < radius**2]
	ycache = y[x**2 + y**2 < radius**2]
	x = xcache
	y = ycache
	if len(x) >= N:
		xs += x[:N]
	else:
		xs[:len(x)] += x
		xrest = N - len(x)
		while xrest > 0:
			x = rand(int(xrest * 4 / 3)) * D - radius
			xcache = x[x**2 + y**2 < radius**2]
			x = xcache
			if len(x) >= xrest:
				xs[len(xs) - xrest:] += x[:xrest]
				break
			else:
				xs[len(xs) - xrest:len(xs) - xrest + len(x)] += x
				xrest -= len(x)
	if len(y) >= N:
		ys += y[:N]
	else:
		ys[:len(y)] += y
		yrest = N - len(y)
		while yrest > 0:
			y = rand(int(yrest * 4 / 3)) * D - radius
			ycache = y[x**2 + y**2 < radius**2]
			y = ycache
			if len(y) >= yrest:
				ys[len(ys) - yrest:] += y[:yrest]
				break
			else:
				ys[len(ys) - yrest:len(ys) - yrest + len(y)] += y
				yrest -= len(y)
	
	return xs, ys

class VortexPoints:
	def __init__(self, N:int|None=None, D:float=1, a0:float=1e-5,
				 polarization:float=0, polarization_type:str='none',
				 walls:bool=False, circle:bool=False, vpin:float=0, pin_type='threshold',
				 probe_type:str = 'uniform', probe_v:float=0, probe_v_freq:float=0,
				 probe_grid=None, probe_grid_v=0,
				 gridx=None, gridy=None, grid_div=None):
		self.walls = walls
		self.circle = circle
		if walls and circle:
			print("Warning: both walls and circle boundary conditions specified. Circle will take precedence.")
		self.a0 = a0 # annihilation distance, in cm
		self.N = N
		self.D = D
		if circle:
			self.xs, self.ys = init_circle_positions(N, D)
		else:
			self.xs = rand(N)*D
			self.ys = rand(N)*D
		self.vx = np.zeros_like(self.xs)
		self.vy = np.zeros_like(self.ys)
		self.signs = np.ones(N, dtype=int)
		self.signs[int(N/2):] = -1
		self.vpin = vpin
		self.pin_type = pin_type

		self.probe_type = probe_type
		self.probe_v = probe_v
		self.probe_v_freq = probe_v_freq
		self.probe_grid = probe_grid
		self.probe_grid_v = probe_grid_v

		# TODO: Add option viable for circular boundary
		match probe_type:
			case 'uniform':
				self._probe_v = self.uniform_probe_v
			case 'grid':
				self._probe_v = self.grid_probe_v
			case 'combined':
				self._probe_v = self.combined_probe_v
			case _:
				raise ValueError(f"Unknown probe type {probe_type}")

		match polarization_type:
			case 'none':
				pass
			case 'skewed':
				print(f"initializing polarized {polarization_type}, {polarization}")
				npos = int(0.5*(polarization+1)*N)
				self.signs[:npos] = +1
				self.signs[npos:] = -1
			case 'jet':
				print(f"initializing polarized {polarization_type}, {polarization}")
				npos = int(0.5*polarization*N)
				nneg = npos
				nrest = N - npos - nneg
				self.xs[:npos] = (rand(npos) + 1)*D/2
				self.signs[:npos] = +1
				self.xs[npos:(npos+nneg)] = rand(nneg)*D/2
				self.signs[npos:(npos+nneg)] = -1
				self.signs[(npos+nneg):(npos+nneg+int(nrest/2))] = +1
				self.signs[(npos+nneg+int(nrest/2)):] = -1
			case 'dipole':
				print("initializing dipole")
				N_random = int(N*(1 - polarization))
				N_dipole = int(N*polarization)

				#the polarized part
				x0 = D*np.sqrt(2)/3/2
				n2 = int(N_dipole/2)
				self.xs[:n2] = D/2 - x0 + randn(n2)*D/10
				self.ys[:n2] = D/2 - x0 + randn(n2)*D/10
				self.signs[:n2] = 1
				self.xs[n2:N_dipole] = D/2 + x0 + randn(n2)*D/10
				self.ys[n2:N_dipole] = D/2 + x0 + randn(n2)*D/10
				self.signs[n2:N_dipole] = -1

				#the random part, positions are already random
				self.signs[N_dipole:int(N_dipole + N_random/2)] = +1
				self.signs[int(N_dipole + N_random/2):] = -1

				self.coerce()
			case 'grid':
				# XXX Doesn't take polarization strength into account, just fills the box with a grid of alternating signs. Maybe add some jitter to make it more realistic?
				n_bunch = int(N/gridx/gridy)
				sigma_x = D/gridx
				sigma_y = D/gridy
				n = 0
				for j in range(gridy):
					sign = 1 - 2*int(j % 2)
					for k in range(gridx):
						cy = (j + 0.5*int((gridy+1)%2))*sigma_y
						cx = (k + 0.5*int((gridx+1)%2))*sigma_x
						self.xs[n:(n+n_bunch)] = cx + randn(n_bunch)*sigma_x/grid_div
						self.ys[n:(n+n_bunch)] = cy + randn(n_bunch)*sigma_y/grid_div
						self.signs[n:(n+n_bunch)] = sign
						sign *= -1
						n += n_bunch
				self.trim()
			case 'pairs':
				# XXX Same comment as for 'grid'.
				sigma = D/grid_div
				npos = int(N/2)
				nneg = npos
				self.xs[:npos] = rand(npos)*D
				self.xs[npos:] = self.xs[:npos] + randn(nneg)*sigma
				self.ys[:npos] = rand(npos)*D
				self.ys[npos:] = self.ys[:npos] + randn(nneg)*sigma
				self.signs[:npos] = +1
				self.signs[npos:] = -1
				self.coerce()
			case _:
				raise ValueError("Unknown polarization type.")
		self.shifts = np.array([-D, 0, D])
		self.to_annihilate = np.zeros(N)
		self.t = 0
		self.step_n = 0
		self.omega = 0
		self.A = 0
	
	def uniform_probe_v(self):
		"""Return a spatially-uniform oscillatory probe velocity.

		The uniform probe flow is useful to model a bulk background
		superflow or applied drive. Its x-component oscillates with
		frequency `probe_v_freq` and amplitude `probe_v`. The y-component
		is zero in this simple model.
		"""
		probe_vx = self.probe_v*np.cos(2*np.pi*self.probe_v_freq*self.t)
		# Return arrays matching vortex count so we can add elementwise.
		return np.repeat(probe_vx, len(self.vx)), np.zeros_like(self.vy)
	
	def grid_probe_v(self):
		"""Return a spatially varying probe velocity with integer wave numbers.

		The grid probe flow produces a simple separable sinusoidal pattern
		across the domain controlled by integers `(n, k)` stored in
		`self.probe_grid`. The amplitude oscillates in time similar to the
		uniform probe.
		"""
		amplitude = self.probe_grid_v*np.cos(2*np.pi*self.probe_v_freq*self.t)
		n, k = self.probe_grid
		# Construct separable spatial dependence; scale by integer wave numbers
		# so higher n and k produce finer spatial structure.
		# XXX Why is it scaled with n and -k?
		spatial_x = n*np.cos(np.pi/self.D*n*self.xs)*np.cos(np.pi/self.D*k*self.ys)
		spatial_y = -k*np.sin(np.pi/self.D*n*self.xs)*np.sin(np.pi/self.D*k*self.ys)
		return amplitude*spatial_x, amplitude*spatial_y
	
	def combined_probe_v(self):
		"""Combine uniform and grid probe flows.

		Useful to superimpose a bulk drive on top of spatial variations.
		"""
		vxu, vyu = self.uniform_probe_v()
		vxg, vyg = self.grid_probe_v()
		return vxu+vxg, vyu+vyg

	def plot(self, ax):
		"""Scatter-plot vortices on matplotlib `ax`.

		Positive vortices are red, negative are blue. This helper is a
		convenience wrapper for quick visualization during debugging and
		for producing saved frames.
		"""
		# XXX Is this ever used?
		ixp = self.signs > 0
		ixn = self.signs < 0
		ax.scatter(self.xs[ixp], self.ys[ixp], color='r')
		ax.scatter(self.xs[ixn], self.ys[ixn], color='b')
	
	def update_velocity(self):
		"""Compute vortex velocities from interactions and add probe flow.

		This function selects the appropriate Taichi kernel depending on
		whether wall boundary conditions are active. The kernels write into
		the `self.vx`/`self.vy` arrays which are then augmented by the
		chosen probe flow (if any).
		"""
		if self.circle:
			# Use image/circle kernel for reflecting circular boundaries
			update_velocity_circle_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.D)
		elif self.walls:
			# Use image/wall kernel for reflecting horizontal boundaries
			update_velocity_walls_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.D)
		else:
			# Use periodic kernel with nearest-image shifts
			update_velocity_ti(self.xs, self.ys, self.signs, self.vx, self.vy, self.shifts)
		
		# Add externally imposed probe flow (uniform, grid, or combined)
		probe_vx, probe_vy = self._probe_v()
		self.vx += probe_vx
		self.vy += probe_vy
	
	def dissipation(self, alpha=0.1, alphap=0, omega=0):
		"""Apply a mutual-friction / pinning model to vortex velocities.

		The model implements three behaviors controlled by `self.pin_type`:
		  - 'threshold': vortices are pinned (no motion) unless their
			kinetic speed exceeds `vpin`; depinned vortices experience
			mutual-friction terms proportional to `alpha` and `alphap`.
		  - 'drag': a continuous drag model that adjusts coefficients based
			on local speed; this uses intermediate `alpha_hat` and `alphap_hat`.
		  - 'none': no pinning; pure mutual friction is applied everywhere.

		Computation steps:
		  1. compute squared speeds v2
		  2. compare to `vpin` to obtain depinned mask
		  3. compute corrected coefficients for drag model when needed
		  4. build new velocities according to selected pin model
		"""

		# Translate velocities to local frames moving on tangent to the rotation of walls
		self.vx += omega*self.ys
		self.vy -= omega*self.xs

		v2 = self.vx**2 + self.vy**2
		# avoid division by zero when vpin==0: inv_beta2 will be inf or nan
		inv_beta2 = v2/self.vpin**2
		depinned = inv_beta2 > 1
		# x = 1/Gamma for depinned vortices, zero otherwise
		x = np.sqrt(np.where(depinned, inv_beta2 - 1, 0))
		# Compute corrected mutual-friction coefficients used in 'drag' model
		alpha_hat = x*(alpha**2 + alpha*x + alphap**2 - 2*alphap + 1)
		alpha_hat /= (alpha**2 + 2*alpha*x + alphap**2 - 2*alphap + x**2 + 1)
		alphap_hat = (alpha**2 + 2*alpha*x + alphap**2 + alphap*x**2 - 2*alphap + 1)
		alphap_hat /= (alpha**2 + 2*alpha*x + alphap**2 - 2*alphap + x**2 + 1)

		if self.pin_type == 'threshold':
			# If depinned: apply mutual-friction modified velocity; else zero
			mf_vx = np.where(depinned, self.vx + alpha*self.vy*self.signs - alphap*self.vx, 0)
			mf_vy = np.where(depinned, self.vy - alpha*self.vx*self.signs - alphap*self.vy, 0)
		elif self.pin_type == 'drag':
			# Use speed-dependent coefficients
			mf_vx = np.where(depinned, self.vx + alpha_hat*self.vy*self.signs - alphap_hat*self.vx, 0)
			mf_vy = np.where(depinned, self.vy - alpha_hat*self.vx*self.signs - alphap_hat*self.vy, 0)
		elif self.pin_type == 'none':
			# No pinning; apply mutual-friction everywhere
			mf_vx = self.vx + alpha*self.vy*self.signs - alphap*self.vx
			mf_vy = self.vy - alpha*self.vx*self.signs - alphap*self.vy
		else:
			raise ValueError(f"Unknown pin type, {self.pin_type}.")

		# Update velocities in-place
		self.vx = mf_vx
		self.vy = mf_vy

		## Translate velocities back to the lab frame
		self.vx -= omega*self.ys
		self.vy += omega*self.xs
	
	def annihilate(self):
		"""Run the appropriate annihilation kernel depending on BCs.

		The kernel marks and/or removes vortex-antivortex pairs closer than
		`self.a0`. For wall boundary conditions, additional checks remove
		vortices that collide with the boundaries.
		"""
		if self.circle:
			annihilate_circle_ti(self.xs, self.ys, self.signs, self.D, self.a0, self.to_annihilate)
		elif self.walls:
			annihilate_walls_ti(self.xs, self.ys, self.signs, self.shifts, self.a0, self.to_annihilate)
		else:
			annihilate_ti(self.xs, self.ys, self.signs, self.shifts, self.a0, self.to_annihilate)
	
	def inject(self, npairs):
				"""Insert `npairs` vortex-antivortex pairs near the box center.

				Strategy:
					- Compute evenly spaced y-positions for positive and negative
						members with small gaussian jitter.
					- If there are inactive slots (sign == 0) reuse them to avoid
						resizing arrays. Otherwise append new entries to arrays.
				Note: when arrays are expanded we also append velocities initialized
				to zero and reset `to_annihilate` to the new length.
				"""
				stepping = self.D/(2*npairs)
				posy = np.linspace(0, self.D-stepping, npairs) + np.random.randn(npairs)*self.D/100
				negy = np.linspace(stepping, self.D, npairs) + np.random.randn(npairs)*self.D/100

				# Reuse freed slots if available
				free = np.sum(self.signs == 0)
				if free > 2*npairs:
						ixfree = np.where(self.signs == 0)[0]
						# place new vortices at the free indices
						self.ys[ixfree[:npairs]] = posy
						self.ys[ixfree[npairs:(2*npairs)]] = negy
						self.xs[ixfree[:(2*npairs)]] = self.D/2
						self.signs[ixfree[:npairs]] = 1
						self.signs[ixfree[npairs:(2*npairs)]] = -1
						return

				# otherwise append new entries to arrays (costly for large arrays)
				self.xs = np.append(self.xs, np.zeros(len(posy) + len(negy)) + self.D/2)
				self.ys = np.append(self.ys, posy)
				self.ys = np.append(self.ys, negy)
		
				self.vx = np.append(self.vx, np.zeros(len(posy) + len(negy)))
				self.vy = np.append(self.vy, np.zeros(len(posy) + len(negy)))

				self.signs = np.append(self.signs, np.ones_like(posy, dtype=int))
				self.signs = np.append(self.signs, -np.ones_like(negy, dtype=int))

				self.N += 2*npairs
				# resize annihilation helper array to match new N
				self.to_annihilate = np.zeros(self.N)

	def step(self, dt):
		"""Advance positions by Euler step dt and perform periodic cleanup.

		The integrator here is explicit Euler: x += v*dt. Every 100 steps we
		compact arrays by removing inactive vortices to keep N small.
		"""
		# Integrate positions
		self.xs += self.vx*dt
		self.ys += self.vy*dt
		# Advance simulation time and step counter
		self.t += dt
		self.step_n += 1
		# Periodically remove inactive vortices to keep arrays compact
		if self.step_n % 100 == 0:
			self.cleanup()
	
	def coerce(self):
		"""
		Adjusts the coordinates of points to ensure they remain within the bounds [0, D) for both x and y axes.
		This method iterates over all points and, if any coordinate (x or y) falls outside the interval [0, D),
		it wraps the coordinate around by adding or subtracting D as necessary. The process repeats until all
		coordinates are within bounds.
		Attributes used:
			self.xs (list or array): The x-coordinates of the points.
			self.ys (list or array): The y-coordinates of the points.
			self.N (int): The number of points.
			self.D (float): Simulation domain size
		"""
		# This function ensures all point coordinates are wrapped into the
		# [0, D) interval, simulating periodic boundaries (torus topology).
		# The loop repeats until no coordinate lies outside the interval; this
		# handles large displacements but typically completes in one pass.

		if self.circle:
			return  # No coercion needed for circular boundary; points outside boundary are annihilated instead.

		while True:
			coerced = 0
			for j in range(self.N):
				if self.xs[j] > self.D:
					self.xs[j] -= self.D
					coerced += 1
				if self.ys[j] > self.D:
					self.ys[j] -= self.D
					coerced += 1
				if self.xs[j] < 0:
					self.xs[j] += self.D
					coerced += 1
				if self.ys[j] < 0:
					self.ys[j] += self.D
					coerced += 1
			if coerced == 0:
				break
	
	def cleanup(self):
		"""
		Removes points that are not active (i.e., have zero vorticity).
		"""
		# Compact arrays by keeping only entries with non-zero sign
		ix_nonzero = abs(self.signs) > 0
		self.xs = self.xs[ix_nonzero]
		self.ys = self.ys[ix_nonzero]
		self.vx = self.vx[ix_nonzero]
		self.vy = self.vy[ix_nonzero]
		self.signs = self.signs[ix_nonzero]
		self.N = len(self.xs)
	
	def trim(self):
		"""
		Trims the points, which the random initial condition placed outside of the simulation domain.
		"""
		# Mark vortices outside the [0,D] box as inactive; they will be
		# removed by the next `cleanup()` call. This is used after certain
		# initialization routines that may place particles outside bounds.
		for j in range(self.N):
			if self.xs[j] > self.D or self.xs[j] < 0:
				self.signs[j] = 0
			if self.ys[j] > self.D or self.ys[j] < 0:
				self.signs[j] = 0
	
	def check(self):
		# In the periodic variant (no walls) total vorticity should be
		# conserved and initially zero if the initialization used equal
		# positive/negative counts. This sanity-check raises if net vorticity
		# is nonzero, helping detect bugs in annihilation or initialization.
		if not self.walls and not self.circle:
			v = sum(self.signs)
			if abs(v) > 0:
				raise RuntimeError("nonzero vorticity")
