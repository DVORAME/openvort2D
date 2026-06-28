import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob
import pandas as pd

from VortexPoints import VortexPoints

KAPPA = 9.96e-4

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--input", type=str, required=True, help="Input directory containing the VortexPoints files.")
	parser.add_argument("--output", type=str, help="Output directory to save the images. If none is provided, images will be saved in the input directory.")
	parser.add_argument("--show", action="store_true", help="Show the plot as interactive animation.")
	parser.add_argument("--save", action="store_true", help="Save the plots as image files.")
	parser.add_argument("--info", action="store_true", help="Add information to the plot.")
	parser.add_argument("--dpi", type=int, default=300, help="DPI for saved images. Default is 300.")

	args = parser.parse_args()

	input = args.input
	if args.output == input or args.output is None:
		output = input
	else:
		output = args.output
		base_output = output
		suffix_k = 1
		if os.path.exists(output):
			while True:
				output = base_output + f"_{suffix_k}"
				if not os.path.exists(output):
					break
				suffix_k += 1
		os.makedirs(output)
	
	vp_files = glob(os.path.join(output, '*.npz'))
	vp_files.sort()
	vp: VortexPoints = np.load(vp_files[0], allow_pickle=True)['arr_0'].item()

	with open(os.path.join(input, 'info.txt')) as file:
		info = dict(file.readline().strip())
	out = pd.read_csv(os.path.join(input, 'out.csv'), sep=',', index_col=0, header=0)

	circle = info.get('circle', False)
	D = float(info.get('D'))

	
	fig, ax = plt.subplots()
	if circle:
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

	if args.show:
		plt.ion()
		plt.show()
	else:
		plt.ioff()

	for i in range(len(vp_files)):
		vp = np.load(vp_files[i], allow_pickle=True)['arr_0'].item()
		if not hasattr(vp, 'step_n'):
			vp.step_n = 0
		pos.set_xdata(vp.xs[vp.signs > 0])
		pos.set_ydata(vp.ys[vp.signs > 0])
		neg.set_xdata(vp.xs[vp.signs < 0])
		neg.set_ydata(vp.ys[vp.signs < 0])
		if circle:
			handle.set_xdata([np.cos(out['phi'][i])*D/2])
			handle.set_ydata([np.sin(out['phi'][i])*D/2])
		if args.info:
			info_text.set_text(f"t = {vp.t:.6e} s\nN = {abs(vp.signs).sum()}\nL = {sum(vp.signs) * KAPPA:.6e} cm^2/s\nphi = {out['phi'][i]:.6e} rad\nomega = {out['omega'][i]:.6e} rad/s")
		if args.show:
			fig.canvas.draw()
			fig.canvas.flush_events()
			plt.pause(0.001)
		if args.save:
			fig.savefig(os.path.join(output, f"frame_{i:08d}.png"), dpi=args.dpi)






