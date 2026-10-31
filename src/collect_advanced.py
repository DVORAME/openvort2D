import glob
from importlib.resources import path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='output', help='Path to the output directory')
parser.add_argument('--n', type=int, default=10, help='Number of output directories to process')
parser.add_argument('--subplots', action='store_true', help='Whether to create subplots for each output directory')
parser.add_argument('--var', nargs="*", type=str, default=[], help='Variables changing in plots')

args = parser.parse_args()
path = args.path

D_frac = 21/23
KAPPA = 9.96e-4
if args.subplots:
	plt.figure()
for i in range(1, args.n + 1):
	# Load the CSV file into a DataFrame
	with open(os.path.join(f'{path}/output_{i}', 'info.txt')) as file:
			s = file.readline().strip()
			info = ast.literal_eval(s)
	df = pd.read_csv(f'{path}/output_{i}/out.csv')
	D = float(info.get('D')) * D_frac
	vp_files = glob.glob(f'{path}/output_{i}/vp_*.npz')
	vp_files.sort()
	vps = np.fromiter(map(lambda f: np.load(f, allow_pickle=True)['arr_0'].item(), vp_files), object)
	N = np.fromiter(map(lambda vp: np.sum((vp.xs ** 2 + vp.ys ** 2) < D**2/4), vps), int)
	# Plotting the data
	if args.subplots:
		if args.n % 2 == 0:
			plt.subplot(args.n // 2, 2, i, )
		else:
			plt.subplot(args.n // 3, 3, i)
	else:
		plt.figure(figsize=(10, 10))
	plt.plot(df['t'], N, label='N')
	plt.plot(df['t'], df['omega'] * np.pi * D**2 / 2 / KAPPA, label='$N_{{id}}$')
	plt.legend()
	plt.xlabel('$t$ / s')
	plt.ylabel('$N$')
	if not args.subplots:
		plt.savefig(f'{path}/output_{i}/plot_cropped.png')
	else:
		plt.title(', '.join([f'{k}={v}' for k, v in info.items() if k in args.var]))
if args.subplots:
	if args.n % 2 == 0:
		plt.gcf().set_size_inches(10, 5 * (args.n // 2))
	else:
		plt.gcf().set_size_inches(15, 5 * (args.n // 3))
	plt.tight_layout()
	plt.savefig(f'{path}/plot_cropped.png')