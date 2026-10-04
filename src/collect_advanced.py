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

args = parser.parse_args()
path = args.path

D_frac = 21/23
KAPPA = 9.96e-4

for i in range(1, 2):
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
	# plt.subplot(5, 2, i)
	plt.figure(figsize=(10, 10))
	plt.plot(df['t'], N, label='N')
	plt.plot(df['t'], df['omega'] * np.pi * D**2 / 2 / KAPPA, label='N_exp')
	plt.title(f'Output {i}')
	plt.savefig(f'{path}/output_{i}/plot_cropped.png')