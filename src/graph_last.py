import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path as path
from glob import glob


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', type=str, required=True, help='Input directory path')

	parser.parse_args()

	vp_files = glob(path.join(parser.i, 'vp_*.npz'))
	vp_files.sort()
	graph_file = np.load(vp_files[-1], allow_pickle=True)
	vp = graph_file['arr_0'].item()
	plt.plot(vp['xs'], vp['ys'], 'o', ms = 2)
	plt.savefig(path.join(parser.i, 'graph_last.png'), dpi=300)