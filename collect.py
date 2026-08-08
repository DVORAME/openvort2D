import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

D = 0.183146054323898
KAPPA = 9.96e-4

for i in range(1, 11):
	# Load the CSV file into a DataFrame
	df = pd.read_csv(f'output_avalanches/output_{i}/out.csv')

	# Plotting the data
	# plt.subplot(5, 2, i)
	plt.figure(figsize=(10, 10))
	plt.plot(df['t'], df['N'], label='N')
	plt.plot(df['t'], df['omega'] * np.pi * D**2 / 2 / KAPPA, label='N_exp')
	plt.title(f'Output {i}')
	plt.savefig(f'output_avalanches/output_{i}/plot.png')