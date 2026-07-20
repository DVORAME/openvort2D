import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for i in range(1, 6):
	# Load the CSV file into a DataFrame
	df = pd.read_csv(f'output_avalanches/output_{i}/out.csv')

	# Plotting the data
	plt.subplot(3, 2, i)
	plt.plot(df['t'], df['N'])
	plt.title(f'Output {i}')

plt.tight_layout()
plt.savefig('output_avalanches/N_t.png')