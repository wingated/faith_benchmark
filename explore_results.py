#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load the results
#results = pd.read_csv('results/results_openai.csv')
#results = pd.read_csv('results/results_gemini.csv')
results = pd.read_csv('results/results_claude.csv')

# Explore the data
print(results.head())
print(results.describe())
# %%

correct = results['model_answer'].value_counts()['A']
total = len(results)

print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}")




# %%
