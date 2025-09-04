#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

all_data = []
# iterate over all subdirectories in results, and then load each csv file into a pandas dataframe
for subdir in os.listdir('results'):
    if not os.path.isdir(os.path.join('results', subdir)):
        continue
    for file in os.listdir(os.path.join('results', subdir)):
        results = pd.read_csv(os.path.join('results', subdir, file))
        #print(results.head())
        #print(results.describe())
        all_data.append(results)
df = pd.concat(all_data)

print( len(df) )


# %%

for model in df['model_name'].unique():
    try:  
        correct = df[df['model_name'] == model]['is_correct'].value_counts()[True]
    except:
        correct = 0
    total = len(df[df['model_name'] == model])
    print(f"{model}: Correct: {correct}, Total: {total}, Accuracy: {correct/total}")



# %%
