import pandas as pd
import numpy as np

#Select path

pathCDL = 'images/MNIST/CDL/'
pathDL = 'images/MNIST/DL/'
pathCSAE = 'images/MNIST/CSAE/'

#Select dict size 

# 3x3, 5x5, 7x7, 9x9, 11x11, 13x13, 15x15, 17x17

df = pd.read_csv(pathCDL + '17x17.csv')

# Searching for best sparsity of each lambda in a dict

lmbda_vals = df.lmbda.unique()
print(lmbda_vals)

for i in lmbda_vals:
    df2 = df.loc[df['lmbda'] == i]
    print('lambda: ' + str(i))
    per_mean = df2.groupby('dict_size', as_index=False)['percent'].mean()
    min_five = per_mean.sort_values('percent', ascending = False)[:5]
    print(min_five)
    min_five.to_csv(pathCSAE + '5x5_best_lambda_' + str(i) + '.csv', index=False)
    
  
df3 = df.loc[df['lmbda'] == 0.5]
df4 = df3.loc[df3['dict_size'] == 25]
print(df4)
#Errors

path_err = 'data_lmbda_'


df_err = pd.read_csv(pathCSAE + '5x5x10/lmbda_0.5/' + path_err + '0.5' + '.csv')

#Calculate average absolute and mean errors for each number

print('***************  Mean of ABSOLUTE ERROR for each number   *****************************')

abs_err = df_err.groupby('number', as_index=False)['mean_absolute_error'].mean()
print(abs_err)
  
    
#print('***************  Mean of MEAN ERROR for each number   *****************************')

#mean_err = df_err.groupby('number', as_index=False)['mean_error'].mean()
#mean_err.drop('number', axis=1, inplace=True)

#res = pd.concat([abs_err, mean_err], axis=1)

abs_err.to_csv(pathCSAE + 'errors_5x5x10.csv', index=False)
