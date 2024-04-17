import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def detect_outliers(data):
    outliers=[]
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
    
    for i in data:
        z_score= (i - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

#read the csv file
data_path = 'labelers_weight.txt'
#set divider of cvs as ; and no coliumn names
df = pd.read_csv(data_path, sep=';', header=None)


#set the column names
df.columns = ['labeler', 'weight','lbl 0','lbl 1','lbl 2','lbl 3','lbl 255']
#set type of labeler as string
df['labeler'] = df['labeler'].astype(str)
#set type of weight as float casting the column
df['weight'] = df['weight'].replace(',', '.')
df['weight'] = df['weight'].astype(float)
dimension = 1024*1024
df['weight'] = df['weight']/dimension*100


print(df)
print(df.info())

#calculate the mean the medianm the mode and teh standard deviation
mean = df['weight'].mean()
median = df['weight'].median()
mode = df['weight'].mode()
std = df['weight'].std()

print('Mean: ', mean)
print('Median: ', median)
print('Standard Deviation: ', std)

outliers = detect_outliers(df['weight'])
plt.hist(outliers, bins=20)
print(len(outliers)/len(df['weight'])*100)
print(len(outliers))
print(len(df['weight']))

#drop the outliers
#df = df[~df['weight'].isin(outliers)]
df_save = df[df['weight'].isin(outliers)]
df_save = df_save['labeler']
df_save.to_csv('outliers.txt', sep=';', index=False)