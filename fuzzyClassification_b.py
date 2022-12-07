
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy import nan
import copy
from sklearn import neighbors
from fuzzyKNN import FuzzyKNN

print('Loading data...')
sepsis_data = pd.read_csv('./data/train_data01.csv')
select_num = int(len(sepsis_data) )
sepsis_data = sepsis_data[:select_num]
print('number of data: ', len(sepsis_data))

print('Loading finished...')

lab_attributes = ['EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Phosphate', 'Potassium', 'Bilirubin_total', 
 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

static_attributes = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp', 'Age', 'Gender','HospAdmTime', 'ICULOS', 'SepsisLabel']

k_dict = {'EtCO2': 3, 
		'BaseExcess': 7, 
		'HCO3': 15, 
		'pH': 13, 
		'PaCO2': 15, 
		'SaO2': 9, 
		'AST': 13, 
		'BUN': 7, 
		'Alkalinephos': 11, 
		'Calcium': 11, 
		'Chloride': 1, 
		'Creatinine': 11, 
		'Bilirubin_direct': 7, 
		'Glucose': 9, 
		'Lactate': 7, 
		'Phosphate': 11, 
		'Potassium': 9, 
		'Bilirubin_total': 13, 
		'TroponinI': 15, 
		'Hct': 15, 
		'Hgb': 11, 
		'PTT': 15, 
		'WBC': 15, 
		'Fibrinogen': 3, 
		'Platelets': 9}



idx = int(sys.argv[1])
attr = lab_attributes[idx]

attr_l = attr + '_nomal_label'
print('Classify', attr_l)
data = sepsis_data[['Patient_id'] + static_attributes+[attr_l]]
true_sample = data.dropna(axis=0,how='any')
X = np.array(true_sample[static_attributes])
y = np.array(true_sample[attr_l])
if len(set(y)) != 2:
	print('Warning', attr_l)
	sys.exit()


test_sample = copy.deepcopy(data)
v = []
for i in range(len(data)):
	value = test_sample[attr_l][i:i+1].values
	idx = test_sample[attr_l][i:i+1].index[0]
	if not np.isnan(value):
		v.append(idx)
# print(len(v))
test_sample = test_sample.drop(index=v)
test = np.array(test_sample[static_attributes])
print('Number of training data:', len(X))
print('Number of predicting data:', len(test))

k_neighbors = k_dict[attr]


fuzzyClf =  FuzzyKNN(k_neighbors)
print('Best K:', k_neighbors)
print('Starting training...')
fuzzyClf.fit(X, y)
print('Starting predicting...')
predicted_label = fuzzyClf.predict(test)

#     print(predicted_label)

indexs = [idx for idx in range(len(sepsis_data))]
for idx in v:
	if idx in indexs:
		indexs.remove(idx)
# print(len(indexs))
print('Writing the predicted data into file...')
for j in range(len(indexs)):
	if np.isnan(sepsis_data.loc[indexs[j], attr_l]):
		sepsis_data.loc[indexs[j], attr_l] = predicted_label[j][1][1]
	else:
		print(indexs[j])
		print(sepsis_data.loc[indexs[j], attr_l])
		print('ERROR!!!')

print('Finished!!!')
	
sepsis_data.to_csv("./data/010/" + str(attr) + ".csv")

