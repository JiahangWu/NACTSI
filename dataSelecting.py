import sys
import os
import pickle
import numpy as np
import pandas as pd
import random
import math


print('Loading data...')
data_path = './data/sepsis_data.csv'
raw_data = pd.read_csv(data_path)
lab_attributes = [	'EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
					'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Phosphate', 'Potassium', 'Bilirubin_total', 
					'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
lab_label = [att + '_nomal_label' for att in lab_attributes]
vital_signal = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp']
demographics = ['Age', 'Gender','HospAdmTime', 'ICULOS', 'SepsisLabel']
p1 = list(raw_data['Patient_id'])
pids = list(set(raw_data['Patient_id']))
pids.sort(key=p1.index)
print('number of patients: ', len(pids))
raw_data = raw_data[['Patient_id'] + vital_signal + lab_attributes + demographics + lab_label]
raw_data_l = raw_data[['Patient_id'] + lab_attributes ]

threshold = 0.1
print("Threshold: ", threshold)

print('Processing and counting...')
train_dataset = pd.DataFrame(columns=['Patient_id'] + vital_signal + lab_attributes + demographics + lab_label)
test_dataset = pd.DataFrame(columns=['Patient_id'] + vital_signal + lab_attributes + demographics + lab_label)
tr_cnt = 0
te_cnt = 0
for i, pid in enumerate(pids):
	one = raw_data_l[raw_data_l['Patient_id'] == pid]
	tmp = raw_data[raw_data['Patient_id'] == pid]
	one = np.array(one)[:, 1:]
	one = np.array(one, dtype=float)
	num_not_nan = len(one[~np.isnan(one)])
	if num_not_nan / (one.shape[0]*one.shape[1]) >= threshold:
		tr_cnt += 1
		train_dataset = train_dataset.append(tmp, ignore_index=True)
	else:
		te_cnt += 1
		test_dataset = test_dataset.append(tmp, ignore_index=True)
	
	print("\rpatient {}/{}, pid {}, train_cnt {}, test_cnt {}.".format(i, len(pids), pid, tr_cnt, te_cnt), end="")
	sys.stdout.flush()
	# if i == 2:
	# 	 break
# print(train_dataset)
# print(test_dataset)

print('Writing .csv files...')
# train_dataset.to_csv('./data/train_data'+str(threshold)+'.csv')
# test_dataset.to_csv('./data/test_data'+str(threshold)+'.csv')

# x = np.array(one_patient)[:, 1:]
# x = np.array(x, dtype=float)
# print(len(x[~np.isnan(x)]))
# print(len(x[np.isnan(x)]))
# print(x.shape)


# x = raw_data[0: 5]

# y = x.drop([0, 1])
# print(y)

# l1 = ['a', 'a', 'b','c','d','b','c','a','a']
# l2 = list(set(l1))
# l2.sort(key=l1.index)
# print(l2)
