import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy import nan
import copy
from sklearn import neighbors
import sys

sepsis_data = pd.read_csv('./data/train_prefilled010.csv')
p1 = list(sepsis_data['Patient_id'])
pids = list(set(sepsis_data['Patient_id']))
pids.sort(key=p1.index)
lab_attributes = ['EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Phosphate', 'Potassium', 'Bilirubin_total', 
'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

vital_signal = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp']
demographics = ['Age', 'Gender','HospAdmTime', 'ICULOS', 'SepsisLabel']

normal_range = {
	'EtCO2_min' : 35,
	'EtCO2_max' : 45,
	'HCO3_min' : 22,
	'HCO3_max' : 28,
	'pH_min' : 7.38,
	'pH_max' : 7.42,
	'PaCO2_min' : 38,
	'PaCO2_max' : 42,
	'SaO2_min' : 94,
	'SaO2_max' : 100,
	'AST_min' : 8,
	'AST_max' : 40,
	'BUN_min' : 6,
	'BUN_max' : 24,
	'Alkalinephos_min' : 40,
	'Alkalinephos_max' : 129,
	'Calcium_min' : 8.6,
	'Calcium_max' : 10.3,
	'Chloride_min' : 96,
	'Chloride_max' : 106,
	'Glucose_min' : 70,
	'Glucose_max' : 99,
	'Lactate_min' : 4.5,
	'Lactate_max' : 19.8,
	'Magnesium_min' : 8.5,
	'Magnesium_max' : 11,
	'Phosphate_min' : 2.8,
	'Phosphate_max' : 4.5,
	'Potassium_min' : 3.6,
	'Potassium_max' : 5.2,
	'Bilirubin_total_min' : 0.1,
	'Bilirubin_total_max' : 1.2,
	'TroponinI_min' : 0,
	'TroponinI_max' : 0.04,
	'PTT_min' : 25,
	'PTT_max' : 35,
	'WBC_min' : 4.5,
	'WBC_max' : 11.0,
	'Fibrinogen_min' : 200,
	'Fibrinogen_max' : 400,
	'Platelets_min' : 150,
	'Platelets_max' : 450,
	'BaseExcess_min' : -2,
	'BaseExcess_max' : +2,
}

total_num = len(pids)
epsilon = 0.01
for j, pid in enumerate(pids):
	print("\rPatient {}/{}.".format(j + 1, total_num), end="")
	sys.stdout.flush()
	one = sepsis_data[sepsis_data['Patient_id'] == pid]
	for attr in lab_attributes:
		miss_postion = one[attr].isna()
		num_not_nan = len(one[attr][~miss_postion])
		
		if num_not_nan <= 1:
			random_missing_indexs = random.sample(list(one[attr][miss_postion].index), int(len(one[attr][miss_postion])/2))
			for idx in random_missing_indexs:
				if attr == 'Creatinine':
					sepsis_data.loc[idx, attr] = np.random.uniform(0.6, 1.3)

				elif attr == 'Bilirubin_direct':
					sepsis_data.loc[idx, attr] = np.random.uniform(0, 0.3)

				elif attr == 'Hct':
					sepsis_data.loc[idx, attr] = np.random.uniform(35.51, 48.6)

				elif attr == 'Hgb':
					sepsis_data.loc[idx, attr] = np.random.uniform(35.51, 48.6)

				else:
					sepsis_data.loc[idx, attr] = np.random.uniform(normal_range[attr+'_min'], normal_range[attr+'_max'])
		
		else:
			if one[attr].max() - one[attr].min() == 0:
				values_position = one[attr][~miss_postion].index
				# print(values_position)
				random_index = random.sample(list(values_position), 2)
				
				sepsis_data.loc[random_index[0], attr] = sepsis_data.loc[random_index[0], attr] + epsilon
				sepsis_data.loc[random_index[1], attr] = sepsis_data.loc[random_index[1], attr] - epsilon

		
		
		
		
sepsis_data.to_csv("./data/train_prefilled010_2.csv")