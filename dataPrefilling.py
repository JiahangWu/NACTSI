import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy import nan
import copy
from sklearn import neighbors
import sys

# sepsis_data = pd.read_csv('./data/fuzzyKNN015.csv')
sepsis_data = pd.read_csv('./data/fuzzyKNN010.csv')
p1 = list(sepsis_data['Patient_id'])
pids = list(set(sepsis_data['Patient_id']))
print(len(pids))
sys.exit()
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

theta = 0.5

for i in range(len(sepsis_data)):
	data = sepsis_data[i:i+1]
	gender = data['Gender'].values[0]
	

	print("\r {}/{}.".format(i + 1, len(sepsis_data)), end="")
	sys.stdout.flush()
			
	for attr in lab_attributes:
		attr_l = attr + '_nomal_label'

		if attr == 'Creatinine' and data[attr_l].values[0] >= theta and np.isnan(data[attr].values[0]):
			if gender == 1:
				sepsis_data.loc[i, attr] = np.random.uniform(0.7, 1.3)
			else:
				sepsis_data.loc[i, attr] = np.random.uniform(0.6, 1.1)

		elif attr == 'Bilirubin_direct' and data[attr_l].values[0] >= theta and np.isnan(data[attr].values[0]):
			sepsis_data.loc[i, attr] = np.random.uniform(0, 0.3)

		elif attr == 'Hct' and data[attr_l].values[0] >= theta and np.isnan(data[attr].values[0]):
			if gender == 1:
				sepsis_data.loc[i, attr] = np.random.uniform(38.3, 48.6)
			else:
				sepsis_data.loc[i, attr] = np.random.uniform(35.51, 44.9)

		elif attr == 'Hgb' and data[attr_l].values[0] >= theta and np.isnan(data[attr].values[0]):
			if gender == 1:
				sepsis_data.loc[i, attr] = np.random.uniform(38.3, 48.6)
			else:
				sepsis_data.loc[i, attr] = np.random.uniform(35.51, 44.9)
		else:
			if data[attr_l].values[0] >= theta and np.isnan(data[attr].values[0]):
				sepsis_data.loc[i, attr] = np.random.uniform(normal_range[attr+'_min'], normal_range[attr+'_max'])
		
	# break


print('\n')
sepsis_data.to_csv("./data/train_prefilled010.csv")