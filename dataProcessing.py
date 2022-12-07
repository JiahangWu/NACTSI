import sys
import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from utils import replace_nan_with_col_mean, construct_delta_matrix



def preprocess_training_data():
	print('Pre-processing training data...')
	data_path = './data/train_prefilled010_2.csv'
	# data_path = './data/sepsis_data.csv'
	raw_data = pd.read_csv(data_path)
	lab_attributes = [	'EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
						'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Phosphate', 'Potassium', 'Bilirubin_total', 
						'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
	vital_signal = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp']
	demographics = ['Age', 'Gender','HospAdmTime', 'ICULOS', 'SepsisLabel']
	p1 = list(raw_data['Patient_id'])
	pids = list(set(raw_data['Patient_id']))
	pids.sort(key=p1.index)
	pids = pids[:int(len(pids)*0.9)]
	# raw_data = raw_data[['Patient_id'] + vital_signal + lab_attributes + demographics]
	raw_data = raw_data[['Patient_id'] + lab_attributes]
	
	total_num = len(pids)
	print('total num:', total_num)
	threshold_num = 5
	num_drop = 1

	training_set = []
	for i, pt in enumerate(pids):
		print("\rPatient {}/{}.".format(i + 1, total_num), end="")
		sys.stdout.flush()
		
		one_pt = raw_data[raw_data['Patient_id'] == pt]
		ts = len(one_pt)

		time_stamps = np.array([ts for ts in range(ts)])
		pt_with_na, pt_ground_truth = one_pt.values[:, 1:], one_pt.values[:, 1:]
		pt_with_na = np.array(pt_with_na, dtype=float)
		pt_ground_truth = np.array(pt_ground_truth, dtype=float)
		ptmax = np.nanmax(pt_ground_truth, axis=0).reshape(1, -1)
		ptmin = np.nanmin(pt_ground_truth, axis=0).reshape(1, -1)

		# Randomly drop num_drop value in each lab_attributes for training
		for j, attr in enumerate(lab_attributes):
			miss_postion = one_pt[attr].isna()
			num_not_nan = len(one_pt[attr][~miss_postion])
			if num_not_nan <= threshold_num:
				continue
			value_indices = np.where(~miss_postion)[0]
			drop_indices = np.random.choice(value_indices, num_drop)
			pt_with_na[drop_indices,j] = np.nan

		pt_with_na, missing_flag = replace_nan_with_col_mean(pt_with_na)
		# Note: NaN in ground truth will not be used. Just to avoid NaN in pytorch which does not support nanmean() etc.
		pt_ground_truth, missing_flag_gt = replace_nan_with_col_mean(pt_ground_truth)
		eval_mask = (~missing_flag_gt) & missing_flag  # 1: locs masked for eval
		observed_mask = (~missing_flag).astype(float)  # 1: observed, 0: missing
		eval_mask = eval_mask.astype(float)
		
		

		delta = construct_delta_matrix(pt_with_na, time_stamps, observed_mask)
		train_data = {
			'pt_with_na': pt_with_na,
			'pt_ground_truth': pt_ground_truth,
			'time_stamps': time_stamps,
			'observed_mask': observed_mask,
			'eval_mask': eval_mask,
			'pt_max': ptmax,
			'pt_min': ptmin,
			'length': pt_with_na.shape[0],
			'delta': delta,
			'pid': pt
		}
		training_set.append(train_data)
		
		# print(train_data['eval_mask'])
	print('\nTrain data processing finished!!!')
	fw = open('./data/trainData01.pkl','wb')
	pickle.dump(training_set, fw, -1)
	fw.close()


def preprocess_testing_data():
	print('Pre-processing testing data...')
	data_path = './data/train_prefilled010_2.csv'
	raw_data = pd.read_csv(data_path)
	lab_attributes = [	'EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
						'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Phosphate', 'Potassium', 'Bilirubin_total', 
						'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
	vital_signal = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp']
	demographics = ['Age', 'Gender','HospAdmTime', 'ICULOS', 'SepsisLabel']
	p1 = list(raw_data['Patient_id'])
	pids = list(set(raw_data['Patient_id']))
	pids.sort(key=p1.index)
	pids = pids[int(len(pids)*0.9):]
	# raw_data = raw_data[['Patient_id'] + vital_signal + lab_attributes + demographics]
	raw_data = raw_data[['Patient_id'] + lab_attributes]
	
	total_num = len(pids)
	print('total num:', total_num)

	testing_set = []
	for i, pt in enumerate(pids):
		
		print("\rPatient {}/{}.".format(i + 1, total_num), end="")
		sys.stdout.flush()
		
		one_pt = raw_data[raw_data['Patient_id'] == pt]
		ts = len(one_pt)

		time_stamps = np.array([ts for ts in range(ts)])
		pt_with_na = (one_pt.values[:, 1:]).astype(float)
		pt_with_na, missing_flag = replace_nan_with_col_mean(pt_with_na)
		ptmax = np.nanmax(pt_with_na, axis=0).reshape(1, -1)
		ptmin = np.nanmin(pt_with_na, axis=0).reshape(1, -1)
		observed_mask = (~missing_flag).astype(float)  # 1: observed, 0: missing
		delta = construct_delta_matrix(pt_with_na, time_stamps, observed_mask)


		
		test_data = {
			'pt_with_na': pt_with_na,
			'time_stamps': time_stamps,
			'observed_mask': observed_mask,
			'pt_max': ptmax,
			'pt_min': ptmin,
			'length': pt_with_na.shape[0],
			'delta': delta,
			'pid': pt
		}
		testing_set.append(test_data)
	print('\nTest data processing finished!!!')
	fw = open('./data/testData01.pkl','wb')
	pickle.dump(testing_set, fw, -1)
	fw.close()
		


if __name__ == '__main__':
	preprocess_training_data()
	preprocess_testing_data()
