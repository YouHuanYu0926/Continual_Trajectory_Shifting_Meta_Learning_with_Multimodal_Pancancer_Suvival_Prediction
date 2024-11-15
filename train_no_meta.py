# %pip install -r requirements.txt

import sys

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
import model.data_loader
from model.model import Model
import model.utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# Const
m_length = 128
BATCH_SIZE = 256
EPOCH = 50
lr = 0.01
K = 5


data_path=DATA_PATH = {'clinical': 'preprocess/meta-test/PC_clinical_emb.csv',
			'mRNA': 'preprocess/meta-test/PC_mRNA_threshold_7.csv',
			'miRNA': 'preprocess/meta-test/PC_miRNA.csv',
			'CNV': 'preprocess/meta-test/PC_CNV_threshold_20.csv'}
modalities_list = [['clinical', 'mRNA', 'miRNA','CNV']]

# setup random seed
model.utils.setup_seed(24)
# detect cuda
device = model.utils.test_gpu()


for modalities in modalities_list:
	if modalities[0] == 'clinical' or modalities[0] == 'CNV':
		lr = 0.01
	else:
		lr = 0.0005
	# create dataset
	mydataset = model.data_loader.MyDataset(modalities, data_path)

	# create sampler
	prepro_clin_data_X, _, prepro_clin_data_y, _ = model.data_loader.preprocess_clinical_data(data_path['clinical'])
	prepro_clin_data_X.reset_index(drop=True)
	prepro_clin_data_y.reset_index(drop=True)
	train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)
	train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[[6]])

	test_c_index_arr = []
	for k, (train, test_val) in enumerate(train_testVal_kfold):
		# Create Train/validation/Test DataLoaders
		x_val, x_test, _, _ = train_test_split(prepro_clin_data_X.iloc[test_val, :], prepro_clin_data_y.iloc[test_val, :][[6]], test_size=0.5, random_state=24, stratify=prepro_clin_data_y.iloc[test_val, :][[6]])
		val, test = list(x_val.index), list(x_test.index)
		dataloaders = model.utils.get_dataloaders(mydataset, train, val, test, BATCH_SIZE)

		# Create survival model
		survmodel = Model(
			modalities=modalities,
			m_length=m_length,
			dataloaders=dataloaders,
			fusion_method='attention',
			trade_off=0.3,
			mode='total', # only_cox
			device=device)
		# Generate run tag
		run_tag = model.utils.compose_run_tag(
			model=survmodel, lr=lr, dataloaders=dataloaders,
			log_dir='.training_logs/', suffix=''
		)

		fit_args = {
		'num_epochs': EPOCH,
		'lr': lr,
		'info_freq': 2,
		'log_dir': os.path.join('.training_logs/', run_tag),
		'lr_factor': 0.5,
		'scheduler_patience': 7,
		}
		# model fitting
		survmodel.fit(**fit_args)

		# Load the best weights on validation set and test the model performance on test set!
		survmodel.test()
		for data, data_label in dataloaders['test']:
			out, event, time = survmodel.predict(data, data_label)
			hazard, representation = out
			test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(), event.cpu().numpy())
			test_c_index_arr.append(test_c_index.item())
		print(f'C-index on Test set: ', test_c_index.item())

	print('Mean and std: ', model.utils.evaluate_model(test_c_index_arr))
	model.utils.save_5fold_results(test_c_index_arr, run_tag)
