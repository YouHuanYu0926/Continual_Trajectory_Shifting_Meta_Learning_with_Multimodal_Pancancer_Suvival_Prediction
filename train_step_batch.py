import os
import glob
import torch
from absl import app, flags, logging
import re

import model.data_loader
from model.model import Model
import model.utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_float("lr", 1e-2, "Learning rate")
EPOCH = 50
DATA_PATH = {'clinical': 'preprocess/meta-test/PC_clinical_emb.csv',
			'mRNA': 'preprocess/meta-test/PC_mRNA_threshold_7.csv',
			'miRNA': 'preprocess/meta-test/PC_miRNA.csv',
			'CNV': 'preprocess/meta-test/PC_CNV_threshold_20.csv'}
modalities_list = [['clinical', 'mRNA', 'miRNA']]
loss_obj = model.loss.Loss(trade_off=0.3, mode='total')
max_CIndex = 0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(argv):
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    data_path=DATA_PATH

    modalities = modalities_list[0]

    # create dataset
    mydataset = model.data_loader.MyDataset(modalities, data_path)

    # create sampler
    prepro_clin_data_X, _, prepro_clin_data_y, _ = model.data_loader.preprocess_clinical_data(data_path['clinical'])
    prepro_clin_data_X.reset_index(drop=True)
    prepro_clin_data_y.reset_index(drop=True)

    directory_path = 'Step/SGD'

    pth_files = glob.glob(os.path.join(directory_path, '*.pth'))
    print(pth_files)
    for file_path in pth_files:
        test_c_index_arr = []

        pattern = r'(\d+)_step\.pth$'

        # Extract the step number from the file path
        matches = re.search(pattern, file_path)

        train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)
        train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[[6]])

        for k, (train, test_val) in enumerate(train_testVal_kfold):
            # Create Train/validation/Test DataLoaders
            x_val, x_test, _, _ = train_test_split(prepro_clin_data_X.iloc[test_val, :],
                                                   prepro_clin_data_y.iloc[test_val, :][[6]], test_size=0.5,
                                                   random_state=24, stratify=prepro_clin_data_y.iloc[test_val, :][[6]])
            val, testData = list(x_val.index), list(x_test.index)
            dataloaders = model.utils.get_dataloaders(mydataset, train, val, testData, FLAGS.batch_size)
            # Create survival model
            survmodel = Model(
                modalities=modalities,
                m_length=128,
                dataloaders=dataloaders,
                fusion_method='attention',
                trade_off=0.3,
                mode='total',  # only_cox
                device=device)
            state_dict = torch.load(file_path)
            survmodel.load_state_dict(state_dict, strict=False)

            # Generate run tag
            run_tag = model.utils.compose_run_tag(
                model=survmodel, lr=FLAGS.lr, dataloaders=dataloaders,
                log_dir='training_logs/SGD/', suffix=' '+str(matches[0])
            )

            fit_args = {
                'num_epochs': EPOCH,
                'lr': FLAGS.lr,
                'info_freq': 2,
                'log_dir': os.path.join('traininglogs/SGD/'+str(matches[0]), run_tag),
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
                test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(),
                                                 event.cpu().numpy())
                test_c_index_arr.append(test_c_index.item())
            print(f'C-index on Test set: ', test_c_index.item())
        print('Mean and std: ', model.utils.evaluate_model(test_c_index_arr))
        model.utils.save_5fold_results(test_c_index_arr, run_tag)


if __name__ == "__main__":
    app.run(main)
