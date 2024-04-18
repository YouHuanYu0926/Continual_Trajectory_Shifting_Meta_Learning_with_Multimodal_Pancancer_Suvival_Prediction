import os
from datetime import datetime

import torch
import torch.nn as nn
from absl import app, flags, logging
from torch.multiprocessing import Process
import model.utils
import model.data_loader
from model.model import Model
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from model import loss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

FLAGS = flags.FLAGS
# Training
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("train_steps", 100000, "Total training steps for a single run")
flags.DEFINE_enum("opt", "sgd", ["adam", "sgd", "rmsprop"], "optimizer")
flags.DEFINE_float("lr", 1e-2, "Learning rate")
flags.DEFINE_float("meta_lr", 1e-2, "Meta learning rate")
flags.DEFINE_integer("aggregate_period", 1, "Aggregate period")
flags.DEFINE_integer("reset_period", 100, "Reset period (K)")
flags.DEFINE_bool("hard_reset", False, "Reset model & opt or not")
flags.DEFINE_integer("print_every", 1000, "Print period")
# Data

EPOCH = 50


DATA_PATH = {'clinical': 'Pc_clinical_emb.csv',
			'mRNA': 'PC_mRNA_threshold_7.csv',
			'miRNA': 'PC_miRNA.csv',
			'CNV': 'PC_CNV_threshold_20.csv'}
modalities_list = [['clinical', 'mRNA', 'miRNA']]
loss_obj = model.loss.Loss(trade_off=0.3, mode='total')
max_CIndex = 0.7
device = torch.device("cpu")
file_path = "Pretrainoutput.txt"
file = open(file_path, "w")  # 使用 "w" 模式打开文件，如果文件不存在将创建新文件

def train_step(model, data_loader, base_param, aggregate, modalities,opt):
    model.train()
    # Sample data from training set
    for data, data_label in data_loader:
        out, event, time = model.predict(data, data_label)
        hazard, representation = out
        loss=loss_obj.forward(representation, modalities, hazard, event.to(device), time.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if aggregate:
            _base_param = {name: w.clone().detach() for name, w in base_param.items()}

            for name, w in model.named_parameters():
                if not w.requires_grad or "fc" in name:
                    continue
                base_param[name].data -= FLAGS.meta_lr * (base_param[name].data - w.data)

            for name, w in model.named_parameters():
                if not w.requires_grad or "fc" in name:
                    continue
                w.data += base_param[name] - _base_param[name]

def test(model, dataloader, test_c_index_arr,step):
    model.eval()
    print('testtest')
    for data, data_label in dataloader:
        out, event, time = model.predict(data, data_label)
        hazard, representation = out

        test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(),
                                         event.cpu().numpy())
        test_c_index_arr.append(test_c_index.item())
        file.write(f"Iteration {step}: Result = {test_c_index.item()}\n")
    print(f'C-index on Test set: ', test_c_index.item())


def run_single_process():
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    data_path = DATA_PATH
    modalities = modalities_list[0]

    # create dataset
    mydataset = model.data_loader.MyDataset(modalities, data_path)

    # preprocess clinical data
    prepro_clin_data_X, _, prepro_clin_data_y, _ = model.data_loader.preprocess_clinical_data(data_path['clinical'])
    prepro_clin_data_X.reset_index(drop=True)
    prepro_clin_data_y.reset_index(drop=True)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(prepro_clin_data_X, prepro_clin_data_y[[6]], test_size=0.2,
                                                        random_state=24, stratify=prepro_clin_data_y[[6]])

    # Create Train/Test DataLoaders
    train_indices = list(X_train.index)
    test_indices = list(X_test.index)
    dataloaders = {}
    dataloaders['train'] = DataLoader(mydataset, batch_size=FLAGS.batch_size, sampler=SubsetRandomSampler(train_indices))
    dataloaders['test'] = DataLoader(mydataset, batch_size=FLAGS.batch_size, sampler=SubsetRandomSampler(test_indices))

    # Create survival model
    survmodel = Model(
        modalities=modalities,
        m_length=128,
        dataloaders=dataloaders,
        fusion_method='attention',
        trade_off=0.3,
        mode='total',  # only_cox
        device=device
    )

    # Synchronize parameters at the beginning
    base_param = {
        name: w.clone().detach() for name, w in survmodel.named_parameters() if w.requires_grad and "fc" not in name
    }
    opt = survmodel.get_opt(FLAGS.lr)

    print(opt)

    # Generate run tag
    run_tag = model.utils.compose_run_tag(
        model=survmodel, lr=FLAGS.lr, dataloaders=dataloaders,
        log_dir='.training_logs/', suffix=''
    )

    for i in range(1, FLAGS.train_steps + 1):
        # model fitting
        train_step(survmodel, dataloaders['train'], base_param, i % FLAGS.aggregate_period == 0, modalities, opt)

        if i % FLAGS.print_every == 0:
            # Evaluate the model performance on test set
            test_c_index_arr = []
            test(survmodel, dataloaders['test'], test_c_index_arr, i)

        if i % FLAGS.reset_period == 0:
            # Resetting model parameters
            if FLAGS.hard_reset:
                survmodel = Model(
                    modalities=modalities,
                    m_length=128,
                    dataloaders=dataloaders,
                    fusion_method='attention',
                    trade_off=0.3,
                    mode='total',  # only_cox
                    device=device
                )
                opt = survmodel.get_opt(FLAGS.lr)
            survmodel.load_state_dict(base_param, strict=False)

        if i % 1000 == 0:
            torch.save(survmodel.state_dict(), f"model_{i}_step.pth")

        print(f"{i} step finish")

    torch.save(survmodel.state_dict(), "final_model.pth")
    print(f"{modalities} round: End")


def run_multi_process(argv):
    run_single_process()


if __name__ == "__main__":
    app.run(run_multi_process)
