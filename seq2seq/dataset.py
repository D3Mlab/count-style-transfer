


from torch.utils.data import DataLoader
import pandas as pd
import torch
from datasets import load_dataset

import torch.utils.data as data_utils
import os


def get_paradetox_train_and_val_datasets():

    dataset = load_dataset("SkolkovoInstitute/paradetox", "en-US", split="train")

    N = len(dataset)

    train_size = int(0.8* N)
    test_size = N - train_size

    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=generator1)

    return train_dataset, val_dataset

def get_paradetox_train_and_val_loaders(batch_size=8):


    train_dataset, val_dataset = get_paradetox_train_and_val_datasets()

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def get_APPDIA_train_and_val_loaders(path='/media/Storage/CTG/text-detox/datasets/APPDIA',batch_size=8):

    train_dataset = load_dataset(path,split="train")
    val_dataset = load_dataset(path,split="validation")
    # train_dataset = pd.read_csv(os.path.join(path,'train.tsv'), sep = '\t')
    # val_dataset = pd.read_csv(os.path.join(path,'validation.tsv'), sep = '\t')

    # print(train_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader