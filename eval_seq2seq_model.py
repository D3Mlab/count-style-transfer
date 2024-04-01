
from src.eval_checkpoint import detox_experiment

from src.detoxifier import Seq2SeqDetoxifier




import argparse
import pandas as pd
from src.dataset import get_APPDIA_train_and_val_loaders, get_paradetox_train_and_val_datasets,get_paradetox_train_and_val_loaders
import os
import torch

parser = argparse.ArgumentParser(description='Eval Seq2Seq')

parser.add_argument('--model_name', type=str, default="facebook/bart-base") # Path to model or model from hugginface, must have BART backbone
parser.add_argument('--tokenizer_name', type=str, default="facebook/bart-base")




parser.add_argument('--save_path', type=str, default="eval_results/test")
parser.add_argument('--dataset', type=str, default="paradetox")
parser.add_argument('--fold', type=str, default="test")
parser.add_argument('--name', type=str, default="Evaluation_name")




parser.add_argument('--make_preds', action='store_true')
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()

assert args.make_preds==True or args.evaluate==True, "at least one of make_preds or evaluate should be True"

if len(args.save_path)==0:
    save_path = os.path.join(args.model_name,args.name)
else:
    save_path = args.save_path


logit_processors = None









detoxifier_class = Seq2SeqDetoxifier





if args.dataset == 'paradetox':
    sep = ','
    if args.fold == 'test':
        test_data_path='datasets/paradetox/paradetox_test.csv'
    
        eval_def = pd.read_csv(test_data_path,sep=sep)
    elif args.fold =='val':
        train_dataset, val_dataset = get_paradetox_train_and_val_loaders(1)
        
        eval_def = pd.DataFrame(val_dataset)
        eval_def = eval_def.applymap(lambda x: x[0])  
    else:
        assert False, 'fold undefined!'
elif args.dataset == 'appdia':
    sep ='\t'
    if args.fold == 'test':
        test_data_path='datasets/APPDIA/test.tsv'
        eval_def = pd.read_csv(test_data_path,sep=sep)
    elif args.fold =='val':
        val_data_path='datasets/APPDIA/validation.tsv'
        eval_def = pd.read_csv(val_data_path,sep=sep)
    else:
        assert False, 'fold undefined!'
else:
    assert False, 'dataset undefined!'


detox = detoxifier_class(model_name=args.model_name,tokenizer_name=args.tokenizer_name)

detox_experiment(save_path,detoxifier=detox,test_dataframe=eval_def,name=args.name,make_preds=args.make_preds, evaluate=args.evaluate)