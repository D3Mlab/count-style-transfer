
from seq2seq.eval_checkpoint import paradetox_experiment

from seq2seq.detoxifier import Seq2SeqDetoxifier, TokenReRankingDetoxifier, MultiLMDetoxifier
from seq2seq.multiobjective.pareto_beam_search import ParetoLogitsProcessor

from seq2seq.multiobjective.scorer import Scorer

import argparse
import pandas as pd
from seq2seq.dataset import get_APPDIA_train_and_val_loaders, get_paradetox_train_and_val_datasets,get_paradetox_train_and_val_loaders
from seq2seq.multiobjective.scalarizers import LinearScalarizer, TorchFunctionScalarizer, RoundRobinScalarizer, HarmonicMeanScalarizer, RoundRobinScottScalarizer, ConstrainedScalarizer
import os
import torch

parser = argparse.ArgumentParser(description='Eval Seq2Seq')

parser.add_argument('--model_name', type=str, default="facebook/bart-base") # Path to model or model from hugginface, must have BART backbone
parser.add_argument('--tokenizer_name', type=str, default="facebook/bart-base")



parser.add_argument(
  "--additional_models",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
#   default=[1, 2, 3],  # default if nothing is provided
)

parser.add_argument('--detoxifier_type', type=str, default="seq2seq")
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--logit_processor', type=str, default="none")

parser.add_argument('--scalarizer', type=str, default="linear")
parser.add_argument('--filter_dominateds', action='store_true')



parser.add_argument('--round_robin', action='store_true')


parser.add_argument('--do_mbr', action='store_true')
parser.add_argument('--use_prefix_prob', action='store_true')

parser.add_argument('--save_path', type=str, default="eval_results/test")
parser.add_argument('--dataset', type=str, default="paradetox")
parser.add_argument('--fold', type=str, default="test")
parser.add_argument('--name', type=str, default="Evaluation_name")

parser.add_argument('--do_types', action='store_true')

parser.add_argument('--use_old_checkpoints', action='store_true')
parser.add_argument('--make_preds', action='store_true')
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()
# print(args)
assert args.make_preds==True or args.evaluate==True, "at least on of make_preds or evaluate should be True"

if len(args.save_path)==0:
    save_path = os.path.join(args.model_name,args.name)
else:
    save_path = args.save_path

detoxifier_classes = {'seq2seq':Seq2SeqDetoxifier,'token_reranking':TokenReRankingDetoxifier,'multi_lm':MultiLMDetoxifier}
logit_processors = None
if args.logit_processor=='pareto':
    logit_processors = [ParetoLogitsProcessor(scorer=Scorer(),model_tokenizer_name=args.tokenizer_name)]


# scalarizers = {'linear':LinearScalarizer([1.0,0.0,0.0])}
scalarizers = {'linear':LinearScalarizer([0.33,0.33,0.33]),'rr':RoundRobinScalarizer(),'harmonic':HarmonicMeanScalarizer(),
'min':TorchFunctionScalarizer(torch.min),'max':TorchFunctionScalarizer(torch.max),'rr_scott':RoundRobinScottScalarizer(),'constrained':ConstrainedScalarizer()
}
scalarizer = scalarizers[args.scalarizer]



kwargs={'logits_processor_list':logit_processors,'scalarizer':scalarizer,'additional_model_names':args.additional_models,'filter_dominateds':args.filter_dominateds,
'do_mbr': args.do_mbr,'round_robin':args.round_robin,'use_prefix_prob':args.use_prefix_prob}


detoxifier_class = detoxifier_classes[args.detoxifier_type]





if args.dataset == 'paradetox':
    sep = ','
    if args.fold == 'test':
        test_data_path='datasets/paradetox/paradetox_test.csv'
    
        eval_def = pd.read_csv(test_data_path,sep=sep)
    elif args.fold =='val':
        train_dataset, val_dataset = get_paradetox_train_and_val_loaders(1)
        # print(val_dataset)
        # sdksjdkj()
        
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
    assert False, 'datset undefined!'


# if args.detoxifier_type == 'multi_lm':
detox = detoxifier_class(model_name=args.model_name,tokenizer_name=args.tokenizer_name,prefix=args.prefix,**kwargs)
# # else:
#     detox = detoxifier_class(model_name=args.model_name,tokenizer_name=args.tokenizer_name,**kwargs)
paradetox_experiment(save_path,detoxifier=detox,test_dataframe=eval_def,do_types=args.do_types,name=args.name,use_old_checkpoints=args.use_old_checkpoints,make_preds=args.make_preds, evaluate=args.evaluate)