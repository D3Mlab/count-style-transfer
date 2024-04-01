




import os.path
import pandas as pd
import numpy as np
from evaluation.eval import evaluate,format_number

import torch
from tqdm import tqdm

import json
import os

def evaluate_results_list(res, test_dataframe,name='exp_name',pred_file_name='preds.txt'):

    toxic_sentences = []
	
    # test_cases = pd.read_csv(test_data_path,sep=sep)
    toxic_sentences = test_dataframe['en_toxic_comment'].values
    refs = test_dataframe['en_neutral_comment'].values
    name_to_add = ''

    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    # per_sent_file_name=os.path.join(None, name_to_add + 'per_sent_'+name+'.csv')
    df, df2 = evaluate(toxic_sentences, res, refs,types=None, name=name,
                       save_path=None,
                       per_sent_file_name=None)
    

    return df, df2



def evaluate_results(save_path, test_dataframe,name='exp_name',pred_file_name='preds.txt'):
    print('Evaling',save_path)
    toxic_sentences = []
	
    # test_cases = pd.read_csv(test_data_path,sep=sep)
    toxic_sentences = test_dataframe['en_toxic_comment'].values
    refs = test_dataframe['en_neutral_comment'].values



    with open(os.path.join(save_path, pred_file_name), mode='r', encoding='utf-8') as f:
        res = f.readlines()


    print("res", len(res), "input", len(toxic_sentences))

    res = [str(r.replace('\n', ' ').replace('\r', ' '))for r in res]
    name_to_add = ''
 
    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    per_sent_file_name=os.path.join(save_path, name_to_add + 'per_sent_'+name+'.csv')
    df, df2 = evaluate(toxic_sentences, res, refs,types=None, name=name,
                       save_path=os.path.join(save_path, name_to_add + 'eval_'+name+'.csv'),
                       per_sent_file_name=per_sent_file_name)
    
    print(df[['model','STA','ref_SIM','SIM','FL','J']].to_string())







def detox_experiment(save_path,detoxifier,test_dataframe,name="exp_name", make_preds=False,evaluate=False):

    
    
    if make_preds:
        toxic_sentences = []
        # test_cases = pd.read_csv(test_data_path,sep=sep)
        toxic_sentences = test_dataframe['en_toxic_comment'].values
        refs = test_dataframe['en_neutral_comment'].values


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res = []
        


        for toxic_sen in tqdm(toxic_sentences):

            pred = detoxifier.get_output(toxic_sen)
            res.append(pred)
        res = [r.replace('\n', ' ').replace('\r', '') for r in res]
    
        with open(os.path.join(save_path, 'preds.txt'), mode='w', encoding='utf-8') as f:
            f.write("\n".join(res))

    if evaluate:
        evaluate_results(save_path, name=name,test_dataframe=test_dataframe)



def get_output(model, tokenizer, x, device='cuda'):
    input_ids =tokenizer.encode(x)
    input_ids = torch.tensor(input_ids).reshape(1,-1).to('cuda')
    # print(input_ids)
    with torch.no_grad():
        beam_output =  model.generate(input_ids,  max_length=50, do_sample=True, early_stopping=True)
    texts = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
    texts = texts[0]


    return texts














def eval_file(save_path,test_data_path='datasets/paradetox/paradetox_test.csv', use_old_checkpoints=False,name="some_name",do_types=True,sep=','):

    toxic_sentences = []
    test_cases = pd.read_csv(test_data_path,sep=sep)
    toxic_sentences = test_cases['en_toxic_comment'].values
    refs = test_cases['en_neutral_comment'].values
    # types = test_cases['Type 1'].values

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res = []
    with open(os.path.join(save_path, 'preds.txt'), mode='r', encoding='utf-8') as f:
        res = f.readlines()

    # try:
        # if not os.path.exists(os.path.join(save_path, 'eval.csv')):
        #     print('Evaling',save_path)
    evaluate_results(save_path, name=name,use_old_checkpoints=use_old_checkpoints,test_data_path=test_data_path,do_types=do_types,sep=sep)
    # except Exception as e: 
    #     print(save_path, "Evaluation had Error!")
    #     print(e)



