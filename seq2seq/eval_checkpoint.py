




import os.path
import pandas as pd
import numpy as np
from paradetox.eval import evaluate,format_number
from paradetox.utils import get_profanity_list, clean_curse
import torch
from tqdm import tqdm
from statics import defaults
import json
import os

def evaluate_results_list(res, test_dataframe, use_old_checkpoints=False,name='few_shot',pred_file_name='preds.txt',do_types=True):

    toxic_sentences = []
	
    # test_cases = pd.read_csv(test_data_path,sep=sep)
    toxic_sentences = test_dataframe['en_toxic_comment'].values
    refs = test_dataframe['en_neutral_comment'].values
    if do_types:
        types = test_cases['Type 1'].values
        # types = test_dataframe['Type second annotation'].values
        
    else:
        types = None


    # with open(os.path.join(save_path, pred_file_name), mode='r', encoding='utf-8') as f:
    #     res = f.readlines()


    # res = [str(r.replace('\n', ' ').replace('\r', ' '))for r in res]
    name_to_add = ''

    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    # per_sent_file_name=os.path.join(None, name_to_add + 'per_sent_'+name+'.csv')
    df, df2 = evaluate(toxic_sentences, res, refs,types=types, name=name,
                       save_path=None,
                       per_sent_file_name=None)
    

    return df, df2



def evaluate_results(save_path, test_dataframe, use_old_checkpoints=False,name='few_shot',pred_file_name='preds.txt',do_types=True):
    print('Evaling',save_path)
    toxic_sentences = []
	
    # test_cases = pd.read_csv(test_data_path,sep=sep)
    toxic_sentences = test_dataframe['en_toxic_comment'].values
    refs = test_dataframe['en_neutral_comment'].values
    if do_types:
        types = test_cases['Type 1'].values
        # types = test_dataframe['Type second annotation'].values
        
    else:
        types = None

    # with open('datasets/paradetox/raw/test_toxic_parallel.txt', encoding='utf-8', mode='r') as f:
    #     toxic_sentences = f.readlines()
    # refs = []
    # with open('datasets/paradetox/raw/test_toxic_parallel_refs.txt', encoding='utf-8', mode='r') as f:
    #     refs = f.readlines()
    # res = []
    with open(os.path.join(save_path, pred_file_name), mode='r', encoding='utf-8') as f:
        res = f.readlines()


    print("res", len(res), "input", len(toxic_sentences))
    # for i, r in enumerate(res):
    #     if len(r) < 4:
            # print('sentence:', [r], 'replacing with source')
            # res[i] = toxic_sentences[i]
        # elif len(r) < 10:
        #     print('sentence:', [r])
    # ref_sentences = ref_sentences[:limit]
    res = [str(r.replace('\n', ' ').replace('\r', ' '))for r in res]
    name_to_add = ''
    # if filter_profanity:
    #     name_to_add += 'filtered_'
    # print("res", len(res), "input", len(toxic_sentences))
    # if filter_profanity:
    #     profanity = get_profanity_list()
    #     res, _ = clean_curse(res, profanity)
    # print("res", len(res), "input", len(toxic_sentences))
    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    per_sent_file_name=os.path.join(save_path, name_to_add + 'per_sent_'+name+'.csv')
    df, df2 = evaluate(toxic_sentences, res, refs,types=types, name=name,
                       save_path=os.path.join(save_path, name_to_add + 'eval_'+name+'.csv'),
                       per_sent_file_name=per_sent_file_name)
    
    print(df[['model','STA','ref_SIM','SIM','FL','J']].to_string())
    if do_types:
        df2 = pd.read_csv(per_sent_file_name)
        # 
    

        df3 = df2.mean(axis=0).to_frame().T
        # print(len(df3))

        df3['type'] = [-1]
        for T in [1,2,3,4,5]:
            # print('Type:',T)
            # print(len(df2.loc[df2['type']==T][['STA','ref_SIM','SIM','FL','J']]))
            # print(df2.loc[df2['type']==T].mean(axis=0))
            df3 = df3.append(df2[df2['type']==T].mean(axis=0).to_frame().T)
        df3 = (df3 * 10000).apply(np.floor)/10000
        print(df3[['STA','ref_SIM','SIM','FL','J','type']].to_string())
        # print(len(df3))
        per_type_name=os.path.join(save_path, name_to_add + 'per_type_'+name+'.csv')
        df3.to_csv(per_type_name)








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



