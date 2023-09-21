


from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import  tokenizers
from datasets import load_dataset
import os
import argparse
import json
from torch import nn
import time
from torch.utils.data import random_split
from losses import *
# from mapping import Mapping 
from transformers import BartConfig

from dataset import get_APPDIA_train_and_val_loaders, get_paradetox_train_and_val_loaders



parser = argparse.ArgumentParser(description='Train LM')
parser.add_argument('--contrastive_loss', action='store_true')

parser.add_argument('--unlikelihood', action='store_true')

parser.add_argument('--add_negatives', action='store_true')
parser.add_argument('--mapping', action='store_true')
parser.add_argument('--model_name', type=str)
parser.add_argument('--lm_name',default="facebook/bart-base", type=str)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--min_delta', default=0.0001, type=float)
parser.add_argument('--patience', default=2, type=int)

parser.add_argument('--dont_save', action='store_true')

parser.add_argument('--save_folder',default="", type=str)


parser.add_argument('--dataset_name',default="paradetox", type=str)


args = parser.parse_args()
min_delta = args.min_delta
alpha = args.alpha
patience = args.patience
num_epochs = args.num_epochs
mapping = args.mapping
lr = args.lr
lm_name = args.lm_name
contrastive_loss = args.contrastive_loss

dataset_name = args.dataset_name

save_folder = args.save_folder

unlikelihood = args.unlikelihood


if dataset_name =='paradetox':
   
    train_dataloader, eval_dataloader = get_paradetox_train_and_val_loaders()
    print('Paradetox dataset!',len(train_dataloader),len(eval_dataloader))
elif dataset_name == 'appdia':
    
    train_dataloader, eval_dataloader = get_APPDIA_train_and_val_loaders()
    print('APPDIA dataset!',len(train_dataloader),len(eval_dataloader))
else:
    assert False, 'Wrong dataset name!'


# dataset = load_dataset("SkolkovoInstitute/paradetox", "en-US", split="train")

# N = len(dataset)

# train_size = int(0.8* N)
# test_size = N - train_size

# generator1 = torch.Generator().manual_seed(42)
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=generator1)

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(val_dataset, batch_size=8)



if mapping:
    # config_loaded = BartConfig.from_pretrained(lm_name)
    # model = Mapping(config_loaded)

    model = BartForConditionalGeneration.from_pretrained(lm_name)


    # model = Mapping(lm_name,768,[192],768) # add numbers to param
else:
    model = BartForConditionalGeneration.from_pretrained(lm_name)


tokenizer = BartTokenizer.from_pretrained(lm_name)



if mapping:
    optimizer = AdamW(model.model.encoder.layers[-1].parameters(), lr=lr)
else:
    optimizer = AdamW(model.parameters(), lr=lr)



num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device is',device)
model.to(device)


progress_bar = tqdm(range(num_training_steps))


model.train()

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def combined_contrastive_loss(model,x,y):
    return compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss5(model,x,y)




def combined_unlikelihood_loss(model,x,y):
    return compute_langauge_modeling_loss(model,x,y) + alpha * unlikelihood_loss(model,x,y)

def combined_contrastive_loss_with_negatives(model,x,y):
    return compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss8(model,x,y)


loss_function = None
if args.contrastive_loss:
    if not args.add_negatives:
        loss_function = combined_contrastive_loss
    else:
        loss_function = combined_contrastive_loss_with_negatives

elif args.unlikelihood:
    loss_function = combined_unlikelihood_loss
else:
    loss_function = compute_langauge_modeling_loss



# EXP_NAME = str(int(time.time()))
EXP_NAME = 'EXP1'

path_prefix = './saved_models/'


if len(save_folder)>0:
    path_prefix = os.path.join(path_prefix,save_folder)


path = os.path.join(path_prefix,EXP_NAME)

if not args.dont_save:
    if not os.path.exists(path):
        os.makedirs(path)


    with open(os.path.join(path,'exp_config.json'), "w") as file:
        my_dict = {'alpha':alpha,'num_epochs':num_epochs,'model_name':args.model_name,'patience':patience,'min_delta':min_delta,'mapping':mapping,
        'lm_name':lm_name,"add_negatives":args.add_negatives,'save_folder':save_folder,'dataset_name':dataset_name,'contrastive_loss':contrastive_loss,'unlikelihood':unlikelihood
       }
        json.dump(my_dict, file)



def save_mapping_checkpint(path,model,epoch):

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.mlp.state_dict(), os.path.join(path,'mlp.pth'))
    with open(os.path.join(path,'info.json'), "w") as file:
        my_dict = {'epoch':epoch}
        json.dump(my_dict, file)

def save_transformers_checkpint(path,model,epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_pretrained(path, from_pt=True) 
    with open(os.path.join(path,'info.json'), "w") as file:
        my_dict = {'epoch':epoch}
        json.dump(my_dict, file)


if mapping:
    # save_checkpint =save_mapping_checkpint
    save_checkpint = save_transformers_checkpint
else:
    save_checkpint = save_transformers_checkpint

def get_loss(model,batch,loss_function):
        x = tokenizer.batch_encode_plus(batch['en_toxic_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        y = tokenizer.batch_encode_plus(batch['en_neutral_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        loss = loss_function(model,x,y)
        return loss

best_val_loss = 99999999.9
for epoch in range(num_epochs):

    for batch in train_dataloader:
        # print(batch)
        # batch = {k: v.to(device) for k, v in batch.items()}
        # x = tokenizer.batch_encode_plus(batch['en_toxic_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        # x = torch.tensor(tokenizer.encode('input: '+batch['en_toxic_comment']+' output:')).unsqueeze(0).to(device)
        # y = tokenizer.batch_encode_plus(batch['en_neutral_comment'], padding=True, truncation=True, return_tensors='pt').to(device)
        # print(batch['en_toxic_comment'])
        # print(x['input_ids'].shape, y['input_ids'].shape, x['attention_mask'].shape)
        # print(x)
        # print(y)

        # print(target_score.shape)

        # loss = loss_function(model,x,y)

        # loss = compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss7(model,x,y)

        # loss = compute_langauge_modeling_loss(model,x,y) + alpha * compute_contrastive_loss5(model,x,y)

        loss = get_loss(model,batch,loss_function)

        # loss = compute_contrastive_loss6(model,x,y)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_description(f"Loss {loss}")
    cc = 0
    total_val_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            # Calculate validation loss
            l = get_loss(model,batch,loss_function)
            total_val_loss+=l
            cc+=1
    val_loss = total_val_loss/cc
    # Check if validation loss has improved
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        # TODO: save checkpoint
        save_checkpint(os.path.join(path,'best'),model,epoch)
        counter = 0
    else:
        counter += 1
    
    # Check if early stopping condition is met
    if counter >= patience:
        print("Early stopping! No improvement in validation loss for {} epochs.".format(patience))
        break
    