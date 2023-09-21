
from transformers import RobertaTokenizer, RobertaForSequenceClassification, LogitsProcessor, AutoTokenizer

import torch

import numpy as np
import torch.nn.functional as F

from seq2seq.multiobjective.scorer import Classifier, Scorer
from botorch.utils.multi_objective import pareto



class ParetoBeamSearch():

    def __init__(self, model_name, tokenizer_name, scorer: Scorer):

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.scorer = scorer


    

class ParetoLogitsProcessor(LogitsProcessor):

    def __init__(self, scorer: Scorer, model_tokenizer_name: str='facebook/bart-base', topk=20,alpha=0.4, sharp=False):
        self.scorer = scorer
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
        self.alpha = alpha
        self.topk = topk
        self.sharp = sharp

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # print('input_ids', input_ids.shape)
        # print('scores', scores.shape)
        # decoded_sequences = self.model_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print('decoded_sequences', decoded_sequences)
        
        next_tokens = torch.argsort(scores, dim=1, descending=True)

        decoded_next_sequences = []
        next_scores = []
        batch_size = scores.shape[0]
        # print('next_tokens[:, :self.topk]',next_tokens[:, :self.topk])
        for i in range(self.topk):
            # print(input_ids.shape,next_tokens[:, i].reshape(-1,1).shape,next_tokens.shape)
            next_sequence = torch.cat([input_ids, next_tokens[:, i].reshape(-1,1)], dim=-1)
            decoded_next_sequence = self.model_tokenizer.batch_decode(next_sequence, skip_special_tokens=True)
            # decoded_next_sequences.append(decoded_next_sequence)
            # print(decoded_next_sequence)

            # TODO: I do not have access to inputs here, need to think about it
            next_scores.append(self.scorer.get_objectives(input=None, preds=decoded_next_sequence))

        
        
        # next_scores = np.array(next_scores).reshape(batch_size,self.topk,3)
        next_scores = np.array(next_scores).reshape(batch_size,self.topk)

        next_scores = torch.from_numpy(next_scores).to('cuda')

        # print('next_scores.shape',next_scores.shape)
        
        multi_objectives = []
        new_next_scores = torch.zeros((batch_size,self.topk,2)).to('cuda')
        for i in range(batch_size):
            # print('scores.shape',scores[i,next_tokens[i,:self.topk]].shape)
            # print('next_scores[i,:].shape',next_scores[i,:].shape)
            new_next_scores[i,:,:] = torch.cat([next_scores[i,:], scores[i,next_tokens[i,:self.topk]]], dim=0).reshape(1, self.topk, 2)

        # print('next_scores',next_scores)

        # print('new_next_scores.shape',new_next_scores.shape)
        
        
        flag = pareto.is_non_dominated(new_next_scores,deduplicate=True)
        # print('a',flag)
        flag = 1.0 - flag.type(torch.float)
        print('b',flag[0,:])
        print(next_tokens[0,:self.topk])
        print(self.model_tokenizer.batch_decode(next_tokens[0,:self.topk], skip_special_tokens=True))
        print(scores[0,next_tokens[0,:self.topk]])
        # sdjkfjd()
        # TODO: make it batch instead of loop
        # print(flag[i,:].shape, scores[i,next_tokens[i,:self.topk]].shape, scores[i,next_tokens[i,:self.topk]].shape)
        for i in range(batch_size):
            # print(scores[i,next_tokens[i,:self.topk]].softmax(dim=-1))
            scores[i,next_tokens[i,:self.topk]] =  scores[i,next_tokens[i,:self.topk]] +  (-float("inf")) *  flag[i,:]
            scores[i,next_tokens[i,self.topk:]] = -float("inf")

        return scores
