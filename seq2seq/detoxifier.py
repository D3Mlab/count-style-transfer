
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartForConditionalGeneration, BartTokenizer, RobertaTokenizer, RobertaForSequenceClassification, LogitsProcessorList, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, T5ForConditionalGeneration
import random

import numpy as np
import torch.nn.functional as F
import torch
import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)

from seq2seq.scorer import Scorer, Classifier


# from seq2seq.multiobjective.multi_decoder import MutliDecoderModel


class Detoxifier():

    def __init__(self,model_name="facebook/bart-base",tokenizer_name="facebook/bart-base",prefix='',**kwargs):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = self.model.to('cuda')
        self.prefix=prefix
    def get_output(self,input_text):
        pass







class Seq2SeqDetoxifier(Detoxifier):

    def __init__(self,model_name="facebook/bart-base",tokenizer_name="facebook/bart-base",prefix='',**kwargs):
        
        super().__init__(model_name, tokenizer_name,prefix)
        # self.roll_out = RollOut()
        self.do_mbr = kwargs['do_mbr']

        self.n_beams=1
        if self.do_mbr:
            self.selector = Selector()
            self.n_beams=5


    def get_output(self,input_text):
        # TODO: make it batch
        # self.tokenizer.encode does not work on a list of texts
        # input_text = 'rewrite in an non-offensive form without using offensive words: '+input_text
        encoding = self.tokenizer.batch_encode_plus([input_text], return_tensors="pt").to('cuda')

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']


        with torch.no_grad():
            # beam_output =  self.model.generate(input_ids,attention_mask=attention_mask,roll_out_scorer=self.roll_out, max_length=50, 
            # do_sample=False,num_beams=10,num_return_sequences=10,
            # output_scores=True,return_dict_in_generate=True,branching_factor=25,rollout_length=5)
            
            beam_output =  self.model.generate(input_ids,attention_mask=attention_mask, max_length=50, 
            do_sample=False, num_beams=self.n_beams, num_return_sequences=self.n_beams, 
            # num_beams=1,num_return_sequences=1,
            output_scores=True,return_dict_in_generate=True)

        texts = self.tokenizer.batch_decode(beam_output['sequences'], skip_special_tokens=True)
        
        if self.do_mbr:
            text = self.selector.select(input_text,texts,beam_output['sequences_scores'])
        else:
            texts = [t for t in texts if len(t)>2]
            if len(texts)>0:
                text = texts[0]
            else:
                text = ''
        
        # text = texts[0]




        # print('input_text-->',input_text)
        # print(input_ids.shape)
        # print(beam_output['sequences'].shape)
        # print('final texts -->',texts)
        # sdjsdl()
        return text




class Selector():

    def __init__(self ,mode='argmax',criterion='j'):
        self.scorer = Scorer(sharp=True)
        # self.softmax_output = F.softmax(x, dim=1)
        self.mode = mode
        self.criterion = criterion

    
    def select(self, input_text, texts, liklihood_scores):
        scores = self.scorer.get_scores(input_text,texts)  
        if self.criterion!='all':
        
            scores = scores[self.criterion]
            if self.mode=='sample':
                return self.sample(texts,liklihood_scores,scores)
            elif self.mode=='argmax':
                return self.argmax(texts,liklihood_scores,scores)
            elif self.mode=='first_true':
                return self.first_true(texts,liklihood_scores,scores)
        else:
            res = {}
            for key in scores.keys():
                res[key] = self.argmax(texts,liklihood_scores,scores[key])
            return res



    def sample(self,texts,liklihood_scores,scores):
        
        # print(scores)
        probs = scores / np.sum(scores)
        # print(probs)
       
        return random.choices(texts, probs)[0] 

    def argmax(self,texts,liklihood_scores,scores):

        idx = np.argmax(scores)
        return texts[idx]


    def first_true(self,texts,liklihood_scores,scores):
        

        combined = list(zip(texts, liklihood_scores))

        # sort the list of tuples by the scores
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

        # separate the sorted tuples back into separate lists
        texts, liklihood_scores = zip(*sorted_combined)
        idx = 0
        for i in range(len(scores)):
            if scores[i]>=0.5:
                idx = i
                break
        
        return texts[idx]


