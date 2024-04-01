
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
    
        self.n_beams=1



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
        

        text = texts[0]

        return text




