from transformers import  RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from scipy.stats import gmean
import random
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm

class Scorer():


    def __init__(self,toxicity_clf='SkolkovoInstitute/roberta_toxicity_classifier',FL_clf='cointegrated/roberta-large-cola-krishna2020',sim_model='sentence-transformers/LaBSE',device='cuda',sharp=False):
    
    

        self.toxicity_clf = Classifier(toxicity_clf,label=0,device=device,sharp=sharp)
        self.FL_clf = Classifier(FL_clf,label=0,device=device,sharp=sharp)
        self.sim_model = SentenceTransformer(sim_model)


    def get_sim_score(self, input,preds):
        sim_scores = []
    
        bs = len(preds)
        inputs_embeddings = self.sim_model.encode(input).reshape(bs,768)
        # print(inputs_embeddings.shape)
        preds_embeddings = self.sim_model.encode(preds).reshape(bs,768)
        for i in range(len(preds)):
            sim_scores.append(np.dot(inputs_embeddings[i,:],preds_embeddings[i,:]))
        return np.array(sim_scores)

    def get_sim_score2(self, input,preds):
        sim_scores = []
    
        bs = len(preds)
        inputs_embeddings = self.sim_model.encode(input).reshape(1,768)
        # print(inputs_embeddings.shape)
        preds_embeddings = self.sim_model.encode(preds).reshape(bs,768)
        for i in range(len(preds)):
            sim_scores.append(np.dot(inputs_embeddings[0,:],preds_embeddings[i,:]))
        return np.array(sim_scores)


    def get_scores(self, input,preds):
        sim_scores = self.get_sim_score2(input,preds)
        nontox_scores = self.toxicity_clf.get_scores(preds)
        fl_scores = self.FL_clf.get_scores(preds)

        gm_scores = [gmean([fl_scores[i],nontox_scores[i],sim_scores[i]]) for i in range(len(fl_scores))]
        j_scores = [fl_scores[i]*nontox_scores[i]*sim_scores[i] for i in range(len(fl_scores))]

        return {'gm':gm_scores,'fl':fl_scores,'sim':sim_scores,'nontox':nontox_scores,'j':j_scores}


    def get_objectives(self, input,preds):
        # sim_scores = self.get_sim_score(input,preds)
        nontox_scores = self.toxicity_clf.get_scores(preds)
        fl_scores = self.FL_clf.get_scores(preds)
        
        # objectives = np.concatenate([nontox_scores,sim_scores,fl_scores],axis=1).reshape(-1,3)
        # print(fl_scores,nontox_scores)
        # objectives = np.concatenate([nontox_scores,fl_scores],axis=0).reshape(-1,2)
        objectives = np.array(nontox_scores).reshape(-1,1)
        # print('objectives.shape',objectives.shape)
        # sdsdsd()
        return objectives


    def get_FL_scores(self, input,preds):
        fl_scores = self.FL_clf.get_scores(preds)
        return fl_scores

    def get_nontoxicity_scores(self, input,preds):
        nontoxicity = self.toxicity_clf.get_scores(preds)
        return nontoxicity

    def get_score(self, input,preds, score_name):
        if score_name=='fl':
            return self.get_FL_scores( input,preds)
        elif score_name=='nontox':
            return self.get_nontoxicity_scores( input,preds)
        elif score_name =='sim':
            return self.get_sim_score(input,preds)
        else:
            assert False,"Score name undefined!"


        

class Classifier():
    def __init__(self, clf_name: str='SkolkovoInstitute/roberta_toxicity_classifier',label=0,sharp=False,device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(clf_name)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(clf_name).to(self.device)
        self.label = label
        self.sharp = sharp
        

    def get_scores(self,preds,batch_size = 5):
        
        results = []
        for i in range(0, len(preds), batch_size):
            batch = self.tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
            with torch.no_grad():
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                # result = self.model(**batch)['logits'][:,1].float().data.tolist()
                probs = F.softmax(self.model(**batch)['logits'], dim=1).float().cpu().numpy()
                
            # print(probs.shape)
            if self.sharp:
                result = np.heaviside(probs[:,self.label]-0.5,0)
                
            else:
                result = probs[:,self.label]
                
                

            results.extend(result)
        return results 
