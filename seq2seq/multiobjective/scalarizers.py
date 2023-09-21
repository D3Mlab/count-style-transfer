

import torch
from botorch.utils.multi_objective import pareto
import time

import numpy as np
class Scalarizer():

    def __init__(self):
        pass


    def scalarize(self,next_token_scores,additional_scores):
        pass

    def fix_size(self,next_token_scores,additional_scores):
        num_vocabs = next_token_scores.shape[-1]
        for i in range(len(additional_scores)):
            if additional_scores[i].shape[-1]>num_vocabs:
                additional_scores[i] = additional_scores[i][...,:num_vocabs]
        return additional_scores


    def filter_dominateds(self, next_token_scores, additional_scores):
        additional_scores = self.fix_size(next_token_scores,additional_scores)

        all_probs = [next_token_scores] + additional_scores
        all_probs = torch.stack(all_probs,dim=2)


        k = 200

        indices_to_remove = next_token_scores < torch.topk(next_token_scores, k)[0][..., -1, None]

        # print(indices_to_remove.shape,all_probs.shape)

        all_probs[:,:,0] = all_probs[:,:,0].masked_fill(indices_to_remove, 0.0)
        all_probs[:,:,1] = all_probs[:,:,1].masked_fill(indices_to_remove, 0.0)
        all_probs[:,:,2] = all_probs[:,:,2].masked_fill(indices_to_remove, 0.0)


        for i, scores in enumerate(additional_scores):
            indices_to_remove = scores < torch.topk(scores, k)[0][..., -1, None]
            # print(indices_to_remove.shape,all_probs.shape)
            # print(torch.sum(indices_to_remove, dim=-1))
            all_probs[:,:,0] = all_probs[:,:,0].masked_fill(indices_to_remove, 0.0)
            all_probs[:,:,1] = all_probs[:,:,1].masked_fill(indices_to_remove, 0.0)
            all_probs[:,:,2] = all_probs[:,:,2].masked_fill(indices_to_remove, 0.0)
        


        merged_size = 20
        indexes  = np.zeros(all_probs.shape[0], merged_size)
        prob_list = np.zeros(all_probs.shape[0], merged_size,3)

        for i in range(all_probs.shape[0]):
            cursor = 0
            for j in range(all_probs.shape[1]):
                if all_probs[i,j,0] > 0:
                    indexes[i,cursor] = j
                    indexes[i,cursor,:] = all_probs[i,j,:]
                    cursor+=1
                    if cursor>merged_size:
                        break


        flag = pareto.is_non_dominated(prob_list, deduplicate=True)





        # TODO: we still need some scoring
        # print(all_probs.shape)
        t = time.time()
        flag = pareto.is_non_dominated(all_probs, deduplicate=True)
        print(time.time()-t)
        asdjskld()
        flag = 1.0 - flag.type(torch.float)
        # print(flag.shape,next_token_scores.shape)
        # jdkfjdfl()
        next_token_scores = next_token_scores  +  (-float("inf")) *  flag
        for i in range(len(additional_scores)):
            additional_scores[i] = additional_scores[i]  +  (-float("inf")) *  flag
        
        return next_token_scores, additional_scores



class LinearScalarizer(Scalarizer):

    def __init__(self,weights,n=50):
        super().__init__()
        self.weights = weights
        self.n = n
        # print(self.weights)


    def scalarize(self,next_token_scores,additional_scores):
        # TODO: first cut at n
        # bs = next_token_scores.shape[0]
        # next_tokens = torch.argsort(next_token_scores, dim=1, descending=True)
        # for i in range(bs):
        #     next_token_scores[i,next_tokens[i,self.n:]] = -float("inf")
        # print(next_token_scores.shape)

        additional_scores = self.fix_size(next_token_scores,additional_scores)
        num_vocabs = next_token_scores.shape[-1]

        final_next_token_scores = self.weights[0] * next_token_scores.clone().detach()
        for i in range(len(additional_scores)):

            # next_tokens = torch.argsort(additional_scores[i], dim=1, descending=True)
        
            # for j in range(bs):
            #     additional_scores[i][j,next_tokens[j,self.n:]] = -float("inf")

            # if additional_scores[i].shape[-1]>num_vocabs:
            #     additional_scores[i] = additional_scores[i][...,:num_vocabs]
            final_next_token_scores += self.weights[i+1]* additional_scores[i]
        return final_next_token_scores





class TorchFunctionScalarizer(Scalarizer):

    def __init__(self,function,n=500):
        super().__init__()
        self.function = function
        self.n = n
        # print(self.weights)


    def scalarize(self,next_token_scores,additional_scores):
        additional_scores = self.fix_size(next_token_scores,additional_scores)


        all_probs = [next_token_scores] + additional_scores
        all_probs = torch.stack(all_probs,dim=0)
        # print(all_probs.shape)
        final_next_token_scores = self.function(all_probs,dim=0).values
        # print(final_next_token_scores)
        # print(final_next_token_scores.shape)
        # dssldk()
        return final_next_token_scores



class HarmonicMeanScalarizer(Scalarizer):

    def __init__(self,n=500):
        super().__init__()
        # self.function = function
        self.n = n
        # print(self.weights)


    def scalarize(self,next_token_scores,additional_scores):

        additional_scores = self.fix_size(next_token_scores,additional_scores)

        num_vocabs = next_token_scores.shape[-1]
        all_probs = [next_token_scores] + additional_scores
        all_probs = torch.stack(all_probs,dim=0)
        # print('asas',all_probs.shape,len(additional_scores))
        # assd()
        final_next_token_scores = ((1/all_probs).mean(dim=0))**(-1)


        # print(next_token_scores.shape)
        # print(final_next_token_scores.shape)
        # sjksdj()

        return final_next_token_scores


class RoundRobinScalarizer(Scalarizer):

    def __init__(self,n=500):
        super().__init__()
        # self.function = function
        self.n = n
        self.iter = 0
        # print(self.weights)


    def scalarize(self,next_token_scores,additional_scores):
        additional_scores = self.fix_size(next_token_scores,additional_scores)

        if self.iter==0:
            return next_token_scores
        else:
            return_id = self.iter
            self.iter+=1
            N = len(additional_scores)+1
            self.iter = self.iter % N
            return additional_scores[return_id]



class RoundRobinScottScalarizer(Scalarizer):

    def __init__(self,n=500):
        super().__init__()
        # self.function = function
        self.n = n
        self.iter = 0
        # print(self.weights)


    def scalarize(self, next_token_scores, additional_scores):
        bs = next_token_scores.shape[0]
        additional_scores = self.fix_size(next_token_scores,additional_scores)
        

        next_tokens = torch.argsort(next_token_scores, dim=1, descending=True)
        additional_next_tokens = []
        for i in range(len(additional_scores)):
            additional_next_tokens.append(torch.argsort(additional_scores[i], dim=1, descending=True))
        
        # indices_to_remove = scores < torch.topk(scores, self.k)[0][..., -1, None]
        # new_next_token_scores = new_next_token_scores.masked_fill(indices_to_remove,0.0)
        new_next_token_scores = torch.zeros_like(next_token_scores)

        all_probs = [next_token_scores] + additional_scores
        all_probs = torch.stack(all_probs,dim=0)

        list_size = 10
        d = 2.0/(list_size*(list_size-1))
        # print('sdsds',len(additional_scores))
        for i in range(bs):
            round_robin = 0
            selected_list = []
            selected_list_probs = []

            decoder_indexes = [0,0,0]
            while len(selected_list)!=list_size:
                if round_robin == 0:
                    current_token_id = next_tokens[i,decoder_indexes[0]]
                else:
                    # print(round_robin,i,decoder_indexes[round_robin])
                    # print(additional_next_tokens)
                    current_token_id = additional_next_tokens[round_robin-1][i,decoder_indexes[round_robin]]
                
                decoder_indexes[round_robin]+=1
                # print('current_token_id',current_token_id)
                # needs checking
                current_objectives = all_probs[:, i, current_token_id]
                # print('current_objectives',current_objectives)

                if current_token_id not in selected_list:
                    selected_list.append(current_token_id)
                    selected_list_probs.append(torch.mean(current_objectives))
                    # idx = list_size - len(selected_list_probs)
                    # selected_list_probs.append(idx*d)
                    round_robin += 1
                    round_robin = round_robin % (len(additional_scores)+1)
            # print(selected_list)
            # print(selected_list_probs)
            for j in range(len(selected_list)):
                new_next_token_scores[i,selected_list[j]] = selected_list_probs[j]
            # TODO: check top tokens from new_next_token_scores and selected_list
            # print(torch.argsort(new_next_token_scores, dim=1, descending=True))
            # print(selected_list)
        # shjsdh()
        return new_next_token_scores
                
                

class ConstrainedScalarizer(Scalarizer):

    def __init__(self,k=10):
        super().__init__()
        self.k = k



    def scalarize(self, next_token_scores, additional_scores):
        # additional_scores = self.fix_size(next_token_scores,additional_scores)
        

        # new_next_token_scores = next_token_scores.clone().detach()

        new_next_token_scores = additional_scores[0].clone().detach()


        bs = next_token_scores.shape[0]
        for scores in [next_token_scores,additional_scores[1]]:
        # for i in [0]:
            # next_tokens = torch.argsort(additional_scores[i], dim=1, descending=True)
            # print('-->',i,next_tokens)
            # print(next_tokens.shape)
            # print(new_next_token_scores.shape,)
            indices_to_remove = scores < torch.topk(scores, self.k)[0][..., -1, None]
            # print(indices_to_remove.shape)
            # print(torch.sum(indices_to_remove, dim=-1))
            new_next_token_scores = new_next_token_scores.masked_fill(indices_to_remove,0.0)
            # for j in range(bs):
                # print(new_next_token_scores[j,next_tokens[j,self.k:]].shape)
                # new_next_token_scores[j,next_tokens[j,self.k:]] = torch.ones_like(new_next_token_scores[j,next_tokens[j,self.k:]]) * (-float("inf"))
   
                # print(new_next_token_scores[j,next_tokens[j,self.k:]])

        # new_next_token_scores =  torch.nn.functional.softmax(new_next_token_scores, dim=-1)
        # print(torch.argsort(new_next_token_scores, dim=1, descending=True))
        # print(torch.sort(new_next_token_scores, dim=1, descending=True))
        return new_next_token_scores
            