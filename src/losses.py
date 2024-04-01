import torch


def generate_negatives(model,input_ids, num_to_generate=6):
    # print('input_ids.shape',input_ids.shape)
    beam_output = model.generate(input_ids,  max_length=50, do_sample=True,  num_beams=num_to_generate, num_return_sequences=num_to_generate, early_stopping=True)
    # print(beam_output)
    texts = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
    return texts, beam_output


def get_score(model,x,y,i,j):
    outputs = model(input_ids=x['input_ids'][i].reshape(1,-1), attention_mask=x['attention_mask'][i].reshape(1,-1), labels=y['input_ids'][j].reshape(1,-1))

    score = -1 *  outputs.loss

    return score


def compute_contrastive_loss(model,x,y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        # TODO: do batch generation
        texts, beam_output = generate_negatives(model, x['input_ids'][i].reshape(1,-1))
        # print(batch['en_toxic_comment'][i],texts)
        target_score = get_score(model,x,y,i,i)
        denum  = target_score
        y_generate = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        for j in range(len(texts)):
            
            s = get_score(model,x,y_generate,i,j)
            # print('s',s)
            denum= denum + s
        # print(target_score,denum)
        loss+= -1 * torch.log(target_score/denum)



    # print(texts)
    loss = loss/batch_size
    return loss




def unlikelihood_loss(model,x,y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    # texts, beam_output = generate_negatives(model, x['input_ids'],num_to_generate=1)
    # y_generate = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs2 = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=x['input_ids'])

    loss =  -1 * torch.log(1-torch.exp( -1 * outputs2.loss))
    return loss





def compute_contrastive_loss5(model,x,y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        # TODO: do batch generation
        # texts, beam_output = generate_negatives(model, x['input_ids'][i].reshape(1,-1), num_to_generate=1)
        target_score = get_score(model, x, y, i, i)
        denum  = target_score
        # y_generate = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        # for j in range(len(texts)):
        j = i # TODO: seems wrong, should be j=i
        s = get_score(model, x, x, i, j)

        # denum= denum + s



        loss+= -1*torch.log(torch.exp(target_score)/ (torch.exp(target_score)+torch.exp(s)))

    loss = loss/batch_size
    return loss






def compute_contrastive_loss8(model,x,y):
    loss = 0
    batch_size = x['input_ids'].shape[0]
    for i in range(batch_size):
        # TODO: do batch generation
        # texts, beam_output = generate_negatives(model, x['input_ids'][i].reshape(1,-1), num_to_generate=1)
        target_score = get_score(model, x, y, i, i)
        denum  = target_score
        # y_generate = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        s = 0
        for j in range(batch_size):
            s += torch.exp(get_score(model, x, x, i, j))

        # denum= denum + s



        loss+= -1*torch.log(torch.exp(target_score)/ (torch.exp(target_score)+s))

    loss = loss/batch_size
    return loss



def compute_langauge_modeling_loss(model,x,y):
        outputs = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=y['input_ids'])
        return  outputs.loss