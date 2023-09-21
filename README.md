# count-style-transfer





##  Training

### Train with LM loss

`python seq2seq/train_seq2seq.py  --model_name model_name --save_folder 'save_folder' --dataset_name 'paradetox'
`

### Train with UT loss
`
python seq2seq/train_seq2seq.py  --model_name some_name  --unlikelihood --alpha alpha_value  --save_folder 'save_folder'  --dataset_name 'appdia'
`
### Train with COUNT loss

`
python seq2seq/train_seq2seq.py  --model_name model_name --contrastive_loss --alpha alpha_value  --save_folder 'save_folder' --dataset_name 'paradetox'
`

## Evalutaing 


`python eval_seq2seq_model.py --model_name 'checkpoint_path' --save_path 'result_save_path' --dataset 'paradetox' --fold 'test' --name 'test_eval' 
`
you can use 'test' or 'val' for --fold'

In all commands you can use whether 'paradetox' or  'appdia' for --dataset_name argument


## Saved checkpoints

### Paradetox
LM loss
UT loss
COUNT loss

### APPDIA
LM loss
UT loss
COUNT loss