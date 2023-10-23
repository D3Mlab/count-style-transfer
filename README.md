# count-style-transfer





##  Training

### Train with LM loss

`python src/train_seq2seq.py  --model_name model_name --save_folder 'save_folder' --dataset_name 'paradetox'
`

### Train with UT loss
`
python src/train_seq2seq.py  --model_name some_name  --unlikelihood --alpha alpha_value  --save_folder 'save_folder'  --dataset_name 'appdia'
`
### Train with COUNT loss

`
python src/train_seq2seq.py  --model_name model_name --contrastive_loss --alpha alpha_value  --save_folder 'save_folder' --dataset_name 'paradetox'
`

## Evalutaing 


`python eval_seq2seq_model.py --model_name 'checkpoint_path' --save_path 'result_save_path' --dataset 'paradetox' --fold 'test' --name 'test_eval' 
`


You can use 'test' or 'val' for --fold'

In all commands you can use whether 'paradetox' or  'appdia' for --dataset_name argument


## Saved checkpoints

You can use the trained models from [here](https://drive.google.com/drive/folders/1yI6tu2IKLxWBGjTHy6q8YAdShwTr2Nfo?usp=sharing).



## Citation
Cite this work using the Bibtex below:
```
@inproceedings{abdollah2023count,
  title={COUNT: COntrastive UNlikelihood Text Style Transfer for Text Detoxification},
  author={Abdollah Pour, Mohammad Mahdi and Farinneya, Parsa and Bharadwaj, Manasa and Verma, Nikhil and Pesaranghader, Ali and Sanner, Scott},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023}
}
```
