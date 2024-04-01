# COUNT: COntrastive UNlikelihood Text Style Transfer for Text Detoxification 

[Paper PDF](https://aclanthology.org/2023.findings-emnlp.579.pdf)

## Abstract

Offensive and toxic text on social media platforms can lead to polarization and divisiveness within online communities and hinders constructive dialogue. Text detoxification is a crucial task in natural language processing to ensure the generation of non-toxic and safe text. Text detoxification is a special case of the Text Style Transfer (TST) problem, where an input text is rephrased to an output text that preserves its content while modifying the style (in this case to a more neutral, non-toxic style). State-of-the-art methods for detoxification use supervised training of encoder-decoder models to produce gold-standard outputs with a standard likelihood-based objective. However, it can be hard for these models to deviate from their pretrained auto-encoder identity mapping. While previous methods have used unlikelihood-based losses to penalize input-to-output copying of toxic content, these methods also unfortunately penalize non-toxic content in the input that would be fine to preserve in the output. To address these issues, we introduce a novel contrastive unlikelihood objective (COUNT) that directly contrasts the gold standard rephrasing with the identity input-to-output mapping to effectively isolate and focus learning on non-toxic style transfer. We benchmark COUNT on two parallel datasets, ParaDetox and APPDIA, showing that it achieves significant improvements in jointly combined fluency, content preservation, and detoxification (i.e., the highest “J” score).




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


`python eval_seq2seq_model.py --model_name 'checkpoint_path' --save_path 'result_save_path' --dataset 'paradetox' --fold 'test' --name 'test_eval'  --make_preds --evaluate 
`


You can use 'test' or 'val' for --fold'

In all commands you can use whether 'paradetox' or  'appdia' for --dataset_name argument


## Saved checkpoints

You can use the trained models from [here](https://drive.google.com/drive/folders/1yI6tu2IKLxWBGjTHy6q8YAdShwTr2Nfo?usp=sharing).



## Citation
Cite this work using the Bibtex below:
```
@inproceedings{pour-etal-2023-count,
    title = "{COUNT}: {CO}ntrastive {UN}likelihood Text Style Transfer for Text Detoxification",
    author = "Pour, Mohammad Mahdi Abdollah  and
      Farinneya, Parsa  and
      Bharadwaj, Manasa  and
      Verma, Nikhil  and
      Pesaranghader, Ali  and
      Sanner, Scott",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.579",
    pages = "8658--8666",
    abstract = "Offensive and toxic text on social media platforms can lead to polarization and divisiveness within online communities and hinders constructive dialogue. Text detoxification is a crucial task in natural language processing to ensure the generation of non-toxic and safe text. Text detoxification is a special case of the Text Style Transfer (TST) problem, where an input text is rephrased to an output text that preserves its content while modifying the style (in this case to a more neutral, non-toxic style). State-of-the-art methods for detoxification use supervised training of encoder-decoder models to produce gold-standard outputs with a standard likelihood-based objective. However, it can be hard for these models to deviate from their pretrained auto-encoder identity mapping. While previous methods have used unlikelihood-based losses to penalize input-to-output copying of toxic content, these methods also unfortunately penalize non-toxic content in the input that would be fine to preserve in the output. To address these issues, we introduce a novel contrastive unlikelihood objective (COUNT) that directly contrasts the gold standard rephrasing with the identity input-to-output mapping to effectively isolate and focus learning on non-toxic style transfer. We benchmark COUNT on two parallel datasets, ParaDetox and APPDIA, showing that it achieves significant improvements in jointly combined fluency, content preservation, and detoxification (i.e., the highest {``}J{''} score).",
}

```
