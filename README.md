
# M3P

This repo provides the code of [M3P](https://arxiv.org/pdf/2006.02635.pdf), a Multitask Multilingual Multimodal Pre-trained model that combines multilingual-monomodal pre-training and monolingual-multimodal pre-training into a unified framework. The model learns universal representations that can map objects that occurred in different modalities or expressed in different languages to vectors in a common semantic space. To verify the generalization capability of M3P, the pre-trained model can be applied for different types of downstream tasks: [multilingual image-text retrieval](#multilingual-image-text-retrieval), [multilingual image captioning](#multilingual-image-captioning), [multimodal machine translation](#multimodal-machine-translation), multilingual natural language inference and multilingual text generation.

![img](M3P/figs/MMMP.png)

# Install and Dependency

- Python 3
- NumPy
- PyTorch (version 1.2+)
- fastBPE (for BPE codes)
- Apex (for fp16 training)
- SentencePiece
- sacrebleu (for generation evaluation)

# Data Ready

Including datasets:
- [x] Multi30K [28, 29]
- [x] MSCOCO [16, 30, 31, 34] 
- [] MILD (*It will be released later.)


Multi30K extended Flickr30K [32] to German (de), French (fr) and Czech
(cs). It contains 31,783 images and provides 5 captions per image in English and German and 1 caption per
image in French and Czech. We use the train, dev, test splits as defined in [32]. MSCOCO contains 123,287
images and provides 5 captions per image in English, but fewer in Chinese and Japanese. STAIR Captions[33]
extended MSCOCO with 820K Japanese captions for COCO images. [31] extended MSCOCO with Chinese captions for 20K images.

Head to reference to download the fine-tuning datasets.
Reference:
[28] Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. Multi30k: Multilingual english-german
image descriptions. arXiv preprint arXiv:1605.00459, 2016.

[29] Desmond Elliott, Stella Frank, Loïc Barrault, Fethi Bougares, and Lucia Specia. Findings of the second
shared task on multimodal machine translation and multilingual image description.

[30] Takashi Miyazaki and Nobuyuki Shimizu. Cross-lingual image caption generation. In ACL, 2016.

[31] Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Coco-cn
for cross-lingual image tagging, captioning and retrieval. In IEEE Transactions on Multimedia, 2019.

[32] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual
denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the
Association for Computational Linguistics, 2:67–78, 2014.

[33] Yuya Yoshikawa, Yutaro Shigeto, and Akikazu Takeuchi. Stair captions: Constructing a large-scale
japanese image caption dataset. arXiv preprint arXiv:1705.00823, 2017.

[34] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3128–3137, 2015.


## Feature Extraction

We use MMF to extract detection features from the image. MMF is a modular framework for vision and language multimodal research. Built on top of PyTorch:
[Feature Extraction](https://github.com/facebookresearch/mmf/tree/6d89e1dede448682d549fb81d073536a31f88548)

*Note: ['bbox', 'captions', 'objects', 'features', 'image_id', 'num_boxes', 'wh'] This feature list is taken as the attributes of h5 file, which is extracted by the above script. The image_id equal to image_file in our project.

## Meta-data

The meta-data is a pickle file about mapping dictionary for image_id and caption. This will generate xxx.pkl file:

'COCO_train2014_000000010073.jpg': ['A couple of slices of pizza sitting on top of a white plate.',
  'The pizza is on the dish and ready to be eaten.',
  'Black olives and cheese pizza slices, with a fork, and sauce in small bowl, all on a plate.',
  'a white plate a fork and a pizza with black olives',
  'A plate of pizza with a fork and a bowl.'],
 'COCO_train2014_000000349905.jpg': ['a couple of kids and a woman with red hair',
  'A woman is holding a boy and a girl.',
  'A smiling woman with two small children in front of a home.',
  'A women who is holding two children on her lap.',
  'This mother is happy that her son and daughter like bananas.'],
  ...


# Pre-trained Models

| Task | Pre-trained Model |
|-----------|:-----------------:|
| Understanding   | [MODEL](https://unicoderrelease.blob.core.windows.net/m3p/m3p_under_weights.tar.gz)    |
| Generiation   | [MODEL](https://unicoderrelease.blob.core.windows.net/m3p/m3p_gen_weights.tar.gz)    |

Same with XLM-R, XLM-R handles the following 100 languages: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

# Downstream tasks

In this section, we will introduce how to fine-tune the pre-trained models on different downstream tasks.
Below notations apply to all commands:

```
$NGPU: number of GPUs used for fine-tuning
$DATA_PATH: path to the image caption files
$RELOAD: path to the pre-trained model
$EXP_NAME: name your experiment
$MODELS: path to store models
$VOCAB_PATH: path to the vocab file
$FEA_PATH: path to the image features
$MILD_PATH: subdirectory of the image features for MILD dataset
$EVAL_PATH: path to save evaluation results
```

## Multilingual image-text retrieval

The task of multilingual image-text retrieval is to find the most relevant images given input texts in different languages, or vice versa. We evaluate M3P on Multi30K, MSCOCO and MILD.

### Fine-tune MSCOCO

This is to fine-tune pre-trained understanding model on MSCOCO (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_rel_steps 'coco' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_understanding True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs 'en' \
    --eval_only False \
```

### Fine-tune Flickr Multi30K

This is to fine-tune pre-trained understanding model on Flickr Multi30K (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_rel_steps 'flicker' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_understanding True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs 'en' \
    --eval_only False \
```

On MILD benchmark we fine-tune M3P with below two settings.

### Fine-tune MILD based on Q-I pairs

This is to fine-tune pre-trained understanding model without using image contexts (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_rel_steps 'mild-img' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_understanding True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs 'en' \
    --eval_only False \
    --is_mild True \
    --qp_type 'q' \
    --seq_per_img 1 \
```
### Fine-tune MILD based on Q-I-C triples

This is to fine-tune pre-trained understanding model where each image and its context always appear together as input (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_rel_steps 'mild-img' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_understanding True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs 'en' \
    --eval_only False \
    --is_mild True \
    --qp_type 'qp' \
    --seq_per_img 1 \
```            

## Multilingual image captioning

The task of multilingual image captioning is to generate captions in specific languages given input images. We evaluate M3P on Multi30K and MSCOCO.

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_modal_steps 'flicker-img' \
    --epoch_size 100000 \
    --max_epoch 25 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_generation True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs $LG \
    --eval_only False \
    --beam_size 10 \
```      

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 32 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_modal_steps 'coco-img' \
    --epoch_size 100000 \
    --max_epoch 25 \
    --max_len 128 \
    --accumulate_gradients 4 \
    --input_fea_dir $FEA_PATH \
    --is_generation True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs $LG \
    --eval_only False \
    --beam_size 10 \
```      

For generation evaluation, you can refer this https://github.com/salaniz/pycocoevalcap or 
follow scareblue command line :
```  
python -m sacrebleu --force -lc -l lg-lg ref.txt < pred.txt
```  

## Multimodal machine translation

The task of multimodal machine translation is to generate sentences in target languages given source sentences together with related images as complementary information. We evaluate M3P on Multi30K. 

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --mild_path $MILD_PATH \
    --cross_modal_steps 'flicker-img' \
    --epoch_size 100000 \
    --max_epoch 25 \
    --max_len 128 \
    --accumulate_gradients 8 \
    --input_fea_dir $FEA_PATH \
    --is_generation True \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --ft_lgs 'en-de' \
    --eval_only False \
    --beam_size 10 \
    --is_mt True \
```      

# How to cite

If you find M3P useful in your work, you can cite the paper as below:

```
@article{huang2020m3p,
  title={M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training},
  author={Haoyang Huang and Lin Su and Di Qi and Nan Duan and Edward Cui and Taroon Bharti and Lei Zhang and Lijuan Wang and Jianfeng Gao and Bei Liu and Jianlong Fu and Dongdong Zhang and Xin Liu and Ming Zhou},
  journal={arXiv},
  year={2020},
  volume={abs/2006.02635}
}
```

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
