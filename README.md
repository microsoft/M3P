
# M3P

This repo provides the code of [M3P](https://arxiv.org/pdf/2006.02635.pdf), a Multitask Multilingual Multimodal Pre-trained model that combines multilingual pre-training and multimodal pre-training into a unified framework via multitask pre-training. Our goal is to learn universal representations that can map objects occurred in different modalities or texts expressed in different languages into a common semantic space. In addition, to explicitly encourage fine-grained alignment between images and non-English languages, we also propose Multimodal Code-switched Training (MCT) to combine monolingual pre-training and multimodal pre-training via a code-switch strategy. Experiments are performed on the [multilingual image-text retrieval](#multilingual-image-text-retrieval) across two benchmark datasets, including MSCOCO and Multi30K. M3P can achieve comparable results for English and new state-of-the-art results for non-English languages.

![img](M3P/figs/Overview.png)

# Install and Dependency

- Python 3
- NumPy
- PyTorch (version 1.2+)
- fastBPE (for BPE codes)
- Apex (for fp16 training)
- SentencePiece

# Data Ready

Including datasets:
- [x] Multi30K 
- [x] MSCOCO 

If you use these resources in your research, please consider citing the following papers:

## Multi30K
English and German data:

Reference:
```
@InProceedings{W16-3210,
  author = 	"Elliott, Desmond
		and Frank, Stella
		and Sima'an, Khalil
		and Specia, Lucia",
  title = 	"Multi30K: Multilingual English-German Image Descriptions",
  booktitle = 	"Proceedings of the 5th Workshop on Vision and Language",
  year = 	"2016",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"70--74",
  location = 	"Berlin, Germany",
  doi = 	"10.18653/v1/W16-3210",
  url = 	"http://www.aclweb.org/anthology/W16-3210"
}
```
French data, Ambiguous COCO evaluation data, and Test 2017 data:

```
@InProceedings{elliott-EtAl:2017:WMT,
  author    = {Elliott, Desmond  and  Frank, Stella  and  Barrault, Lo\"{i}c  and  Bougares, Fethi  and  Specia, Lucia},
  title     = {Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description},
  booktitle = {Proceedings of the Second Conference on Machine Translation, Volume 2: Shared Task Papers},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {215--233},
  url       = {http://www.aclweb.org/anthology/W17-4718}
}
```
Czech data:
```
@inproceedings{barrault2018findings,
  title={Findings of the Third Shared Task on Multimodal Machine Translation},
  author={Barrault, Lo{\"\i}c and Bougares, Fethi and Specia, Lucia and Lala, Chiraag and Elliott, Desmond and Frank, Stella},
  booktitle={Proceedings of the Third Conference on Machine Translation: Shared Task Papers},
  pages={304--323},
  year={2018}
}
```
## MSCOCO
English data:
```
@article{chen2015microsoft,
  title={Microsoft coco captions: Data collection and evaluation server},
  author={Chen, Xinlei and Fang, Hao and Lin, Tsung-Yi and Vedantam, Ramakrishna and Gupta, Saurabh and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  journal={arXiv preprint arXiv:1504.00325},
  year={2015}
}
```

Japanese data:
```
@article{yoshikawa2017stair,
  title={Stair captions: Constructing a large-scale japanese image caption dataset},
  author={Yoshikawa, Yuya and Shigeto, Yutaro and Takeuchi, Akikazu},
  journal={arXiv preprint arXiv:1705.00823},
  year={2017}
}
```
Chinese data:
```
@article{li2019coco,
  title={COCO-CN for Cross-Lingual Image Tagging, Captioning, and Retrieval},
  author={Li, Xirong and Xu, Chaoxi and Wang, Xiaoxu and Lan, Weiyu and Jia, Zhengxiong and Yang, Gang and Xu, Jieping},
  journal={IEEE Transactions on Multimedia},
  volume={21},
  number={9},
  pages={2347--2360},
  year={2019},
  publisher={IEEE}
}
```
## Feature Extraction

We use MMF [Feature Extraction script](https://github.com/facebookresearch/mmf/blob/6d89e1dede448682d549fb81d073536a31f88548/tools/scripts/features/extract_features_vmb.py) to extract detection features from the image. MMF is a modular framework for vision and language multimodal research. Built on top of PyTorch:
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

You can download the pre-trained model from [this link](https://haohua.blob.core.windows.net/minheng/UVL_EXP/UVL_XVL/z3o7zmnowz/checkpoint.pth?sv=2020-04-08&st=2021-05-13T06%3A35%3A03Z&se=2099-12-31T06%3A35%3A00Z&sr=b&sp=r&sig=6bfXa46W3ih1U23J0sgdOAZPk37YFcnpmeCYMj6rnqY%3D).

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
    --emb_dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --n_dec_layers -1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation True \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --lgs $ALL_LGS \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --google_path 'google_captions/obj100' \
    --sbu_path 'google_captions/obj100' \
    --coco_path coco \
    --flicker_path flicker \
    --cross_rel_steps coco-img \
    --mlm_steps '' \
    --epoch_size 150000 \
    --max_epoch 150 \
    --bptt 128 \
    --max_len 64 \
    --fp16 True \
    --validation_metrics valid_I2T_acc,valid_T2I_acc \
    --max_region_num 100 \
    --accumulate_gradients 4 \
    --amp 1 \
    --refine_image False \
    --refine_encoder False \
    --input_fea_dir $FEA_PATH \
    --save_every_epoch 5 \
    --is_generation False \
    --is_understanding True \
    --is_pretrain False \
    --use_new_fea True \
    --t2i_flag True \
    --i2t_flag True \
    --eval_n 50 \
    --sent_ratio 0.8 \
    --word_ratio 0.8 \
    --seq_per_img 5 \
    --eval_images -1 \
    --sample_n 4 \
    --multi_cls_loss_weight 0 \
    --bin_cls_loss_weight 1 \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --google_valid_path $CC_VALID_PATH \
    --train_order_path $ORDER_PATH \
    --cross_lingual_path $CROSS_LINGUAL_PATH \
    --ft_lgs 'en' \
    --src_lgs 'en' \
    --ag_lgs ''\
    --eval_only False \
    --is_freelb False
```

### Fine-tune Flickr Multi30K

This is to fine-tune pre-trained understanding model on Flickr Multi30K (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --emb_dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --n_dec_layers -1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation True \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --lgs $ALL_LGS \
    --data_path $DATA_PATH \
    --vocab_path $VOCAB_PATH \
    --google_path 'google_captions/obj100' \
    --sbu_path 'google_captions/obj100' \
    --coco_path coco \
    --flicker_path flicker \
    --cross_rel_steps flicker-img \
    --mlm_steps '' \
    --epoch_size 150000 \
    --max_epoch 150 \
    --bptt 128 \
    --max_len 64 \
    --fp16 True \
    --validation_metrics valid_I2T_acc,valid_T2I_acc \
    --max_region_num 100 \
    --accumulate_gradients 4 \
    --amp 1 \
    --refine_image False \
    --refine_encoder False \
    --input_fea_dir $FEA_PATH \
    --save_every_epoch 5 \
    --is_generation False \
    --is_understanding True \
    --is_pretrain False \
    --use_new_fea True \
    --t2i_flag True \
    --i2t_flag True \
    --eval_n 50 \
    --sent_ratio 0.8 \
    --word_ratio 0.8 \
    --seq_per_img 5 \
    --eval_images -1 \
    --sample_n 4 \
    --multi_cls_loss_weight 0 \
    --bin_cls_loss_weight 1 \
    --num_workers 4 \
    --eval_path $EVAL_PATH \
    --google_valid_path $CC_VALID_PATH \
    --train_order_path $ORDER_PATH \
    --cross_lingual_path $CROSS_LINGUAL_PATH \
    --ft_lgs 'en' \
    --src_lgs 'en' \
    --ag_lgs ''\
    --eval_only False \
    --is_freelb False
```

# How to cite

If you find M3P useful in your work, you can cite the paper as below:

```
@article{huang2020m3p,
  title={M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training},
  author={Minheng Ni and Haoyang Huang and Lin Su and Edward Cui and Taroon Bharti and Lijuan Wang and Jianfeng Gao and Dongdong Zhang and Nan Duan},
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
