
# M3P

This repo provides the code of [M3P](https://arxiv.org/pdf/2006.02635.pdf), a Multitask Multilingual Multimodal Pre-trained model that combines multilingual-monomodal pre-training and monolingual-multimodal pre-training into a unified framework. The model learns universal representations that can map objects that occurred in different modalities or expressed in different languages to vectors in a common semantic space. To verify the generalization capability of M3P, the pre-trained model can be applied for different types of downstream tasks: [multilingual image-text retrieval](#multilingual-image-text-retrieval), [multilingual image captioning](#multilingual-image-captioning), [multimodal machine translation](#multimodal-machine-translation), multilingual natural language inference and multilingual text generation.

![img](M3P/figs/MMMP.png)

# Install and Dependency

Python 3

NumPy

PyTorch (version 1.2+)

fastBPE (for BPE codes)

Apex (for fp16 training)

SentencePiece

# Data Ready

## Multi30K

In order to fetch mutli30k data correctly, you can clone the repository with --recursive flag:

$ git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset

For multilingual image-text retrieval and multilingual-image-captioning we use raw texts in en and de from task2 and raw texts in fr and cs from task1.
For multimodal machine translation we use raw texts from task1 to build en-de and en-fr bilingual data.

## MSCOCO

coco -en https://cocodataset.org/#download

COCO -zh http://lixirong.net/data/coco-cn/coco-cn-version1805v1.1.tar.gz

COCO -ja https://github.com/STAIR-Lab-CIT/STAIR-captions

## Feature Extraction

We use bottom-up-attention to extract detection features from the image. We use detectron implemented from facebook:
https://github.com/facebookresearch/mmf/blob/6d89e1dede448682d549fb81d073536a31f88548/tools/scripts/features/extract_features_vmb.py

['bbox', 'captions', 'objects', 'features', 'image_id', 'num_boxes', 'wh'] This feature list is taken as the attribute of h5, which is extracted by the above script.

## Meta-data collation

For meta-data, we assign each image_id with its raw caption list. We can build a pickle file, the dictionary for image_id and caption like this:

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
