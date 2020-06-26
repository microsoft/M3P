
# M3P

This repo provides the code of [M3P](https://arxiv.org/pdf/2006.02635.pdf), a Multitask Multilingual Multimodal Pre-trained model that combines multilingual-monomodal pre-training and monolingual-multimodal pre-training into a unified framework. The model learns universal representations that can map objects that occurred in different modalities or expressed in different languages to vectors in a common semantic space. To verify the generalization capability of M3P, the pre-trained model can be applied for different types of downstream tasks: [multilingual image-text retrieval](#multilingual-image-text-retrieval), [multilingual image captioning](#multilingual-image-captioning), [multimodal machine translation](#multimodal-machine-translation), multilingual natural language inference and multilingual text generation.

![img](M3P/figs/MMMP.png)

# Install and Dependency

Python 3

NumPy

PyTorch (version 1.2+)

fastBPE (for BPE codes)

Apex (for fp16 training)

SentencePiece provides Python wrapper that supports both SentencePiece training and segmentation. You can install Python binary package of SentencePiece with.

% pip install sentencepiece

# Data Ready

##Multi30K

##MSCOCO

COCO -zh http://lixirong.net/data/coco-cn/coco-cn-version1805v1.1.tar.gz

COCO -ja https://github.com/STAIR-Lab-CIT/STAIR-captions


# Pre-trained Models

| Task | Pre-trained Model |
|-----------|:-----------------:|
| Understanding   | [MODEL](https://unicoderrelease.blob.core.windows.net/m3p/m3p_under_weights.tar.gz)    |
| Generiation   | [MODEL](https://unicoderrelease.blob.core.windows.net/m3p/m3p_gen_weights.tar.gz)    |

# Downstream tasks

## Multilingual image-text retrieval

The task of multilingual image-text retrieval is to find the most relevant images given input texts in different languages, or vice versa. We evaluate M3P on Multi30K, MSCOCO and MILD.

On MILD benchmark we fine-tune M3P with two settings.

### Fine-tune on Q-I pairs without using image contexts (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --lgs $ALL_LGS \
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
### Fine-tune based on Q-I-C triples, where each image and its context always appear together as input (taking fine-tune on English as an example):

```
python -m torch.distributed.launch --nproc_per_node=$NGPU ./train_x.py --data_path $DATA_PATH \
    --reload_model $RELOAD \
    --dump_path $MODELS \
    --exp_name $EXP_NAME \
    --batch_size 24 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00005 \
    --lgs $ALL_LGS \
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
    --lgs $ALL_LGS \
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
    --lgs $ALL_LGS \
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
    --lgs $ALL_LGS \
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
