
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
    --batch_size 12 \
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
    --coco_path 'COCO/coco14/obj100' \
    --flicker_path flicker \
    --mild_path '/multimedia-nfs/lins/data/QPI' \
    --cross_modal_steps '' \
    --cross_mass_steps '' \
    --cross_ae_steps '' \
    --cross_gan_steps '' \
    --cross_rel_steps 'mild-img' \
    --cross_mlm_steps '' \
    --cross_mrm_steps '' \
    --cross_mrfr_steps '' \
    --mlm_steps '' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --bptt 128 \
    --max_len 128 \
    --fp16 True \
    --validation_metrics valid_I2T_acc,valid_T2I_acc \
    --max_region_num 100 \
    --accumulate_gradients 8 \
    --amp 1 \
    --refine_image False \
    --refine_encoder False \
    --input_fea_dir $FEA_PATH \
    --is_cross_modal True \
    --save_every_epoch 5 \
    --is_generation False \
    --is_understanding True \
    --is_pretrain False \
    --use_new_fea True \
    --t2i_flag True \
    --i2t_flag True \
    --eval_n 50 \
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
    --batch_size 12 \
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
    --coco_path 'COCO/coco14/obj100' \
    --flicker_path flicker \
    --mild_path '/multimedia-nfs/lins/data/QPI' \
    --cross_modal_steps '' \
    --cross_mass_steps '' \
    --cross_ae_steps '' \
    --cross_gan_steps '' \
    --cross_rel_steps 'mild-img' \
    --cross_mlm_steps '' \
    --cross_mrm_steps '' \
    --cross_mrfr_steps '' \
    --mlm_steps '' \
    --epoch_size 100000 \
    --max_epoch 10 \
    --bptt 128 \
    --max_len 128 \
    --fp16 True \
    --validation_metrics valid_I2T_acc,valid_T2I_acc \
    --max_region_num 100 \
    --accumulate_gradients 8 \
    --amp 1 \
    --refine_image False \
    --refine_encoder False \
    --input_fea_dir $FEA_PATH \
    --is_cross_modal True \
    --save_every_epoch 5 \
    --is_generation False \
    --is_understanding True \
    --is_pretrain False \
    --use_new_fea True \
    --t2i_flag True \
    --i2t_flag True \
    --eval_n 50 \
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
    --eval_only False \
    --is_mild True \
    --qp_type 'qp' \
    --seq_per_img 1 \
```            

## Multilingual image captioning

The task of multilingual image captioning is to generate captions in specific languages given input images. We evaluate M3P on Multi30K and MSCOCO.

## Multimodal machine translation

The task of multimodal machine translation is to generate sentences in target languages given source sentences together with related images as complementary information. We evaluate M3P on Multi30K. 

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
