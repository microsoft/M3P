# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

import json
import argparse
import torch
import numpy as np
from torch import nn

from src.slurm import init_signal_handler, init_distributed_mode
from src.data.loader import check_data_params, load_data
from src.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from src.model import check_model_params, build_model
from src.xtrainer import XTrainer
from src.evaluation.xevaluator import XEvaluator

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="/tmp/dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="M3P",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16
    parser.add_argument("--fp16", type=bool_flag, default=True,
                        help="Run model with float16")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder")
    parser.add_argument("--english_only", type=bool_flag, default=True,
                        help="Only use english domain (equal to only use one language)")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=1024,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=12,
                        help="Number of Transformer layers")
    parser.add_argument("--n_dec_layers", type=int, default=-1,
                        help="Number of Decoder Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=True,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--attention_setting", type=str, default="v1", choices=["v1", "v2"],
                        help="Setting for attention module, benefits for distinguish language")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    parser.add_argument("--word_mass", type=float, default=0.5,
                        help="Randomly mask input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="en",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    #path for
    parser.add_argument("--vocab_path", type=str, default="",
                        help="bpe vocab for en")
    parser.add_argument("--input_fea_dir", type=str, default="",
                        help="parent path for features")
    parser.add_argument("--google_path", type=str, default="",
                        help="path to CC")
    parser.add_argument("--sbu_path", type=str, default="",
                        help="path to sbu")
    parser.add_argument("--coco_path", type=str, default="",
                        help="path to coco")
    parser.add_argument("--flicker_path", type=str, default="",
                        help="path to flickr")
    parser.add_argument("--mild_path", type=str, default="",
                        help="path to mild")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch_size")
    parser.add_argument("--seq_per_img", type=int, default=5,
                        help="captions per img")
    parser.add_argument("--max_region_num", type=int, default=100,
                        help="the number of objects")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--min_len", type=int, default=2,
                        help="Minimum length of sentences (after BPE)")
    parser.add_argument("--max_len", type=int, default=60,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")

    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")


    # batch parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_mass", type=str, default="1",
                        help="MASS coefficient")
    parser.add_argument("--lambda_ic", type=str, default="1",
                        help="image_caption coefficient")
    parser.add_argument("--lambda_imlm", type=str, default="1",
                        help="task1 coefficient")
    parser.add_argument("--lambda_ida", type=str, default="1",
                        help="task2 coefficient")
    parser.add_argument("--lambda_tifg", type=str, default="1",
                        help="task3 coefficient")
    parser.add_argument("--lambda_rel", type=str, default="1",
                        help="relation task  coefficient")
    parser.add_argument("--lambda_mrm", type=str, default="1",
                        help="mask region task  coefficient")
    parser.add_argument("--lambda_mrfr", type=str, default="1",
                        help="mask region regression task  coefficient")
    parser.add_argument("--lambda_t2i", type=str, default="1",
                        help="text to image coefficient")
    parser.add_argument("--lambda_i2t", type=str, default="1",
                        help="image to text coefficient")

    # training steps base steps
    #not support currently
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mass_steps", type=str, default="",
                        help="MASS prediction steps")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # for generation step
    parser.add_argument("--cross_modal_steps", type=str, default="",
                        help="ic steps")
    parser.add_argument("--cross_mass_steps", type=str, default="",
                        help="imlm steps")
    parser.add_argument("--cross_ae_steps", type=str, default="",
                        help="ida steps")
    parser.add_argument("--cross_gan_steps", type=str, default="",
                        help="tifg steps")

    #for understanding step
    parser.add_argument("--cross_rel_steps", type=str, default="",
                        help="uvl relation steps")
    parser.add_argument("--cross_mlm_steps", type=str, default="",
                        help="mask lm cross-modal steps")
    parser.add_argument("--cross_mrm_steps", type=str, default="",
                        help="mask region steps")
    parser.add_argument("--cross_mrfr_steps", type=str, default="",
                        help="mrfr steps")

    #text only step
    parser.add_argument("--text_steps", type=str, default="",
                        help="text steps ")

    # reload a pretrained model
    parser.add_argument("--reload_model", type=str, default="",  # ./data/models/mlm_en_2048.pth
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_caption", type=bool_flag, default=False,
                        help="Evaluate BLEU and CIDEr for captioning")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_pretrain", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # AoA Model
    parser.add_argument("--refine_image", type=bool_flag, default=False,
                        help="If use refine module after image embedding")
    parser.add_argument("--refine_layers", type=int, default=6,
                        help="refine_layers for refine module")
    parser.add_argument("--refine_encoder", type=bool_flag, default=False,
                        help="If use refine module after image encoder")

    parser.add_argument("--use_noise", type=bool_flag, default=False,
                    help="whether use noise bart mask for autoencoder ")

    parser.add_argument("--accumulate_gradients", type=int, default=-1,
                        help="accumulated gradients during trianing")
    parser.add_argument("--amp", type=int, default=1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")
    parser.add_argument("--use_memory", type=int, default=0,
                        help="Use externel memory")


    parser.add_argument("--is_cross_modal", type=bool_flag, default=True,
                        help="If use one stream fro image and text")
    parser.add_argument("--is_understanding", type=bool_flag, default=False,
                        help="whether training for understanding tasks")
    parser.add_argument("--is_generation", type=bool_flag, default=False,
                        help="whether training for generation tasks")
    parser.add_argument("--is_pretrain", type=bool_flag, default=False,
                        help="pretrain or finetune dataset")

    parser.add_argument("--use_externel_att", type=bool_flag, default=False,
                        help="If use externel multi-head att during decoder")
    parser.add_argument("--use_enc_att", type=bool_flag, default=False,
                        help="If use encoder-decoder framework for understanding tasks")


    parser.add_argument("--save_every_epoch", type=int, default=1,
                        help="how many epoches for saving")

    parser.add_argument("--multi_reload_model", type=str,
                        default="",
                        help="model weights used to be averaged with read_load_model_weight")

    parser.add_argument("--bin_cls_loss_weight", type=float, default=1,
                        help="the weight of binary classification loss when  finetining")
    parser.add_argument("--multi_cls_loss_weight", type=float, default=1,
                        help="the weight of multiple classification loss when finetining")
    parser.add_argument("--sample_n", type=int, default=2,
                        help="number of samples during retrieval")
    parser.add_argument("--t2i_flag", type=bool_flag, default=True,
                        help="whether sample text 2 image")
    parser.add_argument("--i2t_flag", type=bool_flag, default=True,
                        help="whether sample image 2 text")
    parser.add_argument("--coco_method", type=str,
                        default="CIDEr",
                        help="which evaluation metric selection for coco evaluate")
    parser.add_argument("--eval_n", type=int, default=150,
                        help="n_sentences for evaluation,including retrieval and generation")
    parser.add_argument("--eval_images", type=int, default=-1,
                        help="n_images for evaluation retrieval")
    parser.add_argument("--retrieval_batch", type=int, default=1,
                        help="batches for image retrieval evaluation")
    parser.add_argument("--retrieval_workers", type=int, default=4,
                        help="batches for image retrieval evaluation")
    parser.add_argument("--test_splits", type=int, default=10,
                        help="n_sentences for each ")
    parser.add_argument("--use_new_fea", type=bool_flag, default=False,
                        help="whether use old version pythia")
    parser.add_argument("--eval_path", type=str, default="/tmp/dumped/",
                    help="Experiment results path")
    parser.add_argument("--google_valid_path", type=str, default="./data/google_captions",
                        help="path to CC")
    parser.add_argument("--train_order_path", type=str, default="./data/",
                        help="path to CC")
    parser.add_argument("--cross_lingual_path", type=str, default="./data/",
                        help="path to cross-lingual data")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="workers for multi_thread dataset")
    parser.add_argument("--ft_lgs", type=str, default="en-fr-de",
                        help="Languages for downstream tasks")

    parser.add_argument("--is_mild", type=bool_flag, default=False,
                        help="whether use mild data ")
    parser.add_argument("--qp_type", type=str, default="q",
                        help="q or qp")
    parser.add_argument("--ft_all", type=bool_flag, default=False,
                        help="whether ft on all languages ")
    parser.add_argument("--is_mt", type=bool_flag, default=False,
                        help="whether mmt generation ")
    parser.add_argument("--mt_only_text", type=bool_flag, default=False,
                        help="whether mmt generation only use text ")
    parser.add_argument("--is_ntg", type=bool_flag, default=False,
                        help="whether ntg task ")

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)
    print(data)

    # build model
    # if params.encoder_only:
    model = build_model(params)

    # build trainer, reload potential checkpoints / build evaluator

    trainer = XTrainer(model,data,params)
    evaluator = XEvaluator(trainer, data, params)
    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:
            # MLM steps (also includes TLM if lang2 is not None)

            for lang1, lang2 in shuf_order(params.mlm_steps, params):
                if params.is_understanding:
                    trainer.mlm_step(lang1, lang2, params.lambda_mlm)

            # cross-modal caption steps
            for lang1, lang2 in shuf_order(params.cross_modal_steps, params):
                if params.is_mt:
                    trainer.mt_ic_step(lang1,lang2,params.lambda_ic)
                else:
                    trainer.ic_step(lang1, lang2, params.lambda_ic)

            for lang1, lang2 in shuf_order(params.cross_rel_steps, params):
                trainer.rel_step(lang1, lang2, params.lambda_t2i, params.lambda_i2t)

            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        evaluate_results = []
        import os
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))
            evaluate_results.append(json.dumps(scores))
            with open(os.path.join(params.dump_path, "epoch_{0}.eval_log".format(trainer.epoch)), 'w') as writer:
                for line in evaluate_results:
                    writer.write(line + '\n')

        # end of epoch
        trainer.save_best_model(scores)
        if trainer.epoch % params.save_every_epoch == 0 and params.is_master:
            trainer.save_model('model_pretrain_%i' % trainer.epoch)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
