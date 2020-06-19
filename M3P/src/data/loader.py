# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch
import gc
from .dataset_finetune import CaptionDataset,RetrievalDataset,EvaluateCaptionDataset,EvaluateRetrievalDataset
from .MILD_finetune import MILDCaptionDataset,MILDRetrievalDataset,MILDEvaluateRetrievalDataset,MILDEvaluateCaptionDataset
from .MT_finetune import MTCaptionDataset,EvaluateMTCaptionDataset
import pandas as pd
from .tokenization import XLMRTokenizer

logger = getLogger()

def check_data_params(params):
    """
    Check datasets parameters.
    """
    # data path
    assert os.path.isdir(params.data_path), params.data_path

    # check languages
    params.langs = params.lgs.split('-') if params.lgs != 'debug' else ['en']
    params.ft_lgs = params.ft_lgs.split('-') if params.ft_lgs != 'debug' else ['en']
    assert len(params.langs) == len(set(params.langs)) >= 1
    # assert sorted(params.langs) == params.langs
    params.id2lang = {k: v for k, v in enumerate(sorted(params.langs))}
    params.lang2id = {k: v for v, k in params.id2lang.items()}
    params.n_langs = len(params.langs)

    # CLM steps
    clm_steps = [s.split('-') for s in params.clm_steps.split(',') if len(s) > 0]
    params.clm_steps = [(s[0], None) if len(s) == 1 else tuple(s) for s in clm_steps]
    assert all([(l1 in params.langs) and (l2 in params.langs or l2 is None) for l1, l2 in params.clm_steps])
    assert len(params.clm_steps) == len(set(params.clm_steps))

    # MLM / TLM steps
    mlm_steps = [s.split('-') for s in params.mlm_steps.split(',') if len(s) > 0]
    params.mlm_steps = [(s[0], None) if len(s) == 1 else tuple(s) for s in mlm_steps]
    assert all([(l1 in params.langs) and (l2 in params.langs or l2 is None) for l1, l2 in params.mlm_steps])
    assert len(params.mlm_steps) == len(set(params.mlm_steps))

    # parallel classification steps
    params.pc_steps = [tuple(s.split('-')) for s in params.pc_steps.split(',') if len(s) > 0]
    assert all([len(x) == 2 for x in params.pc_steps])
    assert all([l1 in params.langs and l2 in params.langs for l1, l2 in params.pc_steps])
    assert all([l1 != l2 for l1, l2 in params.pc_steps])
    assert len(params.pc_steps) == len(set(params.pc_steps))

    # machine translation steps
    params.mt_steps = [tuple(s.split('-')) for s in params.mt_steps.split(',') if len(s) > 0]
    assert all([len(x) == 2 for x in params.mt_steps])
    assert all([l1 in params.langs and l2 in params.langs for l1, l2 in params.mt_steps])
    assert all([l1 != l2 for l1, l2 in params.mt_steps])
    assert len(params.mt_steps) == len(set(params.mt_steps))
    assert len(params.mt_steps) == 0 or not params.encoder_only

    # denoising auto-encoder steps
    params.ae_steps = [s for s in params.ae_steps.split(',') if len(s) > 0]
    assert all([lang in params.langs for lang in params.ae_steps])
    assert len(params.ae_steps) == len(set(params.ae_steps))
    assert len(params.ae_steps) == 0 or not params.encoder_only or params.is_cross_modal

    # mass steps
    params.mass_steps = [s for s in params.mass_steps.split(',') if len(s) > 0]
    mass_steps = []
    for src in params.mass_steps:
        for tgt in params.mass_steps:
            if src != tgt:
                mass_steps.append(tuple([src, tgt]))

    #text steps for xnli and ntg
    text_steps = [s.split('-') for s in params.text_steps.split(',') if len(s) > 0]
    params.text_steps = [(s[0], None) if len(s) == 1 else tuple(s) for s in text_steps]
    # params.text_steps = [s for s in params.text_steps.split(',') if len(s) > 0]

    # cross-modal steps
    params.cross_modal_steps = [tuple(s.split('-')) for s in params.cross_modal_steps.split(',') if len(s) > 0]
    # cross-mass and cross-ae
    params.cross_mass_steps = [tuple(s.split('-')) for s in params.cross_mass_steps.split(',') if len(s) > 0]
    params.cross_ae_steps= [tuple(s.split('-')) for s in params.cross_ae_steps.split(',') if len(s) > 0]
    params.cross_gan_steps = [tuple(s.split('-')) for s in params.cross_gan_steps.split(',') if len(s) > 0]

    params.cross_rel_steps = [tuple(s.split('-')) for s in params.cross_rel_steps.split(',') if len(s) > 0]
    params.cross_mlm_steps = [tuple(s.split('-')) for s in params.cross_mlm_steps.split(',') if len(s) > 0]
    params.cross_mrm_steps = [tuple(s.split('-')) for s in params.cross_mrm_steps.split(',') if len(s) > 0]
    params.cross_mrfr_steps = [tuple(s.split('-')) for s in params.cross_mrfr_steps.split(',') if len(s) > 0]

    # check that we can evaluate on BLEU
    assert params.eval_bleu is False or len(params.mt_steps + params.bt_steps + mass_steps) > 0


    #cross-lingual part

    required_mono = set(
        [l1 for l1, l2 in (params.mlm_steps + params.clm_steps) if l2 is None])
    params.mono_dataset = {
        lang: {
            splt: os.path.join(params.cross_lingual_path, '%s.%s.pth' % (lang, splt))
            for splt in ['train', 'valid']
        } for lang in params.langs if lang in required_mono
    }
    assert all([all([os.path.isfile(p) for p in paths.values()]) for paths in params.mono_dataset.values()])

    #para dataset

    # check parallel datasets
    required_para_train = set(params.clm_steps + params.mlm_steps + params.pc_steps + params.mt_steps)
    required_para = required_para_train | set([(l2, l3) for _, l2, l3 in params.bt_steps])
    params.para_dataset = {
        (src, tgt): {
            splt: (os.path.join(params.cross_lingual_path, 'para','%s-%s.%s.%s.pth' % (src, tgt, src,splt)),
                   os.path.join(params.cross_lingual_path, 'para','%s-%s.%s.%s.pth' % (src, tgt, tgt,splt)))
            for splt in ['train', 'valid']
            if splt != 'train' or (src, tgt) in required_para_train or (tgt, src) in required_para_train
        } for src in params.langs for tgt in params.langs
        if src < tgt and ((src, tgt) in required_para or (tgt, src) in required_para)
    }
    for paths in params.para_dataset.values():
        for p1, p2 in paths.values():
            if not os.path.isfile(p1):
                logger.error("{%s} not found"%(p1))
            if not os.path.isfile(p2):
                logger.error("{%s} not found"%(p1))
    assert all([all([os.path.isfile(p1) and os.path.isfile(p2) for p1, p2 in paths.values()]) for paths in
                params.para_dataset.values()])


    tokenizer = XLMRTokenizer(params.vocab_path)
    #params.tokenizer= tokenizer

    params.eos_index =tokenizer.eos_token_id
    params.pad_index = tokenizer.pad_token_id
    params.mask_index = tokenizer.mask_token_id
    params.n_words = tokenizer.vocab_size

def load_captioning_data(params, data):
    data['cross_modal'] = {}
    required_cross_modal_train = set(params.cross_modal_steps)#must need tasks

    for src, tgt in required_cross_modal_train:
        logger.info('============ Cross Modal data (%s-%s)' % (src, tgt))

        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}

        for splt in ['train', 'valid', 'test']:
            if splt=='test' and (src=='sbu' or src=='google'):
                continue
            if params.is_master==False and splt == 'valid':
                continue
            if params.is_master==False and splt == 'test': # multi gpu only support retrieval evaluation
                continue

            # no need to load training data for evaluationpara
            if splt == 'train' and params.eval_only:
                continue
            # cap_path = params.cross_modal_dataset[(src, tgt)][splt]
            if src == 'google' or src == 'sbu':
                if splt == 'valid':
                    _captions = \
                        pd.read_csv(os.path.join(params.data_path, 'uvl_captions', '%s.valid.csv' % src), sep='\t')[
                            'caption']
                else:
                    _captions = pd.read_csv(os.path.join(params.data_path, 'uvl_captions', '%s.csv' % src), sep='\t')[
                        'caption']
            else:
                _caption_dict = {}
                if params.ft_all: #each card with different languages
                    _local_rank = params.local_rank
                    _select_lg = _local_rank%len(params.ft_lgs)
                    lg = params.ft_lgs[_select_lg]
                    _captions = pd.read_pickle(
                        os.path.join(params.data_path, 'uvl_captions', '%s.%s.pkl' % (src, lg)))
                    # _caption_dict[lg] = _captions
                else:
                    if len(params.ft_lgs) > 0:
                        lg = params.ft_lgs[0]
                        _captions = pd.read_pickle(
                            os.path.join(params.data_path, 'uvl_captions', '%s.%s.pkl' % (src, lg)))
                        # _caption_dict[lg] = _captions

                    else:
                        _captions = pd.read_pickle(os.path.join(params.data_path, 'uvl_captions', '%s.pkl' % src))
                        lg = 'en'
                logger.info('select language (%s-%s)' % (str(params.local_rank), lg))
            # create ParallelDataset

            if params.is_generation:
                # if params.is_pretrain:
                #     dataset = VLMPretrainCapDataset(
                #    captions=_captions,params=params,
                #     mode=splt,data_type=src,
                # )
                # else:
                if splt=='test':
                    dataset =EvaluateCaptionDataset( captions=_captions,params=params,
                mode=splt,data_type=src)
                else:
                    dataset = CaptionDataset(
                       captions=_captions,params=params,
                        mode=splt,data_type=src,
                    )

            if splt != 'train':
                dataset.tokens_per_batch = -1

            # if there are several processes on the same machine, we can split the dataset
            # if splt == 'train' and params.n_gpu_per_node > 1 and params.split_data:
            #     n_sent = len(dataset) // params.n_gpu_per_node
            #     a = n_sent * params.local_rank
            #     b = n_sent * params.local_rank + n_sent
            #     dataset.select_data(a, b)

            data['cross_modal'][(src, tgt)][splt] = dataset

            logger.info("")

    logger.info("")

def load_retrieval_data(params, data):
    data['cross_modal'] = {}
    required_cross_modal_train = set(params.cross_rel_steps)  # must need tasks

    for src, tgt in required_cross_modal_train:

        logger.info('============ Cross Modal data (%s-%s)' % (src, tgt))

        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}
        if len(params.ft_lgs)>0:
            data['cross_modal'][(src, tgt)]['test']= {}
        for splt in ['train', 'valid', 'test']:
            if splt=='test' and (src=='sbu' or src=='google'):
                continue
            if params.is_master == False and splt == 'valid':
                continue
            if src=='google' and params.is_master == False and splt == 'test':
                continue
            if src=='sbu' and params.is_master == False and splt == 'test':
                continue

            # no need to load training data for evaluationpara
            if splt == 'train' and params.eval_only:
                continue
            # cap_path = params.cross_modal_dataset[(src, tgt)][splt]
            if src == 'google' or src == 'sbu':
                if splt == 'valid':
                    _captions = \
                        pd.read_csv(os.path.join(params.data_path, 'uvl_captions', '%s.valid.csv' % src), sep='\t')[
                            'caption']
                else:
                    _captions = pd.read_csv(os.path.join(params.data_path, 'uvl_captions', '%s.csv' % src), sep='\t')[
                        'caption']
            else:
                _caption_dict = {}
                if len(params.ft_lgs)>0:
                    for lg in params.ft_lgs:
                        _captions = pd.read_pickle(os.path.join(params.data_path, 'uvl_captions', '%s.%s.pkl' % (src,lg)))
                        _caption_dict[lg] = _captions
                else:
                    _caption_dict['en'] = pd.read_pickle(os.path.join(params.data_path, 'uvl_captions', '%s.pkl' % src))

            # create ParallelDataset
            if params.is_understanding:

                if splt=='test':
                    if len(params.ft_lgs)>0:
                        for lg in params.ft_lgs:
                            dataset = EvaluateRetrievalDataset(caption_dict=_caption_dict, params=params,
                                                           mode=splt, data_type=src,lang=lg)
                            dataset.tokens_per_batch = -1
                            data['cross_modal'][(src, tgt)]['test'][lg] = dataset
                        # else:
                        #     dataset = EvaluateRetrievalDataset(caption_dict=_caption_dict, params=params,
                        #                            mode=splt, data_type=src)
                else:
                    dataset = RetrievalDataset(caption_dict=_caption_dict, params=params,
                                           mode=splt, data_type=src)

            if splt != 'train':
                dataset.tokens_per_batch = -1

            # # if there are several processes on the same machine, we can split the dataset
            # if splt == 'train' and params.n_gpu_per_node > 1 and params.split_data:
            #     n_sent = len(dataset) // params.n_gpu_per_node
            #     a = n_sent * params.local_rank
            #     b = n_sent * params.local_rank + n_sent
            #     dataset.select_data(a, b)

            if len(params.ft_lgs)>0 and splt=='test':
                return
            data['cross_modal'][(src, tgt)][splt] = dataset

            logger.info("")

    logger.info("")

def load_mt_data(params, data):
    data['cross_modal'] = {}
    required_cross_modal_train = set(params.cross_modal_steps)#must need tasks

    for src, tgt in required_cross_modal_train:
        logger.info('============ Multimodal MT data (%s-%s)' % (src, tgt))

        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}

        for splt in ['train', 'valid', 'test']:
            if splt=='test' and (src=='sbu' or src=='google'):
                continue
            if params.is_master==False and splt == 'valid':
                continue
            if params.is_master==False and splt == 'test': # multi gpu only support retrieval evaluation
                continue

            # no need to load training data for evaluationpara
            if splt == 'train' and params.eval_only:
                continue

            _caption_dict = {}
            if len(params.ft_lgs) ==2:
                src_lg = params.ft_lgs[0]
                tgt_lg = params.ft_lgs[1]
                _captions = pd.read_pickle(
                    os.path.join(params.data_path, 'uvl_captions','%s.%s-%s.pkl' % (src, src_lg,tgt_lg)))

            # create ParallelDataset
            if params.is_generation:
                if splt=='test':
                    dataset =EvaluateMTCaptionDataset( captions=_captions,params=params,
                mode=splt,data_type=src)
                else:
                    dataset = MTCaptionDataset(
                       captions=_captions,params=params,
                        mode=splt,data_type=src,
                    )

            logger.info('performing MT direction (%s-%s)' % (src_lg,tgt_lg))
            if splt != 'train':
                dataset.tokens_per_batch = -1

            data['cross_modal'][(src, tgt)][splt] = dataset

            logger.info("")

    logger.info("")

def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    if params.debug_train:
        path = path.replace('train', 'valid')
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    return data

def load_mono_data(params, data):
    """
    Load monolingual data.
    """
    data['mono'] = {}
    data['mono_stream'] = {}

    for lang in params.mono_dataset.keys():

        logger.info('============ Monolingual data (%s)' % lang)

        assert lang in params.langs and lang not in data['mono']
        data['mono'][lang] = {}
        data['mono_stream'][lang] = {}

        for splt in ['train', 'valid']:

            # no need to load training data for evaluation
            if splt == 'train' and params.eval_only:
                continue

            # load data / update dictionary parameters / update data
            mono_data = load_binarized(params.mono_dataset[lang][splt], params)

            # create stream dataset
            data['mono_stream'][lang][splt] = StreamDataset(mono_data['sentences'], mono_data['positions'], params)

            # if there are several processes on the same machine, we can split the dataset
            if splt == 'train' and params.split_data and 1 < params.n_gpu_per_node <= data['mono_stream'][lang][splt].n_batches:
                n_batches = data['mono_stream'][lang][splt].n_batches // params.n_gpu_per_node
                a = n_batches * params.local_rank
                b = n_batches * params.local_rank + n_batches
                data['mono_stream'][lang][splt].select_data(a, b)

            # for online back-translation, we need a non-stream (batched) dataset

            del mono_data
            gc.collect()
            logger.info("")

    logger.info("")

def load_mild_retrieval_data(params, data):
    data['cross_modal'] = {}
    required_cross_modal_train = set(params.cross_rel_steps)  # must need tasks

    for src, tgt in required_cross_modal_train:

        logger.info('============ MIlD data (%s-%s)' % (src, tgt))

        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}
        if len(params.ft_lgs)>0:
            data['cross_modal'][(src, tgt)]['test']= {}
        for splt in ['train', 'valid', 'test']:
            if params.is_master == False and splt == 'valid':
                continue
            # no need to load training data for evaluationpara
            if splt == 'train' and params.eval_only:
                continue
            # cap_path = params.cross_modal_dataset[(src, tgt)][splt]

            _caption_dict = {}
            if params.ft_all:
                _local_rank = params.local_rank
                _select_lg = _local_rank % len(params.ft_lgs)
                lg = params.ft_lgs[_select_lg]
                _captions = pd.read_pickle(os.path.join(params.data_path, 'mild_caption', '%s.%s.pkl' % (src, lg)))
                _caption_dict[lg] = _captions
            else:
                if len(params.ft_lgs)>0:
                    # for lg in params.ft_lgs:
                    lg = params.ft_lgs[0]
                    _captions = pd.read_pickle(os.path.join(params.data_path, 'mild_caption', '%s.%s.pkl' % (src,lg)))
                    _caption_dict[lg] = _captions
                else:
                    lg='en'
                    _caption_dict['en'] = pd.read_pickle(os.path.join(params.data_path, 'mild_caption', '%s.pkl' % src))
            logger.info('select language (%s-%s)' % (str(params.local_rank), lg))
            # create ParallelDataset
            if params.is_understanding:
                if splt=='test':
                    if len(params.ft_lgs)>0:
                        #for lg in params.ft_lgs: # master language will be evaluated
                        dataset = MILDEvaluateRetrievalDataset(caption_dict=_caption_dict, params=params,
                                                       mode=splt, data_type=src,lang=lg)
                        dataset.tokens_per_batch = -1
                        data['cross_modal'][(src, tgt)]['test'][lg] = dataset
                        # else:
                        #     dataset = EvaluateRetrievalDataset(caption_dict=_caption_dict, params=params,
                        #                            mode=splt, data_type=src)
                else:
                    dataset = MILDRetrievalDataset(caption_dict=_caption_dict, params=params,
                                           mode=splt, data_type=src)

            if splt != 'train':
                dataset.tokens_per_batch = -1

            if len(params.ft_lgs)>0 and splt=='test':
                return
            data['cross_modal'][(src, tgt)][splt] = dataset

            logger.info("")

    logger.info("")

def load_mild_captioning_data(params, data):
    data['cross_modal'] = {}
    required_cross_modal_train = set(params.cross_modal_steps)#must need tasks

    for src, tgt in required_cross_modal_train:
        logger.info('============ MIlD data (%s-%s)' % (src, tgt))

        assert (src, tgt) not in data['cross_modal']
        data['cross_modal'][(src, tgt)] = {}

        for splt in ['train', 'valid', 'test']:
            if params.is_master==False and splt == 'valid':
                continue
            if params.is_master==False and splt == 'test': # multi gpu only support retrieval evaluation
                continue

            # no need to load training data for evaluationpara
            if splt == 'train' and params.eval_only:
                continue

            _caption_dict = {}
            if params.ft_all:
                _local_rank = params.local_rank
                _select_lg = _local_rank % len(params.ft_lgs)
                lg = params.ft_lgs[_select_lg]
                _captions = pd.read_pickle(os.path.join(params.data_path, 'mild_caption', '%s.%s.pkl' % (src, lg)))
                _caption_dict[lg] = _captions
            else:
                if len(params.ft_lgs) > 0:
                    lg = params.ft_lgs[0]
                    _captions = pd.read_pickle(
                        os.path.join(params.data_path, 'mild_caption', '%s.%s.pkl' % (src, lg)))
                    _caption_dict[lg] = _captions
                else:
                    lg ='en'
                    _caption_dict['en'] = pd.read_pickle(os.path.join(params.data_path, 'mild_caption', '%s.pkl' % src))
            logger.info('select language (%s-%s)' % (str(params.local_rank), lg))
            # create ParallelDataset

            if params.is_generation:
                _captions = _caption_dict[lg]
                if splt=='test':
                    dataset =MILDEvaluateCaptionDataset( captions=_captions,params=params,
                mode=splt,data_type=src)
                else:
                    dataset = MILDCaptionDataset(
                       captions=_captions,params=params,
                        mode=splt,data_type=src,
                    )

            if splt != 'train':
                dataset.tokens_per_batch = -1

            data['cross_modal'][(src, tgt)][splt] = dataset

            logger.info("")

    logger.info("")

#only tex

def load_data(params):
    """
    Load monolingual data.
    The returned dictionary contains:
        - dico (dictionary)
        - vocab (FloatTensor)
        - train / valid / test (monolingual datasets)
    """
    data = {}
    data['cross_modal'] = {}
    data['text'] = {}
    data['mono_stream'] = {}

    if params.is_understanding:
        if params.is_mild:
            load_mild_retrieval_data(params,data)
        else:
            load_retrieval_data(params,data)
    if params.is_generation:
        if params.is_mild:
            load_mild_captioning_data(params,data)
        elif params.is_mt:
            load_mt_data(params,data)
        else:
            load_captioning_data(params,data)

    # load_mono_data(params,data)

    # monolingual data summary
    logger.info('============ Data summary')
    if params.is_pretrain:
        logger.info('============ Pretrain Data collate')
    else:
        logger.info('============ Finetune Data collate')

    # cross modal data summary
    for (src, tgt), v in data['cross_modal'].items():
        for data_set in v.keys():
            if params.is_generation:
                logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('CrossModal data for captioning', data_set, '%s-%s' % (src, tgt),
                                                                    len(v[data_set])))
            elif params.is_understanding:
                if data_set=='test' and len(params.ft_lgs)>0 and params.is_pretrain==False and params.is_mild==False:
                    for lg in params.ft_lgs:
                        logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Cross-lingual CrossModal data for retrieval', data_set,
                                                                                '%s-%s-lg:%s' % (src, tgt,lg),
                                                                                len(v[data_set][lg])))
                else:
                    logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('CrossModal data for retrieval', data_set,
                                                                            '%s-%s' % (src, tgt),
                                                                            len(v[data_set])))
            else:
                logger.info('no data for training')
            #expand for other dataset and corresponding tasks

    #cross-lingual part
    logger.info('============ Data summary Cross-lingual part')
    for lang, v in data['mono_stream'].items():
        for data_set in v.keys():
            logger.info(
                '{: <18} - {: >5} - {: >12}:{: >10}'.format('Monolingual data', data_set, lang, len(v[data_set])))

    for lang, v in data['text'].items():
        for data_set in v.keys():
            logger.info(
                '{: <18} - {: >5} - {: >12}:{: >10}'.format('Text-only data', data_set, lang, len(v[data_set])))

    logger.info("")

    return data
