# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch
import h5py
import lmdb
import six
import os
import random
from torch.nn import functional as F
import json
import re
import os
from .dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from torch.utils.data.dataset import Dataset
from .tokenization import XLMRTokenizer


logger = getLogger()

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10


class NTGDataset(Dataset):
    def __init__(self, src_texts,tgt_texts, params, mode='train'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        # src_lg = params.ft_lgs[0]
        # tgt_lg = params.ft_lgs[1]
        self.src_texts = src_texts # src->tgt
        self.tgt_texts = tgt_texts

        self.max_len = params.max_len

        self.mode = mode
        self.tokenizer = XLMRTokenizer(params.vocab_path)

        self.params = params

        # feature related for image features

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank


        self.process_caption()

        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))
    def update_captions(self):
        #no need update
        logger.info('epoch ended')

    def process_caption(self):
        _all_captions = []
        for src_text,tgt_text in zip(self.src_texts,self.tgt_texts):
            _all_captions.append((src_text,tgt_text))
        self.raw_caps = _all_captions

    def tokenize(self, sent):
        s = str(sent)
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:self.max_len]
        indexed = np.int32(indexed)
        return indexed

    def __len__(self):
        return len(self.raw_caps)

    def __getitem__(self, index):
        cur_cap = self.raw_caps[index]
        source_sent = self.tokenize(cur_cap[0])
        target_sent = self.tokenize(cur_cap[1])

        img_feas = [source_sent,target_sent]
        return img_feas

class EvaluateNTGDataset(Dataset):
    def __init__(self, src_texts,tgt_texts, params, mode='train'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.src_texts = src_texts  # src->tgt
        self.tgt_texts = tgt_texts

        self.mode = mode
        self.params = params

        self.max_len = params.max_len
        self.tokenizer = XLMRTokenizer(params.vocab_path)
        # feature related for image features

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        # after assign image ids
        _all_captions = []

        for src_text,tgt_text in zip(self.src_texts,self.tgt_texts):
            _all_captions.append((src_text,tgt_text))

        self.raw_caps = _all_captions

        if self.n_gpu_per_node>1:
            each_gpu_len = len(self.raw_caps) // self.n_gpu_per_node
            self.raw_caps = self.raw_caps[self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]
            self.tgt_texts = self.tgt_texts[self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]
            self.each_gpu_len = each_gpu_len

        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))

    def __len__(self):
        return len(self.raw_caps)

    def tokenize(self, sent):
        s = str(sent)
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:self.max_len]
        indexed = np.int32(indexed)
        return indexed

    def __getitem__(self, index):
        # if self.mode=='test':
        #     return self.get_image_iterator(index)
        # index = index

        source_sent = self.tokenize(self.raw_caps[index][0])
        target_sent = self.tokenize(self.raw_caps[index][1])

        img_feas = [source_sent,target_sent]
        return img_feas

