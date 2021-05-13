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
from .tokenization import XLMRTokenizer

logger = getLogger()

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10


class NTGParallelDataset(object):
    def __init__(self, captions_src,captions_tgt, params, mode='train', data_type='ntg',bin_data=None):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.max_vocab = params.max_vocab
        self.min_count = params.min_count

        self.max_len = params.max_len

        self.tokenizer = XLMRTokenizer(params.vocab_path)

        self.mode = mode

        # self.is_xlm = params.use_xlm
        # if self.is_xlm:

        self.params = params

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        # after assign image ids
        _sent1 = []
        _sent2 = []
        _lengths_1 = []
        _lengths_2 = []
        #sent = [self.tokenize(self.raw_caps[sent_id]) for sent_id in sentence_ids]

        if bin_data is not None:
            _sent1 =  bin_data['sent1']
            _sent2 = bin_data['sent2']
            _lengths_1 = bin_data['len1']
            _lengths_2 = bin_data['len2']
        else:
            for _src,_tgt in zip(captions_src,captions_tgt):
                _src_tokenized = self.tokenize(_src)
                _tgt_tokenized = self.tokenize(_tgt)
                _sent1.append(_src_tokenized)
                _sent2.append(_tgt_tokenized)
                _lengths_1.append(len(_src_tokenized))
                _lengths_2.append(len(_tgt_tokenized))

        self.sent1 = _sent1
        self.sent2 = _sent2
        self.lengths_1 = np.array(_lengths_1)
        self.lengths_2 = np.array(_lengths_2)

    def tokenize(self, sent):
        s = str(sent)
        indexed = self.tokenizer.encode(s)
        indexed = indexed
        indexed = np.int32(indexed)
        indexed = indexed[:self.max_len]
        return indexed

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.sent1)


    def batch_sentences(self,sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(1) #sampe

        sent[0] = 0
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = 2

        return sent, lengths

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        if process google dataset : image_ids[x]=x
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            sent1 = self.batch_sentences([self.sent1[sent_id] for sent_id in sentence_ids])
            sent2 = self.batch_sentences([self.sent2[sent_id] for sent_id in sentence_ids])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False,
                    ):
        """
        Return a sentences iterator.
        """

        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.sent1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.sent1)
        assert type(shuffle) is bool and type(group_by_size) is bool
        # assert group_by_size is False or shuffle is True

        # sentence lengths

        lengths = self.lengths_1 + self.lengths_2 + 4


        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(lengths))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])

        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)