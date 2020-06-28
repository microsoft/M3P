# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
import os
import math
import time
import random
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F

def batch_sentences(sentences, lg_ids=None):
    """
    Take as input a list of n sentences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(1)
    if lg_ids is not None:
        lgs = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(4)
    else:
        lgs = None
    sent[0] = 0
    for i, s in enumerate(sentences):
        if lengths[i] > 2:  # if sentence not empty
            sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
        sent[lengths[i] - 1, i] = 2
        if lg_ids is not None:
            lgs[:, i] = lg_ids[i]

    if lgs is None:
        return sent, lengths
    return sent, lengths, lgs


def batch_sentences_v2(sentences, lm_labels=None):
    """
    Take as input a list of n sentences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(1)
    if lm_labels is not None:
        _labels = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(-1)

    sent[0] = 0
    for i, s in enumerate(sentences):
        if lengths[i] > 2:  # if sentence not empty
            sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            if lm_labels is not None:
                lm = np.array(lm_labels[i])
                _labels[1:lengths[i] - 1, i].copy_(torch.from_numpy(lm.astype(np.int64)))
        sent[lengths[i] - 1, i] = 2
        if lm_labels is not None:
            _labels[lengths[i] - 1, i] = -1

    if lm_labels is not None:
        return sent, lengths, _labels
    return sent, lengths


def retrieval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    t2i_batch, i2t_batch = list(zip(*data))

    # t2i
    def generate_inputs(_batch):
        sent, att_feats, img_masks, box_feats, obj_labels, pos_labels, img_ids, langs = zip(
            *_batch)
        # sent = np.array(sent)
        _sent = []
        _pos_labels = []
        _img_ids = []
        _langs = []
        for s, p, i, l in zip(sent, pos_labels, img_ids, langs):
            _sent.extend(s)
            _pos_labels.extend(p)
            _img_ids.extend(i)
            _langs.extend(l)
        pos_labels = _pos_labels
        img_ids = _img_ids

        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)
        x_obj_labels = torch.stack(obj_labels, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
        x_obj_labels = x_obj_labels.view([-1] + list(tuple(x_obj_labels.size()[2:])))

        _inputs = [batch_sentences(_sent, _langs),
                   [x_img,
                    x_img_mask,
                    img_loc,
                    x_obj_labels,
                    pos_labels,
                    img_ids]  # google_replace
                   ]
        return _inputs

    _t2i_out = generate_inputs(t2i_batch) if t2i_batch is not None else None
    _i2t_out = generate_inputs(i2t_batch) if i2t_batch is not None else None
    all_return_results = [_t2i_out, _i2t_out]
    return all_return_results


def caption_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    sent, att_feats, img_masks, box_feats, img_ids = list(zip(*data))

    # t2i
    def generate_inputs():
        # sent = np.array(sent)

        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))

        _inputs = [batch_sentences(sent),
                   [x_img,
                    x_img_mask,
                    img_loc,
                    img_ids]  # google_replace
                   ]
        return _inputs

    all_return_results = generate_inputs()
    return all_return_results


def retrieval_pretrain_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    t2i_batch, i2t_batch = list(zip(*data))

    # t2i
    def generate_inputs(_batch):
        sent, att_feats, img_masks, box_feats, obj_labels, lm_label, itm_label, img_ids, ori_feats, masked_types = zip(
            *_batch)
        # sent = np.array(sent)
        _sent = []
        _img_ids = []
        lm_labels = []
        for s, i, p in zip(sent, img_ids, lm_label):
            _sent.extend(s)
            _img_ids.extend(i)
            lm_labels.extend(p)
        img_ids = _img_ids
        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)
        x_obj_labels = torch.stack(obj_labels, dim=0)
        x_img_ori = torch.stack(ori_feats, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        x_img_ori = x_img_ori.view([-1] + list(tuple(x_img_ori.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
        x_obj_labels = x_obj_labels.view([-1] + list(tuple(x_obj_labels.size()[2:])))

        _inputs = [batch_sentences_v2(_sent, lm_labels),
                   [x_img,
                    x_img_mask,
                    img_loc,
                    x_obj_labels,
                    itm_label,
                    x_img_ori,
                    img_ids]  # google_replace
                   ]
        return _inputs

    _t2i_out = generate_inputs(t2i_batch) if t2i_batch is not None else None
    _i2t_out = generate_inputs(i2t_batch) if i2t_batch is not None else None
    all_return_results = [_t2i_out, _i2t_out]
    return all_return_results


def mt_caption_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    src_sent,tgt_sent, att_feats, img_masks, box_feats, img_ids = list(zip(*data))

    # t2i
    def generate_inputs():
        # sent = np.array(sent)

        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))

        _inputs = [batch_sentences(src_sent),
                   batch_sentences(tgt_sent),
                   [x_img,
                    x_img_mask,
                    img_loc,
                    img_ids]  # google_replace
                   ]
        return _inputs

    all_return_results = generate_inputs()
    return all_return_results


def ntg_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    src_sent,tgt_sent = list(zip(*data))

    # t2i
    _inputs = [batch_sentences(src_sent),
               batch_sentences(tgt_sent),
               ]
    return _inputs