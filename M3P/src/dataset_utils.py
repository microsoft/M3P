import os
import math
import time
import random
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

def batch_sentences(sentences,lm_labels=None,lg_ids=None):
    """
    Take as input a list of n sentences (torch.LongTensor vectors) and return
    a tensor of size (slen, n) where slen is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(1) #pad
    if lm_labels is not None:
        _labels = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(-1)
    if lg_ids is not None:
        lgs = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(4)
    else:
        lgs = None
    sent[0] = 0 #cls
    for i, s in enumerate(sentences):
        if lengths[i] > 2:  # if sentence not empty
            sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            if lm_labels is not None:
                lm = np.array(lm_labels[i])
                _labels[1:lengths[i]-1,i].copy_(torch.from_numpy(lm.astype(np.int64)))
        sent[lengths[i] - 1, i] = 2
        if lm_labels is not None:
            _labels[lengths[i]-1,i] = -1
        if lgs is not None:
            lgs[:,i] = lg_ids[i]

    if lm_labels is not None:
        return sent, lengths,_labels

    return sent,lengths,lgs


def get_loader(params,dataset,data_type, mode):
    if data_type=='google' or data_type=='sbu':
        assert mode != "test"
        train_sampler = RandomSampler(dataset) if params.n_gpu_per_node == 1 else DistributedSampler(dataset)
        eval_sampler = RandomSampler(dataset) #only master evaluate
        eval_bs = 8
        data_loader = DataLoader(dataset, batch_size=params.batch_size if mode=='train ' else eval_bs,
                                 sampler=train_sampler if mode == "train" else eval_sampler,
                                 collate_fn=retrieval_pretrain_collate if params.is_understanding else caption_collate,
                                 num_workers=params.num_workers if mode == "train" else 4)
    elif data_type=='coco' or data_type=='flicker' :
        if mode == "train":
            sampler = RandomSampler(dataset) if params.n_gpu_per_node == 1 else DistributedSampler(dataset)
        elif mode == "valid":
            sampler = SequentialSampler(dataset) #only master evaluate
        else:
            sampler = SequentialSampler(dataset)

        if params.is_generation:
            if params.is_mt:
                data_loader = DataLoader(dataset, batch_size=params.batch_size if mode != "test" else 1,
                                         sampler=sampler,
                                         collate_fn=mt_caption_collate if mode != "test" else mt_caption_eval_collate,
                                         num_workers=params.num_workers)
            else:
                data_loader = DataLoader(dataset, batch_size=params.batch_size if mode != "test" else 1, sampler=sampler,
                                          collate_fn=caption_collate if mode != "test" else caption_eval_collate,
                                         num_workers=params.num_workers)
        if params.is_understanding:
            if mode=='test':
                data_loader = DataLoader(dataset, batch_size=1,
                                         sampler=sampler,
                                         collate_fn=retrieval_eval_collate,
                                         num_workers=4)
            else:
                data_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=sampler,
                                          collate_fn=retrieval_collate,
                                         num_workers=4)

    elif data_type=='mild':
        if mode == "train":
            sampler = RandomSampler(dataset) if params.n_gpu_per_node == 1 else DistributedSampler(dataset)
        elif mode == "valid":
            sampler = SequentialSampler(dataset) #only master evaluate
        else:
            sampler = SequentialSampler(dataset)

        if params.is_generation:
            data_loader = DataLoader(dataset, batch_size=params.batch_size if mode != "test" else 1, sampler=sampler,
                                      collate_fn=caption_collate if mode != "test" else caption_eval_collate,
                                     num_workers=params.num_workers)
        if params.is_understanding:
            if mode=='test':
                data_loader = DataLoader(dataset, batch_size=1,
                                         sampler=sampler,
                                         collate_fn=retrieval_eval_collate,
                                         num_workers=4)
            else:
                data_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=sampler,
                                          collate_fn=retrieval_collate,
                                         num_workers=4)

    elif data_type=='ntg':
        if mode == "train":
            sampler = RandomSampler(dataset) if params.n_gpu_per_node == 1 else DistributedSampler(dataset)
        elif mode == "valid":
            sampler = SequentialSampler(dataset)  # only master evaluate
        else:
            sampler = SequentialSampler(dataset)

        if params.is_generation:
            if params.is_ntg:
                data_loader = DataLoader(dataset, batch_size=params.batch_size if mode != "test" else 1,
                                         sampler=sampler,
                                         collate_fn=ntg_collate,
                                         num_workers=params.num_workers)

    elif data_type=='slide':
        if mode == "train":
            sampler = RandomSampler(dataset) if params.n_gpu_per_node == 1 else DistributedSampler(dataset)
        elif mode == "valid":
            sampler = SequentialSampler(dataset)  # only master evaluate
        else:
            sampler = SequentialSampler(dataset)
            
        data_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=sampler,
                                 collate_fn=slide_collate,
                                 num_workers=params.num_workers)

    return data_loader

def retrieval_pretrain_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    t2i_batch,i2t_batch = list(zip(*data))

    # t2i
    def generate_inputs_t2i(_batch):
        sent,att_feats, img_masks, box_feats,obj_labels,lm_label, itm_label, img_ids,ori_feats,masked_types = zip(
            *_batch)
        #sent = np.array(sent)
        _sent = []
        _img_ids = []
        lm_labels = []
        for s,i,p in zip(sent,img_ids,lm_label):
            _sent.extend(s)
            _img_ids.extend(i)
            lm_labels.extend(p)
        img_ids = _img_ids
        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)
        x_obj_labels = torch.stack(obj_labels, dim=0)
        x_img_ori = torch.stack(ori_feats,dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        x_img_ori = x_img_ori.view([-1] + list(tuple(x_img_ori.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
        x_obj_labels = x_obj_labels.view([-1] + list(tuple(x_obj_labels.size()[2:])))

        _inputs = [batch_sentences(_sent,lm_labels),
                  [x_img,
                   x_img_mask,
                  img_loc,
                  x_obj_labels,
                  itm_label,
                   x_img_ori,
                  img_ids]  # google_replace
                  ]
        return _inputs

    # i2t
    def generate_inputs_i2t(_batch):
        sent, att_feats, img_masks, box_feats, obj_labels, lm_label, itm_label, img_ids, ori_feats, masked_types, clcm_sent, clcm_labels = zip(
            *_batch)
        # sent = np.array(sent)
        _sent = []
        _clcm_sent = []
        _img_ids = []
        lm_labels = []
        for s, ss, i, p in zip(sent, clcm_sent, img_ids, lm_label):
            _sent.extend(s)
            _clcm_sent.extend(ss)
            _img_ids.extend(i)
            lm_labels.extend(p)
        img_ids = _img_ids
        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)
        x_obj_labels = torch.stack(obj_labels, dim=0)
        x_img_ori = torch.stack(ori_feats, dim=0)
        x_clcm_labels = torch.stack(clcm_labels, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        x_img_ori = x_img_ori.view([-1] + list(tuple(x_img_ori.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
        x_obj_labels = x_obj_labels.view([-1] + list(tuple(x_obj_labels.size()[2:])))

        _inputs = [batch_sentences(_sent, lm_labels),
                   batch_sentences(_clcm_sent, None)[:2],
                   [x_clcm_labels,
                    x_img,
                    x_img_mask,
                    img_loc,
                    x_obj_labels,
                    itm_label,
                    x_img_ori,
                    img_ids]  # google_replace
                   ]
        return _inputs

    _t2i_out = generate_inputs_t2i(t2i_batch) if t2i_batch is not None else None
    _i2t_out = generate_inputs_i2t(i2t_batch) if i2t_batch is not None else None
    all_return_results = [_t2i_out,_i2t_out]
    return all_return_results


def retrieval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    t2i_batch,i2t_batch = list(zip(*data))

    # t2i
    def generate_inputs(_batch):
        sent,att_feats, img_masks, box_feats, obj_labels, pos_labels, img_ids,langs = zip(
            *_batch)
        #sent = np.array(sent)
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
        #x_obj_labels = torch.stack(obj_labels, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
        # x_obj_labels = x_obj_labels.view([-1] + list(tuple(x_obj_labels.size()[2:])))

        _inputs = [batch_sentences(_sent,lg_ids=_langs),
                  [x_img,
                   x_img_mask,
                  img_loc,
                    pos_labels,
                  img_ids]  # google_replace
                  ]
        return _inputs

    _t2i_out = generate_inputs(t2i_batch) if t2i_batch is not None else None
    _i2t_out = generate_inputs(i2t_batch) if i2t_batch is not None else None
    all_return_results = [_t2i_out,_i2t_out]
    return all_return_results

def retrieval_eval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    text, text_length,segmentt_ids,img, img_loc, _label = list(zip(*data))

    _inputs = [torch.stack(text,dim=0),torch.stack(text_length,dim=0),torch.stack(segmentt_ids,dim=0),
           torch.stack(img, dim=0).view(-1, 100, img[0].size()[-1]),
           torch.stack(img_loc, dim=0).view(-1, 100, img_loc[0].size()[-1]),
            torch.stack(_label, dim=0)
           ]
    return _inputs

def caption_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    sent, att_feats, img_masks, box_feats, img_ids = list(zip(*data))

    # t2i
    def generate_inputs():
        #sent = np.array(sent)

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

def caption_eval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    att_feats, img_masks, box_feats, img_ids = list(zip(*data))

    # t2i
    def generate_inputs():
        #sent = np.array(sent)

        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))

        _inputs = [x_img,
                   x_img_mask,
                  img_loc,
                  img_ids
                  ]
        return _inputs

    all_return_results = generate_inputs()
    return all_return_results

def slide_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    sent, att_feats, img_masks, box_feats, img_ids,cur_label = list(zip(*data))

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
                    img_ids],
                    cur_label
                   ]
        return _inputs

    all_return_results = generate_inputs()
    return all_return_results


def mt_caption_eval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    src_sent,att_feats, img_masks, box_feats, img_ids = list(zip(*data))

    # t2i
    def generate_inputs():
        #sent = np.array(sent)

        x_img = torch.stack(att_feats, dim=0)
        img_loc = torch.stack(box_feats, dim=0)
        x_img_mask = torch.stack(img_masks, dim=0)

        x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
        img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
        x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))

        _inputs = [batch_sentences(src_sent),
                   x_img,
                   x_img_mask,
                  img_loc,
                  img_ids
                  ]
        return _inputs

    all_return_results = generate_inputs()
    return all_return_results


def ntg_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    src_sent,tgt_sent = list(zip(*data))
    _inputs = [batch_sentences(src_sent),
               batch_sentences(tgt_sent),
               ]
    return _inputs