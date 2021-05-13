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


class SlideDataset(Dataset):
    def __init__(self, captions, params, mode='train', data_type='coco'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.captions = captions # src->tgt

        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        self.mode = mode
        self.tokenizer = XLMRTokenizer(params.vocab_path)

        self.params = params

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.max_len = params.max_len
        self.sample_n = params.sample_n

        assert data_type in ['slide']

        if mode == 'train':
            train_file = os.path.join(params.input_fea_dir, params.slide_path, "train.h5")
            self.precess_reload(train_file)
        if mode == 'valid':
            validfile = os.path.join(params.input_fea_dir, params.slide_path, "valid.h5")
            self.precess_reload(validfile)
        if mode == 'test':
            testfile = os.path.join(params.input_fea_dir, params.slide_path, "test.new.h5")
            self.precess_reload(testfile)

        if self.mode!='train':
            _s1 = True
        else:
            _s1 = False
        self.process_caption(_s1)

        # after assign image ids
        self.data_type = data_type
        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))
    def update_captions(self):
        #no need update
        logger.info('epoch ended')

    def process_caption(self,is_test=False):
        if is_test:
            _all_captions = []
            _img_indexs = {}
            for i, img_id in enumerate(self.image_ids):
                _img_indexs[img_id] = i
                _cap_groups = self.captions[img_id]
                for _pair in _cap_groups:
                    _all_captions.append((_pair[1][1], _pair[-1], img_id))  # also assign label
            self.raw_caps = _all_captions
            self.img_indexs = _img_indexs
        else:
            pos_img_indexs = {}
            neg_img_indexs = {}
            _all_pos = []
            _all_neg = []
            for i,img_id in enumerate(self.image_ids):
                _cap_groups = self.captions[img_id]
                for _pair in _cap_groups:
                    if _pair[-1]==1:
                        _all_pos.append((_pair[1][1],_pair[-1],img_id))
                        pos_img_indexs[img_id]=i
                    else:
                        _all_neg.append((_pair[1][1],_pair[-1],img_id))
                        neg_img_indexs[img_id] = i
            self.raw_caps = _all_pos
            self.raw_neg_caps = _all_neg
            self.pos_img_indexs = pos_img_indexs
            self.neg_img_indexs = neg_img_indexs

            self.all_cap_neg_indices = list(range(0, len(self.raw_neg_caps)))

    def tokenize(self, sent):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:self.max_len]
        indexed = np.int32(indexed)
        return indexed

    def precess_reload(self, filename):

        cur_file = h5py.File(filename, "r")
        _image_ids = cur_file['image_id'][:]
        image_ids = [str(ss, encoding="utf8") for ss in _image_ids]

        wh = cur_file['wh'][:]
        num_boxes = np.ones(len(_image_ids), dtype='int64') * 100

        boxes = cur_file['bbox'][:]
        obj_features = cur_file['features'][:]


        self.image_ids = image_ids
        self.wh = wh
        self.num_boxes = num_boxes
        self.boxes = boxes
        self.obj_features = obj_features

    def __len__(self):
        return len(self.raw_caps)

    def norm_boxes(self, cur_boxes, h, w):
        # devided by image width and height
        x1, y1, x2, y2 = np.hsplit(cur_boxes, 4)
        # for location
        cur_boxes = np.hstack(
            (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
        cur_boxes = cur_boxes / np.linalg.norm(cur_boxes, 2, 1, keepdims=True)
        return cur_boxes

    def get_img_feature(self, index):

        object_features, box, num_boxes, wh = self.obj_features[index][:self.max_region_num], \
                                              self.boxes[index][:self.max_region_num], \
                                              self.num_boxes[index], \
                                              self.wh[index]
        img_id = self.image_ids[index]

        num_boxes = self.max_region_num

        # normalized
        att_feat = object_features.astype('float32')
        # att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        att_feat = torch.FloatTensor(att_feat)
        att_feat = F.normalize(att_feat, dim=-1).numpy()  # keep precision

        h, w = wh.astype('float32')
        loc_features = self.norm_boxes(box.astype('float32'), h, w)

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        return (att_feat, loc_features, image_mask, img_id)

    def __getitem__(self, index):
        # if self.mode=='test':
        #     return self.get_image_iterator(index)
        if self.mode!='train':
            all_raw = self.raw_caps[index]
            cur_cap = all_raw[0]
            cur_label = all_raw[1]
            cur_id = all_raw[2]
            source_sent = self.tokenize(cur_cap)
            sent_id = self.img_indexs[cur_id]
            # sent = self.tokenize(self.raw_caps[index])
            # sent = self.batch_sentences(sent)
            att_feat, box_feat, img_mask, img_id = self.get_img_feature(sent_id)

            att_feat = torch.tensor([att_feat]).float()
            img_mask = torch.tensor([img_mask]).long()
            box_feat = torch.tensor([box_feat]).float()

            img_feas = [source_sent,att_feat, img_mask, box_feat, img_id,cur_label]
        else:
            sample_numbers =  self.sample_n
            neg_cap_ids = random.sample(self.all_cap_neg_indices, sample_numbers - 1)
            # neg_captions = [self.raw_caps[neg_idx] for neg_idx in neg_cap_ids]

            pos_raw = self.raw_caps[index]
            cur_cap = pos_raw[0]
            cur_id = pos_raw[2]
            source_sent = self.tokenize(cur_cap)
            sent_id = self.pos_img_indexs[cur_id]
            att_feat, box_feat, img_mask, img_id = self.get_img_feature(sent_id)

            att_feats = []
            box_feats = []
            img_masks = []
            img_ids = []

            _labels = []

            sent = []

            sent.append(source_sent)
            att_feats.append(att_feat)
            box_feats.append(box_feat)
            img_masks.append(img_mask)
            img_ids.append(img_id)
            _labels.append(1)
            for neg_idx in neg_cap_ids:
               neg_raw = self.raw_neg_caps[neg_idx]
               neg_cap = neg_raw[0]
               neg_cap_id = neg_raw[2]
               neg_source_sent = self.tokenize(neg_cap)

               neg_sent_id = self.neg_img_indexs[neg_cap_id]
               neg_att_feat, neg_box_feat, neg_img_mask, neg_img_id = self.get_img_feature(neg_sent_id)

               sent.append(neg_source_sent)
               att_feats.append(neg_att_feat)
               box_feats.append(neg_box_feat)
               img_masks.append(neg_img_mask)
               img_ids.append(neg_img_id)
               _labels.append(0)


            att_feats = torch.tensor(att_feats).float()
            img_masks = torch.tensor(img_masks).long()
            box_feats = torch.tensor(box_feats).float()

            img_feas = [sent, att_feats, img_masks, box_feats, img_ids,_labels]

        return img_feas


