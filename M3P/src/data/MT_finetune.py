# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
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


class MTCaptionDataset(Dataset):
    def __init__(self, captions, params, mode='train', data_type='coco'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        src_lg = params.ft_lgs[0]
        tgt_lg = params.ft_lgs[1]
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

        self.coco_len = 113287
        assert data_type in ['coco','flicker']
        if data_type == 'coco':
            # open the hdf5 file for image features
            # print('DataLoader loading image feature file: ', params.input_fc_dir, params.input_att_dir, params.input_box_dir)
            if mode == 'train':
                train_file = os.path.join(params.input_fea_dir, params.coco_path, "coco_train_no_dist.h5")
                # self.(train_file)  # get feature by file pointer
                self.this_train_file_path = train_file
            if mode == 'valid':
                validfile = os.path.join(params.input_fea_dir, params.coco_path, "coco_val.h5")
                self.precess_reload(validfile)  # get feautre by memory
                self.process_caption()
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.coco_path, "coco_test5k.h5")
                self.precess_reload(validfile)  # get feautre by memory
                self.process_caption()
        elif data_type == 'flicker':
            if params.use_new_fea == False:
                all_train_files = [None] * 3
                if mode == 'train':
                    train_file = os.path.join(params.input_fea_dir, params.flicker_path, "train.h5")  # FLICKR30
                    self.precess_reload(train_file, True)
                if mode == 'valid':
                    validfile = os.path.join(params.input_fea_dir, params.flicker_path, "dev.h5")
                    self.precess_reload(validfile, True)
                if mode == 'test':
                    testfile = os.path.join(params.input_fea_dir, params.flicker_path, "test.h5")
                    self.precess_reload(testfile, True)
            else:
                if mode == 'train':
                    train_file = os.path.join(params.input_fea_dir, params.flicker_path, "train.h5")
                    self.precess_reload(train_file)
                if mode == 'valid':
                    validfile = os.path.join(params.input_fea_dir, params.flicker_path, "val.h5")
                    self.precess_reload(validfile)
                if mode == 'test':
                    testfile = os.path.join(params.input_fea_dir, params.flicker_path, "test.h5")
                    self.precess_reload(testfile)
            self.process_caption()

        # after assign image ids
        self.data_type = data_type
        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))
    def update_captions(self):
        #no need update
        logger.info('epoch ended')

    def process_caption(self):
        _all_captions = []
        for img_id in self.image_ids:
            src_cap = self.captions[img_id][0]
            tgt_cap = self.captions[img_id][1]
            _all_captions.append((src_cap,tgt_cap))
        self.raw_caps = _all_captions
        assert len(self.raw_caps) == len(self.image_ids)
        self.all_img_neg_indices = list(range(0, len(self.raw_caps)))
        self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))

    def tokenize(self, sent):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = np.int32(indexed)
        return indexed

    def precess_reload(self, filename, is_old_pythia=False):
        if is_old_pythia:
            cur_file = h5py.File(filename, "r")
            _image_ids = cur_file['image_id'][:]
            if self.data_type == 'coco':
                image_ids = [str(ss, encoding="utf8") + '.jpg' for ss in _image_ids]
            else:
                image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
            wh = cur_file['wh'][:]
            if 'num_boxes' not in cur_file:
                num_boxes = np.ones(len(_image_ids), dtype='int64') * 100
            else:
                num_boxes = cur_file['num_boxes'][:]
            boxes = cur_file['boxes'][:]
            obj_features = cur_file['features'][:]
            if 'object' not in cur_file:
                distribution = cur_file['distribution'][:]
                objects = None
            else:
                objects = cur_file['object'][:]
                distribution = None
        else:
            cur_file = h5py.File(filename, "r")
            _image_ids = cur_file['image_id'][:]
            image_ids = [ss for ss in _image_ids]
            wh = cur_file['wh'][:]
            num_boxes = cur_file['num_boxes'][:]
            boxes = cur_file['bbox'][:]
            obj_features = cur_file['features'][:]
            if 'objects' not in cur_file:
                distribution = cur_file['distribution'][:]
                objects = None
            else:
                objects = cur_file['objects'][:]
                distribution = None

        self.image_ids = image_ids
        self.wh = wh
        self.num_boxes = num_boxes
        self.boxes = boxes
        self.obj_features = obj_features
        self.distribution = distribution
        self.objects = objects

    def update_values(self, path_file, is_old_pythia=False):
        if self.mode == 'train' and self.image_ids is None and is_old_pythia == True:
            _image_ids = h5py.File(path_file, 'r')["image_id"]
            if self.data_type == 'coco':
                self.image_ids = [str(ss, encoding="utf8") + '.jpg' for ss in _image_ids]
            else:
                self.image_ids = [str(ss, encoding="utf8") for ss in _image_ids]

            self.wh = h5py.File(path_file, 'r')["wh"]
            self.num_boxes = h5py.File(path_file, 'r')["num_boxes"]
            self.boxes = h5py.File(path_file, 'r')["boxes"]
            self.obj_features = h5py.File(path_file, 'r')["features"]
            cur_pointer = h5py.File(path_file, 'r')
            if 'object' not in cur_pointer:
                self.objects = None
                self.distribution = h5py.File(path_file, 'r')["distribution"]
            else:
                self.objects = h5py.File(path_file, 'r')["object"]
                self.distribution = None
        elif self.mode == 'train' and self.image_ids is None:
            self.image_ids = h5py.File(path_file, 'r')["image_id"]
            self.wh = h5py.File(path_file, 'r')["wh"]
            self.num_boxes = h5py.File(path_file, 'r')["num_boxes"]
            self.boxes = h5py.File(path_file, 'r')["bbox"]
            self.obj_features = h5py.File(path_file, 'r')["features"]
            cur_pointer = h5py.File(path_file, 'r')
            if 'objects' not in cur_pointer:
                self.objects = None
                self.distribution = h5py.File(path_file, 'r')["distribution"]
            else:
                self.objects = h5py.File(path_file, 'r')["objects"]
                self.distribution = None
        self.process_caption()

    def __len__(self):
        if self.data_type=='flicker':
            return len(self.raw_caps)
        if self.data_type=='coco' :
            return self.coco_len*5 if self.mode=='train' else len(self.raw_caps)
        else: #not use
            return 100000 #google and sbu

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
        if self.image_ids is None: #for coco
            self.update_values(self.this_train_file_path)

        sent_id = index
        cur_cap = self.raw_caps[index]
        source_sent = self.tokenize(cur_cap[0])
        target_sent = self.tokenize(cur_cap[1])
        # sent = self.tokenize(self.raw_caps[index])
        # sent = self.batch_sentences(sent)
        att_feat, box_feat, img_mask, img_id = self.get_img_feature(sent_id)

        att_feat = torch.tensor([att_feat]).float()
        img_mask = torch.tensor([img_mask]).long()
        box_feat = torch.tensor([box_feat]).float()

        img_feas = [source_sent,target_sent,att_feat, img_mask, box_feat, img_id]
        return img_feas

class EvaluateMTCaptionDataset(Dataset):
    def __init__(self, captions, params, mode='train', data_type='coco'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.captions = captions

        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        self.mode = mode
        self.params = params

        self.tokenizer = XLMRTokenizer(params.vocab_path)
        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank


        assert data_type in ['coco','flicker']
        if data_type == 'coco':
            # open the hdf5 file for image features
            # print('DataLoader loading image feature file: ', params.input_fc_dir, params.input_att_dir, params.input_box_dir)
            if mode == 'train':
                train_file = os.path.join(params.input_fea_dir, params.coco_path, "coco_train_no_dist.h5")
                self.update_values(train_file)  # get feature by file pointer
            if mode == 'valid':
                validfile = os.path.join(params.input_fea_dir, params.coco_path, "coco_val.h5")
                self.precess_reload(validfile)  # get feautre by memory
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.coco_path, "coco_test5k.h5")
                self.precess_reload(validfile)  # get feautre by memory
        elif data_type == 'flicker':
                if params.use_new_fea == False:
                    all_train_files = [None] * 3
                    if mode == 'train':
                        train_file = os.path.join(params.input_fea_dir, params.flicker_path, "train.h5")  # FLICKR30
                        self.precess_reload(train_file, True)
                    if mode == 'valid':
                        validfile = os.path.join(params.input_fea_dir, params.flicker_path, "dev.h5")
                        self.precess_reload(validfile, True)
                    if mode == 'test':
                        testfile = os.path.join(params.input_fea_dir, params.flicker_path, "test.h5")
                        self.precess_reload(testfile, True)
                else:
                    if mode == 'train':
                        train_file = os.path.join(params.input_fea_dir, params.flicker_path, "train.h5")
                        self.precess_reload(train_file)
                    if mode == 'valid':
                        validfile = os.path.join(params.input_fea_dir, params.flicker_path, "val.h5")
                        self.precess_reload(validfile)
                    if mode == 'test':
                        testfile = os.path.join(params.input_fea_dir, params.flicker_path, "test.h5")
                        self.precess_reload(testfile)

        # after assign image ids
        _all_captions = []

        for img_id in self.image_ids:
            src_cap = self.captions[img_id][0]
            tgt_cap = self.captions[img_id][1]
            _all_captions.append(src_cap)

        self.raw_caps = _all_captions

        assert len(self.raw_caps) == len(self.image_ids) # we only need one source
        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))

    def precess_reload(self, filename, is_old_pythia=False):
        if is_old_pythia:
            cur_file = h5py.File(filename, "r")
            _image_ids = cur_file['image_id'][:]
            if self.data_type == 'coco':
                image_ids = [str(ss, encoding="utf8") + '.jpg' for ss in _image_ids]
            else:
                image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
            wh = cur_file['wh'][:]
            if 'num_boxes' not in cur_file:
                num_boxes = np.ones(len(_image_ids), dtype='int64') * 100
            else:
                num_boxes = cur_file['num_boxes'][:]
            boxes = cur_file['boxes'][:]
            obj_features = cur_file['features'][:]
            if 'object' not in cur_file:
                distribution = cur_file['distribution'][:]
                objects = None
            else:
                objects = cur_file['object'][:]
                distribution = None
        else:
            cur_file = h5py.File(filename, "r")
            _image_ids = cur_file['image_id'][:]
            image_ids = [ss for ss in _image_ids]
            wh = cur_file['wh'][:]
            num_boxes = cur_file['num_boxes'][:]
            boxes = cur_file['bbox'][:]
            obj_features = cur_file['features'][:]
            if 'objects' not in cur_file:
                distribution = cur_file['distribution'][:]
                objects = None
            else:
                objects = cur_file['objects'][:]
                distribution = None

        self.image_ids = image_ids
        self.wh = wh
        self.num_boxes = num_boxes
        self.boxes = boxes
        self.obj_features = obj_features
        self.distribution = distribution
        self.objects = objects

    def __len__(self):
        return len(self.image_ids)

    def tokenize(self, sent):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = np.int32(indexed)
        return indexed

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
        att_feats = []
        box_feats = []
        img_masks = []
        img_ids = []

        sent_id = index
        source_sent = self.tokenize(self.raw_caps[index])
        #target_sent = self.tokenize(self.raw_caps[index][1])

        att_feat, box_feat, img_mask, img_id = self.get_img_feature(sent_id)

        att_feats.append(att_feat)
        box_feats.append(box_feat)
        img_masks.append(img_mask)
        img_ids.append(img_id)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()

        img_feas = [source_sent,att_feats, img_masks, box_feats, img_id]
        return img_feas

