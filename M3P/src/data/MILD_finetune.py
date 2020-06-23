# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#
# Dataloaders for MILD dataset
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
from tqdm import tqdm
from .dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from torch.utils.data.dataset import Dataset
from .tokenization import XLMRTokenizer


logger = getLogger()

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10


class MILDCaptionDataset(Dataset):
    def __init__(self, captions, params, mode='train', data_type='mild'):
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
        self.tokenizer = XLMRTokenizer(params.vocab_path)

        self.params = params

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.en_len = 110000
        self.ft_lg = params.ft_lgs[0] #only use one
        assert data_type in ['mild']
        if data_type == 'mild':
            # open the hdf5 file for image features
            # print('DataLoader loading image feature file: ', params.input_fc_dir, params.input_att_dir, params.input_box_dir)
            lg = self.ft_lg
            if mode == 'train':
                train_file = os.path.join(params.input_fea_dir, params.mild_path, "train.%s.h5"%(lg))
                # self.(train_file)  # get feature by file pointer
                if lg=='en':
                    self.this_train_file_path = train_file
                else:
                    self.precess_reload(train_file)  # get feautre by memory
                    self.process_caption()
            if mode == 'valid':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "valid.%s.h5"%(lg))
                self.precess_reload(validfile)  # get feautre by memory
                self.process_caption()
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "test.%s.h5"%(lg))
                self.precess_reload(validfile)  # get feautre by memory
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
            cur_cap_list = list(self.captions[img_id])
            if len(cur_cap_list) > self.seq_per_img:
                np.random.shuffle(cur_cap_list)
                cur_cap_list = cur_cap_list[:self.seq_per_img]
            elif len(cur_cap_list) < self.seq_per_img:
                while len(cur_cap_list) < self.seq_per_img:
                    cur_cap = random.sample(cur_cap_list, 1)[0]
                    cur_cap_list.append(cur_cap)
            for cur_cap in cur_cap_list:
                if self.params.qp_type=='q':
                    _all_captions.append(cur_cap[0])
                else:
                    _all_captions.append(cur_cap[0]+' </s> ' +cur_cap[1])
        self.raw_caps = _all_captions
        assert len(self.raw_caps) == self.seq_per_img * len(self.image_ids)
        self.all_img_neg_indices = list(range(0, len(self.raw_caps) // self.seq_per_img))
        self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))
    def tokenize(self, sent,max_len=64):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:max_len]
        indexed = np.int32(indexed)
        return indexed
    def precess_reload(self, filename, is_old_pythia=False):

        cur_file = h5py.File(filename, "r")
        _image_ids = cur_file["image_id"][:]
        image_ids = [int(ss) for ss in _image_ids]
        wh = cur_file['wh'][:]
        num_boxes = cur_file['num_boxes'][:]
        boxes = cur_file['bbox'][:]
        obj_features = cur_file['feature'][:]
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

    def update_values(self, path_file):
        if self.mode == 'train' and self.image_ids is None:
            _image_ids = h5py.File(path_file, 'r')["image_id"]
            self.image_ids = [int(ss) for ss in _image_ids]

            self.wh = h5py.File(path_file, 'r')["wh"]
            self.num_boxes = h5py.File(path_file, 'r')["num_boxes"]
            self.boxes = h5py.File(path_file, 'r')["bbox"]
            self.obj_features = h5py.File(path_file, 'r')["feature"]

            self.objects = h5py.File(path_file, 'r')["objects"]
            self.distribution = None

        self.process_caption()

    def __len__(self):
        if self.ft_lg=='en' and self.mode=='train':
            return self.en_len*self.seq_per_img
        else: #not use
            return len(self.raw_caps) #google and sbu

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
        sent = self.tokenize(self.raw_caps[index])
        # sent = self.batch_sentences(sent)

        att_feat, box_feat, img_mask, img_id = self.get_img_feature(sent_id // self.seq_per_img)

        att_feat = torch.tensor([att_feat]).float()
        img_mask = torch.tensor([img_mask]).long()
        box_feat = torch.tensor([box_feat]).float()

        img_feas = [sent,att_feat, img_mask, box_feat, img_id]
        return img_feas

class MILDRetrievalDataset(Dataset):
    def __init__(self, caption_dict, params, mode='train', data_type='mild'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        #support multi language age
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.captions= caption_dict
        self.ft_lgs= params.ft_lgs
        self.n_langs = params.n_langs
        self.max_vocab = params.max_vocab
        self.min_count = params.min_count

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
        self.lang2id = params.lang2id

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.sample_n = params.sample_n

        self.t2i_flag = params.t2i_flag
        self.i2t_flag = params.i2t_flag


        self.en_len = 110000
        self.ft_lg = params.ft_lgs[0]  # only use one
        _local_rank = params.local_rank
        _select_lg = _local_rank % len(params.ft_lgs)
        self.ft_lg = params.ft_lgs[_select_lg]
        #whether fintune on all
        self.is_ft_all= False
        #if len(params.ft_lgs)>1:
        #    self.is_ft_all = True
        assert data_type in ['mild']
        if data_type == 'mild':
            # open the hdf5 file for image features
            # print('DataLoader loading image feature file: ', params.input_fc_dir, params.input_att_dir, params.input_box_dir)
            lg = self.ft_lg
            if mode == 'train':
                if self.is_ft_all:
                    train_file = os.path.join(params.input_fea_dir, params.mild_path, "train.all.h5" )
                else:
                    train_file = os.path.join(params.input_fea_dir, params.mild_path, "train.%s.h5" % (lg))
                # self.(train_file)  # get feature by file pointer
                if lg == 'en':
                    self.this_train_file_path = train_file
                else:
                    self.precess_reload(train_file)  # get feautre by memory
                    self.update_captions()
            if mode == 'valid':
                if self.is_ft_all:
                    validfile = os.path.join(params.input_fea_dir, params.mild_path, "valid.all.h5")
                else:
                    validfile = os.path.join(params.input_fea_dir, params.mild_path, "valid.%s.h5" % (lg))
                self.precess_reload(validfile)  # get feautre by memory
                self.update_captions()
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "test.%s.h5" % (lg))
                self.precess_reload(validfile)  # get feautre by memory
                self.update_captions()

        self.data_type=data_type
        # after assign image ids

    def update_captions(self):
        _all_captions = []
        _all_langs = []
        seq_per_img = self.seq_per_img
        # _all_pag = {}
        for img_id in self.image_ids:
            img_id = int(img_id)
            cur_cap_list = []
            if len(self.ft_lgs)>0:
                for lg in [self.ft_lg]:
                    if img_id not in self.captions[lg]:
                        continue
                    for cap in self.captions[lg][img_id]:
                        cur_cap_list.append((cap,lg))

            if len(cur_cap_list) > seq_per_img:
                np.random.shuffle(cur_cap_list)
                cur_cap_list = cur_cap_list[:seq_per_img]
            elif len(cur_cap_list) < seq_per_img:
                while len(cur_cap_list) < seq_per_img:
                    cur_cap = random.sample(cur_cap_list, 1)[0]
                    cur_cap_list.append(cur_cap)
            for _cur_cap in cur_cap_list:
                #if self.params.qp_type == 'q':
                cur_cap = _cur_cap[0][0] #only q

                _all_captions.append(cur_cap)
                _all_langs.append(_cur_cap[1])
        self.raw_caps = _all_captions
        self.raw_langs = _all_langs
        # self._all_pag = _all_pag
        assert len(self.raw_caps) == seq_per_img * len(self.image_ids)
        self.all_img_neg_indices = list(range(0, len(self.raw_caps) // seq_per_img))
        self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))

    def tokenize(self, sent,max_len=64):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:max_len]
        indexed = np.int32(indexed)
        return indexed

    def precess_reload(self, filename, is_old_pythia=False,is_zh=False):

        cur_file = h5py.File(filename, "r")
        _image_ids = cur_file['image_id'][:]
        image_ids = [int(ss) for ss in _image_ids]

        wh = cur_file['wh'][:]
        num_boxes = cur_file['num_boxes'][:]
        boxes = cur_file['bbox'][:]
        obj_features = cur_file['feature'][:]
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
        if self.mode == 'train' and self.image_ids is None:
            self.image_ids = h5py.File(path_file, 'r')["image_id"]
            self.wh = h5py.File(path_file, 'r')["wh"]
            self.num_boxes = h5py.File(path_file, 'r')["num_boxes"]
            self.boxes = h5py.File(path_file, 'r')["bbox"]
            self.obj_features = h5py.File(path_file, 'r')["feature"]
            cur_pointer = h5py.File(path_file, 'r')
            if 'objects' not in cur_pointer:
                self.objects = None
                self.distribution = h5py.File(path_file, 'r')["distribution"]
            else:
                self.objects = h5py.File(path_file, 'r')["objects"]
                self.distribution = None

        self.update_captions()

    def __len__(self):
        if self.ft_lg=='en' and self.mode=='train':
            return self.en_len*self.seq_per_img
        else: #not use
            return len(self.raw_caps) #google and sbu

    def norm_boxes(self, cur_boxes, h, w):
        # devided by image width and height
        x1, y1, x2, y2 = np.hsplit(cur_boxes, 4)
        # for location
        cur_boxes = np.hstack(
            (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
        cur_boxes = cur_boxes / np.linalg.norm(cur_boxes, 2, 1, keepdims=True)
        return cur_boxes

    def get_img_feature(self, index,is_origin=False):

        object_features, box, num_boxes, wh = self.obj_features[index][:self.max_region_num], \
                                              self.boxes[index][:self.max_region_num], \
                                              self.num_boxes[index], \
                                              self.wh[index]


        img_id = self.image_ids[index]

        num_boxes = self.max_region_num

        if self.objects is None:
            distribution = self.distribution[index][:self.max_region_num]
            objects = distribution.argmax(-1)
        else:
            objects = self.objects[index][:self.max_region_num]

        # normalized
        att_feat = object_features.astype('float32')
        # att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        # att_feat = torch.FloatTensor(att_feat)
        # att_feat = F.normalize(att_feat, dim=-1).numpy()  # keep precision
        if is_origin == False:
            att_feat = torch.FloatTensor(att_feat)
            att_feat = F.normalize(att_feat, dim=-1).numpy()

        h, w = wh.astype('float32')
        loc_features = self.norm_boxes(box.astype('float32'), h, w)

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        return (att_feat, loc_features, image_mask,objects,img_id)

    def sample_images(self, index):
        sample_numbers = self.sample_n
        sent = []
        att_feats = []
        box_feats = []
        img_masks = []
        img_ids = []
        obj_labels = []
        pos_labels = []
        langs = []

        sent_id = index

        all_neg_indices = self.all_img_neg_indices  # sent_length//5
        neg_img_ids = random.sample(all_neg_indices, sample_numbers - 1)
        sample_indices = neg_img_ids
        pos_label = random.randint(0, sample_numbers - 1)

        sample_indices.insert(pos_label, sent_id // self.seq_per_img)

        cur_lg = self.ft_lg
        for img_index in sample_indices:
            # assign same sentence with each image
            #concat caption and passage

            att_feat, box_feat, img_mask, obj_label,img_id = self.get_img_feature(img_index)

            if self.params.qp_type != 'q':
                _cur_caps = self.raw_caps[sent_id]+' </s> '+ self.captions[cur_lg][img_id][0][1]
                sent.append(self.tokenize(_cur_caps))
            else:
                sent.append(self.tokenize(self.raw_caps[sent_id]))

            langs.append(self.lang2id[self.raw_langs[sent_id]])


            att_feats.append(att_feat)
            box_feats.append(box_feat)
            img_masks.append(img_mask)
            img_ids.append(img_id)
            obj_labels.append(obj_label)  # [B,neg_samples,x]

        pos_labels.append(pos_label)  # [B,x]

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()
        obj_labels = torch.tensor(obj_labels).long()

        img_feas = (sent,att_feats, img_masks, box_feats,obj_labels, pos_labels, img_ids,langs)

        return img_feas

    def sample_captions(self,index):
        sample_numbers = self.sample_n
        sent_id = index

        att_feats = []
        box_feats = []
        img_masks = []
        img_ids = []
        obj_labels = []
        pos_labels = []

        sent = []
        langs=[]

        this_caption = self.raw_caps[sent_id]
        this_lang = self.lang2id[self.raw_langs[sent_id]]
        all_neg_indices = self.all_cap_neg_indices
        neg_cap_ids = random.sample(all_neg_indices, sample_numbers - 1)
        neg_captions = [self.raw_caps[neg_idx] for neg_idx in neg_cap_ids]
        pos_label = random.randint(0, sample_numbers - 1)
        sample_captions = neg_captions
        sample_captions.insert(pos_label, this_caption)

        img_index = sent_id//self.seq_per_img
        cur_lg = self.ft_lg
        for cur_caption in sample_captions:  # sample captions
            att_feat, box_feat, img_mask, obj_label,img_id = self.get_img_feature(img_index)

            if self.params.qp_type != 'q':
                _cur_caps = cur_caption + ' </s> ' + self.captions[cur_lg][img_id][0][1]
                sent.append(self.tokenize(_cur_caps))
            else:
                sent.append(self.tokenize(cur_caption))
            langs.append(this_lang)

            att_feats.append(att_feat)
            box_feats.append(box_feat)
            img_masks.append(img_mask)
            img_ids.append(img_id)
            obj_labels.append(obj_label)

        pos_labels.append(pos_label)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()
        obj_labels = torch.tensor(obj_labels).long()

        img_feas = (sent,att_feats, img_masks, box_feats, obj_labels,pos_labels, img_ids,langs)

        return img_feas

    def __getitem__(self, index):
        if self.image_ids is None:
            self.update_values(self.this_train_file_path)
            self.update_captions()

        two_types_input = [None, None]
        if self.t2i_flag:
            two_types_input[0] = self.sample_images(index)
        if self.i2t_flag:
            two_types_input[1] = self.sample_captions(index)

        return two_types_input

class MILDEvaluateCaptionDataset(Dataset):
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

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.ft_lg = params.ft_lgs[0]  # only use one
        assert data_type in ['mild']
        if data_type == 'mild':
            # open the hdf5 file for image features
            # print('DataLoader loading image feature file: ', params.input_fc_dir, params.input_att_dir, params.input_box_dir)
            lg = self.ft_lg
            if mode == 'valid':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "valid.%s.h5" % (lg))
                self.precess_reload(validfile)  # get feautre by memory
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "test.%s.h5" % (lg))
                self.precess_reload(validfile)  # get feautre by memory

        # after assign image ids
        _all_captions = []

        for img_id in self.image_ids:
            cur_cap_list = list(self.captions[img_id])
            if len(cur_cap_list) > self.seq_per_img:
                np.random.shuffle(cur_cap_list)
                cur_cap_list = cur_cap_list[: self.seq_per_img]
            elif len(cur_cap_list) < self.seq_per_img:
                while len(cur_cap_list) <  self.seq_per_img:
                    cur_cap = random.sample(cur_cap_list, 1)[0]
                    cur_cap_list.append(cur_cap)
            for cur_cap in cur_cap_list:
                _all_captions.append(cur_cap)

        self.raw_caps = _all_captions

        assert len(self.raw_caps) ==  self.seq_per_img*len(self.image_ids)
        # self.all_img_neg_indices = list(range(0, len(self.raw_caps) // 5))
        # self.all_cap_neg_indices = list(range(0, len(self.raw_caps)))

    def precess_reload(self, filename, is_old_pythia=False):

        cur_file = h5py.File(filename, "r")
        _image_ids = cur_file['image_id'][:]
        image_ids = [int(ss) for ss in _image_ids]
        wh = cur_file['wh'][:]
        num_boxes = cur_file['num_boxes'][:]
        boxes = cur_file['bbox'][:]
        obj_features = cur_file['feature'][:]
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

        att_feat, box_feat, img_mask, img_id = self.get_img_feature(index)

        att_feats.append(att_feat)
        box_feats.append(box_feat)
        img_masks.append(img_mask)
        img_ids.append(img_id)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()

        img_feas = [att_feats, img_masks, box_feats, img_id]
        return img_feas

class MILDEvaluateRetrievalDataset(Dataset):
    def __init__(self, caption_dict, params, mode='train', data_type='mild',lang='en'):
        # the dataset's order is depended on image features' order
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size

        self.tokenizer = XLMRTokenizer(params.vocab_path)
        self.max_vocab = params.max_vocab
        self.min_count = params.min_count

        # only save
        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        self.mode = mode
        self.lang2id = params.lang2id
        self.n_langs = params.n_langs

        self.seq_per_img = params.seq_per_img

        self.captions = caption_dict[lang]
        self.lang = lang

        if self.lang=='zh':
            self.eval_images =1000
        else:
            self.eval_images= 5000

        # feature related for image features
          # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num
        self.data_type = data_type

        # sanity checks
        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.qp_type = params.qp_type

        all_train_files = []
        assert data_type in ['mild']
        if data_type == 'mild':
            # open the hdf5 file for image features
            if mode == 'test':
                validfile = os.path.join(params.input_fea_dir, params.mild_path, "test.%s.h5" % (lang))
                self.precess_reload(validfile)  # get feautre by memory


    def precess_reload(self, filename):

        cur_file = h5py.File(filename, "r")
        _image_ids = cur_file['image_id'][:]
        image_ids = [int(ss) for ss in _image_ids]
        wh = cur_file['wh'][:]
        num_boxes = cur_file['num_boxes'][:]
        boxes = cur_file['bbox'][:]
        obj_features = cur_file['feature'][:]
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

        seq_per_img = self.seq_per_img

        # process features
        # if self.eval_images==-1:

        self.wh = self.wh[:self.eval_images]
        self.boxes = self.boxes[:self.eval_images][:, :self.max_region_num, :]
        self.obj_features = self.obj_features[:self.eval_images][:, :self.max_region_num, :]
        self.num_boxes = self.num_boxes[:self.eval_images]
        self.image_ids = self.image_ids[:self.eval_images]

        # each_gpu_len = len(self.image_ids) // self.n_gpu_per_node
        # self.image_ids = self.image_ids[
        #                  self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]  # trunct by gpu

        # process all captions
        _all_captions = []
        raw_caps = []

        _all_pag = {}
        cur_lg  = self.lang


        if self.qp_type=='q':

            for img_id in self.image_ids: #each img_id contains different total capitons list
                #get page
                cur_cap_list = list(self.captions[img_id])
                if len(cur_cap_list) > seq_per_img:
                    np.random.shuffle(cur_cap_list)
                    cur_cap_list = cur_cap_list[:seq_per_img]
                elif len(cur_cap_list) < seq_per_img:
                    while len(cur_cap_list) < seq_per_img:
                        cur_cap = random.sample(cur_cap_list, 1)[0]
                        cur_cap_list.append(cur_cap)

                for _cur_cap in cur_cap_list:
                    cur_cap = _cur_cap[0]
                        # cur_cap = _cur_cap[0] +  ' </s> ' + _all_pag[int(img_id)]
                    raw_caps.append(cur_cap)
                    sent_ids = self.tokenize(cur_cap)
                    _all_captions.append(sent_ids)

            sent, lengths = self.batch_sentences(_all_captions)
            self.all_caps = sent
            self.raw_caps = raw_caps
            self.all_caps_length = lengths
            self.all_segment_ids = torch.LongTensor(
                [[self.n_langs] * self.max_region_num + [self.lang2id[self.lang]] * lengths.max().item()] *
                sent.size()[1]) if self.n_langs > 1 else None
            assert len(self.all_caps_length) // seq_per_img == len(self.image_ids)
        else:
            #process tokenizer

            #all query
            _query_dict = {}
            for img_id in self.image_ids:  # each img_id contains different total capitons list
                # get page
                cur_cap_list = list(self.captions[img_id])
                if len(cur_cap_list) > seq_per_img:
                    np.random.shuffle(cur_cap_list)
                    cur_cap_list = cur_cap_list[:seq_per_img]
                elif len(cur_cap_list) < seq_per_img:
                    while len(cur_cap_list) < seq_per_img:
                        cur_cap = random.sample(cur_cap_list, 1)[0]
                        cur_cap_list.append(cur_cap)
                for _cur_cap in cur_cap_list:
                    cur_cap = _cur_cap[0] + ' </s> '

                    _sent_cap = self.tokenize(cur_cap)
                    if img_id not in _query_dict:
                        _query_dict[img_id] = [_sent_cap]
                    else:
                        _query_dict[img_id].append(_sent_cap)
            _pag_dict = {}
            for img_id in self.image_ids:
                cur_cap_list = list(self.captions[img_id])
                cur_page = cur_cap_list[0][1]
                _sent_cap = self.tokenize(cur_page)
                if img_id not in _pag_dict:
                    _pag_dict[img_id] = _sent_cap


            for a_img_id in self.image_ids:
                for img_id in self.image_ids:  # each img_id contains different total capitons list
                    for _cur_q in _query_dict[img_id]:
                        _tokens = list(_cur_q)+list(_pag_dict[a_img_id])
                        _tokens =  np.int32(_tokens)
                        raw_caps.append(_tokens)
                        _all_captions.append(_tokens) #5000*5000

            self.all_caps = _all_captions
            self.raw_caps = raw_caps

        # normalized
        img_box_coordinates = []
        for idx in range(len(self.wh)):
            wh = self.wh[idx]
            box = self.boxes[idx]
            h, w = wh.astype('float32')
            img_box_coordinates.append(torch.Tensor(self.norm_boxes(box.astype('float32'), h, w)))

        # normalized
        self.obj_features = self.obj_features.astype('float32')
        self.all_test_obj_cache = F.normalize(torch.Tensor(self.obj_features), dim=-1)
        self.all_test_box_cache = torch.stack(img_box_coordinates, 0)

        if self.n_gpu_per_node>1:
            each_gpu_len = len(self.image_ids) // self.n_gpu_per_node
            self.raw_caps = self.raw_caps[self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]
            # self.all_test_obj_cache = self.all_test_obj_cache[self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]
            # self.all_test_box_cache = self.all_test_box_cache[
            #                           self.local_rank * each_gpu_len:(self.local_rank + 1) * each_gpu_len]

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = 0
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def tokenize(self, sent):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = np.int32(indexed)
        return indexed

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

    def __getitem__(self, index):
        each_gpu_len = len(self.image_ids) // self.n_gpu_per_node
        index = index+self.local_rank * each_gpu_len

        test_obj_feats = self.all_test_obj_cache[index]
        test_box_coords = self.all_test_box_cache[index]

        if self.qp_type=='q':
            pos_cap_label = torch.zeros(len(self.all_caps_length))  # 5000
            pos_cap_label[index * self.seq_per_img:index * self.seq_per_img + self.seq_per_img] = 1
            return [self.all_caps.transpose(0, 1),
                    self.all_caps_length,
                    self.all_segment_ids,
                    test_obj_feats.unsqueeze(0),
                    test_box_coords.unsqueeze(0),
                    # test_obj_feats.unsqueeze(0).repeat(self.split_len, 1, 1),
                    # test_box_coords.unsqueeze(0).repeat(self.split_len, 1, 1),
                    pos_cap_label]
        else:
            _each_len = len(self.image_ids)*self.seq_per_img
            cur_caps = self.all_caps[index*_each_len:index*_each_len+_each_len]

            sent, lengths = self.batch_sentences(cur_caps)
            cur_seg_ids = torch.LongTensor(
                [[self.n_langs] * self.max_region_num + [self.lang2id[self.lang]] * lengths.max().item()] *
                sent.size()[1]) if self.n_langs > 1 else None
            assert len(lengths) // self.seq_per_img == len(self.image_ids)

            pos_cap_label = torch.zeros(len(lengths))  # 5000
            pos_cap_label[index * self.seq_per_img:index * self.seq_per_img + self.seq_per_img] = 1

            return [sent.transpose(0, 1),
                    lengths,
                    cur_seg_ids,
                    test_obj_feats.unsqueeze(0),
                    test_box_coords.unsqueeze(0),
                    # test_obj_feats.unsqueeze(0).repeat(self.split_len, 1, 1),
                    # test_box_coords.unsqueeze(0).repeat(self.split_len, 1, 1),
                    pos_cap_label]
