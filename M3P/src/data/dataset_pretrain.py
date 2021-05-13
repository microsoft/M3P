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
from .tokenization import XLMRTokenizer
from torch.utils.data.dataset import Dataset

logger = getLogger()

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10


class VLMPretrainRetrievalDataset(Dataset):
    def __init__(self, captions, clager, params, mode='train', data_type='google'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.mask_index = params.mask_index
        self.n_words = params.n_words

        self.mlm_prob = params.word_pred

        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.captions = captions
        self.clager = clager
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

        self.params = params
        self.max_len = params.max_len

        self.tokenizer = XLMRTokenizer(params.vocab_path)

        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.cc_num = 29
        self.sbu_num = 8
        self.sample_n = params.sample_n

        self.coco_len = 113287

        self.t2i_flag = params.t2i_flag
        self.i2t_flag = params.i2t_flag

        self.is_pretrain = params.is_pretrain
        if mode=='valid':
            valid_file = os.path.join(params.google_valid_path, "google_valid_fp16.h5")
            self.precess_reload(valid_file, True)
            self.val_len = len(self.image_ids)
        else:
            if data_type=='google':
                with open(os.path.join(params.train_order_path, "google_train_order.json"), 'r') as f:
                    self.train_order = json.load(f)
                    self.train_order = self.train_order[80:]#sep with cap
            else:
                with open(os.path.join(params.train_order_path, "sbu_train_order.json"), 'r') as f:
                    self.train_order = json.load(f)

        if data_type == 'google':
            all_train_files = []
            if mode == 'train':
                for google_dataset_idx in range(self.cc_num):
                    train_file = os.path.join(params.input_fea_dir, params.google_path,
                                              "train_" + str(google_dataset_idx) + ".h5")
                    # cur_file = h5py.File(train_file, "r", swmr=True)
                    all_train_files.append(train_file)  # reload lately
                self.all_train_files = all_train_files
                self.update(0) #select
        elif data_type == 'sbu':
            # feaFile = '/hdfs/public/nanduan/data/google_captions/obj100'
            all_train_files = []
            if mode == 'train':
                for _idx in range(self.cc_num, self.cc_num + self.sbu_num):
                    train_file = os.path.join(params.input_fea_dir, params.sbu_path,
                                              "train_" + str(_idx) + ".h5")

                    all_train_files.append(train_file)  # reload lately
                self.all_train_files = all_train_files
                self.update(0)

        self.data_type=data_type
        # after assign image ids
        self.split_len = 100000

    def tokenize(self, sent, half=False):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:(self.max_len if not half else self.max_len//2)]
        indexed = np.int32(indexed)
        return torch.tensor(indexed).long()

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
            image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
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

    def update_values(self, is_old_pythia=False):
        path_file = self.this_train_file_path
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
            _image_ids = h5py.File(path_file, 'r')["image_id"]
            self.image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
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

    def update(self, epoch=0):
        local_train_files = self.train_order[epoch][:self.n_gpu_per_node]
        _file_num = local_train_files[self.local_rank]
        if self.params.debug_pretrain:
            _file_num=0
        self.this_train_file_path = self.all_train_files[_file_num]

        logger.info('select train file: ' + self.this_train_file_path)
        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        # self.update_values(self.this_train_file_path)

    def __len__(self):
        return self.val_len if self.mode == "valid" else self.split_len

    def mask_tokens(self, inputs, special_token_mask=None, unmasked=None, mlm_probability=0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        if special_token_mask is not None:
            probability_matrix.masked_fill_(torch.BoolTensor(special_token_mask), value=0.0)
        if unmasked is not None:
            probability_matrix.masked_fill_(torch.BoolTensor(unmasked), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if (masked_indices == False).sum().item() == len(masked_indices): # torch loss not support empty mask
            masked_indices[0] = True
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_index#tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.n_words, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels.numpy().tolist()

    def random_mask_object(self, object_features, object_labels):
        # we need to mask it when training
        masked_object_features = []
        masked_object_labels = []
        # lm_label_ids = []  # share vocabulary with word does not work
        _n_mask = 0
        for i, class_label in enumerate(object_labels):
            prob = random.random()

            if prob < 0.15 and class_label != 0:
                prob /= 0.15
                if prob < 0.9:
                    masked_object_features.append(np.zeros((2048), dtype=np.float32))
                else:
                    masked_object_features.append(object_features[i])

                masked_object_labels.append(int(class_label))
                _n_mask+=1
            else:
                masked_object_features.append(object_features[i])

                masked_object_labels.append(-1)

        if _n_mask==0:
            masked_object_labels[-1] = class_label
            masked_object_features[-1] = object_features[i]
        masked_object_features = np.stack(masked_object_features, 0)  # [BS,dim]

        masked_object_features = torch.FloatTensor(np.stack(masked_object_features, 0))
        att_feat = F.normalize(masked_object_features, dim=-1)
        # # masked_object_features = torch.FloatTensor(np.stack(masked_object_features, 0))
        # # masked_object_features = F.normalize(masked_object_features, dim=-1)
        # att_feat = masked_object_features.astype('float32')
        # att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
        return att_feat.numpy(), masked_object_labels

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

        if is_origin == False:
            att_feat = torch.FloatTensor(att_feat)
            att_feat = F.normalize(att_feat, dim=-1).numpy()

        h, w = wh.astype('float32')
        loc_features = self.norm_boxes(box.astype('float32'), h, w)

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        return (att_feat, loc_features, image_mask, objects,img_id)

    def sample_images(self, index):

        sample_numbers = self.sample_n
        sent = []
        att_feats = []
        box_feats = []
        img_masks = []
        img_ids = []
        obj_labels = []
        lm_labels = []
        masked_types = []
        ori_feats = []

        neg_sample_indices = list(range(self.__len__()))
        neg_sample_indices.pop(index)
        sample_indices = random.sample(neg_sample_indices, sample_numbers - 1)
        itm_label = random.randint(0, sample_numbers - 1)
        sample_indices.insert(itm_label, index)

        cur_image_id = self.image_ids[index]  # enough memory
        if self.data_type == 'google':
            cap_id = int(re.sub("\D", "", cur_image_id))  # google
            cur_cap = self.captions[cap_id]
            if self.clager is None:
                cur_cap = self.captions[cap_id]
            else:
                cur_cap = self.clager.dclag(self.captions[cap_id], 'en', 1, 0)[0]
        else:
            cap_id = int(cur_image_id.split('_')[0])  # sbu
            cur_cap = self.captions[cap_id]
            if self.clager is None:
                cur_cap = self.captions[cap_id]
            else:
                cur_cap = self.clager.dclag(self.captions[cap_id], 'en', 1, 0)[0]

        cur_input_ids = self.tokenize(cur_cap)
        for img_index in sample_indices:
            att_feat, box_feat, img_mask, obj_label, img_id = self.get_img_feature(img_index,True)
            input_ids = cur_input_ids.clone()
            if random.random() > 0.5:  # mask word
                masked = "word"
                input_ids, lm_label_ids = self.mask_tokens(input_ids,mlm_probability=self.mlm_prob)
                object_features = F.normalize(torch.Tensor(att_feat), dim=-1).numpy()
                obj_label_ids = [-1] * self.max_region_num
            else:  # mask object
                masked = "obj"
                object_features, obj_label_ids = self.random_mask_object(att_feat,obj_label)
                input_ids = cur_input_ids.clone()
                lm_label_ids = [-1] * len(input_ids)

            if img_index != index:  # negative sample
                obj_label_ids = [-1] * self.max_region_num
                lm_label_ids = [-1] * len(input_ids)

            sent.append(input_ids.numpy())
            ori_feats.append(att_feat)
            att_feats.append(object_features)
            box_feats.append(box_feat)
            img_masks.append(img_mask)
            img_ids.append(img_id)
            obj_labels.append(obj_label_ids)  # [B,neg_samples,x]
            lm_labels.append(lm_label_ids)
            masked_types.append(masked)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()
        obj_labels = torch.tensor(obj_labels).long()
        ori_feats = torch.tensor(ori_feats).float()


        img_feas = (sent,att_feats, img_masks, box_feats, obj_labels,lm_labels, itm_label, img_ids,ori_feats,masked_types)

        return img_feas

    def sample_captions(self,index):
        sample_numbers = self.sample_n

        att_feats = []
        box_feats = []
        img_masks = []
        img_ids = []
        obj_labels = []
        lm_labels = []
        sent = []
        masked_types = []
        ori_feats = []
        clcm_sent = []
        clcm_labels = []

        neg_sample_indices = list(range(self.__len__()))
        neg_sample_indices.pop(index)
        sample_indices = random.sample(neg_sample_indices, sample_numbers - 1)
        itm_label = random.randint(0, sample_numbers - 1)
        sample_indices.insert(itm_label, index)

        att_feat, box_feat, img_mask, obj_label, img_id = self.get_img_feature(index,True)

        true_tokens = self.tokenize(self.captions[index], half=True)

        for idx in sample_indices:
            lm_label_ids, obj_label_ids = None, None
            cur_image_id = self.image_ids[idx]  # pos or neg  # enough memory

            if self.data_type=='google':
                cap_id = int(re.sub("\D", "", cur_image_id))  # google
                if self.clager is None:
                    cur_cap = self.captions[cap_id]
                else:
                    cur_cap = self.clager.dclag(self.captions[cap_id], 'en', 1, 0)[0]
            else:
                cap_id = int(cur_image_id.split('_')[0])  # sbu
                if self.clager is None:
                    cur_cap = self.captions[cap_id]
                else:
                    cur_cap = self.clager.dclag(self.captions[cap_id], 'en', 1, 0)[0]
            
            cur_cap_tokens = self.tokenize(cur_cap)
            cur_cap_tokens_half = self.tokenize(cur_cap, half=True)
            concated_tokens = torch.cat([true_tokens, cur_cap_tokens_half], dim=0)

            if random.random() > 0.5:  # mask word
                masked = "word"
                input_ids, lm_label_ids = self.mask_tokens(cur_cap_tokens, mlm_probability=self.mlm_prob)
                object_features = F.normalize(torch.Tensor(att_feat), dim=-1).numpy()
                obj_label_ids = [-1] * self.max_region_num  #100

            else:  # mask object
                masked = "obj"
                object_features, obj_label_ids = self.random_mask_object(att_feat, obj_label)
                input_ids = cur_cap_tokens.clone()
                lm_label_ids = [-1] * len(input_ids)

            if idx != index:  # negative sample
                obj_label_ids = [-1] * self.max_region_num
                lm_label_ids = [-1] * len(input_ids)
                clcm_labels.append(0)
            else:
                clcm_labels.append(1)

            sent.append(input_ids.numpy())  # accumulate sentences
            clcm_sent.append(concated_tokens.clone().numpy())
            ori_feats.append(att_feat)
            att_feats.append(object_features)
            box_feats.append(box_feat)
            img_masks.append(img_mask)
            img_ids.append(img_id)
            obj_labels.append(obj_label_ids)
            lm_labels.append(lm_label_ids)
            masked_types.append(masked)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()
        obj_labels = torch.tensor(obj_labels).long()
        ori_feats = torch.tensor(ori_feats).float()
        clcm_labels = torch.tensor(clcm_labels).long()

        img_feas = (sent, att_feats, img_masks, box_feats, obj_labels, lm_labels, itm_label, img_ids,ori_feats,masked_types, clcm_sent, clcm_labels)

        return img_feas

    def __getitem__(self, index):
        if self.image_ids is None:
            self.update_values()
        t2i_inputs = self.sample_images(index) if self.t2i_flag else None
        i2t_inputs = self.sample_captions(index) if self.i2t_flag else None
        three_type_inputs = [t2i_inputs, i2t_inputs]
        return three_type_inputs

class VLMPretrainCapDataset(Dataset):
    def __init__(self, captions, clager, params, mode='train', data_type='google'):
        # diy dataset, image_ids is equal to all captions, each image with 5 caption
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.mask_index = params.mask_index
        self.n_words = params.n_words

        self.mlm_prob = params.word_pred

        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.captions = captions
        self.clager = clager
        self.tokenizer = XLMRTokenizer(params.vocab_path)

        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        self.mode = mode
        self.params = params

        self.max_len = params.max_len
        # feature related for image features
        self.seq_per_img = params.seq_per_img  # number of captions to sample for each image during training

        self.max_region_num = params.max_region_num

        self.data_type = data_type  # for sample cations

        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank

        self.cc_num = 29
        self.sbu_num = 8
        self.sample_n = params.sample_n

        self.coco_len = 113287

        self.t2i_flag = params.t2i_flag
        self.i2t_flag = params.i2t_flag

        self.is_pretrain = params.is_pretrain
        if mode=='valid':
            valid_file = os.path.join(params.google_valid_path, "google_valid_fp16.h5")
            self.precess_reload(valid_file, True)
            self.val_len = len(self.image_ids)
        else:
            if data_type=='google':
                with open(os.path.join(params.train_order_path, "google_train_order.json"), 'r') as f:
                    self.train_order = json.load(f)
            else:
                with open(os.path.join(params.train_order_path, "sbu_train_order.json"), 'r') as f:
                    self.train_order = json.load(f)

        if data_type == 'google':
            all_train_files = []
            if mode == 'train':
                for google_dataset_idx in range(self.cc_num):
                    train_file = os.path.join(params.input_fea_dir, params.google_path,
                                              "train_" + str(google_dataset_idx) + ".h5")
                    # cur_file = h5py.File(train_file, "r", swmr=True)
                    all_train_files.append(train_file)  # reload lately
                self.all_train_files = all_train_files
                self.update(0) #select
        elif data_type == 'sbu':
            # feaFile = '/hdfs/public/nanduan/data/google_captions/obj100'
            all_train_files = []
            if mode == 'train':
                for _idx in range(self.cc_num, self.cc_num + self.sbu_num):
                    train_file = os.path.join(params.input_fea_dir, params.sbu_path,
                                              "train_" + str(_idx) + ".h5")

                    all_train_files.append(train_file)  # reload lately
                self.all_train_files = all_train_files
                self.update(0)

        self.data_type=data_type
        # after assign image ids
        self.split_len = 100000

    def tokenize(self, sent):
        s = sent.rstrip()
        indexed = self.tokenizer.encode(s)
        indexed = indexed[:self.max_len]
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
            image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
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

    def update_values(self, is_old_pythia=False):
        path_file = self.this_train_file_path
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
            _image_ids = h5py.File(path_file, 'r')["image_id"]
            self.image_ids = [str(ss, encoding="utf8") for ss in _image_ids]
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

    def update(self, epoch=0):
        local_train_files = self.train_order[epoch][:self.n_gpu_per_node]
        _file_num = local_train_files[self.local_rank]
        if self.params.debug_pretrain:
            _file_num=0
        self.this_train_file_path = self.all_train_files[_file_num]

        logger.info('select train file: ' + self.this_train_file_path)
        self.image_ids = None
        self.wh = None
        self.num_boxes = None
        self.boxes = None
        self.obj_features = None
        self.objects = None
        self.distribution = None

        # self.update_values(self.this_train_file_path)

    def __len__(self):
        return self.val_len if self.mode == "valid" else self.split_len

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

        if is_origin == False:
            att_feat = torch.FloatTensor(att_feat)
            att_feat = F.normalize(att_feat, dim=-1).numpy()

        h, w = wh.astype('float32')
        loc_features = self.norm_boxes(box.astype('float32'), h, w)

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        return (att_feat, loc_features, image_mask, objects,img_id)

    def __getitem__(self, index):
        if self.image_ids is None:
            self.update_values()
        box_feats = []
        img_masks = []
        img_ids = []
        sent = []
        att_feats = []

        att_feat, box_feat, img_mask, obj_label, img_id = self.get_img_feature(index)

        cur_image_id = self.image_ids[index]  # pos or neg  # enough memory

        if self.data_type == 'google':
            cap_id = int(re.sub("\D", "", cur_image_id))  # google
            if self.clager is None:
                cur_cap = self.captions[cap_id]
            else:
                cur_cap = self.clager.clag(self.captions[cap_id], 'en')
        else:
            cap_id = int(cur_image_id.split('_')[0])  # sbu
            if self.clager is None:
                cur_cap = self.captions[cap_id]
            else:
                cur_cap = self.clager.clag(self.captions[cap_id], 'en')

        cur_cap_tokens = self.tokenize(cur_cap)

        #sent.append(cur_cap_tokens.numpy())  # accumulate sentences
        att_feats.append(att_feat)
        box_feats.append(box_feat)
        img_masks.append(img_mask)
        img_ids.append(img_id)

        att_feats = torch.tensor(att_feats).float()
        img_masks = torch.tensor(img_masks).long()
        box_feats = torch.tensor(box_feats).float()

        img_feas = [cur_cap_tokens, att_feats, img_masks, box_feats, img_ids]
        return img_feas


class StreamDataset(object):

    def __init__(self, sent, pos, params, langs=[]):
        """
        Prepare batches for data iterator.
        """
        self.params = params
        bptt = params.bptt
        bs = params.batch_size
        self.eos = params.eos_index

        self.n_gpu_per_node = params.n_gpu_per_node if hasattr(params, 'n_gpu_per_node') else 1
        self.local_rank = params.local_rank if hasattr(params, 'local_rank') else 0

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens:] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.has_lan = False
        if len(langs) != 0:
            l_buffer = np.zeros(t_size, dtype=sent.dtype) + params.lang2id['en']
            l_buffer[t_size - n_tokens:] = langs
            l_buffer = l_buffer.reshape((bs, n_batches * bptt)).T
            self.langs = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + params.lang2id['en']
            self.langs = l_buffer
            self.has_lan = True

        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.loaded = {i: [] for i in range(params.n_gpu_per_node)}
        self.reload = False
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def reload_check(self, loaded):
        logger.info("reload records [{}]".format(','.join(str(x) for x in loaded[self.local_rank])))
        self.loaded = loaded
        self.reload = True

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning("Invalid split values: %i %i - %i" % (a, b, self.n_batches))
            return
        assert 0 <= a < b <= self.n_batches
        logger.info("Selecting batches from %i to %i ..." % (a, b))

        # sub-select

        self.data = np.copy(self.data[a * self.bptt:b * self.bptt])

        if self.has_lan:
            self.langs = np.copy(self.langs[a * self.bptt:b * self.bptt])

        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def get_iterator(self, shuffle, subsample=1, seed=0):
        """
        Return a sentences iterator.
        """
        if not self.reload:
            self.loaded[self.local_rank].append(0)
        if shuffle:
            if seed is 0:
                seed = np.random.randint(1, 1e6)
            seed += len(self.loaded[self.local_rank])
            logger.warning("GPU {} shuffled with seed {}".format(self.local_rank,
                                                                 seed))
            rng = np.random.RandomState(seed)
        indexes = (rng.permutation if shuffle else range)(self.n_batches // subsample)

        for k, i in enumerate(indexes):
            if shuffle:
                if self.reload and k < self.loaded[self.local_rank][-1]:
                    continue
            a = self.bptt * i
            b = self.bptt * (i + 1)
            self.loaded[self.local_rank][-1] += 1
            self.reload = False
            if self.has_lan:
                yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths, torch.from_numpy(
                    self.langs[a:b].astype(np.int64))
            else:
                yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths
