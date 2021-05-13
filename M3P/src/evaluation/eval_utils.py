import os
import io
import sys
import argparse
import torch
import sys
import numpy as np


def get_images_batchs(batches, att_loader,image_size_annot,all_train_files,data_type='coco',use_new_fea=False):
    """
    Return a sentences iterator, given the associated sentence batches.
    """
    att_feats = []
    box_feats = []
    img_masks = []
    dist_feats = []
    for img_id in batches:
        _,att_feat, box_feat,num_boxes = get_img_feature(img_id, att_loader,image_size_annot,all_train_files,data_type=data_type,use_new_fea=use_new_fea)
        att_feat, box_feat, img_mask = padding_image_features(att_feat, box_feat,num_boxes=num_boxes,dist_features=None)

        att_feats.append(att_feat)
        box_feats.append(box_feat)
        img_masks.append(img_mask)
        # dist_feats.append(dist_feat)

    att_feats = torch.tensor(att_feats).float()
    img_masks = torch.tensor(img_masks).long()
    box_feats = torch.tensor(box_feats).float()
    #dist_feats = torch.tensor(dist_feats).float()

    img_feas = (att_feats, img_masks, box_feats)

    return img_feas



def get_box_feature(index,all_train_files,data_type='coco',use_new_fea=False):
    cur_id = int(index)
    if data_type=='coco':
        if use_new_fea:
            if cur_id < 113287:
                file_idx = 0
                data_idx = cur_id
            elif cur_id < (113287 + 5000) and cur_id >= 113287:
                file_idx = 1
                data_idx = cur_id - 113287
            else:
                file_idx = 2
                data_idx = cur_id - (113287 + 5000)

            obj_features = all_train_files[file_idx]['bbox'][data_idx]
        else:
            file_idx = cur_id // 82783
            data_idx = cur_id % 82783
            obj_features = all_train_files[file_idx]['boxes'][data_idx]
    else:
        if cur_id < 29000:
            file_idx = 0
            data_idx = cur_id
        elif cur_id < 30014 and cur_id >= 29000:
            file_idx = 1
            data_idx = cur_id - 29000
        else:
            file_idx = 2
            data_idx = cur_id - 30014
        if use_new_fea:
            obj_features = all_train_files[file_idx]['bbox'][data_idx]
        else:
            obj_features = all_train_files[file_idx]['boxes'][data_idx]

    return obj_features

def get_num_box_feature(index,all_train_files):
    cur_id = int(index)
    file_idx = cur_id // 82783
    data_idx = cur_id % 82783
    obj_features = all_train_files[file_idx]['num_boxes'][data_idx]
    return obj_features


def get_img_feature(index, att_loader,image_size_annot,all_train_files,norm_att_feat=True,return_dist=False,data_type='coco',use_new_fea=False):
    #att_feat = att_loader.get(str(index))

    if return_dist:
        att_feat, dist_feat = att_loader.get(str(index))
    else:
        att_feat = att_loader.get(str(index))

    # Reshape to K x C
    att_feat = att_feat.reshape(-1, att_feat.shape[-1])
    if norm_att_feat:
        att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)

    box_feat = get_box_feature(str(index),all_train_files,data_type,use_new_fea)
    num_boxes = None#get_num_box_feature(str(index),all_train_files)
    dist_feat = None
    if return_dist:
        dist_feat = dist_feat.reshape(-1, dist_feat.shape[-1])

        dist_feat = dist_feat.astype('float32')
        dist_feat = dist_feat / np.linalg.norm(dist_feat, 2, 1, keepdims=True)

    # devided by image width and height
    x1, y1, x2, y2 = np.hsplit(box_feat.astype('float32'), 4)
    h, w = np.array(image_size_annot[index]).astype('float32')

    # for location
    box_feat = np.hstack((x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)

    return (dist_feat,att_feat, box_feat,num_boxes)


def padding_image_features(att_features, box_features,max_region_num=100,num_boxes=None,dist_features=None):
    if num_boxes is None:
        num_boxes = box_features.shape[0]

    mix_num_boxes = min(int(num_boxes),max_region_num)
    mix_boxes_pad = np.zeros((max_region_num, 5))
    mix_features_pad = np.zeros((max_region_num, 2048))
    #mix_dist_features = np.zeros((max_region_num,1600))

    image_mask = [1] * (int(mix_num_boxes))
    while len(image_mask) < max_region_num:
        image_mask.append(0)

    mix_boxes_pad[:mix_num_boxes] = box_features[:mix_num_boxes]
    mix_features_pad[:mix_num_boxes] = att_features[:mix_num_boxes]
    #mix_dist_features[:mix_num_boxes] = dist_features[:mix_num_boxes]

    return mix_features_pad, mix_boxes_pad, image_mask

