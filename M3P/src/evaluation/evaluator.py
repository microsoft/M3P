# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from ..utils import to_cuda, restore_segmentation, concat_batches

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)

logger = getLogger()


def test_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    origin_input_batch, _ = list(zip(*data))

    def generate_inputs(_batch,is_origin=False):
        if is_origin:
            cap_batch, img_batch, img_coords_batch, img_mask_batch, img_label_batch, obj_feat_ori, mask_img_batch, img_ids = zip(
                *_batch)
        else:
            cap_batch, img_batch, img_coords_batch,img_mask_batch,img_label_batch, obj_feat_ori,mask_img_batch, img_ids, pos_img_label = zip(*_batch)
        # concat_input_ids,concat_input_lengths = batch_sentences(cap_batch)

        if is_origin:
            max_length = cap_batch[0].size(0)
            concat_input_lengths = torch.LongTensor([max_length for s in cap_batch])
        else:
            max_length = cap_batch[0].size(1)
            concat_input_lengths = torch.LongTensor([max_length for s in cap_batch])
            concat_input_lengths = concat_input_lengths.repeat(cap_batch[0].size(0),1)

        if is_origin:
            pos_labels = None
        else:
            pos_labels = pos_img_label

        _inputs = [torch.stack(cap_batch, dim=0),
                  concat_input_lengths,
                  torch.stack(img_batch, dim=0),
                  torch.stack(img_coords_batch, dim=0),
                  torch.stack(img_mask_batch, dim=0),
                  torch.stack(img_label_batch, dim=0),
                  torch.stack(obj_feat_ori, dim=0),
                  torch.stack(mask_img_batch,dim=0),
                  img_ids,  # google_replace
                  pos_labels
                  ]
        return _inputs

    origin_inputs = generate_inputs(origin_input_batch,True) if origin_input_batch[0] is not None else None

    all_return_results = origin_inputs
    return all_return_results

def valid_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    origin_input_batch, both_input_batch = list(zip(*data))
    all_return_results = [None,None, None]

    t2i_batch, i2t_batch = zip(*both_input_batch)

    # t2i
    def generate_inputs(_batch,is_origin=False):
        if is_origin:
            cap_batch, img_batch, img_coords_batch, img_mask_batch, img_label_batch, obj_feat_ori, mask_img_batch, img_ids = zip(
                *_batch)
        else:
            cap_batch, img_batch, img_coords_batch,img_mask_batch,img_label_batch, obj_feat_ori,mask_img_batch, img_ids, pos_img_label = zip(*_batch)
        # concat_input_ids,concat_input_lengths = batch_sentences(cap_batch)

        if is_origin:
            max_length = cap_batch[0].size(0)
            concat_input_lengths = torch.LongTensor([max_length for s in cap_batch])
        else:
            max_length = cap_batch[0].size(1)
            concat_input_lengths = torch.LongTensor([max_length for s in cap_batch])
            concat_input_lengths = concat_input_lengths.repeat(cap_batch[0].size(0),1)

        if is_origin:
            pos_labels = None
        else:
            pos_labels = pos_img_label

        _inputs = [torch.stack(cap_batch, dim=0),
                  concat_input_lengths,
                  torch.stack(img_batch, dim=0),
                  torch.stack(img_coords_batch, dim=0),
                  torch.stack(img_mask_batch, dim=0),
                  torch.stack(img_label_batch, dim=0),
                  torch.stack(obj_feat_ori, dim=0),
                  torch.stack(mask_img_batch,dim=0),
                  img_ids,  # google_replace
                  pos_labels
                  ]
        return _inputs

    t2i_inputs = generate_inputs(t2i_batch) if t2i_batch[0] is not None else None
    # i2t
    i2t_inputs = generate_inputs(i2t_batch) if i2t_batch[0] is not None else None

    origin_inputs = generate_inputs(origin_input_batch,True) if origin_input_batch[0] is not None else None

    all_return_results = [origin_inputs,t2i_inputs, i2t_inputs]
    return all_return_results

def retrieval_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # separate source and target sequences
    origin_input_batch, _ = list(zip(*data))

    def generate_inputs(_batch):

        cap_batch, img_batch, img_coords_batch, img_mask_batch, img_label_batch, obj_feat_ori, mask_img_batch, img_ids,itm_label = zip(
            *_batch)

        max_length = cap_batch[0].size(1)
        concat_input_lengths = torch.LongTensor([max_length for s in cap_batch])
        concat_input_lengths = concat_input_lengths.repeat(cap_batch[0].size(0), 1)
        concat_input_lengths = concat_input_lengths.transpose(0,1)
        pos_labels = itm_label

        _inputs = [torch.stack(cap_batch, dim=0),
                  concat_input_lengths,
                  torch.stack(img_batch, dim=0),
                  torch.stack(img_coords_batch, dim=0),
                  torch.stack(img_mask_batch, dim=0),
                  torch.stack(img_label_batch, dim=0),
                  torch.stack(obj_feat_ori, dim=0),
                  torch.stack(mask_img_batch,dim=0),
                  img_ids,  # google_replace
                  pos_labels
                  ]
        return _inputs

    origin_inputs = generate_inputs(origin_input_batch) if origin_input_batch[0] is not None else None

    all_return_results = origin_inputs
    return all_return_results

class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = params.dico
        self.params = params

    def get_iterator(self, data_set, lang1, lang2=None):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test','test_1k']
        assert lang1 in self.params.langs or lang1 == 'img' or lang2 == 'img'
        assert lang2 is None or lang2 in self.params.langs or lang1 == 'img' or lang2 == 'img'

        dataset = self.data['cross_modal'][(lang1, lang2)][data_set]

        #not use for update

        #sampler = SequentialSampler(dataset)  # the order is fixed for evaluation

        if data_set=='valid':
            sampler = RandomSampler(dataset)
            data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                     collate_fn=valid_collate, num_workers=self.params.num_workers)
        elif data_set=='test_1k':#only for retrieval
            sampler = SequentialSampler(dataset)
            data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                     collate_fn=retrieval_collate, num_workers=self.params.num_workers)
        else:
            sampler = SequentialSampler(dataset)
            data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                     collate_fn=test_collate, num_workers=self.params.num_workers)


        for batch_idx, batch in enumerate(data_loader):
            if data_set=='valid' and self.params.batch_size*(batch_idx+1)>300:break
            yield batch

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        pred_mask = pred_mask.bool()

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)

                for lang in params.mass_steps:
                    self.evaluate_mass(scores, data_set, lang)

                mass_steps = []
                for lang1 in params.mass_steps:
                    for lang2 in params.mass_steps:
                        if lang1 != lang2:
                            mass_steps.append((lang1, lang2))
                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps] + mass_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                # multi-modal translation task (evaluate perplexity and accuracy)

                for lang1, lang2 in set(params.cross_modal_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_ic(scores, data_set, lang1, lang2, eval_bleu)

                #for multi-modal pretraining tasks
                for lang1, lang2 in set(params.cross_mass_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_imlm(scores, data_set, lang1, lang2, eval_bleu)

                for lang1, lang2 in set(params.cross_ae_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_ida(scores, data_set, lang1, lang2, eval_bleu)
                    #self.evaluate_cross_img2img_step(scores, data_set, lang1, lang2)

                #for understanding

                for lang1, lang2 in set(params.cross_mlm_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_cmlm(scores, data_set, lang1, lang2)

                for lang1, lang2 in set(params.cross_mrm_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mrm(scores, data_set, lang1, lang2)

                if params.do_finetune:
                    for lang1, lang2 in set(params.cross_rel_steps):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_i2t(scores, data_set, lang1, lang2)

                    for lang1, lang2 in set(params.cross_rel_steps):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_t2i(scores, data_set, lang1, lang2)
                else:
                    for lang1, lang2 in set(params.cross_rel_steps):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_rel(scores, data_set, lang1, lang2)

                if params.eval_retrieval:
                    for lang1, lang2 in set(params.cross_rel_steps):
                        self.evaluate_image_retrieval(scores, data_set, lang1, lang2)


                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean(
                        [scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean(
                        [scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean(
                        [scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean(
                        [scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

                _mass_step = [l1 for l1 in params.mass_steps]
                if len(_mass_step) > 0:
                    scores['%s_mass_ppl' % data_set] = np.mean(
                        [scores['%s_%s-%s_mass_ppl' % (data_set, lang1, lang1)] for lang1 in _mass_step])
                    scores['%s_mass_acc' % data_set] = np.mean(
                        [scores['%s_%s-%s_mass_acc' % (data_set, lang1, lang1)] for lang1 in _mass_step])

                _cross_modal_step = [(l1, l2) for (l1, l2) in params.cross_modal_steps]

                if params.is_bert_based==False:
                    if len(_cross_modal_step) > 0:
                        scores['%s_IC_ppl' % data_set] = np.mean(
                            [scores['%s_%s-%s_IC_ppl' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             _cross_modal_step])
                        scores['%s_IC_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_IC_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             _cross_modal_step])


                cross_mass_step = [l1 for l1 in params.cross_mass_steps]
                if len(cross_mass_step) > 0:
                    scores['%s_IMLM_ppl' % data_set] = np.mean(
                        [scores['%s_%s-%s_IMLM_ppl' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_mass_step])
                    scores['%s_IMLM_acc' % data_set] = np.mean(
                        [scores['%s_%s-%s_IMLM_acc' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_mass_step])

                cross_ae_step = [l1 for l1 in params.cross_ae_steps]
                if len(cross_ae_step) > 0:
                    scores['%s_IDA_ppl' % data_set] = np.mean(
                        [scores['%s_%s-%s_IDA_ppl' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_ae_step])
                    scores['%s_IDA_acc' % data_set] = np.mean(
                        [scores['%s_%s-%s_IDA_acc' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_ae_step])
                    # scores['%s_cross_img2img_acc' % data_set] = np.mean(
                    #     [scores['%s_%s-%s_cross_modal_img2img_acc' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_ae_step])

                cross_mlm_steps = [l1 for l1 in params.cross_mlm_steps]
                if len(cross_mlm_steps)>0:
                    scores['%s_CMLM_ppl' % data_set] = np.mean(
                        [scores['%s_%s-%s_cmlm_ppl' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_mlm_steps])
                    scores['%s_CMLM_acc' % data_set] = np.mean(
                        [scores['%s_%s-%s_cmlm_acc' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_mlm_steps])

                cross_mrm_steps = [l1 for l1 in params.cross_mrm_steps]
                if len(cross_mrm_steps) > 0:
                    scores['%s_MRM_R1' % data_set] = np.mean(
                        [scores['%s_%s-%s_mrm_r1' % (data_set, lang1, lang2)] for (lang1, lang2) in
                         cross_mrm_steps])
                    scores['%s_MRM_R5' % data_set] = np.mean(
                        [scores['%s_%s-%s_mrm_r1' % (data_set, lang1, lang2)] for (lang1, lang2) in
                         cross_mrm_steps])
                    scores['%s_MRM_R10' % data_set] = np.mean(
                        [scores['%s_%s-%s_mrm_r1' % (data_set, lang1, lang2)] for (lang1, lang2) in
                         cross_mrm_steps])

                cross_rel_steps = [l1 for l1 in params.cross_rel_steps]
                if len(cross_rel_steps) > 0:
                    if params.do_finetune:
                        scores['%s_I2T_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_i2t_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])
                        scores['%s_T2I_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_t2i_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])
                    else:
                        scores['%s_I2T_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_rel_i2t_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])
                        scores['%s_T2I_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_rel_t2i_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])

                if params.multi_eval:
                    for lang1, lang2 in set(params.cross_modal_steps):
                        _eval_rpt = self.evaluate_image_caption(scores,data_set,lang1,lang2)
                else:
                    if params.eval_coco and params.is_master:
                        split = 'val'
                        if data_set == 'valid':
                            split = 'val'
                        else:
                            split = 'test'
                        coco_eval_rpt = self.evaluate_coco(params, params.dump_path, params.sentences_eval,
                                                           params.coco_method, split=split)
                        coco_methods = params.coco_method.split(',')
                        for method in coco_methods:
                            scores['%s_coco_' % data_set + method] = coco_eval_rpt[method]
                    if params.eval_flicker and params.is_master:
                        if data_set == 'valid':
                            split = 'val'
                        else:
                            split = 'test'
                        coco_eval_rpt = self.evaluate_fliker(params, params.dump_path, params.sentences_eval,
                                                             params.coco_method, split=split)
                        coco_methods = params.coco_method.split(',')
                        for method in coco_methods:
                            scores['%s_flicker_' % data_set + method] = coco_eval_rpt[method]


        return scores

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id,
                                                              params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, lang1) if lang2 is None else '%s_%s-%s_mlm_ppl' % (
        data_set, lang1, lang2)
        acc_name = '%s_%s_mlm_acc' % (data_set, lang1) if lang2 is None else '%s_%s-%s_mlm_acc' % (
        data_set, lang1, lang2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.


class XEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model

    def mask_sent(self, x, lengths, rng):

        def random_start(end):
            p = rng.rand()
            if p >= 0.8:
                return 1
            elif p >= 0.6:
                return end - 1
            else:
                return rng.randint(1, end)

        def mask_word(w):
            p = rng.rand()
            if p >= 0.2:
                return self.params.mask_index
            elif p >= 0.05:
                return rng.randint(self.params.n_words)
            else:
                return w

        positions, inputs, targets, outputs, len2 = [], [], [], [], []
        for i in range(lengths.size(0)):
            words = x[:lengths[i], i].tolist()
            l = len(words)
            # Prevent some short sentences will be whole masked
            mask_len = max(1, round(l * self.params.word_mass) - 1)
            start = random_start(l - mask_len + 1)
            len2.append(mask_len)

            pos_i, target_i, output_i, input_i = [], [], [], []
            prev_w = None
            for j, w in enumerate(words):
                if j >= start and j < start + mask_len:
                    output_i.append(w)
                    target_i.append(prev_w)
                    pos_i.append(j - 1)
                    input_i.append(mask_word(w))
                else:
                    input_i.append(w)
                prev_w = w

            inputs.append(input_i)
            targets.append(target_i)
            outputs.append(output_i)
            positions.append(pos_i)

        l1 = lengths.clone()
        l2 = torch.LongTensor(len2)
        x1 = torch.LongTensor(max(l1), l1.size(0)).fill_(self.params.pad_index)
        x2 = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)
        y = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)
        pos = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)

        for i in range(l1.size(0)):
            x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
            x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
            y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
            pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))
        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)

        return x1, l1, x2, l2, y, pred_mask, pos

    def mask_region(self,x_img,image_labels):
        slen,bs = x_img.size()[0],x_img.size()[1]
        pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
        pred_mask = pred_mask.view(slen, bs)
        pred_mask[image_labels!=-1] = 1
        y = image_labels[image_labels>0]
        return x_img,y,pred_mask.bool()

    def evaluate_ic(self, scores, data_set, lang1, lang2, eval_bleu):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        xe_loss = 0
        n_valid = 0

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs,_,_ = batch

            text, text_len = origin_inputs[0:2]
            x1, img_loc, x1_mask = origin_inputs[2:5]
            text = text.transpose(0, 1)

            text_len = (text != self.params.pad_index).sum(dim=0)
            x2 = text[:text_len.max()]  # no padding
            # no padding text
            len2 = text_len

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            #
            len1 = x1_mask.sum(dim=1)
            x1 = x1.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            # cuda
            x1, len1, img_loc, x2, len2, y, x1_mask = to_cuda(x1, len1, img_loc, x2, len2, y, x1_mask)

            # encode source sentence
            enc1 = model('crossfwd', stream_='img', x=x1, lengths=len1, langs=None, causal=False, cross_modal=True,
                         image_loc=img_loc, image_dist=None)
            enc1 = enc1.transpose(0, 1)

            # decode target sentence
            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=None, causal=True, src_enc=enc1,
                         src_len=len1)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text

            # compute perplexity and prediction accuracy
        scores['%s_%s-%s_IC_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_IC_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

    def evaluate_imlm(self, scores, data_set, lang1, lang2, eval_bleu):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []
        rng = np.random.RandomState(0)

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs,_,_ = batch

            text, text_len = origin_inputs[0:2]
            x_img, img_loc, x_img_mask = origin_inputs[2:5]
            text = text.transpose(0, 1)

            text_len = (text != self.params.pad_index).sum(dim=0)
            x1 = text[:text_len.max()]  # no padding
            # no padding text
            len1 = text_len

            (x1, len1, x2, len2, y, pred_mask, positions) = self.mask_sent(x1, len1, rng)

            img_len = x_img_mask.sum(dim=1)
            x_img = x_img.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            # cuda
            x1, len1, x2, len2, y, positions, \
            x_img, img_loc, img_len, x_img_mask = to_cuda(x1, len1, x2, len2, y, positions, x_img, img_loc, img_len,
                                                          x_img_mask)

            enc1 = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                         cache=None,
                         image_loc=img_loc, refine_image=params.refine_image)
            enc1 = enc1.transpose(0, 1)
            img_len = torch.add(img_len, len1)


            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2,
                         langs=None, causal=True,
                         src_enc=enc1, src_len=img_len, positions=positions)
            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_IMLM_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_IMLM_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

    def evaluate_ida(self, scores, data_set, lang1, lang2, eval_bleu):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []
        rng = np.random.RandomState(0)

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs,_,_ = batch

            text, text_len = origin_inputs[0:2]
            x_img, img_loc, x_img_mask = origin_inputs[2:5]
            text = text.transpose(0, 1)

            text_len = (text != self.params.pad_index).sum(dim=0)
            x1 = text[:text_len.max()]  # no padding
            # no padding text
            len1 = text_len

            (x1, len1, x2, len2, y, pred_mask, positions) = self.mask_sent(x1, len1, rng)

            img_len = x_img_mask.sum(dim=1)
            x_img = x_img.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            # cuda
            x1, len1, x2, len2, y, positions, x_img, img_loc, img_len = to_cuda(x1, len1, x2, len2, y, positions,
                                                                                x_img, img_loc, img_len)

            img_enc, img_mask = model('ImageEmbed', x=x_img, lengths=img_len, causal=False, image_loc=img_loc,
                                      refine_image=params.refine_image, image_dist=None)

            enc1 = model('crossfwd', stream_='text', x=x1, lengths=len1, langs=None, causal=False,
                         image_fusion=True, image_enc=img_enc, image_mask=img_mask)
            enc1 = enc1.transpose(0, 1)

            # enc_mask = x1.ne(params.mask_index)
            # enc_mask = enc_mask.transpose(0, 1)
            # decode target sentence

            enc_mask = x1.ne(params.mask_index)
            enc_mask = enc_mask.transpose(0, 1)

            dec2 = model('crossfwd', x=x2, lengths=len2,
                         langs=None, causal=True,
                         src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_IDA_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_IDA_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

    def get_standard_inputs(self,_inputs, is_samples=False,is_cuda=True,is_masked=False):
        #unpack image and text features for trainer
        text, text_len = _inputs[0:2]
        x_img, img_loc, x_img_mask = _inputs[2:5]
        if is_masked:
            x_img = _inputs[-3]

        if is_samples: #mask only for mrm and mrfr , sampling not used
            x_img, img_loc, x_img_mask, x_img_labels, x_ori_img,x_mask_img, x_img_id, x_pos_label = _inputs[2:]
            text = text.view(-1, text.size()[-1])
            x_img = x_img.view([-1] + list(tuple(x_img.size()[2:])))
            img_loc = img_loc.view([-1] + list(tuple(img_loc.size()[2:])))
            x_img_mask = x_img_mask.view([-1] + list(tuple(x_img_mask.size()[2:])))
            x_img_labels = x_img_labels.view([-1] + list(tuple(x_img_labels.size()[2:])))
            x_ori_img = x_ori_img.view([-1] + list(tuple(x_ori_img.size()[2:])))

        text = text.transpose(0, 1)
        # no padding text
        text_len = (text != self.params.pad_index).sum(dim=0)
        text = text[:text_len.max()]  # no padding

        len_img = x_img_mask.sum(dim=1)
        x_img = x_img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        if is_samples:
            if is_cuda:
                text, text_len, x_img, img_loc, len_img, x_img_labels, x_ori_img = to_cuda(text, text_len, x_img, img_loc,
                                                                                           len_img, x_img_labels, x_ori_img)

            return text, text_len, x_img, len_img, img_loc, x_img_labels, x_ori_img, x_img_id, x_pos_label
        if is_cuda:
            text, text_len, x_img, img_loc, len_img = to_cuda(text, text_len, x_img, img_loc, len_img)
        return text, text_len, x_img, len_img, img_loc

    def evaluate_rel(self, scores, data_set, lang1, lang2):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_valid_t2i = 0
        n_valid_i2t = 0

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs, t2i_inputs, i2t_inputs = batch

            #origin_outputs = self.get_standard_inputs(origin_inputs)
            t2i_outputs = self.get_standard_inputs(t2i_inputs,True) if t2i_inputs is not None else None
            i2t_outputs = self.get_standard_inputs(i2t_inputs,True) if i2t_inputs is not None else None

            sample_n = t2i_inputs[0].size()[1] if t2i_inputs is not None else 0
            sample_n = i2t_inputs[0].size()[1] if sample_n==0 and i2t_inputs  is not None else sample_n

            if sample_n==0:
                logger.warning('no sampling for image text matching task')
                assert False

            def get_relation_score( _features):
                x, lengths, x_img, len_img, img_loc = _features
                if params.use_enc_att:
                    enc1 = model('jointfwd', x=x, lengths=lengths, x_img=x_img[:36], lengths_img=len_img - 64, causal=False,
                                 cache=None,
                                 image_loc=img_loc[:36], refine_image=params.refine_image)
                    enc1 = enc1.transpose(0, 1)
                    len1 = torch.add(len_img - 64, lengths)
                else:
                    enc1 = len1 = None

                # for clf decoder
                enc2 = model('jointfwd', x=x, lengths=lengths, x_img=x_img, lengths_img=len_img, causal=False, cache=None,
                             image_loc=img_loc, refine_image=params.refine_image, src_enc=enc1, src_len=len1)
                enc2 = enc2.transpose(0, 1)
                len2 = torch.add(len_img, lengths)
                # enc2_pooled = model.pooled_layer(enc2)
                # relation_scores = model.seq_relationship(enc2_pooled)
                relation_scores = model('predict', tensor=enc2, is_relation=True)
                return relation_scores

            def get_matching_acc_num(_outputs):
                _t2i_scores = get_relation_score(_outputs[0:5])
                t2i_labels = np.array(_outputs[-1])
                matching_label = torch.from_numpy(t2i_labels)
                pred_label = _t2i_scores.view(-1, sample_n).cpu().max(1)[1]
                _acc = (pred_label == matching_label).sum().item()
                return _acc,len(matching_label)


            t2i_acc_num,total_num = get_matching_acc_num(t2i_outputs)
            i2t_acc_num,_ = get_matching_acc_num(i2t_outputs)

            n_valid_t2i += t2i_acc_num
            n_valid_i2t += i2t_acc_num
            n_words+= total_num


        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_rel_t2i_acc' % (data_set, lang1, lang2)] = 100. * n_valid_t2i / n_words
        scores['%s_%s-%s_rel_i2t_acc' % (data_set, lang1, lang2)] = 100. * n_valid_i2t / n_words

    def evaluate_cmlm(self, scores, data_set, lang1, lang2):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_valid = 0
        xe_loss=0
        rng = np.random.RandomState(0)

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs, _, _ = batch

            origin_outputs = self.get_standard_inputs(origin_inputs, False, False)

            # neg_x neg_lengths for sample image
            #
            x, lengths = origin_outputs[:2]

            x_img, len_img, img_loc = origin_outputs[2:5]

            x, y, pred_mask = self.mask_out(x, lengths,rng)

            # cuda
            x, y, pred_mask, lengths, x_img, len_img, img_loc = to_cuda(x, y, pred_mask, lengths, x_img, len_img,
                                                                        img_loc)

            def get_output(_features):
                x, lengths, x_img, len_img, img_loc = _features
                if params.use_enc_att:
                    enc1 = model('jointfwd', x=x, lengths=lengths, x_img=x_img[:36], lengths_img=len_img - 64,
                                 causal=False,
                                 cache=None,
                                 image_loc=img_loc[:36], refine_image=params.refine_image)
                    enc1 = enc1.transpose(0, 1)
                    len1 = torch.add(len_img - 64, lengths)
                else:
                    enc1 = len1 = None

                # for clf decoder
                enc2 = model('jointfwd', x=x, lengths=lengths, x_img=x_img, lengths_img=len_img, causal=False,
                             cache=None,
                             image_loc=img_loc, refine_image=params.refine_image, src_enc=enc1, src_len=len1)

                return enc2

            _features = [x, lengths, x_img, len_img, img_loc]

            tensor = get_output(_features)
            # only mask text
            tensor = tensor[x_img.shape[0]:]

            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_cmlm_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores['%s_%s-%s_cmlm_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words if n_words > 0 else 0.

    def calc_obj_pred_recall(self,all_masked_obj_pred, all_obj_label_ids):
        r1 = 0
        r5 = 0
        r10 = 0
        for i in range(len(all_masked_obj_pred)):
            for j, pred_idx in enumerate(all_masked_obj_pred[i]):
                if pred_idx == all_obj_label_ids[i]:
                    if j < 10:
                        r10 += 1
                    if j < 5:
                        r5 += 1
                    if j < 1:
                        r1 += 1
                    break
        r1 /= (len(all_masked_obj_pred) + 1e-6)
        r5 /= (len(all_masked_obj_pred) + 1e-6)
        r10 /= (len(all_masked_obj_pred) + 1e-6)
        return r1, r5, r10

    def evaluate_mrm(self, scores, data_set, lang1, lang2):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_r1=0
        n_r5=0
        n_r10=0
        rng = np.random.RandomState(0)

        iterator = self.get_iterator(data_set, lang1, lang2)

        for batch in iterator:
            origin_inputs, _, _ = batch

            origin_outputs = self.get_standard_inputs(origin_inputs, False, False, True)

            x, lengths = origin_outputs[:2]
            x_img, len_img, img_loc = origin_outputs[2:5]

            img_labels = origin_inputs[5]

            x_img, y, pred_mask = self.mask_region(x_img, img_labels.transpose(0, 1))

            # cuda
            x, y, pred_mask, lengths, x_img, len_img, img_loc = to_cuda(x, y, pred_mask, lengths, x_img, len_img, img_loc)

            def get_output(_features):
                x, lengths, x_img, len_img, img_loc = _features
                if params.use_enc_att:
                    enc1 = model('jointfwd', x=x, lengths=lengths, x_img=x_img[:36], lengths_img=len_img - 64, causal=False,
                                 cache=None,
                                 image_loc=img_loc[:36], refine_image=params.refine_image)
                    enc1 = enc1.transpose(0, 1)
                    len1 = torch.add(len_img - 64, lengths)
                else:
                    enc1 = len1 = None

                # for clf decoder
                enc2 = model('jointfwd', x=x, lengths=lengths, x_img=x_img, lengths_img=len_img, causal=False,
                             cache=None,
                             image_loc=img_loc, refine_image=params.refine_image, src_enc=enc1, src_len=len1)

                return enc2

            _features = [x, lengths, x_img, len_img, img_loc]

            tensor = get_output(_features)
            tensor = tensor[:x_img.shape[0]]


            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True,is_obj=True)

            # update stats
            try:
                all_masked_obj_pred = word_scores.topk(10, dim=-1)[1].cpu()
                all_obj_label_ids = y.cpu()

                r1, r5, r10 = self.calc_obj_pred_recall(all_masked_obj_pred, all_obj_label_ids)

                n_words += 1
                n_r1 += r1
                n_r5 += r5
                n_r10 += r10
            except:
                continue

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mrm_r1' % (data_set, lang1, lang2)] = n_r1/n_words
        scores['%s_%s-%s_mrm_r5' % (data_set, lang1, lang2)] = n_r5/n_words
        scores['%s_%s-%s_mrm_r10' % (data_set, lang1, lang2)] = n_r10/n_words

    def inference(self,scores,data_set, lang1, lang2):
        params = self.params
        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        t2i_r1 = 0
        t2i_r5 = 0
        t2i_r10 = 0

        all_matching_scores = []
        all_matching_labels = []
        all_matching_ids = []

        iterator = self.get_iterator('test_1k', lang1, lang2)
        total_len = len(self.data['cross_modal'][(lang1, lang2)]['test_1k'])//self.params.batch_size
        for batch in tqdm(iterator,leave=False,total=total_len):

            origin_outputs = self.get_standard_inputs(batch, True, True)
            pos_cap_label = origin_outputs[-1]
            # for save memory
            def get_relation_score( _features):
                x, lengths, x_img, len_img, img_loc = _features
                if params.use_enc_att:
                    enc1 = model('jointfwd', x=x, lengths=lengths, x_img=x_img[:36], lengths_img=len_img - 64, causal=False,
                                 cache=None,
                                 image_loc=img_loc[:36], refine_image=params.refine_image)
                    enc1 = enc1.transpose(0, 1)
                    len1 = torch.add(len_img - 64, lengths)
                else:
                    enc1 = len1 = None

                # for clf decoder
                enc2 = model('jointfwd', x=x, lengths=lengths, x_img=x_img, lengths_img=len_img, causal=False, cache=None,
                             image_loc=img_loc, refine_image=params.refine_image, src_enc=enc1, src_len=len1)
                enc2 = enc2.transpose(0, 1)
                len2 = torch.add(len_img, lengths)
                # enc2_pooled = model.pooled_layer(enc2)
                # relation_scores = model.seq_relationship(enc2_pooled)
                relation_scores = model('predict', tensor=enc2, is_relation=True)
                return relation_scores

            rel_score = get_relation_score(origin_outputs[0:5])

            all_matching_scores.append(rel_score.detach().cpu())

            all_matching_labels.append(torch.from_numpy(np.array(pos_cap_label).reshape(-1,1)))
            all_matching_ids.append(batch[-2])

        all_matching_labels = torch.cat(all_matching_labels, 0)  # 1000 * 5000
        all_matching_scores = torch.cat(all_matching_scores, 0)  # 1000 * 5000
        #all_matching_ids = torch.cat(all_matching_ids,0)
        all_matching_ids = np.concatenate(np.concatenate(np.array(all_matching_ids)))

        assert len(all_matching_labels)%5000==0

        #each imageid with 5000 captions
        all_matching_labels = all_matching_labels.view(-1,5000) # 1000 * 5000
        all_matching_scores = all_matching_scores.view(-1,5000)# 1000 * 5000
        total_img_len = len(all_matching_scores)

        assert len(all_matching_labels)==len(all_matching_scores)
        np.save(os.path.join(params.dump_path, data_set+"_score_" + "epoch_"+str(scores['epoch'])+"rank_"+str(params.local_rank) + ".npy"),
                all_matching_scores.numpy())
        np.save(os.path.join(params.dump_path, data_set+"_label_" + "epoch_" + str(scores['epoch']) + "rank_" + str(
            params.local_rank) + ".npy"),
                all_matching_labels.numpy())
        np.save(os.path.join(params.dump_path,  data_set+"_ids_" + "epoch_" + str(scores['epoch']) + "rank_" + str(
            params.local_rank) + ".npy"),
                all_matching_ids)

        # image to sentence
        i2t_r1 = 0
        i2t_r5 = 0
        i2t_r10 = 0
        # i2t_pos_label = torch.eye(total_img_len).repeat(1, 5).view(total_len, total_img_len).t()  # 1000*5000
        _, pred = all_matching_scores.topk(10, dim=-1)
        for i in range(len(pred)):
            for j, pred_idx in enumerate(pred[i][:10]):
                if all_matching_labels[i][pred_idx] == 1:
                    if j < 1:
                        i2t_r1 += 1
                        i2t_r5 += 1
                        i2t_r10 += 1
                        break
                    if j < 5:
                        i2t_r5 += 1
                        i2t_r10 += 1
                        break
                    if j < 10:
                        i2t_r10 += 1
                        break
        # sentence to image
        all_matching_scores_t = all_matching_scores.t()  # 5000 * 1000
        all_matching_labels_t = all_matching_labels.t()  # 5000 * 1000
        _, pred = all_matching_scores_t.topk(10, dim=-1)
        for i in range(len(pred)):
            for j, pred_idx in enumerate(pred[i][:10]):
                if all_matching_labels_t[i][pred_idx] == 1:
                    if j < 10:
                        t2i_r10 += 1
                    if j < 5:
                        t2i_r5 += 1
                    if j < 1:
                        t2i_r1 += 1

        return t2i_r1 / 5000, t2i_r5 / 5000, t2i_r10 / 5000, \
               i2t_r1 / total_img_len, i2t_r5 / total_img_len, i2t_r10 / total_img_len

    def evaluate_image_retrieval(self,scores,data_set,lang1,lang2):
        #check data by each image with 5 images
        params = self.params

        t2i_r1, t2i_r5, t2i_r10, i2t_r1, i2t_r5, i2t_r10 = self.inference(scores,data_set, lang1, lang2)
        
        logger.info('i2t %s resutls epoch %i R1: %f R5 %f R10 %f' %(data_set,int(scores['epoch']), i2t_r1, i2t_r5, i2t_r10))
        logger.info('t2i %s resutls epoch %i R1: %f R5 %f R10 %f' % (data_set, int(scores['epoch']), t2i_r1, t2i_r5, t2i_r10))

        with open(os.path.join(params.dump_path, "inference.log"), "a") as f:
            f.write(" ".join([str(t2i_r1), str(t2i_r5), str(t2i_r10)]) + "\n")
            f.write(" ".join([str(i2t_r1), str(i2t_r5), str(i2t_r10)]) + "\n")

        scores['%s_%s_retrieval_recall' % (data_set, lang1)] = t2i_r1 + t2i_r5 + t2i_r10 + i2t_r1 + i2t_r5 + i2t_r10


    def get_image_iterator(self,data_set, lang1, lang2=None):

        dataset = self.data['cross_modal'][(lang1, lang2)][data_set]
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                                 collate_fn=test_collate, num_workers=self.params.num_workers)

        for batch_idx, batch in enumerate(data_loader):
            if data_set=='valid' and self.params.batch_size*(batch_idx+1)>500:break
            yield batch

    def evaluate_image_caption(self, scores, data_set,lang1, lang2):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model


        iterator = self.get_image_iterator(data_set, lang1, lang2)
        total_len = len(self.data['cross_modal'][(lang1, lang2)][data_set])

        outs = []
        for batch in tqdm(iterator, leave=False, total=total_len):
            origin_inputs = batch
            x1, img_loc, x1_mask = origin_inputs[2:5]
            img_id = origin_inputs[-2]

            len1 = x1_mask.sum(dim=1)
            x1 = x1.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            encoded = model('crossfwd', stream_='img', x=x1.cuda(), lengths=len1.cuda(), langs=None, causal=False,
                            cross_modal=True,
                            image_loc=img_loc.cuda(), image_dist=None)
            encoded = encoded.transpose(0, 1)

            if data_set == 'test':
                decoded, dec_lengths = model.generate_beam(
                    encoded, len1.cuda(), None, beam_size=2,
                    length_penalty=1,
                    early_stopping=True,
                    max_len=int(1.5 * len1.max().item() + 10))
            else:
                decoded, dec_lengths = model.generate(encoded, len1.cuda(), None,
                                                      max_len=int(1.5 * len1.max().item() + 10))

            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self.params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
                # output translation
                # source = src_sent[i + j].strip()
                target = " ".join([self.dico[sent[k].item()] for k in range(len(sent))])
                import re
                target = target.replace('@@ ', '')
                cur_dict = {}
                cur_dict['caption'] = target
                cur_dict['image_id'] = img_id[0]
                outs.append(cur_dict)

        out_path = os.path.join(params.dump_path, data_set + '_generated_' + "epoch_" + str(scores['epoch']) +
                                "rank_" + str(params.local_rank) + 'caption_translate.json')
        jsObj = json.dumps(outs)
        with open(out_path, "w") as f:
            f.write(jsObj)
            f.close()


        # out_path = os.path.join(params.dump_path, data_set+'_generated_'+ "epoch_" + str(scores['epoch'])+
        #                         "rank_" + str(params.local_rank)+'caption_translate.json')
        # jsObj = json.dumps(outs)
        # with open(out_path, "w") as f:
        #     f.write(jsObj)
        #     f.close()
        #
        # if lang1=='coco':
        #     coco = COCO(os.path.join(params.data_path, pre_path, 'captions_val2014.json'))
        # else:
        #     coco = COCO(os.path.join(params.data_path, pre_path, 'evaluate_flicker.json'))
        # cocoRes = coco.loadRes(out_path)
        #
        # # fp32 beam 1
        # cocoEval = COCOEvalCap(coco, cocoRes)
        # cocoEval.params['image_id'] = cocoRes.getImgIds()
        # # evaluate results
        # cocoEval.evaluate()
        #
        # coco_eval_rpt = cocoEval.eval
        # coco_methods = params.coco_method.split(',')
        # for method in coco_methods:
        #     if lang1=='coco':
        #         scores['%s_coco_' % data_set + method] = coco_eval_rpt[method]
        #     else:
        #         scores['%s_flicker_' % data_set + method] = coco_eval_rpt[method]



