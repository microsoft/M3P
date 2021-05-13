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
import pandas as pd
from tqdm import tqdm
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from ..utils import to_cuda, restore_segmentation, concat_batches
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from ..dataset_utils import get_loader
from ..tokenization import XLMRTokenizer
BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)

logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.params = params

        # create directory to store hypotheses, and reference files for BLEU evaluation
        # if self.params.is_master:
        #     params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
        #     subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
        #     self.create_reference_files()

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

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

            for data_set in ['valid']:
                if params.is_master == False:
                    break
                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                # for lang1, lang2 in params.mlm_steps:
                #     self.evaluate_mlm(scores, data_set, lang1, lang2)

                for lang in params.mass_steps:
                    self.evaluate_mass(scores, data_set, lang)

                mass_steps = []
                for lang1 in params.mass_steps:
                    for lang2 in params.mass_steps:
                        if lang1 != lang2:
                            mass_steps.append((lang1, lang2))
                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps] + mass_steps):
                    self.evaluate_mt(scores, data_set, lang1, lang2, False)


                for lang1,lang2 in params.text_steps:
                    if params.is_ntg:
                        self.evaluate_ntg(scores, data_set, lang1, None, False)

                # multi-modal translation task (evaluate perplexity and accuracy)
                if params.is_generation:
                    for lang1, lang2 in set(params.cross_modal_steps):
                        if params.is_mt:
                            self.evaluate_mt_ic(scores,data_set,lang1,lang2,False)
                        else:
                            self.evaluate_ic(scores, data_set, lang1, lang2, False)

                    # # for multi-modal pretraining tasks
                    # for lang1, lang2 in set(params.cross_mass_steps):
                    #     # eval_bleu = params.eval_bleu and params.is_master
                    #     self.evaluate_imlm(scores, data_set, lang1, lang2, False)
                    #
                    # for lang1, lang2 in set(params.cross_ae_steps):
                    #     # eval_bleu = params.eval_bleu and params.is_master
                    #     self.evaluate_ida(scores, data_set, lang1, lang2, False)
                    #     # self.evaluate_cross_img2img_step(scores, data_set, lang1, lang2)
                if params.is_understanding:
                    if params.is_slide and params.is_master:
                        for lang1,lang2 in set(params.cross_rel_steps):
                            self.evaluate_slide(scores, data_set,lang1,lang2)
                    else:
                        # for understanding
                        if params.is_master:
                            for lang1, lang2 in set(params.cross_rel_steps):
                                self.evaluate_understanding_tasks(scores, data_set, lang1, lang2)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean(
                        [scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean(
                        [scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                # _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                # if len(_mlm_mono) > 0:
                #     scores['%s_mlm_ppl' % data_set] = np.mean(
                #         [scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                #     scores['%s_mlm_acc' % data_set] = np.mean(
                #         [scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

                _mass_step = [l1 for l1 in params.mass_steps]
                if len(_mass_step) > 0:
                    scores['%s_mass_ppl' % data_set] = np.mean(
                        [scores['%s_%s-%s_mass_ppl' % (data_set, lang1, lang1)] for lang1 in _mass_step])
                    scores['%s_mass_acc' % data_set] = np.mean(
                        [scores['%s_%s-%s_mass_acc' % (data_set, lang1, lang1)] for lang1 in _mass_step])

                _cross_modal_step = [(l1, l2) for (l1, l2) in params.cross_modal_steps]

                # cross_mass_step = [l1 for l1 in params.cross_mass_steps]
                # if len(cross_mass_step) > 0:
                #     scores['%s_IMLM_ppl' % data_set] = np.mean(
                #         [scores['%s_%s-%s_IMLM_ppl' % (data_set, lang1, lang2)] for (lang1, lang2) in cross_mass_step])
                #     scores['%s_IMLM_acc' % data_set] = np.mean(
                #         [scores['%s_%s-%s_IMLM_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in cross_mass_step])
                #
                # cross_ae_step = [l1 for l1 in params.cross_ae_steps]
                # if len(cross_ae_step) > 0:
                #     scores['%s_IDA_ppl' % data_set] = np.mean(
                #         [scores['%s_%s-%s_IDA_ppl' % (data_set, lang1, lang2)] for (lang1, lang2) in cross_ae_step])
                #     scores['%s_IDA_acc' % data_set] = np.mean(
                #         [scores['%s_%s-%s_IDA_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in cross_ae_step])
                    # scores['%s_cross_img2img_acc' % data_set] = np.mean(
                    #     [scores['%s_%s-%s_cross_modal_img2img_acc' % (data_set, lang1,lang2)] for (lang1,lang2) in cross_ae_step])

                # undestanding metric

                cross_rel_steps = [l1 for l1 in params.cross_rel_steps]
                if len(cross_rel_steps) > 0 and params.is_slide==False:
                    if params.t2i_flag:
                        scores['%s_I2T_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_rel_i2t_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])
                    if params.i2t_flag:
                        scores['%s_T2I_acc' % data_set] = np.mean(
                            [scores['%s_%s-%s_rel_t2i_acc' % (data_set, lang1, lang2)] for (lang1, lang2) in
                             cross_rel_steps])


            if params.is_generation:
                for lang1, lang2 in set(params.cross_modal_steps):
                    if params.is_master:
                        if lang1=='coco' or lang1=='flicker' or lang1=='mild':
                            if params.is_mt:
                                self.evaluate_mt_image_caption(scores,'test',lang1,lang2)
                            else:
                                self.evaluate_image_caption(scores, 'test', lang1, lang2)

                for lang1,lang2 in params.text_steps:
                    if params.is_ntg and  params.is_master:
                        self.evaluate_ntg_generate(scores, 'test', lang1)

            if params.is_understanding:
                for lang1, lang2 in set(params.cross_rel_steps):  # support
                    data_set = 'test'
                    if lang1=='coco' or lang1=='flicker':
                        # support multi-gpu evaluation
                        for lg in params.ft_lgs:
                            t2i_r1,t2i_r5,t2i_r10,i2t_r1, i2t_r5, i2t_r10 = self.evaluate_image_retrieval(scores, data_set, lang1,
                                                                                                             lang2,lg)
                            with open(os.path.join(params.eval_path, "inference.log"), "a") as f:
                                f.write(" ".join([str(i2t_r1), str(i2t_r5), str(i2t_r10)]) + "\n")

                            scores['%s_%s_%s_t2i_R1' % (lg,data_set, lang1)] = t2i_r1
                            scores['%s_%s_%s_t2i_R5' % (lg,data_set, lang1)] = t2i_r5
                            scores['%s_%s_%s_t2i_R10' % (lg,data_set, lang1)] = t2i_r10
                            scores['%s_%s_%s_i2t_R1' % (lg,data_set, lang1)] = i2t_r1
                            scores['%s_%s_%s_i2t_R5' % (lg,data_set, lang1)] = i2t_r5
                            scores['%s_%s_%s_i2t_R10' % (lg,data_set, lang1)] = i2t_r10
                            scores['%s_%s_%s_Mean_Recall' % (lg,data_set, lang1)] = (t2i_r1+t2i_r5+t2i_r10+i2t_r1+i2t_r5+i2t_r10)/6.0
                    elif lang1=='mild':
                        # support multi-gpu evaluation
                        for lg in params.ft_lgs:
                            t2i_r1, t2i_r5, t2i_r10, i2t_r1, i2t_r5, i2t_r10 = self.evaluate_image_retrieval(scores,
                                                                                                             data_set,
                                                                                                             lang1,
                                                                                                             lang2, lg,seq_per_img=1)
                            with open(os.path.join(params.eval_path, "inference.log"), "a") as f:
                                f.write(" ".join([str(i2t_r1), str(i2t_r5), str(i2t_r10)]) + "\n")
                    elif lang1=='slide':
                        if params.is_master:
                            self.evaluate_slide(scores, data_set, lang1, lang2)

        return scores

    def get_cross_lingual_iterator(self, data_set, lang1, lang2=None, stream=False, is_cross=False,n_sentences=300):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        # assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages


        subsample = 10
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                if self.params.is_ntg:
                    if data_set=='test':
                        iterator = self.data['text'][lang1][data_set].get_iterator(
                            shuffle=False,
                            group_by_size=True,
                            n_sentences=n_sentences,
                            return_indices=True)
                    else:
                        iterator = self.data['text'][lang1][data_set].get_iterator(
                            shuffle=False,
                            group_by_size=True,
                            n_sentences=n_sentences)
                else:
                    iterator = self.data['mono'][lang1][data_set].get_iterator(
                        shuffle=False,
                        group_by_size=True,
                        n_sentences=n_sentences,
                    )
        else:
                _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
                iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences
                )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 or is_cross else batch[::-1]

    def evaluate_clm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_cross_lingual_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

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
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('crossfwd', stream_='text', x=x, lengths=lengths, positions=positions, langs=langs,
                           causal=False)
            #tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, lang1) if lang2 is None else '%s_%s-%s_clm_ppl' % (
            data_set, lang1, lang2)
        acc_name = '%s_%s_clm_acc' % (data_set, lang1) if lang2 is None else '%s_%s-%s_clm_acc' % (
            data_set, lang1, lang2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

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

        for batch in self.get_cross_lingual_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

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

            tensor = model('crossfwd', stream_='text', x=x, lengths=lengths, positions=positions, langs=langs,
                           causal=False)
            # forward / loss
            #tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
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


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class XEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model
        # self.dico = params.dico
        self.tokenizer = XLMRTokenizer(params.vocab_path)

    def get_iterator(self, data_set, lang1, lang2=None, lg=None):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs or lang1 == 'img' or lang2 == 'img'
        assert lang2 is None or lang2 in self.params.langs or lang1 == 'img' or lang2 == 'img'

        if lg is not None:
            dataset = self.data['cross_modal'][(lang1, lang2)][data_set][lg]
        else:
            dataset = self.data['cross_modal'][(lang1, lang2)][data_set]

        eval_loader = get_loader(self.params, dataset, lang1,data_set)

        n_sentences = self.params.eval_n

        for batch_idx, batch in enumerate(eval_loader):
            if data_set=='valid' and batch_idx>n_sentences:
                break
            yield batch# test for over all


    def evaluate_mass(self, scores, data_set, lang):
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang in params.langs

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        params = params
        lang_id = params.lang2id[lang]

        n_words = 0
        xe_loss = 0
        n_valid = 0
        for (x1, len1) in self.get_iterator(data_set, lang):
            (x1, len1, x2, len2, y, pred_mask, positions) = self.mask_sent(x1, len1, rng)

            langs1 = x1.clone().fill_(lang_id)
            langs2 = x2.clone().fill_(lang_id)

            # cuda
            x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)

            # encode source sentence
            enc1 = model('crossfwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            enc_mask = x1.ne(params.mask_index)
            enc_mask = enc_mask.transpose(0, 1)
            # decode target sentence
            dec2 = model('crossfwd', x=x2, lengths=len2,
                         langs=langs2, causal=True,
                         src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)
            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mass_ppl' % (data_set, lang, lang)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mass_acc' % (data_set, lang, lang)] = 100. * n_valid / n_words

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

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = model('crossfwd', stream_='text', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            # decode target sentence
            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1,
                         src_len=len1)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = model.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = model.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu

    def evaluate_ic(self, scores, data_set, lang1, lang2, eval_bleu):
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

        for batch in self.get_iterator(data_set, lang1, lang2):
            (x2, len2,_), (x1, x1_mask, img_loc, img_id) = batch
            #             if params.fp16:
            #                 x1 = x1.to(torch.float16)
            #                 img_loc = img_loc.to(torch.float16)

            if len(params.ft_lgs) > 0:
                lang1_id = params.lang2id[params.ft_lgs[0]]
                langs = x2.clone().fill_(lang1_id)
            else:
                lang1_id = params.lang2id['en']
                langs = x2.clone().fill_(lang1_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            #
            len1 = x1_mask.sum(dim=1)
            x1 = x1.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            if len(params.ft_lgs) > 0:
                lang1_id = params.lang2id[params.ft_lgs[0]]
                langs_img = x1_mask.transpose(0, 1).clone().fill_(lang1_id)
            else:
                lang1_id = params.lang2id['en']
                langs_img = x1_mask.transpose(0, 1).clone().fill_(lang1_id)

            # cuda
            x1, len1, img_loc, x2, len2, y, x1_mask,langs,langs_img = to_cuda(x1, len1, img_loc, x2, len2, y, x1_mask,langs,langs_img)

            # encode source sentence
            enc1 = model('crossfwd', stream_='img', x=x1, lengths=len1, langs=langs_img, causal=False, cross_modal=True,
                         image_loc=img_loc, image_dist=None)
            enc1 = enc1.transpose(0, 1)

            # decode target sentence
            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs, causal=True, src_enc=enc1,
                         src_len=len1)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = model.generate(enc1, len1, None, max_len=max_len)
                else:
                    generated, lengths = model.generate_beam(
                        enc1, len1, None, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

            # compute perplexity and prediction accuracy
        scores['%s_%s-%s_IC_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_IC_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("IC BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_cross_modal_bleu' % (data_set, lang1, lang2)] = bleu

    def evaluate_mt_ic(self, scores, data_set, lang1, lang2, eval_bleu):
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

        for batch in self.get_iterator(data_set, lang1, lang2):
            (x_src,src_len,_),(x2, len2,_), (x1, x1_mask, img_loc, img_id) = batch
            #             if params.fp16:
            #                 x1 = x1.to(torch.float16)
            #                 img_loc = img_loc.to(torch.float16)

            lang0_id = params.lang2id[params.ft_lgs[0]]
            lang_src = x_src.clone().fill_(lang0_id)
            lang1_id = params.lang2id[params.ft_lgs[1]]
            langs = x2.clone().fill_(lang1_id)

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
            x1, len1, img_loc, x2, len2, y, x1_mask,langs,lang_src,x_src,src_len = to_cuda(x1, len1, img_loc, x2, len2, y, x1_mask,langs,lang_src,x_src,src_len)

            # encode source sentence
            if params.mt_only_text:
                encoder_outputs = model('crossfwd', stream_='text', x=x_src, lengths=src_len, langs=lang_src,
                                        causal=False, refine_image=params.refine_image)
                len_all = src_len
            else:
                encoder_outputs = model('jointfwd', x=x_src, lengths=src_len, x_img=x1, lengths_img=len1, causal=False,
                                        langs=None,
                                        image_loc=img_loc, refine_image=params.refine_image)

                len_all = len1 + src_len

            enc1 = encoder_outputs.transpose(0, 1)

            # decode target sentence
            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs, causal=True, src_enc=enc1,
                         src_len=len_all)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = model.generate(enc1, len1, None, max_len=max_len)
                else:
                    generated, lengths = model.generate_beam(
                        enc1, len1, None, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

            # compute perplexity and prediction accuracy
        scores['%s_%s-%s_IC_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_IC_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("IC BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_cross_modal_bleu' % (data_set, lang1, lang2)] = bleu

    def evaluate_mt_image_caption(self, scores, data_set, lang1, lang2):
        import json
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        if data_set == 'valid':
            split = 'val'
        else:
            split = 'test'

        total_len = len(self.data['cross_modal'][(lang1, lang2)][data_set].image_ids)

        outs = []
        tgt_lg = params.ft_lgs[1] #generate
        tgt_lang_id = params.lang2id[tgt_lg]
        for batch in tqdm(self.get_iterator(data_set, lang1, lang2), leave=False, total=total_len):
            (x_src,src_len,_),x1, x1_mask, img_loc, img_id = batch

            len1 = x1_mask.sum(dim=1)
            x1 = x1.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            # langs = x1_mask.transpose(0, 1).clone().fill_(tgt_lang_id)

            if params.mt_only_text:
                lang0_id = params.lang2id[params.ft_lgs[0]]
                lang_src = x_src.clone().fill_(lang0_id)

                encoder_outputs = model('crossfwd', stream_='text', x=x_src.cuda(), lengths=src_len.cuda(), langs=lang_src.cuda(),
                                        causal=False, refine_image=params.refine_image)
                len_all = src_len
            else:

                encoder_outputs = model('jointfwd', x=x_src.cuda(), lengths=src_len.cuda(), x_img=x1.cuda(), lengths_img=len1.cuda(), causal=False,
                                        langs=None,
                                        image_loc=img_loc.cuda(), refine_image=params.refine_image)
                len_all = src_len + len1
            # encoded = encoder_outputs[x1.shape[0]:]  # only text part

            encoded = encoder_outputs.transpose(0, 1)

            if data_set == 'test':
                decoded, dec_lengths = model.generate_beam(
                    encoded, len_all.cuda(), tgt_lang_id, beam_size=params.beam_size,
                    length_penalty=1,
                    early_stopping=True,
                    max_len=int(1.5 * len_all.max().item() + 10))
            else:
                decoded, dec_lengths = model.generate(encoded, len1.cuda(), tgt_lang_id,
                                                      max_len=int(1.5 * len_all.max().item() + 10))


            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self.params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
                # output translation
                # source = src_sent[i + j].strip()
                target = [s.item()for s in sent]
                target = self.tokenizer.decode(target)
                # import re
                # target = target.replace('@@ ', '')
                cur_dict = {}
                cur_dict['caption'] = target
                cur_dict['image_id'] = img_id[0]
                outs.append(cur_dict)

        if lang1 == 'coco':
            ori_path = os.path.join(params.data_path, 'uvl_captions', 'coco.ids.pkl')
            file2imgid = pd.read_pickle(ori_path)
            for line in outs:
                line['image_id'] = file2imgid[line['image_id']]
        elif lang1 == 'flicker':
            ori_path = os.path.join(params.data_path, 'uvl_captions', 'flicker.ids.pkl')
            file2imgid = pd.read_pickle(ori_path)
            for line in outs:
                line['image_id'] = file2imgid[line['image_id']]

        #evaluate tgt lg
        out_path = os.path.join(params.eval_path,
                                "epoch_%s_%s" % (str(scores['epoch']), lang1) + '_caption_translate.%s.json' % (tgt_lg))
        jsObj = json.dumps(outs)

        with open(out_path, "w") as f:
            f.write(jsObj)
            f.close()

        if lang1 == 'mild': return

        if lang1 == 'coco':
            coco = COCO(os.path.join(params.data_path, 'uvl_captions', 'evaluate_coco.mt.%s.json' % (tgt_lg)))
        else:
            coco = COCO(os.path.join(params.data_path, 'uvl_captions', 'evaluate_flicker.mt.%s.json' % (tgt_lg)))

        cocoRes = coco.loadRes(out_path)

        # # fp32 beam 1
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        # evaluate results
        cocoEval.evaluate()
        coco_eval_rpt = cocoEval.eval
        coco_methods = params.coco_method.split(',')
        for method in coco_methods:
            if lang1 == 'coco':
                scores['%s_coco_%s_' % (data_set, tgt_lg) + method] = coco_eval_rpt[method]
            else:
                scores['%s_flicker_%s_' % (data_set, tgt_lg) + method] = coco_eval_rpt[method]

    def evaluate_image_caption(self, scores, data_set, lang1, lang2):
        import json
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        if data_set == 'valid':
            split = 'val'
        else:
            split = 'test'

        total_len = len(self.data['cross_modal'][(lang1, lang2)][data_set].image_ids)

        outs = []
        tgt_lang_id = params.lang2id[params.ft_lgs[0]]
        for batch in tqdm(self.get_iterator(data_set, lang1, lang2), leave=False, total=total_len):
            x1, x1_mask, img_loc, img_id = batch

            len1 = x1_mask.sum(dim=1)
            x1 = x1.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            langs = x1_mask.transpose(0, 1).clone().fill_(tgt_lang_id)

            encoded = model('crossfwd', stream_='img', x=x1.cuda(), lengths=len1.cuda(), langs=langs.cuda(), causal=False,
                            cross_modal=True, refine_image=params.refine_image,
                            image_loc=img_loc.cuda(), image_dist=None)
            encoded = encoded.transpose(0, 1)

            if data_set == 'test':
                decoded, dec_lengths = model.generate_beam(
                    encoded, len1.cuda(), tgt_lang_id, beam_size=params.beam_size,
                    length_penalty=1,
                    early_stopping=True,
                    max_len=int(1.5 * len1.max().item() + 10))
            else:
                decoded, dec_lengths = model.generate(encoded, len1.cuda(), tgt_lang_id,
                                                      max_len=int(1.5 * len1.max().item() + 10))


            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self.params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
                # output translation
                # source = src_sent[i + j].strip()
                target = [s.item()for s in sent]
                target = self.tokenizer.decode(target)
                # import re
                # target = target.replace('@@ ', '')
                cur_dict = {}
                cur_dict['caption'] = target
                cur_dict['image_id'] = img_id[0]
                outs.append(cur_dict)

        if lang1 == 'coco':
            ori_path = os.path.join(params.data_path, 'uvl_captions', 'coco.ids.pkl')
            file2imgid = pd.read_pickle(ori_path)
            for line in outs:
                line['image_id'] = file2imgid[line['image_id']]
        elif lang1 == 'flicker':
            ori_path = os.path.join(params.data_path, 'uvl_captions', 'flicker.ids.pkl')
            file2imgid = pd.read_pickle(ori_path)
            for line in outs:
                line['image_id'] = file2imgid[line['image_id']]

        ft_lg = self.params.ft_lgs[0]

        out_path = os.path.join(params.eval_path,
                                "epoch_%s_%s" % (str(scores['epoch']), lang1) + '_caption_translate.%s.json' % (ft_lg))
        jsObj = json.dumps(outs)

        with open(out_path, "w") as f:
            f.write(jsObj)
            f.close()

        if lang1 == 'mild': return

        if lang1 == 'coco':
            coco = COCO(os.path.join(params.data_path, 'uvl_captions', 'evaluate_coco.%s.json' % (ft_lg)))
        else:
            coco = COCO(os.path.join(params.data_path, 'uvl_captions', 'evaluate_flicker.%s.json' % (ft_lg)))

        cocoRes = coco.loadRes(out_path)

        # # fp32 beam 1
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        # evaluate results
        cocoEval.evaluate()
        coco_eval_rpt = cocoEval.eval
        coco_methods = params.coco_method.split(',')
        for method in coco_methods:
            if lang1 == 'coco':
                scores['%s_coco_%s_' % (data_set, ft_lg) + method] = coco_eval_rpt[method]
            else:
                scores['%s_flicker_%s_' % (data_set, ft_lg) + method] = coco_eval_rpt[method]

    def evaluate_ntg(self, scores, data_set, lang1, lang2=None, eval_bleu=False):
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        xe_loss = 0
        n_valid = 0

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang1]
        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        if data_set=='valid':
            _n_sentences=300
        else:
            _n_sentences =-1
        for batch in self.get_cross_lingual_iterator(data_set, lang1,n_sentences=_n_sentences):
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = model('crossfwd', stream_='text', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            # decode target sentence
            dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1,
                         src_len=len1)

            # loss
            word_scores, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()


            # compute perplexity and prediction accuracy
        scores['%s_%s_NTG_ppl' % (data_set, lang1)] = np.exp(xe_loss / n_words)
        scores['%s_%s_NTG_acc' % (data_set, lang1)] = 100. * n_valid / n_words


    def evaluate_ntg_generate(self, scores, data_set, lang1, lang2=None, eval_bleu=False):
        import json
        params = self.params

        assert data_set in ['valid', 'test']

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang1]
        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        if data_set=='valid':
            _n_sentences=300
        else:
            _n_sentences =self.params.eval_n

        outs = []
        _out_indices = []
        tgt_lg = lang1  # generate
        tgt_lang_id = params.lang2id[tgt_lg]
        for batch in self.get_cross_lingual_iterator(data_set, lang1,n_sentences=_n_sentences):
            (x1, len1), (x2, len2), _indices= batch

            _out_indices.extend(_indices.tolist())

            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = model('crossfwd', stream_='text', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            if data_set == 'test':
                decoded, dec_lengths = model.generate_beam(
                    enc1, len1.cuda(), tgt_lang_id, beam_size=params.beam_size,
                    length_penalty=1,
                    early_stopping=True,
                    max_len=int(1.5 * len1.max().item() + 10))
            else:
                decoded, dec_lengths = model.generate(enc1, len1.cuda(), tgt_lang_id,
                                                      max_len=int(1.5 * len1.max().item() + 10))

            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self.params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
                # output translation
                # source = src_sent[i + j].strip()
                target = [s.item() for s in sent]
                target = self.tokenizer.decode(target)
                # import re
                # target = target.replace('@@ ', '')
                outs.append(target)

        out_path = os.path.join(params.eval_path,
                                "epoch_%s_%s" % (str(scores['epoch']), lang1) + "rank_" + str(
                                    params.local_rank) + '_ntg_.%s.json' % (lang1))
        jsObj = json.dumps(outs)

        with open(out_path, "w") as f:
            f.write(jsObj)
            f.close()

        #write down ref
        np.save(os.path.join(params.eval_path,
                                "epoch_%s_%s" % (str(scores['epoch']), lang1) + "rank_" + str(
                                    params.local_rank) + '_ntg_.%s' % (lang1) + ".npy"),np.array(_out_indices))

    def evaluate_understanding_tasks(self,scores, data_set, lang1, lang2):
        assert data_set in ['valid', 'test']

        t2i_acc,i2t_acc,mlm_acc,mrm_acc = 0,0,0,0
        mlm_loss,mrm_loss =0,0
        t2i_n,i2t_n,mlm_n,mrm_n = 0,0,0,0
        for (t2i_batch, i2t_batch) in self.get_iterator(data_set, lang1, lang2):
            _acc,_n = self.evaluate_t2i(t2i_batch)
            t2i_acc += _acc
            t2i_n += _n
            _acc,_n = self.evaluate_i2t(i2t_batch)
            i2t_acc+=_acc
            i2t_n+=_n
        #     if self.params.is_pretrain:
        #         if self.params.t2i_flag:
        #             n_valid, xe_loss, n_words = self.evaluate_cmlm(t2i_batch)
        #             mlm_acc+=n_valid
        #             mlm_loss+=xe_loss
        #             mlm_n+=n_words
        #             n_valid, xe_loss, n_words = self.evaluate_mrm(t2i_batch)
        #             mrm_acc += n_valid
        #             mrm_loss += xe_loss
        #             mrm_n += n_words
        #         if self.params.i2t_flag:
        #             n_valid, xe_loss, n_words = self.evaluate_cmlm(i2t_batch)
        #             mlm_acc+=n_valid
        #             mlm_loss+=xe_loss
        #             mlm_n+=n_words
        #             n_valid, xe_loss, n_words = self.evaluate_mrm(i2t_batch)
        #             mrm_acc += n_valid
        #             mrm_loss += xe_loss
        #             mrm_n += n_words
        #
        #         # compute perplexity and prediction accuracy
        # if self.params.is_pretrain:
        #     scores['%s_%s-%s_cmlm_ppl' % (data_set, lang1, lang2)] = np.exp(
        #         mlm_loss / mlm_n) if mlm_n > 0 else 1e9
        #     scores['%s_%s-%s_cmlm_acc' % (data_set, lang1, lang2)] = 100. * mlm_acc / mlm_n if mlm_n > 0 else 0.
        #
        #     # compute perplexity and prediction accuracy
        #     scores['%s_%s-%s_mrm_ppl' % (data_set, lang1, lang2)] = np.exp(
        #         mrm_loss / mrm_n) if mrm_n > 0 else 1e9
        #     scores['%s_%s-%s_mrm_acc' % (data_set, lang1, lang2)] = 100. * mrm_acc / mrm_n if mrm_n > 0 else 0.

        scores['%s_%s-%s_rel_t2i_acc' % (data_set, lang1, lang2)] = 100. * t2i_acc / t2i_n
        scores['%s_%s-%s_rel_i2t_acc' % (data_set, lang1, lang2)] = 100. * i2t_acc / i2t_n

    def evaluate_t2i(self,_batch):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        if params.is_pretrain:
            (x1, len1, x1_labels), (img, img_mask, img_loc, obj_labels, pos_labels, img_ori, img_ids) = _batch
            langs = torch.LongTensor(
                [[params.n_langs] * params.max_region_num + [params.lang2id['en']] * len1.max().item()] *
                x1.size()[1]) if params.n_langs > 1 else None
        else:
            (x1, len1,lang_p), (img, img_mask, img_loc, pos_labels, img_ids) = _batch
            lang_p = lang_p.transpose(0, 1)
            lang_img = torch.LongTensor([[params.n_langs] * params.max_region_num] * x1.size()[1])
            langs = torch.cat([lang_img, lang_p], dim=1)  # [img. img_id...sent, sent_id..]
        # (x1, len1), (img, img_mask, img_loc, obj_labels, pos_labels, img_ids) = t2i_batch


        # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        x1, len1, langs, x_img, img_loc, img_len = to_cuda(x1, len1, langs, x_img, img_loc, img_len)

        if params.is_latent:
            encoder_outputs, original_text, original_img, text_kld, img_kld = model('jointfwd', x=x1, lengths=len1,
                                                                                    x_img=x_img, lengths_img=img_len,
                                                                                    causal=False,
                                                                                    langs=langs,
                                                                                    image_loc=img_loc,
                                                                                    refine_image=params.refine_image,
                                                                                    is_latent=True)
        else:
            encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                    langs=langs,
                                    image_loc=img_loc, refine_image=params.refine_image)

        encoder_outputs = encoder_outputs.transpose(0, 1)

        relation_scores = model('predict', tensor=encoder_outputs, is_relation=True)

        _labels = np.array(pos_labels)
        matching_label = torch.from_numpy(_labels)
        pred_label = relation_scores.view(-1, params.sample_n).cpu().max(1)[1]
        _acc = (pred_label == matching_label).sum().item()

        return _acc,len(matching_label)

    def evaluate_i2t(self, _batch):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_valid_ = 0
        if params.is_pretrain:
            (x1, len1, x1_labels), (x2, len2), (clcm_labels, img, img_mask, img_loc, obj_labels, pos_labels, img_ori, img_ids) = _batch
            langs = torch.LongTensor(
                [[params.n_langs] * params.max_region_num + [params.lang2id['en']] * len1.max().item()] *
                x1.size()[1]) if params.n_langs > 1 else None
        else:
            (x1, len1,lang_p), (img, img_mask, img_loc, pos_labels, img_ids) = _batch
            # follow uvp
            lang_p = lang_p.transpose(0, 1)
            lang_img = torch.LongTensor([[params.n_langs] * params.max_region_num] * x1.size()[1])
            langs = torch.cat([lang_img, lang_p], dim=1)  # [img. img_id...sent, sent_id..]

        # (x1, len1), (img, img_mask, img_loc, obj_labels, pos_labels, img_ids) = i2t_batch

        # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        x1, len1, langs, x_img, img_loc, img_len = to_cuda(x1, len1, langs, x_img, img_loc, img_len)

        if params.is_latent:
            encoder_outputs, original_text, original_img, text_kld, img_kld = model('jointfwd', x=x1, lengths=len1,
                                                                                    x_img=x_img, lengths_img=img_len,
                                                                                    causal=False,
                                                                                    langs=langs,
                                                                                    image_loc=img_loc,
                                                                                    refine_image=params.refine_image,
                                                                                    is_latent=True)
        else:
            encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                    langs=langs,
                                    image_loc=img_loc, refine_image=params.refine_image)

        encoder_outputs = encoder_outputs.transpose(0, 1)

        relation_scores = model('predict', tensor=encoder_outputs, is_relation=True)

        _labels = np.array(pos_labels)
        matching_label = torch.from_numpy(_labels)
        pred_label = relation_scores.view(-1, params.sample_n).cpu().max(1)[1]
        _acc = (pred_label == matching_label).sum().item()

        n_valid_ += _acc
        n_words += len(matching_label)

        return n_valid_,n_words

    def get_mask_(self, x, _labels):
        slen, bs = x.size()[0], x.size()[1]
        pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
        pred_mask = pred_mask.view(slen, bs)
        pred_mask[_labels != -1] = 1
        y = _labels[_labels > 0]
        return y, pred_mask.bool()

    def evaluate_cmlm(self, batch):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_valid = 0
        xe_loss = 0
        rng = np.random.RandomState(0)

        (x1, len1, x1_labels), (img, img_mask, img_loc, obj_labels, pos_labels, img_ori, img_ids) = batch

        langs = torch.LongTensor(
            [[params.n_langs] * params.max_region_num + [params.lang2id['en']] * len1.max().item()] *
            x1.size()[1]) if params.n_langs > 1 else None
        # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        # x1, y, pred_mask = self.mask_out(x1, len1, rng)
        y, pred_mask = self.get_mask_(x1, x1_labels)
        # cuda
        x1, len1, langs, x_img, img_loc, img_len, pred_mask, y = to_cuda(x1, len1, langs, x_img, img_loc, img_len,
                                                                         pred_mask, y)

        encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                langs=langs,
                                image_loc=img_loc, refine_image=params.refine_image)

        encoder_outputs = encoder_outputs[x_img.shape[0]:]  # only text part

        try:
            word_scores, loss = model('predict', tensor=encoder_outputs, pred_mask=pred_mask, y=y, get_scores=True)
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
        except:
            # update stats
            n_words += len(y)
            xe_loss += 2.0
            n_valid += 0.5

        return n_valid,xe_loss,n_words

    def evaluate_mrm(self, batch):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        n_words = 0
        n_valid = 0
        xe_loss = 0
        rng = np.random.RandomState(0)

        (x1, len1, x1_labels), (img, img_mask, img_loc, obj_labels, pos_labels, img_ori, img_ids) = batch

        langs = torch.LongTensor(
            [[params.n_langs] * params.max_region_num + [params.lang2id['en']] * len1.max().item()] *
            x1.size()[1]) if params.n_langs > 1 else None
        # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        y, pred_mask = self.get_mask_(img, obj_labels)

        # cuda
        x1, len1, langs, x_img, img_loc, img_len, pred_mask, y = to_cuda(x1, len1, langs, x_img, img_loc, img_len,
                                                                         pred_mask, y)

        encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                langs=langs,
                                image_loc=img_loc, refine_image=params.refine_image)

        encoder_outputs = encoder_outputs[:x_img.shape[0]]  # only text part

        encoder_outputs = encoder_outputs.transpose(0, 1)

        try:
            obj_scores, loss = model('predict', tensor=encoder_outputs, pred_mask=pred_mask, y=obj_labels.cuda().view(-1), get_scores=True,
                                     is_obj=True)

            # update stats
            n_words += len(len1)
            xe_loss += loss.item() * len(len1)
            n_valid += (obj_scores.max(1)[1] == obj_labels.cuda().view(-1)).sum().item()
        except:
            # update stats
            n_words += len(y)
            xe_loss += 2.0
            n_valid += 0.5

        return n_valid,xe_loss,n_words

    def evaluate_image_retrieval(self, scores, data_set, lang1, lang2,lg='en',seq_per_img=5):
        params = self.params
        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        assert data_set in ['test']
        # get set

        _dataset = self.data['cross_modal'][(lang1, lang2)][data_set][lg]

        total_img_len = len(_dataset)  # 5000
        total_len = total_img_len * seq_per_img  # 25000

        test_splits = params.test_splits #the number of captions for each images during retrievaled
        split_len = total_len // test_splits

        all_matching_labels = []
        all_matching_scores = []
        t2i_r1 = 0
        t2i_r5 = 0
        t2i_r10 = 0
        # img_input = test_loader.dataset.all_test_obj_cache.unsqueeze(0).repeat(opt.test_bsz, 1, 1, 1)
        # box_coords = test_loader.dataset.all_test_box_cache.unsqueeze(0).repeat(opt.test_bsz, 1, 1, 1)

        print("\ntest image retrieval (caption to image)")
        t = tqdm(total=total_img_len, initial=0, leave=False)

        for _, batch in tqdm(enumerate(self.get_iterator(data_set, lang1, lang2,lg))):

            concat_input_ids,concat_input_lengths,concat_segment_ids, img_input, box_coords, pos_cap_label = batch

            if params.local_rank != -1:
                total_len = concat_input_ids.size(1)  # 25000
                split_len = concat_input_ids.size(1) // test_splits  # 5000
            img_input = img_input.repeat(1, split_len, 1, 1)
            box_coords = box_coords.repeat(1, split_len, 1, 1)
            img_len = torch.from_numpy(np.ones((img_input.size()[0])) * params.max_region_num).long()

            concat_input_ids, concat_input_lengths, concat_segment_ids, img_input, box_coords,img_len = to_cuda(
                concat_input_ids, concat_input_lengths, concat_segment_ids, img_input, box_coords,img_len)

            img_input = img_input.reshape(-1, img_input.size(-2), img_input.size(-1))
            box_coords = box_coords.reshape(-1, box_coords.size(-2), box_coords.size(-1))
            with torch.no_grad():
                splits = []
                # img_input_splits = torch.split(img_input, split_num, dim=1)
                # box_coords_input_splits = torch.split(box_coords, split_num, dim=1)
                concat_input_ids_splits = torch.split(concat_input_ids, split_len, dim=1)
                concat_segment_ids_splits = torch.split(concat_segment_ids, split_len, dim=1)
                concat_input_lengths_splits = torch.split(concat_input_lengths, split_len, dim=1)
                #calc all captions for these images
                for i in range(total_len // split_len):
                    concat_input_ids_split = concat_input_ids_splits[i]
                    concat_input_lengths_split = concat_input_lengths_splits[i]
                    concat_segment_ids_split = concat_segment_ids_splits[i]

                    concat_input_ids_split = concat_input_ids_split.reshape(-1, concat_input_ids_split.size()[-1])
                    concat_input_lengths_split = concat_input_lengths_split.reshape(-1, )
                    concat_segment_ids_split = concat_segment_ids_split.reshape(-1, concat_segment_ids_split.size()[-1])

                    if params.is_latent:
                        encoder_outputs, _, _, text_kld, img_kld = model('jointfwd',x=concat_input_ids_split.transpose(0,1), lengths=concat_input_lengths_split, x_img=img_input.transpose(0,1), lengths_img=img_len,
                                            causal=False,
                                            langs=concat_segment_ids_split,
                                            image_loc=box_coords.transpose(0,1), refine_image=params.refine_image,
                                                                         is_latent=True)
                    else:
                        encoder_outputs = model('jointfwd', x=concat_input_ids_split.transpose(0,1), lengths=concat_input_lengths_split, x_img=img_input.transpose(0,1), lengths_img=img_len,
                                                causal=False,
                                                langs=concat_segment_ids_split,
                                                image_loc=box_coords.transpose(0,1), refine_image=params.refine_image)

                    encoder_outputs = encoder_outputs.transpose(0, 1)

                    matching_scores_split = model('predict', tensor=encoder_outputs, is_relation=True)

                    splits.append(matching_scores_split.view(-1, split_len))
                matching_scores = torch.cat(splits, dim=-1)  # 8 * 1000  // 8 * 5000

            # pos_cap_label = torch.stack(pos_cap_label, dim=0)
            all_matching_labels.append(pos_cap_label.cpu())
            all_matching_scores.append(matching_scores.detach().cpu().float())
            if (_ + 1) % 5 == 0:
                t.update(params.retrieval_batch * 5)

        all_matching_labels = torch.cat(all_matching_labels, 0)  # 1000 * 5000
        all_matching_scores = torch.cat(all_matching_scores, 0)  # 1000 * 5000
        if params.local_rank != -1:
            np.save(os.path.join(params.eval_path, lang1 + "_score_" +"lang_"+lg+ "epoch_" + str(scores['epoch']) + "rank_" + str(
                params.local_rank) + ".npy"),
                    all_matching_scores.numpy())

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
        all_matching_scores = all_matching_scores.t()  # 5000 * 1000
        all_matching_labels = all_matching_labels.t()  # 5000 * 1000
        _, pred = all_matching_scores.topk(10, dim=-1)
        for i in range(len(pred)):
            for j, pred_idx in enumerate(pred[i][:10]):
                if all_matching_labels[i][pred_idx] == 1:
                    if j < 10:
                        t2i_r10 += 1
                    if j < 5:
                        t2i_r5 += 1
                    if j < 1:
                        t2i_r1 += 1

        return t2i_r1 / total_len, t2i_r5 / total_len, t2i_r10 / total_len, \
               i2t_r1 / total_img_len, i2t_r5 / total_img_len, i2t_r10 / total_img_len

    def evaluate_slide(self,scores, data_set, lang1, lang2):
        params = self.params

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        _acc = 0
        _tol = 0
        all_matching_labels = []

        all_matching_scores = [] # 1000 * 5000
        for _, batch in tqdm(enumerate(self.get_iterator(data_set, lang1, lang2))):

            (x1, len1,_), (img, img_mask, img_loc, img_ids),pos_labels = batch


            # [img. img_id...sent, sent_id..]

            img_len = img_mask.sum(dim=1)
            x_img = img.transpose(0, 1)
            img_loc = img_loc.transpose(0, 1)

            x1, len1, x_img, img_loc, img_len = to_cuda(x1, len1, x_img, img_loc, img_len)


            encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                    langs=None,
                                    image_loc=img_loc, refine_image=params.refine_image)

            encoder_outputs = encoder_outputs.transpose(0, 1)

            relation_scores = model('predict', tensor=encoder_outputs, is_relation=True)

            _labels = np.array(pos_labels)
            matching_label = torch.from_numpy(_labels)
            pred_label = (relation_scores>0.5).cpu().int().view(-1)

            _acc+= (pred_label == matching_label).sum().item()
            all_matching_scores.append(relation_scores.detach().cpu().float())
            all_matching_labels.extend(pos_labels)

            _tol+=len(matching_label)

        acc_name = '%s_%s_slide_acc' % (data_set, lang1)
        scores[acc_name] = 100. * _acc / _tol if _tol > 0 else 0.
        all_matching_scores = torch.cat(all_matching_scores, 0)
        np.save(os.path.join(params.eval_path,
                             "slide_score_" +  "epoch_" + str(scores['epoch']) + "rank_" + str(
                                 params.local_rank)+"_%s"%data_set + ".npy"),
                all_matching_scores.numpy())
        np.save(os.path.join(params.eval_path,
                             "slide_labels_" +  "epoch_" + str(scores['epoch']) + "rank_" + str(
                                 params.local_rank)+"_%s"%data_set + ".npy"),
                np.array(all_matching_labels))


        _print_scores  = all_matching_scores.numpy()
        _scores_labels = np.array(all_matching_labels)
        need_pred = _print_scores[_scores_labels > 0].reshape(-1)
        logger.info("Pos samples acc : %f" % (np.sum(need_pred > 0.5) / len(need_pred)))




def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
