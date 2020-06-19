# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
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
from torch.nn.utils import clip_grad_norm_

from torch import nn

import apex
from .optim import get_optimizer
from .utils import to_cuda, concat_batches
from .utils import parse_lambda_config, update_lambdas
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from .trainer_collate import retrieval_collate,caption_collate,retrieval_pretrain_collate,mt_caption_collate
logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # data iterators
        self.iterators = {}
        # set parameters
        self.set_parameters()
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for name in self.MODEL_NAMES:
                setattr(self, name,
                        nn.parallel.DistributedDataParallel(getattr(self, name), device_ids=[params.local_rank],
                                                            output_device=params.local_rank, broadcast_buffers=True))

        self.set_optimizers()
        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for name in self.MODEL_NAMES:
                    setattr(self, name,
                            apex.parallel.DistributedDataParallel(getattr(self, name), delay_allreduce=True))

        # probability of masking out / randomize / not modify words to predict
        params.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('CLM-%s' % l, []) for l in params.langs] +
            [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
            [('AE-%s' % lang, []) for lang in params.ae_steps] +
            [('MLM-%s' % l, []) for l in params.langs] +
            [('M-BART-%s' % l, []) for l in params.langs] +
            [('M-MASS-%s' % l, []) for l in params.langs] +
            [('NTG-%s' % l, []) for l in params.langs]+
            [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
            [('MA-%s' % lang, []) for lang in params.mass_steps] +
            [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
            [('IC-%s-%s' % (l1, l2), []) for l1, l2 in params.cross_modal_steps] +
            [('IMLM-%s' % l1, []) for l1, l2 in params.cross_mass_steps] +
            [('IDA_FULL-%s' % l1, []) for l1, l2 in params.cross_ae_steps] +
            [('IDA-%s' % l1, []) for l1, l2 in params.cross_ae_steps] +
            [('TIFG-%s' % l1, []) for l1, l2 in params.cross_gan_steps] +
            [('Rel-%s' % l1, []) for l1, l2 in params.cross_rel_steps] +
            [('CMLM-%s' % l, []) for l, l2 in params.cross_mlm_steps] +
            [('MRM-%s' % l, []) for l, l2 in params.cross_mrm_steps] +
            [('MRFR-%s' % l, []) for l, l2 in params.cross_mrfr_steps] +
            [('t2i-%s' % l, []) for l, l2 in params.cross_rel_steps] +
            [('i2t-%s' % l, []) for l, l2 in params.cross_rel_steps]
        )

        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        opt_names = self.optimizers.keys()
        models = [getattr(self, name) for name in self.MODEL_NAMES]

        models, optimizers = apex.amp.initialize(

            models,

            [self.optimizers[k] for k in opt_names],

            opt_level=('O%i' % params.amp)

        )

        for name, model in zip(self.MODEL_NAMES, models):
            setattr(self, name, model)

        self.optimizers = {

            opt_name: optimizer

            for opt_name, optimizer in zip(opt_names, optimizers)

        }

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = []
        for name in self.MODEL_NAMES:
            named_params.extend([(k, p) for k, p in getattr(self, name).named_parameters() if p.requires_grad])
        # model (excluding memory values)

        self.parameters['model'] = [p for k, p in named_params]

        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))

            assert len(v) >= 1

    def set_optimizers(self):

        """
        Set optimizers.
        """
        params = self.params

        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # memory values optimizer
        if params.use_memory:
            self.optimizers['memory'] = get_optimizer(self.parameters['memory'], params.mem_values_optimizer)
        # log

        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()
        params = self.params
        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]
        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)

            for optimizer in optimizers:
                optimizer.step()
        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join(
                "{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        # eos = self.params.eos_index
        # assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            # assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            # new_s.append(eos)
            # assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        # words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            # assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        pred_mask = pred_mask.bool()
        _x_real = x[pred_mask]
        if len(_x_real) == 0:
            pred_mask[0, 0] = 1
            _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def get_cross_lingual_iterator(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join(
            [str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:
                if self.params.is_ntg:
                    iterator = self.data['text'][lang1]['train'].get_iterator(
                        shuffle=True,
                        group_by_size=self.params.group_by_size,
                        n_sentences=-1)
                else:
                    iterator = self.data['mono'][lang1]['train'].get_iterator(
                        shuffle=True,
                        group_by_size=self.params.group_by_size,
                        n_sentences=-1,
                    )
        else:
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_cross_lingual_batch(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        # assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_cross_lingual_iterator(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_cross_lingual_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)

        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_cross_lingual_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:
            (x1, len1) = self.get_cross_lingual_batch(name, lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_cross_lingual_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving models to %s ...' % path)
        data = {}
        for name in self.MODEL_NAMES:
            if self.params.multi_gpu:
                data[name] = getattr(self, name).module.state_dict()
            else:
                data[name] = getattr(self, name).state_dict()

        # data['dico_id2word'] = self.data['dico'].id2word
        # data['dico_word2id'] = self.data['dico'].word2id
        # data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            logger.warning("Saving %s parameters ..." % name)
            data[name] = getattr(self, name).state_dict()
        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning("Saving %s optimizer ..." % name)
                data['%s_optimizer' % name] = self.optimizers[name].state_dict()

        # data['dico_id2word'] = self.data['dico'].id2word
        # data['dico_word2id'] = self.data['dico'].word2id
        # data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning("Reloading checkpoint from %s ..." % checkpoint_path)
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for name in self.MODEL_NAMES:
            getattr(self, name).load_state_dict(data[name])

        # reload optimizers
        for name in self.optimizers.keys():
            if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                logger.warning("Reloading checkpoint optimizer %s ..." % name)
            else:  # instead, we only reload current iterations / learning rates
                logger.warning("Not reloading checkpoint optimizer %s." % name)
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning("No 'num_updates' for optimizer %s." % name)
                        continue
                    logger.warning("Reloading 'num_updates' and 'lr' for optimizer %s." % name)
                    param_group['num_updates'] = data['%s_optimizer' % name]['param_groups'][group_id]['num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning("Checkpoint reloaded. Resuming at epoch %i / iteration %i ..." % (self.epoch, self.n_total_iter))

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_model('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)
                self.save_checkpoint('best-%s' % metric, include_optimizers=True)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
                self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint', include_optimizers=True)
        self.epoch += 1

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def mlm_step(self, lang1, lang2, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'pred')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

        # forward / loss

        tensor = model('crossfwd', stream_='text', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)


        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()


class XTrainer(Trainer):

    def __init__(self, encoder, data, params):

        self.MODEL_NAMES = ['model']

        # model / data / params
        self.model = encoder
        self.data = data
        self.params = params

        # optimizers
        # self.optimizers = {
        #     'encoder': self.get_optimizer_fp('encoder'),
        #     'decoder': self.get_optimizer_fp('decoder'),
        # }

        super().__init__(data, params)

    def get_iterator(self, iter_name, lang1, lang2):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join(
            [str(x) for x in [iter_name, lang1, lang2] if x is not None]))

        # google and sbu need to turn in another files
        dataset = self.data['cross_modal'][(lang1, lang2)]['train']
        if lang1 == 'google' or lang1 == 'sbu':
            dataset.update(self.epoch)

        # update captions each epoch
        if lang1 == 'flicker':
            dataset.update_captions()
        sampler = RandomSampler(dataset) if self.params.n_gpu_per_node == 1 else DistributedSampler(dataset)

        if self.params.is_generation:
            if self.params.is_mt:
                data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                         collate_fn=mt_caption_collate, num_workers=self.params.num_workers)
            else:
                data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                         collate_fn=caption_collate, num_workers=self.params.num_workers)
        else:
            if self.params.is_pretrain:
                data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                         collate_fn=retrieval_pretrain_collate, num_workers=self.params.num_workers)
            else:
                data_loader = DataLoader(dataset, batch_size=self.params.batch_size, sampler=sampler,
                                         collate_fn=retrieval_collate, num_workers=self.params.num_workers)

        logger.info("iterator (%s) done" % ','.join([str(x) for x in [iter_name, lang1, lang2] if x is not None]))

        for batch_idx, batch in enumerate(data_loader):
            yield batch

    def get_batch(self, iter_name, lang1, lang2=None):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang2 == 'img'
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2)
            self.iterators[(iter_name, lang1, lang2)] = iterator
        try:
            x = next(iterator)
        except StopIteration:
            if self.params.is_pretrain:
                self.iterators = {}
            iterator = self.get_iterator(iter_name, lang1, lang2)
            self.iterators[(iter_name, lang1, lang2)] = iterator
            x = next(iterator)
        return x

    def mask_word(self, w):
        _w_real = w
        _w_rand = np.random.randint(self.params.n_words, size=w.shape)
        _w_mask = np.full(w.shape, self.params.mask_index)

        probs = torch.multinomial(self.params.pred_probs, len(_w_real), replacement=True)

        _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
        return _w

    def unfold_segments(self, segs):
        """Unfold the random mask segments, for example:
           The shuffle segment is [2, 0, 0, 2, 0],
           so the masked segment is like:
           [1, 1, 0, 0, 1, 1, 0]
           [1, 2, 3, 4, 5, 6, 7] (positions)
           (1 means this token will be masked, otherwise not)
           We return the position of the masked tokens like:
           [1, 2, 5, 6]
        """
        pos = []
        curr = 1  # We do not mask the start token
        for l in segs:
            if l >= 1:
                pos.extend([curr + i for i in range(l)])
                curr += l
            else:
                curr += 1
        return np.array(pos)

    def shuffle_segments(self, segs, unmasked_tokens):
        """
        We control 20% mask segment is at the start of sentences
                   20% mask segment is at the end   of sentences
                   60% mask segment is at random positions,
        """

        p = np.random.random()
        if p >= 0.8:
            shuf_segs = segs[1:] + unmasked_tokens
        elif p >= 0.6:
            shuf_segs = segs[:-1] + unmasked_tokens
        else:
            shuf_segs = segs + unmasked_tokens

        random.shuffle(shuf_segs)

        if p >= 0.8:
            shuf_segs = segs[0:1] + shuf_segs
        elif p >= 0.6:
            shuf_segs = shuf_segs + segs[-1:]
        return shuf_segs

    def get_segments(self, mask_len, min_len):
        segs = []
        while mask_len >= min_len:
            segs.append(min_len)
            mask_len -= min_len
        if mask_len != 0:
            segs.append(mask_len)
        return segs

    def restricted_mask_sent(self, x, l, min_len=100000):
        """ Restricted mask sents
            if min_len is equal to 1, it can be viewed as
            discrete mask;
            if min_len -> inf, it can be viewed as
            pure sentence mask
            x : [len,batch]
            l: [batch]
        """
        if min_len <= 0:
            min_len = 1
        max_len = 0
        positions, inputs, targets, outputs, = [], [], [], []

        mask_len = round(l[np.argsort(l)[0].item()].item() * self.params.word_mass)
        len2 = [mask_len for i in range(l.size(0))]

        unmasked_tokens = [0 for i in range(l.min().item() - mask_len - 1)]
        segs = self.get_segments(mask_len, min_len)

        for i in range(l.size(0)):
            words = np.array(x[:l[i], i].tolist())  # [LEN(i)]
            shuf_segs = self.shuffle_segments(segs, unmasked_tokens)
            pos_i = self.unfold_segments(shuf_segs)
            output_i = words[pos_i].copy()  # [1,2,5,6]
            target_i = words[pos_i - 1].copy()  # []
            words[pos_i] = self.mask_word(words[pos_i])

            inputs.append(words)
            targets.append(target_i)
            outputs.append(output_i)
            positions.append(pos_i - 1)

        x1 = torch.LongTensor(max(l), l.size(0)).fill_(self.params.pad_index)
        x2 = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        y = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        pos = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        l1 = l.clone()
        l2 = torch.LongTensor(len2)
        for i in range(l.size(0)):
            x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
            x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
            y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
            pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))

        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)
        return x1, l1, x2, l2, y, pred_mask, pos

    def ic_step(self, dataset='coco', input_stream='img', lambda_coeff=1):
        """
        Cross-modal Caption generation step
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # _lang1 = _lang2 = 'en'
        # lang2_id = params.lang2id[_lang2]

        # generate batch

        (x2, len2), (x1, x1_mask, img_loc, img_id) = self.get_batch('txt2img', dataset, input_stream)
        # convert fp16
        #assign lang ids

        if len(params.ft_lgs)>0:
            lang1_id = params.lang2id[params.ft_lgs[0]]
            langs = x2.clone().fill_(lang1_id)
        else:
            lang1_id = params.lang2id['en']
            langs = x2.clone().fill_(lang1_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])  # decide whether or not  predict
        assert len(y) == (len2 - 1).sum().item()

        #
        len1 = x1_mask.sum(dim=1)
        x1 = x1.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)
        if len(params.ft_lgs) > 0:
            lang1_id = params.lang2id[params.ft_lgs[0]]
            langs_img = x1_mask.transpose(0,1).clone().fill_(lang1_id)
        else:
            lang1_id = params.lang2id['en']
            langs_img = x1_mask.transpose(0,1).clone().fill_(lang1_id)

        # cuda

        x1, len1, img_loc, x2, len2, y, x1_mask,langs,langs_img = to_cuda(x1, len1, img_loc, x2, len2, y, x1_mask,langs,langs_img)

        # encode source sentence
        enc1 = model('crossfwd', stream_='img', x=x1, lengths=len1, langs=langs_img, causal=False,
                     image_loc=img_loc)
        enc1 = enc1.transpose(0, 1)

        # decode target sentence
        dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs, causal=True, src_enc=enc1,
                     src_len=len1)

        # loss
        # the last word not apply prediction logic
        _, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('IC-%s-%s' % (dataset, input_stream))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def mt_ic_step(self, dataset='coco', input_stream='img', lambda_coeff=1):
        """
        Cross-modal Caption generation step
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # _lang1 = _lang2 = 'en'
        # lang2_id = params.lang2id[_lang2]

        # generate batch

        (x_src,len_src),(x2, len2),(x1, x1_mask, img_loc, img_id) = self.get_batch('txt2img', dataset, input_stream)
        # convert fp16
        #assign lang ids


        lang0_id = params.lang2id[params.ft_lgs[0]]
        lang_src = x_src.clone().fill_(lang0_id)
        lang1_id = params.lang2id[params.ft_lgs[1]]
        langs = x2.clone().fill_(lang1_id)


        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])  # decide whether or not  predict
        assert len(y) == (len2 - 1).sum().item()

        #
        len1 = x1_mask.sum(dim=1)
        x1 = x1.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        # cuda

        x1, len1, img_loc, x2, len2, y, x1_mask,langs,lang_src,x_src,len_src = to_cuda(x1, len1, img_loc, x2, len2, y, x1_mask,langs,lang_src,x_src,len_src)

        if params.mt_only_text:
            encoder_outputs = model('crossfwd', stream_='text', x=x_src, lengths=len_src, langs=lang_src, causal=False)
            len_all = len_src
        else:
            encoder_outputs = model('jointfwd', x=x_src, lengths=len_src, x_img=x1, lengths_img=len1, causal=False,
                                    langs=None,image_loc=img_loc)
            len_all = len_src + len1

        # encode source sentence
        # enc1 = model('jo', stream_='img', x=x1, lengths=len1, langs=None, causal=False, cross_modal=True,
        #              image_loc=img_loc, refine_image=params.refine_image, refine_encoder=params.refine_encoder,
        #              image_dist=None)
        enc1 = encoder_outputs.transpose(0, 1)


        # decode target sentence
        dec2 = model('crossfwd', stream_='text', x=x2, lengths=len2, langs=langs, causal=True, src_enc=enc1,
                     src_len=len_all)

        # loss
        # the last word not apply prediction logic
        _, loss = model('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('IC-%s-%s' % (dataset, input_stream))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def _mask_object(self, object_features,mask_len=50):
        # we need to mask it when training
        masked_object_features = []
        # lm_label_ids = []  # share vocabulary with word does not work
        _n_mask = 0

        i =0
        max_len = len(object_features)-mask_len
        span_mask =  np.random.poisson(lam=3) % mask_len
        while i< len(object_features):
            prob = random.random()
            if prob < 0.15 :
                prob /= 0.15
                if prob < 0.9:
                    masked_object_features.append(np.zeros((2048), dtype=np.float32))
                    i+=span_mask
                else:
                    masked_object_features.append(object_features[i])

                _n_mask+=1
            else:
                masked_object_features.append(object_features[i])
            i+=1
        if len(masked_object_features)>max_len:
            masked_object_features = masked_object_features[:max_len]
        if len(masked_object_features) < max_len:
            lef_len = max_len-len(masked_object_features)
            masked_object_features.extend([np.zeros((2048), dtype=np.float32)]*lef_len)

        masked_object_features = np.stack(masked_object_features, 0)  # [BS,dim]

        masked_object_features = torch.FloatTensor(np.stack(masked_object_features, 0))
        att_feat = F.normalize(masked_object_features, dim=-1)
        return att_feat.numpy()

    def rel_step(self, dataset='coco', input_stream='img', lambda_1=1, lambda_2=1):
        t2i_batch, i2t_batch = self.get_batch('rel', dataset, input_stream)
        if self.params.t2i_flag:
            self.t2i_step(t2i_batch, dataset, lambda_1)
        if self.params.i2t_flag:
            self.i2t_step(i2t_batch, dataset, lambda_2)

    def t2i_step(self, batches, dataset='coco', lambda_coeff=1):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        (x1, len1, lang_p), (img, img_mask, img_loc, obj_labels, pos_labels, img_ids) = batches

        # follow uvp
        lang_p = lang_p.transpose(0, 1)
        lang_img = torch.LongTensor([[params.n_langs] * params.max_region_num] * x1.size()[1])
        langs = torch.cat([lang_img, lang_p], dim=1)
        # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        x1, len1, langs, x_img, img_loc, img_len = to_cuda(x1, len1, langs, x_img, img_loc, img_len)

        encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                langs=langs,
                                image_loc=img_loc)

        encoder_outputs = encoder_outputs.transpose(0, 1)

        relation_scores = model('predict', tensor=encoder_outputs, is_relation=True)

        def one_hot_labels(_labels):
            nb_classes = params.sample_n
            targets = _labels.reshape(-1)
            one_hot_targets = np.eye(nb_classes, dtype='float32')[targets]
            return torch.from_numpy(one_hot_targets)

        target_labels = one_hot_labels(np.array(pos_labels))

        ce_loss = F.cross_entropy(relation_scores.view(-1, params.sample_n).cpu(),
                                  torch.from_numpy(np.array(pos_labels)))
        bce_loss = F.binary_cross_entropy_with_logits(relation_scores.view(-1).cpu(),
                                                      target_labels.view(-1))

        loss = 0

        _loss = params.multi_cls_loss_weight * ce_loss + params.bin_cls_loss_weight * bce_loss
        loss += _loss

        self.stats['t2i-%s' % (dataset)].append(loss.item())
        loss = lambda_coeff * loss
        # optimize
        self.optimize(loss)

        bs = len1.size(0)
        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += bs * encoder_outputs.size()[1]

    def i2t_step(self, batches, dataset='coco', lambda_coeff=1):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # _lang1 = _lang2 = 'en'
        # lang2_id = params.lang2id[_lang2]

        # generate batch
        (x1, len1, lang_p), (img, img_mask, img_loc, obj_labels, pos_labels, img_ids) = batches

        # follow uvp
        lang_p = lang_p.transpose(0, 1)
        lang_img = torch.LongTensor([[params.n_langs] * params.max_region_num] * x1.size()[1])
        langs = torch.cat([lang_img, lang_p], dim=1)  # [img. img_id...sent, sent_id..]

        img_len = img_mask.sum(dim=1)
        x_img = img.transpose(0, 1)
        img_loc = img_loc.transpose(0, 1)

        x1, len1, langs, x_img, img_loc, img_len = to_cuda(x1, len1, langs, x_img, img_loc, img_len)

        encoder_outputs = model('jointfwd', x=x1, lengths=len1, x_img=x_img, lengths_img=img_len, causal=False,
                                langs=langs,
                                image_loc=img_loc)

        encoder_outputs = encoder_outputs.transpose(0, 1)

        relation_scores = model('predict', tensor=encoder_outputs, is_relation=True)

        def one_hot_labels(_labels):
            nb_classes = params.sample_n
            targets = _labels.reshape(-1)
            one_hot_targets = np.eye(nb_classes, dtype='float32')[targets]
            return torch.from_numpy(one_hot_targets)

        target_labels = one_hot_labels(np.array(pos_labels))

        ce_loss = F.cross_entropy(relation_scores.view(-1, params.sample_n).cpu(),
                                  torch.from_numpy(np.array(pos_labels)))
        bce_loss = F.binary_cross_entropy_with_logits(relation_scores.view(-1).cpu(),
                                                      target_labels.view(-1))

        loss = 0

        _loss = params.multi_cls_loss_weight * ce_loss + params.bin_cls_loss_weight * bce_loss
        loss += _loss

        self.stats['i2t-%s' % (dataset)].append(loss.item())
        loss = lambda_coeff * loss
        # optimize
        self.optimize(loss)

        bs = len1.size(0)
        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += bs * encoder_outputs.size()[1]
