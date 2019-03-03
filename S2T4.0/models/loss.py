import torch
import torch.nn as nn
import models
import data.dict as dict
from torch.autograd import Variable


def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    weight[dict.EOS] = 0.4
    weight[dict.UNK] = 0.5
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab


def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, label_tree = None,level_mask = True,sim_score=0):
    """

    :param hidden_outputs: nlevels, nlabels, batch, hidden_size
    :param decoder:
    :param targets: nlevels, batch, nlabels
    :param criterion:
    :param config:
    :param label_tree:
    :param level_mask:
    :param sim_score:
    :return:
    """
    total = 0 # total predicted label
    correct = 0 # total correctly predicted label
    total_loss = 0
    targets = targets[1:]
    # level mask
    levels = [[] for r in range(4)]  # nlevels, label ids in this level.
    if level_mask:
        for i, level in enumerate(label_tree.levels[1:]):
            levels[i] = [label_tree.label2idx[l] for l in level]
            levels[i] += [label_tree.label2idx['<blank>'], label_tree.label2idx['<unk>'], label_tree.label2idx['</s>'],
                          label_tree.label2idx['<s>']]

    for index,(hddout, tgt) in enumerate(zip(hidden_outputs, targets)):
        # hddout: nlabels, batch, hidden_size
        # tgt: batch * nlabels
        outputs = hddout.view(-1, hddout.size(2)) # nlabels*batch, size
        tgt = tgt[:, 1:].contiguous().view(-1) # discard <s>, nlabels*batch
        scores = decoder.compute_score(outputs) #  nlabels*batch, vocab_size

        if level_mask:
            mask = [l for l in range(len(label_tree.label2idx)) if l not in levels[index]]
            scores[:,mask] = -9999999999 #

        loss = criterion(scores, tgt) + sim_score
        pred = scores.max(1)[1] # nlabels*batch
        num_correct = pred.data.eq(tgt.data).masked_select(tgt.ne(dict.PAD).data).sum()
        num_total = tgt.ne(dict.PAD).data.sum()
        correct += num_correct
        total += num_total
        total_loss += loss
    total_loss.div(total).backward()
    total_loss = total_loss.data[0]

    return total_loss, total, correct, config.tgt_vocab, config.tgt_vocab

