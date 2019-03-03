import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data.dict as dict
import models

import numpy as np

class seq2seq(nn.Module):

    def __init__(self, config, label_tree, src_vocab_size, tgt_vocab_size, use_cuda, pretrain=None, score_fn=None):
        super(seq2seq, self).__init__()
        self.label_tree = label_tree
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None
        self.encoder = models.rnn_encoder(config, src_vocab_size, embedding=src_embedding)
        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, label_tree, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn)
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        # cross entropy loss
        self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()

        if self.use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def compute_loss(self, hidden_outputs, targets, memory_efficiency):
        if memory_efficiency:
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config,label_tree=self.label_tree)

    def forward(self, src, src_len, tgt, tgt_len):
        """

        :param src:
        :param src_len:
        :param tgt:
        :param tgt_len:
        :return:  [context vector], tgt
        """
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        for idx, (level, level_len) in enumerate(zip(tgt, tgt_len)):
            tgt[idx] = torch.index_select(level, dim=0, index=indices)
            tgt_len[idx] = torch.index_select(level_len, dim=0, index=indices)
        # state: (num_layers * num_directions, batch, hidden_size)
        contexts, state = self.encoder(src, lengths.data.tolist())
        # outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
        outputs, final_state = self.decoder(tgt, state, contexts.transpose(0, 1), self.label_tree, tgt_len)
        return outputs, tgt

    def tree_sample(self, src, src_len):
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=1, index=indices)
        contexts, state = self.encoder(src, lengths.data.tolist())
        # tgt: nlevels, batch, nlabels
        tgt = self.decoder.tree_sample(src, self.label_tree, contexts.transpose(0,1), state)
        for idx, level in enumerate(tgt):
            tgt[idx] = torch.index_select(level, dim=0, index=ind)
        return tgt

    def sample(self, src, src_len):

        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        #targets = tgt[1:]

        return sample_ids.t(), alignments.t()


    def beam_sample(self, src, src_len, beam_size = 1):
        """

        :param src: size * batch, tensor
        :param src_len: batch, tensor
        :param beam_size:
        :return:
        """

        #beam_size = self.config.beam_size
        batch_size = src.size(1)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        # seq_len, batch, hiddensize
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # batch*beam_size, seq_len, hiddensize
        contexts = rvar(contexts.data).transpose(0, 1)
        # (batch*beam_size, seq_len, hiddensize) * 2
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda)
                for __ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.
        
        mask = None
        soft_score = None
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder.sample_one(inp, soft_score, decState, contexts, mask)
            soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        #print(allHyps)
        #print(allAttn)
        return allHyps, allAttn
