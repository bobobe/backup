import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models

import numpy as np


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

            # hidden:(level * batch * hiddensize,level*batch*hiddensize)
            # input：(level, batch, em_hidden)
    def forward(self, input, hidden):
        """
        :param input: batch, emb_size.
        :param hidden: (batch, hidden_size), (batch, hidden_size), first of both context and plevel information, second of plevel information.
        :return:
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i])) #每次都是同样的initial hidden，那上一次的hidden作为input
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1) # numlayers， batch, hidden_size
        c_1 = torch.stack(c_1)

        # input， 最后一个cell的hidden， batch, hidden
        # (batch, hidden), (batch, hidden)
        # return input, (h_1[-1], c_1[-1])

        # input: batch, hidden
        # h_1: numlayers, batch, hidden_size
        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        self.config = config

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if not self.config.bidirec:
            return outputs, (h, c)
        else:
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):
    def __init__(self, config, vocab_size, label_tree, embedding=None, score_fn=None, use_cuda = False):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        # self.rnn = nn.ModuleList([StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
        #                 num_layers=config.num_layers, dropout=config.dropout) for i in range(len(label_tree.levels)-1)])
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.rnns = nn.ModuleList([StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout) for i in range(4)])
        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.decoder_hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v = nn.Linear(config.decoder_hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

        # self.init_state_linear_in = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.init_state_linear_in = nn.Linear(config.emb_size+config.decoder_hidden_size, config.decoder_hidden_size)
        # self.init_state_linear_out = nn.Linear(config.decoder_hidden_size+config.emb_size, config.decoder_hidden_size)
        self.init_state_linear_out = nn.Linear(config.decoder_hidden_size + config.emb_size, config.decoder_hidden_size)

        if self.config.global_emb:
            self.gated1 = nn.Linear(config.emb_size, config.emb_size)
            self.gated2 = nn.Linear(config.emb_size, config.emb_size)

    def forward111111111111111111111111111111111111111(self,inputs, init_state, contexts, label_tree, input_lens):
        #inputs(nlevel,batch,T)
        #init_state(h(T,batch,hidden), c(T,batch,hidden))
        #contexts(batch,T,hidden)
        #input_lens(nlevel,batch)
        outputs, attns = [], []
        batch = len(contexts)
        num_level = len(inputs)
       # inputs = torch.stack(inputs)
        #contexts = contexts.repeat(level, 1, 1, 1)#level,batch,T,hidden
        if not self.config.global_emb:
            mask = torch.zeros(5,batch,len(inputs[0][0]))#level,batch,T
            for levidx,lev in enumerate(input_lens):
                for index,indice in enumerate(lev):
                    mask[levidx][index][1:indice.data[0]-1] = 1 # 只保留文本label
            embs = []
            for idx,level in enumerate(inputs):
                embs_ = self.embedding(level)
                embs.append(embs_)
            embs = torch.stack(embs)#level,batch,t,hidden
           # embs = self.embedding(inputs)#level,batch,t,hidden
            embs = embs.mul(Variable(mask.unsqueeze(3)))#level,batch,t,hidden
            state = self.init_state(embs, contexts.repeat(num_level,1,1,1))
            #state:(level * batch * hiddensize,level*batch*hiddensize)

            embs = []
            for idx,level in enumerate(inputs):#embs取到倒数第二位，因为rnn再往后传就没了 
                embs_ = self.embedding(level[:,:-1])
                embs.append(embs_)
            embs = torch.stack(embs)#level,batch,t,hidden
            #embs = self.embedding(inputs[:,:,:-1])#embs取到倒数第二位，因为rnn再往后传就没了
            state = (state[0][:-1],state[1][:-1])#不取最后一层
            for emb in embs.split(1, dim=2): # （每个位置）level,batch,1,em_hidden)
                output, state = self.rnn(emb.squeeze(2)[1:], state)#emb从第二层开始。第一层的state喂给第二层rnn，以此类推
                # output:level*batch*hidden_size
                # state:(h,c):(level*batch*hidden_size,level*batch*hidden_size)
                output, attn_weights = self.attention(output, contexts.repeat(num_level-1,1,1,1))
                # output:level*batch*hidden_size
                # attn_weight: level*batch*T
                output = self.dropout(output)
                outputs += [output]
                attns += [attn_weights]
            outputs = torch.stack(outputs) # nlabels *level* batch * size
            attns = torch.stack(attns)#nlabels *level*batch* T

            outputs = outputs.permute(1,0,2,3).contiguous()#level*nlabels*batch*size
            attns = attns.permute(1,0,2,3).contiguous()#level*nlabels*batch*T
        return outputs, state
        #state:(h,c):(level*batch*hidden_size,level*batch*hidden_size)


    def forward(self, inputs, init_state, contexts, label_tree, input_lens,use_cuda = True):

        """

        :param inputs: tgt， level, batch, seq,
        :param init_state: initial hidden states
        :param contexts:  encoding hiddens
        :return: context vector at each position and state at last position.
        """
        if not self.config.global_emb:
            i = 0
            outputs, attns = [], []
            while i < len(inputs)-1:#每一层
                plen = input_lens[i] #[len of each sample at this level, ]
                plevel = inputs[i] # batch * nlabels, with padding.
                level = inputs[i+1] # batch * nlabels
                # batch, nlabels
                mask = torch.zeros(len(plevel), len(plevel[0]))
                for idx, l in enumerate(plen):
                    # mask out <s> and </s> and padding.（置0）
                    mask[idx][1:l.data[0]-1] = 1

                pembs = self.embedding(plevel)
                # batch, nlabels, embsize
                if(use_cuda):
                    pembs = pembs.mul(Variable(mask.unsqueeze(2)).cuda())#为什么要把padding和<s>mask掉，因为要给下一层使用
                else:
                    pembs = pembs.mul(Variable(mask.unsqueeze(2)))
                state = self.init_state(pembs, contexts, init_state) # both of decoder_hidden_size

                #idx = torch.LongTensor(range(len(level[0]))[:-1])
                embs = self.embedding(level[:, :-1]) # batch, nlabels, emb_size
                level_outputs, level_attns = [], []
                for emb in embs.split(1, dim=1): # batch, 1, emb_size（每个位置）
                    output, state = self.rnn(emb.squeeze(1), state) # 以该位置的embed， 和上一位置的hidden state作为输入
                    #batch * hidden_size
                    output, attn_weights = self.attention(output, contexts) # 以output为key，得到Context的weighted sum，
                    output = self.dropout(output)
                    level_outputs += [output]
                    level_attns += [attn_weights]
                level_outputs = torch.stack(level_outputs) # nlabels * batch * size
                level_attns = torch.stack(level_attns)
                outputs.append(level_outputs) # list of nlevels, with each level nlabels * batch * size
                attns.append(level_attns)
                i += 1
            # outputs: nlevels, nlabels, batch, hidden_size
            # state:
            return outputs, state

    def forward2222222222222222222222222222(self, inputs, init_state, contexts, label_tree, input_lens,use_cuda = True):

        """

        :param inputs: tgt， level, batch, seq,
        :param init_state: initial hidden states
        :param contexts:  encoding hiddens
        :return: context vector at each position and state at last position.
        """
        if not self.config.global_emb:
            i = 0
            outputs, attns = [], []
            while i < len(inputs)-1:#每一层
                plen = input_lens[i] #[len of each sample at this level, ]
                plevel = inputs[i] # batch * nlabels, with padding.
                level = inputs[i+1] # batch * nlabels
                # batch, nlabels
                mask = torch.zeros(len(plevel), len(plevel[0]))
                for idx, l in enumerate(plen):
                    # mask out <s> and </s> and padding.（置0）
                    mask[idx][1:l.data[0]-1] = 1

                pembs = self.embedding(plevel)
                # batch, nlabels, embsize
                if(use_cuda):
                    pembs = pembs.mul(Variable(mask.unsqueeze(2)).cuda())#为什么要把padding和<s>mask掉，因为要给下一层使用
                else:
                    pembs = pembs.mul(Variable(mask.unsqueeze(2)))
                state = self.init_state(pembs, contexts, init_state) # both of decoder_hidden_size

                #idx = torch.LongTensor(range(len(level[0]))[:-1])
                embs = self.embedding(level[:, :-1]) # batch, nlabels, emb_size
                level_outputs, level_attns = [], []
                for emb in embs.split(1, dim=1): # batch, 1, emb_size（每个位置）
                    output, state = self.rnn(emb.squeeze(1), state) # 以该位置的embed， 和上一位置的hidden state作为输入
                    #batch * hidden_size
                    output, attn_weights = self.attention(output, contexts) # 以output为key，得到Context的weighted sum，
                    output = self.dropout(output)
                    level_outputs += [output]
                    level_attns += [attn_weights]
                level_outputs = torch.stack(level_outputs) # nlabels * batch * size
                level_attns = torch.stack(level_attns)
                outputs.append(level_outputs) # list of nlevels, with each level nlabels * batch * size
                attns.append(level_attns)
                i += 1
            # outputs: nlevels, nlabels, batch, hidden_size
            # state:
            return outputs, state
        else:
            outputs, state, attns = [], init_state, []
            embs = self.embedding(inputs).split(1)
            max_time_step = len(embs)
            emb = embs[0]
            output, state = self.rnn(emb.squeeze(0), state)
            output, attn_weights = self.attention(output, contexts)
            output = self.dropout(output)
            soft_score = F.softmax(self.linear(output))
            outputs += [output]
            attns += [attn_weights]

            batch_size = soft_score.size(0)
            a, b = self.embedding.weight.size()

            for i in range(max_time_step-1):
                emb1 = torch.bmm(soft_score.unsqueeze(1), self.embedding.weight.expand((batch_size, a, b)))
                emb2 = embs[i+1]
                gamma = F.sigmoid(self.gated1(emb1.squeeze())+self.gated2(emb2.squeeze()))
                emb = gamma * emb1.squeeze() + (1 - gamma) * emb2.squeeze()
                output, state = self.rnn(emb, state)
                output, attn_weights = self.attention(output, contexts)
                output = self.dropout(output)
                soft_score = F.softmax(self.linear(output))
                outputs += [output]
                attns += [attn_weights]
            outputs = torch.stack(outputs)
            attns = torch.stack(attns)
            return outputs, state
    def init_state1(self, embs, context, activation="tanh"):
        # embs: level,batch, nlabels, emb_size

        x = torch.sum(embs, dim=2) # level,batch, emb_size（把此层的tgt embedding都加起来做为初始s0？）
        gamma_h = self.init_state_linear_in(x).unsqueeze(3)  # level * batch * hiddensize * 1
        if activation == 'tanh':
            gamma_h = F.tanh(gamma_h)
        #context,level,batch,seq_len,hiddensize
        weights = torch.matmul(context, gamma_h).squeeze(3)  # level*batch * seq_len
        weights = F.softmax(weights, dim=2)  # level*batch * seq_len（#初始attention weight）
        #level*batch*1*seq_len,level*batch*seq_len*hiddensize = level*batch*1*hiddensize
        c_t = torch.matmul(weights.unsqueeze(2), context).squeeze(2)  # level*batch * size
        output = F.tanh(self.init_state_linear_out(torch.cat([c_t, x], 2)))#level*batch*hidden_size
        # output: level,batch, decoder_hidden_size
        return output, gamma_h.squeeze(3)#level * batch * hiddensize

    def init_state2(self, embs, context, init_state, activation="tanh"):
        # embs: batch, nlabels, emb_size
        # context: batch, seq_len, hidden_size
        # init_state: (num_layers * num_directions, batch, hidden_size)
        x = torch.sum(embs, dim=1) # batch, emb_size（把此层的tgt embedding都加起来做为初始s0？）
        gamma_h = self.init_state_linear_in(x).unsqueeze(2)  # batch * hiddensize * 1， 只与plevel信息有关
        if activation == 'tanh':
            gamma_h = F.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # batch * seq_len
        weights = F.softmax(weights, dim=1)  # batch * seq_len（#初始attention weight）
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # batch * hiddensize, 根据输入的plevel的embedding, 其得Context seq hiddens 的加权之和。
        output = F.tanh(self.init_state_linear_out(torch.cat([c_t, x], 1))) # batch, decoder_hidden_size， 与Context， plevel信息都有关
        return output, gamma_h.squeeze(2)

    def init_state(self, embs, context, init_state, activation="tanh"):
        # embs: batch, nlabels, emb_size
        # context: batch, seq_len, hidden_size
        # init_state: (num_layers * num_directions, batch, hidden_size)
        num_layers = init_state[0].size(0)
        x = torch.sum(embs, dim=1) # batch, emb_size（把此层的tgt embedding都加起来做为初始s0？）
        x = x.expand(num_layers, -1, -1) # num_layers, batch, emb_size
        x_0 = self.init_state_linear_in(torch.cat([init_state[0], x], -1))
        x_1 = self.init_state_linear_out(torch.cat([init_state[1], x], -1))
        return x_0, x_1


    def compute_score(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        else:
            scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len
        soft_score = None
        mask = None
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], soft_score, state, contexts, mask)
            if self.config.global_emb:
                soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)


    def sample_one(self, input, soft_score, state, contexts, mask):
        if self.config.global_emb:
            batch_size = contexts.size(0)
            a, b = self.embedding.weight.size()
            if soft_score is None:
                emb = self.embedding(input)
            else:
                emb1 = torch.bmm(soft_score.unsqueeze(1), self.embedding.weight.expand((batch_size, a, b)))
                emb2 = self.embedding(input)
                gamma = F.sigmoid(self.gated1(emb1.squeeze())+self.gated2(emb2.squeeze()))
                emb = gamma * emb1.squeeze() + (1 - gamma) * emb2.squeeze()
        else:
            emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, contexts)
        output = self.compute_score(hidden)
        if self.config.mask:
            if mask is not None:
                output = output.scatter_(1, mask, -9999999999)
        return output, state, attn_weigths

    def tree_sample11111111111111111111111111111111(self, src, label_tree, contexts, level_mask = True,max_len=5):
        """predict label tree of a sample."""
        j = 0
        batch = len(src[1]) # for src is not batch first
        plevel = None
        outputs = []
        while j < len(label_tree.levels)-1:
            if type(plevel) == type(None):#第一层
                plevel = torch.LongTensor([label_tree.label2idx["root"]])
                # batch, nlabels
                plevel = plevel.repeat(batch, 1)
                plevel = Variable(plevel)
            # batch * 1
            emb_idx = torch.LongTensor([label_tree.label2idx["<s>"]] * batch)
            emb_idx = Variable(emb_idx)
            pembs = self.embedding(plevel.cuda())
            state = self.init_state(pembs, contexts)  # both of decoder_hidden_size
            level_output = None
            count = torch.zeros(batch)  # to count whether all samples meet the end.
            while sum(count) < batch:
                a = sum(count)
                # batch, emb_size
                emb = self.embedding(emb_idx.cuda())
                output, state = self.rnn(emb.squeeze(1), state)
                # output, batch * size
                output, attn_weights = self.attention(output, contexts)
                output = self.dropout(output)
                # batch * vocal_size
                scores = self.compute_score(output)

                # level mask
                levels = [[] for r in range(4)]
                if level_mask:
                    for i, level in enumerate(label_tree.levels[1:]):
                        levels[i] = [label_tree.label2idx[l] for l in level]
                        levels[i] += [label_tree.label2idx['<blank>'], label_tree.label2idx['<unk>'],
                                      label_tree.label2idx['</s>'], label_tree.label2idx['<s>']]
                    mask = [l for l in range(len(label_tree.label2idx)) if l not in levels[j]]
                    scores[:, mask] = -9999999999  # mask
                # level mask end
                
                # batch * 1
                pred = scores.max(1, keepdim=True)[1]
                for i in range(batch):
                    cur = pred.data[i][0]
                    # prediction ends if predict unk, /s, blank, or predict the same label as before.
                    if cur == label_tree.label2idx["<unk>"] or cur == label_tree.label2idx["</s>"] or cur == label_tree.label2idx["<blank>"]:
                        count[i] = 1
                    elif (type(level_output) != type(None) and cur in level_output.data[i]):
                        # pred[i] = label_tree.label2idx["<unk>"]
                        # to do for predicting the same label as the last step.
                        count[i] = 1

                if type(level_output) == type(None):
                    level_output = pred
                else:
                    # batch, nlabels.
                    level_output = torch.cat([level_output, pred], dim=1)
                emb_idx = pred
            outputs.append(level_output)
            # batch, nlabels
            plevel = level_output

            j += 1
        # list of levels, each level is a tensor of size: batch, nlabels, tensor
        return outputs

    def tree_sample(self, src, label_tree, contexts, init_state, level_mask = True,max_len=25,use_cuda = True):
        #src,(len,batch,hidden)
        #contexts,(batch_size,T,hidden_dim)

        """predict label tree of a sample."""
        j = 0
        batch = len(src[1]) # for src is not batch first
        plevel = None
        outputs = []
        # level mask
        levels = [[] for r in range(4)]
        if level_mask:
            for i, level in enumerate(label_tree.levels[1:]):
                levels[i] = [label_tree.label2idx[l] for l in level]
                levels[i] += [label_tree.label2idx['<blank>'], label_tree.label2idx['<unk>'],
                              label_tree.label2idx['</s>']]

        while j < len(label_tree.levels)-1:
            if type(plevel) == type(None):#第一层
                plevel = torch.LongTensor([label_tree.label2idx["root"]])
                # batch, nlabels
                plevel = plevel.repeat(batch, 1)
                plevel = Variable(plevel)
            # batch * 1
            emb_idx = torch.LongTensor([label_tree.label2idx["<s>"]] * batch)
            emb_idx = Variable(emb_idx)
            if(use_cuda):
            	pembs = self.embedding(plevel.cuda())
            else:
               	pembs = self.embedding(plevel)                
            state = self.init_state(pembs, contexts, init_state)  # both of decoder_hidden_size
            level_output = None
           # count = torch.zeros(batch)  # to count whether all samples meet the end.
            for i in range(self.config.max_tgt_len):
               # a = sum(count)
                # batch, emb_size
                if(use_cuda):
                    emb = self.embedding(emb_idx.cuda())
                else:
                    emb = self.embedding(emb_idx)                   
                output, state = self.rnn(emb.squeeze(1), state)
                # output, batch * size
                output, attn_weights = self.attention(output, contexts)
                output = self.dropout(output)
                # batch * vocal_size
                scores = self.compute_score(output)

                # level mask
                if level_mask:
                    mask = [l for l in range(len(label_tree.label2idx)) if l not in levels[j]]
                    scores[:, mask] = -9999999999  # mask
                # level mask end
                #scores[0,0] = 0 
                #pre mask
                if type(level_output)!=type(None):
                     #level_output(batch,nlabels)
                    for b,lev in enumerate(level_output): 
                        for le in lev:
                            scores[b, le.data.cpu().numpy()[0]] = -9999999999
                       
                # batch * 1
                pred = scores.max(1, keepdim=True)[1]
                '''
                for i in range(batch):
                    cur = pred.data[i][0]
                    # prediction ends if predict unk, /s, blank, or predict the same label as before.
                    if cur == label_tree.label2idx["<unk>"] or cur == label_tree.label2idx["</s>"] or cur == label_tree.label2idx["<blank>"]:
                        count[i] = 1
                    elif (type(level_output) != type(None) and cur in level_output.data[i]):
                        # pred[i] = label_tree.label2idx["<unk>"]
                        # to do for predicting the same label as the last step.
                        count[i] = 1
                '''
                if type(level_output) == type(None):
                    level_output = torch.cat([pred],dim = 1)
                else:
                    # batch, nlabels.
                    level_output = torch.cat([level_output, pred], dim=1)
                emb_idx = pred
            outputs.append(level_output)
            # batch, nlabels
            plevel = level_output

            j += 1
        # nlevels, batch, nlabels
        return outputs


