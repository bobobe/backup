import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict

class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.activation = activation

    #x: level * batch * hidden_size
    #context: level * batch * T * hidden_size
    def forward1(self,x,context):
        """
                :param x: S_t
                :param context: encoding hiddens
                :return: C_t, weights.
                slightly different to what the paper described according to equation (4)
                """
        gamma_h = self.linear_in(x).unsqueeze(3)  # level*batch * size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        #weights: level*batch*T
        weights = torch.matmul(context, gamma_h).squeeze(3)
        weights = self.softmax(weights)
        #level*batch*1*T,level*batach*T*hidden_size = level*batch*1*hidden_size
        c_t = torch.matmul(weights.unsqueeze(2), context).squeeze(2)  # level*batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 2)))#level*batch*hidden_size
        return output, weights
        #output:level*batch*hidden_size
        #weight: level*batch*T

    def forward(self, x, context):
        """
        :param x: batch, hidden_size
        :param context: batch, seq_len, hidden_size
        :return: C_t, weights.
        slightly different to what the paper described according to equation (4)
        """
        gamma_h = self.linear_in(x).unsqueeze(2)    # batch * hidden_size * 1
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)   # batch * seq_len,
        weights = self.softmax(weights)   # batch * seq_len
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1) # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights
