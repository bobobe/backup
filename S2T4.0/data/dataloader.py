import torch
import torch.utils.data as torch_data
import os
import data.utils


padding_idx = 0

class dataset(torch_data.Dataset):

    def __init__(self, src, tgt, raw_src, raw_tgt):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index]

    def __len__(self):
        return len(self.src)


def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)


def padding(data):
    # applied to each batch.
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, raw_src, raw_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long() # idx 0 for padding
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long() # idx 0 for padding
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = s[:end]
    #tgt_len = [length-1 for length in tgt_len]

    #return src_pad.t(), src_len, tgt_pad.t(), tgt_len
    return raw_src, src_pad.t(), torch.LongTensor(src_len), \
           raw_tgt, tgt_pad.t(), torch.LongTensor(tgt_len)

def tree_padding(data):
    # tgt: (label_tree) of number of batch_size in a tuple, label_tree is of size nlevels, nlabels.
    src, tgt, raw_src, raw_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    # create batch for each level.
    blevels = [[] for i in range(5)]#blevels和tat_len都是存放每一层的样本，共5层，每层batch个样本。
    # list of, list of length of each level.
    tgt_len = [[] for i in range(5)]#存放每一层各个样本的长度
    for label_tree in tgt:
        for idx, level in enumerate(label_tree):
            blevels[idx].append(level)
            tgt_len[idx].append(len(level))
    max_len = [max(x) for x in tgt_len]#每一层的最大长度

    for idx, (levels, mlen) in enumerate(zip(blevels, max_len)):
        for level in levels:
            if len(level) < mlen:
                level += [padding_idx] * (mlen-len(level))#每一层长度不够的都在后面加上padding
        blevels[idx] = torch.LongTensor(levels).contiguous()

    for idx, tlen in enumerate(tgt_len):
        tgt_len[idx] = torch.LongTensor(tlen)
    # src_pad of not batch first.
    # blevels, a list of LongTensor, of batch * nlabels
    return raw_src, src_pad.t(), torch.LongTensor(src_len), raw_tgt, blevels, tgt_len

class Label_Sampler():
    def __init__(self, data_source, label_tree):
        self.data_source = data_source
        self.label_tree = label_tree

    def __iter__(self):
        raw_tgt = self.data_source.raw_tgt
        count = {2: [], 3: [], 4: [], 5: []}
        for idx, labels in enumerate(raw_tgt):
            labels_ = [len(label) != 0 for label in labels]
            # try:
            count[sum(labels_)].append(idx)
        indices = []
        for ct, idx in count.items():
            indices += idx

        return iter(indices)

    def __len__(self):
        return len(self.data_source)

def get_loader(dataset, batch_size, shuffle, num_workers, label_tree):
    #print(dataset[0])
    if(label_tree!=None):
        label_sampler = Label_Sampler(dataset, label_tree)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                             # sampler=label_sampler,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=tree_padding)
    return data_loader


# def get_iter(dataset, batch_size=1, shuffle=False, num_worders=1):
#     iter = []
#     for s, t, rs, rt in zip(dataset.src, dataset.tgt, dataset.raw_src, dataset.raw_tgt):
#         batch = (s.unsqueeze(0), t.unsqueeze(0), rs, rt)
#         iter.append(batch)
#     return iter
#


