"""
build a label tree according the label structure.
level-order label frequency statistics.
"""
from collections import deque
import torch

rcv1_v2_levels = 5

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

EXTEND = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]

class Tree_node(object):
    def __init__(self, parent, name, idx, desc=None):
        self.parent = parent
        self.name = name
        # idx of label at the label embedding matrix.
        self.idx = idx
        self.desc = desc
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Label_tree(object):
    def __init__(self, fname):
        self.nodes = {}
        self.root, self.nlabels, self.idx2label, self.label2idx = self.build_tree(fname)
        self.levels = self.level_stat(self.root) # [["Root"], ...]
        self.max_labels_per_level = 4

    def build_tree(self, fname):
        labels = 0
        idx2label = []
        label2idx = {}
        for wd in EXTEND:
            idx2label.append(wd)
            label2idx[wd] = labels
            labels += 1
        root = Tree_node(None, "root", labels)
        self.nodes["root"] = root
        idx2label.append("root")
        label2idx["root"] = labels
        labels += 1
        with open(fname, "r") as f:
            for line in f.readlines()[1:]:
                blks = line.split()
                pname = blks[1].lower()
                name = blks[3].lower()
                # assert pname in self.nodes
                # assert name not in self.nodes
                if pname not in self.nodes:
                    pnode = Tree_node(None, pname, labels)
                    self.nodes[pname] = pnode
                    idx2label.append(pname)
                    label2idx[pname] = labels
                    labels += 1
                if name not in self.nodes:
                    node = Tree_node(self.nodes[pname], name, labels)
                    self.nodes[name] = node
                    idx2label.append(name)
                    label2idx[name] = labels
                    labels += 1
                else:
                    self.nodes[name].parent = self.nodes[pname]
                self.nodes[pname].add_child(self.nodes[name])
        return root, labels, idx2label, label2idx

    '''
    返回label tree每一层的label名字
    levels是一个list，其中包含多个list，每个list是一层的lable名字
    '''
    def level_stat(self, root):
        """ bfs"""
        levels = []
        level = []
        q = deque()
        q.append(root)
        q.append("#")

        while q:
            root = q.popleft()
            if root != "#":
                for child in root.children:
                    q.append(child)
                level.append(root.name)
            else:
                levels.append(level)
                if len(q) == 0:
                    break
                else:
                    level = []
                    q.append("#")

        return levels

    def sample_tree(self, labels):#把一个样本的label转化为树状结构
        """build the tree structure for the one sample."""
        levels = [[] for i in range(rcv1_v2_levels)]
        levels_id = [[] for i in range(rcv1_v2_levels)]

        levels[0].append("root")
        levels_id[0].append(self.label2idx["root"])
        for label in labels:
            for i in range(1, rcv1_v2_levels):
                if label in self.levels[i]:
                    levels[i].append(label)
                    levels_id[i].append(self.label2idx[label])
                    break
        for idx, le in enumerate(levels_id):
            le.insert(0, self.label2idx["<s>"])
            le.append(self.label2idx["</s>"])
            if len(le) == 2:
                le.insert(1, self.label2idx["<unk>"])
        # nlevels * nlabels.
        return levels, levels_id
    
    def convertPred(self,sampleid,end):
        for index ,idx in enumerate(sampleid):
            if idx in end:
                sampleid[index:] = self.label2idx['</s>']
              #  print(sampleid)
                break
        return sampleid




if __name__ == "__main__":
    label_tree = Label_tree("rcv1.topics.hier.orig.txt")
    print(label_tree.nlabels)#label总数
    for level in label_tree.levels:
        print(level)
    print(label_tree.sample_tree(["c172", "ccat", "e121", "genv"]))
    res = []
    for itm in label_tree.nodes:
        res.append(label_tree.nodes[itm].idx)
    print(sorted(res))
    print(label_tree.label2idx["<unk>"])
