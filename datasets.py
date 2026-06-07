from collections import OrderedDict
import enum
import numpy as np
import os
import torch

DatasetSplit = enum.Enum('DatasetSplit', 'train valid test')
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

class WikiText2(torch.utils.data.Dataset):
    """
        PyTorch Dataset for the WikiText2 corpus:
            https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

        Data format: each line in the text file is a sentence.
    """

    def __init__(self, data_dir, context_size, split, use_block_split=True, max_vocab_size=-1):
        self.context_size = context_size
        self.use_block_split = use_block_split
        self.word2idx = {}
        self.idx2word = []
        self.idx_counts = []
        self.non_vocab_tokens = [EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]  # hard-coded. Double-check if correct!

        # build the vocabulary from the training data
        self._build_vocabulary(os.path.join(data_dir, 'train.txt'), max_vocab_size)
        self.data = self._tokenize(os.path.join(data_dir, split.name + '.txt'))

        self.non_vocab_idx = [self.word2idx[t] for t in self.non_vocab_tokens]

    def __len__(self):
        if self.use_block_split:
            return len(self.data) // self.context_size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.use_block_split:
            x = self.data[idx*self.context_size:(idx+1)*self.context_size]
            y = self.data[idx*self.context_size+1:(idx+1)*self.context_size+1].view(-1)
        else:    
            x = torch.tensor([self.word2idx[PAD_TOKEN]] * self.context_size)
            y = torch.tensor([self.word2idx[PAD_TOKEN]] * self.context_size)
            context = min(self.context_size,idx)
            if idx > 0: x[-context:] = self.data[idx-context:idx]
            context = min(self.context_size,idx+1)
            y[-context:] = self.data[idx+1-context:idx+1]

        return x, y
        
    def word_count(self):
        # don't count <pad> as a word
        return len(self.idx2word)

    def _build_vocabulary(self, trainpath, max_vocab_size):
        with open(trainpath, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self._add_word(word)
        self.word2idx, self.idx2word, self.idx_counts = reorder_vocabulary_and_truncate(
            self.word2idx, self.idx2word, self.idx_counts, max_vocab_size
        )
        
        # add special tokens
        for tok in self.non_vocab_tokens:
            self._add_word(tok, add_nonvocab=True)
        self.unk_id = self.word2idx[UNK_TOKEN]
        self.eos_id = self.word2idx[EOS_TOKEN]
        self.pad_id = self.word2idx[PAD_TOKEN]

    def _add_word(self, word, add_nonvocab=False):
        if (not add_nonvocab) and (word in self.non_vocab_tokens):
            pass
        elif word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.idx_counts.append(1)
        else:
            self.idx_counts[self.word2idx[word]] += 1

    def _tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + [EOS_TOKEN]
                ids = []
                for word in words:
                    ids.append(self.word2idx.get(word, self.unk_id))
                idss.append(torch.tensor(ids).type(torch.int64))
        return torch.cat(idss)

def reorder_vocabulary_and_truncate(word2idx, idx2word, idx_counts, max_vocab_size):
    if max_vocab_size <= 0:
        max_vocab_size = len(idx2word)  # use the full vocabulary
    sorted_order = np.argsort(idx_counts)[::-1][:max_vocab_size]  # descending order of word counts
    idx_counts = np.sort(idx_counts)[::-1][:max_vocab_size].tolist()
    idx2word = np.asarray(idx2word)[sorted_order][:max_vocab_size].tolist()
    word2idx = OrderedDict((word, i) for (i, word) in enumerate(idx2word))
    return word2idx, idx2word, idx_counts
