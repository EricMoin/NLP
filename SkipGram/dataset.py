from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

from config import Config
class SkipGramDataset(Dataset):
    # 输入的训练Config
    config:Config
    # 词汇表
    vocab:dict
    # 词汇表大小
    vocab_size:int
    # 索引到词汇表的映射
    index2word:dict
    # 训练数据
    data:list[tuple[int,int]]
    # 所有训练数据
    all_pairs: list[tuple[int,int]]
    # 词数
    word_counts:dict
    # 负采样词频
    word_freq:np.ndarray
    def __init__(self, config:Config):
        self.config = config
        self.get_corpus()
        self.build_vocab()
        self.build_training_pairs()
        self.train_pair_sampling()
        self.calc_nagetive_sampling()
    # 从语料库中读取数据
    def get_corpus(self):
        print("Reading corpus...")
        with open(self.config.corpus_path, 'r', encoding='utf-8') as f:
            # 语料库采样 - 只读取部分行
            all_lines = f.readlines()
            if self.config.corpus_sample_rate < 1.0:
                sample_size = int(len(all_lines) * self.config.corpus_sample_rate)
                sampled_lines = random.sample(all_lines, sample_size)
                print(f"Corpus sampling: using {sample_size}/{len(all_lines)} lines ({self.config.corpus_sample_rate:.2%})")
                corpus = [line.strip().split() for line in sampled_lines]
            else:
                corpus = [line.strip().split() for line in all_lines]
        self.corpus = corpus
    # 构建词汇表
    def build_vocab(self):
        print("Building vocabulary...")
        self.word_counts = Counter([word for sentence in self.corpus for word in sentence])
        print(f"Total unique words: {len(self.word_counts)}")
        
        self.vocab = {word: i for i, (word, count) in enumerate(
            self.word_counts.items()) if count >= self.config.min_count}
        self.vocab['<UNK>'] = len(self.vocab)
        self.index2word = {i: word for word, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size (after min_count={self.config.min_count}): {self.vocab_size}")
    # 构建训练数据
    def build_training_pairs(self):
        print("Creating training pairs...")
        all_pairs = []
        for sentence in tqdm(self.corpus, desc="Processing sentences"):
            word_indices = [self.vocab.get(w, self.vocab['<UNK>']) for w in sentence]
            for i, center in enumerate(word_indices):
                context_indices = list(range(max(0, i-self.config.window_size), i)) + \
                                 list(range(i+1, min(len(word_indices), i+self.config.window_size+1)))
                for context in context_indices:
                    all_pairs.append((center, word_indices[context]))
        self.all_pairs = all_pairs
    # 训练数据采样
    def train_pair_sampling(self):
        if self.config.train_pair_sample_rate < 1.0:
            sample_size = int(len(self.all_pairs) * self.config.train_pair_sample_rate)
            self.data = random.sample(self.all_pairs, sample_size)
            print(f"Training pair sampling: using {sample_size}/{len(self.all_pairs)} pairs ({self.config.train_pair_sample_rate:.2%})")
        else:
            self.data = self.all_pairs
    # 计算负采样词频
    def calc_nagetive_sampling(self):
        self.word_freq = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_freq[i] = self.word_counts.get(self.index2word.get(i, '<UNK>'), 1)
        self.word_freq = self.word_freq ** 0.75
        self.word_freq = self.word_freq / self.word_freq.sum()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        center_word, context_word = self.data[index]
        
        neg_samples = []
        while len(neg_samples) < self.config.negative_samples:
            neg_idx = np.random.choice(self.vocab_size, p=self.word_freq)
            if neg_idx != context_word and neg_idx < self.vocab_size:
                neg_samples.append(neg_idx)
        
        if center_word >= self.vocab_size:
            center_word = self.vocab['<UNK>']
        if context_word >= self.vocab_size:
            context_word = self.vocab['<UNK>']
        
        return torch.tensor(center_word, dtype=torch.long), torch.tensor(context_word, dtype=torch.long), \
               [torch.tensor(idx, dtype=torch.long) for idx in neg_samples]