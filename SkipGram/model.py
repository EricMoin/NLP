import torch
import numpy as np
from config import Config

class SGNSModel(torch.nn.Module):
    config: Config
    # 中心词嵌入
    center_embeddings: torch.nn.Embedding
    # 上下文词嵌入
    context_embeddings: torch.nn.Embedding
    # 词向量
    word_vectors: torch.Tensor
    # 词典大小
    vocab_size: int
    def __init__(self, config: Config,vocab_size:int):
        super(SGNSModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.center_embeddings = torch.nn.Embedding(vocab_size, config.embedding_dim)
        self.context_embeddings = torch.nn.Embedding(vocab_size, config.embedding_dim)
        
        # 改进初始化
        scale = 1.0 / np.sqrt(config.embedding_dim)  # Xavier初始化
        self.center_embeddings.weight.data.uniform_(-scale, scale)
        self.context_embeddings.weight.data.uniform_(-scale, scale)
    
    def forward(self, center_word, context_word, negative_samples):
        # Safety check
        center_word = torch.clamp(center_word, 0, self.vocab_size-1)
        context_word = torch.clamp(context_word, 0, self.vocab_size-1)
        negative_samples = torch.clamp(negative_samples, 0, self.vocab_size-1)
        
        center_embed = self.center_embeddings(center_word)
        context_embed = self.context_embeddings(context_word) 
        # 添加L2归一化提高稳定性
        center_embed = torch.nn.functional.normalize(center_embed, p=2, dim=1)
        context_embed = torch.nn.functional.normalize(context_embed, p=2, dim=1)
        
        pos_score = torch.sum(center_embed * context_embed, dim=1)
        pos_score = torch.clamp(torch.sigmoid(pos_score), min=1e-6, max=1-1e-6)
        
        neg_embeds = self.context_embeddings(negative_samples)
        neg_embeds = torch.nn.functional.normalize(neg_embeds, p=2, dim=2)
        
        neg_score = torch.bmm(neg_embeds, center_embed.unsqueeze(2)).squeeze(2)
        neg_score = torch.clamp(torch.sigmoid(neg_score), min=1e-6, max=1-1e-6)
        
        pos_loss = -torch.mean(torch.log(pos_score))
        neg_loss = -torch.mean(torch.log(1 - neg_score))
        
        return pos_loss + neg_loss, pos_loss.item(), neg_loss.item()
    
    def get_word_vector(self, word_idx):
        if word_idx >= self.vocab_size:
            word_idx = self.vocab_size - 1
        device = next(self.parameters()).device
        return self.center_embeddings(torch.tensor([word_idx], device=device)).cpu().detach().numpy()[0]
    
    # 获取平均向量
    def get_final_embeddings(self):
        # 返回中心词和上下文词向量的平均作为最终表示
        return (self.center_embeddings.weight.data + self.context_embeddings.weight.data) / 2