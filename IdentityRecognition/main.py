import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from config import Config
from model import BERTNERModel
from dataset import NERDataset
from trainer import Trainer


def main():
    config = Config()
    model = BERTNERModel(config=config)
    # 创建训练、验证和测试数据集
    train_dataset = NERDataset(
        config.train_file, config.train_tag_file, model.tokenizer, config.tag2idx, config.max_len
    )
    dev_dataset = NERDataset(
        config.dev_file, config.dev_tag_file, model.tokenizer, config.tag2idx, config.max_len
    )

    # 创建数据整理器以帮助在数据加载时进行动态填充
    data_collator = DataCollatorForTokenClassification(model.tokenizer)

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=data_collator)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=data_collator)

    # 创建模型实例

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 训练模型
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimizer=optimizer,
        config=config
    )

    trainer.train()

    trainer.test()


if __name__ == "__main__":
    main()
