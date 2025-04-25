import os
import numpy as np
import torch


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    device: str
    batch_size: int
    learning_rate: float
    max_len: int
    num_workers: int
    num_epochs: int

    train_file: str
    train_tag_file: str
    dev_file: str
    dev_tag_file: str
    test_file: str
    tag2idx: dict[str, int]
    idx2tag: dict[int, str]
    output_file: str

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.max_len = 128
        self.num_workers = 4
        self.num_epochs = 1
        self.output_file = "prediction.txt"

        self.build_data()

    def build_data(self):
        set_seed(42)  # 设置随机种子以确保结果可复现
        data_path = "data"  # 定义数据目录路径
        # 构造训练、验证和测试文件的完整路径
        self.train_file = os.path.join(data_path, "train.txt")
        self.train_tag_file = os.path.join(data_path, "train_TAG.txt")
        self.dev_file = os.path.join(data_path, "dev.txt")
        self.dev_tag_file = os.path.join(data_path, "dev_TAG.txt")
        self.test_file = os.path.join(data_path, "test.txt")
        # 检查所有必要文件是否存在
        for file in [self.train_file, self.train_tag_file, self.dev_file, self.dev_tag_file, self.test_file]:
            if not os.path.exists(file):
                print(f"找不到文件: {file}")
                return

        # 构建标签集合
        tags = set()
        with open(self.train_tag_file, 'r', encoding='utf-8') as f:
            for line in f:
                tags.update(line.strip().split())

        # 添加特殊标签"X"用于BERT的子词标记
        tags.add("X")

        # 创建标签到索引和索引到标签的映射
        self.tag2idx = {tag: i for i, tag in enumerate(tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(tags)}

        print("标签集合:")
        print(tags)
        print(f"标签数量: {len(tags)}")

    def copy_with(self, **kwargs):
        config = Config()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config
