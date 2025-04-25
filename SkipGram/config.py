import torch


class Config:
    # 语料库路径
    corpus_path: str
    # 测试文件路径
    test_file: str
    # 词嵌入维度
    embedding_dim: int
    # 最小词频
    min_count: int
    # 窗口大小
    window_size: int
    # 负采样数量
    negative_samples: int
    # 训练轮数
    epochs: int
    # 批量大小
    batch_size: int
    # 学习率
    lr: float
    # 语料库采样率
    corpus_sample_rate: float
    # 训练数据采样率
    train_pair_sample_rate: float
    # 设备
    device: torch.device
    # 权重衰减
    weight_decay: float

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_config():
        return Config(
            corpus_path='data/word_based.txt',
            test_file='data/pku_sim_test.txt',
            embedding_dim=200,
            min_count=5,
            window_size=5,
            negative_samples=8,
            epochs=5,
            batch_size=256,
            lr=0.001,
            corpus_sample_rate=1.0,
            train_pair_sample_rate=1.0,
        )

    @staticmethod
    def get_fast_config():
        return Config(
            corpus_path='data/word_based.txt',
            test_file='data/pku_sim_test.txt',
            embedding_dim=100,
            min_count=5,
            window_size=10,
            negative_samples=5,
            epochs=5,
            batch_size=512,
            lr=1e-5,
            corpus_sample_rate=0.1,
            train_pair_sample_rate=0.1,
        )
    @staticmethod
    def get_faster_improved_config():
        return Config(
            corpus_path='data/word_based.txt',
            test_file='data/pku_sim_test.txt',
            embedding_dim=200,
            min_count=5,
            window_size=10,
            negative_samples=8,
            epochs=3,
            batch_size=256,
            lr=1e-3,
            corpus_sample_rate=0.5,
            train_pair_sample_rate=0.5,
            weight_decay=1e-6,
        )


CONFIG = Config.get_config()
CONFIG.device = torch.device('cuda')
for arg in vars(CONFIG):
    print(f"{arg}: {getattr(CONFIG, arg)}")