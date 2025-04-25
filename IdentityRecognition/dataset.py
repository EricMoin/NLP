import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, text_file, tag_file=None, tokenizer=None, tag2idx=None, max_len=128):
        self.texts = []
        self.tags = []
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len

        # 读取文本数据
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text_lines = [line.strip() for line in f]

        # 如果有标签文件，则读取标签
        if tag_file:
            with open(tag_file, 'r', encoding='utf-8') as f:
                self.tag_lines = [line.strip() for line in f]
        else:
            self.tag_lines = None

        # 处理数据
        self.process_data()

    def process_data(self):
        for i, text_line in enumerate(self.text_lines):
            # 如果文本行为空，跳过
            if not text_line:
                continue

            chars = text_line.split()
            # 如果有标签，处理标签，否则使用占位符
            if self.tag_lines:
                tags = self.tag_lines[i].split()
                assert len(chars) == len(tags), f"第{i}行的字符数和标签数不匹配！"
            else:
                tags = ["O"] * len(chars)  # 测试数据用占位符

            self.texts.append(chars)
            self.tags.append(tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        chars = self.texts[idx]
        tags = self.tags[idx]

        # 如果使用BERT，则需要特殊处理
        if self.tokenizer:
            # 对于中文NER，我们可以直接将字符串列表连接为一个字符串
            text = " ".join(chars)

            # 使用tokenizer处理文本
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

            # 从字典中移除batch维度
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # 处理标签
            if self.tag2idx:
                # 转换标签为ID
                label_ids = [self.tag2idx.get(
                    t, self.tag2idx["O"]) for t in tags]

                # 填充标签ID到最大长度
                # 前两个位置是[CLS]和首个实际token的位置
                padding_length = self.max_len - len(label_ids) - 2

                # 标签序列: [CLS_label(X)] + label_ids + [SEP_label(X)] + [PAD_labels(O)]
                label_ids = [self.tag2idx["X"]] + label_ids + \
                    [self.tag2idx["X"]] + [self.tag2idx["O"]] * padding_length
                label_ids = label_ids[:self.max_len]  # 确保不超过max_len

                encoding["labels"] = torch.tensor(label_ids, dtype=torch.long)

            # 存储原始tokens和labels用于后处理
            # 这些不会被collator处理
            encoding["text_idx"] = idx

            return encoding
        else:
            # 简单字符级处理（非BERT）
            input_tensor = torch.tensor(
                [ord(c) % 10000 for c in ''.join(chars)], dtype=torch.long)
            if self.tag2idx:
                label_tensor = torch.tensor(
                    [self.tag2idx.get(t, 0) for t in tags], dtype=torch.long)
                return input_tensor, label_tensor
            return input_tensor, tags
