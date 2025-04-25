import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, DataCollatorForTokenClassification
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Optimizer
from config import Config
from dataset import NERDataset
from model import BERTNERModel


class Trainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, dev_dataloader: DataLoader, optimizer: Optimizer, config: Config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.config = config

        self.model.to(config.device)

    def train(self):
        best_dev_acc = 0
        train_losses = []
        dev_accuracies = []

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")

            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                loss, _ = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask, labels=labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            train_losses.append(avg_train_loss)
            print(
                f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")

            # 在开发集上评估
            dev_accuracy, dev_report = self.evaluate()
            dev_accuracies.append(dev_accuracy)
            print(f"Dev accuracy: {dev_accuracy:.4f}")
            print(dev_report)

            # 保存最佳模型
            if dev_accuracy > best_dev_acc:
                best_dev_acc = dev_accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
                print(
                    f"New best model saved with accuracy: {best_dev_acc:.4f}")

        # 绘制训练损失和开发集准确率曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(dev_accuracies)
        plt.title("Dev Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.close()

        return train_losses, dev_accuracies

    def evaluate(self):
        self.model.eval()
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            for batch in self.dev_dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(
                    self.config.device)
                labels = batch["labels"].to(self.config.device)
                text_indices = batch.get("text_idx", None)

                _, logits = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask)

                # 获取预测结果
                predictions = torch.argmax(logits, dim=2).cpu().numpy()
                labels_np = labels.cpu().numpy()
                mask_np = attention_mask.cpu().numpy()

                # 处理每个样本
                for i in range(len(predictions)):
                    pred = predictions[i]
                    label = labels_np[i]
                    mask = mask_np[i]

                    # 只处理有效token (mask=1)
                    valid_indices = np.where(mask == 1)[0]
                    valid_pred = [self.config.idx2tag[p]
                                  for p in pred[valid_indices]]
                    valid_label = [self.config.idx2tag[l]
                                   for l in label[valid_indices]]

                    # 去掉CLS和SEP对应的标签
                    valid_pred = valid_pred[1:-1]
                    valid_label = valid_label[1:-1]

                    # 获取原始tokens和标签
                    if text_indices is not None:
                        text_idx = text_indices[i].item()
                        orig_tokens = self.dev_dataset.texts[text_idx]
                        orig_labels = self.dev_dataset.tags[text_idx]

                        # 确保预测结果与原始tokens数量相同
                        # 由于BERT分词可能导致不一致，这里做一个简单的对齐
                        if len(valid_pred) > len(orig_tokens):
                            valid_pred = valid_pred[:len(orig_tokens)]
                        elif len(valid_pred) < len(orig_tokens):
                            valid_pred.extend(
                                ["O"] * (len(orig_tokens) - len(valid_pred)))

                        # 去掉X标签
                        valid_pred = [tag for tag in valid_pred if tag != "X"]
                        valid_label = orig_labels  # 使用原始标签

                    all_true_labels.extend(valid_label)
                    all_pred_labels.extend(valid_pred)

        # 计算准确率和其他指标
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        report = classification_report(all_true_labels, all_pred_labels)

        return accuracy, report

    # 预测函数

    def predict(self, dataloader: DataLoader, dataset: Dataset):
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(
                    self.config.device)
                text_indices = batch.get("text_idx", None)

                _, logits = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask)

                # 获取预测结果
                predictions = torch.argmax(logits, dim=2).cpu().numpy()
                mask_np = attention_mask.cpu().numpy()

                # 处理每个样本
                for i in range(len(predictions)):
                    pred = predictions[i]
                    mask = mask_np[i]

                    # 只处理有效token (mask=1)
                    valid_indices = np.where(mask == 1)[0]
                    valid_pred = [self.config.idx2tag[p]
                                  for p in pred[valid_indices]]

                    # 去掉CLS和SEP对应的标签
                    valid_pred = valid_pred[1:-1]

                    # 获取原始tokens
                    if text_indices is not None:
                        text_idx = text_indices[i].item()
                        orig_tokens = dataset.texts[text_idx]

                        # 确保预测结果与原始tokens数量相同
                        if len(valid_pred) > len(orig_tokens):
                            valid_pred = valid_pred[:len(orig_tokens)]
                        elif len(valid_pred) < len(orig_tokens):
                            valid_pred.extend(
                                ["O"] * (len(orig_tokens) - len(valid_pred)))

                    # 去掉X标签
                    pred_filtered = [tag for tag in valid_pred if tag != "X"]

                    # 如果过滤后的标签数量少于token数量，用"O"补齐
                    if text_indices is not None:
                        while len(pred_filtered) < len(orig_tokens):
                            pred_filtered.append("O")

                    all_predictions.append(pred_filtered)

        return all_predictions

    # 主函数

    def test(self):
        self.model.load_state_dict(torch.load("best_model.pt"))
        test_dataset = NERDataset(
            self.config.test_file, self.model.tokenizer, self.config.tag2idx, self.config.max_len
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, collate_fn=data_collator)
        # 使用最佳模型对测试集进行预测
        test_predictions = self.predict(
            self.model, test_dataloader, test_dataset, self.config)

        # 读取原始测试文件，确保行信息正确对应
        test_lines = []
        with open(self.config.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_lines.append(line.strip())

        # 将预测结果按照原始文件的格式写入
        pred_index = 0
        with open(self.config.output_file, 'w', encoding='utf-8') as f:
            for line in test_lines:
                if not line:
                    f.write("\n")
                else:
                    if pred_index < len(test_predictions):
                        f.write(" ".join(test_predictions[pred_index]) + "\n")
                        pred_index += 1
                    else:
                        # 如果预测结果不足，用"O"填充
                        f.write(" ".join(["O"] * len(line.split())) + "\n")
