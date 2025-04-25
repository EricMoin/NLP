# 基于Transformer的命名实体识别系统

本项目实现了一个基于BERT的命名实体识别（NER）系统，用于中文序列标注任务。

## 环境要求

- Python 3.6+
- PyTorch 1.10+
- Transformers 4.18+
- 其他依赖见 requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据格式

- `train.txt`: 训练数据，每行为空格分隔的字符序列
- `train_TAG.txt`: 对应的标签，格式与训练数据一致
- `dev.txt`: 开发集数据
- `dev_TAG.txt`: 开发集标签
- `test.txt`: 测试数据

## 使用方法

1. 确保所有数据文件放在与代码相同的目录中
2. 运行主程序：

```bash
python main.py
```

3. 程序会自动进行以下步骤：
   - 读取并处理训练数据和标签
   - 统计标签集
   - 构建并训练BERT-NER模型
   - 在开发集上评估模型性能
   - 保存训练过程中的loss和准确率曲线图
   - 在测试集上进行预测并生成标签文件 prediction.txt

## 模型说明

- 基础模型：BERT（bert-base-chinese）
- 训练批次大小：32
- 学习率：5e-5
- 训练轮数：5

## 输出文件

- `best_model.pt`: 最佳模型参数
- `training_curves.png`: 训练曲线图
- `prediction.txt`: 测试集预测结果
