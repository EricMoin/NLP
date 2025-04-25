import numpy
import torch
import tqdm
from config import CONFIG
from dataset import SkipGramDataset
from torch.utils.data import DataLoader
from model import SGNSModel
# 导入混合精度训练
from torch.amp import autocast, GradScaler
import numpy as np


def train():
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True
    
    dataset = SkipGramDataset(CONFIG)
    # 增加num_workers提高数据加载效率
    dataloader = DataLoader(dataset, batch_size=CONFIG.batch_size,
                            shuffle=True, num_workers=16, pin_memory=True,
                            prefetch_factor=2)
    
    model = SGNSModel(config=CONFIG, vocab_size=dataset.vocab_size)
    model = model.to(CONFIG.device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=CONFIG.lr, 
                                weight_decay=getattr(CONFIG, 'weight_decay', 0))
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler()
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    # Print model info
    print(f"Using device: {CONFIG.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    
    for epoch in range(CONFIG.epochs):
        total_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        batch_count = 0
        
        # 设置模型为训练模式
        model.train()
        
        progress_bar = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}")
        for batch_idx, (center, context, neg_samples) in enumerate(progress_bar):
            center = center.to(CONFIG.device, non_blocking=True)
            context = context.to(CONFIG.device, non_blocking=True)

            try:
                neg_samples = torch.stack(
                    [ns.to(CONFIG.device, non_blocking=True) for ns in neg_samples], dim=1)

                # 使用自动混合精度
                with autocast(device_type=CONFIG.device.type):
                    loss, pos_loss, neg_loss = model(center, context, neg_samples)
                
                # 使用缩放器处理反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 更新参数
                scaler.step(optimizer)
                scaler.update()

                # Update metrics
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_pos_loss += pos_loss
                    total_neg_loss += neg_loss
                    batch_count += 1
                else:
                    print(f"Warning: NaN loss detected in batch {batch_idx}")

                # 优化进度条更新频率，减少IO开销
                if batch_idx % 50 == 0:  # 从20改为50
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{total_loss/max(1, batch_count):.4f}"
                    })  # 减少显示的指标
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(1, batch_count)
        avg_pos_loss = total_pos_loss / max(1, batch_count)
        avg_neg_loss = total_neg_loss / max(1, batch_count)

        print(f"Epoch {epoch+1}/{CONFIG.epochs} - "
              f"Loss: {avg_loss:.4f} (Pos: {avg_pos_loss:.4f}, Neg: {avg_neg_loss:.4f})")
        
        # 更新学习率
        scheduler.step(avg_loss)

    # 设置模型为评估模式
    model.eval()
    
    # 在CUDA上预分配内存空间以减少碎片
    torch.cuda.empty_cache()
    
    # 先收集测试文件中出现的所有词
    test_words = set()
    with open(CONFIG.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) >= 2:
                test_words.add(words[0])
                test_words.add(words[1])
    
    # 只为测试集中的词和额外的1000个高频词生成向量
    # 这大大减少了不必要的向量计算
    priority_words = {word: idx for word, idx in dataset.vocab.items() 
                     if word in test_words or idx < 1000}
    
    print(f"Extracting vectors for {len(priority_words)} priority words")
    final_embeddings = model.get_final_embeddings().cpu().detach().numpy()
    
    word_vectors = {}
    for word, idx in tqdm.tqdm(priority_words.items(), desc="Extracting vectors"):
        if idx < len(final_embeddings):
            word_vectors[word] = final_embeddings[idx]
        else:
            word_vectors[word] = np.zeros(CONFIG.embedding_dim)
    
    return word_vectors, dataset.vocab


def get_result(word_vectors: dict, output_file: str):
    def cosine_similarity(v1, v2) -> float:
        norm1 = numpy.linalg.norm(v1)
        norm2 = numpy.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return numpy.dot(v1, v2) / (norm1 * norm2)
    with open(CONFIG.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    missing_words = set()
    for line in tqdm.tqdm(lines, desc="Calculating similarities"):
        words = line.strip().split()
        if len(words) >= 2:
            word1, word2 = words[0], words[1]
            sim:float = 0.0
            if word1 in word_vectors and word2 in word_vectors:
                sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
            else:
                if word1 not in word_vectors:
                    missing_words.add(word1)
                if word2 not in word_vectors:
                    missing_words.add(word2)
            results.append(f"{line.strip()}\t{sim:.6f}\n")
    print(f"Found {len(missing_words)} missing words in test set")
    if missing_words:
        print(f"Sample missing words: {list(missing_words)[:10]}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(results)

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    word_vectors, vocab = train()
    get_result(word_vectors, CONFIG.test_file)
