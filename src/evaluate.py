import torch
from bert4rec_on_taobao.src.utils import get_device
import numpy as np
from   tqdm import tqdm 
from torch.utils.data import DataLoader
def calculate_metrics_batch(model, test_dataset, batch_size=128, k_values=[1, 5, 10,50,100],device="cpu"):
    model.eval()
    hrs = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            interactions, candidates, labels = batch
            interactions = interactions.to(device)
            candidates = candidates.to(device)
            # 获取模型预测
            logits = model(interactions)
            
            # 我们只关心最后一个时间步的预测
            last_logits = logits[:, -1, :]
            
            batch_size = interactions.size(0)
            for i in range(batch_size):
                # 获取候选项的得分
                candidate_scores = last_logits[i][candidates[i]]
                
                # 计算排序
                _, indices = torch.sort(candidate_scores, descending=True)
                
                # 找到正样本的排名
                rank = (indices == 0).nonzero().item() + 1  # +1 因为索引从0开始
                
                # 计算 HR 和 NDCG
                for k in k_values:
                    hrs[k].append(1 if rank <= k else 0)
                    ndcgs[k].append(1 / np.log2(rank + 1) if rank <= k else 0)
    # 计算平均值
    for k in k_values:
        hrs[k] = np.mean(hrs[k])
        ndcgs[k] = np.mean(ndcgs[k])
    return hrs, ndcgs

