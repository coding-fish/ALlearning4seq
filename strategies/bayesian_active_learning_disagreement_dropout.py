import numpy as np
import torch
import pdb
from .strategy import Strategy

class BALD(Strategy):
    def __init__(self, dataset, net=None, n_drop=None):
        super(BALD, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        entropies = []
        for seq_idx, sample in enumerate(self.dataset):
            """
                对每条样本，计算其序列的信息熵
                这里采取的策略是，计算序列中感兴趣token的信息熵，并求平均值
                感兴趣的token指，输出中的正召回(非Other标签)token
            """
            seq_entropies = []
            seq_len = sum(sample['attention_mask']) # 样本真实长度，即`mask=1`的个数
            for token_idx in range(seq_len):
                """
                    对样本中的token逐个执行以下步骤：
                    1. argmax取其标签，根据labels中的定义，index(Other)=0，因此只统计argmax非0的token
                    2. 对于关注的token，根据归一化的`token_probs`计算其信息熵entropy，并添加到序列信息熵`seq_entropies`
                """
                token_output = np.array(sample['logits'][token_idx]) # 模型分类器的输出，未归一化，值域(-∞,+∞)
                if np.argmax(token_output) == 0: continue # argmax=0不参与统计

                probs = (token_output - np.min(token_output)) / (np.max(token_output) - np.min(token_output))
                pb = probs.mean(0)
                token_entropy1 = -np.sum(pb * np.log2(pb + np.finfo(float).eps))
                token_entropy2 = (-np.sum(probs * np.log2(probs + np.finfo(float).eps)))
                # token_entropy2 = np.mean(token_entropy2)
                token_entropy2 = token_entropy2 / len(probs)
                uncertainties = token_entropy2 - token_entropy1
                seq_entropies.append(uncertainties)
            """
                `seq_entropies`中保留了序列中预测非other的各个位置的信息熵
                取平均值，作为序列(即这条样本)的信息熵
            """
            seq_mean_entropy = np.mean(seq_entropies)
            if len(seq_entropies) == 0: # 可能整个序列都没有检测到正标签，需要catch这个case
                seq_mean_entropy = 0
                # print(self.dataset[seq_idx]['true_labels'])
            entropies.append((seq_idx, seq_mean_entropy))
        entropies.sort(key=lambda x:x[1])
        #entropies = entropies[::-1] # 排序时只能是升序排列，而我们只关于熵比较高的，因此需要倒过来按降序排列
        pdb.set_trace()
        return entropies[:n]
        # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        # pb = probs.mean(0)
        # entropy1 = (-pb*torch.log(pb)).sum(1)
        # entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        # uncertainties = entropy2 - entropy1
        # return unlabeled_idxs[uncertainties.sort()[1][:n]]
