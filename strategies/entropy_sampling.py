import numpy as np
from .strategy import Strategy
import pdb

class EntropySampling(Strategy):
    def __init__(self, dataset, net=None):
        super(EntropySampling, self).__init__(dataset, net)

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

                token_probs = (token_output - np.min(token_output)) / (np.max(token_output) - np.min(token_output))
                token_probs = token_probs / np.sum(token_probs)
                token_entropy = -np.sum(token_probs * np.log2(token_probs + np.finfo(float).eps))
                seq_entropies.append(token_entropy)
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
        entropies = entropies[::-1] # 排序时只能是升序排列，而我们只关于熵比较高的，因此需要倒过来按降序排列
        # pdb.set_trace()
        return entropies[:n]
