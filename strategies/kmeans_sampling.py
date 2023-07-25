import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import pdb

from .strategy import Strategy


class KMeansSampling(Strategy):
    def __init__(self, dataset, net=None):
        super(KMeansSampling, self).__init__(dataset, net)

    def query(self, n):
        """
            对每条样本，计算其序列的向量，并进行k-means聚类
            已知序列的feature矩阵(seq_len * feature_dim)
            我们使用RNN进行序列嵌入，对每个序列得到(1 * hidden_size)向量表示
        """
        feature_dim = len(self.dataset[0]['features'][0]) # 对当前数据集，特征维度是1024
        rnn = nn.RNN(input_size=feature_dim, hidden_size=32, batch_first=True)  # 使用32维的隐藏状态作为序列的向量表示
        embeddings = []
        with torch.no_grad():
            for seq_idx, sample in enumerate(self.dataset):
                feature_matrix = sample['features']
                seq_len = sum(sample['attention_mask']) # 样本真实长度，即`mask=1`的个数
                feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)[:seq_len]
                # pdb.set_trace()
                _, last_hidden = rnn(feature_tensor.unsqueeze(0)) # 输入维度顺序(batch_size, seq_length, input_size)
                sequence_vector = last_hidden.squeeze().numpy()
                embeddings.append(sequence_vector)
        embeddings = np.array(embeddings)

        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        """
            对序列聚类，选取n_query个聚类中心
            对每个中心，求与之最近的样本
            最后返回n个离各自中心最近的样本
        """
        cluster_idxs = cluster_learner.predict(embeddings) # 得到每个样本，最近中心的编号（样本所属类别）
        centers = cluster_learner.cluster_centers_[cluster_idxs] # 每个样本所属类别的中心
        dis = (embeddings - centers)**2 
        dis = dis.sum(axis=1) # 每个样本与类别中心的距离平方
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
        q_idxs = q_idxs.tolist()

        sample_distances = [(idx, cluster_idxs[idx]) for idx in q_idxs]
        # pdb.set_trace()

        return sample_distances
