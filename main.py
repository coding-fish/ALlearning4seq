# -*- coding: utf-8 -*-
# Author:   renchangyu
# Time:     2023/07/05
import argparse
import json
import numpy as np
import torch
from pprint import pprint
from tqdm import tqdm
import pdb

# 这里做了修改，把原来的目录copy出来一份，改为自己的数据格式
from strategies import RandomSampling, \
                       EntropySampling, \
                       KMeansSampling, \
                       BALD, \
                       LeastConfidence, \
                       MarginSampling, \
                       LeastConfidenceDropout, \
                       MarginSamplingDropout, \
                       EntropySamplingDropout, \
                       KCenterGreedy, \
                       AdversarialBIM, \
                       AdversarialDeepFool


class ActiveLearningDataPicker():
    def __init__(self):
        self.dataset_path = 'data/output.json'      # 数据集文件路径，该数据集中含有1682条样本
        self.dataset_len = 1682                     # 数据集长度，可通过os.system("wc -l data/output.json")可以得到
        self.args = None                            # 超参数
        self.dataset = None                         # 读入的数据集，包括feature,logits,labels等
        self.strategy = None                        # 策略函数
        self.picked_idxs = None                     # 策略选出的待标注数据
        self.output_dir = 'data/'                   # 待标注数据的idx
    
    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_query', type=int, default=1, help="number of queries per round")
        parser.add_argument('--strategy_name', type=str, default="EntropySampling", 
                            choices=["RandomSampling", 
                                    "LeastConfidence", 
                                    "MarginSampling", 
                                    "EntropySampling", 
                                    "LeastConfidenceDropout", 
                                    "MarginSamplingDropout", 
                                    "EntropySamplingDropout", 
                                    "KMeansSampling",
                                    "KCenterGreedy", 
                                    "BALD", 
                                    "AdversarialBIM", 
                                    "AdversarialDeepFool"], help="query strategy")
        parser.add_argument('--debug_mode', type=bool, default=False, help='for faster debug, just load a little samples')
        self.args = parser.parse_args()
        pprint(self.args)

    def load_dataset(self):
        dataset = []
        with open(self.dataset_path) as f:
            for idx, line in tqdm(enumerate(f, start=0), total=self.dataset_len, desc="Dataset processing"):
                if idx == 2 * self.args.n_query and self.args.debug_mode is True:
                                    break
                line = line.strip()
                data_obj = json.loads(line)
                dataset.append(data_obj)
        print(len(dataset))
        # pdb.set_trace()
        self.dataset = dataset

    def get_strategy(self):
        name = self.args.strategy_name
        if name == "RandomSampling":
            self.strategy = RandomSampling(self.dataset)
        elif name == "EntropySampling":
            self.strategy = EntropySampling(self.dataset)
        elif name == "KMeansSampling":
            self.strategy = KMeansSampling(self.dataset)
        elif name == "BALD":
            self.strategy = BALD(self.dataset)
            """
                TODO:下面的目前还没有实现
            """
        # elif name == "LeastConfidence":
        #     self.strategy = LeastConfidence(self.dataset)
        # elif name == "MarginSampling":
        #     self.strategy = MarginSampling(self.dataset)
        # elif name == "LeastConfidenceDropout":
        #     self.strategy = LeastConfidenceDropout(self.dataset)
        # elif name == "MarginSamplingDropout":
        #     self.strategy = MarginSamplingDropout(self.dataset)
        # elif name == "EntropySamplingDropout":
        #     self.strategy = EntropySamplingDropout(self.dataset)
        # elif name == "KCenterGreedy":
        #     self.strategy = KCenterGreedy(self.dataset)
        
        # elif name == "AdversarialBIM":
        #     self.strategy = AdversarialBIM(self.dataset)
        # elif name == "AdversarialDeepFool":
        #     self.strategy = AdversarialDeepFool(self.dataset)
        else:
            raise NotImplementedError

    def pick(self):
        n = self.args.n_query
        self.picked_samples = self.strategy.query(n)
        pprint(self.picked_samples)
        for idx, _ in self.picked_samples:
            # pdb.set_trace()
            ground_truth = self.dataset[idx]['true_labels']
            # print(ground_truth)

    def dump_picked_idxs(self):
        picked_idxs = [x[0] for x in self.picked_samples]
        picked_idxs.sort()
        output_path = self.output_dir + self.args.strategy_name + "_pick_idx.json"
        with open(output_path, 'w') as f:
            line = json.dumps(picked_idxs, ensure_ascii=False)
            print(line, file=f)

if __name__ == "__main__":
    agent = ActiveLearningDataPicker()
    agent.parse()
    agent.load_dataset()
    agent.get_strategy()
    agent.pick()
    agent.dump_picked_idxs()
