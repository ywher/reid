from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid, trackid).
    - num_instances (int): number of instances per identity in a batch. N
    - batch_size (int): number of examples in a batch. K
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size #64
        self.num_instances = num_instances #8
        self.num_pids_per_batch = self.batch_size // self.num_instances #num of vehicle ID per batch, 8
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index) #from vehicel ID to index in image list 
        self.pids = list(self.index_dic.keys()) #len = 1802, vehicle ID list

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid] #how many images of this index
            num = len(idxs)
            if num < self.num_instances:#
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        '''
        {
            key: car_id
            value: list of batch_idexs [[index1, ... index8], ...]
        }
        '''
        for pid in self.pids: #vehicle ID, 1802
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances: #如果这个id的图片数量小于instane num 8, 则可重复采样至8
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances: #如果正好是8个instance
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch) #从现有的car_id中随机选择8个
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0: #移除采样完的car_id
                    avai_pids.remove(pid)

        return iter(final_idxs) #8*8=64

    def __len__(self):
        return self.length

