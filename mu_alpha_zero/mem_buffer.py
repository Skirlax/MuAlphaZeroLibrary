import itertools
import random
from collections import deque
from itertools import chain

import numpy as np
import pymongo
import torch as th
from diskcache import Deque
from torch.utils.data import Dataset, DataLoader

from mu_alpha_zero.General.memory import GeneralMemoryBuffer
from mu_alpha_zero.General.utils import find_project_root
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookAt
from mu_alpha_zero.MuZero.lazy_arrays import LazyArray
from mu_alpha_zero.MuZero.pickler import DataPickler
from mu_alpha_zero.MuZero.utils import scale_action


class MemDataset(Dataset):
    def __init__(self, mem_buffer):
        self.mem_buffer = list(mem_buffer)

    def __len__(self):
        return len(self.mem_buffer)

    def __getitem__(self, idx):
        return self.mem_buffer[idx]


class MemBuffer(GeneralMemoryBuffer):
    def __init__(self, max_size, disk: bool = False, full_disk: bool = False, dir_path: str = None,
                 hook_manager: HookManager or None = None):
        self.max_size = max_size
        self.disk = disk
        self.full_disk = full_disk
        self.dir_path = dir_path
        self.hook_manager = hook_manager if hook_manager is not None else HookManager()
        self.buffer = self.init_buffer(dir_path)
        self.eval_buffer = self.init_buffer(dir_path)
        self.last_buffer_size = 0
        self.priorities = None
        self.is_disk = disk

    def add(self, experience, is_eval: bool = False):
        if not isinstance(experience, tuple):
            raise ValueError("Experience must be a tuple")
        if self.disk and not self.full_disk and not isinstance(experience[-1], LazyArray):
            frame = LazyArray(experience[-1], self.dir_path)
            list_exp = list(experience)
            list_exp[-1] = frame
            experience = tuple(list_exp)

        if not is_eval:
            self.buffer.append(experience)
        else:
            self.eval_buffer.append(experience)

    def init_buffer(self, dir_path: str or None):
        if self.disk and self.full_disk:
            if dir_path is None:
                dir_path = f"{find_project_root()}/Pickles/Data"
            return Deque(maxlen=self.max_size, directory=dir_path)
        else:
            return deque(maxlen=self.max_size)

    def add_list(self, experience_list, percent_eval: int = 10):
        train = experience_list[:int(len(experience_list) * 0.9)]
        test = experience_list[int(len(experience_list) * 0.9):]
        for item in train:
            self.add(item)
        for item in test:
            self.add(item, is_eval=True)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def shuffle(self, is_eval: bool = False):
        if is_eval:  # in-place shuffle
            random.shuffle(self.eval_buffer)
        else:
            random.shuffle(self.buffer)

    def batch(self, batch_size: int, is_eval: bool = False):
        batched_buffer = []
        buf = self.buffer if not is_eval else self.eval_buffer
        buffer_len = len(buf)
        for idx in range(0, buffer_len, batch_size):
            batched_buffer.append(list(buf)[idx:min(idx + batch_size, buffer_len)])

        return batched_buffer

    def __call__(self, batch_size, is_eval: bool = False) -> list:
        return self.batch(batch_size, is_eval=is_eval)

    def __len__(self):
        return len(self.buffer)

    def batch_with_priorities(self, epochs, batch_size, K, alpha=1, is_eval: bool = False):
        buf = self.buffer if not is_eval else self.eval_buffer
        priorities = self.calculate_priorities(batch_size, alpha, K, is_eval=is_eval)
        self.priorities = priorities
        for _ in range(epochs):

            random_indexes = np.random.choice(np.arange(len(buf)),
                                              size=min(len(self.priorities) // K, max(batch_size // K, 1)),
                                              replace=False, p=self.priorities).tolist()
            batch = [list(itertools.islice(buf, i, i + K)) for i in random_indexes]
            pris = [self.priorities[i:i + K] for i in random_indexes]
            self.hook_manager.process_hook_executes(self, self.batch_with_priorities.__name__, __file__, HookAt.ALL,
                                                    args=(batch, pris))
            yield list(chain.from_iterable(batch)), th.tensor(list(chain.from_iterable(pris)), dtype=th.float32)

    def calculate_priorities(self, batch_size, alpha, K, is_eval: bool = False):
        buf = self.buffer if not is_eval else self.eval_buffer
        ps = [abs(buf[i][1] - buf[i][2][2]) for i in range(len(buf))]
        ps = np.array(ps)
        ps = ps ** alpha
        ps = ps / ps.sum()
        self.hook_manager.process_hook_executes(self, self.calculate_priorities.__name__, __file__, HookAt.ALL,
                                                args=(ps,))
        return ps

    def make_fresh_instance(self):
        return MemBuffer(self.max_size, self.disk, self.dir_path)

    def to_dataloader(self, batch_size):
        return DataLoader(MemDataset(self.buffer), batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    def run_on_training_end(self):
        self.hook_manager.process_hook_executes(self, self.run_on_training_end.__name__, __file__, HookAt.ALL)

    def save(self):
        if not self.disk:
            buffer = list(self.buffer)
            th.save(buffer, f"{find_project_root()}/Pickles/memory_buffer.pt")
            return f"{find_project_root()}/Pickles/memory_buffer.pt"
        else:
            return self.buffer.directory


class MuZeroFrameBuffer:
    def __init__(self, frame_buffer_size, noop_action: int, action_space_size: int):
        self.max_size = frame_buffer_size
        self.noop_action = noop_action
        self.action_space_size = action_space_size
        self.buffers = {1: deque(maxlen=frame_buffer_size), -1: deque(maxlen=frame_buffer_size)}

    def add_frame(self, frame, action, player):
        self.buffers[player][-1] = (self.buffers[player][-1][0], action)
        self.buffers[player].append((frame, self.noop_action))

    def concat_frames(self, player):
        frames_with_actions = [th.cat((th.tensor(frame, dtype=th.float32),
                                       th.full((frame.shape[0], frame.shape[1], 1), action, dtype=th.float32)), dim=2)
                               for frame, action in self.buffers[player]]
        # return th.tensor(np.array(frames_with_actions), dtype=th.float32)
        return th.cat(frames_with_actions, dim=2)

    def init_buffer(self, init_state, player):
        for _ in range(self.max_size):
            self.buffers[player].append((init_state, scale_action(self.noop_action, self.action_space_size)))

    def __len__(self, player):
        return len(self.buffers[player])


class MongoDBMemBuffer(GeneralMemoryBuffer):
    def __init__(self):
        self.db = pymongo.MongoClient("localhost", 27017).muzero
        self.calculated_buffer_size = 0
        self.is_disk = False
        self.full_disk = False

    def add(self, experience):
        if not isinstance(experience, dict):
            raise ValueError("Experience must be a dict")
        self.db.game_data.insert(experience)

    def add_list(self, experience_list):
        self.db.game_data.insert_many(experience_list)

    def batch(self, batch_size):
        random_idx = random.randint(0, self.db.game_data.count_documents({}) - batch_size)
        return list(self.db.game_data.find({}).skip(random_idx).limit(batch_size))

    def calculate_priorities(self, batch_size, alpha, K):
        self.calculated_buffer_size = self.db.game_data.count_documents({})
        fields = self.db.game_data.find({}, {"_id": 0, "pred_reward": 1, "t_reward": 1})
        ps = [abs(x["pred_reward"] - x["t_reward"]) ** alpha for x in fields]
        # add ps to db
        document_ids = self.db.game_data.find({}, {"_id": 1})
        for doc_id, p in zip(document_ids, ps):
            self.db.game_data.update_one(doc_id, {"$set": {"priority": p}})

    def update_priorities_if_needed(self, alpha, K):
        if self.calculated_buffer_size < self.db.game_data.count_documents({}):
            self.calculate_priorities(self.calculated_buffer_size, alpha, K)

    def batch_with_priorities(self, epochs, batch_size, K, alpha=1):
        for _ in range(epochs):
            self.update_priorities_if_needed(alpha, K)
            test_p = list(self.db.game_data.find({}, {"priority": 1, "_id": 0}).limit(3))
            # test_p = list(test_p)[0]["priority"]
            priorities = [x["priority"] for x in self.db.game_data.find({}, {"priority": 1, "_id": 0})]
            sum_p = sum(priorities)
            priorities = [p / sum_p for p in priorities]
            indexes = np.random.choice(np.arange(self.db.game_data.count_documents({})),
                                       size=min(self.calculated_buffer_size, batch_size // K), replace=False,
                                       p=priorities).tolist()
            items = [list(self.db.game_data.find({}).skip(x).limit(K)) for x in indexes]
            items = list(chain.from_iterable(items))
            items = tuple(
                [(x["probabilities"], x["vs"], (x["t_reward"], x["game_move"], x["pred_reward"]), x["game_state"]) for x
                 in items])
            yield items, th.tensor(priorities, dtype=th.float32)

    def get_last_greatest_id(self):
        return self.db.game_data.find_one(sort=[("_id", pymongo.DESCENDING)])["_id"]

    def __len__(self):
        return self.db.game_data.count_documents({})

    def drop_game_data(self):
        self.db.game_data.drop()

    def make_fresh_instance(self):
        return MongoDBMemBuffer()


class PickleMemBuffer(GeneralMemoryBuffer):

    def __init__(self, pickle_dir: str):
        self.pickle_dir = pickle_dir
        self.is_disk = True
        self.full_disk = True
        self.pickler = DataPickler(pickle_dir)

    def add(self, experience):
        raise NotImplementedError("Single experience addition shouldn't be performed in PickleMemBuffer, please add "
                                  "entire bach with add_list method")

    def add_list(self, experience_list):
        self.pickler.pickle_buffer(experience_list)  # self.pickler.push_to_consumer(experience_list)

    def batch(self, batch_size):
        raise NotImplementedError("Batch method not implemented for PickleMemBuffer, please use batch_with_priorities")

    def calculate_priorities(self, batch_size, alpha, K):
        vs = self.pickler.load_index(1)
        rews = self.pickler.load_index(2)
        ps = [(abs(vs[i] - rews[i][2]) ** alpha, i) for i in range(len(vs))][:-K]
        sum_p = sum([p[0] for p in ps])
        ps = [(p[0] / sum_p, p[1]) for p in ps]
        return {p[1]: p[0] for p in ps}

    def batch_with_priorities(self, epochs, batch_size, K, alpha=1):
        priorities = self.calculate_priorities(batch_size, alpha, K)
        ps_probs = np.array(list(priorities.values()))
        for _ in range(epochs):
            random_indexes = np.random.choice(np.arange(len(priorities)),
                                              size=min(len(priorities) // K, max(batch_size // K, 1)), replace=False,
                                              p=list(ps_probs)).tolist()
            batch = self.pickler.load_all(batch_size, random_indexes, K)
            pris = [list(priorities.values())[i:i + K] for i in random_indexes]
            tmp = list(chain.from_iterable(pris))
            yield batch, th.tensor(tmp, dtype=th.float32)

    def save(self):
        print("Load using datapickler.load_all(float(inf),...)")
        return self.pickle_dir

    def make_fresh_instance(self):
        return PickleMemBuffer(self.pickle_dir)

    def __len__(self):
        raise NotImplementedError("Length not implemented for PickleMemBuffer.")
