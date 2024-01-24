import random
import torch
import numpy as np
from collections import namedtuple
from core import mod_utils as utils

# Taken and adapted from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, *args):#在函数内部，args 变量将会是一个包含了传递给函数的所有参数的元组。不用*不行
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)#为什么添加none？通过在内存尚未填满时添加 None 或其他占位符来预分配一个空间

        reshaped_args = []
        for arg in args:
            reshaped_args.append(np.reshape(arg, (1, -1)))#（1，-1）表示只有一行，负一表示根据原始数据推断，会一次把args所有元素展平再传入
        self.memory[self.position] = Transition(*reshaped_args)#将重新塑造后的数据作为参数传递给 Transition 类创建一个新的 Transition 对象，('state', 'action', 'next_state', 'reward', 'done')
        self.position = (self.position + 1) % self.capacity

    def add_content_of(self, other):#将另一个回放内存对象的内容添加到当前回放内存对象中。
        """
        将另一个重播缓冲区的内容添加到此重播缓冲区
：param other：另一个重播缓冲区
        """
        latest_trans = other.get_latest(self.capacity)
        for transition in latest_trans:
            self.add(*transition)#这里的*代表对transition进行解包，不然会以单个元组方式传入，而不是很多元素

    def get_latest(self, latest):
        """
       返回其他缓冲区中的最新元素，最新元素位于返回列表的末尾
：param other：另一个重播缓冲区
：param latest：要返回的最新元素数
：return：包含最新元素的列表
        """
        if self.capacity < latest:
            latest_trans = self.memory[self.position:].copy() + self.memory[:self.position].copy()#整个缓冲区
        elif len(self.memory) < self.capacity:
            latest_trans = self.memory[-latest:].copy()#取列表中倒数第 latest 个元素到最后一个元素的切片。
        elif self.position >= latest:#position是当前会整除latest的索引
            latest_trans = self.memory[:self.position][-latest:].copy()#切片两次，第一次0到position，再取最后latest个元素
        else:#position比latest小，0到position个元素加上末尾-latest+self.position个元素
            latest_trans = self.memory[-latest+self.position:].copy() + self.memory[:self.position].copy()
        return latest_trans

    def add_latest_from(self, other, latest):#将其他缓冲区的最新样本添加到此缓冲区

        latest_trans = other.get_latest(latest)
        for transition in latest_trans:
            self.add(*transition)

    def shuffle(self):#打乱memory的顺序
        random.shuffle(self.memory)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        return state, action, next_state, reward, done

    def sample_from_latest(self, batch_size, latest):
        latest_trans = self.get_latest(latest)
        transitions = random.sample(latest_trans, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.device = device

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition: Transition):
        max_prio = self.priorities.max() if self.buffer else 1.0 ** self.prob_alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = len(self.buffer)

        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total) ** (-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=self.device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5) ** self.prob_alpha

    def __len__(self):
        return len(self.buffer)

