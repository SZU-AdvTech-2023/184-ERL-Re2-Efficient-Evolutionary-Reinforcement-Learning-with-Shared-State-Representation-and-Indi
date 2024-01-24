
from torch.autograd import Variable
import random, pickle
import numpy as np
import torch
import os
import gym


class Tracker:#追踪器
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string;self.project_string = project_string#vars_string 是要跟踪的变量的名称列表。project_string 是要保存的文件的特定项目字符串。
        self.foldername = parameters.save_foldername#save_foldername用于指定保存文件的文件夹名称。
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] # 为每个在 vars_string 中的变量创建这样的信息列表，第一个用于存储fitnesses。第二个是fitness的平均值，第三个是要保存到csv文件中的fitnesses值
        self.counter = 0#每次调用 update 方法，counter 属性增加 1。
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)#没有文件夹则创建文件

    def update(self, updates, generation):
        self.counter += 1
        for update, var in zip(updates, self.all_tracker):#updates 是一个更新列表，包含了要跟踪的每个变量的fitness
            if update == None: continue
            var[0].append(update)

       ##如果列表长度超过了 conv_size，则删除最早的更新值，以限制列表的大小。
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        # 更新每个var的平均值
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % 4 == 0:  # 更新到csv文件中
            for i, var in enumerate(self.all_tracker):#enumerate() 函数来同时获取列表 self.all_tracker 中的元素以及它们对应的索引值。
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))#当前generation数和平均值一同更新到要保存到csv文件的值
                filename = os.path.join(self.foldername, self.vars_string[i] + self.project_string)#拼接文件名
                try:#在保存文件时，如果出现异常，例如无法写入文件等，会打印出 'Failed to save progress' 的错误信息。
                    np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')#'%.3f' 表示保存浮点数格式的数据，并且保留三位小数。delimiter=',' 指定了数据之间的分隔符为逗号。
                except:
                    print('Failed to save progress')


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def odict_to_numpy(odict):
    l = list(odict.values())
    state = l[0]
    for i in range(1, len(l)):
        if isinstance(l[i], np.ndarray):
            state = np.concatenate((state, l[i]))
        else: #Floats
            state = np.concatenate((state, np.array([l[i]])))
    return state

def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)

def is_lnorm_key(key):
    return key.startswith('lnorm')
#如果key 是以 'lnorm' 开头的，函数将返回 True



