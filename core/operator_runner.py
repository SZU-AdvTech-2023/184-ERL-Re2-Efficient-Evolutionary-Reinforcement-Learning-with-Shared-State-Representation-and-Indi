import os
import pickle
import numpy as np
import torch
from core import ddpg
from core import mod_neuro_evo


class OperatorRunner:
    def __init__(self, args, env):
        self.env = env
        self.args = args
    def load_genetic_agent(self, source, model):#加载遗传agent
        actor_path = os.path.join(source, 'evo_net_actor_{}.pkl'.format(model))
        buffer_path = os.path.join(source, 'champion_buffer_{}.pkl'.format(model))#根据指定路径source构造了actor和buffer的路径
        agent = ddpg.GeneticAgent(self.args)
        #.load_state_dict是 torch.nn.Module 类的一个方法，用于将一个保存的模型状态字典加载到神经网络模型的参数中。
        agent.actor.load_state_dict(torch.load(actor_path))#从文件路径中加载了预训练模型的参数到 agent 的 actor 神经网络中
        with open(buffer_path, 'rb') as file:
            agent.buffer = pickle.load(file)#pickle.load() 时，需要以二进制模式打开文件（例如 'rb'），并传递文件对象作为参数。
        return agent

    def evaluate(self, agent, trials=10):#用于评估 agent
        results = []
        states = []
        for trial in range(trials):
            total_reward = 0

            state = self.env.reset()
            if trial < 3:
                states.append(state)#前三个状态加入states
            done = False
            while not done:
                action = agent.actor.select_action(np.array(state))#选取动作，和td3一样

                # Simulate one step in environment
                next_state, reward, done, info = self.env.step(action.flatten())#（s’，r，done，其他参数）
                total_reward += reward
                state = next_state
                if trial < 3:
                    states.append(state)

            results.append(total_reward)
        return np.mean(results), np.array(states)#返回平均奖励和状态列表

    def test_crossover(self):#用于执行交叉操作
        source_dir = 'exp/cheetah_sm0.1_distil_save_20/models/'
        models = [1400, 1600, 1800, 2200]
        parent1 = []
        parent2 = []
        normal_cro = []
        distil_cro = []
        p1s, p2s, ncs, dcs = [], [], [], []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if j > i:
                    print("========== Crossover between {} and {} ==============".format(model1, model2))
                    critic = ddpg.Critic(self.args)
                    critic_path = os.path.join(source_dir, 'evo_net_critic_{}.pkl'.format(model2))
                    critic.load_state_dict(torch.load(critic_path))
                    agent1 = self.load_genetic_agent(source_dir, model1)
                    agent2 = self.load_genetic_agent(source_dir, model2)

                    p1_reward, p1_states = self.evaluate(agent1)
                    p2_reward, p2_states = self.evaluate(agent2)
                    parent1.append(p1_reward)
                    parent2.append(p2_reward)
                    p1s.append(p1_states)
                    p2s.append(p2_states)

                    ssne = mod_neuro_evo.SSNE(self.args, critic, None)
                    child1 = ddpg.GeneticAgent(self.args)
                    child2 = ddpg.GeneticAgent(self.args)
                    ssne.clone(agent1, child1)
                    ssne.clone(agent2, child2)#克隆亲代信息

                    ssne.crossover_inplace(child1, child2)

                    c1_reward, c1_states = self.evaluate(child1)
                    normal_cro.append(c1_reward)
                    ncs.append(c1_states)

                    child = ssne.distilation_crossover(agent1, agent2)
                    c_reward, c_states = self.evaluate(child)
                    distil_cro.append(c_reward)
                    dcs.append(c_states)

                    print(parent1[-1])
                    print(parent2[-1])
                    print(normal_cro[-1])
                    print(distil_cro[-1])
                    print()

        save_file = 'visualise/crossover'
        np.savez(save_file, p1=parent1, p2=parent2, nc=normal_cro, dc=distil_cro, p1s=p1s, p2s=p2s, ncs=ncs, dcs=dcs)

    def test_mutation(self):#用于执行突变操作
        models = [800, 1400, 1600, 1800, 2200]
        source_dir = 'exp/cheetah_sm0.1_distil_save_20/models/'

        pr, nmr, smr = [], [], []
        ps, nms, sms = [], [], []
        ssne = mod_neuro_evo.SSNE(self.args, None, None)
        for i, model in enumerate(models):
            print("========== Mutation for {} ==============".format(model))
            agent = self.load_genetic_agent(source_dir, model)
            p_reward, p_states = self.evaluate(agent)
            pr.append(p_reward)
            ps.append(p_states)

            nchild = ddpg.GeneticAgent(self.args)
            ssne.clone(agent, nchild)
            ssne.mutate_inplace(nchild)

            nm_reward, nm_states = self.evaluate(nchild)
            nmr.append(nm_reward)
            nms.append(nm_states)

            dchild = ddpg.GeneticAgent(self.args)
            ssne.clone(agent, dchild)
            ssne.proximal_mutate(dchild, 0.05)
            sm_reward, sm_states = self.evaluate(dchild)
            smr.append(sm_reward)
            sms.append(sm_states)

            print("Parent", pr[-1])
            print("Normal", nmr[-1])
            print("Safe", smr[-1])

        # Ablation for safe mutation
        ablation_mag = [0.0, 0.01, 0.05, 0.1, 0.2]
        agent = self.load_genetic_agent(source_dir, 2200)
        ablr = []
        abls = []
        for mag in ablation_mag:
            dchild = ddpg.GeneticAgent(self.args)
            ssne.clone(agent, dchild)
            ssne.proximal_mutate(dchild, mag)

            sm_reward, sm_states = self.evaluate(dchild)
            ablr.append(sm_reward)
            abls.append(sm_states)

        save_file = 'visualise/mutation'
        np.savez(save_file, pr=pr, nmr=nmr, smr=smr, ps=ps, nms=nms, sms=sms, ablr=ablr, abls=abls,
                 abl_mag=ablation_mag)

    def run(self):
        self.test_crossover()
        self.test_mutation()
