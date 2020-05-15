import random

import gym
import tensorflow as tf
from model import *
from collections import *
from tqdm import tqdm
epsilon = 0.1
alpha = 1
gamma = 1


def greedy_policy(action):
    r = random.random()
    if r >= epsilon:
        return action
    else:
        return random.choice([0, 1])


def make_sample(memory):
    input_state = []
    input_action = []
    input_q = []
    for done,last_state, action, reward, real_q, current_state, max_q in memory:
        input_state.append(last_state)
        input_action.append(action)
        q = real_q + alpha * (reward + gamma * max_q - real_q)
        if done:
            q = real_q + alpha * (reward - real_q)
        input_q.append(q)
    z = list(zip(input_state, input_action, input_q))
    # random.shuffle(z)
    #
    # input_state, input_action, input_q = zip(*z)
    return z
    # return input_state, input_action, input_q


env = gym.make('CartPole-v0')


observation = env.reset()

model = Model()

optimal_action, max_q, all_q = model.eval_all_q([observation])
action = greedy_policy(optimal_action[0])
real_q = all_q[0][action]
memory = deque()
episod = 0

step_count = 0

# for _ in range(100000):
#     # print(action, all_q[0], sep='\t')
#     env.render()
#     state = observation
#     observation, reward, done, info = env.step(random.choice([0,1]))  # take a random action
#     optimal_action, max_q, all_q = model.eval_all_q([observation])
#     real_q = all_q[0][optimal_action[0]]
#     step_count+=1
#     if done:
#         print(step_count)
#         observation = env.reset()
#         step_count = 0

# for i in tqdm(range(10000)):
#     last_state = observation  ## S
#     observation, _, done, info = env.step(action)  # take a random action a
#     current_state = observation
#     optimal_action, max_q, all_q = model.eval_all_q([current_state])
#     step_count += 1
#
#     # alpha = 1.0 / (i + 1)
#     if done:
#         observation = env.reset()
#         episod += 1
#         reward = -1
#         step_count = 0
#         optimal_action, _, all_q = model.eval_all_q([observation])
#     else:
#         reward = 0.1
#     memory.append([done, last_state, action, reward, real_q, current_state, max_q[0]])
#     # print(last_state,action,reward,sep='\t')
#     action = greedy_policy(optimal_action[0])
#     # print(optimal_action,action)
#
#     real_q = all_q[0][action]
#
#     if len(memory) >= 1000:
#         sample = make_sample(memory)
#         for _ in range(0, 1):
#             input_state, input_action, input_q = zip(*random.sample(sample, 32))
#             loss, _ = model.train(input_state, input_action, input_q)
#             # print(loss)
#         memory.popleft()
step_count = 0

env._max_episode_steps = 500
observation = env.reset()
optimal_action, max_q, all_q = model.eval_all_q([observation])
for _ in range(100000):
    env.render()

    a = 1
    if observation[2] > 0:
        a = 1
    else:
        a = 0
    observation, reward, done, info = env.step(a)  # take a random action

    # observation, reward, done, info = env.step(optimal_action[0])  # take a random action
    optimal_action, max_q, all_q = model.eval_all_q([observation])
    step_count += 1
    if done:
        print(step_count)
        observation = env.reset()
        step_count = 0
env.close()
