import os
from collections import deque
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from prometheus_api_client import PrometheusConnect
import redis
from kubernetes import client, config, watch
from kubernetes.client.models.v1_container_image import V1ContainerImage
import json
import time



prom = PrometheusConnect(url="http://192.168.56.11:30568")
def names(self, names):
    self._names = names
V1ContainerImage.names = V1ContainerImage.names.setter(names)
config.load_kube_config('config')
v1 = client.CoreV1Api()




class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 3, 
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size






class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQNAgent:
    # """DQN Agent interacting with environment.
    
    # Attribute:
    #     env (gym.Env): openAI Gym environment
    #     memory (ReplayBuffer): replay memory to store transitions
    #     batch_size (int): batch size for sampling
    #     epsilon (float): parameter for epsilon greedy policy
    #     epsilon_decay (float): step size to decrease epsilon
    #     max_epsilon (float): max value of epsilon
    #     min_epsilon (float): min value of epsilon
    #     target_update (int): period for target model's hard update
    #     gamma (float): discount factor
    #     dqn (Network): model to train and select actions
    #     dqn_target (Network): target model to update
    #     optimizer (torch.optim): optimizer for training dqn
    #     transition (list): transition information including 
    #                        state, action, reward, next_state, done
    #     use_n_step (bool): whether to use n_step memory
    #     n_step (int): step number to calculate n-step td error
    #     memory_n (ReplayBuffer): n-step replay buffer 
    #     """
    

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            n_step (int): step number to calculate n-step td error
        """
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n

        #added
        obs_dim = env.obs_dim
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        # memory for 1-step Learning
        self.memory = ReplayBuffer(
            obs_dim, memory_size, batch_size, n_step=1
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    

    def nodes_available(self, cpu_req, mem_req):
        ready_nodes = []
        available_memory = prom.custom_query(memory_query)
        available_cpu = prom.custom_query(cpu_query)
        for dic1, dic2 in zip(available_cpu, available_memory):
            if dic1['metric']['node'] == 'k3smaster':
                continue
            if float(dic1['value'][1]) > cpu_req and float(dic2['value'][1]) > mem_req:
                ready_nodes.append(dic1['metric']['node'])
        
        return ready_nodes

    


    def select_action_kube(self, state: np.ndarray, cpu, memory) -> np.ndarray:
        """Select an action from the input state."""

        available_nodes = self.nodes_available(cpu, memory)
        available_actions = [i for i, node in enumerate(self.env.ready_nodes) if node in available_nodes]


        # epsilon greedy policy
        if self.epsilon > np.random.random():
            #selected_action = self.env.action_space.sample()
            selected_action = np.random.choice(available_actions)
        else:
            # Get the Q-values for nodes in available_nodes
            q_values = self.dqn(torch.FloatTensor(state).to(self.device))
    
            # Choose the action with the highest Q-value from available_nodes
            selected_action = q_values[available_actions].argmax().item()

            # selected_action = self.dqn(
            #     torch.FloatTensor(state).to(self.device)
            # ).argmax()
            # selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            # but it still stores an n-step transition!!!!
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self._compute_dqn_loss(samples, self.gamma)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance.
        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** self.n_step
            n_loss = self._compute_dqn_loss(samples, gamma)
            loss += n_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def schedule(self, func_name, pod_name, cpu_req, memory_req, done):
        state = self.env.get_state(func_name)
        action = self.select_action_kube(state, cpu_req, memory_req)
        # next_state, reward, done = self.step_kube(action, pod_name)
        next_state, reward, done = self.step_kube(action, pod_name, func_name, done)
        # print('hello')
        # state = next_state
        # score += reward

        update_cnt = 0
        # if episode ends
        if done:
            state, _ = self.env.reset(seed=self.seed)
            scores.append(score)
            score = 0

        # if training is ready
        if len(self.memory) >= self.batch_size:
            loss = self.update_model()
            # losses.append(loss)
            update_cnt += 1
            
            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            # epsilons.append(self.epsilon)
            
            # if hard update is needed
            if update_cnt % self.target_update == 0:
                self._target_hard_update()
    
    def step_kube(self, action, pod, function, done):
        reward = self.env.step(action, pod)
        next_state = self.env.get_state(function)
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            # but it still stores an n-step transition!!!!
            if one_step_transition:
                self.memory.store(*one_step_transition)
                
        print('hello')
        return next_state, reward, done
        # next_state, reward, terminated, truncated, _ = self.env.step(action, pod)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            # but it still stores an n-step transition!!!!
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done
        
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        #state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()
                
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env
        
    def _compute_dqn_loss(
        self, 
        samples: Dict[str, np.ndarray], 
        gamma: float
    ) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(
            dim=1, keepdim=True
        )[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()




import gym
from gym import spaces
import numpy as np

cpu_query = 'sum(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (node) * 1000'
memory_query = 'sum(node_memory_MemAvailable_bytes) by (node) / 1024 / 1024'

class KubeEnvironment(gym.Env):
    def __init__(self, beta=0.5,discount_factor=0.9, max_episode_steps=50, num_of_functions=1):
        self.beta = beta
        self.discount_factor = discount_factor
        self.max_episode_steps = max_episode_steps

        self.ready_nodes = []
        available_cpu = prom.custom_query(cpu_query)
        
        for dic1 in available_cpu:
                
            if dic1['metric']['node'] == 'k3smaster':
                continue
            else:
                self.ready_nodes.append(dic1['metric']['node'])

        self.node_prices = {'k3sg1node1': 5.5, 'k3sg1node2': 6, 'k3sg2node1': 4.5}
        self.function_arrival_rates = {'float-operation': 5}
        self.vm_prices = {'k3sg1node1': 0.172, 'k3sg1node2': 0.172, 'k3sg2node1': 0.086}

        self.obs_dim = ((len(self.ready_nodes) * 4) + (len(self.ready_nodes) * 2) + len(self.ready_nodes) + (len(self.ready_nodes) * 2) + 
                            len(self.ready_nodes) + len(self.ready_nodes) + 2 + 2 + 2 + 1 + 1 + num_of_functions)
        self.action_space = spaces.Discrete(len(self.ready_nodes))

        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.WINDOW_SIZE = 5

        # self.state = self.get_state('float-operation')
        # self.steps_taken = 0

        

    def find_node(self, data, node_name):
        for item in data:
            if 'node' in item['metric'] and item['metric']['node'] == node_name:
                return item
            elif 'instance' in item['metric'] and item['metric']['instance'] == node_name:
                return item
        return None
    
    def get_state(self, func_name):
        state = ()

        #   1. The actual CPU, memory, network (sum of network
        #   bytes received and transmitted) and disk I/O (sum of disk
        #   read and write bytes) bandwidth utilization of each of the
        #   nodes in the cluster at time 𝑡
        node_cpu_utilization = prom.custom_query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')
        node_memory_utilization = prom.custom_query('100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)')
        node_disk_io_utilization = prom.custom_query('sum by (instance) (irate(node_disk_read_bytes_total[1m]) + irate(node_disk_written_bytes_total[1m]))')
        node_network_utilization = prom.custom_query('sum by (instance) (irate(node_network_receive_bytes_total[1m]) + irate(node_network_transmit_bytes_total[1m]))')

        for i in range(1, len(node_cpu_utilization)):
            node_value = node_cpu_utilization[i]['value']
            state += (float(node_value[1]),)

        for i in range(1, len(node_memory_utilization)):
            node_value = node_memory_utilization[i]['value']
            state += (float(node_value[1]),)
        
        for i in range(1, len(node_disk_io_utilization)):
            node_value = node_disk_io_utilization[i]['value']
            state += (float(node_value[1]),)
        
        for i in range(1, len(node_network_utilization)):
            node_value = node_network_utilization[i]['value']
            state += (float(node_value[1]),)
        
        #   2. The CPU and memory capacities of each of the nodes in the cluster
        node_cpu_capacity = prom.custom_query('count without(cpu, mode) (node_cpu_seconds_total{mode="idle"})')
        node_memory_capacity = prom.custom_query('node_memory_MemTotal_bytes')
        
        for i in range(1, len(node_cpu_capacity)):
            node_value = node_cpu_capacity[i]['value']
            state += (float(node_value[1]),)
        
        for i in range(1, len(node_memory_capacity)):
            node_value = node_memory_capacity[i]['value']
            state += (float(node_value[1]),)
        
        #   3. Unit price of each cluster node
        for node in self.node_prices:
            state += (self.node_prices[node],)
        
        #   4. The total of minimum CPU and memory requested by function instances
        #   running in each node at time 𝑡
        cpu_request_sum_by_node_results = prom.custom_query('sum by (node) (kube_pod_container_resource_requests{namespace="openfaas-fn", resource="cpu"})')
        memory_request_sum_by_node_results = prom.custom_query('sum by (node) (kube_pod_container_resource_requests{namespace="openfaas-fn", resource="memory"})')

        for node in self.ready_nodes:
            node_dict = self.find_node(cpu_request_sum_by_node_results, node)
            if node_dict != None:
                state += (float(node_dict['value'][1]),)
            else:
                state += (0,)
        
        for node in self.ready_nodes:
            node_dict = self.find_node(memory_request_sum_by_node_results, node)
            if node_dict != None:
                state += (float(node_dict['value'][1]),)
            else:
                state += (0,)
        
        #   5. The active status of each node. A node is considered
        #   to be active at time 𝑡 if it contains instances of functions for
        #   which user requests are received at the cluster at the time
        active_status_results = prom.custom_query('count(kube_pod_info{namespace="openfaas-fn"}) by (node)')
        for node in self.ready_nodes:
            node_dict = self.find_node(active_status_results, node)
            if node_dict != None:
                state += (1,)
            else:
                state += (0,)
        
        #   6. The number of replicas of function of type 𝑝ₖ already deployed on each node
        function_k_replicas = 'count(kube_pod_container_info{{namespace="openfaas-fn", container="{variable}"}}) by (node)'.format(variable=func_name)
        function_k_replicas_results = prom.custom_query(function_k_replicas)
        for node in self.ready_nodes:
            node_dict = self.find_node(function_k_replicas_results, node)
            if node_dict != None:
                state += (float(node_dict['value'][1]),)
            else:
                state += (0,)
        
        #   7. The minimum CPU and memory requested by the function instance 𝑝ⱼᵏ
        # function_cpu_request = prom.custom_query('kube_pod_container_resource_requests{{namespace="openfaas-fn", resource="cpu", container="{variable}"}}'.format(variable=func_name))
        # function_cpu_request_result = float(function_cpu_request[0]['value'][1])
        # function_memory_request= prom.custom_query('kube_pod_container_resource_requests{{namespace="openfaas-fn", resource="memory", container="{variable}"}}'.format(variable=func_name))
        # function_memory_request_result = float(function_memory_request[0]['value'][1])
        
        function_cpu_request_result = 0.2
        function_memory_request_result = 64
        state += (function_cpu_request_result, function_memory_request_result,)
        
        #   8. Sum of network bytes received during a single request 
        #   execution of 𝑘ᵗʰ function on average
        function_bytes_received = prom.custom_query('container_network_receive_bytes_total{{namespace="openfaas-fn", pod=~"{variable}.*"}}'.format(variable=func_name))
        sum_received = 0

        for dic in function_bytes_received:
            sum_received += int(dic['value'][1])

        # number of invocations of 𝑘ᵗʰ function
        num_of_invocations = prom.custom_query('gateway_function_invocation_total{{code="200", function_name="{variable}.openfaas-fn"}}'.format(variable=func_name))
        num_of_invocations_value = num_of_invocations[0]
        num_of_invocations_value = int(num_of_invocations_value['value'][1])
        
        result = (sum_received / num_of_invocations_value) if num_of_invocations_value != 0 else 0
        state += (result,)

        #   Sum of network bytes transmitted during a single request 
        #   execution of 𝑘ᵗʰ function on average
        function_bytes_transmitted = prom.custom_query('container_network_transmit_bytes_total{{namespace="openfaas-fn", pod=~"{variable}.*"}}'.format(variable=func_name))
        sum_transmitted = 0

        for dic in function_bytes_transmitted:
            sum_transmitted += int(dic['value'][1])

        result = (sum_transmitted / num_of_invocations_value) if num_of_invocations_value != 0 else 0
        state += (result,)

        #   9. Sum of disk read and write bytes during a single request 
        #   execution of 𝑘ᵗʰ function on average
        disk_reads_bytes = prom.custom_query('container_fs_reads_bytes_total{{pod=~"{variable}.*"}}'.format(variable=func_name))
        sum_read = 0

        for dic in disk_reads_bytes:
            sum_read += int(dic['value'][1])

        result = (sum_read / num_of_invocations_value) if num_of_invocations_value != 0 else 0
        state += (result,)

        disk_writes_bytes = prom.custom_query('container_fs_writes_bytes_total{{pod=~"{variable}.*"}}'.format(variable=func_name))
        sum_write = 0

        for dic in disk_writes_bytes:
            sum_write += int(dic['value'][1])

        result = (sum_write / num_of_invocations_value) if num_of_invocations_value != 0 else 0
        state += (result,)

        #   10. Request concurrency on each function instance of type 𝑝ᵏ
        num_of_replicas = prom.custom_query('count(kube_pod_info{{namespace="openfaas-fn", pod=~"{variable}.*"}})'.format(variable=func_name))
        num_of_replicas_value = num_of_replicas[0]
        num_of_replicas_value = int(num_of_replicas_value['value'][1])

        request_concurrency = self.function_arrival_rates[func_name] / num_of_replicas_value
        state += (request_concurrency,)

        #   11. Relative function response time (RFRT) of function
        #   of type 𝑝ᵏ in the cluster at time 𝑡. This is the ratio between
        #   the actual and standard response time (𝑟₀ᵏ) for the function
        
        ## Get the response times from Redis
        response_times = self.redis_client.lrange(func_name, 0, -1)

        ## Calculate and print average response time
        if len(response_times) < self.WINDOW_SIZE:
            ## Adjust window size if the number of response times is less than the specified window size
            window_size = len(response_times)
        else:
            window_size = self.WINDOW_SIZE
        
        if window_size > 0:
            avg_response_time = sum(map(float, response_times)) / window_size
        else:
            avg_response_time = 0.99
        
        state += (avg_response_time/0.99,)

        #   12. The request arrival rates for each different function
        #   deployed in the cluster at time 𝑡
        for key in self.function_arrival_rates:
            state += (self.function_arrival_rates[key],)
        return state

    def get_active_times(self):

        # Get all keys matching the pattern "active_time:*"
        keys = self.redis_client.keys("active_time:*")

        # Initialize an empty dictionary to store node names and their active times
        active_times_dict = {}

        # Iterate over the keys and retrieve the active time for each node
        for key in keys:
            node_name = key.decode().split(":")[1]  # Extract node name from the Redis key
            active_time = self.redis_client.get(key)  # Retrieve active time from Redis
            active_times_dict[node_name] = int(active_time) if active_time else 0  # Store active time in the dictionary

        return active_times_dict

    def step(self, action, pod):
        body = {
        'api_version': 'v1',
        'kind': 'Binding',
        'metadata': {
        'name': pod,
        'namespace': 'openfaas-fn'
        },
        'target': {
        'api_version': 'v1',
        'kind': 'Node',
        'name': self.ready_nodes[action]
        }}
        namespace = 'openfaas-fn'
        
        # cumulative cost of cluster VM usage just before the implementation of the action 𝑎ₜ
        sum_price_before = 0
        active_times = self.get_active_times()
        for at in active_times:
            sum_price_before += active_times[at] * (self.vm_prices[at]/3600)
        
        # Schedule function to a node based on neural net out put
        result = v1.create_namespaced_binding(namespace, body, _preload_content=False)
        response_dict = json.loads(result.data.decode('utf-8'))


        if response_dict['status'] == 'Success':
            print(f'Scheduled pod {pod} to node {action}')
            time.sleep(4)
            #   𝑅₁: The sum of RFRT calculated across all the
            #   deployed functions in the cluster, just before implementation
            #   of the next action, aₜ₊₁.
            deployments = prom.custom_query('kube_deployment_created{namespace="openfaas-fn"}')
            
            R1 = 0
            for deploy in deployments:
                ## Get the response times from Redis
                response_times = self.redis_client.lrange(deployments[0]['metric']['deployment'], -self.WINDOW_SIZE, -1)

                ## Calculate and print average response time
                if len(response_times) < self.WINDOW_SIZE:
                    ## Adjust window size if the number of response times is less than the specified window size
                    window_size = len(response_times)
                else:
                    window_size = self.WINDOW_SIZE
                
                if window_size > 0:
                    avg_response_time = sum(map(float, response_times)) / window_size
                else:
                    avg_response_time = 0.99
                
                R1 += avg_response_time/0.99
            
            # Cumulative cost of cluster VM usage just before the implementation of the action 𝑎ₜ₊₁
            sum_price_after = 0
            active_times = self.get_active_times()
            for at in active_times:
                sum_price_after += active_times[at] * (self.vm_prices[at]/3600)
            
            # R2 The difference in the cumulative cost of cluster
            # VM usage just before the implementation of the action, 𝑎ₜ
            # and just before the implementation of the next action, 𝑎ₜ₊₁
            R2 = sum_price_after - sum_price_before

            # Retrieve current minimum and maximum values from Redis
            min_R1 = float(self.redis_client.get('min_R1') or float('inf'))
            max_R1 = float(self.redis_client.get('max_R1') or float('-inf'))
            min_R2 = float(self.redis_client.get('min_R2') or float('inf'))
            max_R2 = float(self.redis_client.get('max_R2') or float('-inf'))

            # Update minimum and maximum values for R1 and R2
            if R1 < min_R1:
                min_R1 = R1
                self.redis_client.set('min_R1', min_R1)
            if R1 > max_R1:
                max_R1 = R1
                self.redis_client.set('max_R1', max_R1)
            if R2 < min_R2:
                min_R2 = R2
                self.redis_client.set('min_R2', min_R2)
            if R2 > max_R2:
                max_R2 = R2
                self.redis_client.set('max_R2', max_R2)
            
            main_reward = -(self.beta*((R1-min_R1)/(max_R1-min_R1))+(1-self.beta)*((R2-min_R2)/(max_R2-min_R2)))
            
            return main_reward
        else:
            raise MyCustomException("The pod did not schedule successfully")
        
        
class CarRentalEnvironment(gym.Env):
    def __init__(self, max_cars=20, max_move=5, move_cost=2,
                 rent_reward=10, discount_factor=0.9, max_episode_steps=50):
        #super(CarRentalEnvironment, self).__init__()
        
        self.max_cars = max_cars
        self.max_move = max_move
        self.move_cost = move_cost
        self.rent_reward = rent_reward
        self.discount_factor = discount_factor
        self.max_episode_steps = max_episode_steps  # Added maximum episode steps

        self.request_means = {
            1: [5, 3],  # Sunday
            2: [4, 3],  # Monday
            3: [3, 3],  # Tuesday
            4: [2, 1],  # Wednesday
            5: [1, 2],  # Thursday
            6: [4, 5],  # Friday
            7: [3, 5]   # Saturday
        }

        self.return_means = {
            1: [5, 4],  # Sunday
            2: [5, 3],  # Monday
            3: [4, 3],  # Tuesday
            4: [3, 3],  # Wednesday
            5: [2, 1],  # Thursday
            6: [1, 2],  # Friday
            7: [5, 4]   # Saturday
        }

        # Observation space: Number of cars at loc1, number of cars at loc2, and day of the week
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1], dtype=np.int32),  # Minimum values for each component
            high=np.array([max_cars, max_cars, 7], dtype=np.int32),  # Maximum values for each component
            dtype=np.int32
        )

        # Action space: Number of cars to move between locations
        self.action_space = spaces.Discrete(2 * max_move + 1)

        self.state = (0, 0, 1)
        self.steps_taken = 0

    def step(self, action):
        
        assert self.action_space.contains(action), "Invalid action!"

        # Get request and return rates for the current day of the week
        request_mean = self.request_means[self.state[2]]
        return_mean = self.return_means[self.state[2]]

        # Simulate car rental requests and returns based on the means for the day of the week
        rental_requests = [np.random.poisson(request_mean[i]) for i in range(2)]
        rental_returns = [np.random.poisson(return_mean[i]) for i in range(2)]

        # Calculate reward for renting cars
        rent_reward = self.rent_reward * min(self.state[0], rental_requests[0])
        rent_reward += self.rent_reward * min(self.state[1], rental_requests[1])

        # Update the state based on rentals and returns
        cars_at_loc1 = self.state[0] - rental_requests[0]
        if cars_at_loc1 < 0:
            cars_at_loc1 = 0
        cars_at_loc1 += rental_returns[0]
        if cars_at_loc1 > self.max_cars:
            cars_at_loc1 = self.max_cars

        cars_at_loc2 = self.state[1] - rental_requests[1]
        if cars_at_loc2 < 0:
            cars_at_loc2 = 0
        cars_at_loc2 += rental_returns[1]
        if cars_at_loc2 > self.max_cars:
            cars_at_loc2 = self.max_cars
        
        
        #first convert action index to action value
        #we also have to handle negative rewards for cars moved
        real_action = action - 5


        if real_action >= 0 and real_action <= 5:
            cars_at_loc1 -= real_action
            if cars_at_loc1 < 0:
                real_action = real_action + cars_at_loc1
                cars_at_loc1 = 0
    
            cars_at_loc2 += real_action
            if cars_at_loc2 > self.max_cars:
                cars_at_loc2 = self.max_cars
        elif real_action >=-5 and real_action < 0:
            cars_at_loc2 -= abs(real_action)
            if cars_at_loc2 < 0:
                real_action = real_action + abs(cars_at_loc2)
                cars_at_loc2 = 0
    
            cars_at_loc1 += abs(real_action)
            if cars_at_loc1 > self.max_cars:
                cars_at_loc1 = self.max_cars
        

        # Calculate total reward as the sum of rent_reward and moving cost (if any)
        total_reward = rent_reward - abs(real_action) * self.move_cost
        if total_reward < 0:
            total_reward = 0

        next_day = self.state[2] % 7 + 1  # Cycle through days of the week

        self.state = (cars_at_loc1, cars_at_loc2, next_day)
        self.steps_taken += 1

        # Check if the episode is truncated
        if self.steps_taken >= self.max_episode_steps:
            return self.state, total_reward, True, True, {}
            self.steps_taken = 0

        return self.state, total_reward, False, False, {}

    def reset(self, seed):
        # Reset the environment to the initial state
        self.state = (0, 0, 1)
        self.steps_taken = 0

        # Return the state as a NumPy array and an empty dictionary
        return np.array(self.state, dtype=np.int32), {}





# environment
#env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
#env = CarRentalEnvironment()



seed = 777
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True