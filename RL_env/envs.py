import numpy as np
import copy
import pickle
import subprocess
from typing import List, Dict
from src.graph.Strategy.AllocatingStrategy.LRFRedundantScheduling import LRFRedundantSchedulingStrategy
# from src.graph.Strategy.AllocatingStrategy.LRFRedundantScheduling2 import LRFRedundantSchedulingStrategy
from src.net_envs.network.TSNNetworkFactory import TSNNetworkFactory
from src.net_envs.network.TSNNetwork import TSNNetwork
from RL_env.BuildNetwork import BuildNetwork
from src.utils.ConfigFileGenerator2 import ConfigFileGenerator
from src.graph.FlowGenerator import FlowGenerator
from src import config
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WRAPPER ITSELF
def omnet_wrapper(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'

    prefix = ''
    if env.CLUSTER == 'arvei':
        prefix = '/scratch/nas/1/giorgio/rlnet/'

    simexe = prefix + 'omnet/' + sim + '/networkRL'
    simfolder = prefix + 'omnet/' + sim + '/'
    simini = prefix + 'omnet/' + sim + '/' + 'omnetpp.ini'

    try:
        omnet_output = subprocess.check_output([simexe, '-n', simfolder, simini, env.folder + 'folder.ini'])
    except Exception as e:
        omnet_output = e.stdout.decode()

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'ok'

    # vector_to_file([omnet_output], env.folder + OMLOG, 'a')


class ScheduleEnv(object):

    def __init__(self):
        self.current_step = 0
        self.record = []  # 记录每次state的变化
        self.flows = []       # 产生的TSN流集合
        self.num_candidates = 4  # 候选集冗余路径数
        self.num_redundant = 2  # 所需冗余路径数
        self.candidate_path_set: dict = {}  # 各TSN流的路径候选集
        self.max_route_num = 36  # 每个TSN流4个路径候选集的最大总路由跳数
        self.edge_list: list = []  # 记录网络的边
        self.edge_load: Dict = {}  # 记录边上load
        self._F_r: List[int] = []  # 记录所有TSN流的id
        self.current_step_time = 0
        self.current_step_time_len = 2
        self.edge_time_slot_record: Dict[list] = {}  # 记录全部TSN流路径的时间片分配情况
        self.flow_arrival_time_record: Dict = {}  # 记录TSN流的arrival time offset
        self.flow_allocation_num: Dict = {}  # 记录TSN流一条路径所需要的总的时间片数
        self.flow_allocation_num_record: Dict = {} #记录每个step后每个流的执行情况
        self.temp_flow_allocation: Dict = {}
        self.resource_reward: int = 0
        #self.flow_arrival_queue: list = []   # 记录到达的TSN流
        self.successful_flow_queue: list = []  # 记录成功到达的TSN流，任一冗余路径没有到达该TSN流都算失败
        self.current_flow_reward: int = 0
        self.temp_queue: list = []
        self.load_data()    # 加载所有数据

        # 定义动作空间
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.action_represent = {0: [0, 1], 1: [0, 2], 2: [0, 3], 3: [1, 2], 4: [1, 3], 5: [2, 3]}
        self.n_actions = len(self.action_space)

        # 定义状态空间
        # state space的大小 = 所有Flow流最大路径数+所有边数
        self.n_features = self.max_route_num + 1 + len(self.edge_list)  # + 1
        # self.n_features = self.max_route_num + 1
        # 状态空间的信息：TSN流路径信息 + 当前网络负载信息 + #flag位
        #  1.获得TSN流信息
        path_state: list = []
        self.path_states: Dict = {}   # 存储{1：routes[]；2：..}
        for _flow in self.flows:
            self._F_r.append(_flow.flow_id)
        for i in self._F_r:
            x = self.candidate_path_set[i][0]
            path_state.append(i)
            for y in x:
                for z in y:
                    path_state.append(z)
            self.path_states[i] = path_state
            path_state = []
        #  2.初始化边网络负载
        self.edge_load = np.zeros(len(self.edge_list))
        reshaped_path_states = self.pad_flow_info(1)
        # flag = [0]  标志是否为第二次选择路径
        #  3.组合所有state
        # self.state_space = reshaped_path_states
        self.state_space = np.concatenate((reshaped_path_states, self.edge_load))
        self.flow_scheduler = LRFRedundantSchedulingStrategy(self.graph.edge_mapper, self.graph.flow_mapper)
        print('创建环境')


    def load_data(self):
        # 加载TSN数据流
        self.flows = FlowGenerator.load_flows()
        # 加载TSN的拓扑图
        with open(config.graph_filename, "rb") as f:
            self.graph = pickle.load(f)
        # 加载候选路由集合
        self.candidate_path_set = FlowGenerator.load_routes()
        # 获得TSN网络边集合
        self.edge_list: list = [i + 1 for i in range(self.graph.get_edge_num())]


    def pad_flow_info(self, flow_id: int):
        a = self.max_route_num + 1 - len(self.path_states[flow_id])
        reshaped_path_states = np.pad(self.path_states[flow_id], (0, a), 'constant')
        return reshaped_path_states

    def step(self, action):
        self.current_step += 1
        # self.current_step_time += self.current_step_time_len
        reshaped_flow_info: list = []  # 临时存放填充好的更新后的流路径信息
        # 调度: 沿路由分配时间片
        # print('action', action)
        schedule_id = int(self.record[0])   # flow id
        path: list = self.candidate_path_set[schedule_id][0]
        path_ids: list = self.action_represent[action]
        choose_paths: list = []
        time_lens: list = []
        arrival_time_offsets: list = []
        overload_reward: int = 0
        for path_id in path_ids:
            choose_paths.append(path[path_id])
        # choose_paths = [choose_paths]
        self.graph.flow_mapper[schedule_id].routes = choose_paths
        self.flows[schedule_id-1].routes = choose_paths
        # print(choose_paths)
        for choose_path in choose_paths:
        #     #flow_scheduler = LRFRedundantSchedulingStrategy(self.tempgraph.edge_mapper, self.tempgraph.flow_mapper)
              self.failure_queue = self.flow_scheduler.schedule(schedule_id, choose_path, self.current_step_time)
             # print(edge_intervals)
        #     arrival_time_offset = self.flow_scheduler.arrival_time_offset
        #     time_len = self.flow_scheduler.allocation_num * len(choose_path)
        #     time_lens.append(time_len)
        #     arrival_time_offsets.append(arrival_time_offset)
        #     self.flow_arrival_time_record[schedule_id] = arrival_time_offset
        #     i = 0
        #     for edge in choose_path:
        #         self.edge_time_slot_record[edge] = edge_intervals[i]
        #         i += 1
        # self.flow_allocation_num[schedule_id] = time_lens
        # self.flow_allocation_num_record = copy.deepcopy(self.flow_allocation_num)
        # self.flow_arrival_time_record[schedule_id] = arrival_time_offsets
        # 网络负载变化 （网络负载计算方式：current_step_time时存在正在执行的时间片及其后有已分配好的时间片都算当前edge被占用的网络资源）
        # edge_load_sum = 0
        # for ed in self.edge_time_slot_record:
        #     edge_id = ed
        #     for interval in self.edge_time_slot_record[ed]:
        #         if self.current_step_time in interval or interval.lower > self.current_step_time:
        #             edge_load = interval.upper - interval.lower
        #             if edge_load == 0:
        #                 edge_load = 1
        #             edge_load_sum += edge_load
        #     self.edge_load[edge_id-1] = edge_load_sum / 585
        #     edge_load_sum = 0
        for e_id in self.edge_list:
            self.edge_load[e_id - 1] = (self.graph.edge_mapper[e_id].time_slot_allocator.load)*100
        # print(self.edge_load)
        if schedule_id == len(self.flows):
            # print('一个回合')
            # 1.调度所有流
            # self.flow_scheduler.schedule(self._F_r)
            # # 2.配置模拟器所需的 xml文件 (flows文件 / routes文件 / schedules文件)
            # # 2.1构造TSN实际网络
            # tsn_network_factory: TSNNetworkFactory = TSNNetworkFactory()
            # tsn_network: TSNNetwork = tsn_network_factory.product(
            #     self.graph, self.flows,
            #     enhancement_enable=config.XML_CONFIG['enhancement-tsn-switch-enable'])
            # # self.tsn_network = tsn_network
            # node_edge_mac_info = tsn_network_factory.node_edge_mac_info
            # # 2.2create test scenario
            # ConfigFileGenerator.create_test_scenario(tsn_network=tsn_network,
            #                                          graph=self.graph,
            #                                          flows=self.flows,
            #                                          node_edge_mac_info=node_edge_mac_info,
            #                                          solution_name='test_scenario1')
            # 3.启动 Omnet 网络模拟器
            #omnet_wrapper(self)
            # 4.从模拟器中获得奖励 Reward
            # 计算奖励值
            # for e_id in self.edge_list:
            #      self.edge_load[e_id-1] = self.graph.edge_mapper[e_id].time_slot_allocator.load
            # reward = -max(self.edge_load)
            # 来过滤掉链路为 0未经过链路的情况
            min_edge_load = min(filter(lambda x: x > 0, self.edge_load))
            reward = -(max(self.edge_load) - min_edge_load)
            # print('failure queue：', self.failure_queue)
            # print(min_edge_load)
            # print(max(self.edge_load))
            # print(self.edge_load)
            # print('reward', reward)
            done = True
        else:
            reward = 0
            done = False
        # self.edge_load = np.zeros(len(self.edge_list))
        # 转换路径信息
        if self.record[0] == len(self.flows):
            reshaped_flow_info = self.pad_flow_info(1)
        else:
            reshaped_flow_info = self.pad_flow_info(schedule_id + 1)
        # 根据转换的网络资源信息和路径信息构造 next_state
        self.record = np.concatenate((reshaped_flow_info, self.edge_load))
        next_state = copy.deepcopy(self.record)
        # next_state = reshaped_flow_info
        # print(next_state)


        # # reward function
        # # 查看流的到达情况
        # infer_flow_reward: int = 0
        # current_flow_reward: int = 0
        # if schedule_id == len(self.flows):
        #     infer_flow_reward = self.flow_arrival_infer_reward()
        # else:
        #     current_flow_reward = self.flow_arrival_reward()
        #
        # flow_reward = (infer_flow_reward + current_flow_reward) / len(self.flows)
        # # 网络资源情况
        # load_sum: list = []
        # #self.resource_reward: list = []
        # for edge_load in self.edge_load:
        #     load_sum.append(edge_load)
        #     if edge_load > 0.8:
        #         overload_reward = -20
        # #load_sum = load_sum / 120
        # #load_sum = max(load_sum)
        # #self.resource_reward += max(load_sum)
        # #self.resource_reward.append(max(load_sum))
        # self.resource_reward = max(self.edge_load)
        # resource_reward = self.resource_reward

        # 获得 Reward
        # if schedule_id == len(self.flows):
        #     # print('一个回合')
        #     # 1.调度所有流
        #     # self.flow_scheduler.schedule(self._F_r)
        #     # # 2.配置模拟器所需的 xml文件 (flows文件 / routes文件 / schedules文件)
        #     # # 2.1构造TSN实际网络
        #     # tsn_network_factory: TSNNetworkFactory = TSNNetworkFactory()
        #     # tsn_network: TSNNetwork = tsn_network_factory.product(
        #     #     self.graph, self.flows,
        #     #     enhancement_enable=config.XML_CONFIG['enhancement-tsn-switch-enable'])
        #     # # self.tsn_network = tsn_network
        #     # node_edge_mac_info = tsn_network_factory.node_edge_mac_info
        #     # # 2.2create test scenario
        #     # ConfigFileGenerator.create_test_scenario(tsn_network=tsn_network,
        #     #                                          graph=self.graph,
        #     #                                          flows=self.flows,
        #     #                                          node_edge_mac_info=node_edge_mac_info,
        #     #                                          solution_name='test_scenario1')
        #     # 3.启动 Omnet 网络模拟器
        #     #omnet_wrapper(self)
        #     # 4.从模拟器中获得奖励 Reward
        #     # 计算奖励值
        #     # for e_id in self.edge_list:
        #     #      self.edge_load[e_id-1] = self.graph.edge_mapper[e_id].time_slot_allocator.load
        #     reward = max(self.edge_load)
        #     print('reward', reward)
        #     done = True
        # else:
        #     reward = 0
        #     done = False
        # #total_reward = -resource_reward + flow_reward + overload_reward
        # # total_reward = -resource_reward + overload_reward
        # #total_reward = flow_reward
        # #self.current_step_time += self.current_step_time_len
        return next_state, reward, done

    def flow_arrival_infer_reward(self):
        infer_flow_reward: int = 0
        infer_queue = list(set(self._F_r) - set(self.successful_flow_queue))
        #infer_flow_reward = len(self.successful_flow_queue) * 1
        for infer_id in infer_queue:
            flow_infer = max(self.flow_arrival_time_record[infer_id])
            if flow_infer <= self.graph.flow_mapper[infer_id].deadline:
                infer_flow_reward += 1
        return infer_flow_reward

    def flow_arrival_reward(self):
        flow_reward: int = 0
        path_num: int = 0
        #self.temp_queue: list = []
        flag: int = 0
        temp1: int = 0
        if not self.temp_flow_allocation:
            self.temp_flow_allocation = copy.deepcopy(self.flow_allocation_num_record)
        else:
            key1 = list(self.flow_allocation_num_record.keys())
            key2 = list(self.temp_flow_allocation.keys())
            key_id = list(set(key1) - set(key2))
            self.temp_flow_allocation[key_id[0]] = (self.flow_allocation_num_record[key_id[0]])
            temp1 = len(set(self.temp_queue))
            #print('temp1', temp1)
            for flow_id in self.temp_flow_allocation:
                flow_alloaction_record = self.temp_flow_allocation[flow_id]
                for index in range(len(flow_alloaction_record)):
                    flow_alloaction_record[index] -= self.current_step_time_len
                    if flow_alloaction_record[index] <= 0:
                        if self.current_step_time*512 <= self.graph.flow_mapper[flow_id].deadline:
                            path_num += 1
                            flag += 1
                self.temp_flow_allocation[flow_id] = flow_alloaction_record
                if path_num == 2:
                    #temp1 = len(temp_queue)
                    self.temp_queue.append(flow_id)
                    #set(self.successful_flow_queue)
                    #print('self.successful_flow_queue',self.successful_flow_queue)
                path_num = 0
            temp2 = len(set(self.temp_queue))
            #print('temp2',temp2)
            flow_reward = temp2 - temp1
            #for temp_id in temp_queue:
            #    del self.temp_flow_allocation[temp_id]
            #for temp_id in temp_queue:
            #    del self.flow_allocation_num_record[temp_id]
            self.successful_flow_queue = set(self.temp_queue)
            #print('self.successful_flow_queue', self.successful_flow_queue)
        return flow_reward

    def reset(self):
        self.current_step = 0
        # self.update()
        # time.sleep(0.1)
        # return stat
        #self.schedule_sequence = []
        self.record = copy.deepcopy(self.state_space)
        self.edge_load = np.zeros(len(self.edge_list))
        self.current_step_time = 0
        self.edge_time_slot_record = {}
        self.successful_flow_queue = []
        self.temp_flow_allocation = {}
        self.flow_allocation_num_record = {}
        self.flow_allocation_num = {}
        self.flow_arrival_time_record = {}
        self.resource_reward = []
        self.flow_scheduler.reset()
        self.current_flow_reward = 0
        self.temp_queue: list = []
        self.failure_queue = []
        return self.state_space

    def render(self):
        pass


