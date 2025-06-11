from Strategy.AllocatingStrategy.LRFRedundantScheduling import LRFRedundantSchedulingStrategy
from RL_env.Flow import Flow
from typing import List, Tuple
import networkx as nx
from RL_env.TopoGenerator import TopoGenerator
from .Graph import Graph
import config
import matplotlib.pyplot as plt
import copy
import numpy as np

class ScheduleEnv(object):
    def __init__(self):
        self.current_step = 0

        #generate flows
        self.flows: List[Flow] = [
            #  fid = 1, size = 20Kb, period = 150us, source = 1, destinations = [6], deadline = 1ms, path:[1, 4, 10, 13]
            #Flow(1, int(2e4), int(1.5e5), 1, [6], int(1e6), [[[1, 4, 10, 13]]]),
            #  fid = 2, size = 2Kb, period = 300us, source = 1, destinations = [7],  deadline = 2ms,path:[1, 4, 10, 14]
            #Flow(2, int(2e3), int(3e5), 1, [7], int(2e6), [[[1, 4, 10, 14]]]),
            Flow(1, int(1.2e3), int(1e6), 5, [7, 8],  int(2e7),[[[15, 1, 6, 13], [15, 2, 10, 13]], [[15, 1, 6, 14], [15, 2, 10, 14]]]),  # 125B 1000us 20ms
            Flow(2, int(2e3), int(1e6), 6, [5], int(2e7),[[[16, 5, 8, 3], [16, 6, 12, 8, 3]]]),  # 125B 1000us 20ms
            Flow(3, int(1e3), int(1e6), 7, [5, 6], int(2e7),[[[17, 11, 4, 3], [17, 11, 5, 8, 3], [17, 12, 8, 3], [17, 12, 9, 4, 3]], [[17, 11, 7], [17, 12, 9, 7]]]),  # 125B 1000us 20ms
            Flow(4, int(1.6e3), int(1e6), 8, [5], int(5e7),[[[18, 11, 4, 3], [18, 11, 5, 8, 3], [18, 12, 8, 3], [18, 12, 9, 4, 3]]]),  # 200B 1000us 50ms
            Flow(5, int(2.4e3), int(1e6), 6, [7, 8], int(5e7),[[[16, 5, 10, 13], [16, 6, 13]], [[16, 5, 10, 14], [16, 6, 14]]]),  # 300B 1000us 50ms
            Flow(6, int(2.4e3), int(1.5e6), 7, [5, 6], int(5e7),[[[17, 11, 4, 3], [17, 11, 5, 8, 3], [17, 12, 8, 3], [17, 12, 9, 4, 3]], [[17, 11, 7], [17, 12, 9, 7]]]),  # 300B 1500us 50ms
            Flow(7, int(5e3), int(1.5e6), 5, [8], int(1e8),[[[15, 1, 6, 14], [15, 2, 10, 14]]]),  # 625B 1500us 100ms
            Flow(8, int(5e3), int(1.5e6), 6, [8], int(1.5e8),[[[18, 11, 4, 3], [18, 11, 5, 8, 3], [18, 12, 8, 3], [18, 12, 9, 4, 3]]]),  # 625B 1500us 100ms
            Flow(9, int(1.2e3), int(3e6), 8, [5, 6], int(1.5e8),[[[16, 5, 10, 14], [16, 6, 14]]]),  # 150B 3000us 100ms
            Flow(10, int(1.2e3), int(3e6), 8, [5], int(1e8),[[[18, 11, 4, 3], [18, 11, 5, 8, 3], [18, 12, 8, 3], [18, 12, 9, 4, 3]], [[18, 11, 7], [18, 12, 9, 7]]]),  # 150B 3000us 100ms
        ]

        #generate Graph
        self.nodes: List[int] = [1, 2, 3, 4, 5, 6, 7]

        #self.edges: List[Tuple[int]] = [(1, 2), (2, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3),
        #                                (4, 5), (5, 4), (5, 6), (6, 5), (5, 7), (7, 5)]
        self.edges: List[Tuple[int, int]] = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (5, 1), (6, 2), (4, 7), (4, 8)]

        self.graph: nx.Graph = nx.Graph()
        self.graph.add_edges_from(self.edges)
        self.graph = self.graph.to_directed()
        TopoGenerator.draw(self.graph)
        options: dict = {
            'with_labels': True,
            'font_weight': 'bold',
        }
        plt.subplot(121)
        nx.draw(self.graph,**options)
        plt.show()
        self.graph: Graph = Graph(self.graph,
                             self.graph.nodes,
                             self.graph.edges,
                             config.GRAPH_CONFIG['hyper-period'])

        #get edge
        self.edge_list: list = [i+1 for i in range(self.graph.get_edge_num())]
        self.graph.add_flows(self.flows)
        # TODO set bandwidth to edges
        self.graph.set_all_edges_bandwidth(config.GRAPH_CONFIG['all-bandwidth'])  # set bandwidth
        # TODO set error rate to edges
        self.graph.set_all_error_rate(config.GRAPH_CONFIG['all-per'])  # set error rate
        # TODO set propagation delay to edges
        self.graph.set_all_edges_process_delay(config.GRAPH_CONFIG['all-propagation-delay'])
        # TODO set process delay to edges
        self.graph.set_all_edges_process_delay(config.GRAPH_CONFIG['all-process-delay'])
        self.LRFRschedule = LRFRedundantSchedulingStrategy(self.graph.edge_mapper,
                                                           self.graph.flow_mapper)

        #define action space
        self.action_space = []
        for i in range(10):
            self.action_space = self.action_space + [i + 1]
        print(self.action_space)
        self.n_actions = len(self.action_space)

        #define state space
        self.n_features = 4*self.n_actions
        flag = 0

        self.state_space = np.array([
            #  fid = 1, size = 20Kb, period = 150us, deadline = 1ms
            #[1, float(2e4/2e4), float(1.5e5/3e5), float(1e6/2e6),0],
            #  fid = 2, size = 2Kb, period = 300us, deadline = 2ms
            #[2, float(2e3/2e4), float(3e5/3e5),  float(2e6/2e6),0],
            [1, float(1.2e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0],
            [2, float(2e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0],
            [3, float(1e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0],
            [4, float(1.6e3/5e3), float(1e6/3e6), float(5e7/1.5e8),0],
            [5, float(2.4e3/5e3), float(1e6/3e6),  float(5e7/1.5e8),0],
            [6, float(2.4e3/5e3), float(1.5e6/3e6), float(5e7/1.5e8),0],
            [7, float(5e3/5e3), float(1.5e6/3e6), float(1e8/1.5e8),0],
            [8, float(5e3/5e3), float(1.5e6/3e6), float(1.5e8/1.5e8),0],
            [9, float(1.2e3/5e3), float(3e6/3e6), float(1.5e8/1.5e8),0],
            [10,float(1.2e3/5e3), float(3e6/3e6), float(1e8/1.5e8),0]


        ])
        #self.state_space_dealed = np.array([float(2e4/2e4), float(1.5e5/3e5), float(1e6/2e6),0,
        #                                    float(2e3/2e4), float(3e5/3e5),float(2e6/2e6),0])
        self.state_space_dealed = np.array([float(1.2e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0,
                                            float(2e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0,
                                            float(1e3/5e3), float(1e6/3e6), float(2e7/1.5e8),0,
                                            float(1.6e3/5e3), float(1e6/3e6), float(5e7/1.5e8),0,
                                            float(2.4e3/5e3), float(1e6/3e6), float(5e7/1.5e8),0,
                                            float(2.4e3/5e3), float(1.5e6/3e6), float(5e7/1.5e8),0,
                                            float(5e3/5e3), float(1.5e6/3e6), float(1e8/1.5e8),0,
                                            float(5e3/5e3), float(1.5e6/3e6), float(1.5e8/1.5e8),0,
                                            float(1.2e3/5e3), float(3e6/3e6), float(1.5e8/1.5e8),0,
                                            float(1.2e3/5e3), float(3e6/3e6), float(1e8/1.5e8),0])
        self.feature = 4

        #self.record:list = self.state_space
        self.schedule_sequence: list = []
        print(self.state_space)
        #temp_list = []
        #for each_flow in self.flows:
        #    temp_flow = []
        #    for flow_info in (each_flow[0:4]):
        #        temp_flow.append(flow_info)
        #    temp_list.append(temp_flow)
        #self.state_space = temp_list
        print(self.graph.flow_mapper)


    def step(self,action):
        #next state
        flag:int = 0
        if self.record[3+self.feature*action] == 1:
            flag = 1
        self.record[3+self.feature*action] = 1
        next_state = copy.deepcopy(self.record)

        # 记录action顺序即调度顺序
        temp_action = copy.deepcopy(action)+1
        if temp_action not in self.schedule_sequence:
            self.schedule_sequence.append(temp_action)

        # reward function
        num: int = 0
        reward: float = 0
        for i in range(3,len(self.record),self.feature):  # 通过flag的数量判断是否为最终状态
            record:int = self.record[i]
            if record == 1:
                num += 1
        #num = 2
        if num == len(self.action_space):
            done = True
            reward = self.LRFRschedule.schedule(self.schedule_sequence)
            print("reward:", reward)  # Get the total number of time slots used
        else:
            if flag == 1:
                reward = -100
            else:
                reward = 0
            done = False
        """"
        temp_state_list:list = []
        flag:int = 0
        self.edge_list: list = [1, 4, 10, 13, 14]
        #next state
        for i in range(len(self.record)):
            temp_state:list = self.record[i]
            if temp_state[0] == action+1:
                temp:int = temp_state[4]
                if temp == 0:
                    flag = 1                  #标识之前是否被调度过
                temp_state[4] = 1
                temp_state_list.append(temp_state)
            else:
                temp_state_list.append(temp_state)
        next_state = temp_state_list
        self.record = temp_state_list
        

        #记录action顺序即调度顺序
        self.schedule_sequence.append(action)

        #reward function
        num:int = 0
        for i in range(len(self.record)):      #通过flag的数量判断是否为最终状态
            record:list = self.record[i]
            if record[4] == 1:
                num += 1
        num = 2
        if num == len(self.action_space):
            done = True
            reward = self.LRFRschedule.schedule(self.schedule_sequence,self.edge_list)
            print("reward:",reward)# Get the total number of time slots used
        else:
            if flag ==0:
                reward = -100
            reward = 0
            done = False
        """
        return next_state, reward, done

    def reset(self):
        self.current_step = 0
        #self.update()
        #time.sleep(0.1)
        # return state
        self.schedule_sequence = []
        self.record = copy.deepcopy(self.state_space_dealed)
        return self.state_space_dealed

    def render(self):
        pass



#a = ScheduleEnv()











        """    
        self.record.append([self.current_step+1,action])
        self.current_step+=1
        #next state
        if self.current_step == self.n_features
        if state[0] == self.n_features:
            next_state = self.state_space[0]
        next_state = self.state_space[state[0]+1]
        action_used: list = []
        schedule_list : list = []
        relation :list = []
        flow_id_list : list = []
        relation.append(state)
        relation.append(action)
        schedule_list.append(relation)
        schedule_list.sort(key=lambda x: x[1])
        for i in range(len((schedule_list))):
            a = schedule_list[i]
            flow_id_list.append(a[0])
        reward: int = 0

        #reward function
        #judge whether all states are scheduled
        if next_state[0] == 1:
            done = True
            # Avoid action space selected
            if action in action_used:
                reward = -100
            current_flow = self.flows[state[0]]
            reward = LRFRedundantScheduling.schedule(flow_id_list,self.edge_list)      #Get the total number of time slots used
        else:
            done = False
            if action in action_used:
                reward = -100
            reward = 0
        """

    """
    def reset(self):
        self.current_step=0
        self.update()
        time.sleep(0.1)
        # return state
        return self.state_space[0]
    def render(self):
        pass
        a = ScheduleEnv()
    """