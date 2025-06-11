import os
from enum import Enum

from src.type import TIME_GRANULARITY

src_dir: str = os.path.dirname(os.path.abspath(__file__))
pro_dir: str = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
res_dir: str = os.path.join(pro_dir, 'res')
demo_dir: str = os.path.join(res_dir, 'demo1')
train_dir: str = os.path.join(demo_dir, 'train')
test_dir: str = os.path.join(demo_dir, 'test')
# dataset内的信息存储路径
dataset_dir: str = os.path.join(train_dir, 'dataset')
flows_filename: str = os.path.join(dataset_dir, 'flows.json')
routes_filename: str = os.path.join(dataset_dir, 'routes.json')
graph_filename: str = os.path.join(dataset_dir, 'graph')
graph: str = os.path.join(dataset_dir, 'graph.png')
helper_dir: str = os.path.join(dataset_dir, 'helper.txt')
# test_scenario模拟器配置文件存储路径
test_scenario_dir: str = os.path.join(train_dir, 'test_scenario')
# flow_size_res_dir: str = os.path.join(res_dir, 'flow_size')
# graph_size_res_dir: str = os.path.join(res_dir, 'graph_size')
# redundancy_res_dir: str = os.path.join(res_dir, 'redundancy')
solutions_res_dir: str = os.path.join(res_dir, 'solutions')
# flow_routes_repetition_degree_dir: str = os.path.join(res_dir, 'flow_routes_repetition_degree')
template_dir: str = os.path.join(src_dir, 'templates')

GRAPH_CONFIG = {
    'min-flow-size': 64 * 8,  # minimum frame size = 64B, [unit: Byte]
    'hyper-period': int(3e5),  # 300us = 3e5ns, [unit: ns],
    'all-bandwidth': int(1e0),  # 1Gbps = 1bit/ns, [unit: bpns]
    'max-bandwidth': int(1e0),  # maximum bandwidth of all edges
    'all-propagation-delay': 1e2,  # propagation delay of all edges
    'all-process-delay': 5e3,  # process daley of all edges
    'all-per': 0.02,
    'overlapped-routing': True,  # whether routing with overlapping or not
    'time-granularity': TIME_GRANULARITY.NS,  # time granularity, default is ns
    'edge-nodes-distribution-degree': 6,  # distribution degree of edge nodes
    'core-node-num': 8,  # [2, 4, 6, ..., 20]
    'edge-node-num': 8,
    'max-try-times': 50,  # max retry times if the graph is not connected,
    'visible': False,  # whether visualizing or not
}

FLOW_CONFIG = {
    'flow-num': 10,  # number of flow
    'dest-num-set': [1],  # set of the number of destination nodes
    'period-set': [int(3e5), int(1.5e5), int(1e5)],  # set of period, 周期能被最大周期整除
    'hyper-period': int(3e5),  # 300us = 3e5ns, [unit: ns],
    'size-set': [int(1.5e9), int(7e9), int(8e9),int(9e9), int(8.2e9), int(1.6e9)],  # 1500bit, [unit: bit],
    'reliability-set': [0.95, 0.90, 0.94],
    'deadline-set': [int(1e6), int(5e6), int(2e6)],
    'redundancy_degree': 1,
    'max-redundancy-degree': 5,
    'max-hops': 10,  # max hops
    'un-neighbors_degree': 1.0, # avoid source and node connecting at the same node
    'path-num': 4
}

OPTIMIZATION = {
    'enable': False,  # whether enable optimization or not
    'flows-generator': False,  # whether generate new flows or not
    'max_iterations': 50,  # maximum iteration times
    'max_no_improve': 10,  # maximum local search width
    'k': 0.3,  # ratio of removed flows
    'results-root-path': '/src/json/'  # root path of results
}

XML_CONFIG = {
    'tsn_host_pre_name': 'Host',  # prefix of tsn termination host
    'tsn_switch_pre_name': 'Switch',  # prefix of tsn
    'one-flow-one-host': False,  # whether one flow corresponds one host or not
    'multicast-model': True,  # whether all flow follow multicast transmission mode or not
    'static': True,  # whether static forwarding or not
    'enhancement-tsn-switch-enable': True,  # whether enable enhancement function of tsn switch or not
}

TESTING = {
    'round': 5,  # test rounds
    'x-axis-gap': 5,
    'prefix': 1,
    'flow-size': [10, 100],  # the least number of flows
    'draw-gantt-chart': False,
    'save-solution': False,
    'graph-core-size': [10, 20],
    'graph-edge-size': [],
    'generate-flows': False,
    'simulation-time': 1.2  #s
}
