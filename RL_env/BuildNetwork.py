from src.graph.TopoGenerator import TopoGenerator
from src.graph.Strategy.topo_strategy.RandomRegularGraphStrategy import RandomRegularGraphStrategy
from src.graph.Strategy.RedundantRoutingStrategy.RedundantRouting import RedundantRouting
from src.graph.FlowGenerator import FlowGenerator
from src.net_envs.network.TSNNetwork import TSNNetwork
from src.net_envs.network.TSNNetworkFactory import TSNNetworkFactory
from src.utils.ConfigFileGenerator2 import ConfigFileGenerator
from typing import List, Tuple
from src.type import NodeId
from src.graph.Flow import Flow
from src.graph.Graph import Graph
from src import config
import networkx as nx
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildNetwork():
    def __init__(self):
        # 构造网络
        # self.graph 拓扑图
        # self.flows TSN流
        # self.candidate_path_set TSN流候选路由集
        # self.edge_list 拓扑图边集
        # 手动生成网络拓扑图
        ## Generate graph
        # edges: List[Tuple[int, int]] = [(1, 2), (1, 4), (1, 5), (1, 7), (2, 3), (2, 4), (2, 6), (3, 5), (3, 6), (3, 8), (4, 5), (4, 10), (5, 6), (6, 9)]
        # self.graph: nx.Graph = nx.Graph()  # nx.Graph为注释，解释变量类型，self.graph = nx.Graph(),创建空的简单网络图
        # self.graph.add_edges_from(edges)  # 从edges列表中加边
        # self.graph = self.graph.to_directed()
        # TopoGenerator.draw(self.graph)
        # 生成网络拓扑图
        topo_generator: TopoGenerator = TopoGenerator()
        topo_generator.topo_strategy = RandomRegularGraphStrategy(d=4, n=20)
        # topo_generator.topo_strategy = ErdosRenyiStrategy(n=8, m=10)
        self.graph: nx.Graph = topo_generator.generate_core_topo()
        # topo_generator.draw(self.graph)
        attached_edge_nodes_num: int = 20
        # attached_edge_nodes = [7, 8, 9, 10]
        attached_edge_nodes: List[NodeId] = topo_generator.attach_edge_nodes(self.graph, attached_edge_nodes_num)
        print(attached_edge_nodes)
        # 绘制并存储网络拓扑图
        # topo_generator.draw(self.graph)

        # 存储网络拓扑图的详细信息
        helper = '核心节点数：20\n'
        helper += '边缘节点数：' + str(attached_edge_nodes_num) + '\n'
        helper += '边缘节点集合：' + ' '.join(str(i) for i in attached_edge_nodes) + '\n'
        with open(config.helper_dir, "w") as f:
            f.write(helper)

        # 手动生成流
        # self.flows: List[Flow] = [
        #     Flow(1, int(4e3), int(3e5), 7, [9], 0.5, int(1e8)),
        #     Flow(2, int(6e3), int(3e5), 8, [10], 0.5, int(2e8)),
        #     Flow(3, int(5e3), int(3e5), 7, [8], 0.5, int(2e8)),
        #     Flow(4, int(2.5e3), int(3e5), 9, [10], 0.5, int(1e8)),
        #     Flow(5, int(2.6e3), int(3e5), 7, [9], 0.5, int(1e8)),
        #     Flow(6, int(3.2e3), int(3e5), 8, [10], 0.5, int(2e8)),
        #     Flow(7, int(4.2e3), int(3e5), 7, [8], 0.5, int(2e8)),
        #     Flow(8, int(6e3), int(3e5), 9, [10], 0.5, int(1e8)),
        #     Flow(9, int(4e3), int(3e5), 7, [8], 0.5, int(2e8)),
        #     Flow(10, int(2.6e3), int(3e5), 9, [10], 0.5, int(1e8)),
        # ]
        # 生成TSN数据流
        self.flows: List[Flow] = FlowGenerator.generate_flows(edge_nodes=attached_edge_nodes,
                                                         graph=self.graph)
        # 存储TSN数据流
        FlowGenerator.save_flows(self.flows)
        self.graph: Graph = Graph(self.graph,
                                  self.graph.nodes,
                                  self.graph.edges,
                                  config.GRAPH_CONFIG['hyper-period'])
        # 生成冗余路由候选集
        _F_r = []
        for _flow in self.flows:
            _F_r.append(_flow.flow_id)
        #print('All Flows:', _F_r)
        self.path_num = config.FLOW_CONFIG['path-num']
        flow_router = RedundantRouting(self.graph.nodes, self.graph.edges, self.flows, self.path_num, self.graph.nx_graph)
        self.candidate_path_set = flow_router.route(_F_r)
        print('Candidate Path Set:', self.candidate_path_set)
        # 存储冗余路由候选集
        # for _flow in self.flows:
        #     _flow.routes = [[_flow.routes[0][0], _flow.routes[0][1]]]
        #     print(_flow.routes)
        FlowGenerator.save_routes(self.candidate_path_set)

        # 设置边的信息
        # self.graph: Graph = Graph(self.graph,
        #                               self.graph.nodes,
        #                               self.graph.edges,
        #                               config.GRAPH_CONFIG['hyper-period'])
        # self.graph.edge_mapper = flow_router.edge_mapper
        self.edge_list: list = [i + 1 for i in range(self.graph.get_edge_num())]
        #print('self.edge_list:', self.edge_list)
        self.graph.add_flows(self.flows)
        # TODO set bandwidth to edges
        self.graph.set_all_edges_bandwidth(config.GRAPH_CONFIG['all-bandwidth'])  # set bandwidth
        # TODO set error rate to edges
        # self.graph.set_all_error_rate(config.GRAPH_CONFIG['all-per'])  # set error rate
        # TODO set propagation delay to edges
        # self.graph.set_all_edges_process_delay(config.GRAPH_CONFIG['all-propagation-delay'])
        # TODO set process delay to edges
        # self.graph.set_all_edges_process_delay(config.GRAPH_CONFIG['all-process-delay'])

        # 保存graph信息
        with open(config.graph_filename, "wb") as f:
            pickle.dump(self.graph, f)
        # # 构造TSN实际网络
        # tsn_network_factory: TSNNetworkFactory = TSNNetworkFactory()
        # tsn_network: TSNNetwork = tsn_network_factory.product(
        #     self.graph, self.flows,
        #     enhancement_enable=config.XML_CONFIG['enhancement-tsn-switch-enable'])
        # self.tsn_network = tsn_network
        # self.node_edge_mac_info = tsn_network_factory.node_edge_mac_info
        # # create test scenario
        # ConfigFileGenerator.create_test_scenario(tsn_network=self.tsn_network,
        #                                          graph=self.graph,
        #                                          flows=self.flows,
        #                                          node_edge_mac_info=self.node_edge_mac_info,
        #                                          solution_name='test_scenario1')
        print(self.flows[0].routes)


# network = BuildNetwork()
