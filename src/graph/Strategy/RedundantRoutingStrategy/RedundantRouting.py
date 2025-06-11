import logging
from typing import List, Dict, Set
import networkx as nx

from src.graph.Edge import Edge
from src.graph.Flow import Flow
from src.graph.node import Node
from src.type import *

logger = logging.getLogger(__name__)

class RedundantRouting:
    nodes: List[int]
    edges: List[int]
    flows: List[Flow]
    node_mapper: Dict[int, Node]
    edge_mapper: Dict[int, Edge]
    flow_mapper: Dict[int, Flow]
    failure_queue: Set[FlowId]
    candidate_path: Dict[int,list]
    k: int

    def __init__(self, nodes: List[int], edges: List[int], flows: List[Flow], k: int, nx_graph: nx.Graph = None):
        self.graph = nx_graph
        self.nodes = nodes
        self.edges = edges
        self.flows = []
        self.num_redundant = k
        self.node_mapper = {}
        self.edge_mapper = {}
        self.flow_mapper = {}
        self.init_nodes()
        self.init_edges()
        self.add_flows(flows)
        # self.print_nodes()


    def route(self, flow_id_list:List[int]) -> Dict:
        candidate_path: Dict = {}
        #flow_id_lists = flow_id_list
        for fid in flow_id_list:
            candidate_path[fid] = self.get_candidate_path(self.flow_mapper[fid])
        return candidate_path


    def get_candidate_path(self, flow: Flow) -> list:
        f_id = flow.flow_id
        f_source = flow.source
        f_destination = flow.destinations[0]
        routes: List[List[int]] = []  #记录最终的所有候选集路径
        num_candidate_path: int = 0
        neighbor: Dict = {}
        successor: Dict = {}
        predecessor: Dict = {}
        node_num = 0
        available_neighbors:list = []
        #获得第一条冗余路径
        dijkstra_path_n : List[NodeId] = nx.dijkstra_path(self.graph, source=f_source,target=f_destination)
        #print("First path:",dijkstra_path_n)
        #dijkstra_path_e: List[EdgeId] = self.nodes_to_edges(dijkstra_path_n)
        routes.append(dijkstra_path_n)
        #node_num = len(dijkstra_path_n)

        # 获得第一条路径的所有结点的前继结点
        for node in dijkstra_path_n:
            if node == f_source:
                predecessor[node] = -1
            else:
                predecessor[node] = dijkstra_path_n[node_num]
                node_num = node_num + 1
        #print('predecessor:',predecessor)
        # 获得第一条路径的所有结点的后继结点
        node_nums = 0
        for node in dijkstra_path_n:
            if node == f_destination:
                successor[node] = -1
            else:
                node_nums += 1
                successor[node] = dijkstra_path_n[node_nums]
        #print('successsor:',successor)
        #获得所有结点的邻居结点
        for i in self.nodes:
            neighbor[i] = self.get_neigbors(i)
        #print('neighbor:',neighbor)

        #获得其他冗余路径
        tried_path: list = []
        complete_candidate_path_n: list = []
        for node_list in dijkstra_path_n:
            tried_path.append(node_list)
            #print('tried_path',tried_path)
            #print('node_list:',node_list)
            successor_candidate_path_n:List[list] = self.try_candidate_path(node_list, successor, predecessor, neighbor,f_destination)
            #print('successor_candidate_path_n:',successor_candidate_path_n)
            for candidate in successor_candidate_path_n:
                for i in tried_path[::-1]:
                    candidate.insert(0, i)
                complete_candidate_path_n.append(candidate)
            #print('complete_candidate_path_n:',complete_candidate_path_n)
            for candidate_path in complete_candidate_path_n:
                if candidate_path == dijkstra_path_n:
                    logger.info('This path is the same as the first candidate path')
                elif candidate_path not in routes:
                    routes.append(candidate_path)
                    num_candidate_path += 1
                    if num_candidate_path + 1 >= self.num_redundant:
                        break
            #print('routes',routes)
            if num_candidate_path+1 >= self.num_redundant:
                break

        #将冗余路径结点集转换为边集
        routes_e: List[List[EdgeId]] = []
        routes_es = []
        for route in routes:
            #print('route:',route)
            route_e: List[EdgeId] = self.nodes_to_edges(route)
            routes_e.append(route_e)
        routes_es.append(routes_e)
        #self.flow_mapper[f_id].routes = routes_e
        self.flow_mapper[f_id].routes = routes_es
        print('The Routing Paths of Flow %d:'%f_id,routes)

        return self.flow_mapper[f_id].routes

    def try_candidate_path(self,node:int,suceessor:Dict,prdecessor:Dict,neighbor:Dict,destination:int) ->List[List]:
        suceessor_node:int = suceessor[node]
        prdecessor_node:int = prdecessor[node]
        neighbors:list = neighbor[node]
        target_node:int = destination
        pathset:List[List] = []
        pathsets:List[List] = []
        #在求其余路径前删除node与原路径的后继结点相连的边再应用Dij算法
        temp_gragh = self.graph
            #避免dest和node结点间唯一的边被删减无法应用Dij算法
        if suceessor_node==destination:
            neighbor1 = list(nx.neighbors(self.graph,node))
            neighbor2 = list(nx.neighbors(self.graph,suceessor_node))
            common_neighbor = set(neighbor1) & set(neighbor2)
            if common_neighbor:
                temp_gragh.remove_edge(node,suceessor_node)
        else:
            temp_gragh.remove_edge(node,suceessor_node)
        #TopoGenerator.draw(temp_gragh)
        for neighbor in neighbors:
            #不能选择node原有路径的前继结点
            if not neighbor ==prdecessor_node:
                path:list = nx.dijkstra_path(temp_gragh, source=neighbor,target=target_node)
                pathset.append(path)
        #如果求得的路径中node的邻居结点在node前则该路径不符合
        for temp_path in pathset:
            if node not in temp_path:
                pathsets.append(temp_path)
        temp_gragh.add_edge(node,suceessor_node)
        return  pathsets


    def nodes_to_edges(self, node_id_list: List[NodeId]) -> List[EdgeId]:
        edge_id_list: List[EdgeId] = []
        in_node_id: NodeId = None
        out_node_id: NodeId = None
        for i, node_id in enumerate(node_id_list):
            if i == 0:
                in_node_id = node_id
                continue
            out_node_id = node_id
            edge_id: EdgeId = list(filter(
                lambda eid: self.edge_mapper[eid].in_node.node_id == in_node_id and self.edge_mapper[
                    eid].out_node.node_id == out_node_id, self.edge_mapper))[0]
            edge_id_list.append(edge_id)
            in_node_id = out_node_id
        return edge_id_list
    def init_nodes(self) -> bool:
        if self.nodes is None or self.nodes == []:
            logging.info('there is no nodes')
            return False
        for node_id in self.nodes:
            # logger.info('initialize node [' + str(node_id) + ']')
            node: Node = Node(node_id)
            self.node_mapper[node_id] = node
        return True


    def init_edges(self) -> bool:
        if self.edges is None or self.edges == []:
            logging.info('there is no edges')
            return False
        edge_id = 1  # start from 1
        for edge_tuple in self.edges:
            # logger.info(
            #    'initialize edge [' + str(edge_id) + '] <' + str(edge_tuple[0]) + '->' + str(edge_tuple[1]) + '>')
            in_node: int = edge_tuple[0]
            out_node: int = edge_tuple[1]
            _e: Edge = Edge(
                edge_id, in_node=self.node_mapper[in_node], out_node=self.node_mapper[out_node], hp = 0)
            self.edge_mapper[edge_id] = _e
            self.node_mapper[in_node].append_out_edge(_e)
            self.node_mapper[out_node].append_in_edge(_e)
            edge_id += 1
        return True


    def add_flows(self, flows: List[Flow]):
        # add flows to flow list and flow mapper
        if flows is None:
            return
        for _f in flows:
            self.flows.append(_f.flow_id)
            self.flow_mapper[_f.flow_id] = _f


    def get_neigbors(self,node):
        output = []
        layers = dict(nx.bfs_successors(self.graph, source=node))
        nodes = [node]
        # output[i] = []
        for x in nodes:
            output.extend(layers.get(x, []))
        nodes = output
        return output


    def get_max_routenum(self,flow_id_list:list):
        route_num:int = 0
        route_nums:list = []
        for fid in flow_id_list:
            for route in self.flow_mapper[fid].routes:
                route_num += len(route)
            route_nums.append(route_num)
            route_num = 0
        #print(route_nums)
        max_routenum:int = max(route_nums)
        return max_routenum

    def print_nodes(self):
        for _nid in self.nodes:
            self.node_mapper[_nid].to_string()












