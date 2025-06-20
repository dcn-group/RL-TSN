from typing import List, Dict, Set

import networkx as nx

from src.graph.TimeSlotAllocator import TimeSlotAllocator
from src.graph.node import Node
from src.graph.Edge import Edge
from src.graph.Flow import Flow
#from .FlowScheduler import FlowScheduler
#from src.utils.Visualizer import Visualizer, GanttEntry, GanttBlock
import logging
from .computing import lcm_m

logger = logging.getLogger(__name__)


# maybe a single instance
class Graph:
    nodes: List[int]
    edges: List[int]
    flows: List[int]
    nx_graph: nx.Graph
    edge_nodes: List[int]  # edge node id list
    core_nodes: List[int]  # core node id list
    node_mapper: Dict[int, Node]
    edge_mapper: Dict[int, Edge]
    flow_mapper: Dict[int, Flow]
    hyper_period: int
   #flow_scheduler: FlowScheduler
    failure_queue: Set[int]

    def __init__(self, nx_graph: nx.Graph = None, nodes: List[int] = None, edges: List[int] = None, hp: int = 0):
        self.nx_graph = nx_graph
        self.nodes = nodes
        self.edges = edges
        self.flows = []
        self.failure_queue = set()
        self.node_mapper = {}
        self.edge_mapper = {}
        self.flow_mapper = {}
        self.hyper_period = hp
        self.init_nodes()
        self.init_edges()
        self.print_nodes()
        # self.print_edges()

    def get_node_num(self):
        return self.nodes.__len__()

    def get_edge_num(self):
        return self.edges.__len__()

    def print_nodes(self):
        for _nid in self.nodes:
            self.node_mapper[_nid].to_string()

    def print_edges(self):
        for _eid in self.edges:
            self.edge_mapper[_eid].to_string()

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
            logger.info(
               'initialize edge [' + str(edge_id) + '] <' + str(edge_tuple[0]) + '->' + str(edge_tuple[1]) + '>')
            in_node: int = edge_tuple[0]
            out_node: int = edge_tuple[1]
            _e: Edge = Edge(
                edge_id, in_node=self.node_mapper[in_node], out_node=self.node_mapper[out_node], hp=self.hyper_period)
            self.edge_mapper[edge_id] = _e
            self.node_mapper[in_node].append_out_edge(_e)
            self.node_mapper[out_node].append_in_edge(_e)
            edge_id += 1
        return True

    def set_edges_bandwidth(self, b: int):
        # TODO set edge bandwidth
        pass

    def set_all_edges_bandwidth(self, b: int):
        '''
        set bandwidth of all edges
        :param b:
        :return:
        '''
        for edge in self.edge_mapper.values():
            edge.set_bandwidth(b)

    def set_end2switch_edges_bandwidth(self, b: int):
        # TODO set all host-to-switch edges delay
        pass

    def set_switch2switch_edges_bandwidth(self, b: int):
        # TODO set all switch-to-switch edges daley
        pass

    def set_all_edges_propagation_delay(self, prop_d: int):
        '''
        set propagation delay of all edges
        :param prop_d:
        :return:
        '''
        for edge in self.edge_mapper.values():
            edge.set_propagation_delay(prop_d)

    def set_all_edges_process_delay(self, proc_d: int):
        for edge in self.edge_mapper.values():
            edge.set_process_delay(proc_d)

    def set_all_error_rate(self, error_rate: float):
        '''
        set error rate of all edges
        :param error_rate:
        :return:
        '''
        for edge in self.edge_mapper.values():
            edge.error_rate = error_rate

    def add_flows(self, flows: List[Flow]):
        # add flows to flow list and flow mapper
        if flows is None:
            return
        for _f in flows:
            self.flows.append(_f.flow_id)
            self.flow_mapper[_f.flow_id] = _f

    def compute_hyper_period(self):
        p = [flow.period for flow in self.flow_mapper.values()]
        self.hyper_period = int(lcm_m(p))        #对p求最小公倍数
        for edge in self.edge_mapper.values():
            edge.hyper_period = self.hyper_period






    # draw gantt chart for merged time slots allocation blocks
    # def draw_gantt(self):
    #     gantt_entries: List[GanttEntry] = []
    #     for _i, _e in enumerate(self.edges):
    #         _e: Edge = self.edge_mapper[_i + 1]
    #         _allocator: TimeSlotAllocator = _e.time_slot_allocator
    #         _time_slot_len: int = _allocator.time_slot_len
    #         _gantt_blocks: List[GanttBlock] = []
    #         for _j, _block in enumerate(_allocator.allocation_blocks_m):
    #             _caption = 'fid=' + str(_block.flow_id)
    #             _gantt_block: GanttBlock = GanttBlock(_block.interval.lower * _time_slot_len,
    #                                                   (_block.interval.upper + 1) * _time_slot_len, _caption)
    #             _gantt_blocks.append(_gantt_block)
    #         _gantt_entry: GanttEntry = GanttEntry(10 * _i, 'edge ' + str(_e.edge_id), 5, _gantt_blocks)
    #         gantt_entries.append(_gantt_entry)
    #     Visualizer.draw_gantt([0, self.hyper_period], [0, 10 * len(self.edges)], gantt_entries)

    # draw gantt chart for raw time slots allocation blocks

    """
    def draw_gantt(self, title: str = None, filename: str = None):
        '''
        gant chart without merging operation    #画没有合并的时间甘特图
        :return:
        '''
        # random color for flow
        _colors: Dict[int, str] = dict()
        for _fid in self.flows:
            # _colors[_fid] = Visualizer.random_color()
            _colors[_fid] = self.flow_mapper[_fid].color
        gantt_entries: List[GanttEntry] = []
        for _i, _e in enumerate(self.edges):
            _e: Edge = self.edge_mapper[_i + 1]
            _allocator: TimeSlotAllocator = _e.time_slot_allocator
            _time_slot_len: int = _allocator.time_slot_len
            _gantt_blocks: List[GanttBlock] = []
            for _j, _block in enumerate(_allocator.allocation_blocks):
                # _caption: str = 'f=' + str(_block.flow_id) + '\n' + 'p=' + str(_block.phase)
                _caption: str = ''
                _gantt_block: GanttBlock = GanttBlock(
                    _block.interval.lower * _time_slot_len,
                    (_block.interval.upper + 1 - _block.interval.lower) * _time_slot_len,
                    _caption, color=_colors[_block.flow_id])
                _gantt_blocks.append(_gantt_block)
            _gantt_entry: GanttEntry = GanttEntry(10 * _i, 'edge ' + str(_e.edge_id), 5, _gantt_blocks)
            gantt_entries.append(_gantt_entry)
        Visualizer.draw_gantt([0, self.hyper_period], [0, 10 * len(self.edges)], gantt_entries,
                              title=title, filename=filename)
    """
