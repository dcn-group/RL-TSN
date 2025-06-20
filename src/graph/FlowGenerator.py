import copy
import logging
import random

import networkx as nx
import numpy as np
import json
from typing import List, Dict
from math import floor

from src import config
from src.graph.Flow import Flow
from src.type import NodeId

logger = logging.getLogger(__name__)


class FlowGenerator:
    flows: List[Flow]

    def __init__(self):
        pass

    @classmethod
    def compute_hyper_period(cls, flows: List[Flow]):
        # TODO compute hyper period of flows
        # fake method here
        _F: List[Flow] = sorted(flows, key=lambda f: f.period)
        return _F[-1].period

    @staticmethod
    def generate_flow_properties(flow_num: int = 0) -> List[Dict]:
        flow_properties: List[Dict] = []
        for i in range(flow_num):
            properties: Dict = {}
            properties['size'] = \
                config.FLOW_CONFIG['size-set'][random.randint(0, len(config.FLOW_CONFIG['size-set'])) - 1]
            properties['period'] = \
                config.FLOW_CONFIG['period-set'][random.randint(0, len(config.FLOW_CONFIG['period-set'])) - 1]
            properties['reliability'] = \
                config.FLOW_CONFIG['reliability-set'][random.randint(0, len(config.FLOW_CONFIG['reliability-set'])) - 1]
            properties['deadline'] = \
                config.FLOW_CONFIG['deadline-set'][random.randint(0, len(config.FLOW_CONFIG['deadline-set'])) - 1]
            properties['dest-num'] = \
                config.FLOW_CONFIG['dest-num-set'][random.randint(0, len(config.FLOW_CONFIG['dest-num-set'])) - 1]
            flow_properties.append(properties)
        return flow_properties

    @classmethod
    def generate_flows(cls,edge_nodes: List[NodeId] = None, graph: nx.Graph = None, **kwargs) -> List[Flow]:
        '''
        generate flow randomly
        :param graph:
        :param edge_nodes: arrival source nodes
        :return:
        '''
        flow_num: int = 40
        flow_id: int = 1
        pair: Dict = {}
        pairlist = []
        if 'flow_num' in kwargs.keys():
            flow_num = kwargs['flow_num']
        if 'flow_id' in kwargs.keys():
            flow_id = kwargs['flow_id']
        if len(config.FLOW_CONFIG['dest-num-set']) + 1 > len(edge_nodes):
            raise RuntimeError('too less edge nodes')
        _F: List[Flow] = []
        _fid = flow_id
        for x in edge_nodes:
            pair[x] = [0]
        for _i in range(flow_num):
            if 'flow_properties' in kwargs.keys():
                _s: int = kwargs['flow_properties'][_i]['size']
                _p: int = kwargs['flow_properties'][_i]['period']
                _rl: int = kwargs['flow_properties'][_i]['reliability']
                _dl: int = kwargs['flow_properties'][_i]['deadline']
                _dn: int = kwargs['flow_properties'][_i]['dest-num']
            else:
                # _s: int = round(random.uniform(1,9),1) * 1000
                _s: int = random.randint(125, 625) * 8
                _p: int = \
                    config.FLOW_CONFIG['period-set'][random.randint(0, len(config.FLOW_CONFIG['period-set'])) - 1]
                _rl: int = \
                    config.FLOW_CONFIG['reliability-set'][
                        random.randint(0, len(config.FLOW_CONFIG['reliability-set'])) - 1]
                # _dl: int = \
                #     config.FLOW_CONFIG['deadline-set'][random.randint(0, len(config.FLOW_CONFIG['deadline-set'])) - 1]
                _dl: int = random.randint(1, 10) * 1e7
                _dn: int = \
                    config.FLOW_CONFIG['dest-num-set'][random.randint(0, len(config.FLOW_CONFIG['dest-num-set'])) - 1]
            _o: int = \
                edge_nodes[random.randint(0, len(edge_nodes)) - 1]
            #选择目的结点
            #_D = cls.select_dest(_o)
            _D: List[int] = []
            _edge_nodes_t: List[int] = copy.deepcopy(edge_nodes)
            # 1.目的节点不能是已经选择好的源点
            _edge_nodes_t.remove(_o)
            # 2.目的节点不能是之前有过的，即不能有相同源点和终点的流产生
            #if len(pair[_o]) != 0:
            #    for x in pair[_o]:
            #        if x in _edge_nodes_t:
            #            _edge_nodes_t.remove(x)
            # 2.目的节点不能是长度只有2或3的点
            while True:
                _d = _edge_nodes_t[random.randint(0, len(_edge_nodes_t)) - 1]
                distance = nx.dijkstra_path_length(graph,_o,_d)
                if distance > 3:
                    pair[_o].append(_d)
                    break
            _d = [_d]
            #source_neighbor: int = list(graph.neighbors(_o))[0]
            #neighbors: List[int] = list(graph.neighbors(source_neighbor))
            #neighbors.remove(_o)
            #neighbors = list(filter(lambda n: list(graph.neighbors(n)).__len__() == 1, neighbors))
            #_edge_nodes_t = list(set(_edge_nodes_t) - set(neighbors))
            #if neighbors.__len__() >= 1:
            #    _t: List[int] = random.sample(neighbors, int(
            #        np.ceil((1 - config.FLOW_CONFIG['un-neighbors_degree']) * len(neighbors))))
            #    [_edge_nodes_t.append(n) for n in _t]
            #_D = random.sample(_edge_nodes_t, _dn)
            _f: Flow = Flow(_fid, _s, _p, _o, _d, _rl, _dl)
            _F.append(_f)
            _fid += 1
            logger.info(_f)
        print(pair)
        return _F

    @staticmethod
    def select_dest(_o: int):
        dest = 0
        return dest



    @staticmethod
    def smooth_period(hp: int, p: int) -> int:
        # TODO smooth period of all flows
        _n: int = floor(hp / p)
        return int(hp / _n)

    @classmethod
    def generate_r(cls, n: int = 0, hn: List[int] = 0, s: List[int] = [], p: List[int] = [], dn: List[int] = [],
                   rl: List[float] = [], dl: List[int] = []) -> List[Flow]:
        '''
        generate flow randomly
        :param n: number of flows, e.g., 20
        :param hn: list of source nodes, e.g., [1, 6, 7]
        :param s: range of data size per cycle time, e.g., [int(1e4), int(2e4)]
        :param p: range of cycle time, e.g., [int(1e5), int(6e5)]
        :param dn: range of number of destination nodes, e.g., [1, 2]
        :param rl: range of reliability requirement, e.g., [0.97, 0.99]
        :param dl: range of ene-to-end delay requirement, e.g., [int(1e5), int(1.5e5)]
        :return: flows
        '''
        _F: List[Flow] = []
        _P: List[int] = [100000, 150000, 300000, 600000]  # TODO fake period here
        for _i in range(n):
            _fid = _i + 1
            _s: int = random.randint(s[0], s[1])
            # _p: int = random.randint(p[0], p[1])
            _p: int = _P[random.randint(0, len(_P)) - 1]  # TODO fake method here
            _dn: int = random.randint(dn[0], dn[1])
            _rl: int = random.randint(rl[0], rl[1])
            _dl: int = random.randint(dl[0], dl[1])
            _src: int = hn[random.randint(0, len(hn)) - 1]
            while True:
                _D: List[int] = random.sample(hn, _dn)
                _D = list(filter(lambda d: d != _src, set(_D)))
                if len(_D) != 0:
                    break
            _p = cls.smooth_period(p[1], _p)
            _f: Flow = Flow(_fid, _s, _p, _src, _D, _rl, _dl)
            _F.append(_f)
        return _F

    @classmethod
    def generate_s(cls):
        '''
        generate the set of flows specifically
        :return:
        '''
        pass

    @staticmethod
    def _obj2json_helper(obj):
        if type(obj).__name__ == 'set':
            return list(obj)
        else:
            return obj

    @staticmethod
    def _json2obj_helper(d):
        obj = object.__new__(Flow)  # Make instance without calling __init__
        for key, value in d.items():
            if key == 'walked_edges' or key == 'negative_walked_edges':
                value = set(value)
            setattr(obj, key, value)
        return obj

    @classmethod
    def flow2json(cls, flow: Flow):
        json.dumps(flow.__dict__, default=cls._obj2json_helper)

    @classmethod
    def flows2json(cls, flows: List[Flow]) -> str:
        _F: Dict[str] = dict()
        for _i, flow in enumerate(flows):
            #flow.routes_reliability = dict()
            #flow.routes = []
            #flow.walked_edges = set()
            #flow.negative_walked_edges = set()
            _F['f' + str(_i)] = json.dumps(flow.__dict__, default=cls._obj2json_helper)
        return json.dumps(_F)

    @classmethod
    def json2flows(cls, json_str) -> List[Flow]:
        _flows_str: Dict[str] = json.loads(json_str)
        _F: List[Flow] = []
        for _flow_str in _flows_str.values():
            _f: Flow = json.loads(_flow_str, object_hook=cls._json2obj_helper)
            _f.routes_reliability = dict()
            _f.routes = []
            _f.walked_edges = set()
            _f.negative_walked_edges = set()
            _F.append(_f)
        return _F

    @classmethod
    def routes2json(cls, routes: Dict) -> str:
        _F: Dict[str] = dict()
        routeset = dict()
        for _i, route in enumerate(routes):
            routeset['flow_id'] = _i + 1
            routeset['routes'] = routes[_i+1]
            _F['f' + str(_i)] = json.dumps(routeset, default=cls._obj2json_helper)
        return json.dumps(_F)

    @classmethod
    def json2routes(cls, json_str) -> Dict:
        _flows_str: Dict[str] = json.loads(json_str)
        _R = {}
        for _flow_str in _flows_str.values():
            _r = json.loads(_flow_str)
            _R[_r['flow_id']] = _r['routes']
        return _R

    @staticmethod
    def save_flows(flows: List[Flow]):
        with open(config.flows_filename, "w") as f:
            f.write(FlowGenerator.flows2json(flows))

    @staticmethod
    def save_routes(routes: Dict):
        with open(config.routes_filename, "w") as f:
            f.write(FlowGenerator.routes2json(routes))

    @staticmethod
    def load_flows() -> List[Flow]:
        flows: List[Flow] = []
        with open(config.flows_filename, "r") as f:
            flows = FlowGenerator.json2flows(f.read())
        return flows

    @staticmethod
    def load_routes():
        routes = {}
        with open(config.routes_filename, "r") as f:
            routes = FlowGenerator.json2routes(f.read())
        return routes
