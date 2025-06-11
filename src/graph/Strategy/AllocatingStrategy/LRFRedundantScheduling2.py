import copy
import logging
import abc
from typing import List, Set, Dict

from math import ceil

from src.graph.Edge import Edge
from src.graph.Flow import Flow
from src.graph.TimeSlotAllocator import TimeSlotAllocator
from src.type import FlowId

logger = logging.getLogger(__name__)

from src.graph.Strategy.AllocatingStrategy.AEAPAllocatingStrategy import AEAPAllocatingStrategy, AllocatingStrategy



class SchedulingStrategy(metaclass=abc.ABCMeta):
    edge_mapper: Dict[int, Edge]
    flow_mapper: Dict[int, Flow]
    failure_queue: Set[FlowId]
    allocation_num: int
    __allocating_strategy: AllocatingStrategy  # allocating strategy

    def __init__(self, edge_mapper: Dict[int, Edge], flow_mapper: Dict[int, Flow]):
        self.edge_mapper = edge_mapper
        self.flow_mapper = flow_mapper
        self.time_slot_used : int
        self.failure_queue = set()
        self.allocation_num : int = 0
        self.arrival_time_offset = 0
        self.current_time = 0
        self.allocation_record: Dict = {}
        self.edge_last_offset: Dict = {}  # 记录每条边上一个初始分配时刻 （_arrival_time_offset）
        self.__allocating_strategy = AEAPAllocatingStrategy()  # default allocating strategy
        #self._allocator: TimeSlotAllocator

    @abc.abstractmethod
    def schedule(self, flow_id: int, path_id: int, time: int) -> list:
        pass

    def allocate(self, flow: Flow, allocator: TimeSlotAllocator, arrival_time_offset: int) -> int:
        return self.__allocating_strategy.allocate(flow, allocator, arrival_time_offset)

    @property
    def allocating_strategy(self):
        return self.__allocating_strategy

    @allocating_strategy.setter
    def allocating_strategy(self, allocating_strategy: AllocatingStrategy):
        self.__allocating_strategy = allocating_strategy


class LRFRedundantSchedulingStrategy(SchedulingStrategy):

    @staticmethod
    def sort_route(routes: List[List[int]]) -> List[List[int]]:
        """
        sort flows list by priority from <sum of edge delay> to <period> to <hops> [NOTED]
        :param routes: flow routes to sort
        :return: sorted flow routes
        """
        # TODO sort routes of flow
        _routes = sorted(routes, key=lambda r: len(r), reverse=True)
        return _routes

    def schedule(self, flow_id_list: List[FlowId], *args, **kwargs) -> Set[FlowId]:
        for _fid in flow_id_list:
            if not self.schedule_single_flow(self.flow_mapper[_fid]):
                self.failure_queue.add(_fid)
                logger.info('add flow [' + str(_fid) + '] into failure queue')
        # logger.info('FAILURE QUEUE:' + str(self.failure_queue))
        return self.failure_queue

    def schedule_single_flow(self, flow: Flow) -> bool:
        # logger.info('schedule flow [' + str(flow.flow_id) + ']...')
        _all_routes: List[List[List[int]]] = flow.get_routes()
        _union_routes: List[List[int]] = []
        for _e2e_routes in _all_routes:
            for _e2e_route in _e2e_routes:
                _union_routes.append(_e2e_route)
        _union_routes = LRFRedundantSchedulingStrategy.sort_route(_union_routes)
        _ER: List[int] = []  # recover list
        for _e2e_route in _union_routes:
            if not self.schedule_end2end(flow, _e2e_route):
                logger.info('scheduling flow [' + str(flow.flow_id) + '] failure')
                # TODO recover time slots allocation on edge
                for __e2e_route in _ER:
                    for _eid in __e2e_route:
                        self.edge_mapper[_eid].time_slot_allocator.recover_scene()
                return False
            else:
                _ER.append(_e2e_route)
        # logger.info('scheduling flow [' + str(flow.flow_id) + '] successful')
        return True

    def schedule_end2end(self, flow: Flow, route: List[int]) -> bool:
        _arrival_time_offset: int = 0
        _E: List[Edge] = []
        for _eid in route:
            _e: Edge = self.edge_mapper[_eid]  # get edge
            _E.append(_e)
            _allocator: TimeSlotAllocator = _e.time_slot_allocator # get time slot allocator
            _allocator.save_scene()  # save scene
            # _arrival_time_offset: int = self.allocate_aeap_overlap(flow, _allocator, _arrival_time_offset)
            _arrival_time_offset: int = self.allocate(flow, _allocator, _arrival_time_offset)
            # _arrival_time_offset: int = _allocator.allocate_aeap_overlap(flow, _arrival_time_offset)
            # _arrival_time_offset: int = _allocator.allocate_aeap(flow, _arrival_time_offset)
            # TODO fix bug here
            if _arrival_time_offset == -1 or _arrival_time_offset > flow.deadline:
                # recover scene
                for _e in _E:
                    _e.time_slot_allocator.recover_scene()
                return False
        return True

    def reset(self):
        for eid in range(len(self.edge_mapper)):
            self.edge_mapper[eid+1].time_slot_allocator.reset()
