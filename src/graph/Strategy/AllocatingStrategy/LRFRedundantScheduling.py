import logging
import abc
from typing import List, Dict, Set

from src.graph.Edge import Edge
from src.graph.Flow import Flow
from src.graph.TimeSlotAllocator import TimeSlotAllocator
from src.graph.Strategy.AllocatingStrategy import AEAPAllocatingStrategy
from src.type import FlowId
from math import ceil

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


# long-routes-first redundant scheduling strategy
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

    def schedule(self, flow_id: int, route: list, time: int) -> Set[FlowId]:
        # logger.info('schedule flow [' + str(flow_id) + ']...')
        flow = self.flow_mapper[flow_id]
        self.current_time = time
        if not self.schedule_end2end(flow, route):
            self.failure_queue.add(flow_id)
            logger.info('add flow [' + str(flow_id) + '] into failure queue')
        #self.allocation_record = allocation_record
        # edge_intervals = self.schedule_end2end(flow, route)
            #logger.info('scheduling flow [' + str(flow.flow_id) + '] successful')
            #print('allocation_num',self.allocation_num)
        #else:
        #    logger.info('scheduling flow [' + str(flow.flow_id) + '] failure')
        # return edge_intervals
        return self.failure_queue

    #def schedule1(self, flow_id_list: List[FlowId], *args, **kwargs) -> int:
    #    time_slots_used_num: int = 0
    #    for _fid in flow_id_list:
    #        if not self.schedule_single_flow(self.flow_mapper[_fid]):
    #            logger.info('scheduling mutiflow [' + str(_fid) + '] failure')
    #            return 0
    #    edge_list: List[Edge] = list(self.edge_mapper.values())
    #    time_slots_used: List[int] = [edge.time_slot_allocator.time_slot_used for edge in edge_list]
    #    for i in range(len(time_slots_used)):
    #        time_slots_used_num += time_slots_used[i]
    #    print(time_slots_used_num)
    #    return time_slots_used_num

    #def schedule_single_flow(self, flow: Flow) -> bool:
    #    logger.info('schedule flow [' + str(flow.flow_id) + ']...')
    #    _all_routes: List[List[List[int]]] = flow.get_routes()
    #    _union_routes: List[List[int]] = []
    #    for _e2e_routes in _all_routes:
    #        for _e2e_route in _e2e_routes:
    #            _union_routes.append(_e2e_route)
    #    _union_routes = LRFRedundantSchedulingStrategy.sort_route(_union_routes)
    #    _ER: List[int] = []  # recover list
    #    for _e2e_route in _union_routes:
    #        if not self.schedule_end2end(flow, _e2e_route):
    #            logger.info('scheduling flow [' + str(flow.flow_id) + '] failure')
    #            return False
    #        else:
    #            _ER.append(_e2e_route)
    #    logger.info('scheduling flow [' + str(flow.flow_id) + '] successful')
    #    return True

    def schedule_end2end(self, flow: Flow, route: List[int]) -> bool:
        _E: List[Edge] = []
        edge_interval: list = []
        edge_intervals: list = []
        _arrival_time_offset: int = 0
        if self.current_time == 0:
            _arrival_time_offset = 0
            self.edge_last_offset[route[0]] = 0
        else:
            for _block in self.allocation_record:
                if route[0] != _block:
                    _arrival_time_offset = self.current_time * 512
                    self.edge_last_offset[route[0]] = self.current_time
                else:
                    for index in range(len(self.allocation_record[_block])):
                        interval = self.allocation_record[_block][index].interval
                        if self.current_time < interval.upper:
                            _arrival_time_offset = self.edge_last_offset[route[0]] * 512
                        else:
                            _arrival_time_offset = self.current_time * 512
        for _eid in route:
            _e: Edge = self.edge_mapper[_eid]  # get edge
            _E.append(_e)
            _allocator: TimeSlotAllocator = _e.time_slot_allocator  # initialize time slot allocator
            #self._allocator.reset()
            _allocator.save_scene()  # save scene
            # _arrival_time_offset: int = self.allocate_aeap_overlap(flow, _allocator, _arrival_time_offset)
            _arrival_time_offset: int = self.allocate(flow, _allocator, _arrival_time_offset)
            #print('_arrival_time_offset', _arrival_time_offset)
            self.allocation_num: int = ceil(flow.size / _allocator.bandwidth / _allocator.time_slot_len)
            self.allocation_record[_eid] = _allocator.allocation_blocks_m
            #print('self.allocation_record', self.allocation_record[_eid])
            # _arrival_time_offset: int = _allocator.allocate_aeap_overlap(flow, _arrival_time_offset)
            # _arrival_time_offset: int = _allocator.allocate_aeap(flow, _arrival_time_offset)
            # TODO fix bug here
            i = 0
            for block_m in self.allocation_record[_eid]:
                edge_interval.append(block_m.interval)
                a = edge_interval
                #print('block_m.interval',block_m.interval)
            edge_intervals.append(a)
            #print('edge_intervals',edge_intervals)
            edge_interval = []

            if _arrival_time_offset == -1 or _arrival_time_offset > flow.deadline:
               # recover scene
               for _e in _E:
                   _e.time_slot_allocator.recover_scene()
               return False
        #return True
        self.arrival_time_offset = _arrival_time_offset
        return True

    def reset(self):
        for eid in range(len(self.edge_mapper)):
            self.edge_mapper[eid+1].time_slot_allocator.reset()
        self.failure_queue = set()

    # def reset(self):
    #     for eid in range(len(self.edge_mapper)):
    #         self.edge_mapper[eid+1].time_slot_allocator.reset()



