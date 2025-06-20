import logging
import json
import copy
from typing import List, Dict

from math import ceil
from math import floor
import portion as P
import intervals as I
from intervals import Interval

from src import config
from src.graph.Flow import Flow

logger = logging.getLogger(__name__)
MinFrameSize = 64 * 8  # minimal frame size = 64B, unit: Byte


class AllocationBlock:
    flow_id: int
    phase: int
    arrival_time_offset: int
    send_time_offset: int
    interval: Interval #改

    def __init__(self, flow_id, interval: Interval, at_offset: int, st_offset: int, phase: int = 0):   #改
        self.flow_id = flow_id
        self.interval = interval
        self.arrival_time_offset = at_offset
        self.send_time_offset = st_offset
        self.phase = phase


class TimeSlotAllocator:
    edge_id: int
    __hyper_period: int  # hyper period of all flows, [unit: us]
    bandwidth: float  # bandwidth on edge, [unit: bps]
    propagation_delay: int
    process_delay: int
    min_flow_size: int  # minimal flow size, [unit: b]
    flow_times_mapper: Dict[int, List[AllocationBlock]]      #用于记录不同id的流的allocation_block内容
    flow_times_mapper_c: Dict[int, List[AllocationBlock]]
    allocation_blocks: List[AllocationBlock]  # time windows without merging operation 记录一条流的allocation_block,用于存放已经分配的时间片，是排好顺序的
    allocation_blocks_c: List[AllocationBlock]
    allocation_blocks_m: List[AllocationBlock]  # time windows with merging operation
    allocation_blocks_m_c: List[AllocationBlock]
    free_intervals: List[Interval]  # free intervals  改
    time_slot_len: int  # time slot length, [unit: us]
    time_slot_num: int  # number of time slots
    load: float  # load of edge
    load_c: float
    time_slot_used: int  # time slot that be used by flow
    time_slot_used_c: int
    flow_num: int  # number of flow traversed on edge
    flow_num_c: int
    flow_segment_num: int  # number of continuous flow traversed on edge
    flow_segment_num_c: int

    def __init__(self, edge_id: int, hp: int = 0, b: float = 0, s: int = config.GRAPH_CONFIG['min-flow-size'],
                 prop_d: int = 0, proc_d: int = 0):
        self.edge_id = edge_id
        self.__hyper_period = hp
        self.bandwidth = b
        self.min_flow_size = s
        self.propagation_delay = prop_d
        self.process_delay = proc_d
        self.reset()

    @property
    def hyper_period(self):
        return self.__hyper_period

    @hyper_period.setter
    def hyper_period(self, hyper_period: int):
        if hyper_period != self.__hyper_period:
            self.__hyper_period = hyper_period
            self.reset()
            #logger.info('time slots allocation on edge [' + str(self.edge_id) + '] has been reset')
        else:
            logger.info('time slots of edge [' + str(self.edge_id) + '] has no change')

    def to_string(self):          #用于pytest时输出信息
        _B: List[List[int]] = []
        for _block in self.allocation_blocks:
            _interval: Interval = _block.interval   #改
            _B.append([_interval.lower, _interval.upper])
        _B_m: List[List[int]] = []
        for _block_m in self.allocation_blocks_m:
            _interval: Interval = _block_m.interval  #改
            _B_m.append([_interval.lower, _interval.upper])
        _B_f: List[List[int]] = []
        for _block_f in self.free_intervals:
            _B_f.append([_block_f.lower, _block_f.upper])
        o = {
            'edge id': self.edge_id,
            'hyper_period': str(self.__hyper_period) + ' ns',
            'bandwidth': str(self.bandwidth) + ' b/ns',
            'min_flow_size': str(self.min_flow_size) + ' b',
            'time_slot_len': str(self.time_slot_len) + ' ns',
            'time_slot_num': self.time_slot_num,
            'load': str(self.load * 100) + '%',
            'time_slots_used': self.time_slot_used,
            'flow_num': self.flow_num,
            'flow_segment_num': self.flow_segment_num,
            'raw allocation blocks num': len(self.allocation_blocks),
            'raw_allocation_blocks': _B,
            'merged allocation blocks num': len(self.allocation_blocks_m),
            'merged_allocation_blocks': _B_m,
            'free allocation blocks num': len(self.free_intervals),
            'free_allocation_blocks': _B_f
        }
        _json = json.dumps(o)
        logger.info(_json)

    def sort_allocation_blocks(self, blocks: List[AllocationBlock]):
        return sorted(blocks, key=lambda b: b.interval.lower)

    def save_scene(self):
        self.allocation_blocks_c = self.allocation_blocks.copy()
        self.allocation_blocks_m_c = self.allocation_blocks_m.copy()
        self.flow_times_mapper_c = self.flow_times_mapper.copy()
        self.flow_num_c = self.flow_num
        self.flow_segment_num_c = self.flow_segment_num
        self.load_c = self.load
        self.time_slot_used_c = self.time_slot_used

    def recover_scene(self):
        self.allocation_blocks = self.allocation_blocks_c.copy()
        self.allocation_blocks_m = self.allocation_blocks_m_c.copy()
        self.flow_times_mapper = self.flow_times_mapper_c.copy()
        self.flow_num = self.flow_num_c
        self.flow_segment_num = self.flow_segment_num_c
        self.load = self.load_c
        self.time_slot_used = self.time_slot_used_c

    def reset(self):
        self.flow_times_mapper = {}  # clear flow-time-slots mapper when hyper period changes
        self.allocation_blocks = []  # clear time slot allocation
        self.allocation_blocks_m = []  # clear merged time slot allocation
        self.allocation_blocks_c = []
        self.allocation_blocks_m_c = []
        self.load = 0
        self.load_c = 0
        self.time_slot_used = 0
        self.time_slot_used_c = 0
        self.flow_num = 0
        self.flow_num_c = 0
        self.flow_segment_num = 0
        if self.bandwidth != 0 and self.min_flow_size != 0 and self.__hyper_period:
            self.time_slot_len = ceil(self.min_flow_size / self.bandwidth)
            # self.time_slot_num = floor(self.__hyper_period / self.time_slot_len)
            self.time_slot_len = ceil(self.min_flow_size / config.GRAPH_CONFIG['max-bandwidth'])  # TODO fix bug here
            self.time_slot_num = floor(self.__hyper_period / self.time_slot_len)
            self.free_intervals = [P.closed(0, self.time_slot_num - 1)]    #改
        else:
            self.time_slot_len = 0
            self.time_slot_num = 0
            self.free_intervals = []
        # self.to_string()

    def set_bandwidth(self, b: float):
        '''
        change bandwidth will affect time slots at the same time
        :param b: bandwidth
        :return: None
        '''
        if b != self.bandwidth:
            self.bandwidth = b
            self.reset()
            #logger.info('time slots allocation on edge [' + str(self.edge_id) + '] has been reset')
        else:
            logger.info('time slots of edge [' + str(self.edge_id) + '] has no change')

    def set_hyper_period(self, hp: int):
        '''
        change hyper period will affect time slot map at the same time
        :param hp: hyper period
        :return:
        '''
        if hp != self.__hyper_period:
            self.__hyper_period = hp
            self.reset()
            logger.info('time slots allocation on edge [' + str(self.edge_id) + '] has been reset')
        else:
            logger.info('time slots of edge [' + str(self.edge_id) + '] has no change')

    def merge_allocation_blocks(self) -> List[AllocationBlock]:
        # self.allocation_blocks.sort(key=lambda b: b.interval.lower)
        merged_allocation_blocks: List[AllocationBlock] = []
        for block in self.allocation_blocks:
            if not merged_allocation_blocks or merged_allocation_blocks[-1].interval.upper < block.interval.lower:
                _block: AllocationBlock = copy.deepcopy(block)
                merged_allocation_blocks.append(_block)
            elif self._is_same_flow(merged_allocation_blocks[-1].flow_id,
                                    block.flow_id,
                                    merged_allocation_blocks[-1].send_time_offset,
                                    block.send_time_offset,
                                    block.interval.upper - block.interval.lower + 1):
                merged_allocation_blocks[-1].upper = max(merged_allocation_blocks[-1].interval.upper,
                                                                  block.interval.upper)              # 原为merged_allocation_blocks[-1].interval.upper
            else:
                _block: AllocationBlock = copy.deepcopy(block)
                merged_allocation_blocks.append(block)
        return merged_allocation_blocks

    def calculate_free_blocks(self) -> List[Interval]:      #改
        free_blocks: List[Interval] = []                 #改
        lower: int = 0
        for block in self.allocation_blocks_m:
            if block.interval.lower != 0 and lower < block.interval.lower:
                free_blocks.append(P.closed(lower, block.interval.lower - 1))             #改
            lower = block.interval.upper + 1
        if lower < self.time_slot_num:
            free_blocks.append(P.closed(lower, self.time_slot_num - 1))                 #改
        return free_blocks

    def allocate(self, flow: Flow, arrival_time_offset, send_time_offset: int, phase_num: int, allocation_num: int):
        for _phase in range(phase_num):
            _block_num: int = len(self.allocation_blocks)
            _block_m_num: int = len(self.allocation_blocks_m)
            _lower: int = floor(send_time_offset % self.hyper_period / self.time_slot_len)
            # _lower: int = floor(send_time_offset % (self.time_slot_num * self.time_slot_len) / self.time_slot_len)
            _upper: int = _lower + allocation_num - 1
            _blocks: List[AllocationBlock] = []
            # create time slots allocation blocks
            if _lower < self.time_slot_num:
                if _upper < self.time_slot_num:
                    _interval: Interval = P.closed(_lower, _upper)          #改2
                    _block = AllocationBlock(
                        flow.flow_id, _interval,
                        at_offset=arrival_time_offset, st_offset=send_time_offset, phase=_phase)
                    _blocks = [_block]
                else:
                    _lower_0 = _lower
                    _upper_0 = self.time_slot_num - 1
                    _interval_0 = P.closed(_lower_0, _upper_0)       #改
                    _block_0 = AllocationBlock(
                        flow.flow_id, _interval_0,
                        at_offset=arrival_time_offset, st_offset=send_time_offset, phase=_phase)
                    _lower_1 = 0
                    _upper_1 = _upper % self.time_slot_num
                    _interval_1 = P.closed(_lower_1, _upper_1)           #改
                    _block_1 = AllocationBlock(
                        flow.flow_id, _interval_1,
                        at_offset=arrival_time_offset, st_offset=send_time_offset, phase=_phase)
                    _blocks = [_block_0, _block_1]
            else:
                logger.error('fuck damn!')
            # insert directly without merge operation
            for __block in _blocks:
                if len(self.allocation_blocks) == 0:
                    self.allocation_blocks.append(__block)
                else:
                    for _i, block in enumerate(self.allocation_blocks):     # 循环allocation_blockes
                        if __block.interval.lower <= block.interval.lower:    # 如果小于已经分配的interval，则将其插入到之前
                            self.allocation_blocks.insert(_i, __block)
                            break
                        else:
                            if _i >= _block_num - 1:  # append to last       # 如果大于已经分配的interval，则将其插入到之后
                                self.allocation_blocks.insert(_i + 1, __block)
                                break
            if _phase == 0:                              # 一个超周期内只有一次流
                _next_arrival_time_offset = \
                    send_time_offset + flow.period + self.propagation_delay + self.process_delay
            # add blocks to flow-time-slots mapper
            if flow.flow_id in self.flow_times_mapper:
                for __block in _blocks:
                    self.flow_times_mapper[flow.flow_id].append(__block)
            else:
                __blocks: List[AllocationBlock] = _blocks.copy()
                self.flow_times_mapper[flow.flow_id] = __blocks
            # # insert with merging operation
            # for __block in _blocks:
            #     if len(self.allocation_blocks_m) == 0:
            #         self.allocation_blocks_m.append(__block)
            #     else:
            #         for _i, block_m in enumerate(self.allocation_blocks_m):
            #             if __block.interval.lower <= block_m.interval.lower:
            #                 self.allocation_blocks_m.insert(_i, __block)
            #                 if __block.interval.upper in block_m.interval and __block.flow_id == block_m.flow_id:
            #                     __block.interval.upper = block_m.interval.upper
            #                     del self.allocation_blocks_m[_i + 1]
            #                 if _i != 0:
            #                     _pre_block_m: AllocationBlock = self.allocation_blocks_m[_i - 1]
            #                     if __block.interval.lower in _pre_block_m.interval and __block.flow_id == block_m.flow_id:
            #                         __block.interval.lower = _pre_block_m.interval.lower
            #                         del self.allocation_blocks_m[_i - 1]
            #                 break
            #             else:
            #                 if _i >= _block_m_num - 1:
            #                     self.allocation_blocks_m.insert(_i + 1, __block)
            #                     _pre_block_m: AllocationBlock = self.allocation_blocks_m[_i]
            #                     if __block.interval.lower in _pre_block_m.interval and __block.flow_id == block_m.flow_id:
            #                         __block.interval.lower = _pre_block_m.interval.lower
            #                         del self.allocation_blocks_m[_i]
            #                     break
            # if flow not exit, then the number of flow add 1
            if flow.flow_id not in self.flow_times_mapper:
                self.flow_num += 1
            # add to next phase
            send_time_offset += flow.period
        # calculate merged allocation blocks
        self.allocation_blocks_m = self.merge_allocation_blocks()
        # calculate free allocation blocks
        self.free_intervals = self.calculate_free_blocks()
        # calculate time slot used
        _sum: int = 0
        for _block_m in self.allocation_blocks_m:
            _sum += _block_m.interval.upper - _block_m.interval.lower + 1
        self.time_slot_used = _sum
        # add guard band
        total_guard_band: int = 0
        block_num = len(self.allocation_blocks_m)
        for i in range(block_num):
            if i + 1 < block_num and self.allocation_blocks_m[i].interval.upper + 1 != \
                    self.allocation_blocks_m[i + 1].interval.lower:
                total_guard_band += 1
        self.time_slot_used += total_guard_band
        # calculate payload
        self.load = self.time_slot_used / self.time_slot_num

    @staticmethod
    def _is_same_flow(id1: int, id2: int, offset1: int, offset2: int, allocation_num: int) -> bool:
        if id1 == id2:
            if abs(offset1 - offset2) >= allocation_num:
                return False
            else:
                return True
        else:
            return False

    def try_allocate(self, time_offset: int, flow_id: int, allocation_num: int, phase_num: int, bp: int,
                     overlaped=False) -> bool:
        '''
        brute force method to check whether flow can be allocated or not
        :param time_offset:
        :param flow_id:
        :param allocation_num:
        :param phase_num:
        :param bp:flow period
        :return:
        '''
        if self.time_slot_num == 0:
            logger.error('time slots on edge [' + str(self.edge_id) + '] does not initialize')
            return False
        if bp < allocation_num:
            logger.error('required time slots exceed base period')
            return False
        for phase in range(phase_num):
            # _lower: int = floor(time_offset % self.hyper_period / self.time_slot_len)
            _lower: int = floor(time_offset % (self.time_slot_num * self.time_slot_len) / self.time_slot_len)   # 计算所需区间的初始偏移量，不是具体的时间
            _upper: int = _lower + allocation_num - 1    # [_lower,_upper]为所需时间片区间
            _intervals: List[Interval] = []
            if _lower < self.time_slot_num:
                if _upper < self.time_slot_num:
                    _interval: Interval = P.closed(_lower, _upper)      # 全闭区间
                    _intervals: List[Interval] = [_interval]
                else:
                    _lower_0 = _lower
                    _upper_0 = self.time_slot_num - 1    # 如果_upper超过范围，则右区间设为上限值time_slot_num
                    _interval_0 = P.closed(_lower_0, _upper_0)   #改
                    _lower_1 = 0
                    _upper_1 = _upper % self.time_slot_num
                    _intervals = []
                    _interval_1 = P.closed(_lower_1, _upper_1)    #改
                    _intervals = [_interval_0, _interval_1]
            else:
                logger.error('lower bound exceed number of time slots')
                return False
            if overlaped is True:
                for __interval in _intervals:
                    for block in self.allocation_blocks:
                        fid = block.flow_id
                        if __interval.lower in block.interval and __interval.upper in block.interval:
                            if not self._is_same_flow(fid, flow_id, time_offset, block.send_time_offset,
                                                      allocation_num):
                                return False
                        elif __interval.lower in block.interval and __interval.upper > block.interval.upper:
                            if not self._is_same_flow(fid, flow_id, time_offset, block.send_time_offset,
                                                      allocation_num):
                                return False
                        elif __interval.upper in block.interval and __interval.lower < block.interval.lower:
                            if not self._is_same_flow(fid, flow_id, time_offset, block.send_time_offset,
                                                      allocation_num):
                                return False
                        elif block.interval.lower in __interval and block.interval.upper in __interval:
                            if not self._is_same_flow(fid, flow_id, time_offset, block.send_time_offset,
                                                      allocation_num):
                                return False
            else:
                for __interval in _intervals:
                    for block in self.allocation_blocks:
                        if __interval.lower in block.interval and __interval.upper in block.interval:
                            return False
                        elif __interval.lower in block.interval and __interval.upper > block.interval.upper:
                            return False
                        elif __interval.upper in block.interval and __interval.lower < block.interval.lower:
                            return False
                        elif block.interval.lower in __interval and block.interval.upper in __interval:
                            return False
            time_offset += bp      # 在一个超周期内流的周期短，可以产生多个phase的流时，调度一个，后面加周期即可
        return True

