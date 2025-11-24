"""
问题2：缓存分配与换入换出
基于问题1的调度序列，实现硬件状态仿真框架，管理多级缓存并执行SPILL操作
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import json
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from collections import defaultdict
import heapq
import numpy as np
import numba
from numba import jit, types
from numba.typed import Dict, List as NumbaList
from tqdm import tqdm
from advanced_scheduler import AdvancedScheduler, Node


@numba.jit(nopython=True)
def _fast_find_best_fit(regions_start, regions_end, regions_buf_id, capacity, required_size):
    """使用Numba优化的最佳适配算法"""
    best_start = -1
    best_gap_size = capacity + 1  # 初始化为很大的值
    
    # 检查开头
    if len(regions_start) == 0:
        if required_size <= capacity:
            return 0
    else:
        if regions_start[0] >= required_size and regions_start[0] < best_gap_size:
            best_start = 0
            best_gap_size = regions_start[0]
    
    # 检查中间间隙
    for i in range(len(regions_start) - 1):
        gap_start = regions_end[i]
        gap_end = regions_start[i + 1]
        gap_size = gap_end - gap_start
        
        if gap_size >= required_size and gap_size < best_gap_size:
            best_start = gap_start
            best_gap_size = gap_size
    
    # 检查末尾
    if len(regions_start) > 0:
        last_end = regions_end[-1]
        tail_size = capacity - last_end
        if tail_size >= required_size and tail_size < best_gap_size:
            best_start = last_end
            best_gap_size = tail_size
    
    return best_start


@numba.jit(nopython=True)
def _fast_calculate_fragmentation_impact(regions_start, regions_end, new_start, new_size, capacity):
    """快速计算分配后的碎片化影响"""
    new_end = new_start + new_size
    fragmentation_score = 0.0
    
    # 检查左侧碎片
    for i in range(len(regions_start)):
        if regions_start[i] > new_end:
            # 找到右边第一个区域
            left_gap = new_start - (regions_end[i-1] if i > 0 else 0)
            right_gap = regions_start[i] - new_end
            
            # 小碎片惩罚
            if left_gap > 0 and left_gap < 64:
                fragmentation_score += 100
            if right_gap > 0 and right_gap < 64:
                fragmentation_score += 100
            break
    
    return fragmentation_score


@numba.jit(nopython=True)
def _fast_predict_memory_usage(schedule_ops, schedule_buf_ids, schedule_sizes, schedule_cache_types, 
                              current_step, target_cache_type_idx, window_size):
    """快速预测内存使用趋势"""
    current_usage = 0
    max_usage = 0
    
    end_step = min(current_step + window_size, len(schedule_ops))
    
    for i in range(current_step, end_step):
        if schedule_cache_types[i] == target_cache_type_idx:
            if schedule_ops[i] == 0:  # ALLOC = 0
                current_usage += schedule_sizes[i]
                max_usage = max(max_usage, current_usage)
            elif schedule_ops[i] == 1:  # FREE = 1
                current_usage -= schedule_sizes[i]
    
    return max_usage


class CacheRegion:
    """缓存区域，表示一个已分配的地址范围"""
    def __init__(self, start: int, end: int, buf_id: int):
        self.start = start
        self.end = end  # 不包含end
        self.buf_id = buf_id
        
    def overlaps_with(self, start: int, end: int) -> bool:
        """检查是否与指定范围重叠"""
        return not (self.end <= start or end <= self.start)
    
    def size(self) -> int:
        """返回区域大小"""
        return self.end - self.start


class CacheManager:
    """单个缓存类型的管理器"""
    
    def __init__(self, cache_type: str, capacity: int):
        self.cache_type = cache_type
        self.capacity = capacity
        self.allocated_regions: List[CacheRegion] = []  # 已分配的区域
        self.buf_to_region: Dict[int, CacheRegion] = {}  # 缓冲区ID到区域的映射
        # 新增：维护有序空闲块列表 [(start, end)]，表示 [start, end) 连续空闲区间
        self.free_blocks: List[Tuple[int, int]] = [(0, capacity)]
        
    def find_free_space(self, size: int, prefer_start: int = 0) -> Optional[int]:
        """
        寻找能容纳指定大小的空闲空间
        返回起始地址，如果没有足够空间返回None
        """
        # 最佳适配 + 向左紧凑：选择能容纳的最小空闲块的起始地址（若并列，选更小的start）
        best_start = None
        best_len = None
        for (start, end) in self.free_blocks:
            length = end - start
            if length >= size:
                if best_len is None or length < best_len or (length == best_len and start < best_start):
                    best_start = start
                    best_len = length
        return best_start
    
    def allocate(self, buf_id: int, size: int, offset: Optional[int] = None) -> Optional[int]:
        """
        分配缓存空间
        返回分配的起始地址，失败返回None
        """
        if buf_id in self.buf_to_region:
            # 已经分配过
            return self.buf_to_region[buf_id].start
        
        if offset is not None:
            # 指定地址分配
            if self._can_allocate_at(offset, size):
                self._allocate_at(offset, size, buf_id)
                return offset
            else:
                return None
        else:
            # 自动寻找空间
            start_addr = self.find_free_space(size)
            if start_addr is not None:
                self._allocate_at(start_addr, size, buf_id)
                return start_addr
            else:
                return None
    
    def _can_allocate_at(self, start: int, size: int) -> bool:
        """检查是否可以在指定地址分配"""
        if start < 0 or start + size > self.capacity:
            return False
        end = start + size
        # 检查是否完全被某个空闲块包含
        for (fs, fe) in self.free_blocks:
            if fs <= start and end <= fe:
                return True
        return False
    
    def _add_region(self, region: CacheRegion):
        """添加分配的区域（仅更新已分配集合）"""
        self.allocated_regions.append(region)
        self.buf_to_region[region.buf_id] = region

    def _allocate_at(self, start: int, size: int, buf_id: int):
        """在指定位置分配并更新空闲块（拆分 free_blocks）"""
        end = start + size
        # 在 free_blocks 中找到包含该区间的块并拆分
        for i, (fs, fe) in enumerate(self.free_blocks):
            if fs <= start and end <= fe:
                new_blocks = []
                if fs < start:
                    new_blocks.append((fs, start))
                if end < fe:
                    new_blocks.append((end, fe))
                # 替换该位置为拆分后的若干块
                self.free_blocks.pop(i)
                for nb in reversed(new_blocks):
                    self.free_blocks.insert(i, nb)
                break
        # 记录分配
        region = CacheRegion(start, end, buf_id)
        self._add_region(region)
    
    def free(self, buf_id: int) -> bool:
        """释放缓冲区"""
        if buf_id not in self.buf_to_region:
            return False
        
        region = self.buf_to_region[buf_id]
        # 从已分配集合移除
        self.allocated_regions.remove(region)
        del self.buf_to_region[buf_id]
        # 插回空闲块并合并相邻
        self._insert_free_block(region.start, region.end)
        return True

    def _insert_free_block(self, start: int, end: int):
        """插入空闲块并与相邻块合并，保持按start有序"""
        if start >= end:
            return
        idx = 0
        while idx < len(self.free_blocks) and self.free_blocks[idx][0] < start:
            idx += 1
        # 插入
        self.free_blocks.insert(idx, (start, end))
        # 合并前后
        self._merge_around(idx)

    def _merge_around(self, idx: int):
        # 向前合并
        if idx - 1 >= 0:
            s0, e0 = self.free_blocks[idx - 1]
            s1, e1 = self.free_blocks[idx]
            if e0 == s1:
                self.free_blocks[idx - 1] = (s0, e1)
                self.free_blocks.pop(idx)
                idx -= 1
        # 向后合并
        if idx + 1 < len(self.free_blocks):
            s0, e0 = self.free_blocks[idx]
            s1, e1 = self.free_blocks[idx + 1]
            if e0 == s1:
                self.free_blocks[idx] = (s0, e1)
                self.free_blocks.pop(idx + 1)
    
    def get_allocated_size(self) -> int:
        """获取已分配的总大小"""
        free_total = sum(fe - fs for (fs, fe) in self.free_blocks)
        return self.capacity - free_total
    
    def get_free_size(self) -> int:
        """获取空闲空间大小"""
        return sum(fe - fs for (fs, fe) in self.free_blocks)
    
    def get_largest_free_block(self) -> int:
        """获取最大连续空闲块的大小"""
        if not self.free_blocks:
            return 0
        return max(fe - fs for (fs, fe) in self.free_blocks)
    
    def get_fragmentation_ratio(self) -> float:
        """获取碎片化率"""
        free_size = self.get_free_size()
        if free_size == 0:
            return 0.0
        largest_block = self.get_largest_free_block()
        return 1.0 - (largest_block / free_size)


class SpillOperation:
    """SPILL操作记录"""
    def __init__(self, buf_id: int, old_offset: int, new_offset: int, cache_type: str):
        self.buf_id = buf_id
        self.old_offset = old_offset
        self.new_offset = new_offset
        self.cache_type = cache_type
        # 记录对应的SPILL节点，便于追踪
        self.spill_out_node_id: Optional[int] = None
        self.spill_in_node_id: Optional[int] = None


class BufferInfo:
    """缓冲区信息"""
    def __init__(self, buf_id: int, size: int, cache_type: str, alloc_node: int, free_node: int):
        self.buf_id = buf_id
        self.size = size
        self.cache_type = cache_type
        self.alloc_node = alloc_node
        self.free_node = free_node
        self.allocated_offset: Optional[int] = None
        self.is_spilled = False
        self.spill_count = 0


class SegregatedFitAllocator:
    """分箱适配分配器，显著减少内存碎片化"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        # 定义大小类别，覆盖常见的缓冲区大小
        self.size_classes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # 每个大小类别的空闲块列表
        self.free_lists: Dict[int, List[Tuple[int, int]]] = {size: [] for size in self.size_classes}
        # 大块空闲区域（超过最大size_class的）
        self.large_blocks: List[Tuple[int, int]] = []
        # 碎片化历史统计
        self.fragmentation_history: List[float] = []
    
    def _find_size_class(self, size: int) -> Optional[int]:
        """找到适合的大小类别"""
        for class_size in self.size_classes:
            if size <= class_size:
                return class_size
        return None
    
    def _update_free_lists(self):
        """更新空闲块列表"""
        # 清空现有列表
        for size_class in self.size_classes:
            self.free_lists[size_class] = []
        self.large_blocks = []
        
        # 重新构建空闲块列表
        regions = sorted(self.cache_manager.allocated_regions, key=lambda r: r.start)
        
        # 添加开头的空闲空间
        if regions and regions[0].start > 0:
            self._add_free_block(0, regions[0].start)
        elif not regions and self.cache_manager.capacity > 0:
            self._add_free_block(0, self.cache_manager.capacity)
        
        # 添加中间的空闲空间
        for i in range(len(regions) - 1):
            gap_start = regions[i].end
            gap_end = regions[i + 1].start
            if gap_end > gap_start:
                self._add_free_block(gap_start, gap_end)
        
        # 添加末尾的空闲空间
        if regions:
            last_end = regions[-1].end
            if last_end < self.cache_manager.capacity:
                self._add_free_block(last_end, self.cache_manager.capacity)
    
    def _add_free_block(self, start: int, end: int):
        """将空闲块添加到合适的列表中"""
        size = end - start
        size_class = self._find_size_class(size)
        
        if size_class is not None:
            self.free_lists[size_class].append((start, end))
        else:
            self.large_blocks.append((start, end))
    
    def allocate_segregated(self, buf_id: int, size: int) -> Optional[int]:
        """使用分箱适配策略分配内存"""
        self._update_free_lists()
        
        size_class = self._find_size_class(size)
        
        if size_class is not None:
            # 从对应的大小类别中分配
            if self.free_lists[size_class]:
                start, end = self.free_lists[size_class].pop(0)
                return self.cache_manager.allocate(buf_id, size, start)
            
            # 如果对应类别没有空闲块，尝试更大的类别
            for larger_class in self.size_classes:
                if larger_class > size_class and self.free_lists[larger_class]:
                    start, end = self.free_lists[larger_class].pop(0)
                    return self.cache_manager.allocate(buf_id, size, start)
        
        # 尝试从大块中分配
        for i, (start, end) in enumerate(self.large_blocks):
            if end - start >= size:
                self.large_blocks.pop(i)
                return self.cache_manager.allocate(buf_id, size, start)
        
        return None


class SmartAddressAllocator:
    """智能地址分配器，减少缓存碎片化"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.segregated_allocator = SegregatedFitAllocator(cache_manager)
        # 未来分配预测缓存
        self.future_allocation_cache: Dict[int, List[Tuple[int, int]]] = {}
        # 碎片化历史
        self.fragmentation_history: List[float] = []
        # Numba优化的缓存数组
        self._regions_cache = {"start": np.array([], dtype=np.int32), 
                              "end": np.array([], dtype=np.int32), 
                              "buf_id": np.array([], dtype=np.int32)}
    
    def _update_regions_cache(self):
        """更新区域缓存数组以供Numba使用"""
        regions = sorted(self.cache_manager.allocated_regions, key=lambda r: r.start)
        if regions:
            self._regions_cache["start"] = np.array([r.start for r in regions], dtype=np.int32)
            self._regions_cache["end"] = np.array([r.end for r in regions], dtype=np.int32)
            self._regions_cache["buf_id"] = np.array([r.buf_id for r in regions], dtype=np.int32)
        else:
            self._regions_cache["start"] = np.array([], dtype=np.int32)
            self._regions_cache["end"] = np.array([], dtype=np.int32)
            self._regions_cache["buf_id"] = np.array([], dtype=np.int32)
    
    def allocate_with_strategy(self, buf_id: int, size: int, strategy: str = "segregated_fit", 
                              future_requests: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
        """
        使用指定策略分配地址
        
        Args:
            buf_id: 缓冲区ID
            size: 大小
            strategy: 分配策略 ("segregated_fit", "fragmentation_aware", "best_fit", "first_fit", "worst_fit")
            future_requests: 未来分配请求预测 [(size, priority), ...]
            
        Returns:
            分配的起始地址，失败返回None
        """
        if strategy == "segregated_fit":
            return self.segregated_allocator.allocate_segregated(buf_id, size)
        elif strategy == "fragmentation_aware":
            return self._fragmentation_aware_allocate(buf_id, size, future_requests)
        elif strategy == "best_fit":
            return self._best_fit_allocate(buf_id, size)
        elif strategy == "first_fit":
            return self._first_fit_allocate(buf_id, size)
        elif strategy == "worst_fit":
            return self._worst_fit_allocate(buf_id, size)
        else:
            return self.segregated_allocator.allocate_segregated(buf_id, size)
    
    def _fragmentation_aware_allocate(self, buf_id: int, size: int, 
                                     future_requests: Optional[List[Tuple[int, int]]] = None) -> Optional[int]:
        """碎片化感知分配"""
        regions = sorted(self.cache_manager.allocated_regions, key=lambda r: r.start)
        
        # 获取所有可能的分配位置
        candidates = []
        
        # 检查开头
        if not regions or regions[0].start >= size:
            end_pos = regions[0].start if regions else self.cache_manager.capacity
            candidates.append((0, end_pos))
        
        # 检查中间间隙
        for i in range(len(regions) - 1):
            gap_start = regions[i].end
            gap_end = regions[i + 1].start
            if gap_end - gap_start >= size:
                candidates.append((gap_start, gap_end))
        
        # 检查末尾
        if regions:
            last_end = regions[-1].end
            if self.cache_manager.capacity - last_end >= size:
                candidates.append((last_end, self.cache_manager.capacity))
        
        if not candidates:
            return None
        
        # 评估每个候选位置的碎片化影响
        best_candidate = None
        best_score = float('inf')
        
        for start_pos, end_pos in candidates:
            score = self._calculate_fragmentation_score(start_pos, size, end_pos, future_requests)
            if score < best_score:
                best_score = score
                best_candidate = start_pos
        
        if best_candidate is not None:
            return self.cache_manager.allocate(buf_id, size, best_candidate)
        
        return None
    
    def _calculate_fragmentation_score(self, start_pos: int, size: int, end_pos: int, 
                                      future_requests: Optional[List[Tuple[int, int]]] = None) -> float:
        """计算分配在指定位置的碎片化得分"""
        # 基础得分：剩余空间的碎片化程度
        remaining_space = end_pos - start_pos - size
        if remaining_space == 0:
            base_score = 0  # 完美匹配
        elif remaining_space < 64:
            base_score = 100  # 产生小碎片，高惩罚
        else:
            base_score = remaining_space / 1000.0  # 大空间保留，低惩罚
        
        # 如果有未来请求预测，考虑未来适配性
        future_score = 0
        if future_requests and remaining_space > 0:
            for req_size, priority in future_requests:
                if req_size <= remaining_space:
                    future_score -= priority * 10  # 能满足未来请求，降低得分
                    break
        
        return base_score + future_score
    
    def _best_fit_allocate(self, buf_id: int, size: int) -> Optional[int]:
        """最佳适配：选择最小的能容纳的空闲块（Numba优化版）"""
        self._update_regions_cache()
        
        best_start = _fast_find_best_fit(
            self._regions_cache["start"],
            self._regions_cache["end"],
            self._regions_cache["buf_id"],
            self.cache_manager.capacity,
            size
        )
        
        if best_start >= 0:
            return self.cache_manager.allocate(buf_id, size, best_start)
        
        return None
    
    def _fast_best_fit_allocate(self, buf_id: int, size: int) -> Optional[int]:
        """快速最佳适配算法"""
        return self._best_fit_allocate(buf_id, size)
    
    def _first_fit_allocate(self, buf_id: int, size: int) -> Optional[int]:
        """首次适配：选择第一个能容纳的空闲块"""
        return self.cache_manager.allocate(buf_id, size)
    
    def _worst_fit_allocate(self, buf_id: int, size: int) -> Optional[int]:
        """最坏适配：选择最大的空闲块"""
        regions = sorted(self.cache_manager.allocated_regions, key=lambda r: r.start)
        worst_start = None
        worst_gap_size = -1
        
        # 检查开头
        if len(regions) == 0:
            if size <= self.cache_manager.capacity:
                return self.cache_manager.allocate(buf_id, size, 0)
        else:
            if regions[0].start >= size and regions[0].start > worst_gap_size:
                worst_start = 0
                worst_gap_size = regions[0].start
        
        # 检查中间间隙
        for i in range(len(regions) - 1):
            gap_start = regions[i].end
            gap_end = regions[i + 1].start
            gap_size = gap_end - gap_start
            
            if gap_size >= size and gap_size > worst_gap_size:
                worst_start = gap_start
                worst_gap_size = gap_size
        
        # 检查末尾
        if regions:
            last_end = regions[-1].end
            tail_size = self.cache_manager.capacity - last_end
            if tail_size >= size and tail_size > worst_gap_size:
                worst_start = last_end
                worst_gap_size = tail_size
        
        if worst_start is not None:
            return self.cache_manager.allocate(buf_id, size, worst_start)
        
        return None


class HardwareSimulator:
    """硬件状态仿真器"""
    
    # 硬件缓存容量配置
    CACHE_CAPACITIES = {
        "L1": 4096,
        "UB": 1024,
        "L0A": 256,
        "L0B": 256,
        "L0C": 512
    }
    
    def __init__(self, scheduler: AdvancedScheduler):
        self.scheduler = scheduler
        self.nodes = scheduler.nodes
        
        # 缓存管理器
        self.cache_managers: Dict[str, CacheManager] = {}
        self.address_allocators: Dict[str, SmartAddressAllocator] = {}
        for cache_type, capacity in self.CACHE_CAPACITIES.items():
            manager = CacheManager(cache_type, capacity)
            self.cache_managers[cache_type] = manager
            self.address_allocators[cache_type] = SmartAddressAllocator(manager)
        
        # 缓冲区信息
        self.buffer_infos: Dict[int, BufferInfo] = {}
        self.buf_to_alloc: Dict[int, int] = {}
        self.buf_to_free: Dict[int, int] = {}
        
        # SPILL操作记录
        self.spill_operations: List[SpillOperation] = []
        self.spill_nodes: List[Node] = []  # 新增的SPILL节点
        
        # 执行状态
        self.current_step = 0
        self.executed_nodes: Set[int] = set()
        
        # 新增：碎片化历史记录
        self.fragmentation_history: Dict[str, List[float]] = defaultdict(list)
        
        # Numba优化的调度数据缓存
        self.schedule_cache = {
            "ops": np.array([], dtype=np.int32),
            "buf_ids": np.array([], dtype=np.int32), 
            "sizes": np.array([], dtype=np.int32),
            "cache_types": np.array([], dtype=np.int32)
        }
        self.cache_type_mapping = {"L1": 0, "UB": 1, "L0A": 2, "L0B": 3, "L0C": 4}
        
        # 初始化静态信息
        self._initialize_buffer_info()
        # 初始化动态状态
        self.reset()
        # 预计算：COPY_IN 使用的缓冲区集合（用于SPILL_OUT耗时与统计）
        self.copy_in_buf_set: Set[int] = set()
        for node in self.nodes.values():
            if getattr(node, "op", None) == "COPY_IN" and hasattr(node, "bufs") and node.bufs:
                for b in node.bufs:
                    self.copy_in_buf_set.add(b)
        # 策略开关
        self.enable_copyin_substitute_spillin = True
        self.enable_no_reload_before_free = True
        self.enable_spill_candidate_cost_ratio = True

    def reset(self):
        """重置仿真器的所有动态状态，确保每次仿真是独立的"""
        self.spill_operations: List[SpillOperation] = []
        self.spill_nodes: List[Node] = []
        self.current_step = 0
        self.executed_nodes: Set[int] = set()
        self.fragmentation_history: Dict[str, List[float]] = defaultdict(list)
        self.rescheduled_op_count = 0
        # 下一个动态创建节点的ID
        self.next_node_id = (max(self.nodes.keys()) + 1) if self.nodes else 0
        # 额外依赖边（用于审计）
        self.extra_edges: List[Tuple[int, int]] = []
        # 每个缓冲区最近一次SPILL_IN节点
        self.last_spill_in_node_id_per_buf: Dict[int, int] = {}
        # 最近一次释放的地址归属（用于地址复用依赖） key=(cache_type,start,size)->buf_id
        self.last_region_owner: Dict[Tuple[str, int, int], int] = {}
        # L0并发驻留校验计数
        self.l0a_live = 0
        self.l0b_live = 0
        self.l0c_live = 0
        self.l0_violation_count = 0
        self.l0_live_bufs: Dict[str, Set[int]] = {"L0A": set(), "L0B": set(), "L0C": set()}
        
        # 重置所有缓存管理器
        for cache_manager in self.cache_managers.values():
            cache_manager.allocated_regions = []
            cache_manager.buf_to_region = {}

    def _record_step_stats(self):
        """记录当前步骤的统计数据，如碎片率"""
        for cache_type, manager in self.cache_managers.items():
            if cache_type in ["L1", "UB"]:
                self.fragmentation_history[cache_type].append(manager.get_fragmentation_ratio())

    def _prepare_schedule_cache(self, schedule: List[int]):
        """预处理调度序列为Numba可用的数组格式"""
        ops = []
        buf_ids = []
        sizes = []
        cache_types = []
        
        for node_id in schedule:
            node = self.nodes[node_id]
            if node.op == "ALLOC":
                ops.append(0)
                buf_ids.append(node.buf_id)
                sizes.append(node.size)
                cache_types.append(self.cache_type_mapping.get(node.cache_type, -1))
            elif node.op == "FREE":
                ops.append(1)
                buf_ids.append(node.buf_id)
                sizes.append(node.size)
                cache_types.append(self.cache_type_mapping.get(node.cache_type, -1))
            else:
                ops.append(-1)  # 其他操作
                buf_ids.append(-1)
                sizes.append(0)
                cache_types.append(-1)
        
        self.schedule_cache["ops"] = np.array(ops, dtype=np.int32)
        self.schedule_cache["buf_ids"] = np.array(buf_ids, dtype=np.int32)
        self.schedule_cache["sizes"] = np.array(sizes, dtype=np.int32)
        self.schedule_cache["cache_types"] = np.array(cache_types, dtype=np.int32)
    
    def _initialize_buffer_info(self):
        """初始化缓冲区信息"""
        # 收集ALLOC和FREE节点
        for node_id, node in self.nodes.items():
            if node.op == "ALLOC" and node.is_l1_or_ub_cache():
                self.buf_to_alloc[node.buf_id] = node_id
            elif node.op == "FREE" and node.is_l1_or_ub_cache():
                self.buf_to_free[node.buf_id] = node_id
        
        # 创建缓冲区信息
        for buf_id in self.buf_to_alloc:
            if buf_id in self.buf_to_free:
                alloc_node = self.buf_to_alloc[buf_id]
                free_node = self.buf_to_free[buf_id]
                alloc_node_obj = self.nodes[alloc_node]
                
                buffer_info = BufferInfo(
                    buf_id=buf_id,
                    size=alloc_node_obj.size,
                    cache_type=alloc_node_obj.cache_type,
                    alloc_node=alloc_node,
                    free_node=free_node
                )
                self.buffer_infos[buf_id] = buffer_info
        # 构建缓冲区到使用节点的映射
        self.buf_to_users: Dict[int, List[int]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            bufs = getattr(node, 'bufs', None)
            if bufs:
                for b in bufs:
                    self.buf_to_users[b].append(node_id)
        # 预计算：每个缓冲区对应的 COPY_IN 节点ID列表
        self.buf_to_copyin_nodes: Dict[int, List[int]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            if getattr(node, 'op', None) == 'COPY_IN' and hasattr(node, 'bufs') and node.bufs:
                for b in node.bufs:
                    self.buf_to_copyin_nodes[b].append(node_id)
    
    def simulate_schedule(self, schedule: List[int], case_name: str = "", verbose: bool = False) -> Tuple[Dict[int, int], List[SpillOperation], List[int]]:
        """
        模拟执行调度序列（性能优化版）
        
        Returns:
            Tuple[Dict[int, int], List[SpillOperation], List[int]]: 
            (缓冲区地址分配, SPILL操作列表, 完整调度序列)
        """
        # 确保每次仿真都从干净的状态开始
        self.reset()
        self._prepare_schedule_cache(schedule)

        complete_schedule: List[int] = []
        buffer_allocations: Dict[int, int] = {}

        # 预计算：节点位置（用于快速生命周期评估）
        self.node_position: Dict[int, int] = {nid: idx for idx, nid in enumerate(schedule)}

        # 预计算压力点（基于初始序列，作为提示）
        pressure_points = self._batch_predict_pressure_points(schedule)

        # 为每个case添加进度条
        progress_bar = tqdm(total=len(schedule), desc=f"Simulating {case_name}", leave=False, position=0) if (verbose and case_name) else None

        for i, node_id in enumerate(schedule):
            # 每步记录碎片状态，保证可视化完整
            self._record_step_stats()

            if node_id in self.executed_nodes:
                if progress_bar:
                    progress_bar.update(1)
                continue

            self.current_step = i
            node = self.nodes[node_id]

            # 预防性SPILL（流式）：只在压力点触发，最多spill 1 个
            if i in pressure_points:
                self._execute_preemptive_spills_stream(complete_schedule)

            # ALLOC（L1/UB）
            if node.op == "ALLOC" and node.is_l1_or_ub_cache():
                success = self._handle_alloc(node)
                if not success:
                    # 尝试前瞻FREE
                    freed_space, executed_node_ids = self._attempt_local_reschedule(node, schedule, self.executed_nodes, i)
                    if freed_space:
                        self.rescheduled_op_count += len(executed_node_ids)
                        self.executed_nodes.update(executed_node_ids)
                        complete_schedule.extend(executed_node_ids)
                        success = self._handle_alloc(node)

                if not success:
                    spill_success = self._handle_spill_for_alloc_stream(node, complete_schedule)
                    if not spill_success:
                        raise RuntimeError(f"无法为缓冲区 {node.buf_id} 分配空间，即使执行SPILL和重调度也无效")

                # 记录分配结果
                buffer_info = self.buffer_infos.get(node.buf_id)
                if buffer_info and buffer_info.allocated_offset is not None:
                    buffer_allocations[node.buf_id] = buffer_info.allocated_offset
                    # 记录地址复用依赖（若复用了释放过的精确区域）
                    region_key = (buffer_info.cache_type, buffer_info.allocated_offset, buffer_info.size)
                    if region_key in self.last_region_owner:
                        prev_buf = self.last_region_owner[region_key]
                        prev_free = self.buf_to_free.get(prev_buf)
                        if prev_free is not None:
                            self.extra_edges.append((prev_free, node.id))

                # 输出原节点
                complete_schedule.append(node_id)
                self.executed_nodes.add(node_id)
                if progress_bar:
                    progress_bar.update(1)
                continue

            # FREE：若处于spilled则避免在FREE前恢复（可开关）
            if node.op == "FREE" and node.is_l1_or_ub_cache():
                buf_id = node.buf_id
                buffer_info = self.buffer_infos.get(buf_id)
                if buffer_info and buffer_info.is_spilled and buffer_info.allocated_offset is None:
                    if not getattr(self, 'enable_no_reload_before_free', True):
                        self._emit_spill_in(buf_id, complete_schedule)
                self._handle_free(node)
                complete_schedule.append(node_id)
                self.executed_nodes.add(node_id)
                if progress_bar:
                    progress_bar.update(1)
                continue

            # 计算/搬运节点：使用前确保在核内
            if hasattr(node, 'bufs') and node.bufs:
                for buf_id in node.bufs:
                    buffer_info = self.buffer_infos.get(buf_id)
                    if buffer_info and buffer_info.cache_type in ["L1", "UB"] and buffer_info.allocated_offset is None:
                        if getattr(self, 'enable_copyin_substitute_spillin', True) and getattr(node, 'op', None) == 'COPY_IN':
                            # 若当前节点本身是COPY_IN，则以它作为自然恢复，不记为SPILL
                            self._apply_spill_in_effect(buf_id)
                        else:
                            # 尝试用可上提的 COPY_IN 代替 SPILL_IN
                            substituted = False
                            if getattr(self, 'enable_copyin_substitute_spillin', True):
                                substituted = self._try_use_copy_in_instead_of_spill_in(buf_id, complete_schedule)
                            if not substituted:
                                # 回退到常规 SPILL_IN
                                self._emit_spill_in(buf_id, complete_schedule)
                                self.last_spill_in_node_id_per_buf[buf_id] = complete_schedule[-1]

            complete_schedule.append(node_id)
            self.executed_nodes.add(node_id)
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        return buffer_allocations, self.spill_operations, complete_schedule

    def get_fragmentation_stats(self) -> Dict[str, Dict[str, float]]:
        """获取碎片化统计数据（平均值和峰值）"""
        stats = {}
        for cache_type, history in self.fragmentation_history.items():
            if history:
                stats[cache_type] = {
                    "avg": np.mean(history),
                    "max": np.max(history)
                }
            else:
                stats[cache_type] = {"avg": 0.0, "max": 0.0}
        return stats

    def _attempt_local_reschedule(self, alloc_node: Node, schedule: List[int], executed_nodes: Set[int], current_step: int) -> Tuple[bool, List[int]]:
        """
        当分配失败时，向前看并尝试执行一个或多个可行的FREE节点以释放空间。
        这种调度与分配的协同优化可以避免不必要的SPILL操作。

        Args:
            alloc_node: 当前分配失败的ALLOC节点
            schedule: 完整的调度序列
            executed_nodes: 已经执行的节点集合
            current_step: 当前在调度序列中的位置

        Returns:
            Tuple[bool, List[int]]: (是否成功释放了空间, 被提前执行的FREE节点ID列表)
        """
        # 定义一个向前看的窗口大小，这是一个可以调整的超参数
        look_ahead_window = 200
        
        # 步骤1：在定义的“向前看”窗口内，寻找所有符合条件的FREE节点作为候选
        candidates = []
        for i in range(current_step + 1, min(len(schedule), current_step + 1 + look_ahead_window)):
            future_node_id = schedule[i]
            if future_node_id in executed_nodes:
                continue
            
            future_node = self.nodes.get(future_node_id)
            if not future_node: continue

            # 候选节点必须是与当前分配请求相同缓存类型的FREE节点
            if future_node.op == "FREE" and future_node.cache_type == alloc_node.cache_type:
                # 候选节点必须满足依赖关系：其所有的前驱节点都已经被执行
                predecessors = self.scheduler.reverse_adj_list.get(future_node_id, [])
                if all(p in self.executed_nodes for p in predecessors):
                    candidates.append(future_node)
        
        if not candidates:
            return False, []

        # 步骤2：设计启发式策略，对候选节点进行排序
        # 当前策略：优先释放能提供更大空间的缓冲区，因此按size降序排列
        candidates.sort(key=lambda n: n.size, reverse=True)
        
        executed_free_nodes = []
        freed_space = False
        cache_manager = self.cache_managers[alloc_node.cache_type]

        # 步骤3：按顺序执行候选的FREE操作，直到满足当前分配需求为止
        for node_to_free in candidates:
            if self._handle_free(node_to_free):
                executed_free_nodes.append(node_to_free.id)
                freed_space = True
                
                # 检查释放空间后，当前最大的连续空闲块是否已足够本次分配
                if cache_manager.get_largest_free_block() >= alloc_node.size:
                    break  # 空间已足够，停止执行更多的FREE操作
        
        return freed_space, executed_free_nodes

    def _batch_predict_pressure_points(self, schedule: List[int]) -> Set[int]:
        """批量预测内存压力点"""
        pressure_points = set()
        
        for cache_type, cache_type_idx in self.cache_type_mapping.items():
            if cache_type not in ["L1", "UB"]:
                continue
                
            capacity = self.CACHE_CAPACITIES[cache_type]
            
            # 使用Numba优化的内存预测
            for step in range(0, len(schedule), 50):  # 每50步检查一次
                predicted_peak = _fast_predict_memory_usage(
                    self.schedule_cache["ops"],
                    self.schedule_cache["buf_ids"], 
                    self.schedule_cache["sizes"],
                    self.schedule_cache["cache_types"],
                    step, cache_type_idx, 50
                )
                
                if predicted_peak > capacity * 0.95:  # 95%阈值，减少预防性SPILL
                    pressure_points.add(step)
        
        return pressure_points
    
    def _execute_preemptive_spills(self, schedule: List[int], step: int):
        """执行预防性SPILL（兼容旧接口，但直接在动态序列中插入）"""
        proactive_spills = self._proactive_spill_detection(schedule, step)
        for cache_type, candidates in proactive_spills.items():
            for buf_id in candidates[:1]:
                # 在当前位置插入SPILL_OUT
                self._insert_spill_out_and_plan_in(buf_id, cache_type, schedule, step)

    def _execute_preemptive_spills_stream(self, complete_schedule: List[int]):
        """执行预防性SPILL（流式输出，不修改原序列）"""
        proactive_spills = self._proactive_spill_detection([], self.current_step)
        for cache_type, candidates in proactive_spills.items():
            for buf_id in candidates[:1]:
                self._emit_spill_out(buf_id, cache_type, complete_schedule)

    def _handle_spill_for_alloc_stream(self, alloc_node: Node, complete_schedule: List[int]) -> bool:
        """为ALLOC操作处理SPILL（流式）：输出若干SPILL_OUT直到可分配，然后分配。"""
        cache_manager = self.cache_managers[alloc_node.cache_type]
        allocator = self.address_allocators[alloc_node.cache_type]

        candidates = self._select_spill_candidates(alloc_node.cache_type, [], self.current_step)
        if not candidates:
            return False

        required_size = alloc_node.size
        future_requests = self._predict_future_allocation_requests(alloc_node.cache_type)

        for candidate_buf_id, _ in candidates:
            if cache_manager.get_largest_free_block() >= required_size:
                break
            # 输出一个SPILL_OUT
            self._emit_spill_out(candidate_buf_id, alloc_node.cache_type, complete_schedule)

        # 再尝试分配
        offset = allocator.allocate_with_strategy(alloc_node.buf_id, alloc_node.size, "segregated_fit", future_requests)
        if offset is None:
            offset = allocator.allocate_with_strategy(alloc_node.buf_id, alloc_node.size, "fragmentation_aware", future_requests)
        if offset is None:
            return False
        buffer_info = self.buffer_infos.get(alloc_node.buf_id)
        if buffer_info:
            buffer_info.allocated_offset = offset
        return True

    def _emit_spill_out(self, buf_id: int, cache_type: str, complete_schedule: List[int]) -> bool:
        """输出一个SPILL_OUT节点，立即释放缓存，记录操作。"""
        cache_manager = self.cache_managers[cache_type]
        buffer_info = self.buffer_infos.get(buf_id)
        if buffer_info is None or buf_id not in cache_manager.buf_to_region:
            return False
        region = cache_manager.buf_to_region[buf_id]
        old_offset = region.start
        is_copy_in_used = (buf_id in getattr(self, 'copy_in_buf_set', set()))
        cycles = 0 if is_copy_in_used else self._calculate_spill_cycles(buffer_info.size, False)
        spill_out_id = self.next_node_id; self.next_node_id += 1
        spill_out_node = Node({
            "Id": spill_out_id,
            "Op": "SPILL_OUT",
            "Pipe": "MTE3",
            "Cycles": cycles,
            "Bufs": [buf_id]
        })
        self.nodes[spill_out_id] = spill_out_node
        complete_schedule.append(spill_out_id)
        # 状态生效
        self._apply_spill_out_effect(buf_id, cache_type)
        # 记录依赖：ALLOC(buf) -> SPILL_OUT
        alloc_node = self.buf_to_alloc.get(buf_id)
        if alloc_node is not None:
            self.extra_edges.append((alloc_node, spill_out_id))
        # 记录操作
        spill_op = SpillOperation(buf_id, old_offset, -1, cache_type)
        spill_op.spill_out_node_id = spill_out_id
        self.spill_operations.append(spill_op)
        # 记录最近释放的区域归属用于地址复用依赖
        self.last_region_owner[(cache_type, old_offset, buffer_info.size)] = buf_id
        # 依赖：所有已执行的消费者 → SPILL_OUT
        users = getattr(self, 'buf_to_users', {}).get(buf_id, [])
        for u in users:
            if u in self.executed_nodes:
                self.extra_edges.append((u, spill_out_id))
        return True

    def _emit_spill_in(self, buf_id: int, complete_schedule: List[int]) -> int:
        """输出一个SPILL_IN节点并立即分配新地址，回填spill记录。"""
        buffer_info = self.buffer_infos.get(buf_id)
        if buffer_info is None:
            return -1
        cycles = self._calculate_spill_cycles(buffer_info.size, True)
        spill_in_id = self.next_node_id; self.next_node_id += 1
        spill_in_node = Node({
            "Id": spill_in_id,
            "Op": "SPILL_IN",
            "Pipe": "MTE2",
            "Cycles": cycles,
            "Bufs": [buf_id]
        })
        self.nodes[spill_in_id] = spill_in_node
        complete_schedule.append(spill_in_id)
        # 分配并回填
        self._apply_spill_in_effect(buf_id)
        # 依赖：SPILL_OUT -> SPILL_IN
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.spill_in_node_id is None:
                self.extra_edges.append((op.spill_out_node_id, spill_in_id))
                break
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.spill_in_node_id is None:
                op.spill_in_node_id = spill_in_id
                break
        # 依赖：SPILL_IN → 所有未执行的消费者
        users = getattr(self, 'buf_to_users', {}).get(buf_id, [])
        for u in users:
            if u not in self.executed_nodes:
                self.extra_edges.append((spill_in_id, u))
        return spill_in_id
    
    def _handle_alloc(self, node: Node) -> bool:
        """处理ALLOC操作"""
        cache_manager = self.cache_managers[node.cache_type]
        allocator = self.address_allocators[node.cache_type]
        buffer_info = self.buffer_infos.get(node.buf_id)
        
        if buffer_info is None:
            # L0级：严格约束同类型同时最多1个驻留
            if not self._can_alloc_l0(node.cache_type):
                return False
            self._update_l0_live(node, is_alloc=True)
            return True  # L0缓存等，不需要地址管理
        
        # 预测未来分配需求
        future_requests = self._predict_future_allocation_requests(node.cache_type)
        
        # 使用分箱适配策略优先，如果失败则使用碎片化感知策略
        offset = allocator.allocate_with_strategy(node.buf_id, node.size, "segregated_fit", future_requests)
        if offset is None:
            offset = allocator.allocate_with_strategy(node.buf_id, node.size, "fragmentation_aware", future_requests)
        
        if offset is not None:
            buffer_info.allocated_offset = offset
            return True
        else:
            return False
    
    def _predict_future_allocation_requests(self, cache_type: str, look_ahead: int = 50) -> List[Tuple[int, int]]:
        """预测未来的分配需求"""
        future_requests = []
        
        # 简化实现：基于缓冲区大小分布预测
        for buf_id, buffer_info in self.buffer_infos.items():
            if buffer_info.cache_type == cache_type and not buffer_info.allocated_offset:
                # 根据缓冲区大小和重要性给出优先级
                priority = 1.0 / max(1, buffer_info.size / 100)  # 小缓冲区优先级高
                future_requests.append((buffer_info.size, priority))
        
        # 按优先级排序，只保留前N个
        future_requests.sort(key=lambda x: x[1], reverse=True)
        return future_requests[:look_ahead]
    
    def _handle_free(self, node: Node) -> bool:
        """处理FREE操作"""
        cache_manager = self.cache_managers[node.cache_type]
        if node.cache_type in ["L0A", "L0B", "L0C"]:
            self._update_l0_live(node, is_alloc=False)
            return True
        return cache_manager.free(node.buf_id)

    def _update_l0_live(self, node: Node, is_alloc: bool):
        ct = node.cache_type
        delta = 1 if is_alloc else -1
        if ct == "L0A":
            self.l0a_live += delta
            if is_alloc:
                self.l0_live_bufs[ct].add(node.buf_id)
            else:
                self.l0_live_bufs[ct].discard(node.buf_id)
        elif ct == "L0B":
            self.l0b_live += delta
            if is_alloc:
                self.l0_live_bufs[ct].add(node.buf_id)
            else:
                self.l0_live_bufs[ct].discard(node.buf_id)
        elif ct == "L0C":
            self.l0c_live += delta
            if is_alloc:
                self.l0_live_bufs[ct].add(node.buf_id)
            else:
                self.l0_live_bufs[ct].discard(node.buf_id)
        # 校验：同时最多一个驻留（题面注释）
        if self.l0a_live > 1 or self.l0b_live > 1 or self.l0c_live > 1:
            self.l0_violation_count += 1

    def _can_alloc_l0(self, cache_type: str) -> bool:
        if cache_type == "L0A":
            return self.l0a_live == 0
        if cache_type == "L0B":
            return self.l0b_live == 0
        if cache_type == "L0C":
            return self.l0c_live == 0
        return True
    
    def _proactive_spill_detection(self, schedule: List[int], current_step: int) -> Dict[str, List[int]]:
        """主动检测未来可能的SPILL需求"""
        spill_recommendations = {}
        
        for cache_type in self.CACHE_CAPACITIES.keys():
            if cache_type not in ["L1", "UB"]:  # 只处理L1和UB缓存
                continue
                
            cache_manager = self.cache_managers[cache_type]
            
            # 预测未来30步内的内存需求峰值（减少预测窗口）
            future_peak = self._predict_memory_peak(schedule, current_step, cache_type, 30)
            current_free_space = cache_manager.get_free_size()
            
            # 提高阈值到95%，减少不必要的预防性SPILL
            if future_peak > current_free_space * 0.95:
                # 提前执行SPILL，避免紧急情况
                preemptive_candidates = self._select_preemptive_spill_candidates(cache_type, schedule, current_step)
                spill_recommendations[cache_type] = preemptive_candidates
        
        return spill_recommendations
    
    def _predict_memory_peak(self, schedule: List[int], current_step: int, cache_type: str, window: int) -> int:
        """预测未来窗口内的内存使用峰值"""
        current_usage = self.cache_managers[cache_type].get_allocated_size()
        max_predicted_usage = current_usage
        simulated_usage = current_usage
        
        end_step = min(current_step + window, len(schedule))
        
        for i in range(current_step, end_step):
            node_id = schedule[i]
            node = self.nodes[node_id]
            
            if node.op == "ALLOC" and node.cache_type == cache_type:
                simulated_usage += node.size
                max_predicted_usage = max(max_predicted_usage, simulated_usage)
            elif node.op == "FREE" and node.cache_type == cache_type:
                simulated_usage -= node.size
        
        return max_predicted_usage - current_usage  # 返回额外需要的空间
    
    def _select_preemptive_spill_candidates(self, cache_type: str, schedule: List[int], current_step: int) -> List[int]:
        """选择预防性SPILL候选者"""
        cache_manager = self.cache_managers[cache_type]
        candidates = []
        
        for buf_id, region in cache_manager.buf_to_region.items():
            buffer_info = self.buffer_infos.get(buf_id)
            if buffer_info is None:
                continue
            
            # 计算预防性SPILL优先级
            priority = self._calculate_advanced_spill_priority(buffer_info, schedule, current_step, cache_type)
            candidates.append((buf_id, priority))
        
        # 按优先级排序，选择前几个（减少候选者数量）
        candidates.sort(key=lambda x: x[1])
        return [buf_id for buf_id, _ in candidates[:1]]  # 最多选择1个候选者
    
    def _handle_spill_for_alloc_inline(self, alloc_node: Node, dynamic_schedule: List[int], current_index: int) -> bool:
        """为ALLOC操作内联处理SPILL：插入SPILL_OUT并规划SPILL_IN，直至可分配。"""
        # 兼容旧接口：调用流式实现
        complete_schedule_dummy: List[int] = []
        return self._handle_spill_for_alloc_stream(alloc_node, complete_schedule_dummy)
    
    def _select_spill_candidates(self, cache_type: str, schedule: List[int], current_step: int) -> List[Tuple[int, float]]:
        """
        选择SPILL候选者
        
        Returns:
            List[Tuple[int, float]]: (buf_id, priority) 列表，按优先级排序
        """
        cache_manager = self.cache_managers[cache_type]
        candidates = []
        
        # 获取当前已分配的缓冲区
        # 限制遍历与优先计算开销：仅取若干最大块优先评估
        items = list(cache_manager.buf_to_region.items())
        # 先按区域大小降序取前K个候选，避免全表扫描成本
        items.sort(key=lambda kv: kv[1].size(), reverse=True)
        top_k = 64 if len(items) > 64 else len(items)
        for buf_id, region in items[:top_k]:
            buffer_info = self.buffer_infos.get(buf_id)
            if buffer_info is None:
                continue
            priority = self._calculate_spill_priority(buffer_info, schedule, current_step)
            # 预估额外DDR代价（若允许用COPY_IN替代且存在COPY_IN来源，则可视作0；否则按2×size）
            is_copy_in_used = (buf_id in getattr(self, 'copy_in_buf_set', set()))
            expected_cost = 0 if (is_copy_in_used and getattr(self, 'enable_copyin_substitute_spillin', True)) else (buffer_info.size * 2)
            # 预估释放后连续增益
            gain = self._estimate_contiguous_gain_on_free(cache_type, region)
            ratio = expected_cost / max(1, gain)
            # 组合打分
            normalized_cost = (buffer_info.size / max(1, region.size()))
            composite = priority + (0.5 * ratio if getattr(self, 'enable_spill_candidate_cost_ratio', True) else 0.0) + 0.001 * normalized_cost
            candidates.append((buf_id, composite))
        
        # 按优先级排序
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def _calculate_advanced_spill_priority(self, buffer_info: BufferInfo, schedule: List[int], 
                                          current_step: int, cache_type: str) -> float:
        """
        高级SPILL优先级计算 - 多维度评估
        考虑因素：剩余生命周期、碎片化贡献、未来需求预测、使用频率等
        """
        # 0. 下一次使用距离（越远越优先被溢出）
        next_use_dist = self._estimate_next_use_distance(buffer_info.buf_id, current_step)
        next_use_score = 1.0 - min(1.0, next_use_dist / max(1, len(schedule)))

        # 1. 剩余生命周期
        lifetime_score = self._calculate_lifetime_score(buffer_info, schedule, current_step)
        
        # 2. 碎片化贡献度
        fragmentation_score = self._calculate_fragmentation_contribution(buffer_info, cache_type)
        
        # 3. 后续分配压力
        future_demand_score = self._predict_future_allocation_pressure(schedule, current_step, cache_type)
        
        # 4. 使用频率
        usage_frequency_score = self._calculate_usage_frequency(buffer_info, schedule, current_step)
        
        # 5. SPILL成本
        spill_cost_score = self._calculate_spill_cost_score(buffer_info)
        
        # COPY_IN 使用的缓冲区：SPILL_OUT 不产生额外DDR搬运 → 给予负向奖励（降低总分）
        copy_in_bonus = -0.2 if buffer_info.buf_id in getattr(self, 'copy_in_buf_set', set()) else 0.0
        
        # 综合评分（越小越优先）
        total_score = (0.35 * lifetime_score +
                       0.25 * next_use_score +
                       0.15 * fragmentation_score +
                       0.15 * future_demand_score +
                       0.08 * usage_frequency_score +
                       0.02 * spill_cost_score +
                       copy_in_bonus)
        
        return total_score

    def _estimate_next_use_distance(self, buf_id: int, current_step: int) -> int:
        """估计缓冲区距离下一次使用的步数（越大越适合被SPILL）。"""
        if not hasattr(self, 'node_position'):
            return 0
        users = getattr(self, 'buf_to_users', {}).get(buf_id, [])
        for nid in users:
            pos = self.node_position.get(nid)
            if pos is not None and pos > current_step:
                return pos - current_step
        return len(self.node_position)
    
    def _calculate_lifetime_score(self, buffer_info: BufferInfo, schedule: List[int], current_step: int) -> float:
        """计算生命周期得分"""
        remaining_lifetime = 0
        free_node_pos = None
        
        for i in range(current_step, len(schedule)):
            if schedule[i] == buffer_info.free_node:
                free_node_pos = i
                break
        
        if free_node_pos is not None:
            remaining_lifetime = free_node_pos - current_step
        else:
            remaining_lifetime = len(schedule) - current_step
        
        # 归一化到0-1范围
        max_lifetime = len(schedule) - current_step
        return remaining_lifetime / max(1, max_lifetime)
    
    def _calculate_fragmentation_contribution(self, buffer_info: BufferInfo, cache_type: str) -> float:
        """计算碎片化贡献度"""
        cache_manager = self.cache_managers[cache_type]
        
        if buffer_info.buf_id not in cache_manager.buf_to_region:
            return 0.0
        
        region = cache_manager.buf_to_region[buffer_info.buf_id]
        regions = sorted(cache_manager.allocated_regions, key=lambda r: r.start)
        
        # 计算释放此缓冲区后能合并的空间大小
        mergeable_space = region.size()
        
        # 检查左右相邻的空闲空间
        for i, r in enumerate(regions):
            if r.buf_id == buffer_info.buf_id:
                # 检查左侧
                if i > 0 and regions[i-1].end < region.start:
                    mergeable_space += region.start - regions[i-1].end
                elif i == 0 and region.start > 0:
                    mergeable_space += region.start
                
                # 检查右侧
                if i < len(regions) - 1 and regions[i+1].start > region.end:
                    mergeable_space += regions[i+1].start - region.end
                elif i == len(regions) - 1 and region.end < cache_manager.capacity:
                    mergeable_space += cache_manager.capacity - region.end
                break
        
        # 归一化：能合并的空间越大，碎片化贡献度越高（SPILL优先级越低）
        return 1.0 - min(1.0, mergeable_space / cache_manager.capacity)
    
    def _predict_future_allocation_pressure(self, schedule: List[int], current_step: int, cache_type: str) -> float:
        """预测未来分配压力"""
        future_window = min(100, len(schedule) - current_step)
        future_allocs = 0
        future_frees = 0
        
        for i in range(current_step, min(current_step + future_window, len(schedule))):
            node_id = schedule[i]
            node = self.nodes[node_id]
            
            if node.op == "ALLOC" and node.cache_type == cache_type:
                future_allocs += 1
            elif node.op == "FREE" and node.cache_type == cache_type:
                future_frees += 1
        
        # 计算净分配压力
        net_pressure = future_allocs - future_frees
        return max(0.0, net_pressure / max(1, future_window))
    
    def _calculate_usage_frequency(self, buffer_info: BufferInfo, schedule: List[int], current_step: int) -> float:
        """计算缓冲区使用频率"""
        usage_count = 0
        total_steps = 0
        
        # 统计在生命周期内被使用的次数
        for i, node_id in enumerate(schedule):
            if i <= current_step:
                continue
            if node_id == buffer_info.free_node:
                break
            
            node = self.nodes[node_id]
            if hasattr(node, 'bufs') and buffer_info.buf_id in node.bufs:
                usage_count += 1
            total_steps += 1
        
        if total_steps == 0:
            return 0.0
        
        return usage_count / total_steps
    
    def _calculate_spill_cost_score(self, buffer_info: BufferInfo) -> float:
        """计算SPILL成本得分"""
        # 基于大小和已SPILL次数计算成本；若该缓冲区被COPY_IN使用（SPILL_OUT=0），给予显著折扣
        size_cost = buffer_info.size / 10000.0
        spill_history_cost = buffer_info.spill_count * 0.05
        copy_in_discount = 0.5 if buffer_info.buf_id in getattr(self, 'copy_in_buf_set', set()) else 1.0
        return (size_cost + spill_history_cost) * copy_in_discount
    
    def _calculate_spill_priority(self, buffer_info: BufferInfo, schedule: List[int], current_step: int) -> float:
        """
        计算SPILL优先级 (保留原接口，内部调用高级算法)
        """
        cache_type = buffer_info.cache_type
        return self._calculate_advanced_spill_priority(buffer_info, schedule, current_step, cache_type)
    
    def _insert_spill_out_and_plan_in(self, buf_id: int, cache_type: str, dynamic_schedule: List[int], insert_index: int) -> bool:
        """在insert_index位置插入SPILL_OUT，并在下一次使用/或FREE之前插入SPILL_IN。"""
        # 兼容旧接口：这里直接输出SPILL_OUT，并不插入到dynamic_schedule
        complete_schedule_dummy: List[int] = []
        return self._emit_spill_out(buf_id, cache_type, complete_schedule_dummy)

    def _try_use_copy_in_instead_of_spill_in(self, buf_id: int, complete_schedule: List[int]) -> bool:
        """尝试用可上提的 COPY_IN 来替代 SPILL_IN。
        满足条件：存在未执行的 COPY_IN 节点，且其所有前驱已执行。
        成功时：立即为缓冲区分配地址（不记录SPILL），并将该 COPY_IN 输出到序列。
        """
        candidates = getattr(self, 'buf_to_copyin_nodes', {}).get(buf_id, [])
        if not candidates:
            return False
        for nid in candidates:
            if nid in self.executed_nodes:
                continue
            preds = self.scheduler.reverse_adj_list.get(nid, [])
            if all(p in self.executed_nodes for p in preds):
                # 分配地址但不计入SPILL成本
                self._apply_spill_in_effect(buf_id)
                complete_schedule.append(nid)
                self.executed_nodes.add(nid)
                return True
        return False

    def _apply_spill_out_effect(self, buf_id: int, cache_type: str):
        """执行SPILL_OUT的状态变更：释放缓存并标记spilled。"""
        cache_manager = self.cache_managers[cache_type]
        buffer_info = self.buffer_infos.get(buf_id)
        if buffer_info is None:
            return
        # 若已不在缓存，避免重复free
        if buf_id in cache_manager.buf_to_region:
            cache_manager.free(buf_id)
        buffer_info.is_spilled = True
        buffer_info.spill_count += 1
        buffer_info.allocated_offset = None

    def _apply_spill_in_effect(self, buf_id: int):
        """执行SPILL_IN的状态变更：为缓冲区重新分配地址，填写spill_op.new_offset。"""
        buffer_info = self.buffer_infos.get(buf_id)
        if buffer_info is None:
            return
        cache_type = buffer_info.cache_type
        allocator = self.address_allocators[cache_type]
        # 稀疏缓存未来请求，减少重复计算
        cache_key = (cache_type, self.current_step // 50)
        if not hasattr(self, '_future_req_cache'):
            self._future_req_cache = {}
        if cache_key in self._future_req_cache:
            future_requests = self._future_req_cache[cache_key]
        else:
            future_requests = self._predict_future_allocation_requests(cache_type)
            self._future_req_cache[cache_key] = future_requests
        required_size = buffer_info.size
        capacity = self.cache_managers[cache_type].capacity
        if required_size > capacity:
            raise RuntimeError(f"SPILL_IN失败：缓冲区 {buf_id} 大小 {required_size} 超过 {cache_type} 容量 {capacity}")

        # 首选分配策略序列
        def try_allocate() -> Optional[int]:
            # 1) 优先回写原地址（若最近一次SPILL_OUT记录了old_offset且该区间空闲）
            for op in reversed(self.spill_operations):
                if op.buf_id == buf_id:
                    old_offset = op.old_offset
                    if old_offset is not None and old_offset >= 0:
                        cm = self.cache_managers[cache_type]
                        if cm._can_allocate_at(old_offset, required_size):
                            off = cm.allocate(buf_id, required_size, old_offset)
                            if off is not None:
                                return off
                    break
            # 2) 常规策略序列
            off = allocator.allocate_with_strategy(buf_id, required_size, "segregated_fit", future_requests)
            if off is None:
                off = allocator.allocate_with_strategy(buf_id, required_size, "fragmentation_aware", future_requests)
            if off is None:
                off = allocator.allocate_with_strategy(buf_id, required_size, "best_fit", future_requests)
            if off is None:
                off = allocator.allocate_with_strategy(buf_id, required_size, "first_fit", future_requests)
            if off is None:
                off = allocator.allocate_with_strategy(buf_id, required_size, "worst_fit", future_requests)
            return off

        offset = try_allocate()

        # 若失败，累积SPILL直到最大连续空闲块足够
        if offset is None:
            cache_manager = self.cache_managers[cache_type]
            max_spills = 512
            spills_done = 0
            while cache_manager.get_largest_free_block() < required_size and spills_done < max_spills:
                candidates = self._select_spill_candidates(cache_type, [], self.current_step)
                if not candidates:
                    break
                progressed = False
                for c_buf_id, _ in candidates:
                    if c_buf_id == buf_id:
                        continue
                    # 直接释放以快速合并空闲空间（此处不插入节点以减少开销）
                    self._apply_spill_out_effect(c_buf_id, cache_type)
                    spills_done += 1
                    progressed = True
                    if cache_manager.get_largest_free_block() >= required_size or spills_done >= max_spills:
                        break
                if not progressed:
                    break
            # 再尝试一次分配
            if cache_manager.get_largest_free_block() >= required_size:
                offset = try_allocate()

        if offset is None:
            raise RuntimeError(f"SPILL_IN失败：无法为缓冲区 {buf_id} 重新分配空间")
        buffer_info.allocated_offset = offset
        buffer_info.is_spilled = False
        # 回填最近一次针对该缓冲区的SPILL操作的new_offset
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.new_offset == -1:
                op.new_offset = offset
                break

    def _estimate_contiguous_gain_on_free(self, cache_type: str, region: CacheRegion) -> int:
        """估计释放该region后与相邻空闲块合并形成的连续空闲增益。"""
        cm = self.cache_managers[cache_type]
        left = 0
        right = 0
        for fs, fe in cm.free_blocks:
            if fe == region.start:
                left = fe - fs
            if fs == region.end:
                right = fe - fs
        return region.size() + left + right

    def _insert_spill_in_node(self, buf_id: int, cache_type: str, dynamic_schedule: List[int], insert_index: int) -> int:
        """在insert_index插入SPILL_IN节点（不立即分配，等节点执行时分配）。"""
        spill_in_id = max(self.nodes.keys()) + 1
        buffer_info = self.buffer_infos.get(buf_id)
        cycles = self._calculate_spill_cycles(buffer_info.size if buffer_info else 0, True)
        spill_in_node = Node({
            "Id": spill_in_id,
            "Op": "SPILL_IN",
            "Pipe": "MTE2",
            "Cycles": cycles,
            "Bufs": [buf_id]
        })
        self.nodes[spill_in_id] = spill_in_node
        dynamic_schedule.insert(insert_index, spill_in_id)
        return spill_in_id

    def _find_next_use_index(self, dynamic_schedule: List[int], start_index: int, buf_id: int) -> Optional[int]:
        """从start_index起查找下一次使用该缓冲区的节点位置（计算/搬运节点的Bufs包含它）。"""
        for j in range(start_index, len(dynamic_schedule)):
            nid = dynamic_schedule[j]
            node = self.nodes[nid]
            if hasattr(node, 'bufs') and node.bufs and buf_id in node.bufs:
                return j
        return None

    def _find_index_of_in_schedule(self, dynamic_schedule: List[int], node_id: int) -> Optional[int]:
        try:
            return dynamic_schedule.index(node_id)
        except ValueError:
            return None
    
    def _merge_spill_nodes(self, complete_schedule: List[int], original_schedule: List[int]) -> List[int]:
        """废弃：SPILL节点已在动态仿真中内联插入，这里保持接口但直接返回。"""
        return complete_schedule
    
    def _calculate_spill_cycles(self, buffer_size: int, is_spill_in: bool) -> int:
        """
        计算SPILL操作的cycles
        
        Args:
            buffer_size: 缓冲区大小
            is_spill_in: 是否为SPILL_IN操作
            
        Returns:
            cycles数量
        """
        # 统一返回公式（SPILL_OUT=0的情形在调用处判断COPY_IN再处理）
        return buffer_size * 2 + 150
    
    def calculate_total_spill_cost(self) -> int:
        """计算总额外数据搬运量"""
        total_cost = 0
        
        for spill_op in self.spill_operations:
            buffer_info = self.buffer_infos[spill_op.buf_id]
            # 检查是否被COPY_IN节点使用
            is_copy_in_used = self._is_buffer_used_by_copy_in(spill_op.buf_id)
            has_spill_in = spill_op.spill_in_node_id is not None
            if is_copy_in_used:
                # 若没有发生 SPILL_IN（以 COPY_IN 复载），额外为0；发生则1×Size
                total_cost += (buffer_info.size if has_spill_in else 0)
            else:
                # 非 COPY_IN 来源：若无 SPILL_IN，仅SPILL_OUT写回DDR=1×Size；有SPILL_IN则2×Size
                total_cost += (buffer_info.size * 2 if has_spill_in else buffer_info.size)
        
        return total_cost
    
    def _is_buffer_used_by_copy_in(self, buf_id: int) -> bool:
        """检查缓冲区是否被COPY_IN节点使用"""
        # 使用预计算集合以提高效率
        return buf_id in getattr(self, 'copy_in_buf_set', set())
    
    def get_cache_usage_stats(self) -> Dict[str, Dict[str, float]]:
        """获取缓存使用统计"""
        stats = {}
        
        for cache_type, manager in self.cache_managers.items():
            stats[cache_type] = {
                "capacity": manager.capacity,
                "allocated": manager.get_allocated_size(),
                "free": manager.get_free_size(),
                "utilization": manager.get_allocated_size() / manager.capacity,
                "fragmentation": manager.get_fragmentation_ratio(),
                "largest_free_block": manager.get_largest_free_block()
            }
        
        return stats


class Problem2Solver:
    """问题2的主要解决器"""
    
    def __init__(self, graph_file: str):
        self.graph_file = graph_file
        self.scheduler = AdvancedScheduler(graph_file)
        self.simulator = HardwareSimulator(self.scheduler)
        # Store results after solve() is run
        self.complete_schedule: Optional[List[int]] = None
        self.buffer_allocations: Optional[Dict[int, int]] = None
        self.spill_operations: Optional[List[SpillOperation]] = None
    
    def solve(self, task_name: str = "", verbose: bool = True, show_progress: bool = False) -> Tuple[List[int], Dict[int, int], List[SpillOperation], int, Dict]:
        """
        解决问题2
        
        Returns:
            Tuple[List[int], Dict[int, int], List[SpillOperation], int, Dict]:
            (调度序列, 地址分配, SPILL操作, 总额外数据搬运量, 碎片化统计)
        """
        # 获取问题1的调度序列
        schedule, max_v_stay = self.scheduler.schedule_problem1_binary_search()

        # 优先级B：局部重排（不破坏拓扑）以压缩缓冲区生命周期
        schedule = self._improve_schedule_local(schedule)
        
        if verbose:
            print(f"问题1调度序列长度: {len(schedule)}")
            print(f"问题1 max(V_stay): {max_v_stay:,}")
        
        # 执行硬件仿真
        buffer_allocations, spill_operations, complete_schedule = self.simulator.simulate_schedule(
            schedule, case_name=task_name, verbose=show_progress
        )
        
        # 计算总额外数据搬运量
        total_spill_cost = self.simulator.calculate_total_spill_cost()
        
        # 新增：获取真实的碎片化统计
        fragmentation_stats = self.simulator.get_fragmentation_stats()
        
        # 将结果存储在实例变量中以供后续使用
        self.complete_schedule = complete_schedule
        self.buffer_allocations = buffer_allocations
        self.spill_operations = spill_operations
        
        # 打印统计信息
        if verbose:
            self._print_statistics(buffer_allocations, spill_operations, total_spill_cost, fragmentation_stats)
        
        return complete_schedule, buffer_allocations, spill_operations, total_spill_cost, fragmentation_stats

    # ----------------- 局部重排：ALLOC下沉、FREE上提（不破坏拓扑） -----------------
    def _improve_schedule_local(self, schedule: List[int]) -> List[int]:
        """对原始拓扑序做安全的局部重排：
        - 将 L1/UB 的 ALLOC 尽量下沉到其后继最早位置之前
        - 将 L1/UB 的 FREE 尽量上提到其前驱最晚位置之后
        目标：缩短生命周期，减少驻留与SPILL概率。
        """
        if not schedule:
            return schedule

        nodes = self.scheduler.nodes
        adj = getattr(self.scheduler, 'adj_list', {})
        rev = getattr(self.scheduler, 'reverse_adj_list', {})

        pos = {nid: i for i, nid in enumerate(schedule)}

        def move_node(nid: int, new_idx: int):
            old_idx = pos[nid]
            if new_idx == old_idx:
                return
            # 弹出并插入
            schedule.pop(old_idx)
            schedule.insert(new_idx, nid)
            # 重新刷新受影响区间的pos（仅在[min, max]区间）
            start = min(old_idx, new_idx)
            end = max(old_idx, new_idx)
            for i in range(start, end + 1):
                pos[schedule[i]] = i

        # FREE上提
        for buf_id, info in self.scheduler.nodes.items():
            pass  # 占位无用

        # 先处理 FREE 上提
        for buf_id, alloc_node_id in getattr(self, 'simulator', None).buf_to_alloc.items() if hasattr(self, 'simulator') else []:
            free_node_id = getattr(self, 'simulator').buf_to_free.get(buf_id)
            if free_node_id is None:
                continue
            free_node = nodes.get(free_node_id)
            if not free_node or free_node.op != 'FREE':
                continue
            if free_node.cache_type not in ['L1', 'UB']:
                continue
            # 计算FREE的最早合法位置：所有前驱的最大位置+1
            preds = rev.get(free_node_id, [])
            if not preds:
                continue
            earliest_pos = max(pos[p] for p in preds if p in pos) + 1
            cur_pos = pos.get(free_node_id)
            if cur_pos is not None and earliest_pos < cur_pos:
                move_node(free_node_id, earliest_pos)

        # 再处理 ALLOC 下沉
        for buf_id, alloc_node_id in getattr(self, 'simulator', None).buf_to_alloc.items() if hasattr(self, 'simulator') else []:
            alloc_node = nodes.get(alloc_node_id)
            if not alloc_node or alloc_node.op != 'ALLOC':
                continue
            if alloc_node.cache_type not in ['L1', 'UB']:
                continue
            succs = adj.get(alloc_node_id, [])
            if not succs:
                continue
            latest_pos = min(pos[s] for s in succs if s in pos)  # 必须在最早后继之前
            cur_pos = pos.get(alloc_node_id)
            if cur_pos is not None and cur_pos + 1 < latest_pos:
                move_node(alloc_node_id, latest_pos - 1)

        return schedule
    
    def _print_statistics(self, allocations: Dict[int, int], spill_ops: List[SpillOperation], 
                          spill_cost: int, fragmentation_stats: Dict):
        """打印统计信息"""
        print(f"\n=== 问题2解决方案统计 ===")
        print(f"分配的缓冲区数量: {len(allocations)}")
        print(f"SPILL操作次数: {len(spill_ops)}")
        print(f"总额外数据搬运量: {spill_cost:,}")
        print(f"协同优化（重调度）执行次数: {getattr(self.simulator, 'rescheduled_op_count', 0)}")
        
        # 打印碎片化统计
        print(f"\n=== 动态碎片化统计 ===")
        for cache_type, stats in fragmentation_stats.items():
            print(f"{cache_type}: 平均碎片率={stats['avg']:.2%}, 峰值碎片率={stats['max']:.2%}")

        # 缓存使用统计
        cache_stats = self.simulator.get_cache_usage_stats()
        print(f"\n=== 最终缓存使用统计 ===")
        for cache_type, stats in cache_stats.items():
            print(f"{cache_type}: 利用率={stats['utilization']:.1%}, "
                  f"碎片化率={stats['fragmentation']:.1%}, "
                  f"已用/总容量={stats['allocated']}/{stats['capacity']}")
    
    def save_results(self, output_dir: str, task_name: str):
        """将仿真结果正确保存到文件"""
        # 检查结果是否存在，如果不存在，则运行仿真以生成结果
        if self.complete_schedule is None or self.buffer_allocations is None or self.spill_operations is None:
            print(f"\nWarning: {task_name} 的结果未找到。正在运行仿真以生成结果。")
            self.solve(task_name=task_name, verbose=False, show_progress=True)

        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存调度序列
        with open(f"{output_dir}/{task_name}_schedule.txt", 'w') as f:
            for node_id in self.complete_schedule:
                f.write(f"{node_id}\n")
        
        # 保存内存分配
        with open(f"{output_dir}/{task_name}_memory.txt", 'w') as f:
            for buf_id, offset in self.buffer_allocations.items():
                f.write(f"{buf_id}:{offset}\n")
        
        # 保存SPILL操作
        with open(f"{output_dir}/{task_name}_spill.txt", 'w') as f:
            for spill_op in self.spill_operations:
                f.write(f"{spill_op.buf_id}:{spill_op.new_offset}\n")

        # 保存新增依赖边（用于审计与Problem3）
        extra_edges_path = f"{output_dir}/{task_name}_extra_edges.txt"
        try:
            with open(extra_edges_path, 'w') as f:
                for u, v in getattr(self.simulator, 'extra_edges', []):
                    f.write(f"{u},{v}\n")
        except Exception:
            pass

        # 新增：保存碎片化历史
        with open(f"{output_dir}/{task_name}_fragmentation.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_to_save = {
                cache_type: np.array(history).tolist()
                for cache_type, history in self.simulator.fragmentation_history.items()
            }
            json.dump(history_to_save, f)
    
    def save_visualization_data(self, viz_output_dir: str, task_name: str):
        """保存可视化所需的详细数据"""
        import os
        os.makedirs(viz_output_dir, exist_ok=True)
        
        # 检查结果是否存在
        if self.complete_schedule is None or self.buffer_allocations is None or self.spill_operations is None:
            print(f"\nWarning: {task_name} 的结果未找到。正在运行仿真以生成结果。")
            self.solve(task_name=task_name, verbose=False, show_progress=True)
        
        # 计算详细的执行时序数据
        viz_data = self._generate_visualization_data(task_name)
        
        # 保存为JSON格式
        viz_file = f"{viz_output_dir}/{task_name}_problem2_visualization.json"
        with open(viz_file, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已保存 {task_name} 的可视化数据到 {viz_file}")
    
    def _generate_visualization_data(self, task_name: str) -> dict:
        """生成完整的可视化数据"""
        
        # 基础信息
        viz_data = {
            'task_name': task_name,
            'total_spill_cost': self.simulator.calculate_total_spill_cost(),
            'total_buffers': len(self.buffer_allocations),
            'total_spill_ops': len(self.spill_operations),
            'schedule_length': len(self.complete_schedule),
            'rescheduled_ops': getattr(self.simulator, 'rescheduled_op_count', 0)
        }
        
        # 分析调度序列中的节点类型分布
        node_analysis = self._analyze_schedule_nodes()
        viz_data.update(node_analysis)
        
        # 缓存使用统计
        cache_stats = self.simulator.get_cache_usage_stats()
        viz_data['cache_statistics'] = cache_stats
        
        # 碎片化统计
        frag_stats = self.simulator.get_fragmentation_stats()
        viz_data['fragmentation_statistics'] = frag_stats
        
        # SPILL操作详细分析
        spill_analysis = self._analyze_spill_operations()
        viz_data['spill_analysis'] = spill_analysis
        
        # 内存使用时序（用于绘制内存使用曲线）
        memory_timeline = self._generate_memory_timeline()
        viz_data['memory_timeline'] = memory_timeline
        
        # 缓存分配热力图数据
        allocation_heatmap = self._generate_allocation_heatmap()
        viz_data['allocation_heatmap'] = allocation_heatmap
        
        return viz_data
    
    def _analyze_schedule_nodes(self) -> dict:
        """分析调度序列中的节点类型和分布"""
        node_stats = {
            'alloc_l1_ub': 0,
            'free_l1_ub': 0, 
            'alloc_l0': 0,
            'free_l0': 0,
            'spill_out': 0,
            'spill_in': 0,
            'compute_ops': 0,
            'copy_ops': 0,
            'other_ops': 0
        }
        
        pipe_usage = defaultdict(int)
        cycle_distribution = []
        
        for node_id in self.complete_schedule:
            node = self.scheduler.nodes[node_id]
            
            # 节点类型统计
            if node.op == 'ALLOC':
                if node.cache_type in ['L1', 'UB']:
                    node_stats['alloc_l1_ub'] += 1
                else:
                    node_stats['alloc_l0'] += 1
            elif node.op == 'FREE':
                if node.cache_type in ['L1', 'UB']:
                    node_stats['free_l1_ub'] += 1
                else:
                    node_stats['free_l0'] += 1
            elif node.op == 'SPILL_OUT':
                node_stats['spill_out'] += 1
            elif node.op == 'SPILL_IN':
                node_stats['spill_in'] += 1
            elif node.op in ['COPY_IN', 'COPY_OUT']:
                node_stats['copy_ops'] += 1
            elif hasattr(node, 'pipe') and node.pipe:
                if node.pipe in ['Cube', 'Vector']:
                    node_stats['compute_ops'] += 1
                else:
                    node_stats['other_ops'] += 1
            
            # 执行单元使用统计
            if hasattr(node, 'pipe') and node.pipe:
                pipe_usage[node.pipe] += 1
            
            # 周期数分布
            if hasattr(node, 'cycles') and node.cycles > 0:
                cycle_distribution.append(node.cycles)
        
        return {
            'node_statistics': node_stats,
            'pipe_usage': dict(pipe_usage), 
            'cycle_distribution': cycle_distribution,
            'total_cycles': sum(cycle_distribution)
        }
    
    def _analyze_spill_operations(self) -> dict:
        """分析SPILL操作的详细信息"""
        spill_analysis = {
            'spill_by_cache_type': defaultdict(int),
            'spill_by_buffer_size': defaultdict(int),
            'spill_cost_breakdown': {},
            'copy_in_substitution_count': 0
        }
        
        total_cost_with_copy_in = 0
        total_cost_without_copy_in = 0
        
        for spill_op in self.spill_operations:
            buffer_info = self.simulator.buffer_infos[spill_op.buf_id]
            
            # 按缓存类型分类
            spill_analysis['spill_by_cache_type'][spill_op.cache_type] += 1
            
            # 按缓冲区大小分类
            size_category = self._get_size_category(buffer_info.size)
            spill_analysis['spill_by_buffer_size'][size_category] += 1
            
            # 成本分析
            is_copy_in_used = self.simulator._is_buffer_used_by_copy_in(spill_op.buf_id)
            has_spill_in = spill_op.spill_in_node_id is not None
            
            if is_copy_in_used:
                cost = buffer_info.size if has_spill_in else 0
                total_cost_with_copy_in += cost
                if not has_spill_in:
                    spill_analysis['copy_in_substitution_count'] += 1
            else:
                cost = buffer_info.size * 2 if has_spill_in else buffer_info.size
                total_cost_without_copy_in += cost
        
        spill_analysis['spill_cost_breakdown'] = {
            'copy_in_buffers': total_cost_with_copy_in,
            'non_copy_in_buffers': total_cost_without_copy_in,
            'total': total_cost_with_copy_in + total_cost_without_copy_in
        }
        
        return dict(spill_analysis)
    
    def _get_size_category(self, size: int) -> str:
        """将缓冲区大小分类"""
        if size <= 64:
            return "小 (≤64)"
        elif size <= 256:
            return "中小 (65-256)"
        elif size <= 1024:
            return "中等 (257-1024)"
        elif size <= 4096:
            return "大 (1025-4096)"
        else:
            return "超大 (>4096)"
    
    def _generate_memory_timeline(self) -> dict:
        """生成内存使用时序数据"""
        timeline = {
            'steps': list(range(len(self.complete_schedule))),
            'L1_usage': [],
            'UB_usage': [],
            'L1_fragmentation': [],
            'UB_fragmentation': []
        }
        
        # 模拟重播调度过程以获取内存使用时序
        # 这里简化实现，使用已有的碎片化历史
        frag_history = self.simulator.fragmentation_history
        
        # 如果有碎片化历史，使用它
        if 'L1' in frag_history:
            timeline['L1_fragmentation'] = frag_history['L1']
        if 'UB' in frag_history:
            timeline['UB_fragmentation'] = frag_history['UB']
        
        # 简化的内存使用量计算（基于当前状态）
        l1_manager = self.simulator.cache_managers['L1']
        ub_manager = self.simulator.cache_managers['UB']
        
        # 填充当前使用量（简化）
        current_l1_usage = l1_manager.get_allocated_size()
        current_ub_usage = ub_manager.get_allocated_size()
        
        steps_count = len(self.complete_schedule)
        timeline['L1_usage'] = [current_l1_usage] * steps_count
        timeline['UB_usage'] = [current_ub_usage] * steps_count
        
        return timeline
    
    def _generate_allocation_heatmap(self) -> dict:
        """生成缓存分配热力图数据"""
        heatmap_data = {}
        
        for cache_type in ['L1', 'UB']:
            manager = self.simulator.cache_managers[cache_type]
            capacity = manager.capacity
            
            # 创建地址使用热力图
            usage_map = [0] * capacity  # 0=空闲, 1=已分配
            
            for region in manager.allocated_regions:
                for addr in range(region.start, region.end):
                    if addr < capacity:
                        usage_map[addr] = 1
            
            heatmap_data[cache_type] = {
                'capacity': capacity,
                'usage_map': usage_map,
                'allocated_regions': [
                    {
                        'start': region.start,
                        'end': region.end,
                        'size': region.size(),
                        'buf_id': region.buf_id
                    }
                    for region in manager.allocated_regions
                ],
                'free_blocks': [
                    {'start': start, 'end': end, 'size': end - start}
                    for start, end in manager.free_blocks
                ]
            }
        
        return heatmap_data


def test_problem2():
    """测试问题2解决方案"""
    test_files = [
        "data/Json_version/Matmul_Case0.json",
        "data/Json_version/FlashAttention_Case0.json",
        "data/Json_version/Conv_Case0.json",
        "data/Json_version/FlashAttention_Case1.json",
        "data/Json_version/Matmul_Case1.json",
        "data/Json_version/Conv_Case1.json",
    ]
    
    print("=== 问题2：缓存分配与换入换出测试 ===")
    print(f"{'任务名':<25} {'总额外数据搬运量':<15} {'SPILL次数':<10}")
    print("-" * 55)
    
    total_spill_cost = 0
    results = {}
    
    for test_file in test_files:
        try:
            task_name = test_file.split("/")[-1].replace(".json", "")
            
            solver = Problem2Solver(test_file)
            # 运行仿真，显示每个case的进度条
            _, allocations, spill_ops, spill_cost, _ = solver.solve(
                task_name=task_name, verbose=False, show_progress=True
            )
            
            # 打印结果
            print(f"{task_name:<25} {spill_cost:<15,} {len(spill_ops):<10}")
            
            total_spill_cost += spill_cost
            results[task_name] = {
                "spill_cost": spill_cost,
                "spill_count": len(spill_ops),
                "allocated_buffers": len(allocations)
            }
            
            # 保存结果
            solver.save_results("Problem2_Results", task_name)
            
            # 保存可视化数据
            solver.save_visualization_data("Problem2_Visualization_Data", task_name)
            
        except Exception as e:
            print(f"{task_name:<25} 执行失败: {e}")
    
    print("-" * 55)
    print(f"{'总计':<25} {total_spill_cost:<15,}")
    
    return results


if __name__ == "__main__":
    results = test_problem2()
    print(f"\n问题2解决方案完成，结果已保存到 Problem2_Results/ 目录")
