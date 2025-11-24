"""
问题2：缓存分配与换入换出
基于问题1的调度序列，实现硬件状态仿真框架，管理多级缓存并执行SPILL操作
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import json
from typing import Dict, List, Set, Tuple, Optional, NamedTuple, Any
from collections import defaultdict
import heapq
import numpy as np
import numba
from numba import jit, types
from numba.typed import Dict, List as NumbaList
from tqdm import tqdm
from advanced_scheduler import AdvancedScheduler, Node
from hardware_simulator import Problem2Solver as P2BaselineSolver


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

    def free_and_get_region(self, buf_id: int) -> Optional[CacheRegion]:
        """释放缓冲区并返回被释放的区域信息（用于建立复用依赖）。"""
        if buf_id not in self.buf_to_region:
            return None
        region = self.buf_to_region[buf_id]
        # 从已分配集合移除
        self.allocated_regions.remove(region)
        del self.buf_to_region[buf_id]
        # 插回空闲块并合并相邻
        self._insert_free_block(region.start, region.end)
        return region

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
    
    def __init__(self, scheduler: AdvancedScheduler,
                 enable_preemptive_spill: bool = True,
                 enforce_pipe_order_in_runtime: bool = True,
                 enforce_l0_single_live: bool = True,
                 enable_cycle_guard: bool = False):
        self.scheduler = scheduler
        self.nodes = scheduler.nodes
        
        # 缓存管理器
        self.cache_managers: Dict[str, CacheManager] = {}
        self.address_allocators: Dict[str, SmartAddressAllocator] = {}
        for cache_type, capacity in self.CACHE_CAPACITIES.items():
            manager = CacheManager(cache_type, capacity)
            self.cache_managers[cache_type] = manager
            self.address_allocators[cache_type] = SmartAddressAllocator(manager)

        # 配置开关（用于问题三放宽不必要的硬约束）
        self.enable_preemptive_spill: bool = enable_preemptive_spill
        self.enforce_pipe_order_in_runtime: bool = enforce_pipe_order_in_runtime
        self.enforce_l0_single_live: bool = enforce_l0_single_live
        self.enable_cycle_guard: bool = enable_cycle_guard
        
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
        # 阶段D：联合优化参数（可调）
        self.lambda_ddr: float = 0.05  # DDR额外搬运量的权重
        
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
        # 去重集合，避免重复边
        self._extra_edges_set: Set[Tuple[int, int]] = set()
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
        # 释放区间追踪：cache_type -> [(start, end, release_event_node_id)]
        self.released_intervals: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        
        # 重置所有缓存管理器
        for cache_manager in self.cache_managers.values():
            cache_manager.allocated_regions = []
            cache_manager.buf_to_region = {}

    def _record_step_stats(self):
        """记录当前步骤的统计数据，如碎片率"""
        for cache_type, manager in self.cache_managers.items():
            if cache_type in ["L1", "UB"]:
                self.fragmentation_history[cache_type].append(manager.get_fragmentation_ratio())

    def _add_extra_edge(self, u: int, v: int):
        """添加额外依赖边（去重）。"""
        if u == v:
            return
        # 环路守卫：避免新边造成环
        if getattr(self, 'enable_cycle_guard', False):
            try:
                if self._would_create_cycle(u, v):
                    return
            except Exception:
                # 容错：若检查失败，不阻断流程
                pass
        key = (u, v)
        if key in self._extra_edges_set:
            return
        self._extra_edges_set.add(key)
        self.extra_edges.append(key)

    def _would_create_cycle(self, u: int, v: int) -> bool:
        """判断加入边 u->v 是否会引入环（若 v 可达 u，则成环）。"""
        # 组合原始依赖与已添加的额外依赖
        base_adj = getattr(self.scheduler, 'adj_list', {}) or {}
        extra = getattr(self, 'extra_edges', [])
        # 从 v 出发做DFS查找 u
        target = u
        stack = [v]
        visited: Set[int] = set()
        while stack:
            x = stack.pop()
            if x == target:
                return True
            if x in visited:
                continue
            visited.add(x)
            # 后继：原始 + 额外
            for nx in base_adj.get(x, []):
                if nx not in visited:
                    stack.append(nx)
            for (a, b) in extra:
                if a == x and b not in visited:
                    stack.append(b)
        return False

    def _record_release_interval(self, cache_type: str, start: int, end: int, release_event_id: int):
        """记录一次释放产生的空闲区间及其释放事件节点。"""
        if cache_type not in ("L1", "UB"):
            return
        if start >= end:
            return
        self.released_intervals[cache_type].append((start, end, release_event_id))

    def _attach_reuse_dependency(self, cache_type: str, start: int, size: int, alloc_event_node_id: int):
        """当在某空闲区间上重新分配时，建立由释放事件到分配事件的依赖，并消费相应区间。

        - 若新分配区间与多个释放区间有重叠，则为每个重叠的释放事件添加依赖。
        - 消费（移除/切分）已被使用的释放区间片段，避免重复依赖。
        """
        if cache_type not in ("L1", "UB"):
            return
        if size <= 0:
            return
        start_new = start
        end_new = start + size
        intervals = self.released_intervals.get(cache_type, [])
        if not intervals:
            return
        new_list: List[Tuple[int, int, int]] = []
        for (s, e, rid) in intervals:
            # 计算重叠
            overlap_start = max(s, start_new)
            overlap_end = min(e, end_new)
            if overlap_start < overlap_end:
                # 有重叠：建立依赖 rid -> alloc_event
                self._add_extra_edge(rid, alloc_event_node_id)
                # 剩余区间切分（左段）
                if s < overlap_start:
                    new_list.append((s, overlap_start, rid))
                # 右段
                if overlap_end < e:
                    new_list.append((overlap_end, e, rid))
            else:
                # 无重叠，保留原区间
                new_list.append((s, e, rid))
        self.released_intervals[cache_type] = new_list

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
    
    def simulate_schedule(self, schedule: List[int], case_name: str = "", show_progress: bool = False) -> Tuple[Dict[int, int], List[SpillOperation], List[int]]:
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
        pressure_points = self._batch_predict_pressure_points(schedule) if self.enable_preemptive_spill else set()

        # 为每个case添加进度条
        progress_bar = tqdm(total=len(schedule), desc=f"Simulating {case_name}", leave=False, position=1) if (show_progress and case_name) else None

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
            if self.enable_preemptive_spill and (i in pressure_points):
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
                    # 地址复用依赖（广义，基于释放区间重叠）
                    self._attach_reuse_dependency(
                        buffer_info.cache_type,
                        buffer_info.allocated_offset,
                        buffer_info.size,
                        node.id,
                    )
                    # 兼容：精确区域复用的依赖（原有逻辑，去重后安全）
                    region_key = (buffer_info.cache_type, buffer_info.allocated_offset, buffer_info.size)
                    if region_key in self.last_region_owner:
                        prev_buf = self.last_region_owner[region_key]
                        prev_free = self.buf_to_free.get(prev_buf)
                        if prev_free is not None:
                            self._add_extra_edge(prev_free, node.id)

                # 输出原节点
                complete_schedule.append(node_id)
                self.executed_nodes.add(node_id)
                if progress_bar:
                    progress_bar.update(1)
                continue

            # FREE：若处于spilled则先SPILL_IN
            if node.op == "FREE" and node.is_l1_or_ub_cache():
                buf_id = node.buf_id
                buffer_info = self.buffer_infos.get(buf_id)
                if buffer_info and buffer_info.is_spilled and buffer_info.allocated_offset is None:
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
                        # 建立依赖：已执行使用者 -> SPILL_OUT，SPILL_IN -> 未执行使用者（流式语义：在此点前插入SPILL_IN）
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
        
        # 步骤1：在定义的"向前看"窗口内，寻找所有符合条件的FREE节点作为候选
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
            # 阶段D：更精细的选择（若未来窗口中有即将到来的较大ALLOC，可按 required_size 选）
            required_hint = 0
            for nid in range(self.current_step, min(self.current_step + 50, len(self.schedule_cache.get("ops", [])))):
                if self.schedule_cache["ops"][nid] == 0 and self.schedule_cache["cache_types"][nid] == self.cache_type_mapping.get(cache_type, -1):
                    required_hint = max(required_hint, int(self.schedule_cache["sizes"][nid]))
            if required_hint <= 0:
                # 回退：使用原候选
                for buf_id in candidates[:1]:
                    self._emit_spill_out(buf_id, cache_type, complete_schedule)
            else:
                scored = self._select_spill_candidates_for_alloc(cache_type, required_hint, [], self.current_step)
                for buf_id, _ in scored[:1]:
                    self._emit_spill_out(buf_id, cache_type, complete_schedule)

    def _handle_spill_for_alloc_stream(self, alloc_node: Node, complete_schedule: List[int]) -> bool:
        """为ALLOC操作处理SPILL（流式）：输出若干SPILL_OUT直到可分配，然后分配。"""
        cache_manager = self.cache_managers[alloc_node.cache_type]
        allocator = self.address_allocators[alloc_node.cache_type]

        # 阶段D：基于所需大小 required_size 的联合评分挑选候选
        required_size = alloc_node.size
        candidates = self._select_spill_candidates_for_alloc(alloc_node.cache_type, required_size, [], self.current_step)
        if not candidates:
            return False

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
        # 状态生效（记录释放区间）
        self._apply_spill_out_effect(buf_id, cache_type, spill_out_id)
        # 记录依赖：ALLOC(buf) -> SPILL_OUT
        alloc_node = self.buf_to_alloc.get(buf_id)
        if alloc_node is not None:
            self._add_extra_edge(alloc_node, spill_out_id)
        # 记录操作
        spill_op = SpillOperation(buf_id, old_offset, -1, cache_type)
        spill_op.spill_out_node_id = spill_out_id
        self.spill_operations.append(spill_op)
        # 记录最近释放的区域归属用于地址复用依赖
        self.last_region_owner[(cache_type, old_offset, buffer_info.size)] = buf_id
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
        # 分配并回填（可触发显式SPILL_OUT）
        self._apply_spill_in_effect(buf_id, alloc_event_node_id=spill_in_id, complete_schedule=complete_schedule)
        # 依赖：SPILL_OUT -> SPILL_IN
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.spill_in_node_id is None:
                self._add_extra_edge(op.spill_out_node_id, spill_in_id)
                break
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.spill_in_node_id is None:
                op.spill_in_node_id = spill_in_id
                break
        # 依赖：SPILL_IN → 所有未执行的消费者
        users = getattr(self, 'buf_to_users', {}).get(buf_id, [])
        for u in users:
            if u not in self.executed_nodes:
                self._add_extra_edge(spill_in_id, u)
        # 依赖：SPILL_IN → FREE（补齐题面约束）
        free_node_id = self.buf_to_free.get(buf_id)
        if free_node_id is not None:
            self._add_extra_edge(spill_in_id, free_node_id)
        return spill_in_id
    
    def _handle_alloc(self, node: Node) -> bool:
        """处理ALLOC操作"""
        cache_manager = self.cache_managers[node.cache_type]
        allocator = self.address_allocators[node.cache_type]
        buffer_info = self.buffer_infos.get(node.buf_id)
        
        if buffer_info is None:
            # L0级：依据开关决定是否严格单驻留；默认问题2保持严格，问题3放宽为软约束
            if self.enforce_l0_single_live and (not self._can_alloc_l0(node.cache_type)):
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
        # 获取释放的区间并记录释放事件
        region = cache_manager.free_and_get_region(node.buf_id)
        if region:
            self._record_release_interval(node.cache_type, region.start, region.end, node.id)
            return True
        return False

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
            # 引入单位释放空间的成本比作为次排序键，鼓励释放大块
            normalized_cost = (buffer_info.size / max(1, region.size()))
            candidates.append((buf_id, priority + 0.001 * normalized_cost))
        
        # 按优先级排序
        candidates.sort(key=lambda x: x[1])
        return candidates

    def _get_cache_capacity(self, cache_type: str) -> int:
        return self.CACHE_CAPACITIES.get(cache_type, 1)

    def _estimate_ddr_cost_bytes(self, buffer_info: BufferInfo) -> int:
        """按题意评估一次 SPILL 的额外DDR搬运量（字节/单位）。"""
        if buffer_info.buf_id in getattr(self, 'copy_in_buf_set', set()):
            return buffer_info.size  # 仅SPILL_IN
        return buffer_info.size * 2  # SPILL_OUT + SPILL_IN

    def _largest_free_block_if_free(self, cache_type: str, buf_id: int) -> int:
        """假设释放此 buf_id 后，估计最大的连续空闲块大小（不修改真实状态）。"""
        cm = self.cache_managers[cache_type]
        if buf_id not in cm.buf_to_region:
            return cm.get_largest_free_block()
        region = cm.buf_to_region[buf_id]
        lfb_now = cm.get_largest_free_block()
        # 估算与 free_blocks 的相邻合并
        new_block_size = region.size()
        left_size = 0
        right_size = 0
        for (fs, fe) in cm.free_blocks:
            if fe == region.start:
                left_size = fe - fs
            if fs == region.end:
                right_size = fe - fs
        new_block_size += left_size + right_size
        return max(lfb_now, new_block_size)

    def _combined_spill_score_for_alloc(self, buffer_info: BufferInfo, cache_type: str, required_size: int, schedule: List[int], current_step: int) -> float:
        """用于分配失败时的联合评分：越小越优先。
        包含：高级优先级 + λ·DDR成本/容量 - 益处（释放后最大块/需求）。"""
        base = self._calculate_advanced_spill_priority(buffer_info, schedule, current_step, cache_type)
        cap = self._get_cache_capacity(cache_type)
        ddr_cost = self._estimate_ddr_cost_bytes(buffer_info) / max(1, cap)
        lfb_after = self._largest_free_block_if_free(cache_type, buffer_info.buf_id)
        benefit = min(1.0, lfb_after / max(1, required_size))  # 达到阈值则≈1
        # 负向奖励鼓励一次到位
        return base + self.lambda_ddr * ddr_cost - 0.25 * benefit

    def _select_spill_candidates_for_alloc(self, cache_type: str, required_size: int, schedule: List[int], current_step: int) -> List[Tuple[int, float]]:
        """按联合评分为分配失败场景挑选候选者。"""
        cm = self.cache_managers[cache_type]
        items = list(cm.buf_to_region.items())
        items.sort(key=lambda kv: kv[1].size(), reverse=True)
        top_k = 64 if len(items) > 64 else len(items)
        scored: List[Tuple[int, float]] = []
        for buf_id, _ in items[:top_k]:
            bi = self.buffer_infos.get(buf_id)
            if not bi:
                continue
            s = self._combined_spill_score_for_alloc(bi, cache_type, required_size, schedule, current_step)
            scored.append((buf_id, s))
        scored.sort(key=lambda x: x[1])
        return scored
    
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

    def _apply_spill_out_effect(self, buf_id: int, cache_type: str, spill_out_node_id: int):
        """执行SPILL_OUT的状态变更：释放缓存并标记spilled，同时记录释放区间。"""
        cache_manager = self.cache_managers[cache_type]
        buffer_info = self.buffer_infos.get(buf_id)
        if buffer_info is None:
            return
        # 若已不在缓存，避免重复free
        if buf_id in cache_manager.buf_to_region:
            region = cache_manager.free_and_get_region(buf_id)
            if region:
                self._record_release_interval(cache_type, region.start, region.end, spill_out_node_id)
        buffer_info.is_spilled = True
        buffer_info.spill_count += 1
        buffer_info.allocated_offset = None

    def _apply_spill_in_effect(self, buf_id: int, alloc_event_node_id: Optional[int] = None, complete_schedule: Optional[List[int]] = None):
        """执行SPILL_IN的状态变更：为缓冲区重新分配地址，填写spill_op.new_offset，并建立复用依赖。

        若空间不足，通过显式插入 SPILL_OUT 节点释放其他缓冲区（不再静默释放）。
        """
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
            # 通过显式 SPILL_OUT 释放空间（带依赖与代价）
            while cache_manager.get_largest_free_block() < required_size and spills_done < max_spills:
                candidates = self._select_spill_candidates(cache_type, [], self.current_step)
                if not candidates:
                    break
                progressed = False
                for c_buf_id, _ in candidates:
                    if c_buf_id == buf_id:
                        continue
                    if complete_schedule is not None:
                        self._emit_spill_out(c_buf_id, cache_type, complete_schedule)
                    else:
                        # 兜底：仍然释放，但尽量不走这条路径
                        tmp_schedule: List[int] = []
                        self._emit_spill_out(c_buf_id, cache_type, tmp_schedule)
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
        # 建立地址复用依赖（基于释放区间重叠）
        if alloc_event_node_id is not None:
            self._attach_reuse_dependency(cache_type, offset, required_size, alloc_event_node_id)
        # 回填最近一次针对该缓冲区的SPILL操作的new_offset
        for op in reversed(self.spill_operations):
            if op.buf_id == buf_id and op.new_offset == -1:
                op.new_offset = offset
                break

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
            
            if is_copy_in_used:
                # 情况2：仅SPILL_IN产生数据搬运
                total_cost += buffer_info.size
            else:
                # 情况1：SPILL_OUT和SPILL_IN都产生数据搬运
                total_cost += buffer_info.size * 2
        
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


    # ===================== 问题3：基准总运行时间计算 =====================
    def build_dependency_without_pipe(self, schedule: List[int]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """构建不包含同Pipe顺序约束的扩展依赖图（用于可重排的列表调度）。

        - 原始依赖（scheduler.adj_list）
        - 额外依赖（self.extra_edges：SPILL与地址复用）
        不加入按Pipe顺序的边，以便后续对Pipe内顺序进行优化重排。
        """
        node_set = set(schedule)
        succ_map: Dict[int, List[int]] = defaultdict(list)
        pred_map: Dict[int, List[int]] = defaultdict(list)

        def _add_edge(u: int, v: int):
            if u == v:
                return
            if u not in node_set or v not in node_set:
                return
            succ_map[u].append(v)
            pred_map[v].append(u)

        # 原始依赖
        base_adj = getattr(self.scheduler, 'adj_list', {}) or {}
        for u, vs in base_adj.items():
            if u not in node_set:
                continue
            for v in vs:
                _add_edge(u, v)

        # 额外依赖
        for (u, v) in getattr(self, 'extra_edges', []):
            _add_edge(u, v)

        # 确保每个节点存在映射
        for nid in schedule:
            succ_map.setdefault(nid, [])
            pred_map.setdefault(nid, [])

        return succ_map, pred_map

    def _build_extended_dependency_graph(self, schedule: List[int]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """构建扩展依赖图（问题3基线用）：
        - 原始依赖（来自 scheduler.adj_list）
        - 额外依赖（SPILL与地址复用依赖 self.extra_edges）
        - 同一执行单元（Pipe）内严格按调度序列的顺序串行依赖

        Returns:
            (succ_map, pred_map)
        """
        node_set = set(schedule)

        succ_map: Dict[int, List[int]] = defaultdict(list)
        pred_map: Dict[int, List[int]] = defaultdict(list)

        def _add_edge(u: int, v: int):
            if u == v:
                return
            if u not in node_set or v not in node_set:
                return
            succ_map[u].append(v)
            pred_map[v].append(u)

        # 1) 原始依赖
        base_adj = getattr(self.scheduler, 'adj_list', {}) or {}
        for u, vs in base_adj.items():
            if u not in node_set:
                continue
            for v in vs:
                _add_edge(u, v)

        # 2) 额外依赖（SPILL、地址复用）
        for (u, v) in getattr(self, 'extra_edges', []):
            _add_edge(u, v)

        # 3) 同Pipe顺序约束：按 schedule 中出现顺序为每个 Pipe 串起链
        last_by_pipe: Dict[str, int] = {}
        for nid in schedule:
            node = self.nodes.get(nid)
            if not node:
                continue
            pipe = getattr(node, 'pipe', None)
            # 仅对有执行单元的节点添加顺序依赖；ALLOC/FREE等管理节点无Pipe
            if pipe:
                if pipe in last_by_pipe:
                    _add_edge(last_by_pipe[pipe], nid)
                last_by_pipe[pipe] = nid

        # 确保所有节点都有条目
        for nid in schedule:
            succ_map.setdefault(nid, [])
            pred_map.setdefault(nid, [])

        return succ_map, pred_map

    def compute_baseline_runtime(self, schedule: List[int]) -> Tuple[int, Dict[int, Tuple[int, int]]]:
        """计算基准总运行时间（问题3基线）。

        流程：
        1) 基于当前仿真产生的 complete_schedule 构建扩展依赖图
        2) 使用拓扑驱动的最早开始排程，按多执行单元独占约束计算各节点 S/E
        3) 返回 (makespan, per_node_times)

        注意：
        - ALLOC/FREE 等缓存管理节点 cycles=0，且无执行单元约束
        - SPILL 节点的 cycles 已在生成时写入
        """
        if not schedule:
            return 0, {}

        # 依赖图：为满足硬件约束（同一Pipe严格按给定顺序执行），这里始终加入Pipe顺序边
        succ_map, pred_map = self._build_extended_dependency_graph(schedule)

        # 初始化每个执行单元的可用时间
        pipe_available_time: Dict[str, int] = defaultdict(int)

        # 预处理：调度序索引用于就绪集的稳定选择（保持与既有顺序一致）
        order_index = {nid: idx for idx, nid in enumerate(schedule)}

        # 入度
        indeg: Dict[int, int] = {nid: len(pred_map.get(nid, [])) for nid in schedule}

        # 初始就绪集合（小根堆按 schedule 次序）
        ready: List[Tuple[int, int]] = []  # (order_idx, node_id)
        for nid in schedule:
            if indeg.get(nid, 0) == 0:
                heapq.heappush(ready, (order_index[nid], nid))

        start_end: Dict[int, Tuple[int, int]] = {}
        processed_count = 0

        while ready:
            _, nid = heapq.heappop(ready)
            node = self.nodes.get(nid)
            # 计算依赖就绪时间
            preds = pred_map.get(nid, [])
            dep_ready = 0
            for u in preds:
                if u in start_end:
                    dep_ready = max(dep_ready, start_end[u][1])
                else:
                    # 若前驱尚未计算，视为0（理论上不应出现，除非图含环）
                    dep_ready = max(dep_ready, 0)

            # 取资源可用时间
            pipe = getattr(node, 'pipe', None)
            cycles = getattr(node, 'cycles', 0) or 0

            if pipe:
                start_t = max(dep_ready, pipe_available_time[pipe])
                end_t = start_t + cycles
                pipe_available_time[pipe] = end_t
            else:
                # 管理节点：无资源独占，仅依赖约束
                start_t = dep_ready
                end_t = start_t + cycles

            start_end[nid] = (start_t, end_t)
            processed_count += 1

            # 推进后继
            for v in succ_map.get(nid, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(ready, (order_index.get(v, 10**12), v))

        # 环检测：若未处理完所有节点，退化为按原顺序的保守排程
        if processed_count < len(schedule):
            pipe_available_time.clear()
            start_end.clear()
            for nid in schedule:
                node = self.nodes.get(nid)
                dep_ready = 0
                for u in pred_map.get(nid, []):
                    if u in start_end:
                        dep_ready = max(dep_ready, start_end[u][1])
                pipe = getattr(node, 'pipe', None)
                cycles = getattr(node, 'cycles', 0) or 0
                if pipe:
                    s = max(dep_ready, pipe_available_time[pipe])
                    e = s + cycles
                    pipe_available_time[pipe] = e
                else:
                    s = dep_ready
                    e = s + cycles
                start_end[nid] = (s, e)

        makespan = 0
        for _, end_t in start_end.values():
            if end_t > makespan:
                makespan = end_t

        return makespan, start_end

    def compute_runtime_fast(self, schedule: List[int], pred_map: Optional[Dict[int, List[int]]] = None,
                              cycles_map: Optional[Dict[int, int]] = None,
                              pipe_map: Optional[Dict[int, Optional[str]]] = None) -> Tuple[int, Dict[int, Tuple[int, int]]]:
        """快速评估给定拓扑序 schedule 的总运行时间（O(|V|+|E|) 顺序扫描）。

        与 compute_baseline_runtime 的区别：
        - 不构造"同Pipe顺序边"，直接按给定顺序扫描，利用每Pipe的可用时间实现资源独占。
        - 依赖仅使用 pred_map（原始依赖+extra_edges），假设 schedule 已满足拓扑合法。
        适合在局部优化内频繁调用。
        """
        if not schedule:
            return 0, {}

        if pred_map is None:
            _, pred_map = self.build_dependency_without_pipe(schedule)

        if cycles_map is None or pipe_map is None:
            cycles_map = {} if cycles_map is None else cycles_map
            pipe_map = {} if pipe_map is None else pipe_map
            for nid in schedule:
                node = self.nodes.get(nid)
                if nid not in cycles_map:
                    cycles_map[nid] = getattr(node, 'cycles', 0) or 0
                if nid not in pipe_map:
                    pipe_map[nid] = getattr(node, 'pipe', None)

        pipe_available_time: Dict[str, int] = defaultdict(int)
        start_end: Dict[int, Tuple[int, int]] = {}
        makespan = 0

        for nid in schedule:
            preds = pred_map.get(nid, [])
            dep_ready = 0
            for u in preds:
                # 依赖保证在 schedule 中位于前面
                if u in start_end:
                    dep_ready = max(dep_ready, start_end[u][1])
            pipe = pipe_map.get(nid)
            cycles = cycles_map.get(nid, 0)
            if pipe:
                s = max(dep_ready, pipe_available_time[pipe])
                e = s + cycles
                pipe_available_time[pipe] = e
            else:
                s = dep_ready
                e = s + cycles
            start_end[nid] = (s, e)
            if e > makespan:
                makespan = e

        return makespan, start_end

    # ===================== 问题3：关键路径驱动的列表调度 =====================
    def _compute_bottom_level(self, succ_map: Dict[int, List[int]], cycles_map: Dict[int, int]) -> Dict[int, int]:
        """计算每个节点的 bottom-level（到汇点的最长余时）。"""
        # 拓扑逆序遍历：先算出逆拓扑序
        indeg = {u: 0 for u in succ_map}
        for u, vs in succ_map.items():
            for v in vs:
                indeg[v] = indeg.get(v, 0) + 1
        # 计算拓扑序
        q = [u for u, d in indeg.items() if d == 0]
        topo = []
        while q:
            u = q.pop()
            topo.append(u)
            for v in succ_map.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        # 逆序求解 bottom-level
        bl = {u: cycles_map.get(u, 0) for u in succ_map}
        for u in reversed(topo):
            mx = 0
            for v in succ_map.get(u, []):
                mx = max(mx, bl.get(v, 0))
            bl[u] = cycles_map.get(u, 0) + mx
        return bl

    def build_time_optimal_schedule(self, base_schedule: List[int], weights: Tuple[float, float, float, float, float] = (1.0, 0.2, 0.1, 0.1, 1.0), show_progress: bool = False, max_ready: int = 8192, fast_mode: bool = False, per_pipe_cap: int = 512) -> List[int]:
        """事件驱动、资源感知的关键路径列表调度（T1）。

        - 基于 build_dependency_without_pipe 构建拓扑依赖（不加入同Pipe顺序边）
        - 使用 bottom-level 作为主优先级
        - 对就绪节点按"最早可开工时间 est = max(依赖完成时间, pipe 可用时间)"最小优先
        - 通过精确的 pipe 可用时间推进，生成时间友好顺序
        """
        if not base_schedule:
            return []

        # 依赖图（不含Pipe顺序约束）
        succ_map, pred_map = self.build_dependency_without_pipe(base_schedule)

        # 节点属性映射
        cycles_map: Dict[int, int] = {}
        pipe_map: Dict[int, Optional[str]] = {}
        order_index = {nid: idx for idx, nid in enumerate(base_schedule)}  # 稳定性
        for nid in base_schedule:
            node = self.nodes.get(nid)
            cycles_map[nid] = getattr(node, 'cycles', 0) or 0
            pipe_map[nid] = getattr(node, 'pipe', None)

        # 轻量级预计算（减少属性访问与分支开销）
        op_code_map: Dict[int, int] = {}
        cache_idx_map: Dict[int, int] = {}
        size_map: Dict[int, int] = {}
        for nid in base_schedule:
            node = self.nodes.get(nid)
            op = getattr(node, 'op', None)
            if op == 'ALLOC':
                op_code_map[nid] = 0
            elif op == 'FREE':
                op_code_map[nid] = 1
            else:
                op_code_map[nid] = -1
            ct = getattr(node, 'cache_type', None)
            cache_idx_map[nid] = 0 if ct == 'L1' else (1 if ct == 'UB' else -1)
            size_map[nid] = getattr(node, 'size', 0) or 0

        # 关键路径余时（bottom-level）
        bl = self._compute_bottom_level(succ_map, cycles_map)

        # 入度与依赖完成时间
        indeg: Dict[int, int] = {nid: len(pred_map.get(nid, [])) for nid in base_schedule}
        dep_ready_time: Dict[int, int] = defaultdict(int)  # 每个节点的依赖完成时间

        # 每个执行单元的可用时间（资源独占）
        pipe_avail: Dict[str, int] = defaultdict(int)

        # 就绪集合：使用小根堆，按 (est, mem_penalty, -bl, base_order, nid) 选择
        import heapq
        ready_heap: List[Tuple[int, float, int, int, int]] = []
        last_key_est: Dict[int, int] = {}  # 懒更新：记录上次入堆的est
        last_key_mem: Dict[int, float] = {}

        # fast_mode：按Pipe限流，控制就绪堆规模，避免中期雪崩
        per_pipe_ready_count: Dict[str, int] = defaultdict(int)
        deferred_by_pipe: Dict[str, List[Tuple[int, float, int, int, int]]] = defaultdict(list)

        # 软约束：内存压力权重（阶段C）
        mu_usage = 0.05  # 当前占用的权重
        mu_size = 0.10   # 本次增量的权重
        mu_over = 1.00   # 超容量的强惩罚
        nu_free = 0.05   # 提前FREE的奖励（负向成本）

        # 当前序列已放置的"顺序意义上的"驻留（近似问题1的V_stay逻辑）
        mem_usage: Dict[str, int] = {"L1": 0, "UB": 0}
        capacities = {k: v for k, v in self.CACHE_CAPACITIES.items() if k in ("L1", "UB")}

        def compute_est(nid: int) -> int:
            pipe = pipe_map.get(nid)
            if pipe:
                return max(dep_ready_time.get(nid, 0), pipe_avail.get(pipe, 0))
            else:
                # 管理节点无资源独占
                return dep_ready_time.get(nid, 0)

        def compute_mem_penalty(nid: int) -> float:
            op_code = op_code_map.get(nid, -1)
            if op_code == -1:
                return 0.0
            ct_idx = cache_idx_map.get(nid, -1)
            if ct_idx == -1:
                return 0.0
            ct_name = 'L1' if ct_idx == 0 else 'UB'
            cap = capacities.get(ct_name, 1)
            cur = mem_usage.get(ct_name, 0)
            size = size_map.get(nid, 0)
            if op_code == 0:  # ALLOC
                new_cur = cur + size
                over = max(0, new_cur - cap)
                return mu_usage * (cur / cap) + mu_size * (size / cap) + mu_over * (over / cap)
            elif op_code == 1:  # FREE
                return -nu_free * (size / cap)
            return 0.0

        def push_ready(nid: int):
            est = compute_est(nid)
            # fast_mode 默认移除内存罚项，降低排序扰动
            mp = 0.0 if fast_mode else compute_mem_penalty(nid)
            last_key_est[nid] = est
            last_key_mem[nid] = mp
            if fast_mode:
                p = pipe_map.get(nid)
                cap = per_pipe_cap if p else max(64, per_pipe_cap // 2)
                if per_pipe_ready_count[p] >= cap:
                    # 超出该Pipe容量，暂存到延迟队列（以相同优先级排序）
                    heapq.heappush(deferred_by_pipe[p], (est, mp, -bl.get(nid, 0), order_index.get(nid, 10**12), nid))
                    return
                per_pipe_ready_count[p] += 1
            heapq.heappush(ready_heap, (est, mp, -bl.get(nid, 0), order_index.get(nid, 10**12), nid))

        for nid in base_schedule:
            if indeg.get(nid, 0) == 0:
                push_ready(nid)

        new_order: List[int] = []
        start_time: Dict[int, int] = {}
        end_time: Dict[int, int] = {}
        placed: Set[int] = set()

        # 创建进度条
        progress_bar = tqdm(
            total=len(base_schedule), 
            desc="T1-关键路径调度 ", 
            unit="节点",
            disable=not show_progress,
            mininterval=0.1,
            dynamic_ncols=True,
            leave=False,
            position=1
        )

        iter_count = 0
        while ready_heap and len(new_order) < len(base_schedule):
            est_old, mp_old, neg_bl, base_ord, nid = heapq.heappop(ready_heap)
            if nid in placed or indeg.get(nid, 0) != 0:
                continue
            # 懒更新检查
            est_now = compute_est(nid)
            # 仅在最早开工时间变大时执行懒更新，避免因内存罚项微变引起的过度重排
            if est_now > est_old:
                 # 关键值已变化，重新入堆
                last_key_est[nid] = est_now
                heapq.heappush(ready_heap, (est_now, mp_old, neg_bl, base_ord, nid))
                continue

            # 安排该节点
            node_pipe = pipe_map.get(nid)
            cycles = cycles_map.get(nid, 0)
            s = est_now
            e = s + cycles
            start_time[nid] = s
            end_time[nid] = e
            if node_pipe:
                pipe_avail[node_pipe] = e

            # 更新近似"驻留"以反映顺序上的 ALLOC/FREE 对内存压力的影响
            node = self.nodes.get(nid)
            if node is not None:
                op_code = op_code_map.get(nid, -1)
                ct_idx = cache_idx_map.get(nid, -1)
                if ct_idx != -1 and op_code != -1:
                    ct_name = 'L1' if ct_idx == 0 else 'UB'
                    if op_code == 0:  # ALLOC
                        mem_usage[ct_name] = mem_usage.get(ct_name, 0) + size_map.get(nid, 0)
                    elif op_code == 1:  # FREE
                        mem_usage[ct_name] = max(0, mem_usage.get(ct_name, 0) - size_map.get(nid, 0))

            new_order.append(nid)
            placed.add(nid)
            progress_bar.update(1)

            # 释放后继
            for v in succ_map.get(nid, []):
                indeg[v] -= 1
                # 更新依赖完成时间
                dep_ready_time[v] = max(dep_ready_time.get(v, 0), e)
                if indeg[v] == 0:
                    push_ready(v)

            # fast_mode：当前Pipe释放一个名额，从延迟队列补一个就绪节点
            if fast_mode:
                p = node_pipe
                # 释放名额
                per_pipe_ready_count[p] = max(0, per_pipe_ready_count[p] - 1)
                if deferred_by_pipe.get(p):
                    # 取该Pipe的最佳候选，重新计算est后入堆
                    _, _, _, _, cand = heapq.heappop(deferred_by_pipe[p])
                    # 仅当仍然就绪（入度应为0）时推进
                    if indeg.get(cand, 0) == 0 and cand not in placed:
                        est_c = compute_est(cand)
                        mp_c = 0.0 if fast_mode else last_key_mem.get(cand, 0.0)
                        heapq.heappush(ready_heap, (est_c, mp_c, -bl.get(cand, 0), order_index.get(cand, 10**12), cand))
                        per_pipe_ready_count[p] += 1

            # 周期性剪枝：限制就绪堆规模，缓解大型用例中期堆膨胀
            iter_count += 1
            if max_ready > 0 and len(ready_heap) > max_ready and (iter_count & 511) == 0:
                # 仅保留前 max_ready 个最有希望的候选（按现有键）
                ready_heap = heapq.nsmallest(max_ready, ready_heap)
                heapq.heapify(ready_heap)

        progress_bar.close()

        # 容错：若出现未全部排完（极端环/异常），按原序补齐
        if len(new_order) < len(base_schedule):
            seen = set(new_order)
            for nid in base_schedule:
                if nid not in seen:
                    new_order.append(nid)

        return new_order

    def _compute_topo_layers(self, succ_map: Dict[int, List[int]], base_schedule: List[int]) -> Tuple[Dict[int, int], int]:
        """基于 succ_map 计算拓扑层级（Kahn 层），返回 (node->layer, max_layer)。"""
        indeg = {nid: 0 for nid in base_schedule}
        for u, vs in succ_map.items():
            for v in vs:
                if v in indeg:
                    indeg[v] = indeg.get(v, 0) + 1
        # Kahn 层
        from collections import deque
        q = deque([nid for nid in base_schedule if indeg.get(nid, 0) == 0])
        layer = {nid: 0 for nid in q}
        max_layer = 0
        while q:
            u = q.popleft()
            lu = layer.get(u, 0)
            for v in succ_map.get(u, []):
                if v not in indeg:
                    continue
                indeg[v] -= 1
                if indeg[v] == 0:
                    layer[v] = lu + 1
                    if lu + 1 > max_layer:
                        max_layer = lu + 1
                    q.append(v)
        # 对未覆盖的节点（孤立或未遍历）给默认层
        for nid in base_schedule:
            if nid not in layer:
                layer[nid] = 0
        return layer, max_layer

    def build_time_optimal_schedule_segmented(self, base_schedule: List[int], segment_layers: int = 256, overlap_layers: int = 0, show_progress: bool = False) -> List[int]:
        """分段版 T1 调度：
        - 先按拓扑层分组，再对每段（若干层）独立执行 T1 列表调度，最后拼接。
        - 适合上万节点的大图，避免全局就绪堆在中期膨胀导致的卡顿。
        """
        if not base_schedule:
            return []
        # 构建不含Pipe顺序的依赖图
        succ_map, pred_map = self.build_dependency_without_pipe(base_schedule)
        # 计算拓扑层级
        layer, max_layer = self._compute_topo_layers(succ_map, base_schedule)
        # 层到节点（保持 base_schedule 稳定顺序）
        layers_list: List[List[int]] = [[] for _ in range(max_layer + 1 if max_layer >= 0 else 1)]
        for nid in base_schedule:
            l = layer.get(nid, 0)
            if l < 0:
                l = 0
            if l >= len(layers_list):
                # 扩容（极端情况）
                layers_list.extend([[] for _ in range(l - len(layers_list) + 1)])
            layers_list[l].append(nid)
        # 分段调度
        new_order: List[int] = []
        total_nodes = len(base_schedule)
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_nodes, desc="T1-分段调度", unit="节点", leave=False, position=1)
        start = 0
        while start <= max_layer:
            end = min(max_layer, start + segment_layers - 1)
            # 收集段内节点
            seg_nodes: List[int] = []
            # 无重叠或仅向后重叠（谨慎起见默认0，避免毁约）
            for l in range(start, end + 1 + max(0, overlap_layers)):
                if l <= max_layer:
                    seg_nodes.extend(layers_list[l])
            if seg_nodes:
                # 对段内执行一次 T1（局部），使用较小的就绪堆，禁用多余进度
                seg_order = self.build_time_optimal_schedule(seg_nodes, show_progress=False, max_ready=4096)
                new_order.extend(seg_order)
                if pbar:
                    pbar.update(len(seg_order))
            start = end + 1
        if pbar:
            pbar.close()
        return new_order if len(new_order) == len(base_schedule) else base_schedule

    # ===================== 问题3：局部优化与空洞填补 =====================
    def local_refine(self, schedule: List[int], max_iters: int = 20, window: int = 2, attempts_per_iter: int = 200, show_progress: bool = True) -> List[int]:
        """对已生成的调度做局部k-opt（k<=window）交换，尝试减少 makespan。
        策略：
          - 仅做不破坏依赖的相邻交换（或小窗口内重排）
          - 每次接受能使 compute_baseline_runtime 改善的变更
        注意：此方法调用 compute_baseline_runtime 多次，成本较高，建议在小窗口与有限迭代内使用。
        """
        if not schedule:
            return schedule
        best = list(schedule)
        # 预构建依赖与属性映射（不含Pipe顺序），用于快速评估
        succ_map, pred_map = self.build_dependency_without_pipe(schedule)
        cycles_map = {nid: (getattr(self.nodes.get(nid), 'cycles', 0) or 0) for nid in best}
        pipe_map = {nid: getattr(self.nodes.get(nid), 'pipe', None) for nid in best}
        best_T, start_end_best = self.compute_runtime_fast(best, pred_map, cycles_map, pipe_map)

        # 可达性缓存：仅在需要时对 (u,v) 做一次 DFS 并缓存结果
        reach_cache: Dict[Tuple[int, int], bool] = {}

        def reaches(u: int, v: int) -> bool:
            key = (u, v)
            if key in reach_cache:
                return reach_cache[key]
            # 小型DFS，早停
            stack = [u]
            visited = set()
            found = False
            while stack:
                x = stack.pop()
                if x == v:
                    found = True
                    break
                if x in visited:
                    continue
                visited.add(x)
                for y in succ_map.get(x, []):
                    if y not in visited:
                        stack.append(y)
            reach_cache[key] = found
            return found

        def can_swap_adjacent(order: List[int], i: int) -> bool:
            # 仅检查相邻交换 i<->i+1 是否违反可达性：若 i reaches i+1，则不可交换
            a = order[i]
            b = order[i + 1]
            if reaches(a, b):
                return False
            return True

        def detect_largest_pipe_gap(order: List[int], start_end: Dict[int, Tuple[int, int]]):
            """返回最大的管线空洞: (gap_size, pipe, prev_idx, next_idx, gap_start, gap_end)
            若无返回 None。
            """
            # 为每个pipe收集按开始时间排序的节点（来自当前order的相对顺序以保证稳定）
            nodes_by_pipe: Dict[str, List[int]] = defaultdict(list)
            for nid in order:
                p = pipe_map.get(nid)
                if p:
                    nodes_by_pipe[p].append(nid)
            largest = None
            max_gap = 0
            # 位置索引
            pos = {nid: idx for idx, nid in enumerate(order)}
            for p, ns in nodes_by_pipe.items():
                if len(ns) < 2:
                    continue
                # 按 start_time 排序（稳定次序用 pos 破平）
                ns_sorted = sorted(ns, key=lambda x: (start_end.get(x, (0, 0))[0], pos[x]))
                for k in range(len(ns_sorted) - 1):
                    a = ns_sorted[k]
                    b = ns_sorted[k + 1]
                    end_a = start_end.get(a, (0, 0))[1]
                    start_b = start_end.get(b, (0, 0))[0]
                    gap = start_b - end_a
                    if gap > max_gap:
                        max_gap = gap
                        largest = (gap, p, pos[a], pos[b], end_a, start_b)
            return largest

        def try_fill_gap(order: List[int], start_end: Dict[int, Tuple[int, int]], attempts_budget: int) -> Tuple[bool, List[int], int, Dict[int, Tuple[int, int]]]:
            """尝试通过将同Pipe的后续可行节点前移至空洞处来填补空洞。
            返回: (是否改进, 新顺序, 新T, 新start_end)
            """
            gap_info = detect_largest_pipe_gap(order, start_end)
            if not gap_info:
                return False, order, best_T, start_end
            gap, pipe, prev_idx, next_idx, gap_start_t, gap_end_t = gap_info
            if gap <= 0:
                return False, order, best_T, start_end
            pos = {nid: idx for idx, nid in enumerate(order)}
            # 候选从 next_idx 之后扫描，限定尝试次数
            tries = 0
            for cand_pos in range(next_idx + 1, len(order)):
                if tries >= attempts_budget:
                    break
                cand = order[cand_pos]
                if pipe_map.get(cand) != pipe:
                    continue
                # 择优：依赖准备时间估算（当前布局下）
                dep_ready = 0
                for u in pred_map.get(cand, []):
                    dep_ready = max(dep_ready, start_end.get(u, (0, 0))[1])
                # 简单筛选：若依赖准备时间远大于空洞结束，则跳过
                if dep_ready >= gap_end_t:
                    continue
                # 拟插入位置：放在next_idx之前
                insert_at = next_idx
                # 拓扑合法性检查：所有前驱必须位于 insert_at 之前
                ok = True
                for u in pred_map.get(cand, []):
                    if pos.get(u, -1) >= insert_at:
                        ok = False
                        break
                if not ok:
                    continue
                # 试移
                candidate = list(order)
                old = cand_pos
                # 先移除，再插入
                candidate.pop(old)
                if old < insert_at:
                    insert_at -= 1
                candidate.insert(insert_at, cand)
                T_new, se_new = self.compute_runtime_fast(candidate, pred_map, cycles_map, pipe_map)
                tries += 1
                if T_new < best_T:
                    return True, candidate, T_new, se_new
            return False, order, best_T, start_end

        # 添加进度条
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=max_iters, desc="局部优化", leave=False, position=1)

        it = 0
        while it < max_iters:
            improved = False
            n = len(best)
            attempts = 0
            i = 0
            # 先尝试空洞填补：将同Pipe且可行的后续节点前移至最大空洞处
            filled, new_order, new_T, new_start_end = try_fill_gap(best, start_end_best, attempts_per_iter // 2)
            if filled and new_T < best_T:
                best = new_order
                best_T = new_T
                start_end_best = new_start_end
                improved = True
                # 继续下一轮迭代
                it += 1
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"当前T": best_T, "改进": True, "方式": "gap"})
                continue

            # 若空洞填补未奏效，尝试相邻交换，线性扫描，限制每轮最大尝试次数
            while i < n - 1 and attempts < attempts_per_iter:
                candidate = list(best)
                if can_swap_adjacent(candidate, i):
                    candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
                    T, se = self.compute_runtime_fast(candidate, pred_map, cycles_map, pipe_map)
                    attempts += 1
                    if T < best_T:
                        best = candidate
                        best_T = T
                        start_end_best = se
                        improved = True
                        # 接受改进后，立即从上一位置回退一点，继续扫描
                        i = max(0, i - 2)
                        n = len(best)
                        continue
                i += 1
            if not improved:
                break
            it += 1
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"当前T": best_T, "改进": improved, "方式": "swap"})

        if progress_bar:
            progress_bar.close()

        return best

    # ===================== T2：辅助优化（位置重排，不新增SPILL） =====================
    def _jit_alloc_asap_free(self, order: List[int]) -> List[int]:
        if not order:
            return order
        nodes = self.nodes
        # 依赖图（不含Pipe顺序）
        succ_map, pred_map = self.build_dependency_without_pipe(order)
        pos = {nid: i for i, nid in enumerate(order)}
        def move_node(nid: int, new_idx: int):
            old_idx = pos[nid]
            if new_idx == old_idx:
                return
            node_id = order.pop(old_idx)
            if old_idx < new_idx:
                new_idx -= 1
            order.insert(new_idx, node_id)
            start = min(old_idx, new_idx)
            end = max(old_idx, new_idx)
            for i in range(start, end + 1):
                pos[order[i]] = i
        # ALLOC 下沉、FREE 上提（仅 L1/UB）
        for nid in list(order):
            node = nodes.get(nid)
            if not node:
                continue
            op = getattr(node, 'op', None)
            ct = getattr(node, 'cache_type', None)
            if ct not in ('L1', 'UB'):
                continue
            if op == 'ALLOC':
                preds = pred_map.get(nid, [])
                succs = succ_map.get(nid, [])
                if not succs:
                    continue
                earliest = max((pos.get(p, -1) for p in preds), default=-1) + 1
                latest = min((pos.get(s, 10**12) for s in succs), default=10**12) - 1
                cur = pos[nid]
                target = max(earliest, min(latest, cur))
                if target > cur:
                    move_node(nid, target)
            elif op == 'FREE':
                preds = pred_map.get(nid, [])
                succs = succ_map.get(nid, [])
                earliest = max((pos.get(p, -1) for p in preds), default=-1) + 1
                latest = min((pos.get(s, 10**12) for s in succs), default=10**12) - 1 if succs else len(order) - 1
                cur = pos[nid]
                target = max(earliest, min(latest, cur))
                if target < cur:
                    move_node(nid, target)
        return order

    def _prefetch_spill_in(self, order: List[int], lookback: int = 64) -> List[int]:
        if not order:
            return order
        nodes = self.nodes
        _, pred_map = self.build_dependency_without_pipe(order)
        pos = {nid: i for i, nid in enumerate(order)}
        def can_place_before(nid: int, idx: int) -> bool:
            for p in pred_map.get(nid, []):
                if pos.get(p, -1) >= idx:
                    return False
            return True
        def move_node_earlier(nid: int, new_idx: int):
            old_idx = pos[nid]
            if new_idx >= old_idx:
                return
            node_id = order.pop(old_idx)
            order.insert(new_idx, node_id)
            start = new_idx
            end = old_idx
            for i in range(start, end + 1):
                pos[order[i]] = i
        for nid in list(order):
            node = nodes.get(nid)
            if not node or getattr(node, 'op', None) != 'SPILL_IN':
                continue
            cur = pos[nid]
            target = max(0, cur - lookback)
            new_idx = cur
            i = cur - 1
            while i >= target:
                if can_place_before(nid, i):
                    new_idx = i
                    i -= 1
                    continue
                break
            if new_idx < cur:
                move_node_earlier(nid, new_idx)
        return order

    def optimize_T2(self, base_order: List[int], lookback_spillin: int = 64, show_progress: bool = True) -> List[int]:
        if not base_order:
            return base_order
        order = list(base_order)
        # 1) 生命周期压缩
        order = self._jit_alloc_asap_free(order)
        # 2) SPILL_IN 预取（不新增SPILL，仅前移）
        order = self._prefetch_spill_in(order, lookback=lookback_spillin)
        # 3) 局部优化（小迭代，避免雪崩）
        order = self.local_refine(order, max_iters=10, window=3, attempts_per_iter=200, show_progress=show_progress)
        return order

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
        if verbose: print(f"\nSolving {task_name}...")
        if verbose: print("--> 1/3: Running Problem 1 Scheduler...")
        schedule, max_v_stay = self.scheduler.schedule_problem1_binary_search()

        # 优先级B：局部重排（不破坏拓扑）以压缩缓冲区生命周期
        if verbose: print("--> 2/3: Performing local reschedule...")
        schedule = self._improve_schedule_local(schedule)
        
        if verbose:
            print(f"    Initial schedule length: {len(schedule)}")
            print(f"    Initial max(V_stay): {max_v_stay:,}")
        
        # 执行硬件仿真
        if verbose: print("--> 3/3: Simulating hardware execution...")
        buffer_allocations, spill_operations, complete_schedule = self.simulator.simulate_schedule(
            schedule, case_name=task_name, show_progress=show_progress
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
    
    # ===================== 问题3：基线时间便捷接口 =====================
    def compute_baseline_runtime(self, task_name: str = "") -> int:
        """计算问题3的基准总运行时间。
        若尚未运行 solve()，将先运行以生成完整调度序列与额外依赖。
        Returns: makespan
        """
        # 确保已有完整调度
        if self.complete_schedule is None:
            # 运行但不打印详细
            self.solve(task_name=task_name, verbose=False, show_progress=False)
        makespan, _ = self.simulator.compute_baseline_runtime(self.complete_schedule or [])
        return makespan
    
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
    
    for test_file in tqdm(test_files, desc="Problem 2 All Cases", position=0, unit="case"):
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
            
        except Exception as e:
            print(f"{task_name:<25} 执行失败: {e}")
    
    print("-" * 55)
    print(f"{'总计':<25} {total_spill_cost:<15,}")
    
    return results


def test_problem3_baseline():
    """测试问题3基准总运行时间（基线）"""
    test_files = [
        "data/Json_version/Matmul_Case0.json",
        "data/Json_version/FlashAttention_Case0.json",
        "data/Json_version/Conv_Case0.json",
        "data/Json_version/FlashAttention_Case1.json",
        "data/Json_version/Matmul_Case1.json",
        "data/Json_version/Conv_Case1.json",
    ]
    print("\n=== 问题3：基准总运行时间（基线） ===")
    print(f"{'任务名':<25} {'基准总时间(cycles)':<20}")
    print("-" * 50)
    results = {}
    for test_file in tqdm(test_files, desc="Problem 3 Baseline All Cases", position=0, unit="case"):
        task_name = test_file.split("/")[-1].replace(".json", "")
        try:
            solver = Problem2Solver(test_file)
            # 生成完整调度（含SPILL/复用依赖记录）
            solver.solve(task_name=task_name, verbose=False, show_progress=False)
            # 计算基线
            makespan = solver.compute_baseline_runtime(task_name=task_name)
            print(f"{task_name:<25} {makespan:<20}")
            results[task_name] = makespan
        except Exception as e:
            print(f"{task_name:<25} 失败: {e}")
    return results


def test_problem3_pipeline_optimization():
    """比较 T0(基线) / T1(关键路径列表调度) / T2(局部优化) 的示例入口"""
    test_files = [
        "data/Json_version/Matmul_Case0.json",
        "data/Json_version/Matmul_Case1.json",
        "data/Json_version/FlashAttention_Case0.json",
        "data/Json_version/FlashAttention_Case1.json",
        "data/Json_version/Conv_Case0.json",
        "data/Json_version/Conv_Case1.json",
    ]
    print("\n=== 问题3：流水并行优化(T0/T1/T2) ===")
    print(f"{'任务名':<25} {'T0':>12} {'T1':>12} {'T2':>12}")
    print("-" * 70)
    
    for test_file in tqdm(test_files, desc="Problem 3 Optimization All Cases", position=0, unit="case"):
        task_name = test_file.split("/")[-1].replace(".json", "")
        try:
            print(f"\n处理 {task_name}...")
            solver = Problem2Solver(test_file)
            
            # T0: 先跑问题2拿baseline完整序列
            print(f"  T0: 生成基线调度...")
            base_schedule, _, _, _, _ = solver.solve(task_name=task_name, verbose=False, show_progress=True)
            T0 = solver.compute_baseline_runtime(task_name=task_name)

            # T1: 关键路径驱动的列表调度（先对原始节点排序，再重仿真生成完整序列）
            print(f"  T1: 关键路径驱动列表调度...")
            # 仅保留非SPILL节点参与排序，避免二次插入SPILL
            nodes_ref = solver.simulator.nodes
            orig_only = [nid for nid in base_schedule if getattr(nodes_ref.get(nid), 'op', None) not in ("SPILL_OUT", "SPILL_IN")]
            p3_order_orig = solver.simulator.build_time_optimal_schedule(orig_only)
            # 以新顺序重走问题2仿真，得到与之匹配的SPILL与依赖
            solver.simulator.reset()
            _, _, p3_complete = solver.simulator.simulate_schedule(p3_order_orig, case_name=f"{task_name}_T1", verbose=False)
            T1, _ = solver.simulator.compute_baseline_runtime(p3_complete)

            # 回退保护：若不优于T0，则使用T0序列
            if T1 > T0:
                p3_complete = base_schedule
                T1 = T0

            # T2: 局部优化与空洞填补（基于T1的完整序列）
            print(f"  T2: 局部优化与空洞填补...")
            refined = solver.simulator.local_refine(p3_complete, max_iters=30, window=3, show_progress=True)
            T2, _ = solver.simulator.compute_baseline_runtime(refined)
            if T2 > T1:
                T2 = T1

            print(f"{task_name:<25} {T0:>12} {T1:>12} {T2:>12}")
            
            # 显示改进幅度
            if T1 < T0:
                improvement1 = (T0 - T1) / T0 * 100
                print(f"  T1 相对 T0 改进: {improvement1:.2f}%")
            if T2 < T1:
                improvement2 = (T1 - T2) / T1 * 100
                print(f"  T2 相对 T1 改进: {improvement2:.2f}%")
            if T2 < T0:
                total_improvement = (T0 - T2) / T0 * 100
                print(f"  T2 相对 T0 总改进: {total_improvement:.2f}%")
                
        except Exception as e:
            print(f"{task_name:<25} 失败: {e}")
            import traceback
            traceback.print_exc()

class Problem3Solver:
    """问题3的主要解决器：性能优化策略"""
    
    def __init__(self, graph_file: str):
        self.graph_file = graph_file
        self.scheduler = AdvancedScheduler(graph_file)
        # 问题三：放宽不利于优化的约束，关闭昂贵的环路守卫（改用单调顺序构造避免成环）
        self.simulator = HardwareSimulator(
            self.scheduler,
            enable_preemptive_spill=False,
            enforce_pipe_order_in_runtime=False,
            enforce_l0_single_live=False,
            enable_cycle_guard=False,
        )
        # Store results after solve() is run
        self.complete_schedule: Optional[List[int]] = None
        self.buffer_allocations: Optional[Dict[int, int]] = None
        self.spill_operations: Optional[List[SpillOperation]] = None
        self.optimized_schedule: Optional[List[int]] = None
        self.baseline_runtime: Optional[int] = None
        self.optimized_runtime: Optional[int] = None
        self.total_spill_cost: Optional[int] = None
        
        # T1和T2阶段的独立结果存储
        self.t1_schedule: Optional[List[int]] = None
        self.t1_buffer_allocations: Optional[Dict[int, int]] = None
        self.t1_spill_operations: Optional[List[SpillOperation]] = None
        self.t1_runtime: Optional[int] = None
        self.t1_spill_cost: Optional[int] = None
        
        self.t2_schedule: Optional[List[int]] = None
        self.t2_buffer_allocations: Optional[Dict[int, int]] = None
        self.t2_spill_operations: Optional[List[SpillOperation]] = None
        self.t2_runtime: Optional[int] = None
        self.t2_spill_cost: Optional[int] = None
        
        # 新增：存储详细分析数据用于可视化
        self.detailed_analysis_data = {
            'three_stage_comparison': [],
            'optimization_trajectory': [],
            'pipeline_utilization': {},
            'multi_objective_tradeoff': {},
            'convergence_history': [],
            'performance_breakdown': {}
        }
    
    def solve(self, task_name: str = "", verbose: bool = True, show_progress: bool = False) -> Tuple[List[int], Dict[int, int], List[SpillOperation], int, int, int, int, int]:
        """
        解决问题3：性能优化策略
        
        Returns:
            Tuple[List[int], Dict[int, int], List[SpillOperation], int, int, int, int, int]:
            (T2调度序列, 地址分配, SPILL操作, 总额外数据搬运量, T0基线运行时间, T1运行时间, T2运行时间, 最终选择阶段)
        """
        if verbose:
            print(f"\n=== 问题3：性能优化策略 - {task_name} ===")
        
        # 步骤1：获取问题2的基线结果（T0）
        if verbose:
            print("\n--- Stage T0: Simulating baseline schedule ---")
        
        # 使用问题2的求解器生成基线调度（包含SPILL与内存分配）
        p2_solver = P2BaselineSolver(self.graph_file)
        p2_complete, p2_allocations, p2_spill_ops, p2_spill_cost, _ = p2_solver.solve(
            task_name=f"{task_name}", verbose=False, show_progress=show_progress
        )
        complete_schedule = p2_complete
        # 供后续回退/返回使用的基线分配与SPILL列表
        buffer_allocations = p2_allocations
        spill_operations = p2_spill_ops
        
        # 将问题2中产生的新增节点（如SPILL）与额外依赖复制到问题3仿真器环境，确保后续时间评估一致
        for nid in complete_schedule:
            if nid not in self.simulator.nodes:
                src = p2_solver.simulator.nodes.get(nid) or p2_solver.scheduler.nodes.get(nid)
                if src is not None:
                    self.simulator.nodes[nid] = src
        self.simulator.extra_edges = list(getattr(p2_solver.simulator, 'extra_edges', []))
        
        # 计算基线运行时间（在问题3的时间评估器中计算，以统一Pipe独占与依赖处理）
        T0_runtime, _ = self.simulator.compute_baseline_runtime(complete_schedule)
        
        # 基线的数据搬运量采用问题2仿真的结果
        T0_spill_cost = p2_spill_cost
         
        if verbose:
            print(f"T0 Baseline Runtime: {T0_runtime:,} cycles")
            print(f"T0 Spill Cost: {T0_spill_cost:,}")
        
        # 步骤2：关键路径驱动的列表调度优化（T1）
        if verbose:
            print("\n--- Stage T1: Applying critical-path list scheduling ---")
        
        # 仅保留非SPILL节点参与排序，避免二次插入SPILL
        nodes_ref = self.simulator.nodes
        orig_only = [nid for nid in complete_schedule if getattr(nodes_ref.get(nid), 'op', None) not in ("SPILL_OUT", "SPILL_IN")]
        
        # 生成关键路径优化的调度顺序（大图自动采用分段T1）
        if len(orig_only) >= 8000:
            p3_order_orig = self.simulator.build_time_optimal_schedule_segmented(orig_only, segment_layers=256, overlap_layers=0, show_progress=show_progress)
        else:
            p3_order_orig = self.simulator.build_time_optimal_schedule(orig_only, show_progress=show_progress)
        
        # 以新顺序重走问题2仿真，得到与之匹配的SPILL与依赖
        self.simulator.reset()
        _, _, p3_complete = self.simulator.simulate_schedule(p3_order_orig, case_name=f"{task_name}_T1", show_progress=show_progress)
        T1_runtime, _ = self.simulator.compute_baseline_runtime(p3_complete)
        T1_spill_cost = self.simulator.calculate_total_spill_cost()
        
        # 回退保护：若不优于T0，则使用T0序列
        if T1_runtime >= T0_runtime:
            p3_complete = complete_schedule
            T1_runtime = T0_runtime
            T1_spill_cost = T0_spill_cost
        
        # 存储T1阶段结果
        self.t1_schedule = list(p3_complete)
        self.t1_runtime = T1_runtime
        self.t1_spill_cost = T1_spill_cost
        # 获取T1阶段的缓冲区分配和SPILL操作
        t1_allocations = {}
        for buf_id, buffer_info in self.simulator.buffer_infos.items():
            if buffer_info.allocated_offset is not None:
                t1_allocations[buf_id] = buffer_info.allocated_offset
        self.t1_buffer_allocations = t1_allocations
        self.t1_spill_operations = list(self.simulator.spill_operations)
        
        if verbose:
            print(f"T1 Optimized Runtime: {T1_runtime:,} cycles")
            print(f"T1 Spill Cost: {T1_spill_cost:,}")
        
        # 步骤3：局部优化与空洞填补（T2）
        if verbose:
            print("\n--- Stage T2: Applying local refinement & gap filling ---")
        
        # 先运行T2管线（JIT/ASAP + 预取 + 局部优化）
        refined = self.simulator.optimize_T2(p3_complete, lookback_spillin=64, show_progress=show_progress)
        # 评估（不重新插入SPILL，避免新增搬运）
        T2_runtime, _ = self.simulator.compute_baseline_runtime(refined)
        T2_spill_cost = self.simulator.calculate_total_spill_cost()
         
         # 回退保护：若不优于T1，则使用T1序列
        if T2_runtime >= T1_runtime:
            refined = p3_complete
            T2_runtime = T1_runtime
            T2_spill_cost = T1_spill_cost
        
        # 存储T2阶段结果
        self.t2_schedule = list(refined)
        self.t2_runtime = T2_runtime
        self.t2_spill_cost = T2_spill_cost
        # 获取T2阶段的缓冲区分配和SPILL操作
        t2_allocations = {}
        for buf_id, buffer_info in self.simulator.buffer_infos.items():
            if buffer_info.allocated_offset is not None:
                t2_allocations[buf_id] = buffer_info.allocated_offset
        self.t2_buffer_allocations = t2_allocations
        self.t2_spill_operations = list(self.simulator.spill_operations)
        
        if verbose:
            print(f"T2 Optimized Runtime: {T2_runtime:,} cycles")
            print(f"T2 Spill Cost: {T2_spill_cost:,}")
        
        # 计算改进百分比
        T1_improvement = ((T0_runtime - T1_runtime) / T0_runtime * 100) if T0_runtime > 0 else 0
        T2_improvement = ((T0_runtime - T2_runtime) / T0_runtime * 100) if T0_runtime > 0 else 0
        
        # 收集三阶段对比数据
        self.detailed_analysis_data['three_stage_comparison'] = [
            {'stage': 'T0基线', 'runtime': T0_runtime, 'spill_cost': T0_spill_cost, 'improvement': 0.0},
            {'stage': 'T1优化', 'runtime': T1_runtime, 'spill_cost': T1_spill_cost, 'improvement': T1_improvement},
            {'stage': 'T2优化', 'runtime': T2_runtime, 'spill_cost': T2_spill_cost, 'improvement': T2_improvement}
        ]
        
        # 收集多目标权衡数据
        self.detailed_analysis_data['multi_objective_tradeoff'] = {
            'runtime_reduction': T2_improvement,
            'spill_cost_increase': ((T2_spill_cost - T0_spill_cost) / max(T0_spill_cost, 1) * 100),
            'pareto_points': [
                {'runtime': T0_runtime, 'spill_cost': T0_spill_cost, 'stage': 'T0'},
                {'runtime': T1_runtime, 'spill_cost': T1_spill_cost, 'stage': 'T1'},
                {'runtime': T2_runtime, 'spill_cost': T2_spill_cost, 'stage': 'T2'}
            ]
        }
        
        # 打印三阶段对比
        if verbose:
            print(f"\n=== 三阶段对比 ===")
            print(f"{'阶段':<8} {'运行时间(cycles)':<15} {'数据搬运量':<12} {'相对T0改进%':<12}")
            print("-" * 55)
            print(f"{'T0基线':<8} {T0_runtime:<15,} {T0_spill_cost:<12,} {'0.00%':<12}")
            print(f"{'T1优化':<8} {T1_runtime:<15,} {T1_spill_cost:<12,} {T1_improvement:<11.2f}%")
            print(f"{'T2优化':<8} {T2_runtime:<15,} {T2_spill_cost:<12,} {T2_improvement:<11.2f}%")
            
            # 添加详细的问题二vs问题三对比信息
            print(f"\n=== 问题二基准 vs 问题三优化对比 ===")
            print(f"任务名称: {task_name}")
            print(f"问题二基准运行时间: {T0_runtime:,} cycles")
            print(f"问题三优化运行时间: {T2_runtime:,} cycles") 
            print(f"问题二基准数据搬运量: {T0_spill_cost:,} bytes")
            print(f"问题三优化数据搬运量: {T2_spill_cost:,} bytes")
            print(f"执行时间改进: {T2_improvement:.1f}%")
            spill_change = ((T2_spill_cost - T0_spill_cost) / max(T0_spill_cost, 1)) * 100
            print(f"数据搬运量变化: {spill_change:+.1f}%")
        
        # 确定最终选择的阶段
        if T2_runtime < T1_runtime and T2_runtime < T0_runtime:
            final_stage = 2  # T2最优
            final_schedule = refined
            final_allocations = self.simulator.buffer_allocations if hasattr(self.simulator, 'buffer_allocations') else buffer_allocations
            final_spill_ops = self.simulator.spill_operations
            final_runtime = T2_runtime
            final_spill_cost = T2_spill_cost
        elif T1_runtime < T0_runtime:
            final_stage = 1  # T1最优
            final_schedule = p3_complete
            final_allocations = self.simulator.buffer_allocations if hasattr(self.simulator, 'buffer_allocations') else buffer_allocations
            final_spill_ops = self.simulator.spill_operations
            final_runtime = T1_runtime
            final_spill_cost = T1_spill_cost
        else:
            final_stage = 0  # T0最优
            final_schedule = complete_schedule
            final_allocations = buffer_allocations
            final_spill_ops = spill_operations
            final_runtime = T0_runtime
            final_spill_cost = T0_spill_cost
        
        # 存储T2结果（无论最终选择哪个阶段，都保存T2的结果）
        self.complete_schedule = refined  # 始终保存T2结果
        self.buffer_allocations = self.simulator.buffer_allocations if hasattr(self.simulator, 'buffer_allocations') else buffer_allocations
        self.spill_operations = self.simulator.spill_operations
        self.optimized_schedule = refined
        self.baseline_runtime = T0_runtime
        self.optimized_runtime = T2_runtime  # 保存T2运行时间
        self.total_spill_cost = T2_spill_cost  # 保存T2数据搬运量
        
        # 收集更多分析数据
        self._collect_pipeline_utilization_data(task_name)
        self._collect_performance_breakdown_data(task_name, T0_runtime, T1_runtime, T2_runtime, T0_spill_cost, T1_spill_cost, T2_spill_cost)
        
        # 打印最终选择
        if verbose:
            stage_names = {0: "T0基线", 1: "T1优化", 2: "T2优化"}
            print(f"\n=== 最终选择 ===")
            print(f"最优阶段: {stage_names[final_stage]} (运行时间: {final_runtime:,} cycles)")
            print(f"保存结果: T2阶段 (运行时间: {T2_runtime:,} cycles, 数据搬运量: {T2_spill_cost:,})")
        
        return refined, self.simulator.buffer_allocations if hasattr(self.simulator, 'buffer_allocations') else buffer_allocations, self.simulator.spill_operations, T2_spill_cost, T0_runtime, T1_runtime, T2_runtime, final_stage
    
    def save_results(self, output_dir: str, task_name: str):
        """将问题3结果保存到Problem3文件夹，符合题目要求格式"""
        # 检查结果是否存在，如果不存在，则运行求解以生成结果
        if self.complete_schedule is None or self.buffer_allocations is None or self.spill_operations is None:
            print(f"\nWarning: {task_name} 的结果未找到。正在运行求解以生成结果。")
            self.solve(task_name=task_name, verbose=False, show_progress=True)

        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存调度序列 - 格式：每行一个节点Id
        schedule_file = f"{output_dir}/{task_name}_schedule.txt"
        with open(schedule_file, 'w') as f:
            for node_id in self.complete_schedule:
                f.write(f"{node_id}\n")
        
        # 保存缓存分配结果 - 格式：每行 BufId:Offset
        memory_file = f"{output_dir}/{task_name}_memory.txt"
        with open(memory_file, 'w') as f:
            for buf_id, offset in self.buffer_allocations.items():
                f.write(f"{buf_id}:{offset}\n")
        
        # 保存SPILL操作列表 - 格式：每行 BufId:NewOffset（若无SPILL操作则为空文件）
        spill_file = f"{output_dir}/{task_name}_spill.txt"
        with open(spill_file, 'w') as f:
            for spill_op in self.spill_operations:
                f.write(f"{spill_op.buf_id}:{spill_op.new_offset}\n")
        
        print(f"问题3结果已保存到 {output_dir}/")
        print(f"  - {task_name}_schedule.txt: {len(self.complete_schedule)} 个节点")
        print(f"  - {task_name}_memory.txt: {len(self.buffer_allocations)} 个缓冲区分配")
        print(f"  - {task_name}_spill.txt: {len(self.spill_operations)} 个SPILL操作")
    
    def save_t1_results(self, output_dir: str, task_name: str):
        """将T1优化结果保存到problem3_opt1文件夹"""
        # 检查T1结果是否存在
        if self.t1_schedule is None or self.t1_buffer_allocations is None or self.t1_spill_operations is None:
            print(f"\nWarning: {task_name} 的T1结果未找到。请先运行solve方法。")
            return

        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存T1调度序列
        schedule_file = f"{output_dir}/{task_name}_schedule.txt"
        with open(schedule_file, 'w') as f:
            for node_id in self.t1_schedule:
                f.write(f"{node_id}\n")
        
        # 保存T1缓存分配结果
        memory_file = f"{output_dir}/{task_name}_memory.txt"
        with open(memory_file, 'w') as f:
            for buf_id, offset in self.t1_buffer_allocations.items():
                f.write(f"{buf_id}:{offset}\n")
        
        # 保存T1 SPILL操作列表
        spill_file = f"{output_dir}/{task_name}_spill.txt"
        with open(spill_file, 'w') as f:
            for spill_op in self.t1_spill_operations:
                f.write(f"{spill_op.buf_id}:{spill_op.new_offset}\n")
        
        print(f"T1优化结果已保存到 {output_dir}/")
        print(f"  - {task_name}_schedule.txt: {len(self.t1_schedule)} 个节点")
        print(f"  - {task_name}_memory.txt: {len(self.t1_buffer_allocations)} 个缓冲区分配")
        print(f"  - {task_name}_spill.txt: {len(self.t1_spill_operations)} 个SPILL操作")
        print(f"  - T1运行时间: {self.t1_runtime:,} cycles")
        print(f"  - T1数据搬运量: {self.t1_spill_cost:,} bytes")
    
    def save_t2_results(self, output_dir: str, task_name: str):
        """将T2优化结果保存到problem3_opt2文件夹"""
        # 检查T2结果是否存在
        if self.t2_schedule is None or self.t2_buffer_allocations is None or self.t2_spill_operations is None:
            print(f"\nWarning: {task_name} 的T2结果未找到。请先运行solve方法。")
            return

        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存T2调度序列
        schedule_file = f"{output_dir}/{task_name}_schedule.txt"
        with open(schedule_file, 'w') as f:
            for node_id in self.t2_schedule:
                f.write(f"{node_id}\n")
        
        # 保存T2缓存分配结果
        memory_file = f"{output_dir}/{task_name}_memory.txt"
        with open(memory_file, 'w') as f:
            for buf_id, offset in self.t2_buffer_allocations.items():
                f.write(f"{buf_id}:{offset}\n")
        
        # 保存T2 SPILL操作列表
        spill_file = f"{output_dir}/{task_name}_spill.txt"
        with open(spill_file, 'w') as f:
            for spill_op in self.t2_spill_operations:
                f.write(f"{spill_op.buf_id}:{spill_op.new_offset}\n")
        
        print(f"T2优化结果已保存到 {output_dir}/")
        print(f"  - {task_name}_schedule.txt: {len(self.t2_schedule)} 个节点")
        print(f"  - {task_name}_memory.txt: {len(self.t2_buffer_allocations)} 个缓冲区分配")
        print(f"  - {task_name}_spill.txt: {len(self.t2_spill_operations)} 个SPILL操作")
        print(f"  - T2运行时间: {self.t2_runtime:,} cycles")
        print(f"  - T2数据搬运量: {self.t2_spill_cost:,} bytes")
    
    def _collect_pipeline_utilization_data(self, task_name: str):
        """收集流水线利用率数据"""
        try:
            # 统计执行单元使用情况
            pipe_usage = {}
            total_cycles = 0
            node_count_by_pipe = {}
            
            for node_id in self.complete_schedule:
                # 检查多个节点字典源
                node = None
                if hasattr(self, 'simulator') and self.simulator.nodes.get(node_id):
                    node = self.simulator.nodes[node_id]
                elif self.scheduler.nodes.get(node_id):
                    node = self.scheduler.nodes[node_id]
                
                if node:
                    pipe = getattr(node, 'pipe', None)
                    if pipe:
                        cycles = getattr(node, 'cycles', 0) or 0
                        if pipe not in pipe_usage:
                            pipe_usage[pipe] = 0
                            node_count_by_pipe[pipe] = 0
                        pipe_usage[pipe] += cycles
                        node_count_by_pipe[pipe] += 1
                        total_cycles += cycles
            
            # 计算利用率（相对于总执行时间）
            utilization_rates = {}
            for pipe, cycles in pipe_usage.items():
                utilization_rates[pipe] = (cycles / max(total_cycles, 1)) * 100
            
            # 计算平均每个节点的cycles
            avg_cycles_per_node = {}
            for pipe in pipe_usage:
                if node_count_by_pipe[pipe] > 0:
                    avg_cycles_per_node[pipe] = pipe_usage[pipe] / node_count_by_pipe[pipe]
                else:
                    avg_cycles_per_node[pipe] = 0
            
            self.detailed_analysis_data['pipeline_utilization'] = {
                'pipe_usage_cycles': pipe_usage,
                'utilization_rates': utilization_rates,
                'node_count_by_pipe': node_count_by_pipe,
                'avg_cycles_per_node': avg_cycles_per_node,
                'total_cycles': total_cycles,
                'active_pipes': len(pipe_usage),
                'total_nodes': len(self.complete_schedule) if self.complete_schedule else 0
            }
            
            if not pipe_usage:
                print(f"Warning: {task_name} 未找到任何执行单元数据")
                
        except Exception as e:
            print(f"Warning: 收集流水线数据时出错: {e}")
            import traceback
            traceback.print_exc()
            self.detailed_analysis_data['pipeline_utilization'] = {}
    
    def _collect_performance_breakdown_data(self, task_name: str, T0_runtime: int, T1_runtime: int, T2_runtime: int, 
                                          T0_spill_cost: int, T1_spill_cost: int, T2_spill_cost: int):
        """收集性能分解数据"""
        # 计算各阶段的贡献
        t1_contribution = T0_runtime - T1_runtime if T1_runtime < T0_runtime else 0
        t2_contribution = T1_runtime - T2_runtime if T2_runtime < T1_runtime else 0
        
        self.detailed_analysis_data['performance_breakdown'] = {
            'baseline_performance': {
                'runtime': T0_runtime,
                'spill_cost': T0_spill_cost
            },
            'stage_contributions': {
                'T1_critical_path_optimization': {
                    'runtime_reduction': t1_contribution,
                    'spill_cost_change': T1_spill_cost - T0_spill_cost
                },
                'T2_local_refinement': {
                    'runtime_reduction': t2_contribution, 
                    'spill_cost_change': T2_spill_cost - T1_spill_cost
                }
            },
            'final_performance': {
                'runtime': T2_runtime,
                'spill_cost': T2_spill_cost,
                'total_improvement': ((T0_runtime - T2_runtime) / T0_runtime * 100) if T0_runtime > 0 else 0
            }
        }
    
    def save_visualization_data(self, output_dir: str, task_name: str):
        """保存可视化数据到JSON文件"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保所有必要的分析数据都存在
        if not hasattr(self, 'detailed_analysis_data') or not self.detailed_analysis_data:
            print(f"Warning: {task_name} 缺少详细分析数据，重新收集...")
            self._ensure_complete_analysis_data()
        
        # 准备完整的可视化数据
        viz_data = {
            'task_name': task_name,
            'graph_file': self.graph_file,
            'timestamp': str(__import__('datetime').datetime.now()),
            'results': {
                'baseline_runtime': self.baseline_runtime,
                'optimized_runtime': self.optimized_runtime,
                'total_spill_cost': self.total_spill_cost,
                'schedule_length': len(self.complete_schedule) if self.complete_schedule else 0,
                'buffer_allocations_count': len(self.buffer_allocations) if self.buffer_allocations else 0,
                'spill_operations_count': len(self.spill_operations) if self.spill_operations else 0
            },
            'analysis': self.detailed_analysis_data,
            'optimization_summary': self.get_optimization_summary()
        }
        
        # 添加调度序列信息（统计）
        if self.complete_schedule:
            node_stats = self._analyze_schedule_composition()
            viz_data['schedule_analysis'] = node_stats
        
        # 验证数据完整性
        self._validate_visualization_data(viz_data, task_name)
        
        # 保存到JSON文件
        viz_file = os.path.join(output_dir, f"{task_name}_problem3_visualization.json")
        with open(viz_file, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)
        
        print(f"可视化数据已保存到: {viz_file}")
        print(f"  - 三阶段数据: {len(viz_data.get('analysis', {}).get('three_stage_comparison', []))} 个阶段")
        print(f"  - 流水线数据: {len(viz_data.get('analysis', {}).get('pipeline_utilization', {}).get('pipe_usage_cycles', {}))} 个执行单元")
        print(f"  - 多目标数据: {len(viz_data.get('analysis', {}).get('multi_objective_tradeoff', {}).get('pareto_points', []))} 个Pareto点")
        return viz_file
    
    def _validate_visualization_data(self, viz_data: Dict, task_name: str):
        """验证可视化数据的完整性"""
        warnings = []
        
        # 检查基础结果数据
        results = viz_data.get('results', {})
        if not results.get('baseline_runtime'):
            warnings.append("缺少基线运行时间")
        if not results.get('optimized_runtime'):
            warnings.append("缺少优化运行时间")
        
        # 检查分析数据
        analysis = viz_data.get('analysis', {})
        if not analysis.get('three_stage_comparison'):
            warnings.append("缺少三阶段对比数据")
        elif len(analysis['three_stage_comparison']) < 3:
            warnings.append(f"三阶段数据不完整，只有 {len(analysis['three_stage_comparison'])} 个阶段")
        
        if not analysis.get('multi_objective_tradeoff'):
            warnings.append("缺少多目标权衡数据")
        
        pipeline_data = analysis.get('pipeline_utilization', {})
        if not pipeline_data.get('pipe_usage_cycles'):
            warnings.append("缺少流水线使用数据")
        
        if not analysis.get('performance_breakdown'):
            warnings.append("缺少性能分解数据")
        
        # 打印警告
        if warnings:
            print(f"Warning: {task_name} 可视化数据验证发现问题:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print(f"✓ {task_name} 可视化数据验证通过")
    
    def _ensure_complete_analysis_data(self):
        """确保所有分析数据完整"""
        if not hasattr(self, 'detailed_analysis_data'):
            self.detailed_analysis_data = {
                'three_stage_comparison': [],
                'optimization_trajectory': [],
                'pipeline_utilization': {},
                'multi_objective_tradeoff': {},
                'convergence_history': [],
                'performance_breakdown': {}
            }
        
        # 如果缺少三阶段数据，尝试重建
        if not self.detailed_analysis_data.get('three_stage_comparison'):
            if self.baseline_runtime and self.optimized_runtime:
                improvement = ((self.baseline_runtime - self.optimized_runtime) / self.baseline_runtime * 100) if self.baseline_runtime > 0 else 0
                self.detailed_analysis_data['three_stage_comparison'] = [
                    {'stage': 'T0基线', 'runtime': self.baseline_runtime, 'spill_cost': self.total_spill_cost or 0, 'improvement': 0.0},
                    {'stage': 'T1优化', 'runtime': self.optimized_runtime, 'spill_cost': self.total_spill_cost or 0, 'improvement': improvement},
                    {'stage': 'T2优化', 'runtime': self.optimized_runtime, 'spill_cost': self.total_spill_cost or 0, 'improvement': improvement}
                ]
        
        # 如果缺少多目标权衡数据，重建
        if not self.detailed_analysis_data.get('multi_objective_tradeoff'):
            if self.baseline_runtime and self.optimized_runtime:
                runtime_reduction = ((self.baseline_runtime - self.optimized_runtime) / self.baseline_runtime * 100) if self.baseline_runtime > 0 else 0
                self.detailed_analysis_data['multi_objective_tradeoff'] = {
                    'runtime_reduction': runtime_reduction,
                    'spill_cost_increase': 0,  # 默认值
                    'pareto_points': [
                        {'runtime': self.baseline_runtime, 'spill_cost': self.total_spill_cost or 0, 'stage': 'T0'},
                        {'runtime': self.optimized_runtime, 'spill_cost': self.total_spill_cost or 0, 'stage': 'T2'}
                    ]
                }
        
        # 如果缺少流水线数据，重新收集
        if not self.detailed_analysis_data.get('pipeline_utilization'):
            self._collect_pipeline_utilization_data("unknown")
    
    def _analyze_schedule_composition(self) -> Dict:
        """分析调度序列组成"""
        if not self.complete_schedule:
            return {}
        
        node_types = {
            'alloc_nodes': 0,
            'free_nodes': 0,
            'spill_out_nodes': 0,
            'spill_in_nodes': 0,
            'compute_nodes': 0,
            'copy_nodes': 0,
            'other_nodes': 0
        }
        
        pipe_usage = {}
        
        for node_id in self.complete_schedule:
            node = self.scheduler.nodes.get(node_id)
            if node:
                op = getattr(node, 'op', '')
                pipe = getattr(node, 'pipe', '')
                
                if op == 'ALLOC':
                    node_types['alloc_nodes'] += 1
                elif op == 'FREE':
                    node_types['free_nodes'] += 1
                elif op == 'SPILL_OUT':
                    node_types['spill_out_nodes'] += 1
                elif op == 'SPILL_IN':
                    node_types['spill_in_nodes'] += 1
                elif 'COPY' in op:
                    node_types['copy_nodes'] += 1
                elif pipe:
                    node_types['compute_nodes'] += 1
                else:
                    node_types['other_nodes'] += 1
                
                if pipe:
                    pipe_usage[pipe] = pipe_usage.get(pipe, 0) + 1
        
        return {
            'node_types': node_types,
            'pipe_usage': pipe_usage,
            'total_nodes': len(self.complete_schedule)
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化效果总结"""
        if self.baseline_runtime is None or self.optimized_runtime is None:
            return {}
        
        runtime_improvement = (self.baseline_runtime - self.optimized_runtime) / self.baseline_runtime * 100
        spill_cost = self.total_spill_cost or 0
        
        return {
            "baseline_runtime": self.baseline_runtime,
            "optimized_runtime": self.optimized_runtime,
            "runtime_improvement_percent": runtime_improvement,
            "total_spill_cost": spill_cost,
            "spill_operations_count": len(self.spill_operations) if self.spill_operations else 0
        }


def test_problem3():
    """测试问题3完整解决方案"""
    test_files = [
        "data/Json_version/Matmul_Case0.json",
        "data/Json_version/Matmul_Case1.json",
        "data/Json_version/FlashAttention_Case0.json",
        "data/Json_version/FlashAttention_Case1.json",
        "data/Json_version/Conv_Case0.json",
        "data/Json_version/Conv_Case1.json",
    ]
    
    print("=== 问题3：性能优化策略测试 ===")
    print(f"{'任务名':<25} {'T0基线':<12} {'T1优化':<12} {'T2优化':<12} {'最优阶段':<8} {'T2数据搬运量':<12}")
    print("-" * 90)
    
    results = {}
    detailed_results = []  # 存储详细结果用于生成表格
    total_T0 = 0
    total_T1 = 0
    total_T2 = 0
    total_T0_spill = 0
    total_T2_spill = 0
    
    for test_file in tqdm(test_files, desc="Problem 3 All Cases", position=0, unit="case"):
        try:
            task_name = test_file.split("/")[-1].replace(".json", "")
            
            solver = Problem3Solver(test_file)
            # 运行优化，显示详细信息
            _, _, _, T2_spill_cost, T0_runtime, T1_runtime, T2_runtime, final_stage = solver.solve(
                task_name=task_name, verbose=True, show_progress=True
            )
            
            # 获取T0阶段的数据搬运量
            T0_spill_cost = solver.detailed_analysis_data['three_stage_comparison'][0]['spill_cost']
            
            # 确定最优阶段名称
            stage_names = {0: "T0", 1: "T1", 2: "T2"}
            best_stage = stage_names[final_stage]
            
            # 打印结果
            print(f"{task_name:<25} {T0_runtime:<12,} {T1_runtime:<12,} {T2_runtime:<12,} {best_stage:<8} {T2_spill_cost:<12,}")
            
            # 存储详细结果
            detailed_results.append({
                'task_name': task_name,
                'T0_runtime': T0_runtime,
                'T2_runtime': T2_runtime, 
                'T0_spill_cost': T0_spill_cost,
                'T2_spill_cost': T2_spill_cost,
                'time_improvement': ((T0_runtime - T2_runtime) / T0_runtime * 100) if T0_runtime > 0 else 0,
                'spill_change': ((T2_spill_cost - T0_spill_cost) / max(T0_spill_cost, 1) * 100)
            })
            
            # 累计统计
            total_T0 += T0_runtime
            total_T1 += T1_runtime
            total_T2 += T2_runtime
            total_T0_spill += T0_spill_cost
            total_T2_spill += T2_spill_cost
            
            results[task_name] = {
                "T0_runtime": T0_runtime,
                "T1_runtime": T1_runtime,
                "T2_runtime": T2_runtime,
                "T0_spill_cost": T0_spill_cost,
                "T2_spill_cost": T2_spill_cost,
                "final_stage": final_stage
            }
            
            # 保存结果到Problem3文件夹（保存T2结果，保持兼容性）
            solver.save_results("Problem3", task_name)
            
            # 按赛题要求分别保存T1和T2结果
            solver.save_t1_results("problem3_opt1", task_name)
            solver.save_t2_results("problem3_opt2", task_name)
            
            # 保存可视化数据
            solver.save_visualization_data("Problem3_Visualization_Data", task_name)
            
        except Exception as e:
            print(f"{task_name:<25} 执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总计
    T1_improvement = ((total_T0 - total_T1) / total_T0 * 100) if total_T0 > 0 else 0
    T2_improvement = ((total_T0 - total_T2) / total_T0 * 100) if total_T0 > 0 else 0
    total_spill_improvement = ((total_T2_spill - total_T0_spill) / max(total_T0_spill, 1) * 100)
    
    print("-" * 90)
    print(f"{'总计':<25} {total_T0:<12,} {total_T1:<12,} {total_T2:<12,} {'T2':<8} {total_T2_spill:<12,}")
    print(f"{'改进%':<25} {'0.00%':<12} {T1_improvement:<11.2f}% {T2_improvement:<11.2f}% {'':<8} {total_spill_improvement:<11.2f}%")
    
    # 生成LaTeX表格数据
    print("\n" + "="*80)
    print("LaTeX表格数据（可直接复制到问题三.tex）:")
    print("="*80)
    
    for result in detailed_results:
        task_display = result['task_name'].replace('_', '\\_')
        print(f"{task_display} & {result['T0_runtime']:,} & {result['T2_runtime']:,} & {result['T0_spill_cost']:,} & {result['T2_spill_cost']:,} \\\\")
    
    # 计算平均改进比例  
    avg_time_improvement = sum(r['time_improvement'] for r in detailed_results) / len(detailed_results)
    avg_spill_change = sum(r['spill_change'] for r in detailed_results) / len(detailed_results)
    
    print(f"\\midrule")
    print(f"\\textbf{{平均优化比例}} & \\textbf{{-}} & \\textbf{{{avg_time_improvement:+.1f}\\%}} & \\textbf{{-}} & \\textbf{{{avg_spill_change:+.1f}\\%}} \\\\")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # 运行问题3完整测试
    results = test_problem3()
    print(f"\n问题3测试完成，结果已保存到Problem3文件夹")
