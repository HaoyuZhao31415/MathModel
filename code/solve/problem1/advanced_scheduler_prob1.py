#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
"""
基于最前沿技术的高级缓存驻留调度算法
采用JIT分配、早期释放、内存压力感知等先进策略
"""

import json
import heapq
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import math


class Node:
    """计算图节点"""
    def __init__(self, node_data: dict):
        self.id = node_data["Id"]
        self.op = node_data["Op"]
        
        # 缓存管理节点属性
        if self.op in ["ALLOC", "FREE"]:
            self.buf_id = node_data["BufId"]
            self.size = node_data["Size"]
            self.cache_type = node_data["Type"]
        else:
            # 操作节点属性
            self.pipe = node_data.get("Pipe", "")
            self.cycles = node_data.get("Cycles", 0)
            self.bufs = node_data.get("Bufs", [])
            self.buf_id = None
            self.size = 0
            self.cache_type = ""

    def is_cache_node(self) -> bool:
        """判断是否为缓存管理节点"""
        return self.op in ["ALLOC", "FREE"]
    
    def is_l1_or_ub_cache(self) -> bool:
        """判断是否为L1或UB类型缓存"""
        return self.cache_type in ["L1", "UB"]
    
    def memory_delta(self) -> int:
        """计算对V_stay的影响"""
        if not self.is_cache_node() or not self.is_l1_or_ub_cache():
            return 0
        return self.size if self.op == "ALLOC" else -self.size


class AdvancedBufferAnalysis:
    """高级缓冲区分析"""
    def __init__(self, buf_id: int, alloc_node: int, free_node: int, size: int):
        self.buf_id = buf_id
        self.alloc_node = alloc_node
        self.free_node = free_node
        self.size = size
        
        # 生存期分析
        self.usage_nodes = []  # 使用该缓冲区的操作节点
        self.first_usage = None  # 首次使用节点
        self.last_usage = None   # 最后使用节点
        
        # JIT分析
        self.latest_alloc_time = 0  # 最晚分配时间
        self.earliest_free_time = float('inf')  # 最早释放时间
        
        # 压力分析
        self.memory_pressure_score = 0.0  # 内存压力分数
        self.critical_path_weight = 0.0   # 关键路径权重
        
        # 新增：精细生存期分析
        self.usage_intervals = []  # 使用间隔分析
        self.idle_windows = []     # 空闲窗口
        self.usage_density = 0.0   # 使用密度
        self.lifetime_efficiency = 0.0  # 生存期效率
        
        # 新增：预测性分析
        self.future_memory_demand = 0.0  # 未来内存需求预测
        self.peak_contribution = 0.0      # 峰值贡献度
        self.optimization_potential = 0.0  # 优化潜力
        
        # 新增：多目标分析
        self.parallelism_score = 0.0     # 并行度分数
        self.execution_priority = 0.0    # 执行优先级
        self.resource_efficiency = 0.0    # 资源效率


class AdvancedScheduler:
    """基于最前沿技术的高级调度算法"""
    
    def __init__(self, graph_file: str):
        """初始化调度器"""
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Tuple[int, int]] = []
        self.adj_list: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adj_list: Dict[int, List[int]] = defaultdict(list)
        self.in_degree: Dict[int, int] = defaultdict(int)
        
        # 高级分析数据结构
        self.buffer_analysis: Dict[int, AdvancedBufferAnalysis] = {}
        self.buf_to_alloc: Dict[int, int] = {}
        self.buf_to_free: Dict[int, int] = {}
        
        # 调度状态
        self.current_time = 0
        self.memory_pressure_threshold = 0.8  # 内存压力阈值
        
        # 新增：增强内存感知
        self.memory_history = []  # 内存使用历史
        self.peak_prediction_window = 5  # 峰值预测窗口
        self.adaptive_threshold = 0.8  # 自适应阈值
        self.peak_avoidance_enabled = True  # 峰值避免开关
        self.memory_trend = 0.0  # 内存使用趋势
        
        # 新增：执行效率优化
        self.execution_units = {}  # 执行单元使用情况
        self.parallelism_analysis = {}  # 并行度分析
        self.resource_utilization = 0.0  # 资源利用率
        self.critical_path_nodes = set()  # 关键路径节点
        
        # 新增：L0缓存约束
        self.l0_live_buffers: Dict[str, Optional[int]] = {"L0A": None, "L0B": None, "L0C": None}
        self.stalled_l0_allocs: Dict[str, List[int]] = {"L0A": [], "L0B": [], "L0C": []}
        
        # 移除全局优化以保持高效性
        
        self._load_graph(graph_file)
        self._build_graph()
        self._analyze_buffers()
        self._compute_critical_paths()
    
    def _load_graph(self, graph_file: str):
        """加载计算图数据"""
        with open(graph_file, 'r') as f:
            data = json.load(f)
        
        # 加载节点
        for node_data in data["Nodes"]:
            node = Node(node_data)
            self.nodes[node.id] = node
        
        # 加载边
        self.edges = [(src, dst) for src, dst in data["Edges"]]
    
    def _build_graph(self):
        """构建邻接表和计算入度"""
        for src, dst in self.edges:
            self.adj_list[src].append(dst)
            self.reverse_adj_list[dst].append(src)
            self.in_degree[dst] += 1
        
        # 确保所有节点都有入度记录
        for node_id in self.nodes:
            if node_id not in self.in_degree:
                self.in_degree[node_id] = 0
    
    def _analyze_buffers(self):
        """深度分析缓冲区特性"""
        # 1. 建立基本映射
        for node_id, node in self.nodes.items():
            if node.op == "ALLOC" and node.is_l1_or_ub_cache():
                self.buf_to_alloc[node.buf_id] = node_id
            elif node.op == "FREE" and node.is_l1_or_ub_cache():
                self.buf_to_free[node.buf_id] = node_id
        
        # 2. 创建高级分析对象
        for buf_id in self.buf_to_alloc:
            if buf_id in self.buf_to_free:
                alloc_node = self.buf_to_alloc[buf_id]
                free_node = self.buf_to_free[buf_id]
                size = self.nodes[alloc_node].size
                
                analysis = AdvancedBufferAnalysis(buf_id, alloc_node, free_node, size)
                self.buffer_analysis[buf_id] = analysis
        
        # 3. 分析使用模式
        self._analyze_usage_patterns()
        
        # 4. 计算JIT时机
        self._compute_jit_timing()
        
        # 5. 评估内存压力
        self._compute_memory_pressure()
        
        # 6. 精细生存期分析
        self._analyze_detailed_lifetimes()
        
        # 7. 预测性分析
        self._compute_predictive_analysis()
        
        # 8. 多目标优化分析
        self._compute_multi_objective_analysis()
        
        # 9. 执行效率分析
        self._analyze_execution_efficiency()
        
        # 10. 关键路径识别
        self._identify_critical_paths()

        # 11. Pre-compute descendant info for lookahead heuristic
        self._precompute_descendants_and_memory()
    
    def _precompute_descendants_and_memory(self):
        """Pre-computes descendant counts and memory impact for each node."""
        self.descendant_counts = defaultdict(int)
        self.descendant_free_memory = defaultdict(int)
        
        # We need to traverse from the leaves backwards (reverse topological order)
        reverse_topo_order = reversed(self._topological_sort())
        
        for node_id in reverse_topo_order:
            node = self.nodes[node_id]
            
            # A node is its own descendant for calculation purposes
            count = 1
            free_mem = 0
            if node.op == "FREE" and node.is_l1_or_ub_cache():
                free_mem = node.size
                
            for successor_id in self.adj_list[node_id]:
                count += self.descendant_counts[successor_id]
                free_mem += self.descendant_free_memory[successor_id]
            
            self.descendant_counts[node_id] = count
            self.descendant_free_memory[node_id] = free_mem
    
    def _analyze_usage_patterns(self):
        """分析缓冲区使用模式"""
        for node_id, node in self.nodes.items():
            if not node.is_cache_node() and hasattr(node, 'bufs'):
                for buf_id in node.bufs:
                    if buf_id in self.buffer_analysis:
                        analysis = self.buffer_analysis[buf_id]
                        analysis.usage_nodes.append(node_id)
                        
                        # 更新首次和最后使用
                        if analysis.first_usage is None:
                            analysis.first_usage = node_id
                        analysis.last_usage = node_id
    
    def _compute_jit_timing(self):
        """计算JIT分配和早期释放时机"""
        for buf_id, analysis in self.buffer_analysis.items():
            if not analysis.usage_nodes:
                continue
            
            # 计算最晚分配时间：从第一个使用节点回溯
            if analysis.first_usage:
                latest_time = self._compute_latest_alloc_time(analysis.first_usage, analysis.alloc_node)
                analysis.latest_alloc_time = latest_time
            
            # 计算最早释放时间：从最后一个使用节点前推
            if analysis.last_usage:
                earliest_time = self._compute_earliest_free_time(analysis.last_usage, analysis.free_node)
                analysis.earliest_free_time = earliest_time
    
    def _compute_latest_alloc_time(self, usage_node: int, alloc_node: int) -> int:
        """计算最晚分配时间"""
        # 使用反向BFS找到从ALLOC到首次使用的最短路径
        queue = deque([(usage_node, 0)])
        visited = {usage_node}
        
        while queue:
            current, dist = queue.popleft()
            
            if current == alloc_node:
                return max(0, dist - 1)  # 至少在使用前1步分配
            
            for pred in self.reverse_adj_list[current]:
                if pred not in visited:
                    visited.add(pred)
                    queue.append((pred, dist + 1))
        
        return 0  # 如果找不到路径，保守起见早期分配
    
    def _compute_earliest_free_time(self, usage_node: int, free_node: int) -> int:
        """计算最早释放时间"""
        # 使用正向BFS找到从最后使用到FREE的最短路径
        queue = deque([(usage_node, 0)])
        visited = {usage_node}
        
        while queue:
            current, dist = queue.popleft()
            
            if current == free_node:
                return dist + 1  # 在最后使用后1步释放
            
            for succ in self.adj_list[current]:
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, dist + 1))
        
        return float('inf')  # 如果找不到路径，延迟释放
    
    def _analyze_detailed_lifetimes(self):
        """精细生存期分析：检测空闲窗口和使用密度"""
        for buf_id, analysis in self.buffer_analysis.items():
            if not analysis.usage_nodes:
                continue
            
            # 分析使用间隔
            usage_positions = []
            for node_id in analysis.usage_nodes:
                # 计算节点在图中的位置（简化版：使用节点ID排序）
                usage_positions.append(node_id)
            
            usage_positions.sort()
            analysis.usage_intervals = usage_positions
            
            # 检测空闲窗口
            if len(usage_positions) > 1:
                idle_windows = []
                for i in range(len(usage_positions) - 1):
                    gap = usage_positions[i + 1] - usage_positions[i]
                    if gap > 1:  # 存在空闲窗口
                        idle_windows.append((usage_positions[i], usage_positions[i + 1], gap - 1))
                analysis.idle_windows = idle_windows
            
            # 计算使用密度
            total_lifetime = analysis.free_node - analysis.alloc_node if analysis.free_node != float('inf') else len(self.nodes)
            usage_density = len(analysis.usage_nodes) / max(1, total_lifetime)
            analysis.usage_density = usage_density
            
            # 计算生存期效率
            if analysis.idle_windows:
                total_idle_time = sum(window[2] for window in analysis.idle_windows)
                analysis.lifetime_efficiency = 1.0 - (total_idle_time / max(1, total_lifetime))
            else:
                analysis.lifetime_efficiency = 1.0
    
    def _compute_predictive_analysis(self):
        """预测性分析：预测未来内存需求和优化潜力"""
        # 计算全局内存需求趋势
        total_memory = sum(analysis.size for analysis in self.buffer_analysis.values())
        
        for buf_id, analysis in self.buffer_analysis.items():
            # 预测未来内存需求
            # 基于缓冲区大小、使用频率和关键路径权重
            future_demand = (analysis.size * analysis.usage_density * 
                           (1 + analysis.critical_path_weight))
            analysis.future_memory_demand = future_demand
            
            # 计算峰值贡献度
            # 大缓冲区、高使用频率的缓冲区更容易造成峰值
            peak_contribution = (analysis.size / max(1, total_memory)) * analysis.usage_density
            analysis.peak_contribution = peak_contribution
            
            # 计算优化潜力
            # 生存期效率低、空闲窗口多的缓冲区优化潜力大
            optimization_potential = (1 - analysis.lifetime_efficiency) * len(analysis.idle_windows)
            analysis.optimization_potential = optimization_potential
    
    def _compute_multi_objective_analysis(self):
        """多目标优化分析：平衡内存、并行度和执行效率"""
        for buf_id, analysis in self.buffer_analysis.items():
            # 并行度分数：基于使用节点的分布
            if len(analysis.usage_nodes) > 1:
                # 计算使用节点之间的平均距离
                usage_positions = analysis.usage_intervals
                if len(usage_positions) > 1:
                    avg_distance = sum(usage_positions[i+1] - usage_positions[i] 
                                     for i in range(len(usage_positions)-1)) / (len(usage_positions)-1)
                    analysis.parallelism_score = 1.0 / max(1, avg_distance)
                else:
                    analysis.parallelism_score = 0.0
            else:
                analysis.parallelism_score = 0.0
            
            # 执行优先级：综合考虑关键路径和内存压力
            execution_priority = (analysis.critical_path_weight * 0.4 + 
                                analysis.memory_pressure_score * 0.3 +
                                analysis.usage_density * 0.3)
            analysis.execution_priority = execution_priority
            
            # 资源效率：内存使用效率
            if analysis.lifetime_efficiency > 0:
                resource_efficiency = analysis.usage_density / analysis.lifetime_efficiency
            else:
                resource_efficiency = analysis.usage_density
            analysis.resource_efficiency = resource_efficiency
    
    def _analyze_execution_efficiency(self):
        """分析执行效率：并行度、资源利用率等"""
        # 分析执行单元使用情况
        for node_id, node in self.nodes.items():
            if not node.is_cache_node() and hasattr(node, 'pipe'):
                pipe = node.pipe
                if pipe not in self.execution_units:
                    self.execution_units[pipe] = []
                self.execution_units[pipe].append(node_id)
        
        # 计算并行度分析
        for pipe, nodes in self.execution_units.items():
            if len(nodes) > 1:
                # 计算节点间的依赖关系
                dependencies = 0
                for node_id in nodes:
                    for successor in self.adj_list[node_id]:
                        if successor in nodes:
                            dependencies += 1
                
                # 并行度 = 1 - (依赖数 / 最大可能依赖数)
                max_deps = len(nodes) * (len(nodes) - 1)
                parallelism = 1.0 - (dependencies / max(1, max_deps))
                self.parallelism_analysis[pipe] = parallelism
            else:
                self.parallelism_analysis[pipe] = 0.0
        
        # 计算资源利用率
        total_cycles = sum(getattr(node, 'cycles', 1) for node in self.nodes.values() if not node.is_cache_node())
        if total_cycles > 0:
            # 基于关键路径计算资源利用率
            critical_cycles = sum(getattr(self.nodes[node_id], 'cycles', 1) 
                                for node_id in self.critical_path_nodes)
            self.resource_utilization = critical_cycles / total_cycles
    
    def _identify_critical_paths(self):
        """识别关键路径节点"""
        # 使用最长路径算法识别关键路径
        longest_path = {}
        
        # 初始化
        for node_id in self.nodes:
            longest_path[node_id] = 0
        
        # 拓扑排序计算最长路径
        topo_order = self._topological_sort()
        
        for node_id in topo_order:
            node = self.nodes[node_id]
            current_length = longest_path[node_id]
            
            for successor in self.adj_list[node_id]:
                new_length = current_length + getattr(node, 'cycles', 1)
                longest_path[successor] = max(longest_path[successor], new_length)
        
        # 找到最长路径长度
        max_path_length = max(longest_path.values()) if longest_path else 1
        
        # 识别关键路径节点（路径长度接近最大值的节点）
        threshold = max_path_length * 0.8  # 80%阈值
        for node_id, path_length in longest_path.items():
            if path_length >= threshold:
                self.critical_path_nodes.add(node_id)
    
    def _get_execution_efficiency_priority(self, node: Node, ready_nodes: Set[int]) -> float:
        """获取执行效率优先级"""
        if node.is_cache_node():
            return 0.0
        
        # 基础执行效率
        base_efficiency = getattr(node, 'cycles', 1)
        
        # 并行度奖励
        parallelism_bonus = 0.0
        if hasattr(node, 'pipe') and node.pipe in self.parallelism_analysis:
            parallelism_bonus = self.parallelism_analysis[node.pipe] * 10
        
        # 关键路径奖励
        critical_path_bonus = 0.0
        if node.id in self.critical_path_nodes:
            critical_path_bonus = 20
        
        # 资源利用率奖励
        resource_bonus = self.resource_utilization * 5
        
        return base_efficiency + parallelism_bonus + critical_path_bonus + resource_bonus
    
    def _optimize_resource_allocation(self, ready_nodes: Set[int]) -> Set[int]:
        """优化资源分配：选择最优的节点组合"""
        if not ready_nodes:
            return ready_nodes
        
        # 按执行效率排序
        efficiency_scores = {}
        for node_id in ready_nodes:
            node = self.nodes[node_id]
            efficiency_scores[node_id] = self._get_execution_efficiency_priority(node, ready_nodes)
        
        # 选择效率最高的节点
        sorted_nodes = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 考虑资源冲突，选择最优组合
        selected_nodes = set()
        used_pipes = set()
        
        for node_id, score in sorted_nodes:
            node = self.nodes[node_id]
            
            # 检查资源冲突
            if hasattr(node, 'pipe') and node.pipe in used_pipes:
                continue  # 跳过冲突的节点
            
            selected_nodes.add(node_id)
            if hasattr(node, 'pipe'):
                used_pipes.add(node.pipe)
        
        return selected_nodes if selected_nodes else ready_nodes
    
    def _compute_memory_pressure(self):
        """计算内存压力分数"""
        total_memory = sum(analysis.size for analysis in self.buffer_analysis.values())
        
        for buf_id, analysis in self.buffer_analysis.items():
            # 基于大小和使用频率的压力分数
            usage_density = len(analysis.usage_nodes) / max(1, len(self.nodes))
            size_ratio = analysis.size / max(1, total_memory)
            
            # 综合压力分数
            analysis.memory_pressure_score = (size_ratio * 0.6 + usage_density * 0.4)
    
    def _compute_critical_paths(self):
        """计算关键路径权重"""
        # 使用拓扑排序计算最长路径
        longest_path = {}
        
        # 初始化
        for node_id in self.nodes:
            longest_path[node_id] = 0
        
        # 拓扑排序计算最长路径
        topo_order = self._topological_sort()
        
        for node_id in topo_order:
            node = self.nodes[node_id]
            current_length = longest_path[node_id]
            
            for successor in self.adj_list[node_id]:
                new_length = current_length + getattr(node, 'cycles', 1)
                longest_path[successor] = max(longest_path[successor], new_length)
        
        # 计算关键路径权重
        max_path_length = max(longest_path.values()) if longest_path else 1
        
        for buf_id, analysis in self.buffer_analysis.items():
            alloc_path = longest_path.get(analysis.alloc_node, 0)
            free_path = longest_path.get(analysis.free_node, 0)
            
            # 关键路径权重
            analysis.critical_path_weight = (alloc_path + free_path) / (2 * max_path_length)
    
    def _topological_sort(self) -> List[int]:
        """拓扑排序"""
        in_deg = self.in_degree.copy()
        queue = deque([node_id for node_id in self.nodes if in_deg[node_id] == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for successor in self.adj_list[node_id]:
                in_deg[successor] -= 1
                if in_deg[successor] == 0:
                    queue.append(successor)
        
        return result
    
    def _get_advanced_priority(self, node: Node, current_v_stay: int, ready_nodes: Set[int], 
                              total_memory_capacity: int) -> Tuple[int, float, float, float, int]:
        """
        增强的动态优先级计算，融合多种策略和预测性分析
        
        返回：(主要类别, 内存压力权重, 关键路径权重, 多目标权重, 节点ID)
        """
        # 计算当前内存压力
        memory_pressure = current_v_stay / max(1, total_memory_capacity)
        
        # 预测未来内存需求
        future_demand = self._predict_future_memory_demand(ready_nodes)
        
        # 计算多目标权重
        multi_objective_weight = self._calculate_multi_objective_weight(node, ready_nodes, memory_pressure)
        
        if node.op == "FREE" and node.is_l1_or_ub_cache():
            # FREE节点：最高优先级，考虑释放的价值和时机
            buf_id = node.buf_id
            if buf_id in self.buffer_analysis:
                analysis = self.buffer_analysis[buf_id]
                
                # 动态释放价值：考虑当前压力、未来需求和优化潜力
                release_value = (memory_pressure * analysis.memory_pressure_score * 
                               (1 + future_demand) * (1 + analysis.optimization_potential))
                
                # 关键路径权重
                critical_weight = analysis.critical_path_weight
                
                return (0, -release_value, -critical_weight, -multi_objective_weight, node.id)
            else:
                return (0, -node.size, 0, -multi_objective_weight, node.id)
        
        elif not node.is_cache_node():
            # 操作节点：考虑解锁能力和执行效率
            # 考虑能解锁的FREE节点价值
            free_unlock_value = self._calculate_enhanced_free_unlock_value(node.id, ready_nodes, memory_pressure, future_demand)
            
            # 考虑关键路径和执行优先级
            critical_weight = self._get_enhanced_node_critical_weight(node.id)
            
            # 考虑并行度和资源效率
            execution_efficiency = self._calculate_execution_efficiency(node.id, ready_nodes)
            
            return (1, -free_unlock_value, -critical_weight, -execution_efficiency, node.id)
        
        elif node.op == "ALLOC" and node.is_l1_or_ub_cache():
            # ALLOC节点：应用智能JIT策略
            buf_id = node.buf_id
            
            if buf_id in self.buffer_analysis:
                analysis = self.buffer_analysis[buf_id]
                
                # 智能延迟判断：考虑更多因素
                if self._can_smart_delay_allocation(node.id, ready_nodes, analysis, memory_pressure, future_demand):
                    # 延迟分配 - 降低优先级
                    delay_penalty = (memory_pressure * analysis.memory_pressure_score * 
                                   (1 + analysis.peak_contribution))
                    return (3, delay_penalty, analysis.size, multi_objective_weight, node.id)
                else:
                    # 必须现在分配 - 提高优先级
                    urgency_weight = (analysis.memory_pressure_score + analysis.critical_path_weight + 
                                    analysis.execution_priority)
                    return (1, urgency_weight, analysis.size, -multi_objective_weight, node.id)
            else:
                # 没有分析信息的ALLOC节点
                return (3, memory_pressure, node.size, multi_objective_weight, node.id)
        
        else:
            # L0级缓存节点
            return (2, 0, 0, 0, node.id)
    
    def _calculate_free_unlock_value(self, node_id: int, ready_nodes: Set[int], memory_pressure: float) -> float:
        """计算执行该节点后能解锁的FREE节点价值"""
        total_value = 0.0
        
        for successor in self.adj_list[node_id]:
            successor_node = self.nodes[successor]
            
            if successor_node.op == "FREE" and successor_node.is_l1_or_ub_cache():
                # 检查FREE节点是否接近ready
                remaining_deps = sum(1 for pred in self.reverse_adj_list[successor] 
                                   if pred not in ready_nodes and pred != node_id)
                
                if remaining_deps == 0:
                    # 该FREE节点会被解锁
                    buf_id = successor_node.buf_id
                    if buf_id in self.buffer_analysis:
                        analysis = self.buffer_analysis[buf_id]
                        # 内存压力越高，释放价值越大
                        value = successor_node.size * (1 + memory_pressure) * analysis.memory_pressure_score
                        total_value += value
                    else:
                        total_value += successor_node.size * memory_pressure
        
        return total_value
    
    def _predict_future_memory_demand(self, ready_nodes: Set[int]) -> float:
        """预测未来内存需求"""
        future_demand = 0.0
        
        for node_id in ready_nodes:
            node = self.nodes[node_id]
            if node.op == "ALLOC" and node.is_l1_or_ub_cache():
                buf_id = node.buf_id
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    future_demand += analysis.future_memory_demand
        
        return future_demand / max(1, len(ready_nodes))
    
    def _calculate_multi_objective_weight(self, node: Node, ready_nodes: Set[int], memory_pressure: float) -> float:
        """计算多目标权重"""
        if node.op == "FREE" and node.is_l1_or_ub_cache():
            buf_id = node.buf_id
            if buf_id in self.buffer_analysis:
                analysis = self.buffer_analysis[buf_id]
                return (analysis.parallelism_score * 0.3 + 
                       analysis.resource_efficiency * 0.4 + 
                       analysis.execution_priority * 0.3)
        elif not node.is_cache_node():
            # 操作节点的多目标权重
            return self._calculate_execution_efficiency(node.id, ready_nodes)
        elif node.op == "ALLOC" and node.is_l1_or_ub_cache():
            buf_id = node.buf_id
            if buf_id in self.buffer_analysis:
                analysis = self.buffer_analysis[buf_id]
                return (analysis.parallelism_score * 0.2 + 
                       analysis.resource_efficiency * 0.3 + 
                       analysis.execution_priority * 0.5)
        
        return 0.0
    
    def _calculate_enhanced_free_unlock_value(self, node_id: int, scheduled_nodes: list, 
                                           memory_pressure: float, future_demand: float) -> float:
        """增强的FREE解锁价值计算"""
        total_value = 0.0
        scheduled_nodes_set = set(scheduled_nodes)
        
        for successor in self.adj_list[node_id]:
            successor_node = self.nodes[successor]
            
            if successor_node.op == "FREE" and successor_node.is_l1_or_ub_cache():
                # 检查FREE节点是否会被此节点解锁
                is_unlocked = True
                for pred in self.reverse_adj_list[successor]:
                    if pred != node_id and pred not in scheduled_nodes_set:
                        is_unlocked = False
                        break
                
                if is_unlocked:
                    # 该FREE节点会被解锁
                    buf_id = successor_node.buf_id
                    if buf_id in self.buffer_analysis:
                        analysis = self.buffer_analysis[buf_id]
                        # 增强的价值计算：考虑未来需求和优化潜力
                        value = (successor_node.size * (1 + memory_pressure + future_demand) * 
                               analysis.memory_pressure_score * (1 + analysis.optimization_potential))
                        total_value += value
                    else:
                        total_value += successor_node.size * (memory_pressure + future_demand)
        
        return total_value
    
    def _get_enhanced_node_critical_weight(self, node_id: int) -> float:
        """增强的节点关键路径权重"""
        node = self.nodes[node_id]
        base_weight = getattr(node, 'cycles', 1)
        
        # 如果节点使用了关键缓冲区，增加权重
        if hasattr(node, 'bufs'):
            for buf_id in node.bufs:
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    base_weight += (analysis.critical_path_weight * 10 + 
                                  analysis.execution_priority * 5)
        
        return base_weight
    
    def _calculate_execution_efficiency(self, node_id: int, ready_nodes: Set[int]) -> float:
        """计算执行效率"""
        node = self.nodes[node_id]
        
        # 基础执行效率
        base_efficiency = getattr(node, 'cycles', 1)
        
        # 考虑并行度
        parallelism_bonus = 0.0
        if hasattr(node, 'bufs'):
            for buf_id in node.bufs:
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    parallelism_bonus += analysis.parallelism_score
        
        # 考虑资源效率
        resource_efficiency = 0.0
        if hasattr(node, 'bufs'):
            for buf_id in node.bufs:
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    resource_efficiency += analysis.resource_efficiency
        
        return base_efficiency + parallelism_bonus + resource_efficiency
    
    def _can_smart_delay_allocation(self, alloc_node_id: int, ready_nodes: Set[int], 
                                  analysis: AdvancedBufferAnalysis, memory_pressure: float, 
                                  future_demand: float) -> bool:
        """智能延迟分配判断"""
        if not analysis.first_usage:
            return True  # 没有使用节点，可以延迟
        
        # 检查首次使用节点是否即将ready
        first_usage = analysis.first_usage
        
        # 计算到首次使用的距离
        unready_predecessors = 0
        for pred in self.reverse_adj_list[first_usage]:
            if pred not in ready_nodes:
                unready_predecessors += 1
        
        # 智能延迟判断：考虑内存压力、未来需求和优化潜力
        delay_threshold = (analysis.latest_alloc_time * 
                          (1 + memory_pressure) * 
                          (1 + future_demand) * 
                          (1 + analysis.optimization_potential))
        
        return unready_predecessors > delay_threshold
    
    def _update_memory_awareness(self, current_v_stay: int, total_capacity: int):
        """更新内存感知状态"""
        # 记录内存使用历史
        self.memory_history.append(current_v_stay)
        if len(self.memory_history) > self.peak_prediction_window * 2:
            self.memory_history.pop(0)
        
        # 计算内存使用趋势
        if len(self.memory_history) >= 3:
            recent_avg = sum(self.memory_history[-3:]) / 3
            older_avg = sum(self.memory_history[-6:-3]) / 3 if len(self.memory_history) >= 6 else recent_avg
            self.memory_trend = (recent_avg - older_avg) / max(1, older_avg)
        
        # 自适应调整阈值
        self._adapt_memory_threshold(current_v_stay, total_capacity)
    
    def _adapt_memory_threshold(self, current_v_stay: int, total_capacity: int):
        """自适应调整内存阈值"""
        current_pressure = current_v_stay / max(1, total_capacity)
        
        # 基于当前压力和趋势调整阈值
        if self.memory_trend > 0.1:  # 内存使用上升趋势
            self.adaptive_threshold = min(0.9, self.adaptive_threshold + 0.05)
        elif self.memory_trend < -0.1:  # 内存使用下降趋势
            self.adaptive_threshold = max(0.5, self.adaptive_threshold - 0.05)
        
        # 如果当前压力很高，降低阈值
        if current_pressure > 0.8:
            self.adaptive_threshold = max(0.6, self.adaptive_threshold - 0.1)
    
    def _predict_memory_peak(self, ready_nodes: Set[int], current_v_stay: int) -> float:
        """预测内存峰值"""
        if not self.peak_avoidance_enabled:
            return current_v_stay
        
        # 计算即将分配的内存
        pending_allocation = 0
        for node_id in ready_nodes:
            node = self.nodes[node_id]
            if node.op == "ALLOC" and node.is_l1_or_ub_cache():
                pending_allocation += node.size
        
        # 预测峰值
        predicted_peak = current_v_stay + pending_allocation
        
        # 考虑历史趋势
        if self.memory_trend > 0:
            predicted_peak *= (1 + self.memory_trend)
        
        return predicted_peak
    
    def _should_avoid_peak(self, node: Node, current_v_stay: int, total_capacity: int, 
                          ready_nodes: Set[int]) -> bool:
        """判断是否应该避免峰值"""
        if not self.peak_avoidance_enabled:
            return False
        
        # 预测执行该节点后的内存峰值
        predicted_peak = self._predict_memory_peak(ready_nodes, current_v_stay)
        
        # 如果预测峰值过高，且节点是ALLOC，则延迟
        if (node.op == "ALLOC" and node.is_l1_or_ub_cache() and 
            predicted_peak > total_capacity * self.adaptive_threshold):
            return True
        
        return False
    
    def _get_enhanced_memory_priority(self, node: Node, current_v_stay: int, 
                                    total_capacity: int, ready_nodes: Set[int]) -> float:
        """获取增强的内存优先级"""
        # 基础内存压力
        memory_pressure = current_v_stay / max(1, total_capacity)
        
        # 峰值避免权重
        peak_avoidance_weight = 0.0
        if self._should_avoid_peak(node, current_v_stay, total_capacity, ready_nodes):
            peak_avoidance_weight = 1.0
        
        # 趋势权重
        trend_weight = abs(self.memory_trend) if self.memory_trend > 0 else 0
        
        # 综合内存优先级
        return memory_pressure + peak_avoidance_weight + trend_weight
    
    def _get_node_critical_weight(self, node_id: int) -> float:
        """获取节点的关键路径权重"""
        # 简化版：基于节点的cycles
        node = self.nodes[node_id]
        base_weight = getattr(node, 'cycles', 1)
        
        # 如果节点使用了关键缓冲区，增加权重
        if hasattr(node, 'bufs'):
            for buf_id in node.bufs:
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    base_weight += analysis.critical_path_weight * 10
        
        return base_weight
    
    def _can_delay_allocation(self, alloc_node_id: int, scheduled_nodes: list, 
                                analysis: AdvancedBufferAnalysis) -> bool:
        """
        判断是否可以延迟分配.
        仅当此 ALLOC 节点是其 buffer 第一个使用者的唯一未完成的依赖项时，才认为它不可延迟.
        """
        if not analysis.first_usage:
            return True  # 没有使用节点，可以延迟

        first_usage_node_id = analysis.first_usage
        scheduled_nodes_set = set(scheduled_nodes)
        
        # 检查 first_usage 节点是否已调度 (虽然不太可能)
        if first_usage_node_id in scheduled_nodes_set:
            return False

        # 检查此 ALLOC 节点是否是 first_usage 的唯一未调度前驱
        is_urgent = True
        for pred_id in self.reverse_adj_list[first_usage_node_id]:
            if pred_id != alloc_node_id and pred_id not in scheduled_nodes_set:
                # 找到了另一个未被调度的前驱节点.
                # 因此，当前 ALLOC 节点不是紧急的，可以延迟.
                is_urgent = False
                break
                
        return not is_urgent
    
    def schedule(self) -> Tuple[List[int], int]:
        """
        执行高级调度算法
        
        Returns:
            Tuple[List[int], int]: (调度序列, max_V_stay)
        """
        schedule_sequence = []
        current_in_degree = self.in_degree.copy()
        current_v_stay = 0
        max_v_stay = 0
        last_scheduled_node_id: Optional[int] = None
        
        # 估算总内存容量（用于压力计算）
        total_capacity = sum(analysis.size for analysis in self.buffer_analysis.values()) * 2
        
        # 可执行节点集合
        ready_queue: List[Tuple[Tuple[int, float, float, int], int]] = []
        
        def _add_to_ready_queue(node_id: int):
            """Helper to check L0 constraints and add to ready queue."""
            node = self.nodes[node_id]
            
            # 检查L0约束
            if node.op == "ALLOC" and node.cache_type in self.l0_live_buffers:
                cache_type = node.cache_type
                if self.l0_live_buffers[cache_type] is not None:
                    # L0缓存被占用，将节点置于停滞状态
                    self.stalled_l0_allocs[cache_type].append(node_id)
                    return
            
            # 如果不冲突，则计算优先级并加入队列
            priority = self._get_problem1_priority(node, current_v_stay, schedule_sequence)
            heapq.heappush(ready_queue, (priority, node_id))

        # 初始化
        for node_id in self.nodes:
            if current_in_degree[node_id] == 0:
                _add_to_ready_queue(node_id)
        
        step = 0
        while ready_queue:
            # 选择最优节点
            priority, node_id = heapq.heappop(ready_queue)
            node = self.nodes[node_id]
            
            # Re-check L0 constraint before scheduling to handle race conditions
            if node.op == "ALLOC" and node.cache_type in self.l0_live_buffers:
                cache_type = node.cache_type
                if self.l0_live_buffers[cache_type] is not None:
                    # The cache was free when this was added to the queue, but another node took it. Re-stall.
                    self.stalled_l0_allocs[cache_type].append(node_id)
                    continue

            schedule_sequence.append(node_id)
            last_scheduled_node_id = node_id
            
            # 更新内存使用
            delta = node.memory_delta()
            current_v_stay += delta
            max_v_stay = max(max_v_stay, current_v_stay)
            
            # 更新内存感知状态
            self._update_memory_awareness(current_v_stay, total_capacity)
            
            # 更新L0缓存状态
            if node.op == "ALLOC" and node.cache_type in self.l0_live_buffers:
                self.l0_live_buffers[node.cache_type] = node.buf_id
            elif node.op == "FREE" and node.cache_type in self.l0_live_buffers:
                cache_type = node.cache_type
                if self.l0_live_buffers.get(cache_type) == node.buf_id:
                    self.l0_live_buffers[cache_type] = None
                    # 释放L0缓存后，检查并重新激活停滞的ALLOC节点
                    stalled_nodes = self.stalled_l0_allocs[cache_type]
                    self.stalled_l0_allocs[cache_type] = []
                    for stalled_id in stalled_nodes:
                        _add_to_ready_queue(stalled_id)

            # 更新调度状态
            step += 1
            self.current_time = step
            
            # 更新后继节点
            for successor in self.adj_list[node_id]:
                current_in_degree[successor] -= 1
                if current_in_degree[successor] == 0:
                    _add_to_ready_queue(successor)
        
        return schedule_sequence, max_v_stay
    
    def _get_problem1_priority(self, node: Node, current_v_stay: int, scheduled_nodes: list) -> Tuple[int, float, float]:
        """
        A simplified priority function focused purely on minimizing max(V_stay) for Problem 1.
        Uses a lookahead heuristic, JIT allocation, and correct L0 handling.
        
        Priority is a tuple: (category, score, tie_breaker)
        Lower is better.
        Categories:
         -1: L0 FREE (highest priority to unlock resource)
          0: L1/UB FREE
          1: Operation nodes
          2: Urgent ALLOCs (L0 and non-delayable L1/UB)
          3: Delayable L1/UB ALLOCs
          4: Default/Catch-all
        """
        critical_path_tiebreaker = -self._get_enhanced_node_critical_weight(node.id)

        # Category -1: L0 FREE nodes (highest priority to unlock the resource)
        if node.op == "FREE" and node.cache_type in self.l0_live_buffers:
            return (-1, 0.0, critical_path_tiebreaker)
            
        # Category 0: L1/UB FREE nodes
        if node.op == "FREE" and node.is_l1_or_ub_cache():
            return (0, -float(node.size), critical_path_tiebreaker)

        # Category 1: Operation nodes (Lookahead Heuristic)
        if not node.is_cache_node():
            desc_count = self.descendant_counts.get(node.id, 1)
            free_mem = self.descendant_free_memory.get(node.id, 0)
            lookahead_score = free_mem / max(1, desc_count)
            return (1, -lookahead_score, critical_path_tiebreaker)

        # ALLOC nodes
        if node.op == "ALLOC":
            if node.is_l1_or_ub_cache():
                buf_id = node.buf_id
                if buf_id in self.buffer_analysis:
                    analysis = self.buffer_analysis[buf_id]
                    can_delay = self._can_delay_allocation(node.id, scheduled_nodes, analysis)
                    
                    if can_delay:
                        # Category 3: Delayable L1/UB ALLOCs
                        return (3, float(node.size), critical_path_tiebreaker)
                    else:
                        # Category 2: Urgent L1/UB ALLOCs
                        user_crit_weight = 0.0
                        if analysis.first_usage:
                             user_crit_weight = self._get_enhanced_node_critical_weight(analysis.first_usage)
                        score = float(node.size) / (1 + user_crit_weight * 0.1)
                        return (2, score, critical_path_tiebreaker)
            
            elif node.cache_type in self.l0_live_buffers:
                # Category 2: Urgent L0 ALLOCs
                return (2, 0.0, critical_path_tiebreaker)

        # Default for any other case
        return (4, 0.0, critical_path_tiebreaker)

    def validate_schedule(self, schedule: List[int]) -> Tuple[bool, List[str]]:
        """
        验证调度序列的完整性和正确性。

        检查项:
        1. 完整性: 所有节点都被调度且仅调度一次。
        2. 拓扑序: 满足所有依赖边约束。
        3. L0缓存约束: L0A, L0B, L0C上同时最多只有一个缓冲区驻留。

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []

        # 1. 完整性检查
        if len(schedule) != len(self.nodes):
            errors.append(f"调度不完整。预期 {len(self.nodes)} 个节点，实际调度了 {len(schedule)} 个。")
            # 如果节点数不匹配，后续检查可能无意义
            return False, errors

        if set(schedule) != set(self.nodes.keys()):
            missing = set(self.nodes.keys()) - set(schedule)
            extra = set(schedule) - set(self.nodes.keys())
            errors.append(f"调度节点集合与计算图不匹配。缺失: {missing}, 多余: {extra}")
            return False, errors

        # 2. 拓扑序检查
        position = {node_id: i for i, node_id in enumerate(schedule)}
        for src, dst in self.edges:
            if position.get(src, -1) >= position.get(dst, -1):
                errors.append(f"拓扑序错误: 边({src} -> {dst})，位置({position.get(src)} -> {position.get(dst)})")

        # 3. L0缓存约束检查
        live_l0_buffers: Dict[str, Optional[Tuple[int, int]]] = {"L0A": None, "L0B": None, "L0C": None} # value: (buf_id, alloc_node_id)
        for i, node_id in enumerate(schedule):
            node = self.nodes[node_id]
            if node.op == "ALLOC" and node.cache_type in live_l0_buffers:
                cache_type = node.cache_type
                if live_l0_buffers[cache_type] is not None:
                    prev_buf, prev_alloc_node = live_l0_buffers[cache_type]
                    errors.append(
                        f"L0约束冲突: 步骤 {i}, 节点 {node_id} (ALLOC {node.buf_id}) "
                        f"尝试使用 {cache_type}, 但它已被 "
                        f"buffer {prev_buf} (在节点 {prev_alloc_node} 分配) 占用。"
                    )
                live_l0_buffers[cache_type] = (node.buf_id, node.id)
            elif node.op == "FREE" and node.cache_type in live_l0_buffers:
                cache_type = node.cache_type
                if live_l0_buffers.get(cache_type) is not None and live_l0_buffers[cache_type][0] == node.buf_id:
                     live_l0_buffers[cache_type] = None

        is_valid = not errors
        return is_valid, errors

    def calculate_max_v_stay(self, schedule: List[int]) -> int:
        """计算max(V_stay)"""
        v_stay = 0
        max_v_stay = 0
        
        for node_id in schedule:
            node = self.nodes[node_id]
            delta = node.memory_delta()
            v_stay += delta
            max_v_stay = max(max_v_stay, v_stay)
        
        return max_v_stay


def test_advanced_scheduler():
    """测试高级调度算法"""
    test_files = [
        "data/Json_version/Matmul_Case0.json",
        "data/Json_version/Matmul_Case1.json",
        "data/Json_version/FlashAttention_Case0.json",
        "data/Json_version/FlashAttention_Case1.json",
        "data/Json_version/Conv_Case0.json",
        "data/Json_version/Conv_Case1.json"
    ]
    
    print("=== 高级调度算法测试 ===")
    print(f"{'任务名':<25} {'高级算法max(V_stay)':<20}")
    print("-" * 50)
    
    results = {}
    for test_file in test_files:
        try:
            task_name = test_file.split("/")[-1].replace(".json", "")
            
            # 高级算法
            scheduler = AdvancedScheduler(test_file)
            schedule, max_v_stay = scheduler.schedule()
            
            # 验证
            is_valid, errors = scheduler.validate_schedule(schedule)
            
            print(f"{task_name:<25} {max_v_stay:<20,}")
            
            if not is_valid:
                print(f"  ⚠️  调度序列无效 ({len(errors)} 处错误):")
                # 只打印前5个错误以避免刷屏
                for error in errors[:5]:
                    print(f"    - {error}")
            
            results[task_name] = max_v_stay
            
        except Exception as e:
            print(f"{task_name:<25} 执行失败: {e}")
    
    return results


if __name__ == "__main__":
    results = test_advanced_scheduler()
    
    print(f"\n总体内存峰值: {sum(results.values()):,}")
    print(f"平均峰值: {sum(results.values()) // len(results):,}")
