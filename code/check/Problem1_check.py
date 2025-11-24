#!/usr/bin/env python3
"""
通用调度序列验证工具
可以验证任意方法的调度序列结果，只需要配置文件路径即可

使用方法：
1. 修改 SCHEDULE_CONFIGS 配置，指定要验证的方法和文件路径
2. 运行脚本即可对比验证多种方法

主要功能：
- 验证调度序列是否满足题目约束条件
- 计算V_stay值
- 对比多种方法的性能
- 提供详细的验证报告

验证内容：
- 调度序列完整性（包含所有节点，无重复）
- 拓扑序约束（依赖关系满足）
- ALLOC/FREE配对（缓存分配释放匹配）
- 缓冲区生命周期（操作在分配释放之间）
- V_stay计算和对比
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import json
import os
import csv
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import time

# ==================== 配置区域 ====================
# 在这里配置要验证的方法和对应的文件路径模式
SCHEDULE_CONFIGS = {
    # 方法名: (文件夹路径, 文件名模式)
    # 文件名模式中 {case} 会被替换为具体的测试用例名
    "初始贪心": ("初始贪心", "{case}/{case}_调度序列.txt"),
    "改进贪心": ("改进贪心", "{case}.txt"),
    # 可以添加更多方法，例如：
    "机理建模": ("机理建模序列/Problem1", "{case}_schedule.txt"),
    "优化算法": ("优化算法序列", "{case}/{case}_schedule.txt"),
    "Problem1_Global_Optimized": ("Problem1_Global_Optimized", "{case}_schedule.txt"),
}

# 测试用例列表
TEST_CASES = [
    "Conv_Case0",
    "Conv_Case1", 
    "Matmul_Case0",
    "Matmul_Case1",
    "FlashAttention_Case0",
    "FlashAttention_Case1"
]

# CSV数据文件路径
CSV_DATA_PATH = "Data/CSV版本"
# ==================== 配置区域结束 ====================

class Node:
    """节点类"""
    def __init__(self, node_data: dict):
        self.id = int(node_data['Id'])
        self.op = node_data['Op']
        
        # 缓存管理节点属性
        if self.op in ['ALLOC', 'FREE']:
            self.buf_id = int(node_data.get('BufId')) if node_data.get('BufId') else None
            self.size = int(node_data.get('Size', 0)) if node_data.get('Size') else 0
            self.cache_type = node_data.get('Type', '')
        else:
            # 操作节点属性
            self.pipe = node_data.get('Pipe', '')
            self.cycles = int(node_data.get('Cycles', 0)) if node_data.get('Cycles') else 0
            # 处理Bufs字段
            if 'BufId' in node_data and node_data.get('BufId'):
                self.bufs = [int(node_data['BufId'])]
            else:
                bufs_str = node_data.get('Bufs', '')
                if bufs_str:
                    bufs_str = bufs_str.strip('"')
                    if bufs_str:
                        try:
                            self.bufs = [int(bufs_str)]
                        except ValueError:
                            self.bufs = [int(x.strip()) for x in bufs_str.split(',') if x.strip()]
                    else:
                        self.bufs = []
                else:
                    self.bufs = []
            self.buf_id = None
            self.size = 0
            self.cache_type = ''

class UniversalValidator:
    """通用调度序列验证器"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)
        
    def load_graph_data_csv(self, case_name: str) -> bool:
        """从CSV文件加载计算图数据"""
        try:
            # 重置数据结构
            self.nodes = {}
            self.edges = []
            self.adjacency_list = defaultdict(list)
            self.reverse_adjacency_list = defaultdict(list)
            
            # 加载节点数据
            nodes_file = f"{CSV_DATA_PATH}/{case_name}_Nodes.csv"
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = Node(row)
                    self.nodes[node.id] = node
            
            # 加载边数据
            edges_file = f"{CSV_DATA_PATH}/{case_name}_Edges.csv"
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    start = int(row['StartNodeId'])
                    end = int(row['EndNodeId'])
                    self.edges.append((start, end))
            
            # 构建邻接表
            for start, end in self.edges:
                self.adjacency_list[start].append(end)
                self.reverse_adjacency_list[end].append(start)
            
            print(f"加载完成: {len(self.nodes)}个节点, {len(self.edges)}条边")
            return True
            
        except Exception as e:
            print(f"加载图数据失败: {e}")
            return False
    
    def load_schedule(self, schedule_file: str) -> List[int]:
        """加载调度序列"""
        try:
            schedule = []
            with open(schedule_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        schedule.append(int(line))
            return schedule
        except Exception as e:
            print(f"加载调度序列失败 ({schedule_file}): {e}")
            return []
    
    def validate_schedule_completeness(self, schedule: List[int], method_name: str) -> bool:
        """验证调度序列完整性"""
        print(f"\n=== 验证调度序列完整性 ({method_name}) ===")
        
        schedule_set = set(schedule)
        node_set = set(self.nodes.keys())
        
        if schedule_set != node_set:
            missing_nodes = node_set - schedule_set
            extra_nodes = schedule_set - node_set
            
            if missing_nodes:
                print(f"❌ 缺失节点: {len(missing_nodes)}个")
            if extra_nodes:
                print(f"❌ 多余节点: {len(extra_nodes)}个")
            return False
        
        if len(schedule) != len(schedule_set):
            duplicates_count = len(schedule) - len(schedule_set)
            print(f"❌ 重复节点: {duplicates_count}个")
            return False
        
        print("✅ 调度序列完整性验证通过")
        return True
    
    def validate_topological_order(self, schedule: List[int], method_name: str) -> bool:
        """验证拓扑序约束"""
        print(f"\n=== 验证拓扑序约束 ({method_name}) ===")
        
        position = {node_id: i for i, node_id in enumerate(schedule)}
        violations = []
        
        for src, dst in self.edges:
            if position[src] >= position[dst]:
                violations.append((src, dst, position[src], position[dst]))
        
        if violations:
            print(f"❌ 发现 {len(violations)} 个拓扑序违反")
            for i, (src, dst, src_pos, dst_pos) in enumerate(violations[:5]):
                print(f"  违反 {i+1}: 节点{src}(位置{src_pos}) -> 节点{dst}(位置{dst_pos})")
            if len(violations) > 5:
                print(f"  ... 还有 {len(violations)-5} 个违反")
            return False
        
        print("✅ 拓扑序约束验证通过")
        return True
    
    def validate_alloc_free_pairing(self, schedule: List[int], method_name: str) -> bool:
        """验证ALLOC和FREE节点配对"""
        print(f"\n=== 验证ALLOC/FREE配对 ({method_name}) ===")
        
        buf_alloc_count = defaultdict(int)
        buf_free_count = defaultdict(int)
        
        for node_id in schedule:
            node = self.nodes[node_id]
            if node.op == 'ALLOC':
                buf_alloc_count[node.buf_id] += 1
            elif node.op == 'FREE':
                buf_free_count[node.buf_id] += 1
        
        violations = []
        all_buf_ids = set(buf_alloc_count.keys()) | set(buf_free_count.keys())
        
        for buf_id in all_buf_ids:
            alloc_count = buf_alloc_count[buf_id]
            free_count = buf_free_count[buf_id]
            
            if alloc_count != free_count:
                violations.append((buf_id, alloc_count, free_count))
        
        if violations:
            print(f"❌ 发现 {len(violations)} 个ALLOC/FREE配对错误")
            for buf_id, alloc_count, free_count in violations[:5]:
                print(f"  BufId {buf_id}: ALLOC={alloc_count}, FREE={free_count}")
            return False
        
        print("✅ ALLOC/FREE配对验证通过")
        return True
    
    def validate_buffer_lifecycle(self, schedule: List[int], method_name: str) -> bool:
        """验证缓冲区生命周期"""
        print(f"\n=== 验证缓冲区生命周期 ({method_name}) ===")
        
        buf_alloc_pos = {}
        buf_free_pos = {}
        
        for i, node_id in enumerate(schedule):
            node = self.nodes[node_id]
            if node.op == 'ALLOC':
                buf_alloc_pos[node.buf_id] = i
            elif node.op == 'FREE':
                buf_free_pos[node.buf_id] = i
        
        violations = []
        
        for node_id in schedule:
            node = self.nodes[node_id]
            if node.op not in ['ALLOC', 'FREE'] and node.bufs:
                node_pos = schedule.index(node_id)
                
                for buf_id in node.bufs:
                    if buf_id in buf_alloc_pos and buf_id in buf_free_pos:
                        alloc_pos = buf_alloc_pos[buf_id]
                        free_pos = buf_free_pos[buf_id]
                        
                        if not (alloc_pos < node_pos < free_pos):
                            violations.append((node_id, buf_id, node_pos, alloc_pos, free_pos))
        
        if violations:
            print(f"❌ 发现 {len(violations)} 个缓冲区生命周期违反")
            for i, (node_id, buf_id, node_pos, alloc_pos, free_pos) in enumerate(violations[:5]):
                print(f"  违反 {i+1}: 节点{node_id}(位置{node_pos})使用BufId{buf_id}, 但ALLOC在{alloc_pos}, FREE在{free_pos}")
            return False
        
        print("✅ 缓冲区生命周期验证通过")
        return True
    
    def calculate_v_stay(self, schedule: List[int], method_name: str) -> Dict:
        """计算V_stay"""
        print(f"\n=== 计算V_stay ({method_name}) ===")
        
        cache_types = set()
        for node in self.nodes.values():
            if node.op in ['ALLOC', 'FREE'] and node.cache_type:
                cache_types.add(node.cache_type)
        
        cache_residency = {cache_type: 0 for cache_type in cache_types}
        max_cache_residency = {cache_type: 0 for cache_type in cache_types}
        
        total_residency = 0
        max_total_residency = 0
        peak_step = 0
        
        alloc_count = 0
        free_count = 0
        
        for step, node_id in enumerate(schedule):
            node = self.nodes[node_id]
            
            if node.op == 'ALLOC':
                alloc_count += 1
                cache_residency[node.cache_type] += node.size
                total_residency += node.size
                
                max_cache_residency[node.cache_type] = max(
                    max_cache_residency[node.cache_type], 
                    cache_residency[node.cache_type]
                )
                
                if total_residency > max_total_residency:
                    max_total_residency = total_residency
                    peak_step = step
                    
            elif node.op == 'FREE':
                free_count += 1
                cache_residency[node.cache_type] -= node.size
                total_residency -= node.size
        
        print(f"ALLOC操作数: {alloc_count}")
        print(f"FREE操作数: {free_count}")
        print(f"最终总驻留: {total_residency}")
        print(f"峰值V_stay: {max_total_residency} (步骤 {peak_step})")
        
        for cache_type in sorted(cache_types):
            print(f"  {cache_type}: {max_cache_residency[cache_type]}")
        
        return {
            'v_stay': max_total_residency,
            'peak_step': peak_step,
            'cache_breakdown': max_cache_residency,
            'alloc_count': alloc_count,
            'free_count': free_count,
            'final_residency': total_residency
        }
    
    def validate_single_method(self, case_name: str, method_name: str, schedule_file: str) -> Dict:
        """验证单个方法的调度序列"""
        print(f"\n{'='*80}")
        print(f"验证 {method_name} 方法: {case_name}")
        print(f"调度文件: {schedule_file}")
        print(f"{'='*80}")
        
        # 加载数据
        if not self.load_graph_data_csv(case_name):
            return {"success": False, "error": "数据加载失败"}
        
        schedule = self.load_schedule(schedule_file)
        if not schedule:
            return {"success": False, "error": "调度序列加载失败"}
        
        print(f"调度序列长度: {len(schedule)}")
        
        # 执行各项验证
        results = {
            "method": method_name,
            "case": case_name,
            "success": True,
            "completeness": False,
            "topological_order": False,
            "alloc_free_pairing": False,
            "buffer_lifecycle": False,
            "v_stay_info": {}
        }
        
        try:
            results["completeness"] = self.validate_schedule_completeness(schedule, method_name)
            results["topological_order"] = self.validate_topological_order(schedule, method_name)
            results["alloc_free_pairing"] = self.validate_alloc_free_pairing(schedule, method_name)
            results["buffer_lifecycle"] = self.validate_buffer_lifecycle(schedule, method_name)
            results["v_stay_info"] = self.calculate_v_stay(schedule, method_name)
            
            all_constraints_satisfied = all([
                results["completeness"],
                results["topological_order"],
                results["alloc_free_pairing"],
                results["buffer_lifecycle"]
            ])
            
            results["success"] = all_constraints_satisfied
            
            print(f"\n{'='*60}")
            if all_constraints_satisfied:
                print(f"🎉 {method_name} 方法所有约束验证通过！")
                print(f"V_stay: {results['v_stay_info']['v_stay']}")
            else:
                print(f"❌ {method_name} 方法约束验证失败")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"验证过程中出现错误: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def validate_all_methods(self) -> Dict:
        """验证所有配置的方法"""
        print(f"\n{'='*100}")
        print("通用调度序列验证工具")
        print("验证配置的所有方法和测试用例")
        print(f"{'='*100}")
        
        all_results = {}
        
        for case in TEST_CASES:
            print(f"\n{'='*100}")
            print(f"开始验证测试用例: {case}")
            print(f"{'='*100}")
            
            case_results = {}
            
            for method_name, (folder_path, file_pattern) in SCHEDULE_CONFIGS.items():
                try:
                    # 构建完整的文件路径
                    schedule_file = os.path.join(folder_path, file_pattern.format(case=case))
                    
                    # 检查文件是否存在
                    if not os.path.exists(schedule_file):
                        print(f"⚠️  文件不存在: {schedule_file}")
                        case_results[method_name] = {"success": False, "error": "文件不存在"}
                        continue
                    
                    # 验证方法
                    result = self.validate_single_method(case, method_name, schedule_file)
                    case_results[method_name] = result
                    
                except Exception as e:
                    print(f"❌ {method_name} 方法验证失败: {e}")
                    case_results[method_name] = {"success": False, "error": str(e)}
            
            all_results[case] = case_results
        
        return all_results
    
    def print_summary(self, all_results: Dict):
        """打印汇总结果"""
        print(f"\n{'='*120}")
        print("验证结果汇总")
        print(f"{'='*120}")
        
        # 构建表头
        methods = list(SCHEDULE_CONFIGS.keys())
        header = f"{'测试用例':<20}"
        for method in methods:
            header += f" {method:<12}"
        for method in methods:
            header += f" {method}_V_stay"[:12] + " " * max(0, 12 - len(f"{method}_V_stay"))
        print(header)
        print("-" * len(header))
        
        # 统计数据
        method_stats = {method: {"total": 0, "passed": 0} for method in methods}
        
        # 打印每个测试用例的结果
        for case in TEST_CASES:
            if case in all_results:
                case_results = all_results[case]
                
                # 状态行
                status_line = f"{case:<20}"
                for method in methods:
                    if method in case_results:
                        result = case_results[method]
                        method_stats[method]["total"] += 1
                        
                        if result.get("success"):
                            status_line += f" {'✅ 通过':<12}"
                            method_stats[method]["passed"] += 1
                        else:
                            status_line += f" {'❌ 失败':<12}"
                    else:
                        status_line += f" {'N/A':<12}"
                
                # V_stay行
                for method in methods:
                    if method in case_results and case_results[method].get("success"):
                        v_stay = case_results[method]["v_stay_info"]["v_stay"]
                        status_line += f" {v_stay:<12}"
                    else:
                        status_line += f" {'N/A':<12}"
                
                print(status_line)
        
        # 统计结果
        print(f"\n{'='*80}")
        print("验证统计结果")
        print(f"{'='*80}")
        total_cases = len(TEST_CASES)
        
        for method in methods:
            stats = method_stats[method]
            if stats["total"] > 0:
                pass_rate = stats["passed"] / stats["total"] * 100
                print(f"{method}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
        # 性能对比（如果有多个方法都通过）
        if len(methods) > 1:
            print(f"\n{'='*80}")
            print("性能对比分析")
            print(f"{'='*80}")
            
            for case in TEST_CASES:
                if case in all_results:
                    case_results = all_results[case]
                    valid_methods = []
                    
                    for method in methods:
                        if method in case_results and case_results[method].get("success"):
                            v_stay = case_results[method]["v_stay_info"]["v_stay"]
                            valid_methods.append((method, v_stay))
                    
                    if len(valid_methods) > 1:
                        valid_methods.sort(key=lambda x: x[1])  # 按V_stay排序
                        best_method, best_v_stay = valid_methods[0]
                        print(f"{case}: 最优方法是 {best_method} (V_stay: {best_v_stay})")
        
        print(f"\n{'='*80}")
        print("验证完成！")
        print("说明：")
        print("1. 本工具可以验证任意配置的调度方法")
        print("2. 只需修改脚本顶部的 SCHEDULE_CONFIGS 配置即可")
        print("3. 验证内容包括完整性、拓扑序、ALLOC/FREE配对、缓冲区生命周期等约束")
        print("4. 约束验证通过是调度序列有效的必要条件")
        print(f"{'='*80}")

def main():
    """主函数"""
    validator = UniversalValidator()
    
    # 验证所有配置的方法
    all_results = validator.validate_all_methods()
    
    # 打印汇总结果
    validator.print_summary(all_results)

if __name__ == "__main__":
    main()