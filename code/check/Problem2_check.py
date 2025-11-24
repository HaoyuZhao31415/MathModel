#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2通用验证脚本 - 缓存分配与SPILL操作验证
===============================================

功能：验证问题2的两种方法（机理建模和算法建模）的结果合理性
验证内容：
1. 文件格式验证（memory.txt, schedule.txt, spill.txt）
2. 缓存地址分配合理性（地址不重叠、容量约束）
3. SPILL操作有效性（格式正确、数量统计）
4. 总额外数据搬运量计算和对比
5. 性能指标分析

使用方法：
只需修改 PROBLEM2_CONFIGS 配置即可验证不同方法
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import os
import json
import csv
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# ==================== 配置区域 ====================
PROBLEM2_CONFIGS = [
    {
        'name': '机理建模',
        'base_path': 'C:/Users/dzdragon/Desktop/A题/A题/2/机理建模',
        'file_patterns': {
            'memory': '{case}_memory.txt',
            'schedule': '{case}_schedule.txt', 
            'spill': '{case}_spill.txt'
        }
    },
    {
        'name': '算法建模',
        'base_path': 'C:/Users/dzdragon/Desktop/A题/A题/2/算法建模',
        'file_patterns': {
            'spill': '{case}_spill.txt'
        }
    }
]

# 测试用例
TEST_CASES = ['Conv_Case0', 'Conv_Case1', 'FlashAttention_Case0', 'FlashAttention_Case1', 'Matmul_Case0', 'Matmul_Case1']

# 数据文件路径
DATA_DIR = 'C:/Users/dzdragon/Desktop/A题/A题/data'

# 硬件缓存容量约束（题目表1）
CACHE_LIMITS = {
    'L1': 4096,
    'UB': 1024,
    'L0A': 256,
    'L0B': 256,
    'L0C': 512
}

# ==================== 数据结构定义 ====================
@dataclass
class Node:
    id: int
    op: str
    type: str = ""
    size: int = 0
    cache_type: str = ""
    bufs: List[int] = None
    pipe: str = ""
    
    def __post_init__(self):
        if self.bufs is None:
            self.bufs = []

@dataclass
class BufferAllocation:
    buf_id: int
    offset: int
    size: int
    cache_type: str

@dataclass
class SpillOperation:
    buf_id: int
    new_offset: int
    
@dataclass
class ValidationResult:
    method_name: str
    case_name: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict

class Problem2Validator:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.buffer_info = {}  # buf_id -> {size, cache_type, has_copy_in}
        
    def load_graph_data(self, case_name: str):
        """加载计算图数据"""
        print(f"\n=== 加载 {case_name} 数据 ===")
        
        # 加载节点数据
        nodes_file = f"{DATA_DIR}/CSV版本/{case_name}_Nodes.csv"
        if not os.path.exists(nodes_file):
            nodes_file = f"{DATA_DIR}/Json版本/{case_name}.json"
            return self.load_json_data(case_name)
            
        self.nodes = {}
        self.buffer_info = {}
        
        with open(nodes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 处理Size字段
                size_str = row.get('Size', '0').strip()
                size = int(size_str) if size_str and size_str.isdigit() else 0
                
                # 处理Bufs字段
                bufs_str = row.get('Bufs', '').strip()
                if not bufs_str:
                    bufs_str = row.get('BufId', '').strip()  # 尝试BufId字段
                
                bufs = []
                if bufs_str:
                    if bufs_str.startswith('"') and bufs_str.endswith('"'):
                        bufs_str = bufs_str[1:-1]  # 去掉引号
                    bufs = [int(x) for x in bufs_str.split(',') if x.strip().isdigit()]
                
                # 处理缓存类型字段
                cache_type = row.get('CacheType', '').strip()
                if not cache_type:
                    cache_type = row.get('Type', '').strip()
                
                node = Node(
                    id=int(row['Id']),
                    op=row['Op'],
                    type=row.get('Type', ''),
                    size=size,
                    cache_type=cache_type,
                    bufs=bufs,
                    pipe=row.get('Pipe', '')
                )
                self.nodes[node.id] = node
                
                # 收集缓冲区信息
                if node.op in ['ALLOC', 'FREE'] and node.bufs:
                    buf_id = node.bufs[0]
                    if buf_id not in self.buffer_info:
                        self.buffer_info[buf_id] = {
                            'size': node.size,
                            'cache_type': node.cache_type,
                            'has_copy_in': False
                        }
                    
                    # 检查是否有COPY_IN操作使用此缓冲区
                    if node.op == 'COPY_IN':
                        for buf in node.bufs:
                            if buf in self.buffer_info:
                                self.buffer_info[buf]['has_copy_in'] = True
        
        # 加载边数据
        edges_file = f"{DATA_DIR}/CSV版本/{case_name}_Edges.csv"
        self.edges = []
        
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 处理不同的列名
                    from_id = row.get('From') or row.get('StartNodeId')
                    to_id = row.get('To') or row.get('EndNodeId')
                    
                    if from_id and to_id:
                        self.edges.append((int(from_id), int(to_id)))
        
        print(f"加载完成: {len(self.nodes)} 个节点, {len(self.edges)} 条边")
        print(f"缓冲区信息: {len(self.buffer_info)} 个缓冲区")
        
    def load_json_data(self, case_name: str):
        """加载JSON格式数据"""
        json_file = f"{DATA_DIR}/Json版本/{case_name}.json"
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.nodes = {}
        self.buffer_info = {}
        
        for node_data in data['nodes']:
            node = Node(
                id=node_data['id'],
                op=node_data['op'],
                type=node_data.get('type', ''),
                size=node_data.get('size', 0),
                cache_type=node_data.get('cache_type', ''),
                bufs=node_data.get('bufs', []),
                pipe=node_data.get('pipe', '')
            )
            self.nodes[node.id] = node
            
            # 收集缓冲区信息
            if node.op in ['ALLOC', 'FREE'] and node.bufs:
                buf_id = node.bufs[0]
                if buf_id not in self.buffer_info:
                    self.buffer_info[buf_id] = {
                        'size': node.size,
                        'cache_type': node.cache_type,
                        'has_copy_in': False
                    }
                
                # 检查COPY_IN
                if node.op == 'COPY_IN':
                    for buf in node.bufs:
                        if buf in self.buffer_info:
                            self.buffer_info[buf]['has_copy_in'] = True
        
        self.edges = [(edge['from'], edge['to']) for edge in data['edges']]
        
    def load_method_results(self, method_config: Dict, case_name: str) -> Dict:
        """加载方法结果文件"""
        results = {}
        base_path = method_config['base_path']
        patterns = method_config['file_patterns']
        
        for file_type, pattern in patterns.items():
            file_path = os.path.join(base_path, pattern.format(case=case_name))
            
            if os.path.exists(file_path):
                if file_type == 'memory':
                    results[file_type] = self.parse_memory_file(file_path)
                elif file_type == 'schedule':
                    results[file_type] = self.parse_schedule_file(file_path)
                elif file_type == 'spill':
                    results[file_type] = self.parse_spill_file(file_path)
            else:
                results[file_type] = None
                
        return results
    
    def parse_memory_file(self, file_path: str) -> List[BufferAllocation]:
        """解析memory.txt文件"""
        allocations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    try:
                        buf_id, offset = line.split(':')
                        buf_id = int(buf_id)
                        offset = int(offset)
                        
                        if buf_id in self.buffer_info:
                            allocation = BufferAllocation(
                                buf_id=buf_id,
                                offset=offset,
                                size=self.buffer_info[buf_id]['size'],
                                cache_type=self.buffer_info[buf_id]['cache_type']
                            )
                            allocations.append(allocation)
                    except ValueError:
                        continue
                        
        return allocations
    
    def parse_schedule_file(self, file_path: str) -> List[int]:
        """解析schedule.txt文件"""
        schedule = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    schedule.append(int(line))
                    
        return schedule
    
    def parse_spill_file(self, file_path: str) -> Tuple[List[SpillOperation], Dict]:
        """解析spill.txt文件"""
        spill_ops = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    try:
                        buf_id, new_offset = line.split(':')
                        buf_id = int(buf_id)
                        new_offset = int(new_offset)
                        
                        spill_op = SpillOperation(buf_id=buf_id, new_offset=new_offset)
                        spill_ops.append(spill_op)
                    except ValueError:
                        continue
        
        # 直接计算统计信息，不依赖文件末尾的统计数据
        stats = {
            'total_spills': len(spill_ops),
            'total_transfer': 0  # 将在validate_spill_operations中计算
        }
                        
        return spill_ops, stats
    
    def validate_memory_allocation(self, allocations: List[BufferAllocation]) -> Tuple[bool, List[str], List[str]]:
        """验证缓存地址分配"""
        errors = []
        warnings = []
        
        if not allocations:
            errors.append("缺少内存分配文件或文件为空")
            return False, errors, warnings
        
        # 按缓存类型分组
        cache_groups = defaultdict(list)
        for alloc in allocations:
            cache_groups[alloc.cache_type].append(alloc)
        
        # 验证每个缓存类型
        for cache_type, allocs in cache_groups.items():
            if cache_type not in CACHE_LIMITS:
                warnings.append(f"未知缓存类型: {cache_type}")
                continue
                
            # 检查容量约束
            max_addr = 0
            for alloc in allocs:
                end_addr = alloc.offset + alloc.size
                max_addr = max(max_addr, end_addr)
                
            if max_addr > CACHE_LIMITS[cache_type]:
                errors.append(f"{cache_type}缓存超出容量限制: 使用{max_addr}, 限制{CACHE_LIMITS[cache_type]}")
            
            # 检查地址重叠（需要结合调度序列的生命周期）
            # 这里简化处理，只检查明显的重叠
            sorted_allocs = sorted(allocs, key=lambda x: x.offset)
            for i in range(len(sorted_allocs) - 1):
                curr = sorted_allocs[i]
                next_alloc = sorted_allocs[i + 1]
                
                if curr.offset + curr.size > next_alloc.offset:
                    # 可能重叠，但需要检查生命周期
                    warnings.append(f"{cache_type}缓存可能存在地址重叠: Buf{curr.buf_id}[{curr.offset}:{curr.offset+curr.size}] 与 Buf{next_alloc.buf_id}[{next_alloc.offset}:{next_alloc.offset+next_alloc.size}]")
        
        return len(errors) == 0, errors, warnings
    
    def validate_spill_operations(self, spill_ops: List[SpillOperation], stats: Dict) -> Tuple[bool, List[str], List[str]]:
        """验证SPILL操作"""
        errors = []
        warnings = []
        
        # 验证SPILL操作的缓冲区存在性
        for spill_op in spill_ops:
            if spill_op.buf_id not in self.buffer_info:
                errors.append(f"SPILL操作引用不存在的缓冲区: Buf{spill_op.buf_id}")
        
        # 计算理论额外搬运量
        theoretical_transfer = 0
        for spill_op in spill_ops:
            if spill_op.buf_id in self.buffer_info:
                buf_info = self.buffer_info[spill_op.buf_id]
                if buf_info['has_copy_in']:
                    theoretical_transfer += buf_info['size']  # 仅SPILL_IN
                else:
                    theoretical_transfer += 2 * buf_info['size']  # SPILL_OUT + SPILL_IN
        
        # 更新计算出的额外搬运量
        stats['total_transfer'] = theoretical_transfer
        
        return len(errors) == 0, errors, warnings
    
    def calculate_metrics(self, method_results: Dict) -> Dict:
        """计算性能指标"""
        metrics = {}
        
        # 内存分配指标
        if method_results.get('memory'):
            allocations = method_results['memory']
            cache_usage = defaultdict(int)
            
            for alloc in allocations:
                max_addr = alloc.offset + alloc.size
                cache_usage[alloc.cache_type] = max(cache_usage[alloc.cache_type], max_addr)
            
            metrics['cache_usage'] = dict(cache_usage)
            metrics['cache_utilization'] = {
                cache_type: usage / CACHE_LIMITS.get(cache_type, 1) * 100
                for cache_type, usage in cache_usage.items()
                if cache_type in CACHE_LIMITS
            }
        
        # SPILL操作指标
        if method_results.get('spill'):
            spill_ops, stats = method_results['spill']
            metrics['spill_count'] = len(spill_ops)
            metrics['total_transfer'] = stats.get('total_transfer', 0)
            
            # 按缓存类型统计SPILL
            spill_by_cache = defaultdict(int)
            for spill_op in spill_ops:
                if spill_op.buf_id in self.buffer_info:
                    cache_type = self.buffer_info[spill_op.buf_id]['cache_type']
                    spill_by_cache[cache_type] += 1
            
            metrics['spill_by_cache'] = dict(spill_by_cache)
        
        return metrics
    
    def validate_method(self, method_config: Dict, case_name: str) -> ValidationResult:
        """验证单个方法的结果"""
        method_name = method_config['name']
        print(f"\n=== 验证 {method_name} 方法 {case_name} ===")
        
        # 加载结果文件
        method_results = self.load_method_results(method_config, case_name)
        
        # 初始化验证结果
        result = ValidationResult(
            method_name=method_name,
            case_name=case_name,
            passed=True,
            errors=[],
            warnings=[],
            metrics={}
        )
        
        # 验证文件存在性
        required_files = ['spill']  # 所有方法都应该有spill文件
        if method_name == '机理建模':
            required_files.extend(['memory', 'schedule'])
        
        for file_type in required_files:
            if method_results.get(file_type) is None:
                result.errors.append(f"缺少{file_type}文件")
                result.passed = False
        
        # 验证内存分配（仅机理建模）
        if method_results.get('memory'):
            memory_passed, memory_errors, memory_warnings = self.validate_memory_allocation(method_results['memory'])
            result.passed = result.passed and memory_passed
            result.errors.extend(memory_errors)
            result.warnings.extend(memory_warnings)
        
        # 验证SPILL操作
        if method_results.get('spill'):
            spill_ops, stats = method_results['spill']
            spill_passed, spill_errors, spill_warnings = self.validate_spill_operations(spill_ops, stats)
            result.passed = result.passed and spill_passed
            result.errors.extend(spill_errors)
            result.warnings.extend(spill_warnings)
            
            # 更新指标
            result.metrics['spill_count'] = len(spill_ops)
            result.metrics['total_transfer'] = stats['total_transfer']
        
        # 计算其他指标
        metrics = self.calculate_metrics(method_results)
        result.metrics.update(metrics)
        
        return result
    
    def run_validation(self):
        """运行完整验证"""
        print("=" * 60)
        print("问题2验证脚本 - 缓存分配与SPILL操作验证")
        print("=" * 60)
        
        all_results = []
        
        for case_name in TEST_CASES:
            print(f"\n{'='*20} {case_name} {'='*20}")
            
            # 加载计算图数据
            self.load_graph_data(case_name)
            
            # 验证每种方法
            case_results = []
            for method_config in PROBLEM2_CONFIGS:
                result = self.validate_method(method_config, case_name)
                case_results.append(result)
                all_results.append(result)
            
            # 输出案例结果
            self.print_case_results(case_results)
        
        # 输出总结
        self.print_summary(all_results)
    
    def print_case_results(self, case_results: List[ValidationResult]):
        """输出单个案例的结果"""
        case_name = case_results[0].case_name
        print(f"\n{case_name} 验证结果:")
        print("-" * 50)
        
        for result in case_results:
            status = "✓ 通过" if result.passed else "✗ 失败"
            print(f"{result.method_name:12} | {status}")
            
            if result.errors:
                for error in result.errors:
                    print(f"  错误: {error}")
            
            if result.warnings:
                for warning in result.warnings[:3]:  # 只显示前3个警告
                    print(f"  警告: {warning}")
            
            # 显示关键指标
            metrics = result.metrics
            if 'spill_count' in metrics:
                print(f"  SPILL次数: {metrics['spill_count']}")
            if 'total_transfer' in metrics:
                print(f"  额外搬运量: {metrics['total_transfer']}")
        
        # 对比分析
        if len(case_results) == 2:
            self.print_comparison(case_results)
    
    def print_comparison(self, case_results: List[ValidationResult]):
        """输出对比分析"""
        if len(case_results) != 2:
            return
            
        result1, result2 = case_results
        print(f"\n对比分析 ({result1.method_name} vs {result2.method_name}):")
        
        # SPILL次数对比
        spill1 = result1.metrics.get('spill_count', 0)
        spill2 = result2.metrics.get('spill_count', 0)
        
        if spill1 > 0 and spill2 > 0:
            improvement = (spill1 - spill2) / spill1 * 100
            print(f"  SPILL次数: {spill1} vs {spill2} (改进: {improvement:.1f}%)")
        
        # 额外搬运量对比
        transfer1 = result1.metrics.get('total_transfer', 0)
        transfer2 = result2.metrics.get('total_transfer', 0)
        
        if transfer1 > 0 and transfer2 > 0:
            improvement = (transfer1 - transfer2) / transfer1 * 100
            print(f"  额外搬运量: {transfer1} vs {transfer2} (改进: {improvement:.1f}%)")
    
    def print_summary(self, all_results: List[ValidationResult]):
        """输出总结"""
        print("\n" + "=" * 60)
        print("验证总结")
        print("=" * 60)
        
        # 按方法分组
        method_results = defaultdict(list)
        for result in all_results:
            method_results[result.method_name].append(result)
        
        for method_name, results in method_results.items():
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)
            
            print(f"\n{method_name}:")
            print(f"  通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
            
            # 统计指标
            total_spills = sum(r.metrics.get('spill_count', 0) for r in results)
            total_transfer = sum(r.metrics.get('total_transfer', 0) for r in results)
            
            print(f"  总SPILL次数: {total_spills}")
            print(f"  总额外搬运量: {total_transfer}")
            
            # 失败案例
            failed_cases = [r.case_name for r in results if not r.passed]
            if failed_cases:
                print(f"  失败案例: {', '.join(failed_cases)}")
        
        print(f"\n验证完成！共验证 {len(all_results)} 个结果")

def main():
    """主函数"""
    validator = Problem2Validator()
    validator.run_validation()

if __name__ == "__main__":
    main()