#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
问题3验证脚本 - 基于问题2验证逻辑
===============================================

功能：使用问题2的验证逻辑来验证问题3的结果合理性
验证内容：
1. 文件格式验证（memory.txt, schedule.txt, spill.txt）
2. 缓存地址分配合理性（地址不重叠、容量约束）
3. SPILL操作有效性（格式正确、数量统计）
4. 总额外数据搬运量计算和对比
5. 性能指标分析

使用方法：
验证问题3的结果文件
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# ==================== 配置区域 ====================
PROBLEM3_CONFIGS = [
    {
        'name': '问题3结果',
        'base_path': 'C:/Users/dzdragon/Desktop/A题/A题/Problem3/Problem3',
        'file_patterns': {
            'memory': '{case}_memory.txt',
            'schedule': '{case}_schedule.txt', 
            'spill': '{case}_spill.txt'
        }
    }
]

# 测试用例
TEST_CASES = ['Conv_Case0', 'Conv_Case1', 'FlashAttention_Case0', 'FlashAttention_Case1', 'Matmul_Case0', 'Matmul_Case1']

# 数据文件路径
DATA_DIR = 'C:/Users/dzdragon/Desktop/A题/A题/data'

# 硬件缓存资源大小（题目表1）
CACHE_LIMITS = {
    'L1': 4096,
    'UB': 1024,
    'L0A': 256,
    'L0B': 256,
    'L0C': 512
}

@dataclass
class Node:
    """节点数据结构"""
    id: int
    op: str
    type: str = ''
    size: int = 0
    cache_type: str = ''
    bufs: List[int] = None
    pipe: str = ''
    
    def __post_init__(self):
        if self.bufs is None:
            self.bufs = []

@dataclass
class ValidationResult:
    """验证结果数据结构"""
    passed: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    spill_count: int = 0
    extra_transfer: int = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class Problem3Validator:
    """问题3验证器 - 基于问题2验证逻辑"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.buffer_info = {}  # buf_id -> {size, cache_type, has_copy_in}
        
    def load_graph_data(self, case_name: str):
        """加载计算图数据"""
        print(f"\n=== 加载 {case_name} 数据 ===")
        
        # 优先尝试JSON格式
        json_file = f"{case_name}.json"
        if os.path.exists(json_file):
            return self.load_json_data(case_name)
        
        # 尝试CSV格式
        nodes_file = f"{DATA_DIR}/CSV版本/{case_name}_Nodes.csv"
        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"找不到数据文件: {nodes_file}")
            
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
        json_file = f"{case_name}.json"
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.nodes = {}
        self.buffer_info = {}
        
        for node_data in data['Nodes']:
            node = Node(
                id=node_data['Id'],
                op=node_data['Op'],
                type=node_data.get('Type', ''),
                size=node_data.get('Size', 0),
                cache_type=node_data.get('Type', ''),
                bufs=node_data.get('Bufs', []),
                pipe=node_data.get('Pipe', '')
            )
            self.nodes[node.id] = node
            
            # 收集缓冲区信息
            if node.op == 'ALLOC' and node_data.get('BufId') is not None:
                buf_id = node_data['BufId']
                self.buffer_info[buf_id] = {
                    'size': node.size,
                    'cache_type': node.type,
                    'has_copy_in': False
                }
            elif node.op == 'COPY_IN' and node.bufs:
                for buf in node.bufs:
                    if buf in self.buffer_info:
                        self.buffer_info[buf]['has_copy_in'] = True
        
        # 加载边数据
        self.edges = [(edge[0], edge[1]) for edge in data['Edges']]
        
        print(f"加载完成: {len(self.nodes)} 个节点, {len(self.edges)} 条边")
        print(f"缓冲区信息: {len(self.buffer_info)} 个缓冲区")
    
    def load_memory_allocation(self, file_path: str) -> Dict[int, int]:
        """加载内存分配结果"""
        memory_allocation = {}
        
        if not os.path.exists(file_path):
            return memory_allocation
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    try:
                        buf_id, offset = line.split(':', 1)
                        memory_allocation[int(buf_id)] = int(offset)
                    except ValueError:
                        continue
        
        return memory_allocation
    
    def load_spill_operations(self, file_path: str) -> List[Tuple[int, int]]:
        """加载SPILL操作列表"""
        spill_operations = []
        
        if not os.path.exists(file_path):
            return spill_operations
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    try:
                        buf_id, new_offset = line.split(':', 1)
                        spill_operations.append((int(buf_id), int(new_offset)))
                    except ValueError:
                        continue
        
        return spill_operations
    
    def validate_memory_allocation(self, memory_allocation: Dict[int, int]) -> ValidationResult:
        """验证内存分配合理性"""
        result = ValidationResult()
        
        # 按缓存类型分组
        cache_buffers = defaultdict(list)
        
        for buf_id, offset in memory_allocation.items():
            if buf_id in self.buffer_info:
                buf_info = self.buffer_info[buf_id]
                cache_type = buf_info['cache_type']
                size = buf_info['size']
                
                cache_buffers[cache_type].append({
                    'buf_id': buf_id,
                    'offset': offset,
                    'size': size,
                    'end': offset + size
                })
        
        # 验证缓存容量约束
        for cache_type, buffers in cache_buffers.items():
            if cache_type not in CACHE_LIMITS:
                result.warnings.append(f"未知缓存类型: {cache_type}")
                continue
            
            limit = CACHE_LIMITS[cache_type]
            
            for buf in buffers:
                if buf['end'] > limit:
                    result.errors.append(f"{cache_type}缓存容量超限: Buf{buf['buf_id']}[{buf['offset']}:{buf['end']}] > {limit}")
                    result.passed = False
            
            # 检查地址重叠（警告级别）
            for i, buf1 in enumerate(buffers):
                for buf2 in buffers[i+1:]:
                    if (buf1['offset'] < buf2['end'] and buf2['offset'] < buf1['end']):
                        result.warnings.append(
                            f"{cache_type}缓存可能存在地址重叠: "
                            f"Buf{buf1['buf_id']}[{buf1['offset']}:{buf1['end']}] 与 "
                            f"Buf{buf2['buf_id']}[{buf2['offset']}:{buf2['end']}]"
                        )
        
        return result
    
    def calculate_extra_transfer(self, spill_operations: List[Tuple[int, int]]) -> int:
        """计算额外数据搬运量"""
        total_extra_transfer = 0
        
        for buf_id, new_offset in spill_operations:
            if buf_id in self.buffer_info:
                buf_info = self.buffer_info[buf_id]
                size = buf_info['size']
                has_copy_in = buf_info['has_copy_in']
                
                if has_copy_in:
                    # 被COPY_IN使用的缓冲区，只有SPILL_IN产生额外搬运
                    total_extra_transfer += size
                else:
                    # 未被COPY_IN使用的缓冲区，SPILL_OUT和SPILL_IN都产生额外搬运
                    total_extra_transfer += size * 2
        
        return total_extra_transfer
    
    def validate_case(self, config: Dict, case_name: str) -> ValidationResult:
        """验证单个测试用例"""
        print(f"\n=== 验证 {config['name']} 方法 {case_name} ===")
        
        result = ValidationResult()
        
        try:
            # 构建文件路径
            base_path = config['base_path']
            patterns = config['file_patterns']
            
            # 加载内存分配（如果存在）
            memory_allocation = {}
            if 'memory' in patterns:
                memory_file = os.path.join(base_path, patterns['memory'].format(case=case_name))
                memory_allocation = self.load_memory_allocation(memory_file)
            
            # 加载SPILL操作
            spill_operations = []
            if 'spill' in patterns:
                spill_file = os.path.join(base_path, patterns['spill'].format(case=case_name))
                spill_operations = self.load_spill_operations(spill_file)
            
            # 验证内存分配
            if memory_allocation:
                memory_result = self.validate_memory_allocation(memory_allocation)
                result.passed = result.passed and memory_result.passed
                result.errors.extend(memory_result.errors)
                result.warnings.extend(memory_result.warnings)
            
            # 计算性能指标
            result.spill_count = len(spill_operations)
            result.extra_transfer = self.calculate_extra_transfer(spill_operations)
            
        except Exception as e:
            result.passed = False
            result.errors.append(f"验证过程出错: {str(e)}")
        
        return result
    
    def validate_all(self):
        """验证所有测试用例"""
        print("="*60)
        print("问题3验证 - 基于问题2验证逻辑")
        print("="*60)
        
        all_results = {}
        
        for case_name in TEST_CASES:
            print(f"\n{'='*20} {case_name} {'='*20}")
            
            # 加载计算图数据
            try:
                self.load_graph_data(case_name)
            except Exception as e:
                print(f"加载数据失败: {e}")
                continue
            
            case_results = {}
            
            # 验证每个配置
            for config in PROBLEM3_CONFIGS:
                result = self.validate_case(config, case_name)
                case_results[config['name']] = result
            
            all_results[case_name] = case_results
            
            # 输出结果
            print(f"\n{case_name} 验证结果:")
            print("-" * 50)
            
            for method_name, result in case_results.items():
                status = "✓ 通过" if result.passed else "✗ 失败"
                print(f"{method_name:<15} | {status}")
                
                # 显示错误
                for error in result.errors[:3]:  # 只显示前3个错误
                    print(f"  错误: {error}")
                if len(result.errors) > 3:
                    print(f"  ... 还有 {len(result.errors) - 3} 个错误")
                
                # 显示警告
                for warning in result.warnings[:3]:  # 只显示前3个警告
                    print(f"  警告: {warning}")
                if len(result.warnings) > 3:
                    print(f"  ... 还有 {len(result.warnings) - 3} 个警告")
                
                # 显示性能指标
                print(f"  SPILL次数: {result.spill_count}")
                print(f"  额外搬运量: {result.extra_transfer}")
        
        # 输出总结
        self.print_summary(all_results)
    
    def print_summary(self, all_results: Dict):
        """输出验证总结"""
        print("\n" + "="*60)
        print("验证总结")
        print("="*60)
        
        for method_name in [config['name'] for config in PROBLEM3_CONFIGS]:
            passed_count = 0
            total_spill = 0
            total_transfer = 0
            total_cases = 0
            
            for case_name, case_results in all_results.items():
                if method_name in case_results:
                    result = case_results[method_name]
                    if result.passed:
                        passed_count += 1
                    total_spill += result.spill_count
                    total_transfer += result.extra_transfer
                    total_cases += 1
            
            if total_cases > 0:
                print(f"\n{method_name}:")
                print(f"  通过率: {passed_count}/{total_cases} ({passed_count/total_cases*100:.1f}%)")
                print(f"  总SPILL次数: {total_spill}")
                print(f"  总额外搬运量: {total_transfer}")
        
        print(f"\n验证完成！共验证 {sum(len(case_results) for case_results in all_results.values())} 个结果")

def main():
    validator = Problem3Validator()
    validator.validate_all()

if __name__ == "__main__":
    main()