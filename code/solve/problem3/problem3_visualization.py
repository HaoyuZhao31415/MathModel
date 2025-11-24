#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题三高质量科研可视化分析
采用现代科研期刊风格的可视化展示
分析时间优化、多目标权衡和算法收敛性
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
import platform
from typing import Dict, List, Optional, Tuple

# 设置顶级期刊绘图风格
def setup_journal_style():
    """设置顶级科研期刊的绘图风格"""
    plt.style.use('default')
    
    # 字体设置
    if platform.system() == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    elif platform.system() == 'Darwin':
        chinese_fonts = ['Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'Heiti SC']
    else:
        chinese_fonts = ['Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    
    # 科研论文级别的rcParams设置
    journal_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': chinese_fonts + ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 24,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'patch.linewidth': 0.5,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 600,
        'figure.dpi': 120,
        'axes.unicode_minus': False,
        'text.usetex': False,
    }
    
    plt.rcParams.update(journal_params)
    return True

# 顶级期刊配色方案
JOURNAL_COLORS = {
    'modern_scientific': {
        'primary': '#2E86AB',      # 深海蓝
        'secondary': '#A23B72',    # 深紫红
        'success': '#F18F01',      # 活力橙
        'danger': '#C73E1D',       # 科研红
        'warning': '#FFB563',      # 温暖橙
        'info': '#3D5A80',         # 学术蓝
        'purple': '#7209B7',       # 深紫
        'brown': '#8B5A3C',        # 大地棕
        'accent1': '#1B9AAA',      # 青绿色
        'accent2': '#EE6C4D',      # 珊瑚红
        'light_blue': '#06FFA5',   # 亮青色
        'dark_green': '#2B9348'    # 深绿色
    }
}

# 初始化期刊风格
setup_journal_style()
current_colors = JOURNAL_COLORS['modern_scientific']


class Problem3JournalVisualizer:
    """问题三高质量科研期刊可视化器"""
    
    def __init__(self, viz_data_dir="Problem3_Visualization_Data", 
                 figures_dir="problem3_journal_figures"):
        self.viz_data_dir = viz_data_dir
        self.figures_dir = figures_dir
        
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
            
        self.cases = [
            "Matmul_Case0", "Matmul_Case1", 
            "FlashAttention_Case0", "FlashAttention_Case1",
            "Conv_Case0", "Conv_Case1"
        ]
        
        # 加载所有可视化数据
        self.viz_data = {}
        self._load_visualization_data()
    
    def _load_visualization_data(self):
        """加载所有案例的可视化数据"""
        print("正在加载问题三可视化数据...")
        
        for case_name in self.cases:
            viz_file = os.path.join(self.viz_data_dir, f"{case_name}_problem3_visualization.json")
            if os.path.exists(viz_file):
                with open(viz_file, 'r', encoding='utf-8') as f:
                    self.viz_data[case_name] = json.load(f)
                print(f"  ✓ 已加载 {case_name}")
            else:
                print(f"  ✗ 未找到 {case_name} 的可视化数据")
    
    def create_comprehensive_analysis(self):
        """创建综合分析图表"""
        print("开始生成问题三高质量科研可视化图表...")
        
        # 1. 执行时间优化对比
        self._create_runtime_optimization_comparison()
        
        # 2. 多目标权衡分析  
        self._create_multi_objective_tradeoff()
        
        # 3. 三阶段优化效果分析
        self._create_three_stage_analysis()
        
        # 4. 流水线利用率分析
        self._create_pipeline_utilization_analysis()
        
        # 5. 性能改进贡献分解
        self._create_performance_contribution_analysis()
        
        print("✓ 问题三高质量可视化图表生成完成")
    
    def _create_runtime_optimization_comparison(self):
        """创建执行时间优化对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 收集数据
        cases = []
        t0_runtimes = []
        t2_runtimes = []
        improvement_percentages = []
        
        # 标题映射
        title_mapping = {
            'Matmul_Case0': 'Matmul\nCase 0', 'Matmul_Case1': 'Matmul\nCase 1',
            'FlashAttention_Case0': 'FlashAttn\nCase 0', 
            'FlashAttention_Case1': 'FlashAttn\nCase 1',
            'Conv_Case0': 'Conv\nCase 0', 'Conv_Case1': 'Conv\nCase 1'
        }
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                results = data.get('results', {})
                
                baseline_runtime = results.get('baseline_runtime', 0)
                optimized_runtime = results.get('optimized_runtime', 0)
                
                cases.append(title_mapping.get(case_name, case_name))
                t0_runtimes.append(baseline_runtime)
                t2_runtimes.append(optimized_runtime)
                
                # 计算改进百分比
                if baseline_runtime > 0:
                    improvement = ((baseline_runtime - optimized_runtime) / baseline_runtime) * 100
                    improvement_percentages.append(improvement)
                else:
                    improvement_percentages.append(0)
        
        # 子图1：执行时间对比
        x = np.arange(len(cases))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, t0_runtimes, width, label='问题二基准 (T0)', 
                       color=current_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax1.bar(x + width/2, t2_runtimes, width, label='问题三优化 (T2)', 
                       color=current_colors['success'], alpha=0.8, edgecolor='white', linewidth=1)
        
        ax1.set_ylabel('执行时间 (周期)', fontweight='bold')
        ax1.set_title('(a) 执行时间对比', fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cases, rotation=0)
        ax1.legend(fontsize=16)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    # 智能显示数值
                    if height >= 1000000:
                        label = f'{height/1000000:.1f}M'
                    elif height >= 1000:
                        label = f'{height/1000:.0f}K'
                    else:
                        label = f'{int(height)}'
                    
                    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 子图2：改进百分比
        bars3 = ax2.bar(range(len(cases)), improvement_percentages, 
                       color=current_colors['danger'], alpha=0.8, edgecolor='white', linewidth=1)
        
        ax2.set_ylabel('执行时间改进比例 (%)', fontweight='bold')
        ax2.set_title('(b) 执行时间改进效果', fontweight='bold', pad=20)
        ax2.set_xticks(range(len(cases)))
        ax2.set_xticklabels(cases, rotation=0)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # 添加百分比标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 美化所有子图
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(os.path.join(self.figures_dir, 'problem3_runtime_optimization.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✓ 已生成执行时间优化对比图")
    
    def _create_multi_objective_tradeoff(self):
        """创建多目标权衡分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 收集Pareto前沿数据
        case_data = {}
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                analysis = data.get('analysis', {})
                multi_obj = analysis.get('multi_objective_tradeoff', {})
                pareto_points = multi_obj.get('pareto_points', [])
                
                if pareto_points:
                    case_data[case_name] = pareto_points
        
        # 子图1：Pareto前沿展示
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.cases)))
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for i, (case_name, points) in enumerate(case_data.items()):
            if points:
                runtimes = [p['runtime'] for p in points]
                spill_costs = [p['spill_cost'] for p in points]
                stages = [p['stage'] for p in points]
                
                # 归一化数据用于更好的可视化
                if max(runtimes) > 0:
                    norm_runtimes = [r / max(runtimes) for r in runtimes]
                else:
                    norm_runtimes = runtimes
                    
                if max(spill_costs) > 0:
                    norm_spill_costs = [s / max(spill_costs) for s in spill_costs]
                else:
                    norm_spill_costs = spill_costs
                
                short_name = self._get_short_name(case_name)
                ax1.scatter(norm_runtimes, norm_spill_costs, 
                           color=colors[i], marker=markers[i % len(markers)], 
                           s=120, alpha=0.8, edgecolor='white', linewidth=2, 
                           label=short_name)
                
                # 连接同一案例的点
                if len(norm_runtimes) > 1:
                    ax1.plot(norm_runtimes, norm_spill_costs, '--', 
                           color=colors[i], alpha=0.5, linewidth=1.5)
                
                # 标注阶段
                for j, (nr, ns, stage) in enumerate(zip(norm_runtimes, norm_spill_costs, stages)):
                    ax1.annotate(stage, (nr, ns), xytext=(5, 5), 
                               textcoords='offset points', fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('归一化执行时间', fontweight='bold')
        ax1.set_ylabel('归一化数据搬运量', fontweight='bold')
        ax1.set_title('(a) 多目标Pareto前沿分析', fontweight='bold', pad=20)
        ax1.legend(fontsize=14, loc='upper right')
        
        # 子图2：权衡效果统计
        cases_names = []
        runtime_reductions = []
        spill_increases = []
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                analysis = data.get('analysis', {})
                multi_obj = analysis.get('multi_objective_tradeoff', {})
                
                runtime_reduction = multi_obj.get('runtime_reduction', 0)
                spill_increase = multi_obj.get('spill_cost_increase', 0)
                
                cases_names.append(self._get_short_name(case_name))
                runtime_reductions.append(runtime_reduction)
                spill_increases.append(spill_increase)
        
        x = np.arange(len(cases_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, runtime_reductions, width, label='执行时间减少 (%)', 
                       color=current_colors['success'], alpha=0.8, edgecolor='white')
        bars2 = ax2.bar(x + width/2, spill_increases, width, label='数据搬运量增加 (%)', 
                       color=current_colors['warning'], alpha=0.8, edgecolor='white')
        
        ax2.set_ylabel('变化百分比 (%)', fontweight='bold')
        ax2.set_title('(b) 多目标权衡效果统计', fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cases_names, rotation=0)
        ax2.legend(fontsize=16)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top', 
                           fontsize=11, fontweight='bold')
        
        # 美化所有子图
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(os.path.join(self.figures_dir, 'problem3_multi_objective_tradeoff.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✓ 已生成多目标权衡分析图")
    
    def _create_three_stage_analysis(self):
        """创建三阶段优化效果分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        
        # 案例标题映射
        title_mapping = {
            'Matmul_Case0': 'Matmul Case 0',
            'Matmul_Case1': 'Matmul Case 1', 
            'FlashAttention_Case0': 'FlashAttn Case 0',
            'FlashAttention_Case1': 'FlashAttn Case 1',
            'Conv_Case0': 'Conv Case 0',
            'Conv_Case1': 'Conv Case 1'
        }
        
        # 重新安排子图顺序
        case_order = [
            'Matmul_Case0', 'FlashAttention_Case0', 'Conv_Case0',       # 上排
            'Matmul_Case1', 'FlashAttention_Case1', 'Conv_Case1'        # 下排
        ]
        
        # 子图标签
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        
        for i, case_name in enumerate(case_order):
            if case_name not in self.viz_data:
                continue
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 绘制该案例的三阶段分析
            self._draw_three_stage_for_case(ax, case_name, 
                                          title_mapping.get(case_name, case_name), 
                                          subplot_labels[i])
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(os.path.join(self.figures_dir, 'problem3_three_stage_analysis.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✓ 已生成三阶段优化分析图")
    
    def _draw_three_stage_for_case(self, ax, case_name: str, display_title: str, subplot_label: str):
        """为单个案例绘制三阶段分析"""
        if case_name not in self.viz_data:
            ax.text(0.5, 0.5, f'{display_title}\n数据加载失败', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
        data = self.viz_data[case_name]
        analysis = data.get('analysis', {})
        three_stage = analysis.get('three_stage_comparison', [])
        
        if not three_stage:
            ax.text(0.5, 0.5, f'{display_title}\n无三阶段数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # 提取数据
        stages = [stage['stage'] for stage in three_stage]
        runtimes = [stage['runtime'] for stage in three_stage]
        improvements = [stage['improvement'] for stage in three_stage]
        
        # 绘制运行时间柱状图
        colors = [current_colors['primary'], current_colors['secondary'], current_colors['success']]
        bars = ax.bar(range(len(stages)), runtimes, color=colors, alpha=0.8, edgecolor='white')
        
        # 添加改进百分比标签
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            
            # 显示运行时间
            if height >= 1000000:
                time_label = f'{height/1000000:.1f}M'
            elif height >= 1000:
                time_label = f'{height/1000:.0f}K'
            else:
                time_label = f'{int(height)}'
                
            ax.annotate(time_label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 显示改进百分比（除了T0）
            if i > 0 and improvement != 0:
                ax.annotate(f'{improvement:.1f}%', 
                           xy=(bar.get_x() + bar.get_width()/2, height * 0.7),
                           ha='center', va='center', fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           color='red' if improvement > 0 else 'blue', fontweight='bold')
        
        # 设置标题和标签
        ax.set_xlabel('优化阶段', fontsize=12)
        ax.set_ylabel('执行时间 (周期)', fontsize=12)
        ax.set_title(display_title, fontweight='bold', pad=10)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages)
        
        # 添加子图标签
        ax.text(0.02, 0.98, subplot_label, transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    def _create_pipeline_utilization_analysis(self):
        """创建流水线利用率分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 收集所有案例的流水线数据
        all_pipe_usage = defaultdict(int)
        case_utilizations = {}
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                analysis = data.get('analysis', {})
                pipeline = analysis.get('pipeline_utilization', {})
                
                utilization_rates = pipeline.get('utilization_rates', {})
                pipe_usage = pipeline.get('pipe_usage_cycles', {})
                
                case_utilizations[case_name] = utilization_rates
                
                # 累计所有管道使用量
                for pipe, cycles in pipe_usage.items():
                    all_pipe_usage[pipe] += cycles
        
        # 子图1：总体流水线使用分布
        if all_pipe_usage:
            pipes = list(all_pipe_usage.keys())
            cycles = list(all_pipe_usage.values())
            
            # 按使用量排序
            sorted_data = sorted(zip(pipes, cycles), key=lambda x: x[1], reverse=True)
            pipes, cycles = zip(*sorted_data)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(pipes)))
            bars1 = ax1.bar(range(len(pipes)), cycles, color=colors, alpha=0.8, edgecolor='white')
            
            ax1.set_ylabel('总执行周期数', fontweight='bold')
            ax1.set_title('(a) 执行单元总体使用分布', fontweight='bold', pad=20)
            ax1.set_xticks(range(len(pipes)))
            ax1.set_xticklabels(pipes, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, cycle in zip(bars1, cycles):
                height = bar.get_height()
                if height >= 1000000:
                    label = f'{height/1000000:.1f}M'
                elif height >= 1000:
                    label = f'{height/1000:.0f}K'
                else:
                    label = f'{int(height)}'
                    
                ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 子图2：各案例利用率对比
        if case_utilizations:
            # 使用左子图中使用量最高的6个执行单元
            available_pipes = []
            if all_pipe_usage:
                # 按使用量排序，取前6个（与左子图保持一致）
                sorted_pipes = sorted(all_pipe_usage.keys(), key=lambda x: all_pipe_usage[x], reverse=True)
                available_pipes = sorted_pipes[:6]
            
            if available_pipes:
                x = np.arange(len(self.cases))
                width = 0.8 / len(available_pipes)
                colors = plt.cm.Set3(np.linspace(0, 1, len(available_pipes)))
                
                for i, pipe in enumerate(available_pipes):
                    utilizations = []
                    for case_name in self.cases:
                        if case_name in case_utilizations:
                            util = case_utilizations[case_name].get(pipe, 0)
                            utilizations.append(util)
                        else:
                            utilizations.append(0)
                    
                    offset = (i - len(available_pipes)/2 + 0.5) * width
                    bars = ax2.bar(x + offset, utilizations, width, 
                                 label=pipe, color=colors[i], alpha=0.8, edgecolor='white')
                
                ax2.set_ylabel('利用率 (%)', fontweight='bold')
                ax2.set_title('(b) 各案例执行单元利用率对比', fontweight='bold', pad=20)
                ax2.set_xticks(x)
                ax2.set_xticklabels([self._get_short_name(case) for case in self.cases], rotation=45, ha='right')
                ax2.legend(fontsize=14)
                ax2.set_ylim(0, 100)
        
        # 美化所有子图
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(os.path.join(self.figures_dir, 'problem3_pipeline_utilization.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✓ 已生成流水线利用率分析图")
    
    def _create_performance_contribution_analysis(self):
        """创建性能改进贡献分解图"""
        fig, ax1 = plt.subplots(1, 1, figsize=(18, 10))
        
        # 子图1：各阶段贡献堆叠图
        cases_short = []
        t1_contributions = []
        t2_contributions = []
        baselines = []
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                analysis = data.get('analysis', {})
                perf_breakdown = analysis.get('performance_breakdown', {})
                
                baseline_perf = perf_breakdown.get('baseline_performance', {})
                stage_contrib = perf_breakdown.get('stage_contributions', {})
                
                baseline = baseline_perf.get('runtime', 0)
                t1_contrib = stage_contrib.get('T1_critical_path_optimization', {}).get('runtime_reduction', 0)
                t2_contrib = stage_contrib.get('T2_local_refinement', {}).get('runtime_reduction', 0)
                
                cases_short.append(self._get_short_name(case_name))
                baselines.append(baseline)
                t1_contributions.append(t1_contrib)
                t2_contributions.append(t2_contrib)
        
        # 计算剩余时间（优化后）
        remaining_times = []
        for i in range(len(baselines)):
            remaining = baselines[i] - t1_contributions[i] - t2_contributions[i]
            remaining_times.append(max(0, remaining))
        
        x = np.arange(len(cases_short))
        
        # 堆叠柱状图
        bars1 = ax1.bar(x, remaining_times, label='优化后运行时间', 
                       color=current_colors['primary'], alpha=0.8, edgecolor='white')
        bars2 = ax1.bar(x, t2_contributions, bottom=remaining_times, label='T2局部优化贡献', 
                       color=current_colors['success'], alpha=0.8, edgecolor='white')
        
        bottom_for_t1 = np.array(remaining_times) + np.array(t2_contributions)
        bars3 = ax1.bar(x, t1_contributions, bottom=bottom_for_t1, label='T1关键路径优化贡献', 
                       color=current_colors['warning'], alpha=0.8, edgecolor='white')
        
        ax1.set_ylabel('执行时间 (周期)', fontweight='bold')
        ax1.set_title('性能改进贡献分解', fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cases_short, rotation=0)
        ax1.legend(fontsize=16)
        
        # 添加总时间标签
        for i, (remaining, t1_contrib, t2_contrib) in enumerate(zip(remaining_times, t1_contributions, t2_contributions)):
            total = remaining + t1_contrib + t2_contrib
            if total > 0:
                if total >= 1000000:
                    label = f'{total/1000000:.1f}M'
                elif total >= 1000:
                    label = f'{total/1000:.0f}K'
                else:
                    label = f'{int(total)}'
                    
                ax1.annotate(label, xy=(i, total),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 美化图表
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(os.path.join(self.figures_dir, 'problem3_performance_contribution.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✓ 已生成性能改进贡献分析图")
    
    def _get_short_name(self, case_name: str) -> str:
        """获取案例的简短名称"""
        short_name = case_name.replace('FlashAttention', 'FlashAttn')
        return short_name.replace('_Case', '\nCase').replace('_', '\n')
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print("\n" + "=" * 80)
        print("问题三高质量科研可视化分析报告")
        print("=" * 80)
        
        total_baseline = 0
        total_optimized = 0
        total_improvement = 0
        valid_cases = 0
        
        print(f"\n{'测试用例':<25} {'基线运行时间':<15} {'优化运行时间':<15} {'改进比例':<10} {'数据搬运量':<12}")
        print("-" * 75)
        
        for case_name in self.cases:
            if case_name in self.viz_data:
                data = self.viz_data[case_name]
                results = data.get('results', {})
                
                baseline_runtime = results.get('baseline_runtime', 0)
                optimized_runtime = results.get('optimized_runtime', 0)
                spill_cost = results.get('total_spill_cost', 0)
                
                if baseline_runtime > 0:
                    improvement = ((baseline_runtime - optimized_runtime) / baseline_runtime) * 100
                else:
                    improvement = 0
                
                print(f"{case_name:<25} {baseline_runtime:<15,} {optimized_runtime:<15,} {improvement:<9.1f}% {spill_cost:<12,}")
                
                total_baseline += baseline_runtime
                total_optimized += optimized_runtime
                valid_cases += 1
        
        if valid_cases > 0:
            overall_improvement = ((total_baseline - total_optimized) / total_baseline) * 100 if total_baseline > 0 else 0
            
            print("-" * 75)
            print(f"{'总计':<25} {total_baseline:<15,} {total_optimized:<15,} {overall_improvement:<9.1f}% {'':<12}")
            
            print(f"\n关键发现:")
            print(f"• 总体执行时间改进: {overall_improvement:.1f}%")
            print(f"• 平均每案例改进: {overall_improvement:.1f}%")
            print(f"• 成功优化案例数: {valid_cases}/{len(self.cases)}")
            
            # 多目标权衡分析
            avg_runtime_reduction = 0
            avg_spill_increase = 0
            tradeoff_cases = 0
            
            for case_name in self.cases:
                if case_name in self.viz_data:
                    data = self.viz_data[case_name]
                    analysis = data.get('analysis', {})
                    multi_obj = analysis.get('multi_objective_tradeoff', {})
                    
                    runtime_reduction = multi_obj.get('runtime_reduction', 0)
                    spill_increase = multi_obj.get('spill_cost_increase', 0)
                    
                    avg_runtime_reduction += runtime_reduction
                    avg_spill_increase += spill_increase
                    tradeoff_cases += 1
            
            if tradeoff_cases > 0:
                avg_runtime_reduction /= tradeoff_cases
                avg_spill_increase /= tradeoff_cases
                
                print(f"\n多目标权衡效果:")
                print(f"• 平均执行时间减少: {avg_runtime_reduction:.1f}%")
                print(f"• 平均数据搬运量变化: {avg_spill_increase:.1f}%")
                
                if avg_spill_increase <= 10:  # 数据搬运量增加不超过10%
                    print(f"• 权衡效果: 优秀（数据搬运量增幅控制在{avg_spill_increase:.1f}%以内）")
                elif avg_spill_increase <= 20:
                    print(f"• 权衡效果: 良好（数据搬运量增幅为{avg_spill_increase:.1f}%）")
                else:
                    print(f"• 权衡效果: 需改进（数据搬运量增幅达{avg_spill_increase:.1f}%）")


def main():
    """主函数"""
    print("=== 问题三高质量科研可视化分析 ===\n")
    
    # 检查可视化数据是否存在
    viz_data_dir = "Problem3_Visualization_Data"
    if not os.path.exists(viz_data_dir):
        print("未找到可视化数据目录，正在运行问题三求解器生成数据...")
        
        # 运行问题三求解器
        try:
            from hardware_simulator_prob3 import test_problem3
            test_problem3()
        except ImportError:
            print("错误：无法导入hardware_simulator_prob3模块")
            return
        
        if not os.path.exists(viz_data_dir):
            print("错误：无法生成可视化数据，请检查hardware_simulator_prob3.py")
            return
    
    # 创建可视化器并生成图表
    visualizer = Problem3JournalVisualizer()
    
    if not visualizer.viz_data:
        print("错误：未找到任何可视化数据")
        return
    
    # 生成所有可视化图表
    visualizer.create_comprehensive_analysis()
    
    # 生成汇总报告
    visualizer.generate_summary_report()
    
    print(f"\n✓ 所有图表已保存到 {visualizer.figures_dir}/ 目录")
    print("✓ 问题三高质量科研可视化分析完成")


if __name__ == "__main__":
    main()
