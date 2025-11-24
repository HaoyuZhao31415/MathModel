"""
增强版调度结果可视化工具 - 改进算法对比版
创建原算法vs改进算法的对比可视化展示

运行方式：
1. 默认分析（对比6个测试用例）：
   python enhanced_visualizer.py

2. 自定义对比分析：
   from enhanced_visualizer import ImprovedAlgorithmVisualizer
   visualizer = ImprovedAlgorithmVisualizer()
   visualizer.create_comparison_visualization()

输出文件：
- algorithm_comparison_memory.png - 内存使用对比图（6个测试用例）
- algorithm_comparison_details.png - 详细统计信息对比图
- chinese_font_test.png - 中文字体测试图

功能特性：
- 原算法vs改进算法全面对比
- 6个测试用例并排展示
- 节点分布交错显示，便于区分
- 详细性能改进统计
- 智能中文字体检测
"""
#本程序及代码是在人工智能工具辅助下完成的，人工智能工具名称:ChatGPT ，版本:5，开发机构/公司:OpenAI，版本颁布日期2025年8月7日。
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict
from advanced_scheduler import AdvancedScheduler, Node

# 设置顶级期刊绘图风格
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 设置期刊级别的绘图参数
def setup_journal_style():
    """设置顶级科研期刊的绘图风格"""
    # 基础样式设置
    plt.style.use('default')
    
    # 字体设置 - 优先使用微软雅黑，确保中文正常显示
    primary_fonts = ['Microsoft YaHei', 'Arial', 'Helvetica', 'DejaVu Sans']
    
    # 根据操作系统设置备用中文字体
    if platform.system() == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    elif platform.system() == 'Darwin':
        chinese_fonts = ['Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'Heiti SC']
    else:
        chinese_fonts = ['Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    
    # 科研论文级别的rcParams设置 - 字体全面放大
    journal_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': chinese_fonts + primary_fonts,
        'font.size': 20,           # 基础字体大幅加大
        'axes.titlesize': 24,      # 标题字体大幅加大
        'axes.labelsize': 22,      # 轴标签字体大幅加大
        'xtick.labelsize': 20,     # x轴刻度字体大幅加大
        'ytick.labelsize': 20,     # y轴刻度字体大幅加大  
        'legend.fontsize': 20,     # 图例字体大幅加大
        'figure.titlesize': 26,    # 图形标题字体大幅加大
        
        # 线条和标记
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'patch.linewidth': 0.5,
        
        # 坐标轴
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        
        # 刻度
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # 图形
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 600,  # 高DPI用于期刊发表
        'figure.dpi': 120,
        
        # 其他
        'axes.unicode_minus': False,
        'text.usetex': False,  # 避免LaTeX依赖问题
    }
    
    plt.rcParams.update(journal_params)
    
    # 设置seaborn样式增强
    sns.set_palette("husl")
    return True

# 顶级期刊配色方案
JOURNAL_COLORS = {
    # 现代科研期刊配色 - Nature/Science/Cell 风格
    'modern_scientific': {
        'primary': '#2E86AB',      # 深海蓝 - 主要数据
        'secondary': '#A23B72',    # 深紫红 - 次要数据  
        'success': '#F18F01',      # 活力橙 - 成功/改进
        'danger': '#C73E1D',       # 科研红 - 关键/警告
        'warning': '#FFB563',      # 温暖橙 - 注意事项
        'info': '#3D5A80',         # 学术蓝 - 信息展示
        'purple': '#7209B7',       # 深紫 - 特殊标记
        'brown': '#8B5A3C',        # 大地棕 - 辅助色
    },
    
    # 高对比度科研配色 - 适合印刷
    'high_contrast': {
        'primary': '#003f5c',      # 深蓝
        'secondary': '#bc5090',    # 紫红
        'success': '#ff6361',      # 珊瑚红
        'danger': '#ffa600',       # 金橙
        'warning': '#665191',      # 深紫
        'info': '#2f4b7c',         # 海军蓝
        'purple': '#a05195',       # 梅红
        'brown': '#d45087',        # 玫红
    },
    
    # 经典学术配色 - IEEE/ACM 风格
    'academic_classic': {
        'primary': '#1f4e79',      # 学术蓝
        'secondary': '#c5504b',    # 学术红
        'success': '#70ad47',      # 学术绿
        'danger': '#ffc000',       # 学术黄
        'warning': '#7030a0',      # 学术紫
        'info': '#0070c0',         # 信息蓝
        'purple': '#843c0c',       # 棕橙
        'brown': '#538135',        # 深绿
    }
}

# 初始化期刊风格
setup_journal_style()

# 获取当前使用的配色方案 - 使用现代科研期刊配色
current_colors = JOURNAL_COLORS['modern_scientific']


class EnhancedVisualizer:
    """增强版可视化器"""
    
    def __init__(self, graph_file: str, schedule: list):
        """初始化"""
        self.scheduler = AdvancedScheduler(graph_file)
        self.schedule = schedule
        
        # 完整模拟整个调度过程
        self.full_memory_history = []
        self.full_step_info = []
        self._simulate_full_schedule()
    
    def get_nodes_by_range(self, start_idx: int = 0, end_idx: int = 500):
        """根据调度序列范围获取节点"""
        end_idx = min(end_idx, len(self.schedule))
        target_nodes = self.schedule[start_idx:end_idx]
        return target_nodes
    
    def find_interesting_nodes(self, num_nodes: int = 30, start_idx: int = 0):
        """找到更有趣的节点组合，包含不同类型的操作"""
        # 扩大搜索范围以适应大规模节点
        search_range = min(start_idx + num_nodes * 3, len(self.schedule))
        
        # 分类收集不同类型的节点
        alloc_nodes = []
        free_nodes = []
        operation_nodes = []
        
        for i, node_id in enumerate(self.schedule[start_idx:search_range], start_idx):
            node = self.scheduler.nodes[node_id]
            if node.op == 'ALLOC' and node.is_l1_or_ub_cache():
                alloc_nodes.append((i, node_id))
            elif node.op == 'FREE' and node.is_l1_or_ub_cache():
                free_nodes.append((i, node_id))
            elif not node.is_cache_node():
                operation_nodes.append((i, node_id))
        
        # 选择有代表性的节点
        selected_nodes = []
        
        # 根据节点数量调整比例
        if num_nodes <= 50:
            alloc_count = min(num_nodes // 3, len(alloc_nodes))
            op_count = min(num_nodes // 3, len(operation_nodes)) 
            free_count = min(num_nodes // 3, len(free_nodes))
        else:
            # 大规模节点时更均匀分布
            alloc_count = min(num_nodes // 4, len(alloc_nodes))
            op_count = min(num_nodes // 2, len(operation_nodes))
            free_count = min(num_nodes // 4, len(free_nodes))
        
        # 选择节点
        selected_nodes.extend([node_id for _, node_id in alloc_nodes[:alloc_count]])
        selected_nodes.extend([node_id for _, node_id in operation_nodes[:op_count]])
        selected_nodes.extend([node_id for _, node_id in free_nodes[:free_count]])
        
        # 如果还不够，直接按顺序添加
        if len(selected_nodes) < num_nodes:
            remaining = num_nodes - len(selected_nodes)
            end_range = min(start_idx + num_nodes, len(self.schedule))
            for node_id in self.schedule[start_idx:end_range]:
                if len(selected_nodes) >= num_nodes:
                    break
                if node_id not in selected_nodes:
                    selected_nodes.append(node_id)
        
        return selected_nodes[:num_nodes]
    
    def _simulate_full_schedule(self):
        """完整模拟整个调度过程，记录所有步骤的缓存状态"""
        current_memory = 0
        
        for i, node_id in enumerate(self.schedule):
            node = self.scheduler.nodes[node_id]
            
            # 记录步骤信息
            step_info = {
                'step': i,
                'node_id': node_id,
                'op': node.op,
                'is_cache': node.is_cache_node(),
                'is_l1_ub': node.is_l1_or_ub_cache() if node.is_cache_node() else False,
                'memory_delta': 0,
                'cache_type': getattr(node, 'cache_type', ''),
                'size': getattr(node, 'size', 0),
                'pipe': getattr(node, 'pipe', ''),
                'cycles': getattr(node, 'cycles', 0),
                'bufs': getattr(node, 'bufs', [])
            }
            
            # 计算内存变化
            if node.is_cache_node() and node.is_l1_or_ub_cache():
                delta = node.memory_delta()
                current_memory += delta
                step_info['memory_delta'] = delta
            
            # 记录当前内存状态
            self.full_memory_history.append(current_memory)
            self.full_step_info.append(step_info)
    
    def create_comprehensive_visualization(self, start_idx: int = 0, end_idx: int = None):
        """创建综合可视化展示 - 单图模式"""
        # 设置默认范围
        if end_idx is None:
            end_idx = len(self.schedule)
        
        end_idx = min(end_idx, len(self.schedule))
        start_idx = max(0, start_idx)
        
        if start_idx >= end_idx:
            print("错误：起始索引必须小于结束索引")
            return None
        
        print(f"可视化节点范围: [{start_idx}, {end_idx}), 共{end_idx - start_idx}个节点")
        
        # 提取指定范围的数据
        range_steps = list(range(start_idx, end_idx))
        range_memory = self.full_memory_history[start_idx:end_idx]
        range_info = self.full_step_info[start_idx:end_idx]
        
        # 创建单图布局
        fig = plt.figure(figsize=(20, 12))
        
        # 创建子图网格 2x3
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # 1. 主要内存使用时间线 (占据上半部分)
        ax1 = fig.add_subplot(gs[0, :])
        self._draw_main_memory_timeline(ax1, range_steps, range_memory, range_info, start_idx)
        
        # 2. 节点类型分布饼图
        ax2 = fig.add_subplot(gs[1, 0])
        self._draw_node_type_pie(ax2, range_info)
        
        # 3. 执行单元使用情况
        ax3 = fig.add_subplot(gs[1, 1])
        self._draw_execution_units_bar(ax3, range_info)
        
        # 4. 关键统计信息
        ax4 = fig.add_subplot(gs[1, 2])
        self._draw_key_statistics(ax4, range_memory, range_info, start_idx, end_idx)
        
        plt.suptitle(f'调度结果分析 - 步骤[{start_idx}:{end_idx}] (总共{len(self.schedule)}步)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        filename = f'schedule_analysis_{start_idx}_{end_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印关键统计信息
        self._print_range_statistics(range_info, range_memory, start_idx, end_idx)
        
        return filename
    
    def _draw_main_memory_timeline(self, ax, range_steps, range_memory, range_info, start_idx):
        """绘制主要内存使用时间线"""
        # 绘制内存使用曲线
        ax.plot(range_steps, range_memory, 'b-', linewidth=2, alpha=0.8, label='内存使用量')
        ax.fill_between(range_steps, range_memory, alpha=0.3, color='lightblue')
        
        # 标记峰值
        if range_memory:
            peak_memory = max(range_memory)
            peak_idx = range_memory.index(peak_memory)
            peak_step = range_steps[peak_idx]
            
            ax.plot(peak_step, peak_memory, 'ro', markersize=8)
            ax.annotate(f'峰值: {peak_memory}\n步骤: {peak_step}', 
                       xy=(peak_step, peak_memory),
                       xytext=(peak_step + len(range_steps)//10, peak_memory),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red', fontweight='bold')
            
            # 统计信息
            avg_memory = np.mean(range_memory)
            ax.axhline(y=avg_memory, color='green', linestyle='--', alpha=0.7, label=f'平均: {avg_memory:.0f}')
        
        # 在顶部标记不同类型的节点
        alloc_steps = []
        free_steps = []
        op_steps = []
        
        for i, info in enumerate(range_info):
            step = range_steps[i]
            if info['op'] == 'ALLOC' and info['is_l1_ub']:
                alloc_steps.append(step)
            elif info['op'] == 'FREE' and info['is_l1_ub']:
                free_steps.append(step)
            elif not info['is_cache']:
                op_steps.append(step)
        
        # 在图的顶部绘制节点类型标记
        max_memory = max(range_memory) if range_memory else 1000
        marker_height = max_memory * 1.1
        
        if alloc_steps:
            ax.scatter(alloc_steps, [marker_height] * len(alloc_steps), 
                      c='red', marker='^', s=30, alpha=0.7, label=f'ALLOC ({len(alloc_steps)})')
        if free_steps:
            ax.scatter(free_steps, [marker_height] * len(free_steps), 
                      c='green', marker='v', s=30, alpha=0.7, label=f'FREE ({len(free_steps)})')
        if op_steps:
            ax.scatter(op_steps, [marker_height] * len(op_steps), 
                      c='blue', marker='o', s=20, alpha=0.5, label=f'操作 ({len(op_steps)})')
        
        ax.set_xlabel('调度步骤')
        ax.set_ylabel('内存使用量')
        ax.set_title('内存使用量时间线及节点分布', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置合理的y轴范围
        if range_memory:
            ax.set_ylim(0, max_memory * 1.2)
    
    def _draw_node_type_pie(self, ax, range_info):
        """绘制节点类型分布饼图"""
        type_counts = {'ALLOC': 0, 'FREE': 0, '操作': 0, '其他': 0}
        
        for info in range_info:
            if info['op'] == 'ALLOC':
                type_counts['ALLOC'] += 1
            elif info['op'] == 'FREE':
                type_counts['FREE'] += 1
            elif not info['is_cache']:
                type_counts['操作'] += 1
            else:
                type_counts['其他'] += 1
        
        # 过滤掉数量为0的类型
        filtered_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        if filtered_counts:
            colors = ['#ff6b6b', '#51cf66', '#339af0', '#ffd93d']
            wedges, texts, autotexts = ax.pie(filtered_counts.values(), 
                                             labels=filtered_counts.keys(),
                                             colors=colors[:len(filtered_counts)],
                                             autopct='%1.1f%%', 
                                             startangle=90)
            
            # 美化文本
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('节点类型分布', fontsize=12, fontweight='bold')
    
    def _draw_execution_units_bar(self, ax, range_info):
        """绘制执行单元使用情况"""
        pipe_counts = defaultdict(int)
        
        for info in range_info:
            if not info['is_cache'] and info['pipe']:
                pipe_counts[info['pipe']] += 1
        
        if pipe_counts:
            pipes = list(pipe_counts.keys())
            counts = list(pipe_counts.values())
            
            bars = ax.bar(pipes, counts, color='#339af0', alpha=0.7, edgecolor='black')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('使用次数')
            ax.set_xlabel('执行单元')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, '该范围内无操作节点', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('执行单元使用频次', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _draw_key_statistics(self, ax, range_memory, range_info, start_idx, end_idx):
        """绘制关键统计信息"""
        ax.axis('off')  # 隐藏坐标轴
        
        # 计算统计信息
        total_steps = len(range_info)
        alloc_count = sum(1 for info in range_info if info['op'] == 'ALLOC' and info['is_l1_ub'])
        free_count = sum(1 for info in range_info if info['op'] == 'FREE' and info['is_l1_ub'])
        op_count = sum(1 for info in range_info if not info['is_cache'])
        
        total_allocated = sum(info['size'] for info in range_info if info['op'] == 'ALLOC' and info['is_l1_ub'])
        total_freed = sum(info['size'] for info in range_info if info['op'] == 'FREE' and info['is_l1_ub'])
        
        peak_memory = max(range_memory) if range_memory else 0
        avg_memory = np.mean(range_memory) if range_memory else 0
        final_memory = range_memory[-1] if range_memory else 0
        
        # 创建统计信息文本
        stats_text = f"""
关键统计信息:

范围: 步骤 {start_idx} - {end_idx}
总步数: {total_steps}

节点分布:
• ALLOC节点: {alloc_count}
• FREE节点: {free_count}  
• 操作节点: {op_count}

内存统计:
• 峰值使用: {peak_memory:,}
• 平均使用: {avg_memory:,.0f}
• 结束时使用: {final_memory:,}

缓存操作:
• 总分配量: {total_allocated:,}
• 总释放量: {total_freed:,}
• 净增长: {total_allocated - total_freed:,}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='sans-serif',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        ax.set_title('关键统计信息', fontsize=12, fontweight='bold')
    
    def _print_range_statistics(self, range_info, range_memory, start_idx, end_idx):
        """打印范围统计信息"""
        print(f"\n=== 范围 [{start_idx}:{end_idx}] 统计信息 ===")
        
        # 节点类型统计
        alloc_count = sum(1 for info in range_info if info['op'] == 'ALLOC' and info['is_l1_ub'])
        free_count = sum(1 for info in range_info if info['op'] == 'FREE' and info['is_l1_ub'])
        op_count = sum(1 for info in range_info if not info['is_cache'])
        other_count = len(range_info) - alloc_count - free_count - op_count
        
        print(f"节点类型分布:")
        print(f"  ALLOC节点: {alloc_count}")
        print(f"  FREE节点: {free_count}")
        print(f"  操作节点: {op_count}")
        print(f"  其他节点: {other_count}")
        
        # 内存统计
        if range_memory:
            peak_memory = max(range_memory)
            avg_memory = np.mean(range_memory)
            final_memory = range_memory[-1]
            
            print(f"\n内存使用统计:")
            print(f"  峰值: {peak_memory:,}")
            print(f"  平均: {avg_memory:,.0f}")
            print(f"  结束时: {final_memory:,}")
        
        # 执行单元统计
        pipe_counts = defaultdict(int)
        for info in range_info:
            if not info['is_cache'] and info['pipe']:
                pipe_counts[info['pipe']] += 1
        
        if pipe_counts:
            print(f"\n执行单元使用:")
            for pipe, count in sorted(pipe_counts.items()):
                print(f"  {pipe}: {count}")
    


def test_chinese_font():
    """测试中文字体显示"""
    try:
        import matplotlib.font_manager as fm
        
        # 获取当前字体设置
        current_font = plt.rcParams['font.sans-serif'][0]
        print(f"当前使用字体: {current_font}")
        
        # 创建简单测试图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.7, '中文显示测试', fontsize=16, ha='center', fontweight='bold')
        ax.text(0.5, 0.5, '节点类型分布图表', fontsize=14, ha='center')
        ax.text(0.5, 0.3, '内存使用量统计信息', fontsize=12, ha='center')
        ax.text(0.5, 0.1, 'ALLOC/FREE/操作节点', fontsize=10, ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('中文字体测试图', fontsize=18, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 中文字体测试完成，已生成 chinese_font_test.png")
        return True
        
    except Exception as e:
        print(f"✗ 中文字体测试失败: {e}")
        return False


class AlgorithmVisualizer:
    """算法可视化器 - 支持启发式与二分法对比"""
    
    def __init__(self):
        """初始化可视化器"""
        self.test_cases = [
            "Matmul_Case0",
            "Matmul_Case1", 
            "FlashAttention_Case0",
            "FlashAttention_Case1",
            "Conv_Case0",
            "Conv_Case1"
        ]
        
        # 存储两种算法的数据
        self.heuristic_data = {}
        self.binary_data = {}
        
        self._load_all_results()
    
    def _load_all_results(self):
        """优先从保存的可视化数据中加载结果，如果不存在则实时运行调度算法"""
        print("正在加载调度算法结果...")
        
        for case_name in self.test_cases:
            try:
                # 首先尝试从保存的可视化数据中加载
                viz_data_path = f"Problem1_Visualization_Data/{case_name}_visualization_data.json"
                loaded_from_file = False
                
                try:
                    with open(viz_data_path, 'r') as f:
                        viz_data = json.load(f)
                    
                    print(f"  -> 从保存的数据加载 {case_name}...")
                    
                    # 加载启发式数据
                    h_data = viz_data['heuristic']
                    self.heuristic_data[case_name] = {
                        'visualizer': None,  # 不需要创建visualizer对象
                        'memory_history': h_data['memory_history'],
                        'step_info': h_data['step_info'],
                        'schedule': h_data['schedule'],
                        'max_v_stay': h_data['max_v_stay']
                    }
                    
                    # 加载二分法数据
                    b_data = viz_data['binary']
                    self.binary_data[case_name] = {
                        'visualizer': None,  # 不需要创建visualizer对象
                        'memory_history': b_data['memory_history'],
                        'step_info': b_data['step_info'],
                        'schedule': b_data['schedule'],
                        'max_v_stay': b_data['max_v_stay']
                    }
                    
                    print(f"✓ 已从文件加载 {case_name} (启发式: {h_data['max_v_stay']:,}, 二分法: {b_data['max_v_stay']:,})")
                    loaded_from_file = True
                    
                except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                    print(f"  -> 无法从文件加载 {case_name}，将重新运行算法... ({e})")
                
                # 如果无法从文件加载，则重新运行调度算法
                if not loaded_from_file:
                    graph_file = f"data/Json_version/{case_name}.json"
                    scheduler = AdvancedScheduler(graph_file)
                    
                    # 1. 启发式算法
                    print(f"  -> 正在为 {case_name} 运行启发式算法...")
                    schedule_h, max_v_stay_h = scheduler.schedule(show_progress=False)
                    print(f"     启发式完成. 峰值 V_stay: {max_v_stay_h:,}")
                    
                    visualizer_h = EnhancedVisualizer(graph_file, schedule_h)
                    self.heuristic_data[case_name] = {
                        'visualizer': visualizer_h,
                        'memory_history': visualizer_h.full_memory_history,
                        'step_info': visualizer_h.full_step_info,
                        'schedule': visualizer_h.schedule,
                        'max_v_stay': max_v_stay_h
                    }
                    
                    # 2. 二分法算法
                    print(f"  -> 正在为 {case_name} 运行二分法算法...")
                    schedule_b, max_v_stay_b = scheduler.schedule_problem1_binary_search(show_progress=False)
                    print(f"     二分法完成. 峰值 V_stay: {max_v_stay_b:,}")
                    
                    visualizer_b = EnhancedVisualizer(graph_file, schedule_b)
                    self.binary_data[case_name] = {
                        'visualizer': visualizer_b,
                        'memory_history': visualizer_b.full_memory_history,
                        'step_info': visualizer_b.full_step_info,
                        'schedule': visualizer_b.schedule,
                        'max_v_stay': max_v_stay_b
                    }
                
                # 从数据中获取峰值进行打印
                h_peak = self.heuristic_data[case_name]['max_v_stay']
                b_peak = self.binary_data[case_name]['max_v_stay']
                print(f"✓ 已处理 {case_name} (启发式: {h_peak:,}, 二分法: {b_peak:,})")
                
            except Exception as e:
                print(f"✗ 处理 {case_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    def create_visualization(self):
        """创建算法对比可视化"""
        print("\n开始生成算法对比可视化...")
        
        # 创建第一张图：二分法内存使用量及节点分布（6个Case）
        self._create_binary_memory_figure()
        
        # 创建第二张图：详细统计信息（仅二分法，6个Case柱状图）
        self._create_binary_statistics_figure()
        
        # 创建第三张图：性能改进总结（与基准数据对比）
        self._create_improvement_summary_figure()
        
        print("✓ 算法对比可视化生成完成")
    
    def _create_binary_memory_figure(self):
        """创建二分法内存使用量及节点分布图（6个Case）- 现代科研期刊风格"""
        # 现代科研期刊配色方案
        journal_colors = {
            'primary': current_colors['primary'],      # 主要曲线 - 深海蓝
            'secondary': current_colors['secondary'],  # 次要元素 - 深紫红
            'alloc': current_colors['danger'],         # ALLOC节点 - 科研红
            'free': current_colors['success'],         # FREE节点 - 活力橙
            'operation': current_colors['purple'],     # 操作节点 - 深紫
            'average': current_colors['warning'],      # 平均线 - 温暖橙
            'peak': current_colors['info']             # 峰值标记 - 学术蓝
        }
        
        # 创建科研论文级图形 - 适合期刊发表的尺寸
        fig_width = 20    # 增大宽度适合论文版面
        fig_height = 14   # 增大高度确保清晰度
        fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        
        # 去掉大标题
        
        # 子图标题映射（保留英文名称）
        title_mapping = {
            'Matmul_Case0': 'Matmul Case 0',
            'Matmul_Case1': 'Matmul Case 1', 
            'FlashAttention_Case0': 'FlashAttention Case 0',
            'FlashAttention_Case1': 'FlashAttention Case 1',
            'Conv_Case0': 'Conv Case 0',
            'Conv_Case1': 'Conv Case 1'
        }
        
        # 重新安排子图顺序：左列矩阵乘法，中列FlashAttention，右列卷积
        case_order = [
            'Matmul_Case0', 'FlashAttention_Case0', 'Conv_Case0',       # 上排
            'Matmul_Case1', 'FlashAttention_Case1', 'Conv_Case1'        # 下排
        ]
        
        # 子图标签
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        
        for i, case_name in enumerate(case_order):
            if case_name not in self.binary_data:
                continue
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 绘制二分法结果
            if case_name in self.binary_data:
                self._draw_journal_quality_plot(ax, case_name, self.binary_data[case_name], 
                                               journal_colors, title_mapping.get(case_name, case_name), i)
            else:
                ax.text(0.5, 0.5, f'{title_mapping.get(case_name, case_name)}\n数据加载失败', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
            
            # 在左下角添加子图标签
            ax.text(0.02, 0.02, subplot_labels[i], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='bottom', ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 期刊级布局调整
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        
        # 保存高质量图片
        filename = 'journal_binary_memory_analysis.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"✓ 已生成期刊级质量图片: {filename}")
    
    def _draw_journal_quality_plot(self, ax, case_name, data, colors, display_title, subplot_idx=0):
        """绘制期刊级质量的内存使用图 - 现代科研期刊风格"""
        memory_history = data['memory_history']
        step_info = data['step_info']
        max_v_stay = data['max_v_stay']
        
        if not memory_history:
            ax.text(0.5, 0.5, '无数据可用', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # 创建步骤索引
        steps = np.array(range(len(memory_history)))
        memory_array = np.array(memory_history)
        
        # 绘制主要内存使用曲线 - 马卡龙样式
        ax.plot(steps, memory_array, color=colors['primary'], linewidth=2.5, 
               alpha=0.9, label='内存使用量', zorder=3)
        
        # 渐变填充区域
        ax.fill_between(steps, memory_array, alpha=0.25, color=colors['primary'], zorder=1)
        
        # 绘制期刊级节点标记
        self._draw_journal_node_markers(ax, step_info, steps, colors, 
                                      max(memory_history) if memory_history else 0)
        
        # 统计信息计算
        avg_memory = np.mean(memory_array)
        std_memory = np.std(memory_array)
        
        # 绘制统计线条 - 期刊级样式
        ax.axhline(y=avg_memory, color=colors['average'], linestyle='--', 
                  linewidth=1.5, alpha=0.8, label=f'均值: {avg_memory:.0f}', zorder=2)
        
        # 标准差带（可选）
        if std_memory > avg_memory * 0.1:  # 只在标准差显著时显示
            ax.fill_between(steps, avg_memory - std_memory, avg_memory + std_memory,
                          alpha=0.1, color=colors['average'], zorder=0)
        
        # 峰值标记 - 期刊级样式
        if max_v_stay > 0:
            peak_idx = memory_history.index(max_v_stay)
            
            # 峰值点标记
            ax.plot(peak_idx, max_v_stay, marker='D', color=colors['peak'], 
                   markersize=8, markeredgecolor='white', markeredgewidth=1.5, 
                   zorder=4, label=f'峰值: {max_v_stay:,}')
            
            # 峰值标注 - 更简洁的样式
            offset_x = len(steps) * 0.05
            offset_y = max_v_stay * 0.05
            ax.annotate(f'{max_v_stay:,}', 
                       xy=(peak_idx, max_v_stay),
                       xytext=(peak_idx + offset_x, max_v_stay + offset_y),
                       fontsize=9, fontweight='bold', color=colors['peak'],
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                               edgecolor=colors['peak'], alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color=colors['peak'], 
                                     connectionstyle="arc3,rad=0.1"))
        
        # 去掉子标题，只保留轴标签 - 字体放大
        ax.set_ylabel('内存使用量 (字节)', fontsize=16)
        ax.set_xlabel('调度步骤', fontsize=16)
        
        # 坐标轴美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 期刊级图例 - 调整位置避免遮挡散点，去掉散点描边
        legend_elements = [
            plt.Line2D([0], [0], color=colors['primary'], linewidth=2.5, label='内存使用量'),
            plt.Line2D([0], [0], color=colors['average'], linestyle='--', linewidth=1.5, label='均值'),
            plt.scatter([], [], c=colors['alloc'], marker='^', s=40, alpha=0.8, label='ALLOC', edgecolors='none'),
            plt.scatter([], [], c=colors['free'], marker='v', s=40, alpha=0.8, label='FREE', edgecolors='none'),
            plt.scatter([], [], c=colors['operation'], marker='o', s=25, alpha=0.8, label='操作', edgecolors='none')
        ]
        
        # 图例统一右下角 - 字体放大
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14, 
                 frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        
        # 自动调整y轴范围以显示所有重要信息
        y_max = max(memory_array) * 1.15
        y_min = min(0, min(memory_array) * 0.95)
        ax.set_ylim(y_min, y_max)
        
        # 格式化刻度标签
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    def _draw_journal_node_markers(self, ax, step_info, steps, colors, max_memory):
        """绘制期刊级质量的节点标记 - 现代科研期刊风格"""
        # 分类收集节点
        alloc_steps = []
        free_steps = []
        op_steps = []
        
        for i, info in enumerate(step_info):
            if i >= len(steps):
                break
            step = steps[i]
            if info['op'] == 'ALLOC' and info['is_l1_ub']:
                alloc_steps.append(step)
            elif info['op'] == 'FREE' and info['is_l1_ub']:
                free_steps.append(step)
            elif not info['is_cache']:
                op_steps.append(step)
        
        # 计算标记位置 - 使用不同高度避免重叠
        base_height = max_memory * 1.05 if max_memory > 0 else 1.05
        level_spacing = max_memory * 0.03 if max_memory > 0 else 0.03
        
        alloc_height = base_height + 2 * level_spacing
        free_height = base_height + level_spacing
        op_height = base_height
        
        # 马卡龙级标记样式 - 无描边
        marker_size = 20
        alpha = 0.8
        
        # 绘制节点标记 - 去掉描边
        if alloc_steps:
            ax.scatter(alloc_steps, [alloc_height] * len(alloc_steps), 
                      c=colors['alloc'], marker='^', s=marker_size, alpha=alpha,
                      edgecolors='none', zorder=3)
        if free_steps:
            ax.scatter(free_steps, [free_height] * len(free_steps), 
                      c=colors['free'], marker='v', s=marker_size, alpha=alpha,
                      edgecolors='none', zorder=3)
        if op_steps:
            # 为了避免过多的操作节点标记，只显示关键的操作节点
            if len(op_steps) > 50:
                # 对于太多的操作节点，使用采样
                sample_indices = np.linspace(0, len(op_steps)-1, 30, dtype=int)
                op_steps_sampled = [op_steps[i] for i in sample_indices]
            else:
                op_steps_sampled = op_steps
                
            ax.scatter(op_steps_sampled, [op_height] * len(op_steps_sampled), 
                      c=colors['operation'], marker='o', s=marker_size//2, alpha=alpha,
                      edgecolors='none', zorder=3)
    
    def _draw_single_algorithm_plot(self, ax, case_name, data, colors, algorithm_name, side):
        """绘制单个算法的内存使用图 - 兼容性保留函数"""
        # 为了保持向后兼容性，调用新的期刊级绘图函数
        self._draw_journal_quality_plot(ax, case_name, data, colors, case_name)
    
    def _create_memory_overview_figure(self):
        """创建内存使用量图（6行子图）"""
        fig, axes = plt.subplots(6, 1, figsize=(20, 24))
        fig.suptitle('高级调度算法 - 内存使用量及节点分布', fontsize=20, fontweight='bold', y=0.98)
        
        colors = {
            'memory_curve': '#51cf66',
            'alloc': '#ffd93d',
            'free': '#339af0', 
            'operation': '#845ec2'
        }
        
        for i, case_name in enumerate(self.test_cases):
            ax = axes[i]
            
            if case_name not in self.results_data:
                ax.text(0.5, 0.5, f'{case_name}\n数据加载失败', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(case_name, fontsize=14, fontweight='bold')
                continue
            
            # 获取算法数据
            memory_history = self.results_data[case_name]['memory_history']
            step_info = self.results_data[case_name]['step_info']
            
            # 创建步骤索引
            steps = list(range(len(memory_history)))
            
            # 绘制内存使用曲线
            ax.plot(steps, memory_history, color=colors['memory_curve'], 
                   linewidth=2, alpha=0.8, label='内存使用')
            
            # 填充区域
            ax.fill_between(steps, memory_history, alpha=0.3, color=colors['memory_curve'])
            
            # 绘制节点分布
            self._draw_staggered_node_markers(ax, step_info, steps, 
                                            colors, max(memory_history) if memory_history else 0)
            
            # 标记峰值
            peak_memory = max(memory_history) if memory_history else 0
            
            if peak_memory > 0:
                peak_idx = memory_history.index(peak_memory)
                ax.plot(peak_idx, peak_memory, 's', color=colors['memory_curve'], 
                       markersize=8, markeredgecolor='white', markeredgewidth=2)
                
                # 标注峰值
                ax.annotate(f'峰值: {peak_memory:,}', 
                           xy=(peak_idx, peak_memory),
                           xytext=(peak_idx + len(steps)//10 if steps else 0, peak_memory),
                           arrowprops=dict(arrowstyle='->', color=colors['memory_curve']),
                           fontsize=10, color=colors['memory_curve'], fontweight='bold')
            
            # 设置标题和标签
            title = f'{case_name} - 峰值: {peak_memory:,}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('内存使用量')
            ax.grid(True, alpha=0.3)
            
            # 统计信息
            avg_memory = np.mean(memory_history) if memory_history else 0
            ax.axhline(y=avg_memory, color='orange', linestyle='--', alpha=0.7, 
                      label=f'平均: {avg_memory:.0f}')
            
            # 只在最下面的图显示图例和x轴标签
            if i == len(self.test_cases) - 1:
                ax.set_xlabel('调度步骤')
                # 添加节点类型图例
                legend_elements = [
                    plt.Line2D([0], [0], color=colors['memory_curve'], linewidth=2, label='内存使用'),
                    plt.Line2D([0], [0], color='orange', linestyle='--', label='平均值'),
                    plt.scatter([], [], c=colors['alloc'], marker='^', s=50, label='ALLOC节点'),
                    plt.scatter([], [], c=colors['free'], marker='v', s=50, label='FREE节点'),
                    plt.scatter([], [], c=colors['operation'], marker='o', s=30, label='操作节点')
                ]
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.tick_params(labelbottom=False)
        
        plt.tight_layout()
        plt.savefig('scheduling_memory_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 已生成 scheduling_memory_overview.png")
    
    def _draw_staggered_node_markers(self, ax, step_info, steps, colors, max_memory):
        """绘制交错的节点标记"""
        # 分类收集节点
        alloc_steps = []
        free_steps = []
        op_steps = []
        
        for i, info in enumerate(step_info):
            step = steps[i]
            if info['op'] == 'ALLOC' and info['is_l1_ub']:
                alloc_steps.append(step)
            elif info['op'] == 'FREE' and info['is_l1_ub']:
                free_steps.append(step)
            elif not info['is_cache']:
                op_steps.append(step)
        
        # 计算交错的高度位置
        base_height = max_memory * 1.05 if max_memory > 0 else 1.05
        offset = max_memory * 0.02 if max_memory > 0 else 0.02
        
        alloc_height = base_height + 2*offset
        free_height = base_height + offset
        op_height = base_height
        alpha = 0.9
        size = 20
        
        # 绘制节点标记
        if alloc_steps:
            ax.scatter(alloc_steps, [alloc_height] * len(alloc_steps), 
                      c=colors['alloc'], marker='^', s=size, alpha=alpha,
                      edgecolors='black', linewidth=0.5)
        if free_steps:
            ax.scatter(free_steps, [free_height] * len(free_steps), 
                      c=colors['free'], marker='v', s=size, alpha=alpha,
                      edgecolors='black', linewidth=0.5)
        if op_steps:
            ax.scatter(op_steps, [op_height] * len(op_steps), 
                      c=colors['operation'], marker='o', s=size//2, alpha=alpha,
                      edgecolors='black', linewidth=0.5)
    
    def _create_binary_statistics_figure(self):
        """创建详细统计信息图（仅二分法，6个Case柱状图）- 现代科研期刊风格"""
        # 科研论文级图形设置
        fig_width = 16    # 增大宽度
        fig_height = 12   # 增大高度
        fig, (ax_main, ax_memory) = plt.subplots(2, 1, figsize=(fig_width, fig_height), 
                                                height_ratios=[3, 1], gridspec_kw={'hspace': 0.3})
        
        # 去掉大标题
        
        # 准备数据
        cases = []
        alloc_counts = []
        free_counts = []
        op_counts = []
        peak_memories = []
        
        # 标题映射（保留英文名称）
        title_mapping = {
            'Matmul_Case0': 'Matmul\nCase 0',
            'Matmul_Case1': 'Matmul\nCase 1', 
            'FlashAttention_Case0': 'FlashAttention\nCase 0',
            'FlashAttention_Case1': 'FlashAttention\nCase 1',
            'Conv_Case0': 'Conv\nCase 0',
            'Conv_Case1': 'Conv\nCase 1'
        }
        
        for case_name in self.test_cases:
            if case_name in self.binary_data:
                cases.append(title_mapping.get(case_name, case_name))
                step_info = self.binary_data[case_name]['step_info']
                peak_memories.append(self.binary_data[case_name]['max_v_stay'])
                
                # 统计节点类型
                alloc_count = sum(1 for info in step_info if info['op'] == 'ALLOC' and info['is_l1_ub'])
                free_count = sum(1 for info in step_info if info['op'] == 'FREE' and info['is_l1_ub'])
                op_count = sum(1 for info in step_info if not info['is_cache'])
                
                alloc_counts.append(alloc_count)
                free_counts.append(free_count)
                op_counts.append(op_count)
        
        # 现代科研分组柱状图设计
        x = np.arange(len(cases))
        width = 0.25
        
        # 使用现代科研配色
        colors = [current_colors['danger'], current_colors['success'], current_colors['primary']]
        alphas = [0.9, 0.9, 0.9]
        
        bars1 = ax_main.bar(x - width, alloc_counts, width, label='ALLOC节点', 
                          color=colors[0], alpha=alphas[0], edgecolor='white', linewidth=0.5)
        bars2 = ax_main.bar(x, free_counts, width, label='FREE节点', 
                          color=colors[1], alpha=alphas[1], edgecolor='white', linewidth=0.5)
        bars3 = ax_main.bar(x + width, op_counts, width, label='操作节点', 
                          color=colors[2], alpha=alphas[2], edgecolor='white', linewidth=0.5)
        
        # 期刊级数值标签
        for bars, color in zip([bars1, bars2, bars3], colors):
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # 只在有数据时显示标签
                    ax_main.annotate(f'{height:,}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                                   ha='center', va='bottom', fontsize=12,
                                   fontweight='bold', color='black')
        
        # 主图设置（中文）- 字体放大
        ax_main.set_xlabel('测试用例', fontsize=18, fontweight='bold')
        ax_main.set_ylabel('节点数量', fontsize=18, fontweight='bold')
        # 去掉子标题
        ax_main.set_xticks(x)
        ax_main.set_xticklabels(cases, fontsize=16)
        
        # 调整纵轴范围确保图例不遮挡数据
        ax_main.set_ylim(0, 16000)
        
        # 期刊级图例移至右上角 - 字体放大
        ax_main.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True, 
                      shadow=False, framealpha=0.9)
        
        # 坐标轴美化
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        
        # 添加主图标签 - 移至左下角
        ax_main.text(0.02, 0.02, '(a)', transform=ax_main.transAxes, 
                    fontsize=14, fontweight='bold', va='bottom', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 第二个子图：内存峰值条形图
        self._create_memory_peak_subplot(ax_memory, cases, peak_memories)
        
        # 添加子图标签 - 移至左下角
        ax_memory.text(0.02, 0.02, '(b)', transform=ax_memory.transAxes, 
                      fontsize=14, fontweight='bold', va='bottom', ha='left',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 保存高质量图片
        filename = 'journal_binary_statistics.png'
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(filename, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"✓ 已生成期刊级统计图片: {filename}")
        self._print_binary_statistics()
    
    def _create_memory_peak_subplot(self, ax, cases, peak_memories):
        """创建内存峰值子图"""
        x = np.arange(len(cases))
        
        # 统一颜色的条形图
        bars = ax.bar(x, peak_memories, color=current_colors['info'], alpha=0.8,
                     edgecolor='white', linewidth=0.8)
        
        # 数值标签
        for i, (bar, peak) in enumerate(zip(bars, peak_memories)):
            height = bar.get_height()
            ax.annotate(f'{peak:,}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12,
                       fontweight='bold', color='black')
        
        # 子图设置（中文）- 字体放大
        ax.set_xlabel('测试用例', fontsize=16)
        ax.set_ylabel('峰值内存 (字节)', fontsize=16)
        # 去掉子标题
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=14)
        
        # 坐标轴美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        
        # 格式化y轴标签
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    def _draw_case_statistics(self, ax, data, algorithm_name):
        """绘制单个案例的统计信息"""
        step_info = data['step_info']
        peak_memory = data['max_v_stay']
        
        # 创建内部网格 1x2
        inner_grid = ax.get_subplotspec().subgridspec(1, 2, wspace=0.1)
        
        ax_pie = ax.figure.add_subplot(inner_grid[0])
        ax_bar = ax.figure.add_subplot(inner_grid[1])

        # 1. 绘制节点类型分布环形图
        self._draw_node_type_donut(ax_pie, step_info, peak_memory)
        
        # 2. 绘制执行单元使用情况条形图
        self._draw_pipe_usage_bar(ax_bar, step_info)
    
    def _create_improvement_summary_figure(self):
        """创建性能改进总结图 - 拆分为两个独立图表（中文马卡龙风格）"""
        # 四种算法的基准数据
        algorithm_data = {
            '贪心算法': {
                'Matmul_Case0': 163840, 'Matmul_Case1': 1179648,
                'FlashAttention_Case0': 64264, 'FlashAttention_Case1': 248976,
                'Conv_Case0': 150807, 'Conv_Case1': 570168
            },
            '改进贪心': {
                'Matmul_Case0': 131328, 'Matmul_Case1': 1048832,
                'FlashAttention_Case0': 55048, 'FlashAttention_Case1': 229520,
                'Conv_Case0': 111930, 'Conv_Case1': 549530
            },
            '遗传算法': {
                'Matmul_Case0': 64256, 'Matmul_Case1': 536320,
                'FlashAttention_Case0': 13442, 'FlashAttention_Case1': 94792,
                'Conv_Case0': 42190, 'Conv_Case1': 260028
            },
            '自适应调度': {}
        }
        
        # 添加二分法数据
        for case_name in self.test_cases:
            if case_name in self.binary_data:
                algorithm_data['自适应调度'][case_name] = self.binary_data[case_name]['max_v_stay']
        
        # 准备绘图数据
        cases = []
        case_mapping = {
            'Matmul_Case0': 'Matmul\nCase 0', 'Matmul_Case1': 'Matmul\nCase 1',
            'FlashAttention_Case0': 'FlashAttention\nCase 0', 'FlashAttention_Case1': 'FlashAttention\nCase 1',
            'Conv_Case0': 'Conv\nCase 0', 'Conv_Case1': 'Conv\nCase 1'
        }
        
        algorithm_results = {alg: [] for alg in algorithm_data.keys()}
        
        for case_name in self.test_cases:
            if case_name in algorithm_data['自适应调度']:
                cases.append(case_mapping.get(case_name, case_name))
                for alg_name in algorithm_data.keys():
                    algorithm_results[alg_name].append(algorithm_data[alg_name][case_name])
        
        # 创建第一个图：四种算法对比
        self._create_algorithm_comparison_figure(cases, algorithm_results)
        
        # 创建第二个图：提升比例对比
        self._create_improvement_percentage_figure(cases, algorithm_results)
        
        print(f"✓ 已生成算法对比图表")
        self._print_four_algorithm_statistics(algorithm_data['贪心算法'], 
                                            algorithm_data['改进贪心'],
                                            algorithm_data['遗传算法'], 
                                            algorithm_data['自适应调度'])
    
    def _create_algorithm_comparison_figure(self, cases, algorithm_results):
        """创建四种算法对比图 - 第一个独立图表（科研论文风格）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))   # 增大尺寸
        
        # 去掉大标题
        
        x = np.arange(len(cases))
        width = 0.18
        
        # 从浅蓝到深蓝的渐进配色，突出自适应调度优势
        colors = [
            '#B6D7FF',      # 贪心算法 - 浅蓝色（较差性能）
            '#7BB3F0',      # 改进贪心 - 中蓝色（中等性能）  
            '#4A90E2',      # 遗传算法 - 较深蓝色（良好性能）
            '#003366'       # 自适应调度 - 深蓝色（最佳性能，突出显示）
        ]
        
        # 绘制柱状图，突出自适应调度
        bars = []
        for i, (alg_name, values) in enumerate(algorithm_results.items()):
            offset = (i - 1.5) * width
            # 为自适应调度（最后一个算法）设置特殊效果
            if i == len(algorithm_results) - 1:  # 自适应调度
                bar = ax.bar(x + offset, values, width, label=alg_name, 
                            color=colors[i], alpha=1.0, edgecolor='#001133', linewidth=2.0)
            else:
                bar = ax.bar(x + offset, values, width, label=alg_name, 
                            color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
            bars.append(bar)
            
            # 添加数值标签 - 智能显示
            for j, (rect, value) in enumerate(zip(bar, values)):
                height = rect.get_height()
                if value > 1000000:
                    label = f'{value/1000000:.1f}M'
                elif value > 1000:
                    label = f'{value/1000:.0f}K'
                else:
                    label = f'{value:,}'
                
                # 交替显示标签位置避免重叠，不旋转
                y_offset = 3 + (i % 2) * 15
                # 为自适应调度使用深蓝色标签
                label_color = '#003366' if i == len(algorithm_results) - 1 else 'black'
                label_weight = 'bold' if i == len(algorithm_results) - 1 else 'bold'
                ax.annotate(label,
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, y_offset),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12,
                           fontweight=label_weight, color=label_color,
                           rotation=0)
        
        # 坐标轴设置（中文）- 字体放大
        ax.set_xlabel('测试用例', fontsize=18, fontweight='bold')
        ax.set_ylabel('内存峰值 (字节)', fontsize=18, fontweight='bold')
        # 去掉子标题
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=16)
        
        # 图例移至右上角 - 字体放大
        ax.legend(loc='upper right', fontsize=16, frameon=True, 
                 fancybox=True, shadow=False, framealpha=0.9)
        
        # 坐标轴美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        
        # 添加子图标签
        ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 格式化y轴
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 保存高质量图片
        filename = 'journal_algorithm_comparison.png'
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(filename, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"✓ 已生成算法对比图片: {filename}")
    
    def _create_improvement_percentage_figure(self, cases, algorithm_results):
        """创建提升比例对比图 - 第二个独立图表（科研论文风格）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))   # 增大尺寸
        
        # 去掉大标题
        
        # 计算提升比例（以贪心算法为基准）
        greedy_values = algorithm_results['贪心算法']
        improvement_data = {}
        
        for alg_name, values in algorithm_results.items():
            if alg_name != '贪心算法':
                improvements = []
                for i, (baseline, current) in enumerate(zip(greedy_values, values)):
                    if baseline > 0:
                        improvement = ((baseline - current) / baseline) * 100
                        improvements.append(improvement)
                    else:
                        improvements.append(0)
                improvement_data[alg_name] = improvements
        
        x = np.arange(len(cases))
        width = 0.25
        
        # 现代科研配色（排除贪心算法）
        colors = [
            current_colors['secondary'],   # 改进贪心 - 深紫红  
            current_colors['success'],     # 遗传算法 - 活力橙
            current_colors['primary']      # 自适应调度 - 深海蓝
        ]
        
        # 绘制提升比例柱状图
        bars = []
        for i, (alg_name, improvements) in enumerate(improvement_data.items()):
            offset = (i - 1) * width
            bar = ax.bar(x + offset, improvements, width, label=f'{alg_name}相对贪心算法', 
                        color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            bars.append(bar)
            
            # 添加百分比标签
            for rect, improvement in zip(bar, improvements):
                height = rect.get_height()
                ax.annotate(f'{improvement:.1f}%',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top', 
                           fontsize=12, fontweight='bold', color='black')
        
        # 添加0%参考线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # 坐标轴设置（中文）- 字体放大
        ax.set_xlabel('测试用例', fontsize=18, fontweight='bold')
        ax.set_ylabel('内存峰值减少比例 (%)', fontsize=18, fontweight='bold')
        # 去掉子标题
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=16)
        
        # 图例放在右上角 - 字体放大
        ax.legend(loc='upper right', fontsize=16, frameon=True, 
                 fancybox=True, shadow=False, framealpha=0.9)
        
        # 坐标轴美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        
        # 添加子图标签
        ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 保存高质量图片
        filename = 'journal_improvement_percentage.png'
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(filename, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"✓ 已生成提升比例图片: {filename}")
    
    def _create_journal_comparison_chart(self, ax, cases, algorithm_results):
        """创建期刊级质量的算法对比图"""
        x = np.arange(len(cases))
        width = 0.18
        
        # 期刊级配色方案
        colors = [
            current_colors['danger'],      # Greedy - 红色
            current_colors['secondary'],   # Improved Greedy - 橙色  
            current_colors['success'],     # Genetic - 绿色
            current_colors['primary']      # Adaptive Scheduler - 蓝色
        ]
        
        # 绘制柱状图
        bars = []
        for i, (alg_name, values) in enumerate(algorithm_results.items()):
            offset = (i - 1.5) * width
            bar = ax.bar(x + offset, values, width, label=alg_name, 
                        color=colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            bars.append(bar)
        
        # 期刊级数值标签 - 只在重要位置显示
        for i, bar_group in enumerate(bars):
            for j, bar in enumerate(bar_group):
                height = bar.get_height()
                if height > 0:
                    # 使用科学记数法简化标签
                    if height >= 1000000:
                        label = f'{height/1000000:.1f}M'
                    elif height >= 1000:
                        label = f'{height/1000:.0f}K'
                    else:
                        label = f'{height:.0f}'
                    
                    ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 2), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8, 
                               fontweight='bold', rotation=0)
        
        # 期刊级轴设置
        ax.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Memory Usage (bytes)', fontsize=12, fontweight='bold')
        ax.set_title('Comparative Analysis of Four Scheduling Algorithms', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=10)
        
        # 期刊级图例
        ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, 
                 shadow=False, framealpha=0.95, ncol=2)
        
        # 美化轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    def _create_improvement_chart(self, ax, cases, algorithm_results):
        """创建改进效果图"""
        greedy_values = algorithm_results['Greedy']
        improvements = {}
        
        for alg_name, values in algorithm_results.items():
            if alg_name != 'Greedy':
                improvements[alg_name] = [
                    ((g - v) / g * 100) if g > 0 else 0 
                    for g, v in zip(greedy_values, values)
                ]
        
        x = np.arange(len(cases))
        width = 0.25
        colors = [current_colors['secondary'], current_colors['success'], current_colors['primary']]
        
        for i, (alg_name, imp_values) in enumerate(improvements.items()):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, imp_values, width, label=alg_name,
                         color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # 添加数值标签
            for bar, imp in zip(bars, imp_values):
                height = bar.get_height()
                ax.annotate(f'{imp:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Test Cases', fontsize=11)
        ax.set_ylabel('Improvement (%)', fontsize=11)
        ax.set_title('Performance Improvement vs Greedy Algorithm', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=9, rotation=45)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_performance_summary(self, ax, algorithm_results):
        """创建性能摘要表"""
        ax.axis('off')
        
        # 计算总体统计
        totals = {}
        averages = {}
        for alg_name, values in algorithm_results.items():
            totals[alg_name] = sum(values)
            averages[alg_name] = sum(values) / len(values) if values else 0
        
        # 计算改进百分比
        greedy_total = totals['Greedy']
        improvements = {}
        for alg_name in totals:
            if alg_name != 'Greedy':
                improvements[alg_name] = ((greedy_total - totals[alg_name]) / greedy_total * 100) if greedy_total > 0 else 0
        
        # 创建表格数据
        table_data = []
        table_data.append(['Algorithm', 'Total Peak', 'Avg Peak', 'Improvement'])
        table_data.append(['-' * 10, '-' * 10, '-' * 10, '-' * 11])
        
        for alg_name in algorithm_results.keys():
            total = f"{totals[alg_name]/1000000:.1f}M" if totals[alg_name] >= 1000000 else f"{totals[alg_name]/1000:.0f}K"
            avg = f"{averages[alg_name]/1000000:.2f}M" if averages[alg_name] >= 1000000 else f"{averages[alg_name]/1000:.0f}K"
            imp = f"{improvements.get(alg_name, 0):.1f}%" if alg_name != 'Greedy' else 'Baseline'
            table_data.append([alg_name[:12], total, avg, imp])
        
        # 绘制表格
        y_start = 0.9
        for i, row in enumerate(table_data):
            y_pos = y_start - i * 0.12
            for j, cell in enumerate(row):
                x_pos = 0.02 + j * 0.22
                fontweight = 'bold' if i == 0 else 'normal'
                fontsize = 10 if i == 0 else 9
                ax.text(x_pos, y_pos, cell, transform=ax.transAxes, 
                       fontsize=fontsize, fontweight=fontweight, 
                       ha='left', va='center')
        
        ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    def _draw_peak_comparison_bar(self, ax):
        """绘制峰值对比柱状图"""
        cases = []
        heuristic_peaks = []
        binary_peaks = []
        
        for case_name in self.test_cases:
            if case_name in self.heuristic_data and case_name in self.binary_data:
                cases.append(case_name.replace('_Case', '\nCase'))
                heuristic_peaks.append(self.heuristic_data[case_name]['max_v_stay'])
                binary_peaks.append(self.binary_data[case_name]['max_v_stay'])
        
        x = np.arange(len(cases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, heuristic_peaks, width, label='启发式算法', color='#ff6b6b', alpha=0.8)
        bars2 = ax.bar(x + width/2, binary_peaks, width, label='二分法算法', color='#51cf66', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('测试用例')
        ax.set_ylabel('内存峰值')
        ax.set_title('内存峰值对比')
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_improvement_percentage(self, ax):
        """绘制改进百分比"""
        cases = []
        improvements = []
        
        for case_name in self.test_cases:
            if case_name in self.heuristic_data and case_name in self.binary_data:
                heuristic_peak = self.heuristic_data[case_name]['max_v_stay']
                binary_peak = self.binary_data[case_name]['max_v_stay']
                
                if heuristic_peak > 0:
                    improvement = ((heuristic_peak - binary_peak) / heuristic_peak) * 100
                    cases.append(case_name.replace('_Case', '\nCase'))
                    improvements.append(improvement)
        
        colors = ['#51cf66' if imp > 0 else '#ff6b6b' for imp in improvements]
        bars = ax.bar(cases, improvements, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{imp:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        ax.set_xlabel('测试用例')
        ax.set_ylabel('改进百分比 (%)')
        ax.set_title('二分法相对启发式的改进')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    def _draw_overall_comparison(self, ax):
        """绘制总体统计对比"""
        ax.axis('off')
        
        # 计算总体统计
        total_heuristic = sum(data['max_v_stay'] for data in self.heuristic_data.values())
        total_binary = sum(data['max_v_stay'] for data in self.binary_data.values())
        avg_heuristic = total_heuristic / len(self.test_cases)
        avg_binary = total_binary / len(self.test_cases)
        total_improvement = ((total_heuristic - total_binary) / total_heuristic) * 100
        
        stats_text = f"""
总体性能对比

总内存峰值:
• 启发式算法: {total_heuristic:,}
• 二分法算法: {total_binary:,}
• 改进: {total_improvement:.1f}%

平均内存峰值:
• 启发式算法: {avg_heuristic:,.0f}
• 二分法算法: {avg_binary:,.0f}
• 改进: {((avg_heuristic - avg_binary) / avg_heuristic) * 100:.1f}%

最佳改进案例:
{self._get_best_improvement_case()}

算法特点:
• 启发式: 基于复杂权重和预测
• 二分法: 预算约束 + 可行性检测
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='sans-serif',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    def _draw_efficiency_comparison(self, ax):
        """绘制算法效率对比"""
        ax.axis('off')
        
        efficiency_text = f"""
算法效率分析

时间复杂度:
• 启发式: O((N+E) log N) + 后处理
• 二分法: O(log S · (N+E) log N)

空间复杂度:
• 启发式: O(N+E)
• 二分法: O(N+E)

收敛性:
• 启发式: 局部最优，依赖启发式质量
• 二分法: 全局最优，保证收敛

适用场景:
• 启发式: 快速近似，适合实时调度
• 二分法: 精确优化，适合离线分析

稳定性:
• 启发式: 中等，依赖参数调优
• 二分法: 高，确定性算法
        """
        
        ax.text(0.05, 0.95, efficiency_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='sans-serif',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    def _get_best_improvement_case(self):
        """获取最佳改进案例"""
        best_case = ""
        best_improvement = 0
        
        for case_name in self.test_cases:
            if case_name in self.heuristic_data and case_name in self.binary_data:
                heuristic_peak = self.heuristic_data[case_name]['max_v_stay']
                binary_peak = self.binary_data[case_name]['max_v_stay']
                
                if heuristic_peak > 0:
                    improvement = ((heuristic_peak - binary_peak) / heuristic_peak) * 100
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_case = f"{case_name}: {improvement:.1f}%"
        
        return best_case if best_case else "无数据"
    
    def _create_statistics_overview_figure(self):
        """创建详细统计信息图，包含节点类型分布和执行单元使用情况"""
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('高级调度算法 - 详细状态信息', fontsize=20, fontweight='bold', y=0.98)
        
        outer_grid = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.4)

        for i, case_name in enumerate(self.test_cases):
            ax_main = fig.add_subplot(outer_grid[i])
            ax_main.set_title(case_name, fontsize=16, fontweight='bold', pad=20)
            ax_main.axis('off')

            if case_name not in self.results_data:
                ax_main.text(0.5, 0.5, '数据加载失败', ha='center', va='center', fontsize=12)
                continue

            step_info = self.results_data[case_name]['step_info']
            peak_memory = max(self.results_data[case_name]['memory_history']) if self.results_data[case_name]['memory_history'] else 0

            # 创建内部网格 1x2
            inner_grid = ax_main.get_subplotspec().subgridspec(1, 2, wspace=0.1)
            
            ax_pie = fig.add_subplot(inner_grid[0])
            ax_bar = fig.add_subplot(inner_grid[1])

            # 1. 绘制节点类型分布环形图
            self._draw_node_type_donut(ax_pie, step_info, peak_memory)
            
            # 2. 绘制执行单元使用情况条形图
            self._draw_pipe_usage_bar(ax_bar, step_info)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('scheduling_statistics_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 已生成 scheduling_statistics_overview.png")
        self._print_overall_statistics()

    def _draw_node_type_donut(self, ax, step_info, peak_memory):
        """为统计图绘制节点类型环形图和关键信息"""
        type_counts = {'ALLOC': 0, 'FREE': 0, '操作': 0, '其他': 0}
        for info in step_info:
            if info['op'] == 'ALLOC' and info['is_l1_ub']: type_counts['ALLOC'] += 1
            elif info['op'] == 'FREE' and info['is_l1_ub']: type_counts['FREE'] += 1
            elif not info['is_cache']: type_counts['操作'] += 1
            else: type_counts['其他'] += 1

        filtered_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        if filtered_counts:
            colors = ['#ff6b6b', '#51cf66', '#339af0', '#ffd93d']
            wedges, texts = ax.pie(filtered_counts.values(), 
                                   colors=colors[:len(filtered_counts)],
                                   startangle=90, radius=1.0,
                                   wedgeprops=dict(width=0.4, edgecolor='w'))
            
            legend_labels = [f'{k} ({v:,})' for k, v in filtered_counts.items()]
            ax.legend(wedges, legend_labels, title="节点类型", loc="upper left", bbox_to_anchor=(0, 0))
            ax.text(0, 0, f'峰值内存\n{peak_memory:,}', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_title('节点分布', fontsize=12, pad=15)

    def _draw_pipe_usage_bar(self, ax, step_info):
        """为统计图绘制执行单元使用条形图"""
        pipe_counts = defaultdict(int)
        for info in step_info:
            if not info['is_cache'] and info['pipe']:
                pipe_counts[info['pipe']] += 1
        
        if pipe_counts:
            sorted_pipes = sorted(pipe_counts.items(), key=lambda item: item[1], reverse=True)
            pipes = [item[0] for item in sorted_pipes]
            counts = [item[1] for item in sorted_pipes]
            
            bars = ax.barh(pipes, counts, color='#339af0', alpha=0.8, edgecolor='black', height=0.6)
            ax.invert_yaxis()
            ax.set_xlabel('使用次数')
            ax.tick_params(axis='x', rotation=0)
            
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='y', labelsize=8)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, f'{width:,}',
                        ha='left', va='center', fontsize=8)
            ax.set_xlim(right=max(counts) * 1.15)
        else:
            ax.text(0.5, 0.5, '无操作节点', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('执行单元使用', fontsize=12, pad=15)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    def _print_binary_statistics(self):
        """打印二分法统计信息"""
        print("\n=== 二分法算法性能统计 ===")
        
        total_binary = sum(data['max_v_stay'] for data in self.binary_data.values())
        print(f"二分法算法总峰值: {total_binary:,}")
        print()
        
        print(f"{'测试用例':<25} {'二分法峰值':<15} {'节点数':<10}")
        print("-" * 55)
        
        for case_name in self.test_cases:
            if case_name in self.binary_data:
                b_peak = self.binary_data[case_name]['max_v_stay']
                total_nodes = len(self.binary_data[case_name]['schedule'])
                print(f"{case_name:<25} {b_peak:<15,} {total_nodes:<10,}")
    
    def _print_four_algorithm_statistics(self, greedy_data, improved_greedy_data, genetic_data, advanced_scheduler_data):
        """打印四种算法对比统计信息"""
        print("\n=== 四种算法性能对比统计 ===")
        
        total_greedy = sum(greedy_data.values())
        total_improved_greedy = sum(improved_greedy_data.values())
        total_genetic = sum(genetic_data.values())
        total_advanced = sum(advanced_scheduler_data.values())
        
        total_improved_improvement = ((total_greedy - total_improved_greedy) / total_greedy) * 100
        total_genetic_improvement = ((total_greedy - total_genetic) / total_greedy) * 100
        total_advanced_improvement = ((total_greedy - total_advanced) / total_greedy) * 100
        
        print(f"贪心算法总峰值: {total_greedy:,}")
        print(f"改进贪心总峰值: {total_improved_greedy:,}")
        print(f"遗传算法总峰值: {total_genetic:,}")
        print(f"改进列表调度总峰值: {total_advanced:,}")
        print()
        print(f"改进贪心相对贪心改进: {total_improved_improvement:.1f}%")
        print(f"遗传算法相对贪心改进: {total_genetic_improvement:.1f}%")
        print(f"改进列表调度相对贪心改进: {total_advanced_improvement:.1f}%")
        print()
        
        print(f"{'测试用例':<25} {'贪心算法':<12} {'改进贪心':<12} {'遗传算法':<12} {'改进列表调度':<12} {'改进贪心%':<10} {'遗传算法%':<10} {'改进列表调度%':<12}")
        print("-" * 120)
        
        for case_name in self.test_cases:
            if case_name in greedy_data and case_name in advanced_scheduler_data:
                greedy_peak = greedy_data[case_name]
                improved_peak = improved_greedy_data[case_name]
                genetic_peak = genetic_data[case_name]
                advanced_peak = advanced_scheduler_data[case_name]
                
                improved_improvement = ((greedy_peak - improved_peak) / greedy_peak) * 100
                genetic_improvement = ((greedy_peak - genetic_peak) / greedy_peak) * 100
                advanced_improvement = ((greedy_peak - advanced_peak) / greedy_peak) * 100
                
                print(f"{case_name:<25} {greedy_peak:<12,} {improved_peak:<12,} {genetic_peak:<12,} {advanced_peak:<12,} {improved_improvement:<10.1f} {genetic_improvement:<10.1f} {advanced_improvement:<12.1f}")
    
    def _print_three_algorithm_statistics(self, greedy_data, improved_greedy_data, genetic_data):
        """打印三种算法对比统计信息（保留原函数以兼容性）"""
        print("\n=== 三种算法性能对比统计 ===")
        
        total_greedy = sum(greedy_data.values())
        total_improved_greedy = sum(improved_greedy_data.values())
        total_genetic = sum(genetic_data.values())
        
        total_improved_improvement = ((total_greedy - total_improved_greedy) / total_greedy) * 100
        total_genetic_improvement = ((total_greedy - total_genetic) / total_greedy) * 100
        genetic_vs_improved = ((total_improved_greedy - total_genetic) / total_improved_greedy) * 100
        
        print(f"贪心算法总峰值: {total_greedy:,}")
        print(f"改进贪心总峰值: {total_improved_greedy:,}")
        print(f"遗传算法总峰值: {total_genetic:,}")
        print()
        print(f"改进贪心相对贪心改进: {total_improved_improvement:.1f}%")
        print(f"遗传算法相对贪心改进: {total_genetic_improvement:.1f}%")
        print(f"遗传算法相对改进贪心改进: {genetic_vs_improved:.1f}%")
        print()
        
        print(f"{'测试用例':<25} {'贪心算法':<15} {'改进贪心':<15} {'遗传算法':<15} {'改进贪心改进%':<12} {'遗传算法改进%':<12}")
        print("-" * 100)
        
        for case_name in self.test_cases:
            if case_name in greedy_data:
                greedy_peak = greedy_data[case_name]
                improved_peak = improved_greedy_data[case_name]
                genetic_peak = genetic_data[case_name]
                
                improved_improvement = ((greedy_peak - improved_peak) / greedy_peak) * 100
                genetic_improvement = ((greedy_peak - genetic_peak) / greedy_peak) * 100
                
                print(f"{case_name:<25} {greedy_peak:<15,} {improved_peak:<15,} {genetic_peak:<15,} {improved_improvement:<12.1f} {genetic_improvement:<12.1f}")
    
    def _print_improvement_statistics(self, baseline_data):
        """打印改进统计信息（保留原函数以兼容性）"""
        print("\n=== 与基准数据对比统计 ===")
        
        total_baseline = sum(baseline_data.values())
        total_binary = sum(data['max_v_stay'] for data in self.binary_data.values())
        total_improvement = ((total_baseline - total_binary) / total_baseline) * 100
        
        print(f"基准数据总峰值: {total_baseline:,}")
        print(f"二分法算法总峰值: {total_binary:,}")
        print(f"总体改进: {total_improvement:.1f}%")
        print()
        
        print(f"{'测试用例':<25} {'基准峰值':<15} {'二分法峰值':<15} {'改进%':<10}")
        print("-" * 70)
        
        for case_name in self.test_cases:
            if case_name in self.binary_data and case_name in baseline_data:
                baseline_peak = baseline_data[case_name]
                binary_peak = self.binary_data[case_name]['max_v_stay']
                improvement = ((baseline_peak - binary_peak) / baseline_peak) * 100
                print(f"{case_name:<25} {baseline_peak:<15,} {binary_peak:<15,} {improvement:<10.1f}")
    
    def _print_overall_statistics(self):
        """打印总体统计信息"""
        print("\n=== 高级算法性能统计 ===")
        
        total_peak = sum(max(self.results_data[case]['memory_history']) for case in self.test_cases if case in self.results_data and self.results_data[case]['memory_history'])
        print(f"所有案例内存峰值总计: {total_peak:,}")
        print()
        
        print(f"{'测试用例':<25} {'高级算法峰值':<20} {'总节点数':<15}")
        print("-" * 65)
        
        for case_name in self.test_cases:
            if case_name in self.results_data:
                data = self.results_data[case_name]
                peak = max(data['memory_history']) if data['memory_history'] else 0
                total_nodes = len(data['schedule'])
                print(f"{case_name:<25} {peak:<20,} {total_nodes:<15,}")


def main():
    """
    主函数 - 调度算法可视化分析
    
    生成四张图表：
    1. algorithm_comparison_memory.png - 二分法内存使用量及节点分布（6个Case）
    2. algorithm_comparison_details.png - 详细统计信息（仅二分法，6个Case柱状图）
    3. algorithm_improvement_summary.png - 四种算法性能对比（贪心、改进贪心、遗传算法、改进列表调度）
    4. chinese_font_test.png - 中文字体测试图
    """
    print("=== 调度算法可视化分析 ===\n")
    print("📋 功能说明：")
    print("   • 二分法算法内存使用量及节点分布分析")
    print("   • 6个测试用例详细统计信息展示")
    print("   • 四种算法性能对比分析（贪心、改进贪心、遗传算法、改进列表调度）")
    print("   • 性能改进统计和算法效果评估")
    print()
    
    # 首先测试中文字体
    print("测试中文字体支持...")
    font_ok = test_chinese_font()
    
    if not font_ok:
        print("警告：中文字体可能无法正确显示")
    
    print()
    
    try:
        # 创建算法对比可视化器
        visualizer = AlgorithmVisualizer()
        
        # 生成对比可视化
        visualizer.create_visualization()
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

def demo_single_case_analysis():
    """演示单个测试用例的详细分析（保留原有功能）"""
    print("\n=== 单个测试用例详细分析演示 ===")
    
    task_name = "FlashAttention_Case0"
    graph_file = f"data/Json_version/{task_name}.json"
    
    print(f"正在分析任务: {task_name}")
    
    try:
        # 实时运行调度器
        print(f"  -> 正在为 {task_name} 运行高级调度算法...")
        scheduler = AdvancedScheduler(graph_file)
        schedule, max_v_stay = scheduler.schedule()
        print(f"     ...完成. 峰值 V_stay: {max_v_stay:,}")

        # 创建可视化器
        visualizer = EnhancedVisualizer(graph_file, schedule)
        total_steps = len(visualizer.schedule)
        
        print(f"总步数: {total_steps}")
        print(f"内存使用峰值: {max(visualizer.full_memory_history):,}")
        
        # 分析完整调度过程
        filename = visualizer.create_comprehensive_visualization(0, None)
        print(f"已生成单个用例分析文件: {filename}")
        
    except Exception as e:
        print(f"单个用例分析失败: {e}")

if __name__ == "__main__":
    # 运行主要的对比分析
    main()
    
    # 可选：演示单个测试用例的详细分析
    # demo_single_case_analysis()
