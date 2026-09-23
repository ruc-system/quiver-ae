import argparse
import sqlite3
import pandas as pd
import sys
import os

# --- 关键修改 1: 设置无头模式 (Headless Mode) ---
import matplotlib
matplotlib.use('Agg') # 必须在导入 pyplot 之前设置，专用于服务器环境

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
# 尝试设置字体，防止中文乱码（服务器上如果没有中文字体，可能会显示方框，可以改回英文）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# CONFIG: 数据库路径
db_path = 'my_profile_report.sqlite'
# 输出图片的文件名
output_image = 'profile_report.png'
# 跳过前 N 个 Query（warmup），0 表示不跳过
skip_warmup_queries = 0

# --- NVTX 定义 (保持不变) ---
main_stages = [
    'BuildDistanceTable',
    'GraphSearch', 'GatherCandidates', 'DeduplicateCandidates',
    'UploadCandidates', 'PQDistanceAccum',
    'FetchApproxResults', 'CPUSelectTopR', 'HeuristicRerank'
]
main_stage_labels = [
    'PQ Prep (GPU)',
    'Graph (CPU)', 'Gather (CPU)', 'Dedup (CPU)',
    'Upload (H2D)', 'PQ Scan (GPU)',
    'Fetch (GPU/Wait)', 'Sort (CPU)', 'Rerank (Mixed)'
]

sub_stages = {
    'Wait_GPU': ['WaitCompStream', 'WaitD2HCopy'],
    'Rerank_IO': ['Rerank.IOBatchFetch', 'Rerank.IOBatchFetchTail'],
    'Rerank_L2': ['Rerank.L2Batch', 'Rerank.L2BatchTail']
}
all_targets = main_stages + [item for sublist in sub_stages.values() for item in sublist]
parent_label = 'ProcessQuery'

def get_data_from_db(db_file):
    if not os.path.exists(db_file):
        print(f"错误: 找不到文件 {db_file}")
        return None

    try:
        con = sqlite3.connect(db_file)
    except Exception as e:
        print(f"无法打开数据库: {e}")
        return None

    query_labels = [parent_label] + all_targets
    formatted_labels = ','.join([f"'{s}'" for s in query_labels])
    
    sql = f"""
    SELECT n.start, n.end, (n.end - n.start) as duration, n.text as name, n.globalTid as thread_id
    FROM NVTX_EVENTS as n WHERE n.text IN ({formatted_labels}) ORDER BY n.start
    """
    try:
        df = pd.read_sql_query(sql, con)
        con.close()
        return df
    except Exception as e:
        print(f"查询执行失败: {e}")
        con.close()
        return None

def process_data(df):
    if df is None or df.empty: return None
    parents = df[df['name'] == parent_label].copy()
    children_pool = df[df['name'].isin(all_targets)].copy()
    if parents.empty: return None

    # 跳过 warmup：按 start 排序后取后面的正式查询
    skip = max(0, skip_warmup_queries)
    if skip > 0:
        parents = parents.sort_values('start').iloc[skip:].reset_index(drop=True)
        if parents.empty:
            print(f"跳过 {skip} 个 warmup 后无剩余 Query。")
            return None
        print(f"已跳过前 {skip} 个 warmup Query，分析剩余 {len(parents)} 个正式 Query...")
    else:
        print(f"正在分析 {len(parents)} 个 Query 请求...")

    results = []

    for idx, parent in parents.iterrows():
        p_start, p_end = parent['start'], parent['end']
        p_dur, tid = parent['duration'], parent['thread_id']
        if p_dur <= 0: continue

        current_children = children_pool[
            (children_pool['start'] >= p_start) & 
            (children_pool['end'] <= p_end) & 
            (children_pool['thread_id'] == tid)
        ]
        
        row_raw = {'Total_ns': p_dur}
        accounted_sum = 0
        for stage in main_stages:
            s_dur = current_children[current_children['name'] == stage]['duration'].sum()
            row_raw[f'{stage}_ns'] = s_dur
            accounted_sum += s_dur
        row_raw['Unaccounted_ns'] = max(0, p_dur - accounted_sum)

        wait_dur = current_children[current_children['name'].isin(sub_stages['Wait_GPU'])]['duration'].sum()
        io_dur = current_children[current_children['name'].isin(sub_stages['Rerank_IO'])]['duration'].sum()
        l2_dur = current_children[current_children['name'].isin(sub_stages['Rerank_L2'])]['duration'].sum()
        rerank_total = current_children[current_children['name'] == 'HeuristicRerank']['duration'].sum()
        fetch_total = current_children[current_children['name'] == 'FetchApproxResults']['duration'].sum()
        
        row_raw['Wait_GPU_ns'] = wait_dur
        row_raw['Rerank_IO_ns'] = io_dur
        row_raw['Rerank_L2_ns'] = l2_dur
        row_raw['Rerank_Total_ns'] = rerank_total
        row_raw['Fetch_Total_ns'] = fetch_total
        
        results.append(row_raw)

    return pd.DataFrame(results)

def plot_analysis(df, filename):
    if df is None or df.empty:
        print("没有足够的数据用于绘图。")
        return

    avg_data = df.mean()
    total_ns = avg_data['Total_ns']
    if total_ns <= 0: return

    # 创建画布，稍微调大一点以便保存
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Query Performance Profile Analysis (Avg of {len(df)} Queries)', fontsize=20, y=0.95)

    colors_main = sns.color_palette("Set2", len(main_stages) + 1)
    colors_bottleneck = sns.color_palette("husl", 5)
    colors_rerank = sns.color_palette("pastel", 3)

    # --- 图 1: 主阶段 ---
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    sizes = [avg_data[f'{s}_ns'] for s in main_stages] + [avg_data['Unaccounted_ns']]
    labels = main_stage_labels + ['Unaccounted (Gap)']
    
    threshold = total_ns * 0.01
    f_sizes, f_labels, f_colors = [], [], []
    for i, size in enumerate(sizes):
        if size > threshold:
            f_sizes.append(size)
            f_labels.append(labels[i])
            f_colors.append(colors_main[i])
            
    ax1.pie(f_sizes, labels=f_labels, autopct='%1.1f%%', 
            startangle=90, colors=f_colors, wedgeprops={'edgecolor': 'white'})
    ax1.set_title("1. Logic Stages Breakdown (Wall Time)", fontsize=14)

    # --- 图 2: 瓶颈分析 ---
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    cpu_compute_ns = (avg_data['GraphSearch_ns'] + avg_data['GatherCandidates_ns'] + 
                      avg_data['DeduplicateCandidates_ns'] + avg_data['CPUSelectTopR_ns'] + 
                      avg_data['Rerank_L2_ns'])
    gpu_wait_ns = avg_data['Wait_GPU_ns']
    storage_io_ns = avg_data['Rerank_IO_ns']
    gpu_transfer_active_ns = max(0, avg_data['Fetch_Total_ns'] - gpu_wait_ns)
    
    bottleneck_values = [cpu_compute_ns, gpu_wait_ns, storage_io_ns, gpu_transfer_active_ns]
    bottleneck_labels = ['CPU Pure Compute', 'CPU Waiting for GPU', 'Storage I/O (Rerank)', 'GPU/PCIe Active (Est.)']
    bottleneck_pcts = [v / total_ns * 100 for v in bottleneck_values]
    
    y_pos = np.arange(len(bottleneck_labels))
    bars = ax2.barh(y_pos, bottleneck_pcts, color=colors_bottleneck)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bottleneck_labels, fontsize=12)
    ax2.set_xlabel('Percentage of Total Query Time (%)', fontsize=12)
    ax2.set_title("2. Resource Bottleneck Analysis (Where time goes?)", fontsize=14)
    ax2.set_xlim(0, max(bottleneck_pcts) * 1.2)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', ha='left', va='center', fontweight='bold')

    # --- 图 3: 重排透视 ---
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    rerank_total = avg_data['Rerank_Total_ns']
    if rerank_total > 0:
        io_ns = avg_data['Rerank_IO_ns']
        l2_ns = avg_data['Rerank_L2_ns']
        other_ns = max(0, rerank_total - io_ns - l2_ns)
        rr_sizes = [io_ns, l2_ns, other_ns]
        rr_labels = ['I/O (Read Vectors)', 'Compute (L2 Dist)', 'Other (Heap/Ctrl)']
        
        rr_threshold = rerank_total * 0.02
        f_rr_sizes, f_rr_labels, f_rr_colors = [], [], []
        for i, size in enumerate(rr_sizes):
            if size > rr_threshold:
                f_rr_sizes.append(size)
                f_rr_labels.append(rr_labels[i])
                f_rr_colors.append(colors_rerank[i])

        ax3.pie(f_rr_sizes, labels=f_rr_labels, autopct='%1.1f%%',
                startangle=140, colors=f_rr_colors, wedgeprops={'edgecolor': 'white'})
        title = f"3. Rerank Internals\n(Rerank accounts for {rerank_total/total_ns*100:.1f}% of Total)"
        ax3.set_title(title, fontsize=14)
    else:
        ax3.text(0.5, 0.5, "No Rerank Data", ha='center', va='center')
        ax3.set_axis_off()

    # --- 图 4: Wait 效率 ---
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    fetch_total = avg_data['Fetch_Total_ns']
    if fetch_total > 0:
        wait_ns = avg_data['Wait_GPU_ns']
        active_ns = max(0, fetch_total - wait_ns)
        wait_pct = wait_ns / fetch_total * 100
        active_pct = active_ns / fetch_total * 100
        
        ax4.barh([0], [active_pct], color=colors_bottleneck[3], label='GPU/PCIe Active')
        ax4.barh([0], [wait_pct], left=[active_pct], color=colors_bottleneck[1], label='CPU Waiting')
        
        ax4.set_yticks([])
        ax4.set_xlabel('Percentage (%)')
        ax4.set_title(f"4. Fetch Stage Efficiency: How much time was CPU waiting? ({wait_pct:.1f}%)", fontsize=14)
        ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
        ax4.set_xlim(0, 100)
        
        interp = "Good! GPU is busy." if wait_pct < 30 else "Warning! CPU is blocked waiting for GPU."
        ax4.text(50, 0.4, interp, ha='center', color='red' if wait_pct >=30 else 'green', fontweight='bold')
    else:
         ax4.set_axis_off()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # --- 关键修改 2: 保存而不是弹窗 ---
    print(f"正在保存图片到 {filename} ...")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print("完成！请使用 scp 或 SFTP 下载查看。")
    # plt.show() # 千万别在 SSH 下调用这个

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 NVTX profile 报告")
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=0,
        metavar="N",
        help="跳过前 N 个 Query（warmup），只分析后续正式查询。默认 0 表示不跳过",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=db_path,
        help=f"数据库路径，默认 {db_path}",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=output_image,
        help=f"输出图片路径，默认 {output_image}",
    )
    args = parser.parse_args()

    skip_warmup_queries = max(0, args.skip_warmup)
    db_path = args.db
    output_image = args.output

    raw_df = get_data_from_db(db_path)
    processed_df = process_data(raw_df)
    plot_analysis(processed_df, output_image)