import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, linregress

folder_path = '/data/zlt/RemoteSensePET/outputs/rtc_s/WuhanMetro/t_attnsplit_boxquery_mixedloss_contxt128_refinedata/vis/gt_nac'

left=0
right=35

all_gt = []
all_pred = []

all_gt = []
all_pred = []

results=[]

pattern = re.compile(r'gt(\d+)_pred(\d+)')

for filename in os.listdir(folder_path):
    match = pattern.search(filename)
    if match:
        pred_value = float(match.group(1))
        gt_value = float(match.group(2))
        
        if left < gt_value < right:
            all_gt.append(gt_value)
            all_pred.append(pred_value)
        filename = filename.split('_gt')[0]
        results.append({
                'Filename': filename,
                'GT Value': gt_value,
                'Pred Value': pred_value
            })

all_gt = np.array(all_gt).astype(float)
all_pred = np.array(all_pred).astype(float)

# df_results = pd.DataFrame(results)
# excel_output_path = '/data/zlt/RemoteSensePET/outputs/SOY/s_noencoder_dist/r2/file_gt_pred_data.xlsx'
# df_results.to_excel(excel_output_path, index=False)

# 确保我们有足够的数据点来计算统计量
if len(all_gt) > 0 and len(all_pred) > 0:
    # 计算R²值、MAE、Pearson相关系数及其P值 和 Bias
    r2 = r2_score(all_gt, all_pred)
    mae = mean_absolute_error(all_gt, all_pred)
    corr_coef, p_value = pearsonr(all_gt, all_pred)
    bias = np.mean(all_pred - all_gt)

    slope, intercept, r_value, p_value_reg, std_err = linregress(all_gt, all_pred)
    # 创建图形并绘制散点图
    fig = plt.figure(figsize=(8, 6))
    
    # set(ax, 'XTick', [0,1], 'YTick', [1,1])
    
    f_size = 18
    # 设置较大的字体大小
    
    plt.rcParams.update({'font.size': f_size})

    plt.scatter(all_gt, all_pred, color='#FA9857',alpha=0.7, s=100, edgecolors='none')

    # 绘制完美预测线
    x_min, x_max = plt.xlim()
    x_vals = np.array([left, right+10])
    lims = [left, right+10]        
    # lims = [0, 155]
    plt.plot(lims, lims, 'k--', alpha=0.4, label='y = x')
    
    
    # x_vals = np.array([0, 155])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='#9E0142', label=f'y = {slope:.2f}x + {intercept:.2f}')

    plt.title(f'$R^2$ for GT ranging from {left} to {right}', fontsize=f_size)
    # plt.title(f'$R^2$ for all', fontsize=f_size)
    plt.xlabel('Ground Truth', fontsize=f_size-4)
    plt.ylabel('Predition', fontsize=f_size-4)
    plt.grid(False)

    stats_text = f'$R^2$ = {r2:.3f}\n' 
    plt.text(0.05, 0.75, stats_text, transform=plt.gca().transAxes,
            bbox=dict(facecolor='none', edgecolor='none')) 
    stats_text = f'MAE={mae:.2f}'
    plt.text(0.05, 0.89, stats_text, transform=plt.gca().transAxes,
            bbox=dict(facecolor='none', edgecolor='none')) 

    # 显示图形
    plt.savefig(f'point_{left}_t_{right}.pdf')