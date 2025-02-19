import plotly.graph_objects as go
import numpy as np
# 数据
params = [6.18, 1.88, 5.86, 17.06, 11.72, 1.60, 1.62, 119.13]  # 参数量
compute = [433.25, 2334.68, 2256.18, 256.11, 1029.60, 271.04, 9482.25, 990.62]  # 计算量
inference_time = [251.42, 346.17, 549.91, 110.80, 964.19, 254.88, 873.50, 318.37]  # 推理时间（假设数据）
performance = [40.3536, 39.5792, 40.1087, 39.6509, 40.1362, 39.7196, 39.5581, 40.0952]  # PSNR
methods = ['Ours','DeamNet','MIRNet_v2', 'NAFNet','Restormer_s','RetinexFormer','DRANet','CGNet']
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

fig = go.Figure(data=[go.Scatter(
    x=inference_time,
    y=performance,
    mode='markers+text',  # 添加 text 模式
    marker=dict(
        size=12,
        color=colors,  # 对数变换
        colorscale='Plasma',
        opacity=0.8,
        showscale=False,
        colorbar=dict(
            title='计算量',
            x=1.0,
            y=0.5,
            len=0.5
        )
    ),
    text=methods,  # 为每个点添加标签
    textposition='top center',  # 标签位置：点的正上方
    textfont=dict(size=12, color='black')  # 将字体大小设置为 8
)])

fig.update_layout(
    title='Inference time(ms) vs PSNR(dB)',
    font=dict(family='Times New Roman', size=12, color='black'),  # 设置全局字体为新罗马
    xaxis_title='Inference time(ms)',
    yaxis_title='PSNR(dB)',
    plot_bgcolor='rgba(0,0,0,0)',  # 设置绘图区域背景色为透明
    margin=dict(l=0, r=0, b=0, t=40),
    xaxis=dict(
        title_standoff=10,  # 减少 X 轴标题与坐标轴之间的距离
        showline=True,  # 显示坐标轴线
        linecolor='black',  # 坐标轴线颜色
        mirror=False,  # 不显示镜像
        showgrid=False,  # 不显示网格线
        ticks='outside',  # 刻度线显示在外部
        tickcolor='black',  # 刻度线颜色
        ticklen=5,  # 刻度线长度
        zeroline=False  # 不显示零线
    ),
    yaxis=dict(
        title_standoff=5,  # 减少 X 轴标题与坐标轴之间的距离
        showline=True,  # 显示坐标轴线
        linecolor='black',  # 坐标轴线颜色
        mirror=False,  # 不显示镜像
        showgrid=False,  # 不显示网格线
        ticks='outside',  # 刻度线显示在外部
        tickcolor='black',  # 刻度线颜色
        ticklen=5,  # 刻度线长度
        zeroline=False  # 不显示零线
    )
)
#fig.show()
import time
fig.write_image("C:/Users/Carlos/Desktop/output.pdf",format="pdf")
time.sleep(2)
fig.write_image("C:/Users/Carlos/Desktop/output.pdf",format="pdf")
#fig.write_image("C:/Users/Carlos/Desktop/output.pdf", format="pdf", width=800, height=600)