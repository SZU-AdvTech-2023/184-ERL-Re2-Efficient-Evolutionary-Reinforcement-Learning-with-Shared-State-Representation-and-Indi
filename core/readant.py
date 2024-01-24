import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
def read(path,keys,a,b,encoding):#a,b是第一个读取数据和第二个要读取数据所在的位置
    if(encoding):
        with open(path, 'r',encoding='utf-16') as file:
            lines = file.readlines()
    else:
        with open(path, 'r') as file:
            lines = file.readlines()
    # 初始化列表来保存Frames和Test_Score的值
    frames = []
    test_scores = []
    # 提取Frames和Test_Score的值
    for line in lines:
        if keys in line:
            line = line.strip().split()
            frames.append(float(line[a]))
            test_scores.append(float(line[b]))
    return frames,test_scores
def draw(frames,test_scores,color,label):
    x_smooth = frames
    y_smooth = savgol_filter(test_scores, window_length=109, polyorder=3)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color=color, label=label)
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed2", "Test_Score", 7, 14, False)
max=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed2", "Test_Score", 3, 14, False)
draw(frames,max,'red','seed=2')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed3", "Test_Score", 7, 14, True)
max=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed3", "Test_Score", 3, 14, True)
draw(frames,max,'blue','seed=3')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed7", "Test_Score", 7, 14, True)
max=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\\antseed7", "Test_Score", 3, 14, True)
draw(frames,max,'green','seed=7')

plt.legend(loc='best')
# 设置图表标题和轴标签
plt.title("Ant-v2")
plt.ylabel('reward')
# 显示网格线
plt.grid(True)
# 显示图表
plt.show()

