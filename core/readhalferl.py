import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    y_smooth = savgol_filter(test_scores, window_length=59, polyorder=3)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color=color, label=label)

test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed2", "Test_Score", 7, 14, False)
max1=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed2", "Test_Score", 3, 14, False)
x1,y1=np.array(frames),np.array(max1)
#draw(frames,max1,'red','seed=2')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed3", "Test_Score", 7, 14, False)
max1=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed3", "Test_Score", 3, 14, False)
x2,y2=np.array(frames),np.array(max1)
#draw(frames,max1,'green','seed=7')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed7", "Test_Score", 7, 14, False)
max1=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\half\distilseed7", "Test_Score", 3, 14, False)
x3,y3=np.array(frames),np.array(max1)
#draw(frames, test_scores, 'blue', 'seed=3')
f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')
f3 = interp1d(x3, y3, kind='linear', fill_value='extrapolate')
interpolation_range = np.linspace(min(min(x1), min(x2), min(x3)), max(max(x1), max(x2), max(x3)), 1000)
y1_interpolated = f1(interpolation_range)
y2_interpolated = f2(interpolation_range)
y3_interpolated = f3(interpolation_range)
average_curve = (y1_interpolated + y2_interpolated+y3_interpolated) / 3
draw(interpolation_range,average_curve,'brown','distil')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\half\erlseed2", "Test_Score", 7, 14, False)
max1=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\half\erlseed2", "Test_Score", 3, 14, False)
x1,y1=np.array(frames),np.array(max1)
#draw(frames,max1,'red','seed=2')
frames, test_scores = read("D:\ERL-RE2\ERL-Re2-main\logs\half\erlseed3", "Total T", 2, 10, False)
x2,y2=np.array(frames),np.array(test_scores)
#draw(frames,max1,'green','seed=7')
test_scores,ddpg=read("D:\ERL-RE2\ERL-Re2-main\logs\half\erlseed7", "Test_Score", 7, 14, False)
max1=np.maximum(test_scores, ddpg)
frames,_=read("D:\ERL-RE2\ERL-Re2-main\logs\half\erlseed7", "Test_Score", 3, 14, False)
x3,y3=np.array(frames),np.array(max1)
#draw(frames, test_scores, 'blue', 'seed=3')
f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')
f3 = interp1d(x3, y3, kind='linear', fill_value='extrapolate')
interpolation_range= np.linspace(min(min(x1), min(x2), min(x3)), max(max(x1), max(x2), max(x3)), 1000)
y1_interpolated = f1(interpolation_range)
y2_interpolated = f2(interpolation_range)
y3_interpolated = f3(interpolation_range)
average_curve = (y1_interpolated + y2_interpolated+y3_interpolated) / 3
draw(interpolation_range,average_curve,'blue','ERL-RE2')
plt.legend(loc='best')
# 设置图表标题和轴标签
plt.title("Ant-v2")
plt.ylabel('reward')
# 显示网格线
plt.grid(True)
# 显示图表
plt.show()