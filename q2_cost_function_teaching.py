'''
此文是由 youtube 免費影片教學內容，根據解說，邊聽邊寫而成(含部分調整)
# machine learning : 尋找最佳路徑準備
# 雖然是最沒效率的方式，卻是一開始最容易讀懂的方法
'''

import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import q1_new_pyplot as q1
import numpy as np  # 本次雙迴圈會用到
import torch
import torch.nn as nn
import torch.optim as optim

# 讀取 csv, local端下載
data = pd.read_csv("./csv/salary.csv", encoding="utf-8")

# 讀取 *.csv, 兩行，有keys: YearsExperience, Salary, 作為 x,y 
# y_list = y.tolist()   # 轉成 Python list
# y_dict = y.to_dict()  # 轉成 Python dict
x = data["YearsExperience"].to_numpy()
y = data["Salary"].to_numpy()


## cost function：
## cost = (真實數據 - 預測值)^2 的 平方合
w, b = 0, 0
y_pred = w * x + b

# # 找出成本
def compute_cost(x, y, w, b):

    # q1.plot_predict(w, b)
    y_pred = w * x + b
    cost = ((y - y_pred) ** 2).sum()  / len(x)
    # log = f'平方合={cost * len(x)}, x:{len(x)}個, 平均={cost}'
    # print(log)
    return cost


# 設定 ws, bs 範圍都在 -100 ~ 100 之間
ws = np.arange(-100, 101, 1)
bs = np.arange(-100, 101, 1)
costs = np.zeros((201, 201))

i = 0
for w in ws:
    j = 0
    for b in bs:
        costs[i][j] = compute_cost(x, y, w, b)
        j += 1
    i += 1


# 建立 3D構圖
ax = plt.axes(projection='3d')
ax.view_init(45,-120)
ax.xaxis.set_pane_color((1,1,1))
ax.yaxis.set_pane_color((1,1,1))
ax.zaxis.set_pane_color((1,1,1))

# 產生 二維 grid
b_grid, w_grid = np.meshgrid(bs, ws)
# 研究 https://wangyeming.github.io/2018/11/12/numpy-meshgrid/

# 中文語言
mlp.rc('font', family='Microsoft JhengHei, arial',)

ax.plot_surface(w_grid, b_grid, costs, cmap='coolwarm', alpha=0.8)
ax.plot_wireframe(w_grid, b_grid, costs, color='black', alpha=0.2)
ax.set_title('w b 對應的 Cost Function')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost')

min_cost = np.min(costs)
# 找出最小成本的 index
w_index, b_index = np.where(costs == min_cost)  # 找出最小成本的 index
# 找出最小 w, b
w = ws[w_index]
b = bs[b_index]
print(w,b, min_cost)

plt.show()

# y = mx + b
# y = 9x + 29








