# 有效率地找出 w & b
# gradient descent 梯度下降
# 開始進入正題

# (真實數據 - 預測值)^2
# = (y - y_pred) ** 2
# = (y - w * x + b) ** 2
# 找 cost function 微分 | dx, dy
# dc
# 斜率 = -2x( y - (w * x + b)) | dx
# 斜率 = 2x( y - w * x + b ) | dx
# 斜率 = 2 ( y - w * x + b ) | dy
# 斜率 = w - 斜率 * 學習率 (learning rate)
# 學習率 (learning rate) = 步伐大小 step_size

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 csv, local端下載
data = pd.read_csv("./csv/salary.csv", encoding="utf-8")

# 讀取 *.csv, 兩行，有keys: YearsExperience, Salary, 作為 x,y 
# y_list = y.tolist()   # 轉成 Python list
# y_dict = y.to_dict()  # 轉成 Python dict
x = data["YearsExperience"]
y = data["Salary"]


# 計算梯度
def compute_gradient(x, y, w, b):
    w_gradient = x * ( w * x + b - y)   # 因為 step 是逐步下降的, 有2倍只是加大步伐, 所以去除
    w_gradient = w_gradient.mean()
    b_gradient = ( w * x + b - y)   # 因為 step 是逐步下降的, 有2倍只是加大步伐, 所以去除
    b_gradient = b_gradient.mean()
    # print(f"w_gradient:{w_gradient}, b_gradient: {b_gradient}")
    return w_gradient, b_gradient


# 找出成本
def compute_cost(x, y, w, b):

    y_pred = w * x + b
    cost = ((y - y_pred) ** 2).mean()
    # log = f'平方合={cost * len(x)}, x:{len(x)}個, 平均={cost}'
    # print(log)
    return cost


w, b, learning_rate = 0, 0, 0.001

def gradient_descent( 
        x, y, w, b, compute_gradient, compute_cost,
        learning_rate=1.0e-3, run_iter=1000, p_iter=None ):
    
    c_hist, w_hist, b_hist = [], [], []
    if not p_iter:
        p_iter = run_iter   # 預設只印一次

    for i in range(run_iter):
        w_gradient, b_gradient = compute_gradient(x, y, w, b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = compute_cost(x, y, w, b)

        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        if i % p_iter == (run_iter-1):
            print(f"cost={cost: 2f}, w={w: 2f}, b={b: 2f}")

    return w, b, w_hist, b_hist, c_hist


w, b = 0, 0
learning_rate = 1.0e-3
run_iter = 1000
w, b, w_hist, b_hist, c_hist = \
    gradient_descent(x, y, w, b, compute_gradient, compute_cost, learning_rate, run_iter)

# print(f"w_gradient: {w_gradient}")

# print(f"x: {x.tolist()}")  # 列出所有 x 值
# print(f"y: {y.tolist()}")  # 列出所有 y 值
# print(f"y.mean(): {y.mean()}")  # 平均薪資
# print(f"w_gradient: {w_gradient}")  # 梯度值



# # w - w_gradient * learning_rate
# w , b, learning_rate = 0, 0, 0.001  # 推薦 0.001
# w_gradient, b_gradient = compute_gradient(x, y, w, b)
# w -= w_gradient * learning_rate
# b -= b_gradient * learning_rate



# print(f"w={w}")
# print(f"b={b}")
# print(f"b_gradient: {b_gradient}")














