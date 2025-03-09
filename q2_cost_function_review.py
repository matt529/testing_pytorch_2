'''
此文是由 q2_cost_function_teaching.py 修改後的內容, 學習仍以原版為主
修正的目的是利用 numpy , 減少對 forloop 的依賴, 增加 running 速度

p.s. 裡面很多內容有依賴 chatgpt 進行修改至我看得懂，但仍不熟的地步
'''

import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import q1_new_pyplot as q1
import numpy as np  # 本次雙迴圈會用到
# import torch
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


# 對 w, b 找出最佳解
# 定義 w, b 的範圍
w_range = np.arange(-100, 101, 1)  # w = -100 ~ 100
b_range = np.arange(-100, 101, 1)  # b = -100 ~ 100

# 生成 (w, b) 組合
W, B = np.meshgrid(w_range, b_range)  # W, B 都是 201x201 矩陣
# 擴展 W, B 讓它們和 x 相容
W = np.expand_dims(W, axis=0)  # 變成 (1, 201, 201)
B = np.expand_dims(B, axis=0)  # 變成 (1, 201, 201)
# 擴展 x, y 讓它們和 W, B 相容
x = x.reshape(-1, 1, 1)         # 變成 (33, 1, 1)
y = y.reshape(-1, 1, 1)         # 變成 (33, 1, 1)
# 計算所有的 y 預測值
y_pred = x.reshape(-1, 1, 1) * W + B  # (33, 201, 201)
# 計算平方誤差
error_squared = (y_pred - y)**2  # (33, 201, 201)
# 計算平均 cost 值
cost_values = np.mean(error_squared, axis=0)  # cost_values 變成 (201, 201)
# find the index of minimum cost
# min_cost_values_index = np.argmin(cost_values)  # 找出最小的index of minimum cost
min_cost_index = np.unravel_index(np.argmin(cost_values), cost_values.shape)
( index_w , index_b ) = min_cost_index
best_w = W[0][index_w][index_b]
best_b = B[0][index_w][index_b]

new_plt = q1.plot_predict(best_w, best_b)
new_plt.show()

print(f"最小 Cost: {np.min(cost_values)}")
print(f"最佳 w: {best_w}")
print(f"最佳 b: {best_b}")
print(f"對應的最小 Cost: {cost_values[min_cost_index]}")

# 最小 Cost: 32.69484848484849
# 最佳 w: 9
# 最佳 b: 29
# 對應的最小 Cost: 32.69484848484849








# print(x)

# # 生成訓練數據
# x_tensor  = torch.tensor(x, dtype=torch.float32).view(-1, 1).cuda()
# # view() 僅為改變矩陣本身結構, 總元素是不變的
# y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).cuda()   
# # x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# # y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# # print(x[:5].shape) .shape是測數目


# # # 定義模型
# model = nn.Linear(1, 1).cuda()

# # # 定義損失函數和優化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)  # 優化器

# # 訓練模型
# for i in range(1000):
    
#     optimizer.zero_grad()   # 清空優化器
#     output = model(x_tensor)       
#     loss = criterion(output, y_tensor)
#     loss.backward()
#     optimizer.step()
#     if i==999:
#         print(f"次數i= {i+1}, Loss: {loss.item()}")

# # 測試結果， Loss落差頗大，
# # test_input = torch.tensor([[5.0]])
# # predicted = model(test_input)
# # print(predicted.item())  # 預測 5 對應的輸出值


# # 用 y = mx + b 計算預測值
# x_line = np.linspace(min(x), max(x), 100)  # 產生 x 範圍
# y_line = w * x_line + b  # y = mx + b
# print(x_line)
# print(y_line)
# # 繪製回歸線
# plt.plot(x_line, y_line, label="Predicted Line", color="red")

# # 圖片標籤
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.title("Linear Regression Fit")
# plt.legend()
# plt.show()



# # 產生所有可能的 (w, b) 組合
# W, B = np.meshgrid(w_range, b_range)  # 創建 2D 網格，n 階矩陣
# Y_pred = W[:, :, np.newaxis] * x.values + B[:, :, np.newaxis]  # 計算預測值

# # 計算成本函數 (平方誤差)
# cost_matrix = np.mean((y.values - Y_pred) ** 2, axis=2)  

# # 找到最小的 cost
# min_index = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
# best_w, best_b = W[min_index], B[min_index]

# print(f'最佳解：w={best_w}, b={best_b}, 最小成本={cost_matrix[min_index]}')



# plot_predict(20,-40)
# plot_predict(20,-100)
# font_path = "Traditional_Chinese_font.ttf"  # 確保檔案路徑正確
# custom_font = font_manager.FontProperties(fname=font_path)

# plt.title("標題", fontproperties=custom_font)
# plt.show()

# # y = w*x + b
# x = data["YearsExperience"]
# y = data["Salary"]

# plt.scatter(x, y)




# def plot_predict(w, b):
#     y_pred = w * x + b
#     plt.plot(x, y_pred, color="blue", label="predict")       # 直線
#     plt.scatter(x, y, color="red", label="real_data" )        # 點
#     plt.title("years-salary relationship")
#     plt.xlabel("Years of experience(Y)")
#     plt.ylabel("Salary of experience(K)")
#     plt.xlim(0, 12)
#     plt.ylim(-60,140)
#     plt.legend()
#     plt.show()

# w = 0
# b = 0
# plot_predict(w, b)


    










