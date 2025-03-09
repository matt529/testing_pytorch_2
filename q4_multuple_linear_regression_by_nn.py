'''
修改自 
q4_multuple_linear_regression.py and 
q4_multiple_linear_regression_with_nn_by_chatgpt
考慮使用 PyTorch 的 nn.Module 和 optim 模塊來建立並優化模型
'''
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 讀取資料並預處理
data = pd.read_csv("./csv/salary2.csv", encoding="utf-8")
education_map = {"高中以下":0, "大學":1, "碩士以上":2}
data["EducationLevel"] = data["EducationLevel"].map(education_map)

# One-Hot Encoding
encoder = OneHotEncoder()    # 自動刪除第一個 dummy variable
encoder.fit(data[["City"]])
city_encoded = encoder.transform(data[["City"]]).toarray()
data[["CityA", "CityB", "CityC"]] = city_encoded
data.drop(columns=["City", "CityC"], inplace=True) 

# 分割特徵 x 和標籤 y
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]].to_numpy()
y = data["Salary"].to_numpy()

# 分割訓練和測試集
'''
# 將 x, y 整理成 4 份 data 用來訓練和驗證, 部分數據固定不變
# 分割成 4 種，訓練用特徵、訓練用標籤、測試用特徵、測試用標籤
# random_state = 87 固定一種隨機方式
# code 長度超過 80, 修改成 2行方便閱讀
'''
vectors_list = train_test_split(x, y, test_size=0.2, random_state=87)
x_train, x_test, y_train, y_test = vectors_list

# 轉為 torch tensor 並搬移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# 定義線性回歸模型，(註：複製貼上的...)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 輸入大小，輸出1個預測值

    def forward(self, x):
        return self.linear(x)
    

# 初始化模型
inpit_size = x_train.shape[1]   # *.shape 會產生[28,4]==[訓練數, 特徵]
model = LinearRegressionModel(inpit_size).to(device)
# model2 = LinearRegressionModel(inpit_size).cuda()

# 定義損失函數和優化器
criterion = nn.MSELoss()    # 損失函數
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 優化器


# 訓練模型
epochs = 10000
for epoch in range(epochs):
    # 前向傳播
    y_pred = model(x_train)
    
    # 計算損失
    loss = criterion(y_pred, y_train)
    
    # 反向傳播和優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每 1000步 print
    if (epoch + 1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')


# 在測試集上進行預測 ( PyTorch 在該區域內不計算梯度，這樣可以節省計算資源)
with torch.no_grad():
    y_pred_test = model(x_test)

# 顯示結果
print("Actual  | Predicted")
print("-------------------")
for actual, pred in zip(y_test, y_pred_test):
    print(f"{actual.item():6.1f}  | {pred.item():6.1f}  | {abs(actual - pred).item():6.1f}")

# 計算誤差
error = torch.abs(y_test - y_pred_test)
print(f"Mean Absolute Error: {torch.mean(error).item():.1f}")


