'''
以下是 chatgpt 所寫，
修改自 q4_multuple_linear_regression.py
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
education_map = {"高中以下": 0, "大學": 1, "碩士以上": 2}
data["EducationLevel"] = data["EducationLevel"].map(education_map)

# One-Hot Encoding
encoder = OneHotEncoder()
encoder.fit(data[["City"]])
city_encoded = encoder.transform(data[["City"]]).toarray()
data[["CityA", "CityB", "CityC"]] = city_encoded
data.drop(columns=["City", "CityC"], inplace=True)

# 分割特徵 x 和標籤 y
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]].to_numpy()
y = data["Salary"].to_numpy()

# 分割訓練和測試集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

# 轉換為 PyTorch 張量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)

# 定義線性回歸模型，(註：複製貼上的...)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 輸入大小，輸出1個預測值

    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_size = x_train.shape[1]
model = LinearRegressionModel(input_size).to(device)

# 定義損失函數和優化器
criterion = nn.MSELoss()  # 均方誤差損失
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

    # 每1000步打印一次
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 在測試集上進行預測
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
