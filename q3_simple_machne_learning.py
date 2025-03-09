import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 讀取資料
file_path = './csv/salary.csv'
data = pd.read_csv(file_path, encoding='utf-8')
x = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values.reshape(-1, 1)

# 轉換為 Tensor
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 切分訓練集與測試集
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=87)

# 轉移至 GPU（若可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# 定義模型
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 單輸入、單輸出

    def forward(self, x):
        return self.linear(x)

# 初始化模型、損失函數、優化器
model = SimpleLinearRegression().to(device)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 100000
prev_loss = float('inf')

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清空梯度
    y_pred = model(x_train)  # 前向傳播
    loss = loss_function(y_pred, y_train)  # 計算損失
    loss.backward()  # 反向傳播
    optimizer.step()  # 更新權重

    # 監控收斂情況
    if abs(prev_loss - loss.item()) < 1e-6:
        break
    prev_loss = loss.item()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss={loss.item():.3f}')

# 測試預測
y_pred_test = model(x_test).detach()

# 結果輸出
print("\n測試結果對比：")
print("Actual  | Predicted  | Error")
print("-----------------------------")
for actual, pred in zip(y_test[:10], y_pred_test[:10]):
    print(f"{actual.item():6.1f}  | {pred.item():6.1f}  |\
           {abs(actual.item() - pred.item()):6.1f}")

print("-----------------------------")
error = torch.abs(y_test - y_pred_test)
print(f"Mean Absolute Error: {torch.mean(error).item():6.1f}")
