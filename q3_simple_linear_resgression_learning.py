# 從 多元線性回歸 自我練習寫 simple linear regression
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # 初始化 data
    data = pd.read_csv(file_path, encoding='utf-8')
    x = data['YearsExperience']
    x = torch.tensor(x.values, dtype=torch.float32)
    y = data['Salary']
    y = torch.tensor(y.values, dtype=torch.float32)

    # train_test_split 本身會代出 4 個值 (train_x, train_y, test_x, test_y)
    return train_test_split(x, y, test_size=0.2, random_state=87)
    

# 計算梯度
def compute_gradient(x, y, w, b):
    '''
    y_pred  = w * x + b,
    error   = (y - y_pred)**2
            = (y - (w*x + n))**2
            = (y - w*x - b)**2
    w_gradient  = d/dw (y - w*x - b)**2
                = 2 *(y - w*x - b) * d/dw (y - w*x - b)
                = 2 *(y - y_pred) * (-x)
                = -2x * ( y - y_pred )
                = 2x * ( y_pred - y )
                == x *( y_pred - y )
    加入 learning_rate 後，常數可省略成
                
    b_gradient  = 2 *(y - y_pred) * (-1)
                = -2 * ( y - y_pred )
                = 2 * ( y_pred - y ) 
                == ( y_pred - y )
    '''
    y_pred = w * x + b
    w_gradient = torch.mean(x * (y_pred - y))
    b_gradient = torch.mean(y_pred - y)

    return w_gradient, b_gradient



# 計算成本
def compute_cost(x, y, w, b):
    y_pred = w * x + b
    cost = torch.mean((y - y_pred) ** 2)
    return cost

    
# 梯度下降
def gradient_descent(x, y, w, b, learning_rate, run_iter):
    prev_cost = float('inf')
    for _ in range(run_iter):
        w_gradient, b_gradient = compute_gradient(x, y, w, b)
        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient
        cost = compute_cost(x, y, w, b)

        # 若 cost 前後變化太小, 就終止 forloop
        if abs(prev_cost - cost) < 1e-6:  # 加收斂條件
            break
        prev_cost = cost

        if _ % 1000 == 0:
            print(f'Iter {_}: Cost={cost:.3f}, w={w:.3f}, b={b:.3f}')

    return w, b



# 主流程
if __name__ == '__main__':

    # 若有 GPU 使用 GPU 進行運算，否則使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 data
    file_path = './csv/salary.csv'
    data = load_and_preprocess_data(file_path)
    x_train, x_test, y_train, y_test = data

    # 轉換為 GPU tensor
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # 初始的 weight 和 bias
    w = torch.tensor(0.0, device=device)  # 初始 weight
    b = torch.tensor(0.0, device=device)  # 初始 weight
    learning_rate = 0.001
    run_iter = 10000


    # 訓練模型
    w_train, b_train = gradient_descent( x_train, y_train, w, b,learning_rate,\
                                        run_iter)
    
    # 變成普通 Tensor，不參與梯度計算
    w_train = w_train.detach()
    b_train = b_train.detach()

    # print(f'w_gradient: {w_gradient}, b_gradient: {b_gradient}')

    # 測試預測 y = w * x + b
    y_pred_test = w_train * x_test + b_train

    # 結果輸出（格式化）
    print("\n測試結果對比：")
    print("Actual  | Predicted  | Error")
    print("-----------------------------")
    for actual, pred in zip(y_test[:10], y_pred_test[:10]):  # 只顯示前10筆
        print(f"{actual:6.1f}  | {pred:6.1f}  | {abs(actual - pred):6.1f}")

    print("-----------------------------")
    error = torch.abs(y_test - y_pred_test)
    print(f"Mean Absolute Error: {torch.mean(error):6.1f}")


