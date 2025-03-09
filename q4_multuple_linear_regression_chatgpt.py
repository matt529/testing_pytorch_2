'''
本篇是由 chatgpt 自行撰寫的內容, 幾乎不做修正, 
了解自己和 gpt 撰寫的差距在哪 ~ -.-", 
不過 gpt 有說, @ 這個運算式, 能使 code 美化, 仍需了解原理，
但如果不懂原理還是用 * 會比較好讀懂。
看那精美的撰寫方式 ~~~~~~~~~~~~~~~~
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# 讀取 CSV 並進行預處理
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")
    
    # 將 EducationLevel 轉換為數值
    education_map = {"高中以下": 0, "大學": 1, "碩士以上": 2}
    data["EducationLevel"] = data["EducationLevel"].map(education_map)
    
    # One-Hot Encoding 處理 City 欄位
    encoder = OneHotEncoder(drop="first")  # 自動刪除第一個 dummy variable
    city_encoded = encoder.fit_transform(data[["City"]]).toarray()
    city_columns = encoder.get_feature_names_out(["City"])
    data[city_columns] = city_encoded
    data.drop(columns=["City"], inplace=True)

    # 分割特徵與標籤
    x = data.drop(columns=["Salary"]).to_numpy()
    y = data["Salary"].to_numpy()

    return train_test_split(x, y, test_size=0.2, random_state=87)


# 計算 Cost Function
def compute_cost(x, y, w, b):
    y_pred = x @ w + b  # 使用矩陣乘法提高效率
    return np.mean((y - y_pred) ** 2)


# 計算梯度
def compute_gradients(x, y, w, b):
    y_pred = x @ w + b
    error = y_pred - y
    
    w_gradient = np.mean(x * error[:, np.newaxis], axis=0)  # 批量計算梯度
    b_gradient = np.mean(error)

    return w_gradient, b_gradient


# 梯度下降演算法
def gradient_descent(x, y, w, b, learning_rate=1e-3, run_iter=10000):
    for i in range(run_iter):
        w_gradient, b_gradient = compute_gradients(x, y, w, b)
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

        if i % 1000 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iter {i}: Cost={cost:.3f}, w={w}, b={b:.3f}")

    return w, b


# 主程式
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_preprocess_data("./csv/salary2.csv")

    # 初始化權重與超參數
    w_init = np.random.randn(x_train.shape[1])  # 隨機初始化
    b_init = 0
    learning_rate = 0.01
    run_iter = 10000

    # 訓練模型
    w_train, b_train = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, run_iter)

    # 測試預測
    y_pred_test = x_test @ w_train + b_train


