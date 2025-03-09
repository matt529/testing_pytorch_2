'''
改編自q4_multuple_linear_regression_chatgpt.py --
仿照 chatgpt 原檔, 重新編寫成自己習慣使用的撰寫方式, 
不一定比較好, 希望這樣能幫助自己吸收
'''
'''
草稿筆記
1 準備好資料
2 設定一個模型 (ex 直線) (y-y_pred)**2
3 設定 cost function
4 設定優化器 optimizer

多元線性回歸 multuple linear regression
y = w1x1 + w2x2 +... + wnxn + b
'''
'''
資料預處理
	Label Encoding：
		EducationLevel 改為 0,1,2 數值

	OneHotEncoder：獨熱向量 (One-Hot)
		City 進行分割在簡化

	Cost Function：
	cost= ( 真實數據 - 預測直 )**2，距離**2，越小越好
		= ( y - y_pred )**2
		= ( y - (w1x1+w2x2+w3x3+ b) )**2
	y_pred
		= ( w* x ).sum(axis=1) + b
		= 

	梯度計算(gradient)：
		w1斜率(w1_gradient)
		= 2 * x1 * (y_pred -y) 又 learning rate 2倍不影響學習
		= x1 * (y_pred -y)

		w1_gradient = x1 * (y_pred -y)
		w2_gradient = x2 * (y_pred -y)
		w3_gradient = x3 * (y_pred -y)
		w4_gradient = x4 * (y_pred -y)
		wi_gradient = xi * (y_pred -y)
		b_gradient = (y_pred - y).mean()

	Feature Scaling：Standardization


'''
# 讀取 csv, local端下載
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# data[]: YearsExperience, EducationLevel, City, Salary
data = pd.read_csv("./csv/salary2.csv", encoding="utf-8")

# 將 EducationLevel 轉換為數值
education_map = {"高中以下":0, "大學":1, "碩士以上":2}
data["EducationLevel"] = data["EducationLevel"].map(education_map)

# One-Hot Encoding 'City'
encoder = OneHotEncoder()    # 自動刪除第一個 dummy variable
encoder.fit(data[["City"]])
city_encoded = encoder.transform(data[["City"]]).toarray()
data[["CityA", "CityB", "CityC"]] = city_encoded
data.drop(columns=["City", "CityC"], inplace=True)  # 刪除 City 和被簡化的 C

# data.drop(["City", "CityC"], axis=1) axis=1 行，橫列縱行，與線代相反喔!
# 特徵與標籤
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]].to_numpy()
y = data["Salary"].to_numpy()


# 初值設定完成!
#================================================================


'''
# a,b,c,d = x_train[0]
# print(f"a:{type(a)} b:{type(b)} c:{type(c)} d:{type(d)}")

# 月薪(y) = w1*年資 + w2*學歷 + w3*CityA + w4*CityB + b
# [w1, w2, w3, w4], b # 這邊一開始隨興取值
# w = np.array([1,2,3,4])
# b = 0

# # 預測值
# y_pred = (x_train * w).sum(axis=1) + b
# # print(f"y_pred: {y_pred}")

# # 設定 cost function
# # cost = (真實數據 - 預測值)**2
# cost = ((y_train - y_pred)**2).mean()
# print(f"cost: {cost}")
'''

# 計算成本 cost function
def compute_cost(x, y, w, b):
    '''
    w = np.array([1,2,3,4])
    b = 0
    cost = (真實數據 - 預測值)**2
    cost = ((y_train - y_pred)**2).mean()
    print(f"cost: {cost}")
    y_pred = (x_train * w).sum(axis=1) + b
    '''
    # 預測值
    # y_pred = (w * x).sum(axis=1) + b == y_pred = x @ w + b
    y_pred = x @ w + b  # 矩陣乘法
    # 計算 cost function
    # cost = ((y - y_pred)**2).mean()
    # cost  = np.mean((y - y_pred)**2)  # 同上，但使用 numpy 
    cost = torch.mean((y - y_pred)**2)  # 同上，但使用 torch tensor
    return cost

# w = np.array([1,2,2,4])
# b = 0
# compute_cost(x_train, y_train, w, b)


# gradient 梯度計算
def compute_gradients(x, y, w, b):
    '''
    # 設定 optimizer - gradient descent = 根據斜率改變參數
    # y_pred：
    # y_pred = (x_train * w).sum(axis=1) + b

    # b 斜率：
    # b_gradient = (y_pred - y_train).mean()
    # print(f"b_gradient: {b_gradient}")

    # # wi 斜率：
    # wi_gradient = (x_train[:, i] * (y_pred - y_train)).mean()
    '''
    # y_pred：
    # y_pred = (w * x).sum(axis=1) + b == y_pred = x @ w + b
    y_pred = x @ w + b
    error = y_pred - y
    # error_newaxis = error[:, np.newaxis]    # 原(n,) 需要轉為 (n,1)
    error_newaxis = error[:, None] # 原(n,) 需要轉為 (n,1)
    # w_gradient = np.mean( x * error_newaxis, axis=0 )
    # b_gradient = np.mean(error)
    w_gradient = torch.mean( x * error_newaxis, axis=0 )
    b_gradient = torch.mean(error)

    '''
    # print(f"x = {x.shape[1]}")
    # w_gradient = np.zeros(x.shape[1])   # x.shape[1] 為 x 有幾個維度: 4:int
    # for i in range(x.shape[1]):
    #     w_gradient[i] = (x[:, i] * (error)).mean()
    # w_gradient = np.mean(error)
    '''

    return w_gradient, b_gradient




# gradient descent 梯度下降
def gradient_descent( 
        x, y, w, b,
        learning_rate=0.001, run_iter=10000 ):
    '''
    learning_rate：學習率，
    run_iter：迭代次數，
    p_iter：每 n 次 print() 設為 None，後續會被設定成最後一次印出
    # w , b = np.array([1,2,2,4]), 1
    # w_gradient, b_gradient = compute_gradients(x_train, y_train, w, b)
    # learning_rate = 0.001

    # # print(f"cost = {compute_cost(x_train, y_train, w, b)}")

    # w = w - w_gradient * learning_rate
    # b = b - b_gradient * learning_rate

    # print(f"cost = {compute_cost(x_train, y_train, w, b)}")
    '''

    # 影片教學有設定, 目前修改不用
    # c_hist, w_hist, b_hist = [], [], []

    for i in range(run_iter):
        w_gradient, b_gradient = compute_gradients(x, y, w, b)
        w = w - w_gradient * learning_rate
        b = b - b_gradient * learning_rate
        cost = compute_cost(x, y, w, b)

        # w_hist.append(w)
        # b_hist.append(b)
        # c_hist.append(cost)

        # if i % p_iter == (run_iter-1):
        #     print(f"cost={cost: 2f}, w={w}, b={b}")
        # 每 1000 forloop 才印一次
        if i % 1000 == 0:
            [ w1, w2, w3, w4 ] = w
            print(f"cost={cost: 2f},w=[{w1: 2f},{w2: 2f},{w3: 2f},{w4: 2f}], \
                    b={b: 2f}" )

    # return w, b, w_hist, b_hist, c_hist, cost
    return w, b, cost


if __name__ == "__main__":


    '''
    # 分割成 4 種，訓練用特徵、訓練用標籤、測試用特徵、測試用標籤
    # random_state = 87 固定一種隨機方式
    # code 長度超過 80, 修改成 2行方便閱讀
    '''
    # 將 x, y 整理成 4 份 data 用來訓練和驗證, 部分數據固定不變
    vectors_list = train_test_split(x, y, test_size=0.2, random_state=87)

    # 轉為 PyTorch Tensor 並搬移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(vectors_list[0], dtype=torch.float32).to(device)
    x_test = torch.tensor(vectors_list[1], dtype=torch.float32).to(device)
    y_train = torch.tensor(vectors_list[2], dtype=torch.float32).to(device)
    y_test = torch.tensor(vectors_list[3], dtype=torch.float32).to(device)

    # 初始化，加入參數
    # np.random.randn(x_train.shape[1])  # 隨機初始化
    w_init = np.array([1,2,2,4])
    b_init = 0
    # w, b 丟入 GPU
    w_init = torch.tensor(w_init, dtype=torch.float32).to(device)
    b_init = torch.tensor(b_init, dtype=torch.float32).to(device) 
    learning_rate = 0.01
    run_iter = 10000

    # 訓練模型
    w_train, b_train, cost = \
        gradient_descent( 
            x_train, y_train, w_init, b_init, learning_rate, run_iter)

    print(f'-'*20)
    print(f'w={w_train}, b={b_train}, cost={cost:.2f}.....')
    print(f'-'*20)


    # 以測試集 檢驗訓練模型 w, b
    y_pred_test = x_test @ w_train + b_train

    print("Actual  | Predicted")
    print("-------------------")
    for actual, pred in zip(y_test, y_pred_test):
        print(f"{actual:6.1f}  | {pred:6.1f}  | {abs(actual - pred):6.1f}")


    print(y_test.shape, y_pred_test.shape)

    error = torch.abs(y_test - y_pred_test)
    print("Mean Absolute Error:", error)
    print(f"Mean Absolute Error: {torch.mean(error):6.1f}")







