# machine learning question 1: 基本檔建立
# 學習 matplotlib
# 學習 y = w*x + b
'''
引用 salary.csv, 呈現 YearsExperience and salary 分布
學習 matplotlib 製圖：直線 plot、點陣圖 scatter、字體加入
 
'''

import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager    # 加入新語言，這次沒用

# 讀取 csv, local端下載
data = pd.read_csv("./csv/salary.csv", encoding="utf-8")

# 讀取 *.csv, 兩行，有keys: YearsExperience, Salary, 作為 x,y 
# y_list = y.tolist()   # 轉成 Python list
# y_dict = y.to_dict()  # 轉成 Python dict
x = data["YearsExperience"]
y = data["Salary"]


'''
w, b = 0, 0
p_pred = w * x + b  # 目標是找出一條最能代表點陣圖分佈的直線, 初值 w, b 初始為 0
'''


# 不同字體安裝方式，font_manager.addfont("Traditional_Chinese_font.ttf")
mlp.rc('font', family='Microsoft JhengHei, arial',)

# 繪出 matplotlib 
def plot_predict(w, b):
    y_pred = w * x + b
    plt.plot(x, y_pred, color="blue", label="predict")  # 直線
    plt.scatter(x, y, color="red", label="real_data" )  # 點陣圖
    plt.title("Years-Salary relationship")
    plt.xlabel("年資 (年) ：Years of experience")
    plt.ylabel("薪資 (千元) ： Salary of experience")
    plt.xlim(0, 12)
    plt.ylim(-60,140)
    plt.legend()
    return plt


# w、b自由使用,得出2個常用的 matlab 圖
plot_predict(10, 10).show()  



