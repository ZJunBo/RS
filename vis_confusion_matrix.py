"""
# -*- coding:utf-8 -*-
@Project : DCA-master
@File : vis_confusion_matrix.py
@Author : ZhangJunBo
@Time : 2023/3/17 下午6:42
"""
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
path = "/home/zjb/PycharmProjects/uda/log/ast_old/urban/confusion_matrix-1679473593.338917.npy"
confusion = np.load(path)

d = confusion/confusion.sum(0)

confusion = np.around(d, decimals=2)
print(confusion)
# coding=utf-8
# from sklearn.metrics import confusion_matrix

save_flg = True

# confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))  # 设置图片大小

# 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色

plt.imshow(confusion, cmap='Reds')
# plt.colorbar()  # 右边的colorbar

# 2.设置坐标轴显示列表
indices = range(len(confusion))
classes = ['Backg.', 'Build.', 'Road', 'Water', 'Barren', 'Forest', 'Agricul.']

# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, classes, rotation =45, fontsize=18)  # 设置横坐标方向，rotation=45为45度倾斜
plt.yticks(indices, classes, fontsize=18)

# 3.设置全局字体
# 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
# ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
plt.rcParams['font.sans-serif'] = ['TimesNewRoman']
plt.rcParams['axes.unicode_minus'] = False



# 4.设置坐标轴标题、字体
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title('Confusion matrix')

# plt.xlabel('预测值')
# plt.ylabel('真实值')
# plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  # 可设置标题大小、字体

# 5.显示数据
normalize = True
fmt = '.2f' if normalize else 'd'
thresh = confusion.max() / 2.

for i in range(len(confusion)):  # 第几行
    for j in range(len(confusion[i])):  # 第几列
        plt.text(j, i, format(confusion[i][j], fmt),
                 fontsize=18,  # 矩阵字体大小
                 horizontalalignment="center",  # 水平居中。
                 verticalalignment="center",  # 垂直居中。
                 color="white" if confusion[i, j] > thresh else "black")

# 6.保存图片
if save_flg:

    plt.savefig('confuse_2.png', dpi = 350)

# 7.显示
# plt.tick_params(labelsize = 18)

plt.show()