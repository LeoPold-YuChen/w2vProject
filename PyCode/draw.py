from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(classes, embedding, vocabb, superU):
    plt.ion()
    if superU:
        print(np.array(embedding).shape)
        print(embedding)

    plt.rcParams['font.sans-serif'] = ['simsun']  # 使用黑體字體來顯示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

    # 轉置數據，使 classes 和 embedding 對調
    data = np.array(embedding).T

    # 繪製熱圖
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="YlGn")

    # 顯示所有科別名稱
    ax.set_xticks(np.arange(len(data[0])))
    ax.set_xticklabels(classes)  # 原本是 vocabb
    ax.set_yticks(np.arange(len(vocabb)))
    ax.set_yticklabels(vocabb)  # 原本是 classes

    # 設置標籤的旋轉角度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 在每個方格中標註數值
    for i in range(len(vocabb)):
        for j in range(len(data[0])):
            text = ax.text(j, i, round(data[i, j], 2),
                           ha="center", va="center", color="black")

    ax.set_title("Annotated Heatmap of Medical Department Embeddings")
    fig.tight_layout()
    plt.show()
    plt.pause(.1)  # 显示1s
    plt.close()

