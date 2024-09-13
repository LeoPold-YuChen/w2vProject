# 使用w2v大型語言模型實作智慧健康助理

### 建立環境
```
pip install -r requirements.txt
```

### 跑程式去建議掛哪一個科別
```
python main.py
```

### 成果展示1.
輸入'左下方的牙齒在喝到冰水時會敏感'

可以看到重點有抓到牙齒

![img](https://i.imgur.com/dDA6t6c.png)

### 成果展示2.

10筆測資:

![img](https://imgur.com/DWP1zY1.png)

12筆測資:

![img](https://imgur.com/KRHFSG1.png)
---
### Env:
缺少權重檔與一些詞典，因此會無法使用(wiki權重檔3GB)

### Feature:
1. 把w2v換成bert，泛用性應該可以提升很多
2. 內科難抓，可能與資料集有關
3. 把svm換成randomForest效果應該會更好
4. 泌尿科與婦產科可以再分得更清楚，應該可以再接randomForest

### 筆記:
1. svm的ovr代表一對多
2. svm適合資料量相符的環境，本次牙科嚴重缺乏
3. standerScaler的fit_transform與transform
4. w2v調整好參數就盡量不要動，否則接下來的ML模型整個都會亂掉

### 心得:
1. NLP的base model直接選bert比較好(先用最好再慢慢修剪(加入其他ML做區分))
2. 先做Classification再嘗試Clustering
