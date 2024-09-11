import pandas as pd
import pickle
# from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def trainSVM(classes, updateVectors, cut_jieba, model):
    a = pd.read_csv("../Ref/threeClass.csv")
    x = a.iloc[:, 0].tolist()
    y = a.iloc[:, 1].tolist()
    filtered_x, filtered_y, vectors = w2v(x, y, model)
    new_train_x = preprocessing(vectors)
    svm(new_train_x, filtered_y)


def svm(ssModel, y):
    svmModel = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
    svmModel.fit(ssModel, y)
    pickle.dump(svmModel, open('../Ref/svmModel.pkl', 'wb'))


def preprocessing(vectors):
    '''
    資料縮放是根據column來進行
    Choose
    ------
    standardScaler: Formula -> (X-X_mean)/X_std

    RobustScaler:   由於會捨棄outlier，且使用者可能輸入outlier，
                    屆時將無法辨別，故放棄使用。
    '''
    s = StandardScaler()
    new_train_x = s.fit_transform(vectors)
    pickle.dump(s, open('../Ref/ssModel.pkl', 'wb'))
    print(f'ss平均值: {s.mean_}')
    return new_train_x


def w2v(x, y, model):
    filtered_x = [word for word in x if word in model.wv]
    filtered_y = [y[i] for i, word in enumerate(x) if word in model.wv]
    vectors = [model.wv[word] for word in filtered_x]
    # print(f'filtered_x: {filtered_x}')
    # print(f'vectors: {np.array(vectors).shape}')
    # print(f'filtered_y: {np.array(filtered_y).shape}')
    return filtered_x, filtered_y, vectors
