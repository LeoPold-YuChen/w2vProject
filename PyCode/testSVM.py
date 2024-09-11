import pickle


classes = ['耳鼻喉科', '眼科', '牙科']


def testSVM(classes, jbVec, jbVoc):
    svmM, ssM = load()  # ssM standardScaler Model
    NjbVec = ssM.transform(jbVec)
    res = svmM.predict(NjbVec)
    res = res.tolist()
    a = res.count(0)
    b = res.count(1)
    c = res.count(2)
    total = [a, b, c]
    # print(f'{a} {b} {c} {max(a,b,c)}')
    # 沒做當2個出現次數一樣時的處理，
    # 現在會預設跑到同類最前面那個[3,9,9] -> 1
    return classes[total.index(max(total))]


def load():
    svmM = pickle.load(open('../Ref/svmModel.pkl', 'rb'))
    ssM = pickle.load(open('../Ref/ssModel.pkl', 'rb'))
    return svmM, ssM
