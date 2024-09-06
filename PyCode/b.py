'''Use more then 2 times'''
import jieba


def cut(sentence):
    cut_jieba = []
    a = jieba.cut(sentence, cut_all=False)
    cut_jieba.append(list(a))
    # print(f'精確: {cut_jieba}')
    return cut_jieba


def stopDict_2(vocab, vectors):
    stopwords_file = "../Ref/stopwords-zh-v2.txt"

    # 使用集合來存儲停用詞
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)

    # 使用列表推導式來過濾詞彙和向量
    filtered_vocab = [word for word in vocab if word not in stopwords]
    filtered_indices = [i for i, word in enumerate(vocab) if word not in stopwords]

    # 根據過濾的索引過濾向量
    vectors = vectors[filtered_indices]

    return filtered_vocab, vectors


def stopDict(vocab, vectors):
    stopwords_file = "../Ref/stopwords-zh-v2.txt"
    stopwords = []
    with open(stopwords_file, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.append(line.strip())

    filtered_vocab = []
    filtered_vectors = []
    for i, word in enumerate(vocab):
        if word not in stopwords:
            filtered_vocab.append(word)
            filtered_vectors.append(i)
    # filtered_vocab = [word for word in vocab if word not in stopwords]
    vocab = filtered_vocab  # filtered_vocab is [4, 6, 7, 11, 12, 14, 15]
    vectors = vectors[filtered_vectors]
    return vocab, vectors


'''
使用者的輸入，
回傳切字vocab與在wiki的索引位置
'''


def inputProcessing(cut_jieba, vocab):
    a = []
    vocabb = []
    for i in cut_jieba:
        for m in i:
            if m in vocab:
                for j, k in enumerate(vocab):
                    if m == k:
                        a.append(j)
    # print(a) #index
    for i in a:
        # print(vocab[i])
        vocabb.append(vocab[i])
    return a, vocabb


'''
Finish 1Step
'''


def happy(vocabb, model):
    maxres = []
    totalres = []
    classes = ['內科', '泌尿科', '婦產科', '耳鼻喉科', '眼科', '牙科']
    for i in classes:
        res = []
        for j in vocabb:
            if ('手' or '腳') in j and '外科' == i:
                res.append(1)
            elif '泌尿科' == i:
                res.append(model.wv.similarity(i, j)-0.05)
            elif '牙科' == i:
                res.append(model.wv.similarity(i, j)-0.1)
            elif '內科' == i:
                res.append(model.wv.similarity(i, j)+0.05)
            else:
                res.append(model.wv.similarity(i, j))
        maxres.append(max(res))
        totalres.append(res[:])
    print(f'7科類別相似最高: {max(maxres)}\n各科相似度分析: {maxres}')
    print(f'相似度最高科名: {classes[maxres.index(max(maxres))]}')
    return classes, totalres
