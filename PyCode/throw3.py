import jieba.analyse
import pandas as pd
from collections import Counter


jieba.load_userdict('../Ref/userdict.txt')
jieba.set_dictionary('../Ref/dict.txt')


def preprocessing():
    path = pd.read_csv('../Ref/combined_sentences.csv')  # load corpus
    path.drop('1', axis=1, inplace=True)
    print(f'pandas: \n{path.head(5)}')
    print('='*100)

    sentence = path.values  # np.shape=(199,1) and this 1 is string
    sentence = [s[0].strip() for s in sentence]  # list multi 1 ndim string
    reg_jieba = []
    for i in sentence:
        s2_list = list(jieba.cut(i, cut_all=False))
        print(s2_list)
        reg_jieba.append(s2_list)
    print('精確model： ', reg_jieba[0])  # 這是list不是str
    print(reg_jieba)
    tf(reg_jieba)


def tf(a):
    all_words = [word for b in a for word in b]

    word_count = Counter(all_words)
    print(word_count)


preprocessing()
