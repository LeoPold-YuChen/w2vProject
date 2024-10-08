'''
Only run once
'''


import os
import sys
import jieba
from b import stopDict_2
from gensim.models import word2vec
import logging

model = None


# show position
def show():
    print('='*100)
    print(f'現在所在位置:\n{os.getcwd()}')
    print('='*100)
    return None


# envirment configuration
def addEnv():
    sys.path.append('../jieba-zh_TW')
    jieba.dt.cache_file = '../jieba-zh_TW.cache'
    # print("添加環境變數:")
    # for path in sys.path:
    #     print(path)


def loadJiebaDict():
    logging.getLogger('jieba').setLevel(logging.ERROR)
    jieba.load_userdict('../Ref/userdict-corpus-v2.txt')
    jieba.set_dictionary('../Ref/dict.txt')


def Load():
    global model
    model = word2vec.Word2Vec.load('../Ref/word2vec.zh.300.model')


def w2vLoad():  # w2v+stopDict
    # 9sec
    # model = word2vec.Word2Vec.load('../Ref/word2vec.zh.300.model')
    if model is None:
        raise ValueError("Model is not loaded. Please call load_model() first.")
    vocab = model.wv.index_to_key
    vectors = model.wv[vocab]
    # 11sec
    vocab, vectors = stopDict_2(vocab, vectors)
    return vocab, vectors, model


Load()
