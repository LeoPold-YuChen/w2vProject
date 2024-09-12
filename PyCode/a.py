import envSetting
from b import cut, happy, inputProcessing
from time import time
import draw


def s(sentence):
    # envSetting.show()
    envSetting.addEnv()  # jieba position
    envSetting.loadJiebaDict()  # userdict + dict
    # print(sentence)
    cut_jieba = cut(sentence)  # cut
    start = time()  # 17sec -> 2.5sec
    vocab_w2v, vectors_w2v, model = envSetting.w2vLoad()
    print(time()-start)

    a, vocabb = inputProcessing(cut_jieba, vocab_w2v)
    updateVectors = []
    for i in a:  # combine wordEmbedding
        updateVectors.append(vectors_w2v[i])
    print(f'停用辭典後: {vocabb}')
    if not vocabb:
        print('jieba或輸入是空的')
        return
    # print(np.array(updateVectors).shape) 詞嵌入向量

    '''Ans'''
    classes, embedding = happy(vocabb, model, updateVectors)
    '''Plot'''
    if not (classes == 0 and embedding == 0):
        draw.plot(classes, embedding, vocabb)
