import envSetting
from b import stopDict_2, cut, happy, inputProcessing
import time
import draw
from gensim.models import word2vec
import threading
'''
Basic:

Attention:
1. 記得git clone https://github.com/ldkrsi/jieba-zh_TW.git


sentence example:
1. '昨天搬重物時肋骨位置感到劇烈疼痛，可能是因為受傷或肌肉拉傷。'
2. '今天手指僵硬，可能因氣候變冷或過度使用手部，建議適當保暖。'
3. '未來有機會交到女友嗎，希望是會讓心臟跳得很快的女友'
4. '左下方的牙齒再喝到冰水時會敏感'
'''


model = None


def load_model(event):
    global model
    model = word2vec.Word2Vec.load('../Ref/word2vec.zh.300.model')
    event.set()  # 通知主執行緒模型已經加載完成


# 建立一個事件對象
aaa = threading.Event()
aa = threading.Thread(target=load_model, args=(aaa,))
aa.start()
sentence = input('請輸入症狀:\n')
# envSetting.show()
envSetting.addEnv()  # jieba position
envSetting.loadJiebaDict()  # userdict + dict
# print(sentence)
cut_jieba = cut(sentence)  # cut

start = time.time()
aaa.wait()
print(f'multithread is used {time.time()-start} second')

start = time.time()  # 17sec
vocab_w2v, vectors_w2v, model = envSetting.w2vLoad(model)
print(time.time()-start)

'''Officiallsy'''
start = time.time()  # 7sec -> 0.5sec
vocab, vectors = stopDict_2(vocab_w2v, vectors_w2v)
print(time.time()-start)

a, vocabb = inputProcessing(cut_jieba, vocab)

print(vocabb)

'''Ans'''
classes, embedding = happy(vocabb, model)

'''Plot'''
draw.plot(classes, embedding, vocabb)
