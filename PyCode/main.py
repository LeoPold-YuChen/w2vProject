# add: 醫生，護士，護理人員 in stopdict
import envSetting
from b import cut, happy, inputProcessing
import time
import draw
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

totalSentence = ["我最近經常胃部不適，感覺有點不對勁。",
                 "我一直有排尿的問題，這讓我很困擾。",
                 "我最近有一些妊娠紋方面的問題，不知道怎麼辦。",
                 "我耳朵裡一直有聲音，而且還有點耳鳴。",
                 "我的視力最近變得模糊，感覺有點擔心。",
                 "我的牙齒有點敏感而且時常痛，讓我很不舒服。",
                 "有時候我會感覺呼吸有點困難，這種情況持續了有一段時間。",
                 "我有尿頻和尿急的問題，這種情況一直困擾著我。",
                 "我最近經常經痛，這讓我感到很煩惱。",
                 "我有點耳朵痛而且聽力似乎有下降的情況。"]
for i in totalSentence:
    # sentence = input('請輸入症狀:\n')
    sentence = i
    # envSetting.show()
    envSetting.addEnv()  # jieba position
    envSetting.loadJiebaDict()  # userdict + dict
    # print(sentence)
    cut_jieba = cut(sentence)  # cut

    start = time.time()  # 17sec
    vocab_w2v, vectors_w2v, model = envSetting.w2vLoad()
    print(time.time()-start)

    # '''Officiallsy''' 腦殘，多寫一次stopdict
    # start = time.time()  # 7sec -> 0.5sec
    # vocab, vectors = stopDict_2(vocab_w2v, vectors_w2v)
    # print(time.time()-start)
    # print(np.array(vectors).shape)

    a, vocabb = inputProcessing(cut_jieba, vocab_w2v)
    updateVectors = []
    for i in a:  # combine wordEmbedding
        updateVectors.append(vectors_w2v[i])
    print(f'停用辭典後: {vocabb}')
    # print(np.array(updateVectors).shape) 詞嵌入向量

    '''Ans'''
    classes, embedding = happy(vocabb, model, updateVectors)
    '''Plot'''
    try:
        i
    except NameError:
        if classes == 0 and embedding == 0:
            draw.plot(classes, embedding, vocabb)
