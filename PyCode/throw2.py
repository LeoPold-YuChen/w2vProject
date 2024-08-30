
import jieba.analyse
import pandas as pd

# ================================================ #


def DictPreprocessingCutAns():
    # 載入詞典
    jieba.load_userdict('../Ref/userdict.txt')
    jieba.set_dictionary('../Ref/dict.txt')
    # 前處理
    path = pd.read_csv('../Ref/combined_sentences.csv')
    path.drop('1', axis=1, inplace=True)
    print(f'pandas:\n{path.head(5)}')
    print('='*100)

    # 切字
    sentence = path.values
    sentence = [s[0].strip() for s in sentence]
    # print(sentence[3])
    reg = []
    reg_jieba = []

    for i in sentence:
        s2_list = jieba.cut(i, cut_all=False)
        reg = ' '.join(s2_list)
        reg_jieba.append(reg)

    print('accModel： ', reg_jieba[0])  # 這是list不是str
    reg_jieba = str(reg_jieba)
    tags = jieba.analyse.extract_tags(reg_jieba, topK=200, withWeight=True)

    for tag in tags:
        print('word:', tag[0], 'tf-idf:', tag[1])
    # for tag in tags:
    #   if tag[1]<0.3:
    #     print('word:', tag[0], 'tf-idf:', tag[1])
    #   else:
    #     pass
    print('='*100)

    # 停用詞表
    stopwords_file = "../Ref/stopwords-zh.txt"
    # sentence = tags
    sentence = " ".join([word for word, score in tags])
    # 載入停用詞表
    stopwords = set()
    with open(stopwords_file, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())
    # 分詞並移除停用詞
    words = jieba.cut(sentence)
    filtered_words = []
    for word in words:
        if word not in stopwords:
            filtered_words.append(word)
    ans = "".join(filtered_words)
    print("分詞結果:", ans)
    return ans
# ================================================ #
