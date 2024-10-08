from a import s
from sklearn.metrics import classification_report, accuracy_score
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


if __name__ == '__main__':
    aa = '1'
    while (aa != '0'):
        superU = 0
        metricsX = []
        metricsY = []
        aa = input('plz選擇編號:\n(1)使用者自己輸入(2)預設單筆(3)預設12筆(0)Exit: ')
        if aa == 'superU':
            superU = 1
            print('進入superU')
            aa = input('plz選擇: ')
        if aa == '0':
            break
        elif aa == '1':
            a = input('plz輸入字串: ')
            s(a, superU)
        elif aa == '2':
            a = '我耳朵裡一直有聲音，而且還有點耳鳴。'
            s(a, superU)
        elif aa == '3':
            a = ["我最近經常心臟跳很快，感覺有點不對勁。",
                 "我一直有排尿的問題，這讓我很困擾。",
                 "我最近有一些妊娠紋方面的問題，不知道怎麼辦。",
                 "我耳朵裡一直有聲音，而且還有點耳鳴。",
                 "我的視力最近變得模糊，感覺有點擔心。",
                 "我的牙齒有點敏感而且時常痛，讓我很不舒服。",
                 "有時候我會感覺呼吸有點困難，這種情況持續了有一段時間。",
                 "我有尿頻和尿急的問題，這種情況一直困擾著我。",
                 "我最近經常經痛，這讓我感到很煩惱。",
                 "我有點耳朵痛而且聽力似乎有下降的情況。"]
            a = ["最近我常常感到心跳加速，心臟有些不適，希望檢查一下內部器官的健康狀況。",
                 "我感覺最近體力下降，時常感到疲倦，可能需要做個全身檢查，了解身體狀況。",
                 "我經常有尿頻的問題，尤其在晚上更為明顯，影響了我的睡眠質量。",
                 "有時候我感到尿道有刺痛的感覺，尿尿的時候特別不舒服。",
                 "我最近發現有些妊娠紋，而且這些紋路變得越來越明顯，不知道是否需要諮詢專業醫生。",
                 "我經常感到腹部有疼痛感，特別是月經期間，可能需要檢查一下是否有婦科問題。",
                 "我耳朵裡總是有嗡嗡聲，並且伴有輕微的耳鳴現象，不知道是否需要做耳朵檢查。",
                 "有時我感覺呼吸有些困難，耳朵也有點痛，可能與耳鼻喉科有關。",
                 "最近我的視力變得模糊，看東西不清楚，感覺需要去眼科檢查一下。",
                 "我經常感到眼睛乾澀，視力似乎有所下降，可能需要檢查一下眼睛的健康。",
                 "我感覺牙齒對冷熱特別敏感，常常感到牙痛，希望去牙科看看是否有問題。",
                 "最近我發現牙齦有些腫脹，刷牙時出血，可能需要牙科醫生的檢查和治療。"]
            for i in a:
                metricsX.append(s(i, superU))
        else:
            continue
        if len(metricsX) > 0:
            metricsY = ['內科', '泌尿科', '婦產科', '耳鼻喉科', '眼科', '牙科',
                        '耳鼻喉科', '泌尿科', '婦產科', '耳鼻喉科']
            metricsY = ['內科', '內科', '泌尿科', '泌尿科', '婦產科', '婦產科', '耳鼻喉科', '耳鼻喉科', '眼科', '眼科', '牙科', '牙科']
            print("Accuracy:", accuracy_score(metricsX, metricsY))
            # print("Classification Report:\n",
            #       classification_report(metricsX, metricsY))
