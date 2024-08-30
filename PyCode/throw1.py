'''
python強制預設ANSI因此必須加上面那行
Ref: https://blog.csdn.net/zc666ying/article/details/105601764

記得git clone https://github.com/ldkrsi/jieba-zh_TW.git
'''

import envSetting
from project.PyCode.throw2 import DictPreprocessingCutAns


envSetting.show()
envSetting.addEnv()


# sys.setdefaultencoding('UTF8')    # jiebaba.jiebaEnvirments()
a = DictPreprocessingCutAns()

print('finish')
