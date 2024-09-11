import threading
from collections import deque
import time


def a(dq):
    for i in range(10):
        dq.append(i)
        time.sleep(0.1)


def b(dq):
    for i in range(1, 10):
        dq.append(i*10)
        time.sleep(0.1)


dq = deque()
aa = threading.Thread(target=a, args=(dq,))
bb = threading.Thread(target=b, args=(dq,))
aa.start()
bb.start()
time.sleep(3)
bb.join()
aa.join()
while dq:
    print(dq.popleft())
print('finish')
