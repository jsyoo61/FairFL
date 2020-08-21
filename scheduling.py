# Small Test
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(42) # seed 42기준

U = 100 # hospital
U_cnt = list()  # hospital cnt
E = 10  # center
E_cnt = list()  # center cnt
A = 6   # anthena
# A_cnt = A

count = 0
flag1 = E
flag2 = 0
flag3 = 0
timer = list()
y_index = list()
count_a = 76

# set hospital
for idx in range(U):
  U_cnt.append(random.randint(0,128))

# set center
for idx in range(E):
  E_cnt.append(random.randint(1000, 1024))

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

while flag2 == 0 and count < count_a:
  count = count + 1

  if count >= count_a - 20:
    if U < 300:
      # print("check", E, flag1)
      timer.append(E-flag1)
      y_index.append(U)
      flag1 = E
      U = U + 10
      U_cnt = list()
      E_cnt = list()
      for idx in range(U):
        U_cnt.append(random.randint(0,128))
      for idx in range(E):
        E_cnt.append(random.randint(1000, 1024))

      count = 0

    else:
      flag2 = 1

  E_index = np.argmax(E_cnt)
  if E_cnt[E_index] <= 0:
    print(E_cnt[E_index])
    continue
  # print(E_cnt[E_index]) # 센터 중 제일 빈 공간이 큰 센터를 찾아냄

  # 빈 공간에 맞춰서 연결할 안테나의 개수를 정함
  for idx in range(A):
    if E_cnt[E_index] > 128 * A: # 128 * 6 = 768
      A_temp = A
    else:
      A_temp = int(E_cnt[np.argmax(E_cnt)]/128)

  for idx in range(A_temp):
    if E_cnt[E_index] // 128 == 0:
      U_index = find_nearest(U_cnt, E_cnt[E_index])
      E_cnt[E_index] = E_cnt[E_index] - U_cnt[U_index]
      U_cnt[U_index] = 0

      if E_cnt[E_index] <= 0:
        print('1', flag1)
        print(flag1)
        flag1 = flag1 - 1
        break

    else:
      U_index = np.argmax(U_cnt)
      print(E_cnt[E_index], count, U_cnt[U_index])
      E_cnt[E_index] = E_cnt[E_index] - U_cnt[U_index]
      U_cnt[U_index] = 0

      if E_cnt[E_index] <= 0:
        print('2', flag1)
        flag1 = flag1 - 1
        break

  # set hospital
  for idx in range(U):
    U_sum = random.randint(0, 16)
    if U_cnt[idx] + U_sum < 128:
      U_cnt[idx] = U_cnt[idx] + U_sum
    else:
      break

  if flag1 == 0:
    flag2 = 1

print("센터가 가지는 공간 : ", E_cnt)
plt.plot(y_index, timer)


# %%
# 1번 plot Random
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(42) # seed 42기준

U = 100 # hospital
U_cnt = list()  # hospital cnt
E = 10  # center
E_cnt = list()  # center cnt
A = 6   # anthena

count = 0
flag1 = E
flag2 = 0
flag3 = 0
timer = list()
y_index = list()
count_a = 76

# set hospital
for idx in range(U):
  U_cnt.append(random.randint(0,128))

# set center
for idx in range(E):
  E_cnt.append(random.randint(1000, 1024))

# check
# print(E_cnt)
# print(E_cnt[random.randint(0, len(E_cnt))])

while flag2 == 0 and count < count_a: # base steps : 64, 80% : 51, 120% : 76
  count = count + 1 # step
  # print(count, count_a)
  if count >= count_a - 20:  
    if U < 300:
      timer.append(E-flag1)
      y_index.append(U)
      flag1 = E
      U = U + 10
      U_cnt = list()
      E_cnt = list()
      for idx in range(U):
        U_cnt.append(random.randint(0,128))
      for idx in range(E):
        E_cnt.append(random.randint(1000, 1024))
    else:
      flag2 = 1

    count = 0

  E_index = random.randint(0, len(E_cnt)-1) # random center choice
  A_temp = random.randint(0, A) # random anthena count choice

  if E_cnt[E_index] < 0:
    continue

  # random anthena count choice
  for idx in range(A_temp):
    U_index = random.randint(1, len(U_cnt)-1)
    E_cnt[E_index] = E_cnt[E_index] - U_cnt[U_index]
    U_cnt[U_index] = 0
    if E_cnt[E_index] <= 0:
      flag1 = flag1 - 1
      break

  for idx in range(U):
    U_sum = random.randint(0, 16)
    if U_cnt[idx] + U_sum < 128:
      U_cnt[idx] = U_cnt[idx] + U_sum

  if flag1 == 0:
    flag2 = 1

print(E_cnt, timer)
plt.plot(y_index, timer)
