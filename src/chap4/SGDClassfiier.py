import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv')

# 이거는 4-1절에서 했던걸 토대로...!
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 훈련세트와 테스트세트로 쪼개기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 특성을 표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input) # 훈련세트에서 학습한 통계값으로
test_scaled = ss.transform(test_input) # 테스트 세트도 변환하기!!

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
# loss 매개변수 : 손실함수의 종류 지정
# max_iter 매개변수 : 수행할 에포크 횟수를 지정

sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)

print('훈련 세트 점수: ', sc.score(train_scaled, train_target))
print('테스트 세트 점수: ', sc.score(test_scaled, test_target))

### 에포크와 과대/과소적합

import numpy as np
sc = SGDClassifier(loss = 'log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)
# fit()을 사용하지 않고 partial_fit() 만 사용하려면, 훈련 세트에 있는 전체 클래스 레이블을 partial_fit() 에 전달해주어야 한다.

for _ in range(0, 300):
  sc.partial_fit(train_scaled, train_target, classes = classes) # 훈련세트의 클래스 레이블 전부 전달!
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch') # train_score[0] 이 0번째 epoch의 훈련 세트 score
plt.ylabel('score')
plt.show()

sc = SGDClassifier(loss='log', max_iter = 100, tol = None, random_state=42) # tol은 뭐임?
sc.fit(train_scaled, train_target)
print('훈련 세트 점수: ', sc.score(train_scaled, train_target))
print('테스트 세트 점수: ', sc.score(test_scaled, test_target))

# cf. hinge 손실 함수
sc = SGDClassifier(loss='hinge', max_iter = 100, tol = None, random_state=42) # tol은 뭐임?
sc.fit(train_scaled, train_target)
print('훈련 세트 점수: ', sc.score(train_scaled, train_target))
print('테스트 세트 점수: ', sc.score(test_scaled, test_target))