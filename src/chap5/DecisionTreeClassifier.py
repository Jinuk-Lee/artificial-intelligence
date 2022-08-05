import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()

wine.info()
wine.describe()

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
"처음 3개의 열을 넘파이 배열로 바꿔서 data배열에 저장하고 마지막 class열을 넘파이 배열로 바꿔서 target배열에 저장" \
"이제 훈련세트와 테스트 세트로 나누겠습니다."
from sklearn.model_selection import train_test_split


train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
"샘플 개수를 20퍼 정도만 테스트 세트로 나눔"

print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
"standardscaler클래스로 훈련세트를 전처리 실행"

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
"표준점수로 변환된 train_scaled 와 test_scaled 로 로지스틱 회귀를 실행합니다." \
"그런데 두 세트 점수가 모두 낮으니 과소적합되었습니다."

print(lr.coef_, lr.intercept_)

# 결정 트리
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
"사이킷런에있는 결정트리 클래스로 모델을 훈련시킵니다. 그러니까 훈련세트가 과대적합이 되었습니다."

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
"그리고 plot_Tree()로 결정트리를 그림으로 출력해봅니다."
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
"너무 많아서 복잡하니까 깊이를 제한해서 출력해봅니다." \
"결정트리에서 예측하는 방법은 리프노드에서 가장많은 클래스가 예측클래스가 됩니다. "

"그리고 불순도라는게 있는데 결정트리가 최적의 질문을 찾기위한 기준이라고 합니다. 그리고 " \
"부모와 자식노드사이의 불순도 차이를 정보이득이라고 하는데 그 정보이득이 최대화가 되도록 학습한다고 합니다."

# 가지치기
"무작정 트리를 끝까지 자라나게 하면 훈련세트에 과대적합되어 일반화가 잘 안된다고 합니다." \
"그래서 가지치기를 하는 데 가장 간단한 방법은 트리의 깊이를 지정하는 것이라고 합니다."
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

"훈련세트의 성능은 낮아졌지만 테스트 세트의 성능은 거의 그대로입니다."
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

print(dt.feature_importances_)