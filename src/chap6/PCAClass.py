"""# 주성분 분석"""

"""## PCA 클래스"""

"**사이킷런은 PCA 클래스로 주성분 분석 알고리즘을 제공한다.**" \
"과일 사진 데이터를 다운로드하여 넘파이 배열로 적재한다."
!wget https: // bit.ly / fruits_300_data - O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)
"사이킷런은 sklearn.decomposition 모듈 아래 PCA 클래스로 주성분 분석 알고리즘을 제공한다." \
"- PCA 클래스의 객체를 만들 때, `n_components` 매개변수에 주성분의 개수를 지정해야 한다."
"비지도 학습이기 때문에 `fit()` 메소드를 제공하지 않는다."
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

"배열의 크기를 확인 합니다."
print(pca.components_.shape)


"주성분을 그림으로 그려봅니다."
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr)  # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다.
    rows = int(np.ceil(n / 10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols * ratio, rows * ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:  # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


draw_fruits(pca.components_.reshape(-1, 100, 100))

"주성분을 찾앗으므로 원본 데이터를 주성분에 투영해서 턱성의 개수를 50개로 줄일 수 있습니다."
print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)

print(fruits_pca.shape)

"""## 원본 데이터 재구성"""
"10,000개의 특성을 50개로 줄였기 때문에 어느 정도 손실이 발생할 수밖에 없습니다." \
"하지만 최대한 분산이 큰 방향으로 데이터를 투영했기 때문에 원본 데이터를 상당 부분 재구성할 수 있다." \
"PCA 클래스는 이를 위해 inverse_transform() 메서드를 제공한다."
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

"이 복원한 10,000개의 데이터를 100 * 100 크기로 바꾸어 100개씩 나누어 출력한다."
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start + 100])
    print("\n")

"""## 설명된 분산"""
"설명된 분산은 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지를" \
" 기록한 값을 설명된 분산이라고 한다."
"PCA 클래스의 explained_variance_ratio_ 에 각 주성분의 설명된 분산 비율이 기록되어 있다" \
"이를 이용해  총 분산 비율을 얻을 수 있다.."
print(np.sum(pca.explained_variance_ratio_))

"적절한 주성분의 개수를 찾기 위해 plot() 함수로 설명된 분산을 그래프로 출력한다."
plt.plot(pca.explained_variance_ratio_)
plt.show()

"""## 다른 알고리즘과 함께 사용하기"""

"과일 사진 원본 데이터와 PCA로 축소한 데이터를 지도 학습에 " \
"적용해 보고 어떤 차이가 있는지 알아 본다."
"3개의 과일 사진을 분류해야하므로 로지스틱 회귀모델을 모델을 만듭니다."
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

"타깃값을 만들기 위해 파이썬 리스트와 정수를 곱하면 리스트 안의 원소를 정수만큼 반복합니다." \
target = np.array([0] * 100 + [1] * 100 + [2] * 100)

"상능을 가늠해 보기위해 교차검증을 수행합니다."
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

"이 값을 PCA로 축소한 fruits_pca 를 사용했을 때와 비교한다."
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

"설명된 분산의 50%에 달하는 주성분을 찾도록 PCA 모델을 만들어 본다."
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

"몇 개의 주성분을 찾았는지 확인해본다."
print(pca.n_components_)

"이 모델로 원본 데이터를 변환한다."
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

"교차 검증의 결과를 알아본다."
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

"차원 축소된 데이터를 사용해 k-평균 알고리즘으로 클러스터를 찾아본다."
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

"KMeans 가 찾은 레이블을 사용해 과일 이미지를 출력한다."
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

"- 훈련 데이터의 차원을 줄이면 또 하나 얻을 수 있는 장점은 **시각화**이다." \
"`km.labels_` 를 사용해 클러스터별로 나누어 산점도를 그려본다."
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()