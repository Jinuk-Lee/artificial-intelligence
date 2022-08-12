import wget
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
"wget 명령으로 데이터를 다운로드"
import numpy as np
"넘파이 함수를 사용해서 npy파일을 읽어 넘파이 배열을 준비합니다." \
"k- 평균 모델을 훈련하기 위해 (샘플 개수, 너비, 높이)크기의 3자원 배열을( 샘플 개수, 너비*높이 )크기를" \
" 가진 2차원 배열로 변경 합니다."
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


# KMeans 클래스
"사이킷런의 k - 평균 알고리즘은 sklearn.cluster 모듈 아래에 KMeans 클래스에 구현되어있습니다." \
"여기서는 비지도 학습이므로 fit()메소드에서 타깃 데이터를 사용하지 않습니다."
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

"군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장된다."
"클러스터는 3개를 사용해서 0,1,2"
print(km.labels_)

"레이블 0, 1, 2로 분류된 샘플의 개수를 확인한다."
print(np.unique(km.labels_, return_counts=True))

import matplotlib.pyplot as plt
"각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위해 " \
"유틸리티 함수 draw_fruits() 를 만들어 본다."
def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
"draw_fruits() 함수는 (샘플 개수, 너비, 높이)의 3차원 배열을 " \
"입력받아 가로로 10개씩 이미지를 출력한다."
draw_fruits(fruits[km.labels_==0])
"- `km.labels_==0` 과 같이 쓰면 `km.labels_` 배열에서 값이 0인 위치는 `True` , " \
"그 외는 모두 `False` 가된다."
"넘파이는 이런 불리언 배열을 이용해 원소를 선택할 수 있는데, " \
"이를 불리언 인덱싱이라고 한다."
"불리언 인덱싱을 적용하면 `True` 인 위치의 원소만 추출한다."
draw_fruits(fruits[km.labels_==1])
draw_fruits(fruits[km.labels_==2])


# 클러스터 중심
"이미지로 출력하려면 100 * 100 크기의 2차원 배열로 바꿔야 한다."
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
"KMeans 클래스는 훈련 데이터 샘플에서 클러스터 중심까지 거리로 변환해 주는 " \
"transform() 메소드를 갖고 있다."
print(km.transform(fruits_2d[100:101]))
"KMeans 클래스는 가장 가까운 클러스터 중심을 " \
"예측 클래스로 출력하는 predict() 메서드를 제공한다."
print(km.predict(fruits_2d[100:101]))
"샘플을 확인 해본다."
draw_fruits(fruits[100:101])
"알고리즘이 최적의 클러스터를 찾기 위해 반복한 횟수는 " \
"KMeans 클래스의 n_iter_ 속성에 저장된다."
print(km.n_iter_)


# 최적의 k 찾기
"클러스터 개수k를 2~6까지 바꿔가며 KMeans 클래스를 5번 훈련합니다." \
"그리고 fit() 메서드로 모델을 훈련한 후 inertia_속성에 저장된 이너셔값을 inertia 리스트에 " \
"추가합니다. 마지막으로 그 리스트에 저장된 값을 그래프로 출력합니다."
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()