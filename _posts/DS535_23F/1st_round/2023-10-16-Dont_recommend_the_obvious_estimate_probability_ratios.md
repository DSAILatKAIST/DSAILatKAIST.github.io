---
title:  "[RecSys 2022] Don't recommend the obvious: estimate probability ratios"
permalink: 2023-10-16-Dont_recommend_the_obvious_estimate_probability_ratios.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [RecSys 2022] Don't Recommend the Obvious: Estimate Probability Ratios

# Title
**Don't Recommend the Obvious: estimate probability ratios**

# 1. Introduction
순차적 추천 시스템은 과거 사용자와 아이템 간의 상호 작용의 순서를 고려하여 사용자가 다음에 소비할 아이템을 예측한다. 이러한 순차적 추천 시스템은 광범위한 분야에서 사용되어 왔지만, 이런 순차적 추천 시스템에는 사용자에게 개인 맞춤화 된 아이템을 추천하는 것이 아니라 일반적으로 인기 있는 아이템을 추천하는 경우가 많다는 한계점이 존재한다. 
이에 최근 일부 논문에서는 보통의 사용자들에게 인기가 있는 항목들 사이에서 특정 사용자에게 개인화된 다음 아이템을 얼마나 잘 찾는가 측정하는 **Popularity-Sampled Metrics**를 사용하여 추천 시스템을 평가한다. 저자들은 이러한 Popularity-Sampled Metrics가 PMI(Point-wise Mutual Information)와 밀접한 관련이 있다는 것을 보이며, Popularity-Sampled Metrics를 최적화하기 위해 PMI를 피팅하는 두 가지 방법을 제시한다.  

# 2. Sequential Recommendations
- 좋은 추천은 context에 따라 변화하며 현재 sequential recommendation의 아키텍처는 이벤트의 시계열을 자연스럽게 모델링하여 사용자의 변화하는 관심사를 포착한다. 따라서 저자들은 개인 맞춤화된 추천 시스템에 관심이 있기 때문에 유연한 신경 순차 접근 방식을 채택했지만, 이러한 방법은 MF(Matrix Fatorization)를 포함한 모든 확률 추천 시스템에 일반화 될 수 있다. 
- 또한 현재의 추세를 고려해 Transformer 기반 시스템을 이용해 실험하며 이러한 시스템은 다양한 데이터 셋에서 최신 성능을 보여주고 있다. 하지만 실험을 간단하게 유지하기 위해 sequence에 다른 추가적인 맥락 정보는 포함하고 있지 않다.  

- 표기법
```
1. 사용자 셋: U = {u~1~, u~2~, ... , u~|U|~}
2. 아이템 셋: V = {v~1~, v~2~, ... , v~|V|~}'
3. 사용자 u가 상호 작용한 아이템 셋(발생 순서대로):  S~u~ = [v~1~^u^, v~2~^u^, ... , v~n~~u~^u^]
```

# 3. Metrics and Evaluation
1. 평가 데이터 분할: 'Leave-One-Out' 분할 방법을 통해 각 사용자에 대해 아이템 셋의 마지막 아이템을 test, 그 앞의 아이템을 validation, 나머지 아이템들은 train에 사용합니다. 이때 이러한 분할에 있어 일부 train 아이템이 validation이나 test 항목보다 나중에 나타날 수 있기 때문에 실제 추천 시스템의 현실과는 일치하지 않는다는 단점이 있으나, 이는 이전 연구 결과들과 직접 비교하기 위함이다. 

2. HIT@k 평가 메트릭: 저자들은 HIT@k 성능 지표를 사용하며 이는 다음과 같이 정의될 수 있다. 
![HIT@k 수식](https://i.ibb.co/17JxrSC/HIT-k.png)
이때 r은 test 아이템 v~i~ 의 순위이며, 후보 아이템 목록 C = [c~1~, ... , c~|C|~] 와 함께 조건부 확률을 사용하여 정렬되었을 때 해당 순위가 k 이하인 경우를 나타낸다. 

3.  후보 아이템 목록 선택: 후보 아이템 C를 선택하는 데에는 아이템들의 인기에 비례하여 샘플링을 하거나 무작위로 샘플링하는 등 다양한 방법이 있을 수 있다. 

# 4. Point-wise Mutual Information 
Point-wise Mutual Information(PMI)은 두 개의 사건 x와 y간의 연관성을 측정하기 위해 사용되는 지표로 두 사건이 함께 발생할 확률을 독립적인 경우에 기대되는 확률로 나눈 로그값을 나타낸다. 이를 통해 두 사건이 독립적인 경우에 비해 얼마나 자주 함께 발생하는지 측정한다. 

![PMI 공식](https://i.ibb.co/fM05bSv/PMI.png)

PMI는 추천 시스템에서 사용자와 아이템 간의 연관성을 평가하는 데에 사용되며 특히 사용자 에게 개인화된 추천을 제공하고 사용자가 일반적으로 인기 있는 항목을 추천 받는 것을 방지한다. 

# 5. Relation between Probability Ratios and Sampled Metrics
![Metrics 수식](https://i.ibb.co/bmkN50K/Metrics.png)

저자들은 샘플링된 메트릭과 확률 비율 추정 사이의 관계를 명확하게 하기 위해 HIT@k 메트릭과 유사한 분류 작업을 보인다. 
분류 작업에서, 사용자의 시퀀스 S~u~ 에 대한 아이템 목록을 생성하며, 이 중 하나는 실제 다음에 상호작용이 있을 아이템이고 이 분류 작업의 목표는 모델이 실제 아이템의 인덱스를 추론하는 것이다. 이러한 Bayes Classificcation은 균일한 사전 확률을 갖는 클래스인 k의 조건부 확률을 최대화하는 것이며, 이를 통해 아이템의 순서를 정할 수 있다. 

- 후보 아이템들이 균일하게 샘플링 되었다면 P~C~(v)=1/|v| 가 상수가 되면서, 아이템들은 다음 아이템을 예측하는 Probability Ratio, p(v|S~U~), 로 아이템들의 순위가 매겨진다. 
- 만약, 후보 아이템들이 인기 분포를 따를 경우 아이템들의 순위는 위 식과 같이 PMI로 매겨진다
즉, Probability Ratio 또는 PMI로 아이템들의 순위를 매길 수 있다. 

또한, 비복원 추출 방식으로 샘플링하는 것이 classification에 있어 당연해보일 수 있지만, 복원 추출방식으로 샘플링하는 것이 샘플 크기 |C|에 대한 의존성을 제거하고 PMI와 더 깔끔하게 연결될 수 있다. 

# 6. Methods for Fitting the Point-wise Mutual Information
저자들은 Training 단계에서 PMI를 추정하는 두 가지 접근 방식을 제시한다. 
 
### 6-1. Ratio Estimation by Classification
첫 번째 방법은 evaluation(section 5)에 사용하고 있는 것과 동일한 classification을 사용하여 모델을 훈련시키는 것이다. 이떄 마지막 layer는 Softmax 함수를 사용하며 아래 과정을 통해 마지막 layer가 PMI에 비례한다는 것을 알 수 있다. ![6-1 Ratio Estimation by Classification](https://i.ibb.co/2jPNDTg/6-1.png) 

하지만 이러한 접근 방식은 대규모의 somftmax를 다뤄야한다느 단점이 있으며 이는 시간과 비용이 많이 들 수 있다는 문제점을 갖는다. 또한 대규모 데이터셋에서는 적절한 성능을 달성하는 것이 어려울 수 있으며, 모델이 모든 항목에 대해 Probability ratio를 정확하게 추정하기 어렵다.

### 6-2. Embedded Prior Model
![6-2Ebbedded Prior Model](https://i.ibb.co/WHPkwKQ/6-2.png)
2번째 방법은 PMI와 인기 분포를 나타내는 두 개의 함수를 사용하는 것이며, 이후 저자들은 다음 아이템을 예측하는 모델로 만들기 위해 두 가지 다른 loss function을 사용하고 이를 최소화하는 방법으로 학습시켰다. 


# 7. Experiments 
- **Research Question**
1. RQ.1 Probability raio 추정은 Popularity-sampled HIT@k 메트릭을 개선하는가
2.  RQ.2 training에 있어 후보 아이템 샘플 사이즈를 조정하고 비복원 추출 방식을 사용하는 것이 기존 Popularity-sampled HIT@k 성능과 어떻게 다른가
3.  RQ.3  복원 추출 방법을 사용했을 때 HIT@k의 샘플 사이즈를 조정했을 때 두 가지 방법들의 성능은 어떻게 달라지는가 
4.  RQ.4  Probability ratio 추정 방법이 덜 인기가 있는 아이템을 추천하는가 

- **Datasets and Experimental Setup**
저자들은 영화 추천 웹사이트 MovieLens에서 수집한 영화 데이터 셋을 사용하였으며, 각각 1백만 개와 2천만 개의 영화 평가를 포함하고 있는 ML-1M, ML-20M 두 가지 버전의 데이터 셋을 사용하였다. 이때 실제 평가 점수는 고려되지 않았으며 사용자가 평가를 한 것 자체를 긍정적인 action으로 간주하였다. 또한 이전 연구들에서는 사용자와 영화와의 상호작용이 5회 미만인 경우를 제거하였지만, 통계적 재현을 위해 ML-20M에서는 이 단계를 생략하고 ML-1M에서는 5회 미만의 영화들이 제거되었다.
전처리 후의 데이터셋 요약 통계는 다음과 같다 

![Dataset Statistics after Pre-processing](https://i.ibb.co/T46QT2t/7-2.png)

저자들은 사용자의 최종 상호작용을 테스트 아이템으로 사용하며 아이템 추천 순위 목록의 정확성을 평가하기 위해 HIT@k 메트릭스를 사용한다. 또한 TOP k 추천 항목의 평균 인덱스를 확인해 모델이 추천하는 아이템이 전체 아이템 분포에서 어디에 위치하는지 측정하고, 이 값이 높을 수록 이 모델이 꼬리쪽에 분포한 아이템을 더 추천한다는 것이다. 

- **Model Architectures and Implementation Details**
저자들은 training 과정에서 PMI 를 추정하는 것의 효과성을 검증하기 위해 두 가지 순차 추천 시스템인 SasRec과 Bert4Rec을 사용한다. 이 두 모델은 Transformer를 기반으로 하는 순차 추천 시스템으로 사용자가 이전에 상호작용한 아이템 시퀀스를 고려해 다음에 상호작용할 아이템의 조건부 확률을 추정한다. 

>SasRec: 단방향 Transformer encoder를 사용하며 아이템 시퀀스를 왼쪽에서 오른쪽으로만 학습한다. 

> Bert4Rec: SasRec과 유사한 모델 아키텍처를 갖고 있지만 양방향 메커니즘으로 양쪽에서의 constext를 학습하여 유저의 시퀀스 학습


- **Conditional Probability vs. Probability Ratio**
RQ.1에 대해 답하기 위해 저자들은 SasRec과 Bert4Rec의 조건부 확률과 probability ratio를 비교했다. 결론적으로 아래 표와 같이 Probability ratio를 추정하는 것이 일관되게 popularity sampled 된 HIT@k 메트릭스의 성능을 향상시킨다는 것을 알 수 있다. 
![conditional probability vs, probability ratio](https://i.ibb.co/gzSKBHX/7-3.png)

- **Number of Samples in Classification task**
아래 그림은 RQ.2에 대한 답을 제공할 수 있다. 아래 그림을 통해 Classification의 샘플 수를 늘리면 Popularity-sampled 된 HIT@k 메트릭스의 성능은 향상된다는 것을 알 수 있으며, 또한 균일 분포에서 인기 분포로의 변화는 큰 변화를 주지만 샘플 추출 방법은 결과에 큰 영향을 미치지 않는다는 것을 알 수 있다. 

![number of samples in classification task](https://i.ibb.co/7VvpRfw/7-4.png)

- **Numbers of Samples in HIT@k with Replacement**
RQ.3에 답하기 위해 저자들은 복원추출된 HIT@k 메트릭을 고려하였으며, 그 이유는 비복원 추출로 |C|를 증가시키면 점점 균일하게 샘플링된 메트릭스와 유사해지기 때문이다. 위 그림 중 왼쪽 그림처럼 샘플 수가 제품 수에 비해 작을 때 즉 |C|<<|V| 인 경우 classification은 embedded prior model 보다 비슷하거나 약간 더 좋은 성능을 보인다. 하지만 샘플 수가 제품 수와 동일한 수준으로 증가하면 embedded prior model이 classification을 능가한다. 따라서 대량의 아이템 사이에서 몇 안되는 후보를 선택해야 하는 경우 embedded prior model을 사용할 것을 저자들은 권장했다. 

- **Popularity of Recommended Items**
저자들은 RQ.4에 답하기 위해 PMI를 이용해 추정하는 것이 추천하는 데에 있어 덜 인기있는 항목을 고려하는지 확인하였다. 아래 표를 통해 Probability ratio를 이용한 방법이 원래의 Bert4Rec 모델보다 추천 영화 상위 10개의 평균 인덱스가 더 높다는 것을 알 수 있으며 이는 이 모델이 대중적으로 덜 인기가 있는 영화를 더 많이 추천한다는 것을 의미한다. 따라서 Probability ratio를 추정하는 것이 대중적으로 덜 인기가 있는 아이템을 추천하는 데에 도움이 된다고 결론 지을 수 있다. 
 ![popularity of recommended items](https://i.ibb.co/PTCxQBP/7-6.png)

# 8. Conclusion
이 연구는 추천 시스템이 당연하게 인기 있는 아이템이 아닌 개인화된 아이템을 추천하는 시스템을 구축하기 위한 방법을 제안했다. 또한 저자들은  **1) 조건부 확률과 사전 확률을 별개로 추정하는 방법 2) 분류를 통한 Ratio 추정 3) Embedded prior model** 세 가지 방법 중 두 번째와 세 번째 방법이 더 뛰어나며 두 번째 방법은 샘플 사이즈가 작을 때 효율적이며, 대량의 아이템 사이에서 후보 아이템 수가 많을 때는 세 번째 방법이 더 효율적일 수 있다고 결론지었다. 

반면 저자들은 조건부 확률을 사용하는 것보다 PMI가 더 좋은 성능을 내지만, 사용자에 대한 정보가 없는 등의 상황에서는 그렇지 않을 수 있다는 것을 언급하였다. 또한 인기가 없는 꼬리 부분에 위치한 아이템들의 비율을 추정하는 데에 있어 노이즈가 많을 수 있으며, 이로 인해 이러한 방법인 주가 되어 사용되어진다면 사용자에게 관련이 없거나 이상한 항목을 보여줄 위험이 있을 수 있다는 것을 언급하였다. 

# Author Information
Roberto Pellegrini
Amazon Development Centre Scotland, United Kingdom
Wenjie Zhao
Amazon Development Centre Scotland, United Kingdom
Iain Murray
Amazon Development Centre Scotland, United Kingdom and School of Informatics, University of Edinburgh, United Kingdom


# Reference & Additional Materials
Roberto Pellegrini, Wenjie Zhao, and Iain Murray. 2022. Don’t recommend the obvious: estimate probability ratios. In Proceedings of the 16th ACM Conference on Recommender Systems (RecSys '22). Association for Computing Machinery, New York, NY, USA, 188–197. https://doi.org/10.1145/3523227.3546753
