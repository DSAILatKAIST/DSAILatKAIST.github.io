---
title:  "[SIGIR 2022] User-controllable Recommendation Against Filter Bubbles"
permalink: 2023-10-16-User-controllable_Recommendation_Against_Filter_Bubbles.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [SIGIR-22]User-controllable_Recommendation_Against_Filter_Bubbles



## **1. Problem Definition**[](https://dsailatkaist.github.io/template.html#1-problem-definition)

과거 몇 년 동안 추천시스템은 '개인화된 정보 필터링'이라는 명목으로 정보의 과잉을 감당하기 어려운 사용자들에게 불필요한 정보를 필터링 해주어 빠른 발전을 거두었지만, '필터버블(Filter Bubble)'에 대한 논의는 항상 이어져 왔었다. 

필터버블이란, 추천시스템이 사용자-아이템 간 상호작용을 기반으로 기존 사용자의 선호도에 일치하는 아이템만 계속해서 노출시키는 현상을 가리킨다.  이런 현상이 반복되면, 유사한 아이템만 노출되는 확률이 계속해서 커지게 되고, 사용자가 다양한 카테고리의 콘텐츠를 접할 수 있는 기회가 점점 줄어지게 된다.

즉, 장기적인 관점에서 필터버블은 아이템 혹은 콘텐츠의 다양성과 오리지널리티를 추천시스템에서 배제시키게 되고, 이는 필연적으로 정보의 편식으로 인해 사용자로 하여금 왜곡 효과를 낳게 된다.

따라서, 필터버블을 완화하는 것 또한 추천시스템에서 중요한 과제로 자리매김하게 되었다.

## **2. Motivation**[](https://dsailatkaist.github.io/template.html#2-motivation)

필터버블을 완화하는 방안으로 기존 연구에서는 '다양성(Diversity)'과 공정성(Fairness)'을 높이는 방법을 제안했었다.

- 추천에서의 다양성:
다양성은 추천 목록에서 유사성이 다른 아이템을 생성하도록 장려하는 방안이다. 이는 후처리 및 엔드-투-엔드 방법으로 나눌 수 있다. 전자는 몇몇 모델에 의해 생성된 추천 목록을 다시 순위 지정을 통해 다양화시키며, 후자는 모델 훈련 및 예측 과정에서 정확도와 다양성의 균형을 맞추는 방향으로 진행된다. 그러나 이러한 방식들 역시 단순히 사용자에게 다양한 아이템을 추천한 후 사용자의 피드백을 통해 새로운 아이템 카테고리를 발굴해가는 것으로 많은 시간이 필요하다. 심지어 다양한 아이템을 추천하는 단계에서 사용자 선호도와 관련 없는 아이템을 많이 가져올 수 있다는 단점이 있다. 

- 추천에서의 공정성:
추천시스템에서 공정성을 달성하기 위해서는 다양한 사용자 그룹 또는 아이템 카테고리 간에 균형을 맞추어야 한다. 이것은 특정 그룹에게 더 많은 추천을 하도록 조절하거나 특정 카테고리의 아이템을 다른 것보다 자주 포함시키는 것을 의미하여 필터버블을 어느 정도 완화할 수 있다. 그러나 이렇게 균형을 맞추는 과정에서, 일부 사용자 또는 아이템 그룹에 대한 추천 정확성을 희생시킬 수 있다. 예를 들어, 만약 특정 사용자 그룹에게 더 많은 공정성을 부여하려면 그 그룹에 대한 추천 목록에서 다른 사용자 그룹의 선호를 무시하고 그 그룹에 맞춰야 할 수 있기 때문에 그 그룹에 대한 정확한 추천이 희생되고 사용자 경험이 저하될 수 있다.

이렇듯 기존 접근 방식은 다양성, 공정성을 고려하여 필터버블을 완화하지만 정확성과 사용자 경험을 희생해야 한다는 단점이 있다. 

또한, 사용자의 피드백을 통해서 추천시스템이 모델 훈련-예측의 무한 루프를 도는 과정에서, 사용자는 추천 결과를 수동적으로만 받아들이게 때문에 진정한 '개별맞춤' 결과를 생성하기까진 많은 시간과 비용을 필요로 하다. 즉, 비록 추천시스템이 사용자의 선호도를 기반으로 결과를 생성하긴 하지만, 다양성과 공정성 향상 과정에서 다시 불필요한 정보까지 포함시킬 수 있기 때문에 사용자가 자신에게 필요한 정보만을 얻기 위해선 생성된 추천 결과에 대해 'like' 혹은 'dislike' 등 지속적인 피드백을 제공하고 학습을 시켜야 한다.

따라서, 동 연구는 사용자가 직접 컨트롤을 통해 자신이 원하는 추천 결과를 생성할 수 있게끔 'User-Controllable Recommendation System(UCRS)' 방안을 제시했다.

![rsloop](https://i.ibb.co/9wXYx34/rsloop.png)

사용자가 제어할 수 있는 추천시스템인 UCRS는 기존 추천시스템 외에 아래 3가지 기능을 추가시켰다.

 1. 필터버블의 수준을 측정하여 사용자에게 알려주는 필터버블 경고 기능
 2. 4가지 수준의 제어 명령 기능
 3. 사용자 제어에 따라 추천결과를 조정하는 응답 메커니즘

이로써, 동 연구는 사용자가 추천시스템의 동작을 제어할 수 있는 방법을 제공하여 사용자의 참여를 촉진하고, 사용자가 더 많은 표현력과 제어권을 가지게 함으로써 사용자 경험을 향상시키는 것을 목적으로 삼았다.

이와 더불어, 동 연구는 앞서 말한 응답 메거니즘 단계에서 사용자 A가 1년 전에는 액션 영화를 많이 시청했지만 현재는 코미디 영화에 더 관심이 있을 수 있는 것처럼 시간 지남에 따라 사용자의 선호도가 변할 수 있다는 점까지 인식하였다. 이에 따라, 사용자 표현의 오래된 정보가 추천에 미치는 영향을 완화하는 데 중점을 둔 'User-Controllable Inference(UCI)' 프레임 워크가 제안되었다.

UCI는 사용자가 제어 명령을 제공하면,  반 사실(counterfactual)인 대조적 추론을 사용하여 과거에 나온 사용자 표현의 영향을 줄이는 방안이다. 예를 들어, 여성인 사용자 B는 여성 사용자 그룹이 선호하는 영화 리스트에 질려 남성 사용자 그룹이 선호하는 영화 리스트를 추천시키는 제어 명령을 내렸다고 가정하자. 이 때, 반 사실인 대조적 추론은 '사용자 B가 남성이라면 추천 리스트는 어떻게 변할까?'라는 질문의 대한 예측이라고 생각할 수 있다. 즉, UCI는 오래된 사용자 표현이 폐기되는 반 사실 세계를 상상하고 이런 반 사실 조건에 맞는 새로운 추천 결과를 생성한다. 이로써 과거 사용자-아이템 간 상호작용 패턴에 극한되지 않고 사용자가 원하는 추천을 얻을 수 있다.

## 3. Preliminary
동 연구는 Method를 확립하기 전에 다양한 사용자 그룹에 대한 필터버블 결과를 분석하는 사전 실험을 수행했다:

 1.  대표적인 추천 모델인 Factorization Machine (FM)을 DIGIX-Video, Amazon-Book 및 ML-1M과 같은 세 개의 공개 데이터셋에 훈련시킴.
 2. 각 사용자에 대해 상위 10개의 추천 아이템을 수집함.
 3. 사용자 그룹을 ID, 성별 및 연령을 고려한 사용자 특성(User Features)과 아이템 카테고리에 대한 관심도 등을 고려한 사용자 상호작용(User Interactions) 두 가지 요인으로 분류함.
 4. FM이 생성한 추천결과와 사용자 그룹에 따른 사용자의 과거 상호작용 패턴을 비교함.

분석 결과로는, 사용자 특성과 아이템 특성(Item Features) 2가지 측면에서 필터버블이 존재한다는 사실이 발견되었다.

![fbresult](https://i.ibb.co/0c9Qt8R/fbresult.png)

이미지 2(a)는 DIGIX-Video의 남성 및 여성 사용자별 상위 3개 아이템 카테고리에 대한 과거 분포를 시각화한 결과이다. 여성 사용자 그룹은 로맨스 영화를 더 선호한 반면, 남성 사용자 그룹은 액션 영화를 더 선호했다. 그 결과, 이미지 2(b)와 2(c)에서 알 수 있듯이, 추천 결과에서도 편향된 분포를 유지하게 되었다.

이러하듯이, 사용자는 계속해서 유사한 아이템을 추천 받게 된다. 추천 모델은 이러한 편향을 강화하고 상위 특정 카테고리를 더 노출시키는 경향으로 이어져 결국 남성과 여성 사용자 그룹 간의 심각한 분리를 야기시키게 된다.

또한, 이미지 2(d) 및 2(e)는 Amazon-Book 및 ML-1M 데이터셋에 대해 사용자 상호작용에 따라 나눈 결과이다. 동 결과에서도 과거 사용자 상호작용 패턴에 따른 카테고리 편향 증폭이 발견되었다. 즉, 사용자으로부터 가장 큰 관심을 받은 카테고리가 이후 추천 목록에서도 증가된 것이다. 이는 필터버블의 강화를 초래하고 사용자의 관심을 좁혀 사용자 그룹 분리로 이어지게 된다.


## **4. Method**[](https://dsailatkaist.github.io/template.html#3-method)
전반적으로 동 연구는 커버리지(Coverage), 격리지수(Isolation Index), 최다 카테고리 지배도(Majority Category Domination, MCD) 등 지표를 통해 필터버블의 수준을 실시간으로 감지하고 사용자에게 알람을 보내는 경고 기능을 구현했다.

또한, 제어 명령 기능과 관련해서는 앞서 사전 실험에서 사용자 특성과 아이템 특성 2가지 측면에서 필터버블이 존재한다는 사실을 발견함에 따라, UCRS를 설계할 때에도 사용자 특성과 아이템 특성 각각의 방면에서 제어시스템을 구현했다. 

마지막으로, 사용자 제어에 따라 추천결과를 조정하는 반 사실 대조적 추론 응답 메커니즘을 통해 필터버블을 완화하는 동시에 정확성은 유지하고 사용자가 원하는 결과를 얻을 수 있는 추천시스템을 구현했다.

![ucrs](https://i.ibb.co/7Nrkm21/ucrs.png)

### 4.1 필터버블 감지 지표
추천시스템에서 필터버블을 감지한다는 것은 다양성을 제한하고 특정 그룹 내에서 사용자를 고립시키는 개인화된 추천의 심각도를 측정하는 것을 의미한다. 추천 목록의 항목 카테고리 수를 계산하는 Coverage와 서로 다른 사용자 특성 그룹 간의 분리를 평가하는 Isolation Index와 같은 지표를 사용하여 필터버블의 심각도를 정량화할 수 있다. MCD는 아이템 특성과 관련된 필터버블을 감지하는 용도로 사용되어 가장 자주 추천되는 아이템 카테고리의 비율을 확인할 수 있다. 시간이 지남에 따라 MCD가 증가하면 아이템 카테고리와 관련된 필터 버블의 심각성이 커지고 있음을 나타낸다. 이러한 지표들을 통해 사용자에게 실시간으로 필터버블 경고를 보내고 이를 제어할지 여부를 결정하게 도울 수 있다.

### 4.2 제어 명령 기능
사용자의 과거 상호작용 데이터인 𝐷가 주어졌을 때, 기존의 추천 모델은 추천 𝑅을 예측하기 위해 𝑃(𝑅|𝐷)를 사용한다. 그러나 UCRS는 추가로 사용자 제어인 𝐶를 고려하며 사용자 개입(𝑑𝑜(𝐶))을 통해 𝑃(𝑅|𝐷, 𝑑𝑜(𝐶))를 추정할 것을 제시했다. 이때, 사용자 제어과 관련해서 UCRS는 사용자 및 항목별로 'Fine-grained controls'와 'Coarse-grained controls'를 2수준으로 나눠 총 4가지 유형의 사용자 제어를 제시했다.

#### 4.2.1 사용자 특성 제어(User-feature Controls)
N가지 사용자 특성과 사용자 𝑢가 있을 때, 사용자 𝑢는 $x_ u = [ x^1_ {u}, \ldots, x^n_ {u}, \ldots,  x^N_ {u} ], \text{ where }  x^n_ {u} \in \{0,1 \}$ 로 나타내며, 여기서 $x^n_ {u} \in \{0,1 \}$
는 사용자 𝑢가 사용자 특성 $x^n$을 가지고 있는지 여부를 나타낸다. 예를 들어, 특성 $x_ {1}$과 $x_ {2}$가 남성과 여성을 나타낸다면, $x_ {u}$ = [0, 1]은 사용자 𝑢가 여성임을 뜻한다. 

사용자 특성 제어는 'Fine-grained controls'와 'Coarse-grained controls'를 2수준으로 또 나뉘게 된다.  'Fine-grained controls'에서는 사용자가 세분화된 사용자 특성을 활용하여 구체적인 제어를 통해 다른 사용자 그룹의 관심사에 맞는 추천을 받을 수 있게 된다(예: 30대 사용자가 10대가 선호하는 동영상을 추천 받기). 

반면에 'Coarse-grained controls'를 사용하면 사용자의 기존 특성만을 제한하는 방안으로 추천을 줄이게 된다. 예를 들어, '연령=30'처럼 사용자의 나이 특성을 제거함으로써 기존 자신의 사용자 그룹에 제시되었던 추천을 줄임으로써 필터버블을 완화할 수 있다. 

#### 4.2.2 아이템 특성 제어(Item-feature Controls)
사용자 기능 제어는 사용자 특성과 관련된 필터버블을 해결하지만 사용자 상호 작용의 영향은 고려하지 않는다. 사용자 특성 제어를 보완하기 위해 아이템 특성 제어가 도입되어 아이템 특성에 따라 추천 목록을 조정할 수 있다. 이러한 제어 명령을 사용하면 액션 영화와 같은 특정 카테고리에 속하는지 여부와 같은 아이템의 특성을 고려하여 추천을 지정할 수 있게 된다.

M개의 아이템 특성과 아이템 i는 $h _{i} = [ h^1 _{i}, \ldots, h^m _{i}, \ldots,  h^M _{i} ], \text{ where }  h^n _{i} \in \{0,1 \}$ 로 나타내며, 여기서 $h^m _{i} \in \{0,1 \}$ 는 아이템i가 아이템 특성 $h^m$ 을 가지고 있는지 여부를 나타낸다.

 'Fine-grained controls'에서는 사용자가 특정 아이템 카테고리의 추천을 늘리도록 허용할 수 있다. 예를 들어 로맨스 영화와 같은 특정 카테고리의 아이템을 더 많이 받을 수 있게끔 구체적인 제어 명령을 내릴 수 있다.

 'Coarse-grained controls'에서는 사용자의 과거 상호작용에서 가장 큰 아이템 카테고리의 추천을 줄이도록 진행된다. 

종합적으로 보면,  UCRS의 'Fine-grained controls'에서는 사용자가 세분화된 사용자 혹은 아이템 특성을 활용하여 구체적인 제어 명령을 통해 특정 추천 목표를 달성하는 방안이고,  'Coarse-grained controls'는 세분화된 특성에 대해서 사용자가 구체적인 제어 명령을 지시할 필요없이 간단하게 필터버블을 완화하는 방법을 구현했다.


### 4.3 반 사실적(Counterfactual) 응답 메커니즘
'Fine-grained controls'에는 연령과 같이 변경된 사용자 특성을 기반으로 추천을 생성하여 다양한 사용자 그룹에서 사실과 다른 질문에 답하는 추론 과정이 포함된다. 

UCI는 사용자의 이전 상호작용 패턴에 반대되는 반 사실적인 조건 아래에서 새로운 추천을 생성하게 된다. 이를 통해 사용자가 과거 패턴에 제한받지 않고 원하는 추천을 받을 수 있다.


## **5. Experiment**[](https://dsailatkaist.github.io/template.html#4-experiment)

### **Experiment setup**[](https://dsailatkaist.github.io/template.html#experiment-setup)

 **Dataset**
 - DIGIX-Video
 - ML-1M
 - Amazon-Book

**Baseline**
 Factorization Machine(FM)과  Neural Factorization Machine(NFM) 2가지 추천 모델에 대해 아래와 같은 베이스라인을 적용함
- woUF: 사용자 특성 없이 학습된 포로토타입
- maskUF:  사용자 특성을 삭제하는 포로토타입
- changeUF: 사용자 특성을 변경한 포로토타입
- Fairco: 공정성 기반 ranking 포로토타입
- Diversity: 다양성 기반 re-ranking 포로토타입
- Reranking: UCI의 한 변형으로, counterfactual 추론과 target category 예측을 제외한 포로토타입
- C-UCI: 'Coarse-grained controls' UCI 포로토타입
- F-UCI: 'Fined-grained controls' UCI 포로토타입

**Evaluation Metric**
- Recall: 정확도 척도
- NDCG: 정확도 척도
- Weighted-NDCG: 카테고리 선호도 우선순위 척도
- Coverage: 다양성 척도
- Isolation Index: 격리지수 척도
- MCD: 최다 카테고리 비율 척도
- DIS-EUC: 유클리디언 거리 척도
- TCD(Target Category Domination): 목표 카테고리 비율 척도


### **Result**[](https://dsailatkaist.github.io/template.html#result)
![rs1](https://i.ibb.co/DGLCcZ2/rs1.png)


![rs2](https://i.ibb.co/kD7nSjt/rs2.png)

![rs3](https://i.ibb.co/WpSs466/rs3.png)


-   3가지 데이터 세트를 대상으로 실험한 결과, UCI 프레임워크가 사용자 제어를 기반으로 원하는 항목을 더 많이 추천하는 효과가 입증되었으며, 정확성과 다양성 측면에서 유망한 성과를 보였다.
-   UCI 프레임워크는 사용자 제어를 기반으로 더 많은 원하는 항목을 효과적으로 추천하여 사용자 만족도와 추천 생태계 참여도를 높일 수 있는 결론을 얻을 수 있다.

## **6. Conclusion**[](https://dsailatkaist.github.io/template.html#5-conclusion)
-   동 연구는 사용자가 필터버블 완화를 능동적으로 제어할 수 있도록 하는 UCRS (User-Controllable Recommender System) 를 제안하여 사용자 특성 및 과거 상호작용을 기반으로 유사한 항목을 과도하게 추천하는 문제를 해결하는 시도를 보였다.
-   UCRS 프로토타입은 필터버블의 심각도를 감지하고 사용자에게 4가지 제어 명령을 제공하여 사용자로 하여금 추천시스템의 제어권을 능동적으로 실시할 수 있도록 추천시스템 생태계에 방향을 제시했다.
-   세 가지 데이터셋을 대상으로 한 실험을 통해 UCRS 프로토타입과 UCI 프레임워크는 정확성과 다양성 측면에서 유망한 성과를 보였으며 사용자 만족도와 추천 생태계에 대한 참여도를 높일 수 있을 것으로 기대된다.
-  단, 동 연구는 UCRS 프로토타입과 UCI 프레임워크를 평가할 때 온라인 실시간 데이터 대신 오프라인 데이터셋을 사용했고, 일부 사용자가 필터버블을 완화하고 제어 기능을 제공할 의향이 있다고 가정을 했기 때문에 현실 데이터와 비교하면 strong assumption일 것으로 판단된다.

----------

## **Author Information**[](https://dsailatkaist.github.io/template.html#author-information)

-   Minkyung Choi
    -   Affiliation:
 [Human Factors and Ergonomics Lab – Human Factors and Ergonomics Lab (HFEL) (kaist.ac.kr)](http://hfel.kaist.ac.kr/)
 
    -   Research Topic:
    Data Science, Computer Vision, VR

## **Reference & Additional materials**[](https://dsailatkaist.github.io/template.html#6-reference--additional-materials)

-   Github Implementation:
https://github.com/WenjieWWJ/UCR

-   Reference:
Wang, W., Feng, F., Nie, L., & Chua, T. S. (2022, July). User-controllable recommendation against filter bubbles. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ (pp. 1251-1261).
