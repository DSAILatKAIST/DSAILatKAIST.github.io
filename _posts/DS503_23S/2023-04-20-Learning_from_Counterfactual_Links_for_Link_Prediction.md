---
title:  "[ICML 2022] Learning from Counterfactual Links for Link Prediction"
permalink: Learning_from_Counterfactual_Links_for_Link_Prediction.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [ICML 2022] [Learning from Counterfactual Links for Link Prediction](https://arxiv.org/pdf/2106.02172.pdf)

## **1. Introduction**
Link Prediction Task는 관측된 그래프에 기반하여 node pair간의 edge가 존재할 likelihood를 예측하고자 하는 것이 목표이다. 이러한 task는 여러 분야에 적용해볼 수 있는데, 대표적으로 영화 추천과 Chemical Interaction Prediction, Knowledge Graph Completion 등이 있다.
이를 해결하기 위해 Graph Machine Learning을 도입하여 nodes의 representation을 학습하고, pair of nodes의 representations간의 association을 학습함으로써 Link의 존재 확률에 대해 추론해오는 방식이 최근 흐름이었다. 

하지만 이렇게 association만을 학습할 경우 중요한 요소를 캡처할 만하지 않아 test data에 대한 예측에 실패하는 경우가 발생할 수 있다. 이를 알아보기 위해 Social Network를 예시로 들어 설명해보자면, Adam과 Alice는 같은 neighborhood에 살고 있고, 그들은 가까운 친구 관계이다 ( 친구 관계를 Link로 볼 수 있다 ). 하지만 같은 neighborhood와 친구 관계 간의 association에 너무 집중하면, 공통 관심사나 가족 관계와 같이 친구 관계에 영향을 줄 수 있는 다양한 factors를 놓칠 수 있다. 이렇기에 단순한 association을 뛰어넘어 좀 더 정교한 예측을 하기 위해서 이들은 다음과 같은 __Counterfactual Question__ ( 이에 대한 개념은 Background에서 자세히 설명할 예정이다 )에 대한 대답을 하고자 했다.

> __“만약 Alice와 Adam이 같은 neighborhood에 살지 않았다고 해도, 여전히 그들은 가까운 친구일까?”__

 즉, 이러한 질문에 graph learning model이 대답해봄으로써 causal relationship을 학습할 수 있다면, link prediction의 성능을 높일 수 있을 것이라는 점이 논문의 motivation이라고 할 수 있다. 이러한 motivation을 활용하여 CFLP라는 framework를 제안하였다.

---

## **2. Background**
Key Idea를 이해하기 위해서, Causal Inference와 이 논문의 전신이라고 볼 수 있는 [Learning Representations for Counterfactual Inference](http://proceedings.mlr.press/v48/johansson16.pdf)을 간략히 알아볼 필요성이 있다.

### __Causal Inference__
먼저 Causal이란, 확률분포를 생성하는 함수적 구조로써, 데이터 생성 과정에서 불변이라고 간주하는 것이다 ([참고자료](https://horizon.kias.re.kr/15780/)). 따라서 확률 변수 $X$가 원인, $Y$가 그에 따른 결과라고 할 때, $X$가 변하면 $Y$의 확률분포가 변하겠지만, $Y$를 변화시킨다고 해서 $X$가 변하지는 않는 구조를 생각해볼 수 있다. 이에 대해 좀 더 쉽게 이해해보기 위해서, 통계 공부를 하면서 들어 봤을 상관관계와 인과관계에 대해 예시를 통해서 간략히 알아보자. 

![causality](https://user-images.githubusercontent.com/74266785/232045122-75317f6a-72f3-4a85-b015-1ebdbf183427.PNG)

위 그림은 아이스크림의 판매량($X$)과 상어가 사람을 공격한 횟수($Y$)를 동시에 나타낸 그래프이다. 이러한 그래프를 보면, 아이스크림의 판매량과 상어가 사람을 공격한 횟수 간 양의 상관관계가 있음을 파악해볼 수 있는데, 이들 간에 인과관계 또한 있다고 볼 수 있을까? 심층적인 이해를 위해 다음 질문을 고려해보자.

> - 아이스크림 판매를 중지하면 ($X$를 감소시키면) 상어의 공격횟수가 줄어들까 ($Y$가 감소할까)?
> - 상어의 공격횟수가 증가하면 ($Y$를 증가시키면), 아이스크림의 판매량이 증가할까 ($X$가 증가할까)?

“상식” 선에서 이러한 질문은 거짓임을 알 수 있고, 따라서 유의미한 인과관계는 없다고 볼 수 있다. 이를 좀 더 깊게 살펴보자면, 기온(Z)라는 제3의 변수가 각각에 영향을 미치게 되어 허위 상관(Spurious Correlation)을 갖게 되었다고 표현한다. 

다음으로는 이 논문의 제목에서도 등장하는 Counterfactual의 개념에 대해 간단히 알아보고자 한다. Counterfactual은 한국어로 해석해보면 반사실인데, 말 그대로 현실에서 일어나지 않은 사실에 대한 논의를 말하고자 하는 개념이다. 예시를 들어 알아보자면, 집을 가야하는 상황에서 갈림길이 주어졌는데, 하나는 고속도로이고, 다른 하나는 국도로 가는 길이 있다. 이때 국도를 선택하였고, 결과적으로는 길이 막혀서 2시간이라는 시간이 걸리게 되었는데, 집에 생각해보면서 “만약 고속도로를 탔었다면, 더 빨리 도착했었을 텐데” 와 같은 후회를 해볼 수 있는 것이다. 좀 더 Formal하게 말해보자면 Counterfactual은 정확히 동일한 조건에서 하나의 측면(선행 사건)만 달랐을 경우 ( Causal Inference Framework 하에서는 이러한 작업을 Intervention이라고 한다 ), 두 결과가 얼마나 다를지에 대한 고찰이라고 볼 수 있다. 즉 이 상황에서는 국도를 타고 집에 올 때까지 걸린 시간( 실제 발생한 사건 )과 그 당시에 고속도로를 타고 집에 올 때까지 걸린 시간 ( 반사실 ) 간의 차이를 고찰해 봄으로써, 도로가 나의 귀가 시간에 실질적으로 영향을 끼칠 수 있었는지 회고적으로 생각해보자는 철학이 담겨 있다. 이에 대한 좀 더 자세하고 수학적인 논의는 [Casual Inference in Statistics: A Primer]( https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf)의 4단원을 참고하거나 Pearl’s Ladder of Causation을 검색하여 추가적으로 학습해보기를 바란다.

앞선 Introduction에서 말했던 Counterfactual Question을 돌이켜보면, neighborhood가 하나의 측면(선행사건)이라고 볼 수 있고 이것이 달랐을 경우에 결과가 어떻게 될지 알아보고자 한다는 의미가 된다.

Causal Inference에 대한 자세한 설명의 경우, [Casual Inference in Statistics: A Primer]( https://www.datascienceassn.org/sites/default/files/CAUSAL%20INFERENCE%20IN%20STATISTICS.pdf)을 참고하기를 바란다.

### __[ICML'16] Learning Representations for Counterfactual Inference__
이 논문의 경우 Machine Learning의 관점에서 Counterfactual Question에 대해 답하고자 한 것이 Motivation이라고 볼 수 있다.

Counterfactual Question을 구성하는 요소는 총 3가지가 있는데, $T$는 potential intervention(treatment)의 집합, $X$는 contexts의 집합, $Y$는 potential outcome이다. 간단한 예시를 들자면 환자 $x \in X$에 대해 생각해볼 수 있는 intervention으로는 환자에게 주어지는 treatments(여기서는 치료의 의미를 가진다)를 고려해볼 수 있고, 각 intervention에 대응되는 potential outcome으로 혈당량 $Y_ t(x)$를 생각해볼 수 있다. 여기서 중요한 사실은
> 주어진 context(여기서는 환자)에 대해 현실에서는 단 하나의 potential outcome만 관찰할 수 있다.

이다. 즉, 실제로 관측된 것이 아닌 counterfactual(potential outcome)에 대해서는 적절한 값을 추론해야 하는 것이다. 이러한 조건 속에서 관심 포인트는 바로 각 context $x$에 대해 Individualized Treatment Effect (ITE)를 계산해보고 싶은 것이다. 이 값의 의미를 알아보기 위해 우선 Binary한 intervention(여기서 부터는 treatment라는 용어로 다시 설명하겠다) set  $T = \left\lbrace 0,1\right\rbrace$ ( 통상적으로 값이 1일 경우에는 "treated", 0일 경우에는 "control"라고 표현한다. ) 을 가정하자. 이 경우 $ITE = Y_ {1}(x)-Y_ {0}(x)$으로 정의 되는데, 즉 ITE는 한 context에 대해서 treatment에 따른 potential outcome의 값의 차이를 의미한다. 이때 한 outcome밖에 알 수 없다는 것을 알고 있으므로, 자연스럽게 ITE를 추정해야 함을 알 수 있다. 이를 위한 기본적인 approach는 direct modeling으로, 
> 주어진 $n$ samples $\left\lbrace(x_ i,t_ i,y_ i^F)\right\rbrace_ {i=1}^n$ where $ y_ i^F = t_ iY_ 1(x_ i) + (1-t_ i)Y_ 0(x_ i)$ (Factual Outcome) 에 대해 함수 $ h : X \times T \rightarrow Y$ s.t $ h(x_ i, t_ i) \approx y_ i^F$를 잘 학습하자.

이렇게 될 경우, estimated ITE는 다음과 같다.

![ITE Estimation](https://user-images.githubusercontent.com/74266785/232067724-d22f156d-9597-4050-8172-396e61177285.PNG)

이 논문에서는 이러한 direct modeling을 응용하여 Context x를 잘 표현하는 representation $\Phi$와 Outcome을 예측하는 함수 $h$를 잘 학습하기 위해

> 1. <span style="color:red">관측된 factuals에 대한 low-error prediction 달성
> 2. <span style="color:blue">관측되지 않은 counterfactuals에 대한 low-error prediction 달성
> 3. <span style="color:green">treated와 control populations간의 distribution을 비슷하게 만들어주기 ( __Balancing__ 이라고도 표현한다 )

의 목표를 이루고자 하고, 다음과 같은 Objective를 감소시키는 방향으로써 최적의 parameters를 찾고자 한다.

![prev_loss_colored](https://user-images.githubusercontent.com/74266785/232071334-cea8d083-f1d5-4603-ac8a-850fafbcdf4f.PNG)

(1)과 (2)에서는 기존 ML에서 Representation Learning과 Regression Task를 Empirical Risk Minimization를 바탕으로 진행하는 것과 유사한데, (2)의 경우는 살짝 다른 것이 관측되지 않은 counterfactual을 estimation하기 위해 supervision signal로 주고자 관측된 context중에서 가장 비슷한 context의 outcome을 surrogate value로 차용한다는 점이다. 수식적으로 나타내면 context $x_i$에 대해 Metric space 상에서 적절한 metric $d$에 대해 $j(i) \in \text{argmin}_ {j=1,...,n \;\text{s. t}\; t_ j=1-t_ i}d(x_j, x_i)$를 생각해봄으로써 계산해볼 수 있다. 이러한 점이 Counterfactual Inference를 실질적으로 계산하기 위해 진행해주는 중요한 작업이라고 볼 수 있다. (3)에서 population간의 distribution을 유사하게 해준다는 것은 factual로부터 counterfactual을 일반화할 때 data의 불확실한 정보로부터 학습하는 것을 방지해주고자 하는 목적이 있다. 예를 들어 관측된 sample의 gender feature에서 특정 treatment A에 대해 남자에 대한 정보가 없다면, 남자가 그러한 treatment A에 어떻게 반응할 것인지의 Counterfactual에 대한 예측은 조심스럽게 할 필요성이 있다는 것이다. 따라서 Randomized Controlled Trials에서 selection bias를 없애기 위해 treatment 집단과 control 집단의 특성을 Randomized Assignment를 통해 비슷하게 만들어주는 것과 유사한 작업을 해준다고 볼 수 있다. 

이러한 과정을 거쳐서 Counterfactual Question에 대한 답을 머신러닝의 관점에서 해볼 수 있고, 만약 이 논문에 대해 더 자세한 설명과 뒷받침되는 이론을 접해보고 싶다면, 직접 [Learning Representations for Counterfactual Inference](http://proceedings.mlr.press/v48/johansson16.pdf)를 참고하기를 바란다.

---

## **3. Problem Definition**

### __Link Prediction Formulation__
기본적인 Notation의 경우, 대다수의 Graph ML에서 차용하는 표준적인 Notation으로써, 아래와 같다.

![Notation](https://user-images.githubusercontent.com/74266785/232073611-cfd538c3-89bd-402d-a98e-5005476eced5.PNG)

Link Prediction의 공통적인 concept은 주어진 observed graph $G$에서 모든 pairs of nodes의 link existence를 예측하는 것이다. 즉, True $A$와 유사한 $\hat{A}$를 만들고자 하는 것인데, 그래프 머신러닝의 관점에서 생각해보면, 결국 test data에서 link existence를 예측하기 위한 node representation $z$를 잘 학습하는 것으로 귀결된다.

---

## **4.Proposed Method**

### __Leveraging Causal Model__
Learning Representations for Counterfactual Inference 섹션에서 설명했던 내용을 Link prediction task에 적용해보면 된다. $A$는 observed adjacency matrix로, 관측된 factual outcome을 의미하고 $A^{CF}$는 treatment가 다를 때의 counterfactual link 정보를 담고 있는 unobserved matrix로써, counterfactual outcome을 의미한다. 이때 $T\in\left\lbrace 0,1\right\rbrace^{N\times N}$을 binary factual treatment matrix로 정의하는데, 이때 $T_ {i,j}$는 node pair $(v_ i,v_ j)$의 treatment를 의미한다. $T^{CF}$는 counterfactual treatment matrix로써 Binary treatment 상황이기에, $T_ {i,j}^{CF}=1-T_ {i,j}$로 정의된다.

여기서 제안하는 방법론의 목표는 ITE를 잘 찾고자 하는 고전적인 Causal Inference와는 달리 ITE 값을 이용하여, edge existence를 잘 예측할 수 있는 node의 representation을 잘 학습하고자 하는 것이다. 이러한 차이점은 아래의 figure에서 확인해볼 수 있다.

![Figure 2](https://user-images.githubusercontent.com/74266785/232075721-d613a290-f394-4334-a0db-b8f00f146d4d.PNG)

### __Treatment Variable__

Motivation에서도 neighborhood과 같은 graph structural information이 너무 강력하여 link existence 예측에 essential한 다른 factor를 찾는 것에 어려움을 줄 수 있다고 언급하였는데, 이는 곧 link prediction performance의 sub-optimal한 결과를 내놓을 수 있다는 의미이다. 따라서 저자는 이러한 information을 treatment로써 고려하였다. treatment의 후보로써 일반성을 잃지 않기 위해 unsupervised approach를 이용하였는데, graph clustering/mining method를 통해 두 node가 같은 cluster로 할당이 되면 $T_ {i,j}=1$ 아니면, $T_ {i,j}=0$이 되는 형태로 각 context에 treatment를 할당하였다.

### __Counterfactual Link__

실제로 한 context에 대해서 하나의 treatment에 대한 값만 관측할 수밖에 없기 때문에, [Learning Representations for Counterfactual Inference](http://proceedings.mlr.press/v48/johansson16.pdf)에서 했듯이, nearest neighbor를 통해 1) treatment가 다르면서 2) 가장 비슷한 node pair를 찾아 그 outcome을 counterfactual link로 정의한다. 여기서 efficient한 비교를 위해서 unsupervised graph representation learning method [MVGRL](https://arxiv.org/pdf/2006.05582.pdf)와 margin을 도입하여 node-level embedding space 상에서 비교를 진행한다. 따라서 모든 node pair에 대해 counterfactual link를 아래와 같이 정의하여 계산할 수 있다.

![equation3](https://user-images.githubusercontent.com/74266785/232079126-6360c11a-1c99-4d6c-a683-1b8a639ebdb6.PNG)

이때 $\gamma$라는 hyperparameter가 존재하여 조건을 만족하지 못하는 pair가 발생할 수 있는데, 이 경우 counterfactual link를 찾을 수 없다고 판단하여 아래와 같이 counterfactual treatment matrix $T_ {i,j}^{CF}$와 adjacency matrix $A_ {i,j}^{CF}$를 정의한다.

![T, A def](https://user-images.githubusercontent.com/74266785/232079616-2a31a0a7-0a6c-4f17-9aea-dd98623b5073.PNG)

### __Learning from Counterfactual Distributions__

$P^F$를 observed contexts와 treatment의 factual distribution, $P^{CF}$를 observed context와 반대되는 treatment로 구성된 counterfactual distribution이라고 하자. 

이에 대응되는 empirical factual distribution $\hat{P}^F=\left\lbrace (v_i,v_j,T_ {i,j} ) \right\rbrace_ {i,j=1}^N \sim P^F$, empirical counterfactual distribution $\hat{P}^{CF}=\left\lbrace (v_i,v_j,T_ {i,j}^{CF} ) \right\rbrace_ {i,j=1}^N \sim P^{CF}$로 저자들은 정의하였다. 이러한 정의는 이 논문의 방법론이 traditional link prediction method와는 달리 counterfactual outcome또한 이용함과 동시에 후술할 population간의 discrepancy를 설명하기 위해 등장했다고 보인다.

### __Learning from Counterfactual Links (Training Framework)__

최종적으로 Model에 들어가는 Input은 (1) observed adjacency matrix $A$와 raw feature $X$, (2) factual과 Counterfactual treatment matrix $T^F$, $T^{CF}$, (3) counterfactual link data $A^{CF}$이다. Output으로는 prediction logit으로 $\hat{A}$과 $\hat{A}^{CF}$를 얻게 된다.

Model의 경우 Graph Encoder와 link decoder로 구성되며, GCN과 같은 graph encoder를 통해 얻은 node representation ($z_i, z_j$)을 Hadamard product로 합쳐준 후, Treatment값과 함께 MLP에 태워 logit 값을 얻게 된다. 이러한 과정은 아래의 그림과 같다.

![a_hat](https://user-images.githubusercontent.com/74266785/232082645-3c2b9920-e42f-4c35-b53e-6565e254d04e.PNG)

Loss의 경우, [Learning Representations for Counterfactual Inference](http://proceedings.mlr.press/v48/johansson16.pdf)에서처럼 factual과 counterfactual에 대한 값을 잘 예측하기 위해 classification에서 자주 쓰이는 Cross-Entropy Loss를 이용한다.

![Loss](https://user-images.githubusercontent.com/74266785/232083066-db9d2141-7064-4f90-ae94-85fb915739af.PNG)

여기서 그치는 것이 아니라, treated와 control population간의 discrepancy를 줄이기 위한 term을 하나 더 설정한다. 이에 대해 자세히 설명해보자면, Counterfactual Learning을 진행함에 있어서 생길 수 있는 문제로, inference 단계에서 Covariant Shift의 형태로 training data와 test data distribution 간의 gap이 생길 수 있다. 따라서 이를 위해 discrepancy distance를 도입하여 regularization의 형태로써 두 distribution간의 gap을 줄이고자 한다. 이때 실질적인 계산을 위해서, 대응되는 learned representation간의 distance를 줄이는 방식으로 진행되고 이에 대한 Loss는 아래와 같다.

![L_disc](https://user-images.githubusercontent.com/74266785/232083856-545957cd-0dbb-4120-a2f9-e806c0413634.PNG)

따라서 최종 Loss는 아래와 같다.

![Total Loss](https://user-images.githubusercontent.com/74266785/232084088-4009a0f7-25af-4880-b1ba-4f2d630fc8d7.PNG)

실질적인 Implementation위해 필요한 Pseudo Code는 아래와 같다.

![CFLP](https://user-images.githubusercontent.com/74266785/232084288-681da1a5-87bb-4f99-86c4-44851f077e3f.PNG)

여기서 decoder fine-tuning을 따로 진행하는 이유는, discrepancy regularization term인 $L_ {disc}$의 경우 graph encoder $f$에 의해 학습된 node pair의 representations으로부터 계산되는데, 이러면 decoder g는 두 empirical distribution 간의 discrepancy가 충분히 해소되지 않은 상황에서의 node representations으로 학습하기 때문이다. 따라서 이를 보완하기 위해 early stage에서 좋은 quality의 node representations를 학습하게 한 후, encoder는 freeze하고 factual data를 통해 g를 fine-tuning 하게 된다.

끝으로, 이들의 framework를 아래 한 장의 그림으로 요약해볼 수 있다.

![Summary](https://user-images.githubusercontent.com/74266785/232084877-264a63a6-c3c8-4b5b-985f-c79b6a0976b1.PNG)

---

## **5. Experiment**

### __Setup__

위 논문에서는 총 다섯 개의 Benchmark datasets을 이용하는데, citation network ( CORA, CITESEER, PUBMED), social network ( FACEBOOK ), drug-drug interaction network (OGB-DDI)이다. Treatment Variable로는 K-core를 기본값으로 사용하고 CFLP의 encoder로는 GCN, GSAGE, JKNet을 이용하였다.

### __Link Prediction__

Link prediction의 performance를 비교하기 위해 선택한 baseline으로는 Node2Vec, MVGRL, VGAE, SEAL, LGLP, GNNs with MLP decoder를 이용하였다. 이에 대한 Metric으로 Hits@20과 ROC-AUC를 이용한다. 여기서 Hits@20은 top-20 개에서 실제로 올바른 link를 의미하는 것으로써, precision과 동등하다고 볼 수 있다. 실험 결과는 아래와 같다.

![Table2](https://user-images.githubusercontent.com/74266785/232090170-70524e0a-1c3d-4eae-aad2-fbcb7b397ca6.PNG)
![Table3](https://user-images.githubusercontent.com/74266785/232090181-b94fb47a-3d86-41dc-92df-10d9739f2171.PNG)

결과를 해석해보자면, Hits@20에서는 JKNet으로 CFLP 방식을 이용하였을 때 5개의 dataset에 대하여 좋은 성능을 내었음을 알 수 있고, AUC를 측정하였을 때에도 좋은 성능을 내었음을 알 수 있다. 결과에 대한 해석을 저자가 많이 서술해놓지 않아 개인적으로 아쉬웠던 부분이다. 대부분의 Dataset에서 좋은 성능을 낼 수 있었던 원인으로는, 다른 Method들과는 달리 Counterfactual Links(Treatement Variable은 다르면서, 나와 가장 유사한 Link) 에 대한 개념을 고려하여 좀 더 풍부한 학습을 진행할 수 있었기 때문에, Test dataset에 대해 추론하는 과정에서 더 좋은 성능을 낼 수 있었다고 본다.

### __ATE with Different Treatments__

독자의 경우 Setup에서 Treatment를 정하는 기준이 궁금했을 것이다. 이 섹션을 보면 궁금증을 해소할 수 있을 것이다.

저자들은 Counterfactual link를 만들 때 필요한 Treatment Variable을 선정하는 방식을 제안하기 위해  2가지의 질문에 답하고자 한다. 

> Q1. CFLP가 observed averaged treatment effect (ATE)를 충분히 학습할 수 있는가?\
> Q2. estimated ATE와 prediction performance와는 어떤 관계가 있는가?

( 여기서 ATE는 각 context에 대해 ITE값의 평균을 취해주는 것으로써, $E_ {z \sim Z}\text{ITE}(z)$으로 정의되고, 큰 값일수록 Treatment와 Outcome이 강한 causal relation을 갖고 있음을 의미한다. )
만약 Q1의 답이 맞다면, Q2에 대한 대답을 통해 관측된 ATE를 바탕으로 어떻게 treatment를 선정하는지 알 수 있다. 이를 위해 Q1에 대한 답을 하기 위해서는 observed ATE와 estimated ATE 간의 비교를 진행해야 한다. 이를 위해 아래의 수식을 바탕으로 각각이 정의된다. ( 정의의 배경을 이해하고 싶다면 [ICML 16'] Learning Representations for Counterfactual Inference 섹션의 estimated ITE를 생각해보자 )

![ATE Def](https://user-images.githubusercontent.com/74266785/232092671-35073238-7710-4c45-afbf-1cae692754d8.PNG)

사용할 Treatment Variables의 후보군으로는 graph clustering이나 community detection method로써, K-core, Stochastic block model (SBM), spectral clustering (SpecC), propagation clustering (PropC), Louvain, common neighbors (CommN), Katz index, hierarchical clustering이다. Encoder로는 공통적으로 JKNet을 이용하였으며, 실험 결과는 아래와 같다.

![Table4](https://user-images.githubusercontent.com/74266785/232093427-7f87bebd-ca90-4588-bfa0-91ab6bc72cdb.PNG)
![Table5](https://user-images.githubusercontent.com/74266785/232093439-d250e83c-84a1-4ffa-99db-f7b98bad8863.PNG)

여기서 저자들이 관찰한 것은 첫 번째로 observed ATE와 estimated ATE의 ranking이 positively correlate 되어있음을 Kendell’s ranking coefficient로 계산하여 확인하였다는 것이다. 여기서 Kendell’s ranking coefficient는 비모수 통계적 검정기법 중 하나로, 두 변수의 순위 상관관계를 측정하고자 할 때 이용할 수 있다. 1에 가까울 수록 순위 상관관계의 경향성이 높다고 할 수 있고, 이에 대한 더 자세한 설명은 [Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)을 참고하기를 바란다. CORA와 CITESEER에서 각각 0.67과 0.57로서 두 ATE간의 순위가 양의 상관관계를 가지고 있고, 이를 바탕으로 CFLP가 causal relationship을 잘 학습해낼 수 있다고 결론지었다. 이는 Q1에 대한 대답이라고 볼 수 있다. 

두 번째로는 두 ATE값 모두 link prediction의 성능과 음의 상관관계를 가짐을 알 수 있었는데 이를 바탕으로 그들은 CFLP로 모델을 훈련시키기 전에 (좋은 성능을 낼 수 있는) 적절한 treatment를 고를 수 있다고 주장하였다. 이때 ATE의 값이 낮으면 weak causal relation을 의미하게 되는데, 이렇게 outcome과 가장 약한 causal relationship을 갖는 treatment를 이용하고자 하는 것은 모델이 좀 더 outcome에 essential한 factor를 학습할 수 있도록 도와준다고 해석해볼 수 있다. 이러한 결론을 Q2에 대한 대답으로써 이해해볼 수 있다.

개인적으로 Q1의 경우 어떻게 답할지 궁금했었는데, 간단한 비모수 검정 기법을 이용한 점이 인상적이었다.

---

## **6. Conclusion**

위 논문은 counterfactual link의 novel concept을 제안하고, causal inference의 framework를 활용하여 link prediction의 성능을 끌어올렸다. [Learning Representations for Counterfactual Inference](http://proceedings.mlr.press/v48/johansson16.pdf)과 Causal Inference에 이론적 기반을 두고, 이를 link prediction의 task에 적용하였는데, global graph structure와 link existence 간의 causal relation을 알아내고자 하는 간단한 아이디어로 좋은 성능을 낼 수 있었다는 점이 인상적이었다.

---

## **7. Future Work**
해볼 수 있는 Future Work로는 단순히 이 개념을 Link에만 국한할 것이 아니라, counterfactual graph의 개념을 도입하여 graph learning에 이용해볼 수 있을 것으로 생각된다. 물론 이를 위해서는 Link의 유사도를 측정했던 방식과는 달리 Graph간의 유사성을 측정할 수 있는 graph kernel과 같은 Metric을 이용해야할 것이다. 이를 통해 counterfactual graph를 training에서 true와 같이 살펴본다면, 좀 더 좋은 representation learning을 할 수 있을 것으로 기대된다. 또한 Node나 Edge type이 다를 수 있는 경우, 즉 Heterogeneous graph에도 적용해볼 수 있다면 좋은 future work가 될 수 있을 것이라고 생각한다. 응용 방안으로는 추천시스템의 데이터를 user-item의 graph 형태로 보고, user-item link에 대한 Counterfactual link를 얻어내어 학습에 이용해봄으로써 좀 더 정확한 추천 성능을 얻어낼 수 있을 것이라고 생각한다.

---

## **Author**

__이준모 ( Junmo Lee )__

- Affiliation : KAIST ISysE
- Research Topic : Causal Inference, Interpretable Machine Learning

---

## **Reference & Implementation**

1. Johansson, F., Shalit, U., & Sontag, D. (2016, June). Learning representations for counterfactual inference. In International conference on machine learning (pp. 3020-3029). PMLR.
2. 수학도가 인공지능 연구에 기여하는 방법, 임성빈, HORIZON, 2020
3. Glymour, M., Pearl, J., & Jewell, N. P. (2016). Causal inference in statistics: A primer. John Wiley & Sons.

- [Implementation](https://github.com/DM2-ND/CFLP)