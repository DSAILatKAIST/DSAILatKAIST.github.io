---
title:  "[AAAI 2023] Simple and Efficient Heterogeneous Graph Neural Network"
permalink: 2023-10-16-Simple_and_Efficient_Heterogeneous_Graph_Neural_Network.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Simple and Efficient Heterogeneous Graph Neural Network

## 1. Problem Definition

Heterogeneous Graph Neural Networks(HGNN)은 기존 Graph Neural Network(GNN)에서 사용하는 attention이나 multi-layer 구조 등의 매커니즘을 그대로 사용해왔다. 하지만 homogeneous graph를 위해 디자인된 GNN에서 사용하는 매커니즘을 Heterogeneous graph에 적용했을 때, 정말 효과가 있는지에 대한 분석은 이루어지지 않았다. 본 논문에서는 이러한 매커니즘들의 효과성에 대한 분석을 바탕으로, Heterogeneous graph를 효율적으로 모델링할 수 있는 Simple and Efficient Heterogeneous Graph Neural Networks(SeHGNN)을 제안한다.

## 2. Motivation

이전의 Heterogeneous Graph Neural Network(HGNN)은 GNN에서 사용하는 메커니즘이 heterogeneous 그래프에 효과가 있는지에 대한 분석은 거의 하지 않은 채, 이를 그대로 사용하면서 heterogeneous 그래프의 representation learning을 수행해왔다. 본 논문에서는 attention이나 multi-layer 구조가 heterogeneous 그래프를 모델링하는데 효과적인지에 대해 분석하는 과정에서 중요한 두 가지 발견을 하였고, 이를 바탕으로 SeHGNN 아키텍처를 설계하였다. Heterogeneous 그래프에 대한 기존 매커니즘의 효과성에 대한 분석 과정과 그에 따른 발견은 아래와 같다.


<li> attention에 대한 연구 </li><br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/MDVHbCW/figure1.png" alt="figure1" border="0"></a>

HGNN은 Figure 1에 나타난 것처럼 서로 다른 모듈이나 파라미터들을 사용하여 계산되는 여러 attention을 사용한다. 이러한 attention들은 두 가지 유형으로 분류할 수 있는데, 첫 번째는 같은 relation의 neighbor들 사이에서 계산되는 neighbor attention이고, 두 번째는 서로 다른 relation 사이에서 계산되는 semantic attention이다. 본 논문에서는 attention의 효과성을 살펴보기 위해 attention을 사용한 경우와 사용하지 않은 경우에 대한 비교를 수행하였다.<br>
이때, attention의 사용 양상은 Heterogeneous graph를 모델링하는 유형에 따라 나뉘어 지는데, HAN과 같이 metapath 기반 방법은 neighbor aggregation 단계와 semantic fusion 단계 각각에서 두 가지 attention을 뚜렷하게 구분하여 사용한다. 반면, HGB와 같이 metapath를 사용하지 않는 방법은 relation-specific한 임베딩을 사용하여 1-hop neighbor의 attention을 계산해서, 두 가지 attention 유형을 구분하는 것이 어려울 수 있기 때문에, attention의 영향을 제거하기 위해 추가 계산을 수행해야 한다. 구체적으로 각 노드의 이웃의 attention 값을 relation별로 평균화하여 neighbor attention을 제거하거나, 각 relation 내에서 정규화하여 각 relation이 최종 결과에 동일하게 기여하도록 조정할 수 있는데 이것은 semantic attention을 제거하는 것과 같다.
 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/Ns6wjmB/table1.png" alt="table1" border="0"></a>


이를 바탕으로 HAN과 HGB에 대해 각 요소를 제거하면서 DBLP 데이터와 ACM 데이터에 대해 node classification을 수행해 실험하였고 그 결과를 Table 1에 정리하였다. 여기서 '*'는 neighbor attention를 제거하는 것을 의미하고 '†'는 semantic attention를 제거하는 것을 의미한다. Table 1의 결과에서, semantic attention이 없는 모델은 성능이 감소하는 것을 나타내는 반면, neighbor attention이 없는 모델은 그렇지 않음을 보여준다. 이를 통해 semantic attention은 HGNN에서도 필수적이며, neighbor attention은 필요하지 않다는 것을 발견했고, 추가적으로 neighbor attention의 경우 다양한 SGC(Stochastic Gradient Community)기반의 연구에서 단순 mean aggregtion이 attention 모듈을 사용한 aggregation과 동일한 효과를 가질 수 있다는 것을 확인하였다고 하면서, mean aggregation으로 대체할 수 있음을 언급한다.


<li> multi-layer 구조에 대한 연구 </li><br>

neighbor attention이 없는, metapath를 사용하지 않는 방법은 각 relation 내에서 neighbor의 feature를 먼저 평균화한 다음, 다른 relation의 결과를 fusion하는 형태를 지닌다. 따라서 이들은 multi-layer 구조를 가지고 있으며, 각 레이어에서 1-hop metapath만 사용하는 metapath 기반 방법으로 변환할 수 있다. 따라서 본 논문에서는 metapath 기반 방법에서의 레이어 수와 metapath 수의 영향에 중점을 두고 실험을 수행하였다. metapath 기반 방법인 HAN에 대한 실험을 수행하면서 각 variant의 구조를 나타내는 숫자 list를 사용하였는데, 예를 들어 ACM 데이터셋에서 (1,1,1,1)은 각 레이어에서 1-hop metapath "PA" 및 "PS"를 사용하는 네 개의 레이어 네트워크를 나타내며, (4)는 4-hop 이상의 metapath가 없는 single-layer 네트워크를 나타낸다. 이러한 list는 receptive field의 크기도 보여준다. 예를 들어 (1,1,1,1), (2,2), (4)는 동일한 receptive field의 크기를 가지며 4-hop neighbor를 포함한다. 본 논문에서는 마찬가지로 DBLP 데이터와 ACM 데이터에 대해 실험을 수행하여 Table 2에 정리하였고 이 결과를 기반으로 두 번째 발견을 도출했다.<br>

<a href="https://imgbb.com/"><img src="https://i.ibb.co/ZJp6f07/table2.png" alt="table2" border="0"></a> 

Table 2에 나타난 것처럼 single-layer 구조와 긴 metapath를 사용한 모델이, multi-layer 구조와 짧은 metapath를 사용한 모델보다 우수한 성능을 보이는 것을 볼 수 있다. single-layer와 긴 metapath를 사용한 모델은 동일한 receptive field 크기에서 더 나은 성능을 달성하는데, 이는 multi-layer 네트워크가 각 레이어마다 semantic들을 fusion하기 때문에 고수준 의미를 구별하기 어렵게 만든다는 사실로 설명할 수 있다. 예를 들어, ACM 데이터에서 network 구조로 (4)와 같은 형태를 가진 모델에서 multi-hop metapath를 사용하면, 동일한 저자로부터 쓰여진 (PAP) 또는 익숙한 저자 (PAPAP)와 같은 고수준 의미를 구별할 수 있지만, 모든 중간 벡터가 서로 다른 semantic의 혼합을 나타내므로 4개 layer 네트워크 (1,1,1,1)에서는 이러한 차이를 구분할 수 없다. 더 나아가, 최대 metapath 길이를 증가시킴으로써 모델의 성능을 향상시키는데 도움이 되며, 다양한 의미를 갖는 더 많은 metapath를 도입할 수 있다고 설명한다.<br>


위의 두 가지 발견을 바탕으로 제안된 SeHGNN에서는 각 metapath 범위에서 mean aggregation을 사용하여 모델의 성능을 희생시키지 않으면서 중복되는 neighbor attention를 피할 수 있고, single-layer 네트워크 구조를 사용하면서 단순하지만 더 긴 metapath를 사용하여 receptive field를 확장함으로써 더 나은 성능을 얻는 것을 보여준다. 더불어 attention 모듈이 없는 neighbor aggregation 부분은 linear 연산만 포함하고 학습 가능한 파라미터가 없으므로, neighbor aggregation을 매 트레이닝 에폭마다 수행하는 것이 아니라 전처리 단계에서 한 번만 실행할 수 있도록 하여 훈련 시간을 크게 줄일 수 있다. 즉, 이러한 최적화를 통해 네트워크 구조를 간소화하고 효율적으로 만드는 것이 SeHGNN의 핵심 포인트이다.

## Methodology


<a href="https://ibb.co/cYBCPdK"><img src="https://i.ibb.co/PDL9R8s/figure2.png" alt="figure2" border="0"></a><br /><a target='_blank' href='https://usefulwebtool.com/'>writing keyboard</a><br />

SeHGNN의 아키텍처는 Simplified Neighbor Aggregation과 Multi-layer Feature Projection, 그리고 Transformer-based Semantic Fusion의 세 가지 주요 요소로 구성된다. Figure 2에서 SeHGNN과 다른 metapath 기반 HGNN 간의 차이를 볼 수 있는데, SeHGNN은 <b>neighbor aggregation을 전처리 단계에서 사전 계산</b>하므로, 매 트레이닝 에폭에서 반복적인 neighbor aggregation의 과도한 복잡성을 피할 수 있다는 점이 주요한 특징이다. 각 구성 요소를 세부적으로 살펴보면 다음과 같다.

<li>Simplified Neighbor Aggregation</li><br>

간소화된 neighbor aggregation은 전처리 단계에서 단 한 번 수행되는데, 주어진 모든 metapath의 집합 $\Phi_X$에 대한 다른 semantic의 feature matrix들의 list를 아래와 같이 생성한다.

$M = \{X_P : P \in \Phi_X\}$

일반적으로 각 노드 $v_i$에 대해, 각 주어진 metapath로부터 metapath 기반 이웃의 feature를 aggregate하기 위해 mean aggregation을 사용하며 semantic feature vector들의 list를 다음과 같이 출력하는데,

$mi = \{z_P^i = \frac{1}{||SP||} \sum_{p(i,j)\in SP} x_j : P \in \Phi_X\}$

본 논문에서는 metapath 기반 neighbor collection을 간소화하기 위해 새로운 방법을 제안한다. HAN과 같은 기존의 metapath 기반 방법은 각 metapath에 대해 모든 metapath 기반 이웃을 열거하는 metapath neighbor 그래프를 구축하며, 이는 metapath의 길이에 따라 metapath 인스턴스의 수가 기하급수적으로 증가하므로 높은 부하를 초래했다. 본 논문에서는 GCN의 레이어별 전파에서 영감을 얻어 각 노드의 최종 기여 가중치를 인접 행렬의 곱셈을 사용하여 계산한다. 

$X_c = \{x_0^{cT}; x_1^{cT}; \ldots; x_{||V_c||-1}^{cT}\} \in \mathbb{R}^{||V_c|| \times d_c}$

여기서 $d_c$는 feature dimension이고, $X_c$는 $c$ 유형에 속하는 모든 노드의 초기 feature matrix를 나타낸다. 그런 다음 간소화된 neighbor aggregation 과정은 다음과 같이 표현될 수 있다.

$XP = \hat{A}_{c,c1}\hat{A}_{c1,c2}\ldots \hat{A}_{cl-1,cl}X^{cl}$

여기서 $P = c1c2 ... cl$은 $l$-hop metapath이며, $\hat{A}_{ci,ci+1}$은 노드 유형 $c_i$와 $c_{i+1}$ 간의 인접 행렬 $A_{ci,ci+1}$의 row-normalized된 형태이다.


여기에 레이블을 추가 입력으로 통합하면 모델 성능을 향상시킬 수 있다는 것을 입증한 이전 연구(Wang and Leskovec 2020; Wang et al. 2021b; Shi et al. 2021)를 활용하기 위해, raw feature들을 aggregation하는 것과 유사하게, 레이블을 one-hot 형식으로 표현하고 다양한 metapath를 통해 전파한다. 이 과정은 일련의 행렬 $\{Y_P : P \in \Phi_Y\}$ 을 생성하며, 이러한 행렬은 해당 metapath neighbor 그래프의 레이블 분포를 반영한다. metapath $P \in \Phi_Y$ 의 두 끝점은 노드 분류 작업에서 대상 노드 유형 $c$여야 한다. metapath $P = cc_1c_2 \ldots c_{l-1}c \in \Phi_Y$ 가 주어지면, 레이블 전파 과정은 다음과 같이 표현될 수 있다.

$Y^P = rm \_ diag(\hat{A}^P)Y^c, \, \hat{A}^P = \hat{A}_{c,c1}\hat{A}_{c1,c2}\ldots \hat{A}_{cl-1,c}\,$

여기서 $Y^c$는 raw label matrix이다. $Y^c$에서 training set에 속하는 노드에 해당하는 행은 one-hot 형태의 label 값을 가지며, 다른 행은 0으로 채워진다. 레이블 유출을 방지하기 위해 각 노드가 자신의 실제 레이블 정보를 받지 않도록 하기 위해 인접 행렬의 곱셈 결과에서 대각선에 있는 값을 제거한다. 레이블 전파는 neighbor aggregation 단계에서 실행되며 나중에 학습을 위한 추가 입력으로 semantic 행렬을 생성한다.


<li>Multi-layer Feature Projection</li><br>

feature projection 단계는 서로 다른 metapath의 semantic 벡터가 다른 차원을 가지거나 다양한 데이터 공간에 위치할 수 있기 때문에, 이러한 semantic 벡터를 동일한 데이터 공간으로 projection하는 과정이다. 일반적으로, 각 metapath $P$에 대한 semantic-specific한 transformation matrix $W^P$를 정의하고 ${H^′P = W^PX^P}$ 를 계산한다. 더 나은 representation을 위해, 각 metapath $P$에 대해 multi-layer perception 블록 $MLP_P$를 사용하며, 이 블록은 두 개의 연속적인 linear layer 사이에 normalization layer, non-linear layer 및 dropout layer를 포함한다. 이 과정을 다음과 같이 나타낸다.

$H'_P = \text{MLP}_P(X_P)$


<li>Transformer-based Semantic Fusion</li><br>

semantic fusion 단계는 semantic feature 벡터를 융합하고 각 노드에 대한 최종 임베딩 벡터를 생성한다. 단순한 weighted sum 형식 대신, 본 논문에서는 각 semantic 쌍 간의 상호 관계를 더 탐색하기 위해 트랜스포머 기반의 semantic fusion 모듈을 제안하였다. 트랜스포머 기반의 semantic fusion 모듈은 미리 정의된 metapath list $\Phi = \{P_1, \ldots, P_K\}$ 와 각 노드에 대한 projected된 semantic 벡터 $\{h'_{P1}, \ldots, h'_{PK}\}$ 을 고려하여 semantic 벡터 쌍 간의 상호 attention를 학습하도록 설계되었다. 각 semantic 벡터 
$h^{'Pi}$ 에 대해 이 모듈은 이 벡터를 query 벡터 $q^{Pi}$, key 벡터 $k^{Pi}$ 및 value 벡터 $v^{Pi}$로 매핑한다. 상호 attention 가중치 $\alpha(P_i, P_j)$ 는 소프트맥스 normalization 후의 query 벡터 $q^{Pi}$와 key 벡터 $k^{Pi}$의 dot product 결과이다. current semantic $P_i$의 출력 벡터 $h^{Pi}$는 모든 value 벡터 $v^{Pj}$의 weighted sum과 residual connection을 포함한다. semantic fusion 과정은 다음과 같이 표현될 수 있다.

$q^{Pi} = W_Q h'^{Pi}$ , $k^{Pi} = W_K h'^{Pi}$ , $v^{Pi} = W_V h'^{Pi}$ , $P_i \in \Phi$ <br>

$\alpha(Pi,Pj) = \frac{exp(q^{Pi} \cdot k^{{Pj}^T})}{\sum_{Pt\in\Phi} exp(q^{Pi} \cdot k^{{Pt}^T})}$ <br>

$h^{Pi} = \beta \sum_{P_j\in\Phi} \alpha(P_i,P_j) v^{P_j} + h'^{P_i}$ <br>

여기서 $W_Q$, $W_K$, $W_V$, β는 모든 metapath 간에 공유되는 학습 가능한 파라미터이다.

각 노드의 최종 임베딩 벡터는 모든 출력 벡터의 연결로 이루어지는데, node classification과 같은 downstream 작업을 위해 또 다른 MLP가 사용되어 예측 결과를 생성하며, 이는 다음과 같이 표현될 수 있다.

$Pred = \text{MLP}([h^{P1} || h^{P2} || \ldots || h^{P|\Phi|}])$


## Experiment

본 논문에서는 DBLP, ACM, IMDB 및 Freebase와 같은 널리 사용되는 heterogeneous 그래프 4개와 OGB 챌린지에서 가져온 큰 규모의 ogbn-mag 데이터셋을 사용하여 실험을 진행했고, node classification의 성능 비교를 통해 제안한 방법의 효과성을 검증했다. 

<a href="https://ibb.co/dpyvcbK"><img src="https://i.ibb.co/fN7WS80/table3.png" alt="table3" border="0"></a>

<li>Results on HGB Benchmark</li><br>
Table 3은 네 가지 데이터셋에서 SeHGNN의 성능을 HGB 벤치마크의 여러 baseline들과 비교한 결과를 제시하며, 1st 행은 네 가지 metapath 기반 방법, 2nd 행은 metapath를 사용하지 않는 네 가지 방법을 나타낸다. SeHGNN이 Freebase 데이터셋의 micro-f1을 제외하고 모든 baseline 대비 최상의 성능을 달성하였다. 

추가적으로 Motivation 파트에서 언급한 두 가지 발견을 검증하고, 다른 모듈의 중요성을 결정하기 위해 ablation study도 수행하였는데, Table 3의 4th 행은 SeHGNN에 네 가지 변형을 가한 각각의 경우에 대한 결과를 나타낸다. Variant#1은 neighbor aggregation 단계에서 HAN과 같이 각 metapath에 대해 GAT를 사용한 경우이다. Variant#2는 각 레이어가 독립적인 neighbor aggregation 및 semantic fusion 단계를 갖는 두 개의 레이어 구조를 사용하며, 각 레이어의 metapath의 최대 hop이 SeHGNN의 절반으로 놓고 SeHGNN과 Variant#2가 동일한 수용 영역 크기를 갖도록 한 경우이다. SeHGNN과 Variant#1과 Variant#2 사이의 성능 차이를 통해 Motivation에서 언급한 두 가지 발견에 대한 내용이 SeHGNN에도 적용된다는 것을 확인할 수 있다. Variant#3는 추가 입력으로 레이블을 포함하지 않는 경우이고, Variant#4는 HAN과 같이 weighted sum fusion으로 트랜스포머 기반의 semantic fusion을 대체한 경우이다. 특히, SeHGNN에 뒤쳐지지만, Variant#3은 Freebase 데이터셋의 micro-f1을 제외한 대부분의 baseline들보다 우수한 성능을 보여준다. 이러한 결과는 레이블 전파와 트랜스포머 기반 fusion의 활용이 모델 성능을 향상시킨다는 것을 입증하는 증거로 볼 수 있다.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/TmM0Ldh/table4.png" alt="table4" border="0"></a>

<li>Results on  Ogbn-mag</li><br>

본 논문에서는 다섯 번째 데이터셋으로 ogbn-mag을 사용하여 성능을 비교하였다. ogbn-mag 데이터셋은 일부 유형의 노드의 초기 feature가 부족하고, target type 노드가 연도에 따라 분할되어 training 노드와 test 노드가 다른 데이터 분포를 갖게 된다는 문제점을 갖고 있다. 기존의 다른 방법들은 일반적으로 이러한 어려움을 다루기 위해 ComplEx (Trouillon et al. 2016)와 같은 비지도 표현 학습 알고리즘을 사용하여 추가 임베딩을 생성하고, multi-stage learning을 활용하여 마지막 학습 단계에서 확신 있는 예측을 가진 test 노드를 선택하고 이러한 노드를 training set에 추가하여 새로운 단계에서 모델을 다시 훈련한다고 한다(Li, Han, and Wu 2018; Sun, Lin, and Zhu 2020; Yang et al. 2021). 본 논문의 저자는 이러한 방법들을 사용하거나 사용하지 않는 결과를 이용하여 비교하였다. 추가 임베딩이 없는 방법의 경우 무작위로 초기화된 초기 feature 벡터를 사용했다고 한다.

Table 4는 대규모 데이터셋 ogbn-mag에 대해 baseline과 비교한 결과를 보여준다. 결과는 SeHGNN이 동일한 조건에서 다른 방법을 능가한다. 무작위로 초기화된 특징을 가진 SeHGNN이 추가 표현 학습 알고리즘에서 잘 훈련된 임베딩을 가진 다른 방법보다 우수한 성능을 보이는데, 이는 SeHGNN이 그래프 구조로부터 더 많은 정보를 학습한다는 것을 보여주는 증거이다.

<li>Time Analysis</li><br>
본 논문에서는 또한 실행 시간에 대한 비교 분석도 수행하였다.


<a href="https://imgbb.com/"><img src="https://i.ibb.co/f8wwp0k/table5.png" alt="table5" border="0"></a>



먼저, Table 5에서 보여지듯이 SeHGNN의 시간 복잡도를 HAN과 HGB와 비교하는 이론적 분석을 수행한다. SeHGNN과 HAN은 k개의 metapath와 single-layer 구조를 가정하고, HGB는 $l$개의 레이어 구조를 갖는 것으로 가정하여 분석한다. metapath의 최대 hop도 $l$로 설정하여 receptive field의 크기를 동일하게 유지한다. 대상 유형 노드의 수는 n이며 입력 및 hidden 벡터의 차원은 $d$이다. HAN에서 metapath neighbor 그래프의 평균 neighbor 수는 $e_1$이고, HGB에서 multi-layer aggregation 중에 관련된 neighbor 수는 $e_2$ 이다. $e_1$ 과 $e_2$는 metapath의 길이와 레이어 수인 $l$과 함께 지수적으로 증가한다. 위 다섯 개의 데이터셋에서 본 논문은 최대 metapath 수십 개를 사용하지만, 레이어 $l$ ≥ 3에 대해 각 노드는 평균 수천 개의 neighbor들로부터 정보를 aggregation한다. 일반적으로 $e_1$ ≫ $k^2$, $e_2$ ≫ $k^2$이다. 따라서 SeHGNN의 이론적 복잡성은 HAN과 HGB보다 훨씬 낮은 것을 확인할 수 있다.


<a href="https://imgbb.com/"><img src="https://i.ibb.co/9GcRN9c/figure3.png" alt="figure3" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>image hosting without registration</a><br />


이론적 분석을 검증하기 위해 SeHGNN의 시간을 이전에 나온 HGNN들과 비교하는 실험을 수행하였고, Figure 3은 각 모델의 평균 시간 단위로 학습 시간에 따른 micro-f1 점수 달성 정도를 보여준다. 이는 SeHGNN이 학습 속도와 모델 성능 모두에서 우수함을 나타낸다.


## Conclusion

본 논문은 heterogeneouos 그래프 representation learning을 위한 SeHGNN이라는 새로운 방법을 제안한다. 이 방법은 attention 사용 여부와 네트워크 구조에 따른 효과성에 대한 두 가지 주요 발견을 기반으로 제안되었다. 본 논문에서는 light mean aggregation을 사용하여 neighbor aggregation을 사전에 계산함으로써 구조 정보를 효과적으로 포착하면서, neighbor attention이 과도하게 사용되는 것을 방지하고 반복적인 neighbor aggregation도 피할 수 있도록 하였다. 이와 함께 receptive field를 확장하고 semantic 정보를 더 잘 활용하기 위해 긴 metapath를 사용하는 single-layer 구조와, 트랜스포머 기반의 semantic fusion 모듈을 사용하여 모델의 효과성을 향상시켰다. 













