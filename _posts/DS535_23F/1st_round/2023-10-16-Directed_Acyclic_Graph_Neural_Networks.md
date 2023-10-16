---
title:  "[ICLR 2021] Directed Acyclic Graph Neural Networks"
permalink: 2023-10-16-Directed_Acyclic_Graph_Neural_Networks.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [ICLR 2021]DIRECTED ACYCLIC GRAPH NEURAL NETWORKS

# 1. INTRODUCTION

그래프 구조화된 데이터는 다양한 분야에서 흔히 발견되며, 그래프 신경망(Graph Neural Networks, GNNs)은 그래프 구조와 노드 특성을 모두 사용하여 벡터 형태의 표현을 생성합니다. 이러한 표현은 분류, 회귀 및 그래프 디코딩에 사용될 수 있습니다. 잘 알려져 있는 GNNs는 이웃 노드 간의 반복적인 메시지 전달을 통해 노드 표현을 업데이트하고 그래프 representaion을 생성합니다. Relational inductive 편향(neighborhood aggregation)은 GNNs이 graph-agnostic neural networks을 능가할 수 있도록 돕습니다. 본 논문의 내용 이해를 돕기 위해  메시지 전달 신경망(Message-Passing Neural Network, MPNN) 구조를 fomalize하겠습니다, 이 구조는 모든 레이어 $l$에서 그래프 $\mathcal{G}$의 모든 노드 $v$에 대한 표현 $h_v^l$를 계산하고 최종 그래프 representaion $h_{\mathcal{G}}$를 계산합니다.

$
h_v^l = \text{COMBINE}^{l} h^{(l-1)}_v, \text{AGGREGATE}^l( \{ h^{(l-1)}_u|u \in N(v) \}), \quad l = 1,.., L, \quad (1)
$

$
h_{\mathcal{G}} = \text{READOUT} ( \{ h_v^L \,,\, v \in V \}), \quad (2)
$

여기서 $h_v^0$는 input feature이고 $\mathcal{N}(v)$는 $v$의 이웃 노드들을, $\mathcal{V}$는 $\mathcal{G}$의 노드들의 집합을, $L$은 layer의 개수를 의미합니다. 

방향 그래프(Directed Graphs)가 방향 비순환 그래프(Directed Acyclic Graphs, DAGs)가 되려면 그래프의 간선들이 노드들에 대한 부분적 순서를 정의해야 합니다. 이 부분적 순서는 강력한 귀납적 편향(inductive bias)으로, 신경망에 자연스럽게 incorporate 하고자 하는 것 입니다. 예를 들어, DAG로 표현된 신경 구조는 acyclic dependency of computation을 정의하며, 이것은 구조를 비교하고 성능을 예측할 때 중요한 정보입니다. 따라서 이 정보는 예측 능력을 높이기 위해 architecture representation에 포함되어야 합니다

이를 위해 본 논문에서는 DAGNN(Directed acyclic graph neural networks)이라는 모델을 제안합니다. 

이 모델은 partial order의한 DAG에 대한 representation을 생성합니다. 특히 이러한 순서를 사용하면 모든 이전 노드의 표현을 기반으로 노드 표현을 순차적으로 업데이트할 수 있습니다. 이렇게 하면 후속 노드가 없는 노드가 전체 그래프의 정보를 소화할 수 있습니다.

# 2. THE DAGNN MODEL

DAG 는 사이클이 없는 방향 그래프입니다. DAG를 $\mathcal{G=(V,E)}$로 표시하고, 여기서 $\mathcal{V}$는 노드 집합

$\mathcal{E} \subset \mathcal{V}\times\mathcal{V}$ 는 간선 집합을 나타냅니다. 집합 $S$에 대한 partial order는 transitive하고 asymmetric한 binary relation인 $\le$ 입니다. 이 논문에서는 DAG에서 self-loop를 금지합니다. Partial order가 있는 집합 $S$를 poset이라고 하며 $(S,\le)$로 나타냅니다.

어떤 DAG에 대해 노드 집합 $\mathcal{V}$에 대한 유일한 partial order $\le$를 정의할 수 있으며, 이때 모든 요소 $u,v \in\mathcal{V}$에 대해 $u\le v$인 경우 $u$에서 $v$로의 directed path가 있고, 이러한 directed path가 있다면 $u\le v$입니다. 반대로, 어떤 poset $(S,\le)$에 대해서는 $S$를 노드 집합으로 사용하고,  $u\le v$일 때 $u$에서 $v$로의 directed path가 하나 이상인 DAG가 존재할 수 있습니다.
{% raw %}
DAG에서 (직접적인) predecessors가 없는 모든 노드를 소스(source)라고 하며, 이러한 소스 노드들을 $S$ 집합에 모읍니다. 마찬가지로, (직접적인) successors가 없는 모든 노드를 목표(target)라고 하며, 이러한 목표 노드들을 $\mathcal{T}$ 집합에 모읍니다. 추가적으로 입력 노드 feature의 집합인 $\mathcal{X}=\{{h_v}^0, v \in \mathcal{V}\}$를 정의합니다.
{% endraw %}
## 2.1 MODEL

DAGNN의 주요 아이디어는 DAG에서 정의된 부분적 순서에 따라 노드를 처리하는 것입니다.

MPNN과의 주요 차이점은 다음과 같습니다. 이전 layer 정보 대신 현재 layer 정보를 사용하여 $v$의 현재 layer representaion을 계산합니다.  $v$의 이웃 전체 집합 $\mathcal{N}(v)$가 아닌 direct-predecessor 집합 $\mathcal{P}(v)$에있는 노드들만을 활용해 aggregate합니다.  이러한 차이점은 최종 "readout"에서도 직접적인 차이를 유발합니다.

이를 식으로 나타내면 다음과 같습니다.

$h^{l}_v = F^{l}(h_v^{l-1},G^l(\{h_u^l \vert u\in \mathcal{P}(v)\},h_v^{l-1}))\quad l=1,...,L,\quad (3)$

$h_\mathcal{G} = R^{l}({h^{l}_v, l=0,1,...,L,v \in \mathcal{T}})\quad(4)$ 

여기서 $\mathcal{P(v)}$는 $v$의 direct predecessor 집합을, $\mathcal{T}$는 successors가 없는 노드들의 집합을 나타냅니다. 또한 $G^l,F^l$은 각각 $AGGREGATE^l,COMBINE^l$을 나타내고 $R$은 $READOUT$을 나타냅니다.

다음은 위의 식 (3),(4)의 instantiation에 대해 설명하겠습니다.

### One layer

본 논문에서는 aggregate operator $G^l$을 위해 attention mechanism을 사용하였습니다. 

![image1](https://github.com/sh0613/1234/assets/130838113/ab07f719-b164-42be-bf6e-130207285c39)

$l$번째 layer에 있는 노드 $v$에 대해 $m_v^l$은 모든 direct predecessor $u$의 $h^l_u$의weighted combination입니다. 

Weighting coefficients $\alpha_{vu}^l$은 일반적인 attention mechanisms의 query-key design을 따릅니다. 

![image2](https://github.com/sh0613/1234/assets/130838113/b8ed88cf-2147-4d4a-8637-359fbc2e7f92)

여기서 $w^l_1,w^l_2$는 model parameter입니다. 본 논문에서는 dot product form대신 additive form을 사용하였습니다. 이로 인해 parameter수를 줄일 수 있습니다. 또한 additive form을 사용하여 쉽게 edge attribution을 모델에 포함할 수 있습니다.

다음은  combine operator $F^l$에 대해 설명하겠습니다. 식으로 표현하면 다음과 같습니다.

![image3](https://github.com/sh0613/1234/assets/130838113/5c93f73e-f4ef-4bb0-9a53-9e5d19f65bff)

$F^l$은 메세지 $m_v^l$과 이전 단계의 노드 $v$의 representaion $h_v^{l-1}$을 combine하여 현재 단계의 노드$v$의 representaion을 업데이트 합니다. 여기서 recurrent 구조(GRU)를 사용합니다. 이를 사용하여 partial order와 유사한 순서대로 processing 할 수 있게됩니다.( $h_v^{l-1}$, $m_v^{l}$, $h_v^{l}$ 을 각각 input, past state, update state(output)으로 생각하면 됩니다.)

### Bidirectional processing

본 논문에서 DAG의 방향을 역순으로한 그래프와 그에 해당하는 notation은 그에 해당하는 그래프의 notation에 tilde를 붙입니다. 예를 들어 그래프$\mathcal{G}$의 역방향 그래프는 $\tilde{\mathcal{G}}$로 $\tilde{\mathcal{G}}$의 노드 representaion은 $\tilde{h_v^l}$로 표기합니다. 방향이 바뀌므로 $\mathcal{T}$와 $\tilde{\mathcal{S}}$ 는 같아지게 됩니다.

### Readout

$l$개의 layer를 거친 후 노드 representataion을 이용해 그래프 representataion(Read out)를 계산해야 합니다.

![image4](https://github.com/sh0613/1234/assets/130838113/86e942c5-58d1-46a6-8052-b7f6192e7c06)

그래프$\mathcal{G}$와 $\tilde{\mathcal{G}}$ 의 각 $l$번째 layer를 통과한 노드 representataion을 concat하여 maxpooling한 후 다시 concat하여 fully-connected layer를 통과시켜 readout을 얻습니다. 만약 그래프가 undirected 그래프라면 우측의  $\tilde{\mathcal{G}}$ 없이  $\mathcal{G}$만을 활용하여 readout을 계산하면 됩니다.

### Edge attributes

$\tau(u,v)$를 $edge(u,v)$의 attribute라고 하겠습니다. 그리고 $y_{\tau}$를 $\tau$의 edge representation이라고 하겠습니다. Edge attribute를 기존의 (6)식에 포함시켜

![image5](https://github.com/sh0613/1234/assets/130838113/1ca5b9d8-dc44-4e34-825c-6ea5aa9134ae)

와 같이 나타낼 수 있습니다. 더 적은 parameter를 위해 $w^l_3 = w^l_1$ 로 두어도 모델이 잘 작동합니다.

전체적인 구조를 나타내면 아래와 같습니다.

![image6](https://github.com/sh0613/1234/assets/130838113/1892be28-905b-40ae-9180-7d6b6b9fce26)

## 2.2 TOPOLOGICAL BATCHING

MPNN과의 주요 차이점은 DAGNN이 partial order를 따라야 하기때문에 aggregator 연산시 노드를 순서대로 처리해야 합니다. 계산 효율을 위해 병렬 컴퓨팅 리소스 (예: GPU)를 더 효과적으로 활용하기 위해 동시성을 최대한 활용하는 것이 중요합니다. dependency가 없는 노드들은 그들의 predecessor가 모두 처리된 경우에 아래 그림과 같이 동시에 처리될 수 있습니다.

![image7](https://github.com/sh0613/1234/assets/130838113/f7041181-b168-4feb-8807-59c0a0df42b1)

이러한 아이디어를 구체화 하기위해 본 논문에서는 topological batching을 고려합니다. 
{% raw %}
topological batching은 아래와 같은 세가지 속성을 만족하도록 노드 집합을 순서를 가진 배치$\mathcal\{{B}_i\} _{i\ge0}$로 나눕니다.
{% endraw %}
$(i) \quad B_i$ 는 disjoint 이고 그들의 union은 노드집합 $\mathcal{V}$가 됩니다.

$(ii)\quad$ $u, v\in\mathcal{B_i}$인 노드 pair에 대해 $u$ 에서 $v$로 가는 path 또는 그 역방향인 path는 존재하지 않습니다.

$(iii)\quad i >0$인 모든 $i$에 대해 $\mathcal{B_i}$에 $\mathcal{B_{i-1}}$의 노드중 하나를 head로 하는 노드가 존재합니다(즉 이전의 배치에 존재하는 노드에서 그 다음의 배치에 존재하는 노드로 가는 edge가 존재하게 됩니다.)

Partitioning procedure는 Direct predecessors가 없는 모든 노드들인 $\mathcal{S}$는 초기 배치를 형성하고,반복적으로 방금 형성된 배치를 그래프에서 제거하고 이러한 노드들에서 발생하는 엣지도 제거한 후 남은 그래프에서 Direct predecessors가 없는 노드들이 다음 배치를 형성하는 방식으로 이뤄집니다.

이러한 topological bathching은 각 배치 내의 모든 노드가 병렬로 처리될 수 있도록 모든 순차 배치의 최소 개수를 생성합니다.

![image8](https://github.com/sh0613/1234/assets/130838113/8854c8e2-331b-43cd-948f-d1d5dbae6f7a)

## 2.3 PROPERTIES

이 섹션에서는 DAGNN에 properties에 관해 요약하겠습니다.

$\mathcal{M}$을 그래프에서 그래프 representation으로 바꾸는 함수 $\mathcal{M}: \mathcal{V} \times \mathcal{E} \times \mathcal{X} \rightarrow \mathcal{h_G}$ 로 정의하겠습니다. 

DAGNN은 아래와 같은 성질을 만족합니다.


$ Theorem  2.\text{ The graph representation }\mathcal{h_g} \text{ is invariant to node indexing if all }G^l,F^l\text{and R are so.} $

$ Corollary  3.\text{ The functions } G^l, F^l,\text{and }R \text{ defined in (5)–(8) are invariant to node indexing.}\text{ Hence, the resulting graph representation. } \mathcal{h_G}\text{ is, too.} $

$ Theorem  4.\text{ The mapping M is injective if } 
G^l, F^l
\text{ and R, considered as multiset functions, are so.} $

# 3. COMPARISON TO RELATED MODELS

### Tree-LSTM, DAG-RNN  vs  DAGNN

Tree-LSTM과 DAG-RNN 모두 recurrent한 구조를 사용하지만 hidden state가 단순한 합의 형태나 element-wise product의 형태로 되어있습니다.  또한 Tree-LSTM의 경우 child수(DAGNN의 경우로 생각하자면 predecessor)가 같아야하고 이러한 child가 순서대로 정렬되어야 한다는 한계가 있습니다. 또한 두 모델 모두 single terminal node(leaf node)가 존재한다고 가정해야 합니다. 

### D-VAE vs DAGNN

DAGNN과 D-VAE의 encoder는 비슷한 구조를 가졌습니다. 그러나 두가지의 차이점이 존재합니다.

첫 번째는 D-VAE는 aggregator로 gated sum을 사용하지만 DAGNN은 attention을 사용해서 $h_u^l$의 정보 뿐만아니라 $h_v^{l-1}$의 정보도 활용합니다.

두 번째는 D-VAE는 layer에 대한 개념이 없습니다. 반대로 DAGNN은 multiple layer를 사용합니다. 

# 4. EVALUATION

## 4.1 DATASETS, TASKS, METRICS, AND BASELINES

먼저 데이터셋 , TASK , METRICS, 베이스라인에 대해 간단히 설명하겠습니다.

본 논문에서는 OGBG-CODE dataset을 사용하였습니다. NA dataset, BN dataset을 사용하였습니다. 

OGBG-CODE dataset은 DAG로 파싱된 452,741개의 파이썬 함수를 포함합니다.(여기서 OGBG-CODE-15는 데이터의 15%만 사용했음을 의미합니다.) 이 데이터는 TOK task, LP task를 위해 사용됩니다. TOK task은 함수 이름을 구성하는 토큰 예측을 LP task은 DAG의 최장 경로 길이 예측을 의미합니다. 이를 위한 Metric으로는 TOK의 경우에는 F1 score를 LP의 경우에는 accuracy를 사용하였습니다.

 

Baseline으로 TOK task의 경우 Node2Token, TargetInGraph 과 4 개의 GNN모델(iGCN,GIN,GAT,GG-NN)과 두 개의 hierarchical pooling approaches인 SAGPool과 ASAP모델과 D-VAE 인코더를  사용하였고 LP task의 경우 MajorityInValid와 4개의 GNN모델, SAGPool,ASAP모델, D-AVE인코더가 사용되었습니다. 여기서 각 모델 이름 뒤에 붙은 VN은 모든 노드와 연결되어있는 virtual한 node를 포함 시켰음을 의미합니다.

 NA dataset은 ENAS 소프트웨어로부터 생성된 19,020개의 neural architectures를 포함합니다. 이 데이터는 CIFAR-10 데이터셋에 대한 아키텍처 성능을 예측하기 위한 Task를 위해 사용되었습니다.

BN dataset은 bnlearn패키지(R 패키지)로 부터 생성된 200,000개의 베이지안 네트워크를 포함합니다.이 데이터는 Asia 데이터셋에 대한 베이지안 네트워크(BN)가 얼마나 잘 맞는지를 측정하는 BIC 점수를 예측하는 것입니다. NA와 BN의 task를 위한 metric으로는 RMSE와 Pearson의 상관계수를 사용하였습니다. 

이 Task의 경우 Baseline으로 S-VAE, GraphRNN,GCN, DeepGMG, D-VAE를 사용하였습니다.

![image9](https://github.com/sh0613/1234/assets/130838113/e2e17d68-4f40-44f3-b134-32b5166ab9ca)

## 4.2 RESULTS AND DISCUSSTION

**Prediction performance, token prediction (TOK), Table 1.**

DAGNN이 가장 우수한 성능을 발휘합니다.대부분의 MPNN 모델 (표의 중간 부분)은 메시지 전달에 다섯 개의 레이어를 사용하지만, DAGNN과 D-VAE의 전반적으로 우수한 성능은 네트워크 깊이에 제한이 없는 DAG 아키텍처가 inductive bias를 이용할 수 있다는 것을 나타냅니다.

**Prediction performance, length of longest path (LP), Table 1.**

DAGNN과 D-VAE가 거의 완벽한 정확도를 달성한 점은 주목할 만합니다. 이 결과는 inductive bias 추론 알고리즘 (이 경우, 경로 추적)과 일치할 때 모델이 추론을 더 쉽게 수행하고 샘플 효율성을 높일 수 있도록 학습한다는 Xu et al. (2020)의 이론을 뒷받침합니다.

**Prediction performance, scoring the DAG, Table 2.**

"NA와 BN에 대해서도 DAGNN이 D-VAE보다 우수한 성능을 발휘하는데, 이어서 D-VAE가 다른 네 가지 베이스라인보다 우수한 성능을 보입니다.D-VAE는 DAG bias를 통합하는 이점을 보여주지만, DAGNN은 architectural component의 우수성을 입증하며, 이는 ablation study에서  확인할 수 있습니다.

![image10](https://github.com/sh0613/1234/assets/130838113/6d1657aa-35ed-4fbe-ba04-c7dce9529982)

**Ablation study, Table 3.**

Table 3에서는  aggregator에서 attention 을 gated sum으로 대체했을 때, GRU를 완전 연결 레이어로 대체했을때, readout 수정, edge attribute 제거의 결과를 각각 보여줍니다.  Attention을 Gated-sum으로 바꿨을 때 성능의 차이가 가장 큰 것을 볼 수 있습니다. 그러나 LP-15의 결과는 다른 결과와 다르게 성능이 더 좋아진 것을 볼 수 있습니다. BN에서 readout을 변형시켰을 때 더 성능이 좋아지는 것을 볼 수 있습니다. 이는 풀링이 모든 노드에 대한 overemphasis를 보정해주는 것으로 보입니다.

![image11](https://github.com/sh0613/1234/assets/130838113/034b9367-db19-4c41-b4b9-f5800b2a3167)

![image12](https://github.com/sh0613/1234/assets/130838113/b146a877-d016-4790-8090-63820df83b84)

**Sensitivity analysis, Table 4 and Figure 4**

Table 4는 layer개수에 따른 성능을 나타냅니다.최고의 성능은 일반적으로 두 개 또는 세 개의 Layer에서 나타납니다. 즉, 하나의 layer로는 충분하지 않으며, 세 개 이상의 레이어는 두 개의 레이어보다 이점을 제공하지 않는 것을 볼 수 있습니다.

# CONCLUSTIONS

DAGNN은 DAG가 가지는 partial order에 의한 강력한 inductive bias을 representation 학습을 향한 중요한 도구로 활용합니다. 이 inductive bias을 이용해 DAGNN이 MPNN(메시지 전달 신경망)을 대표적인 대표적인 데이터셋과 작업에서 능가하는 것을 볼 수 있었습니다. DAGNN이 순열 불변성(permutation invariant) 및 injective한 성질을 가지는 것을 볼 수 있습니다.





### Author

- Seonghyeon Jo
    - Graduate School of Data Science, KAIST, Daejeon, Korea