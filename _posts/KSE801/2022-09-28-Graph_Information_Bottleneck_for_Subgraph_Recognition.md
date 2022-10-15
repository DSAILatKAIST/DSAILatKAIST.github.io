---
title:  "[ICLR 2020] Graph Information Bottleneck for Subgraph Recognition"
permalink: Graph_Information_Bottleneck_for_Subgraph_Recognition.html
tags: [reviews]
---

$$
\begin{aligned}
\mathcal{L}_ {VIB} = \frac{1}{N} \sum_{i=1}^{N} \int\nolimits p(z|x_{i})\log{q_{\phi}(y_{i}|z)} dz  - \beta \mathrm{KL}(p(z|x_{i})|r(z)),
\end{aligned}
$$

---
description: >-
  Wang, Junshan et al./ Graph Information Bottleneck for Subgraph Recognition/
  KDD-2022
---
# Graph Information Bottleneck for Subgraph Recognition
## **1. Problem Definition**
> **Graph classification 작업에 중요한 역할을 하는 압축된 데이터를 추출한다.**  


레이블을 기반으로 분류하는 작업은 다양한 분야에서 적용될 수 있고 딥러닝 학습에서 근본적인 문제라고 할 수 있습니다. 그러나 실제 데이터에서는 분류작업에 관계없는 노이즈 정보가 포함되어 있을 가능성이 높으며, 이것은 원 데이터에서는 추가적인 정보를 제공하여 고유한 특성을 보유하도록 하지만, 실제로 분류작업을 하는데 있어 부정적인 영향을 미칩니다. 이러한 문제에 기인하여 분류 작업에 결정적인 역할을 하는 압축된 정보를 인식하도록 하는 문제가 제안되었습니다. 예를 들어 원자를 node로 정의하고 원자간 결합을 edge로 정의한 분자 그래프에서 분자의 functional group를 나타내는 subgraph를 추출하는 것을 목표로 할 수 있습니다. 
  
![image](https://distill.pub/2021/gnn-intro/graph_xai.bce4532f.png)
  
## **2. Motivation**  

> **Graph 데이터를 Information Bottleneck의 관점에서 접근한다.**   

subgraph 인식이라는 문제가 중요한 과제로 대두되면서, 그래프의 label을 예측하는데 있어서, 정보 손실을 최소화하면서 압축된 subgraph를 어떻게 추출할 수 있을지에 대해서 개발하는 작업이 지속되고 있습니다. 최근 Information Bottleneck(IB)이라는 정보 이론 분야에서 이러한 문제에 대해서 다루고 있습니다. 이는 그래프 label의 정보를 가장 잘 담고 있는 정보를 유지한 채로 원본 데이터에서 압축된 방식으로 추출하는 것을 목표로 하고 있습니다. deep learning으로 강화된 IB는 컴퓨터 비전, 강화학습, 자연어 처리 분야에 적용되어, 다양한 분야에서 적절한 feature를 학습할 수 있도록 하는데 성공하였습니다.   
그러나 현재의 IB 방법은 상호간에 관계적인 정보와 이산적인 데이터를 담고 있는 그래프 데이터를 처리하는데 어려움이 있습니다. IB가 정보 손실을 최소화하면서 원본 그래프의 하위 그래프와 같은 불규칙한 그래프 데이터를 압축하는 것은 여전히 어려운 일이라고 할 수 있습니다.   
따라서 본 모델에서는 앞서 언급된 subgraph graph 인식 문제를 해결하기 위해서 그래프 데이터에 IB 원리를 발전시켜서, 새로운 원리인 GIB(Graph Information Bottleneck) 방법을 제안합니다. 기존의 IB는 숨겨진 임베딩 공간에서 최적인 representation을 학습하여 main task에 유용한 정보를 추출하는 한편, GIB에서는 graph level에서 중요한 subgraph를 추출하도록 합니다.  


## **3. Method**
> **Preliminaries**
  
논문에서 제안한 방법론을 이해하기 위해서 몇 가지 Notation과 `GNN`의 개념을 소개하겠습니다.

N개의 그래프로 구성된 집합 $\lbrace ( \mathcal{G}_ 1, Y_ 1),\dots,(\mathcal{G}_ N, Y_N) \rbrace$에서 $\\mathcal{G}_ n$은 n번째 그래프를 나타내고, $Y_n$는 n번째 그래프에 해당하는 레이블을 나타냅니다.   
$\mathcal{G}_ n=(\mathbb{V},\mathbb{E}, A, X)$에서 해당 그래프는 속하는 노드 집합 $\mathbb{v}=\lbrace V_i|i=1,\dots, M_n \rbrace$,   
edge 집합 $\mathbb{E}=\lbrace (V_i, V_j)|i>j; V_i,V_j \text{ is connected} \rbrace$,  
인접행렬 $A\in \lbrace 0,1 \rbrace^\lbrace M_n\times M_n \rbrace$,   
feature 행렬 $X\in \mathbb{R}^{ M_n\times d}$로 구성되어 있습니다.   
$\mathcal{G}_ {sub}$는 특정 subgraph를 나타내고,  $\overline{\mathcal{G}}_ {sub}$는 $\mathcal{G}_ {sub}$를 제외한 나머지 부분을 의미합니다. $f:\mathbb{G} \rightarrow \mathbb{R} / [0,1,\cdots,n] $는 그래프에서 실수값으로 mapping하는 함수를 의미하고, 여기서 $\mathbb{G}$는 input graph의 도메인입니다. 

> **Graph Convolutional Network** 
  
Graph Convolutional network(GCN)은 그래프 분류 작업에 널리 사용됩니다. node feature $X$와 인접행렬 $A$를 가지는 그래프 $\mathcal{G} = (\mathbb{V},\mathbb{E})$ 를 가정할 때, GCN은 다음의 과정을 통해 노드 임베딩 $X^{'}$을 도출합니다.

$$ X^{'} = \mathrm{GCN}(A,X;W) = \mathrm{ReLU}(D^{-\frac{1}{2}}\hat{A}D^{-\frac{1}{2}}W) $$

여기서 D는 노드의 차수를 담은 대각 행렬이고, W는 모델 파라미터를 의미합니다.


> **Graph Information Bottleneck**

먼저 Graph Information Bottleneck 현상과 IB subgraph를 정의합니다.  

![image](http://snap.stanford.edu/gib/venn.png)  

Information Bottleneck 현상의 원리를 일반화하여 불규칙하고 복합적인 그래프의 정보 표현을 학습하며, 이를 통해 그래프 정보 병목 현상(GIB)을 도출합니다.  
그래프 $\mathcal{G}$와 레이블 $Y$가 있을 때, GIB는 가장 유용하지만 압축된 representation $Z$를 탐색합니다. 탐색 과정은 아래와 같습니다. 

$$ \max_{Z}{I(Y,Z)}  \text{ s.t. } I(\mathcal{G} ,Z) \leq I_{c} $$

여기서 $I_c$는 $\mathcal{G}$와 $Z$ 사이의 정보량의 제한을 나타내기 위해 사용됩니다. 즉, 압축된 형태로 나타내는 것을 목적으로 하기 때문에 $\mathcal{G}와 $Z$ 사이의 Mutual Information을 최소화하는 방식으로 최적화가 진행됩니다. 여기에서 Lagrange multiplier $\beta$를 도입하여, 제약 조건을 제거합니다.

$$ \max_{Z}{ I(Y,Z) - \beta I(\mathcal{G},Z) } $$

위의 식은 Graph Information Bottleneck의 핵심 과정을 나타냅니다. 여기서, subgraph 인식에서는 그래프 속성 측면에서 중요한 정보 손실을 최소화하면서 정보를 최대한 압축하는데에 집중하고 있습니다.  
최종적으로 도출된 핵심 subgraph(IB-subgraph)는 유용하면서도 최소한의 그래프를 담아야 하고, 그 결과는 다음과 같이 표현될 수 있습니다.  

$$ \max_{\mathcal{G}_ {sub}\in \mathcal{G}_ {sub}} I(Y,\mathcal{G}_{sub})-\beta I(\mathcal{G},\mathcal{G}_{sub}) $$  

이와 같이 도출된 IB-subgraph는 그래프 분류 개선, 그래프 해석, 그래프 잡음 제거 등 여러 그래프 학습 과제에 적용할 수 있습니다. 그러나 위의 식의 GIB 목적함수는 mutual information과 그래프의 이산적인 특성으로 인해서 최적화하기가 어렵습니다. 그래서 이러한 목적함수를 최적화하기 위해 subgraph를 도출하는 방법에 대한 접근방식이 추가적으로 필요합니다.

  
> **Graph Information Bottleneck의 목적함수 최적화 과정**  
* Maximization of $I(Y,\mathcal{G}_ {sub})$  
위의 GIB 목적함수는 2개의 부분으로 구성됩니다. 먼저 식의 첫 번째 항 $I(Y,\mathcal{G}_ {sub})$를 살펴봅니다. 이 항은 $\mathcal{G}_ {sub}$ 와 $Y$ 간의 연관성을 측정하는 부분입니다. $I(Y; \mathcal{G}_ {sub})$는 다음과 같이 확장할 수 있습니다.  

$$ I(Y,\mathcal{G}) = \int p(y,\mathcal{G}_ {sub}) \log{{p(y|\mathcal{G}_ {sub})}} dy d \mathcal{G}_ {sub} + \mathrm{H}(Y)  $$  

$H(Y)$는 $Y$의 엔트로피으로 고정된 값이므로 무시합니다. 또한 
$$p(y,\mathcal{G}_ {sub}) \approx \frac{1}{N} \sum_{i=1}^{N} \delta_ {y_ i}(y) \delta_ {\mathcal{G}_ {sub, i}}(\mathcal{G}_ {sub})$$와 같이 
$p(y,\mathcal{G}_ {sub})$를 근사할 수 있습니다. 여기서 $\mathcal{G}_ {sub}$는 출력 하위 그래프이고 $Y$는 그래프 레이블입니다. 실제 사후 확률 $p(y|\mathcal{G}_ {sub})$를 variational approximation $q_{\phi_{1}}(y|\mathcal{G}_ {sub})$으로 대체함으로써, 우리는 위의 식에서 첫 번째 항의 tractable한 최소값을 다음과 같이 얻을 수 있습니다.  

$$ I(Y,\mathcal{G}_ {sub}) \geq \int p(y,\mathcal{G}_ {sub}) \log{{q_{\phi_{1}}(y|\mathcal{G}_ {sub})}} dy \ d\mathcal{G}_ {sub} \\
\approx \frac{1}{N} \sum_{i=1}^{N} q_{\phi_{1}}(y_{i}|\mathcal{G}_ {sub_{i}}) =: -\mathcal{L}_ {cls}(q_{\phi_{1}}(y|\mathcal{G}_ {sub}),y_{gt}) $$

여기서 $y_{gt}$는 그래프의 groundtruth 레이블입니다. 위의 식은 Y와 Gsub 사이의 classification loss를 $\mathcal{L}_ {cls}$로 최소화함으로써 I(Y;\mathcal{G}_ {sub})를 최대화한다는 것을 나타냅니다. 직관적으로 살펴보아도 $\mathcal{L}_ {cls}$를 최소화하여 분류 정확도를 높인다면, subgraph가 그래프 레이블을 정확하게 예측할 수 있다는 것을 의미합니다. 실제로 classification $Y$에 대해서는 cross entropy loss를 사용하고, regression Y에 대해서는 mean square error(MSE)를 선택합니다.

*  Minimization of $I(\mathcal{G},\mathcal{G}_ {sub})$   
이제 두 번째 항 $I(\mathcal{G},\mathcal{G}_ {sub})$에 대한 최소화 작업을 진행합니다.  KL-divergence의 Donsker-Varadhan representation을 적용하여,  $I(\mathcal{G},\mathcal{G}_ {sub})$을 다음과 같이 나타낼 수 있습니다.  

$$I(\mathcal{G},\mathcal{G}_ {sub}) = \sup \limits_{f_{\phi_2}:\mathbb{G} \times \mathbb{G} \rightarrow \mathbb{R}} \mathbb{E}_ {\mathcal{G},\mathcal{G}_ {sub}\in p(\mathcal{G},\mathcal{G}_ {sub})}f_{\phi_{2}}(\mathcal{G},\mathcal{G}_ {sub})-\log{\mathbb{E}_ {\mathcal{G} \in p(\mathcal{G}),\mathcal{G}_ {sub}\in p(\mathcal{G}_ {sub})}e^{f_{\phi_{2}}(\mathcal{G},\mathcal{G}_ {sub})}}$$

여기서 $f_{\phi_{2}}$는 앞에서 말씀드렸다시피 그래프 집합에서 실수 집합으로 매핑되는 네트워크입니다. 이 식의 유도 과정에 집중하기보다는 결과 해석과 모델 설계 방향에 대해 말씀드리고자 합니다.  
앞서 나온 식을 바탕으로 설계된 모델의 아키텍처는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/67723054/196002229-221dee3c-a226-4541-a28f-d65f5232380a.PNG)

먼저 GNN을 사용하여 $\mathcal{G}$와 $\mathcal{G}_ sub$ 로부터 임베딩을 추출하고, $\mathcal{G}$와 $\mathcal{G}$ 임베딩을 concat시켜서, MLP의 입력으로 설정합니다. 그 결과 output은 subgraph $\mathcal{G}$와 나머지 부분 $\overline{\mathcal{G}}_ {sub}$에 해당하는지의 여부를 결정하는 sampling matrix를 도출하게 됩니다. $p(\mathcal{G}_ {sub})$에 대한 샘플링 방법과 함께 $I(Y,\mathcal{G}_ {sub})$에 대한 다음과 같은 최적화 문제에 도달하게 됩니다.   
  
$$\max_{\phi_{2}} \mathcal{L}_ {\mathrm{MI}}(\phi_{2},\mathcal{G}_ {sub}) =  \frac{1}{N}\sum_ {i=1}^{N}f_{\phi_{2}}(\mathcal{G}_ {i},\mathcal{G}_ {sub,i})-\log{\frac{1}{N}\sum_{i=1,j\neq i}^{N}e^{f_{\phi_{2}}(\\mathcal{G}_ {i},\mathcal{G}_ {sub,j})}}$$  
  
그래프 데이터의 Mutual Information 근사과정을 통해서 $I(Y,\mathcal{G}_ {sub})$의 최대화 과정과 $I(\mathcal{G},\mathcal{G}_ {sub})$의 최소화 과정을 결합하여 최종적으로 다음과 같은 최적화 과정을 설계할 수 있습니다.  inner loop에서는 IB-subgraph를 통해 나온 임베딩을 활용하여 $I(\mathcal{G},\mathcal{G}_ {sub})$를 최소화하는 과정을 거칩니다. $I(\mathcal{G},\mathcal{G}_ {sub})$에 대한 좋은 추정이 이루어지면, outer loop에서 mutual information, classification loss, connectivity loss를 사용하여 GIB의 목적함수를 최적화합니다.

$$
\min \limits_{\mathcal{G}_ {sub},\phi_ {1}} \mathcal{L}(\mathcal{G}_ {sub},\phi_{1},\phi_{2}^{* }) = \mathcal{L}_ {cls}(q_ {\phi_ {1}}(y|\mathcal{G}_ {sub}),y_ {gt}) + \beta \mathcal{L}_ {\rm MI}(\phi_ {2}^{* },\mathcal{G}_ {sub}) 
$$  

$$
\text{ s.t. }  \phi_{2}^{*} = \arg\max_{\phi_ {2}}\mathcal{L}_ {\mathrm{MI}}(\phi_{2},\mathcal{G}_ {sub}) 
$$  


먼저 inner loop에서 $\phi_{2}^{* }$를 $\phi_{2}^{* }$로 최적화하고, 이후에 outer loop에서 $\phi_{2}^{* }$를 활용하여 $I(\mathcal{G},\mathcal{G}_ {sub})$에 대한 minimization 작업을 진행하고 classification loss $\mathcal{L}_ {cls}$를 기반으로 $Y$와 $\mathcal{G}$간의 mutual information을 최대화시킨다. 이 과정에서 $\phi_{1}$과 $\mathcal{G}_ {sub}$가 IB-subgraph를 생성하도록 최적화하게 됩니다.


## 4. Experiment  

실험은 총 3가지 실험을 진행하였고, graph classification과 graph interpretation(해석), graph denoising(노이즈 제거)의 관점에서 본 논문에서 제안된 GIB(Graph Information Bottleneck)을 평가합니다. 

> **Graph Classification**  

GIB 알고리즘은 subgraph $\mathcal{G}_ {sub}$의 representation을 활용하여 $\mathcal{L}_ {cls}$을 기반으로 graph classification task를 진행합니다. 이 과정에서 GCN, GAT, GIN, GraphSAGE를 포함한 여러가지 backbone에 GIB를 연결합니다. 이 모델들을 다양한 aggregation 방식들의 모델과 비교합니다.  
* pooling 기반 aggregation : SortPool, ASAPool, DiffPool, EdgePool, AttPool
* mean/sum 기반 aggregation : GCN, GraphSAGE, GIN, GAT

MUTAG, PROTEINS, IMDB-BINARY, DD 총 4가지의 데이터셋으로 graph classification을 진행합니다. 그 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/67723054/196009252-9bdf3ea0-80a4-459a-8aff-1e567e1eacce.PNG)  

위의 결과에서 논문에서 제안한 방법과 baseline을 종합적으로 평가하였습니다. 다양한 backbone에서 GIB를 학습시키고, subgraph에서 representation을 aggregation하여 성능을 측정한 결과, 1개의 데이터셋(MUTAG)을 제외하고 나머지 데이터셋에서 가장 좋은 성능을 낸다는 것을 확인할 수 있습니다. 이는 GIB가 그래프 분류의 핵심적인 구조를 추출할 수 있으며 이것이 그래프 분류 작업에 있어서 결정적인 역할을 할 수 있다는 것을 의미합니다.
  
> **Graph Interpretation**

GIB의 가장 두드러진 특징은 interpretation 과정을 진행할 수 있다는 것입니다. 이는 그래프의 핵심 특성을 반영할 수 있는 하위 구조를 찾는 작업입니다. 이 interpretation 작업과 classification 작업을 동시에 진행함으로써 많은량의 유용한 정보를 도출해낼 수 있다는 점에서 큰 의미가 있습니다. 본 실험에서는 GIB와 attention 메커니즘을 비교합니다. 즉, graph prediction에 대한 node의 정보를 aggregation합니다. attention score를 50%(Att05), 70%(Att07)로 설정하여 해석가능한 subgraph를 추출합니다. 그래프와 그에 해당하는 subgraph 사이에 나타나는 예측 차이에 대한 평균과 분산을 측정합니다. 공정한 비교를 위해서 모든 방법에 대해 GCN backbone을 사용하였으며, $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 제외하고 학습을 진행하였습니다.  

250000여개의 분자를 포함하는 ZINC dataset을 사용하고, 4가지 분자의 특성에 대해서 그래프 해석 작업을 진행합니다. QED는 0에서 1까지의 범위로 약물의 유사성을 측정합니다. DRD2는 0에서 1까지의 범위로 분자가 도파민 수용체에 대해서 활성일 확률을 나타냅니다. HLM-CLint와 MLM-CLint는 시험관 내에서 인간과 쥐의 간 마이크로솜의 대사 안정성을 추정한 값입니다. 각 작업에 대해서 training set, validation set, test set의 비율을 85%, 5%, 10%로 분할합니다.  

![image](https://user-images.githubusercontent.com/67723054/196010188-1a19c246-6835-4238-a857-5ac9a21ed0bf.PNG)

위의 결과는 graph interpretation 작업에 대한 정량적인 성능을 보여줍니다. GIB에 의해 도출된 subgraph에 대해서 전체 모델 중에서 가장 작은 차이를 보여주고 있기 때문에, IB-subgraph가 입력 그래프와 가장 유사한 특성을 가지기 때문에 GIB가 높은 그래프 해석 능력을 가진다는 것을 확인할 수 있습니다. 또한 $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 각각 제거한 방법의 결과도 함께 제시하고 있습니다. GIB가  $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 제외한 성능보다 우수하기 때문에 모델의 모든 부분이 성능 개선에 기여한다는 것을 알 수 있습니다.  

![image](https://user-images.githubusercontent.com/67723054/196010588-594f8ace-fb01-4038-808d-abd7a439aba4.PNG)  

위의 그림은 graph interpretation 작업에 대해서 한눈에 보기 쉽게 나타나는 정성적인 결과입니다. GIB가 HLM-CLint와 LML-CLint의 관점에서 왼쪽의 원래 그래프와 가장 유사한 특성을 보여준다는 것을 알 수 있습니다. 게다가 어떤 subgraph가 그러한 결과에 기인하였는지 붉은색의 부분으로 표시되어있습니다. 이를 통해서 어떤 subgraph가 모델의 전체 특성을 지배하는지 비교적 정확하게 알 수 있습니다.  



표 2는 그래프 해석 작업에 대한 다양한 방법의 정량적 성능을 보여줍니다. GIB에 의해 발견된 하부 구조는 입력 분자와 가장 유사한 특성을 가지므로 GIB는 정확한 그래프 해석(IBsubgraph)을 생성할 수 있습니다. 그런 다음 Lcon 과 LMI 를 삭제하여 방법의 두 가지 변형을 유도합니다. GIB는 또한 변형보다 성능이 우수하므로 모델의 모든 부분이 성능 개선에 기여함을 나타냅니다. 실제로 우리는 Lcon을 제거하면 부분 그래프 생성에 대한 지속적인 완화로 인해 불안정한 훈련 과정을 초래한다는 것을 관찰했습니다. 그림 2에서 GIB는 화학 전문가에 의해 확인된 분자의 특성에 대해 보다 간결하고 합리적인 해석을 생성합니다. 더 많은 결과는 부록에 나와 있습니다. 표 4에서는 압축된 하위 그래프가 더 많은 토폴로지 정보를 보존해야 하므로 그래프당 연결이 끊긴 하위 구조의 평균 수를 비교합니다. GIB는 그래프 속성을 더 잘 해석하기 위해 더 간결한 하위 그래프를 생성합니다.




 QED는 범위(0, 1:0) 내에서 경계를 이루는 분자의 약물 유사성을 측정합니다. DRD2는 분자가 (0; 1:0)에 결합된 도파민 2형 수용체에 대해 활성일 확률을 측정합니다. HLM-CLint 및 MLM-CLint는 시험관 내 인간 및 마우스 간 마이크로솜 대사 안정성의 추정값입니다(mL/min/g의 기본 10 로그). 각 작업에 대해 QED 0:85, DRD2 0:50, HLM-CLint 2, MLM-CLint 2로 분자를 샘플링합니다. 우리는 이러한 분자의 85%를 훈련에, 5%를 검증에, 10%를 테스트에 사용합니다. 
그래프 분류 개선: 그래프 분류 개선을 위해 GIB는 하위 그래프 정보를 집계하여 그래프 표현을 생성합니다. 우리는 GCN [19], GAT [30], GIN [32] 및 GraphSAGE [14]를 포함한 다양한 백본에 GIB를 연결합니다. 제안된 방법을 다음과 비교합니다.
분류 정확도 측면에서 평균/합계 집계 [19, 30, 14, 32] 및 풀링 집계 [38, 26, 35, 6].

$N$개의 노드를 가진 그래프 $$\mathcal{G}= \lbrace \mathcal{V},\mathcal{E} \rbrace$$가 주어지고, $$X = \lbrace x_{1}, x_{2}, ..., x_{N} \rbrace$$ 을 node feature의 집합이라고 하고, $$A$$를 node들의 관계를 표현하는 adjacency matrix라고 하겠습니다.
$$l-th$$ hidden layer에서의 $$v_{i}$$의 hidden representation을 $$h_{i}^{(l)}$$ 이라고 할 때, 이 $$h_{i}^{(l)}$$는 다음과 같이 계산됩니다:
$$h_{i}^{(l)} = \sigma(\sum_{j \subset \mathcal{N}(i)} \mathcal{A_{ij}}h_{j}^{(l-1)}W^{(l)})$$
이 때, $$\mathcal{N}(i)$$ 는 $$v_{i}$$의 neighbors를 의미하고,   
$$\sigma ( \bullet )$$는 activation function, $$W^{(l)}$$은 $$l-th$$ layer의 transform matrix를 나타냅니다.
Continual Learning setting에서, 데이터는 그래프의 형태를 띠고 연속적으로 들어옵니다. 이는 다음과 같이 표현이 가능합니다.
$$\mathcal{G} = (\mathcal{G}^1, \mathcal{G}^2, ..., \mathcal{G}^T)$$
where $$\mathcal{G^t} = \mathcal{G}^{t-1}+\Delta \mathcal{G}^t$$
여기서 $$\mathcal{G} = (A^t, X^t)$$ 는 attributed graph at time $$t$$이고, $$\Delta \mathcal{G} = (\Delta A^t , \Delta X^t)$$는 time $$t$$에서의 node attribute와 network의 structure의 변화량을 나타냅니다.
이 때 Streaming `GNN`은 traditional `GNN`을 streaming setting으로 확장한 것이 됩니다. Streaming graph가 있을 때, Continual Learning의 목적은 $$(\theta^1, \theta^2, ..., \theta^T)$$ 를 배우는 것입니다. 이 때 $$\theta^t$$ 는 time $$t$$ 에서의 `GNN` parameter를 의미합니다.
> **Model Framework**
저자들은 이 논문에서 `SGNN-GR`이라는 방법론을 제시합니다. 모델 구조는 아래 그림과 같습니다.
![image](https://user-images.githubusercontent.com/99710438/194887946-3f736cc4-1c2c-47ca-97aa-4516da0ae42e.png)
**모델을 요약하자면, 아래와 같습니다.**
* 새로운 task가 오면 `GAN`으로 sequence를 생성(이게 `replay buffer`가 되는 것이죠)해서 이번 task의 그래프와 **같이** `GNN`을 학습합니다.
* 이러면 이 `GNN`은 **현재 그래프를 학습함과 동시에 이전의 정보까지 기억**하게 될 것입니다.
* 또한 이번 task에서 새롭게 생성된 node들과 그것들로부터 영향받은 node들을 다시 `GAN`의 input으로 주어 학습시킵니다.
* 이러면 다음 task에서는 `GAN`은 더 양질의 `reaply buffer`를 만들어 낼 수 있을 것입니다.
지금부터 `SGNN-GR`의 자세한 내용을 살펴보겠습니다. 위 그림을 잘 참고하면서 아래 설명을 따라오시기 바랍니다.
가장 먼저, Streaming GNN의 time $$t$$에서의 loss는 다음과 같습니다.
$$\mathcal{L}(\theta^t ; \mathcal{G}^t) = \mathcal{L}(\theta^t ; \mathcal{G}_A^t) + \lambda \mathcal{R} (\theta^{t-1} ; \mathcal{G}_S^t)$$
우변의 첫 항은 incremental learning에 관한 것이고, 두 번째 항은 historical knowledge에 관한 것입니다.
본 논문에서 $$\mathcal{G}_A^t$$ 는 graph의 affected part, $$\mathcal{G}_S^t$$ 는 grpah의 stable part로 정의합니다.
이 때 $$\Delta \mathcal{G}^t \subset \mathcal{G}_A^t$$ 이고 $$\mathcal{G}_S^t \subset \mathcal{G}^{t-1}$$ 입니다. 몇몇 node들이 새롭게 바뀐 node들에 대해서 영향을 받는 것입니다.
각 time step에서 모델은 main model(`GNN`)과 Generative Model로 구성됩니다. 위 그림에서 확인할 수 있듯이, Generative Model은 $$\mathcal{G}_ A^t$$에서 바뀐 node들과 $$\mathcal{G}^{t-1}$$에서의 replayed node를 training data로 받습니다. 이 때 replayed node는 이전 time step의 Generative Model로부터 나옵니다.
이 논문에서는 Generative Model로 `GAN`을 사용하였습니다. `GAN`에 대한 자세한 설명은 생략하며, 원 논문은 [여기](https://dl.acm.org/doi/abs/10.1145/3422622)를 참고하시기 바랍니다.
`GNN` 모델도 changed node와 replayed node를 똑같이 input으로 받습니다.
Main model의 loss function은 다음과 같습니다.
$$\mathcal{L}_ {GNN} (\theta^t) = r \mathbf{E}_ {v \sim \mathcal{G}_ A^t } \[ l(F_{\theta^t}(\upsilon), y_{\upsilon} ) \] + (1-r) \mathbf{E}_ {v' \sim G_{\phi^{t-1}}} \[ l(F_{\theta^t}(\upsilon '), F_{\theta^{t-1}}(\upsilon ')\] $$
여기서 $$v$$는 changed node, $$v'$$는 replayed node입니다. 즉, 이 모델은 새로 들어온 node와 이전에 학습했던 node(replayed)를 동시에 학습합니다.
> **Generative Model for Node Neighborhood**
앞서 언급한대로, 일반적인 Generative model(ex. `GAN`)은 주로 computer vision 분야에서 활발하게 연구되었으나, graph data는 structure에 dependent하기 때문에, edge의 생성은 independent한 event가 아니라 jointly structured 되어야 합니다.
`NetGan`이나 `GraphRNN`같은 Graph Generative model들이 있지만, 이는 전체 그래프를 생성하기 위함이지 node의 neighborhood를 생성하기 위함이 아니어서, 저자들은 `ego network`라는 node neighborhood 생성모델을 제시합니다. 이 `ego network`는 `GAN`의 프레임워크와 유사하지만, 그래프 상에서의 random walks with restart, 즉 `RWRs`를 학습하는 방향으로 사용합니다.
`RWRs`는 일반적인 `Random Walk`모델에서 일정 확률로 starting node로 돌아가고, 그렇지 않으면 neighborhood node로 넘어갑니다. 이는 기존 `RWRs`가 `Random Walk`보다 훨씬 적은 step으로 explore가 가능하게 한다고 합니다.
지금부터 generator에 관한 설명을 보겠습니다.
저자들은 node간의 dependency를 capture하기 위해 **m**이라는 graph state를 정의합니다. 각 walk step에서 $$m_l$$과 $$v_l$$을 계산하는데, 이 때의 input은 last state $$m_{l-1}$$과 last input $$s_{l-1}$$입니다. 이 $$s_{l-1}$$은 node idnetity $$v_{l-1}$$과 node attribute $$x_{l-1}$$을 포함하고 있습니다.
Current state $$m_ l$$은 neural network $$f$$로 계산됩니다.
Generator의 update process는 다음과 같습니다.
$$m_l = f(m_{l-1}, s_{l-1}),$$
$$v_l = softmax(m_l \cdot W_{up,adj}),$$
$$x_l = m_l \cdot W_{up,fea},$$
$$s_l = (v_l \oplus x_l) \cdot W_{down}$$
여기서 $$W_{up}, W_{down}$$은 차원을 맞춰주기 위한 projection matrix라고 생각하시면 됩니다.
저자들은 `WGAN` 프레임워크를 사용해 모델을 학습을 진행했고, 위의 그림에서 확인할 수 있듯이 이 generator는 새로운 그래프 $$\mathcal{G}_ t$$ 에서 `RWRs`로 생성된 Sequence들을 input으로 받아 학습을 진행하고, 다음 task에서 `replay buffer`에 넣을 sequence를 뱉어줍니다. `GNN`은 이 sequence까지 포함해 학습하여 `catastrophic forgetting`을 방지합니다.
> **Incremental Learning on Graphs**
지금부터는 Continual Learning이 어떻게 이루어지는지 보겠습니다.
먼저 저자들은 affected nodes를 정의합니다.
그래프가 time step에 따라 변하면서, 새로운 node나 edge가 생성되면 주위 K(`GNN`의 layer 수)-hop 이내의 neighborhood만 change 됩니다. (`GNN`의 layer가 2개라면, 한 node가 변할 때 그 node와 edge 2개 이내로만 연결되어 있는 node들만 변한다는 의미입니다.)
Changed node중에 **크게 변한 것들**이 있을 것이고, **유의미한 변화가 없는 것들**이 있을 것입니다. 이 **크게 변한 것들**이 전체적인 neighborhood의 패턴을 바꿀 가능성이 있는 node 들이라, 학습에 사용해야하는데, 크게 변했다는 것을 어떻게 확인할 수 있을까요?
저자들은 아래와 같은 influenced degree를 정의하고 그 influence degree가 threshold $$\delta$$ 보다 크다면 affected node라고 취급합니다.
$$ \mathcal{V}_ C^t = \lbrace v| \lVert F_{\theta^{t-1}} (v, \mathcal{G}^t) - F_{\theta^{t-1}} (v, \mathcal{G}^{t-1}) \rVert > \delta \rbrace$$
위 식을 해석해보면, 어떤 node $$v$$의 이전 그래프 $$\mathcal{G}^{t-1}$$에서의 representation와 현재 그래프 $$\mathcal{G}^t$$에서의 representation이 많이 차이난다면, 이 node는 이전 그래프에서 현재 그래프로 넘어오면서 영향을 받았다고 보는 겁니다. 꽤 직관적인 해석입니다.
이런 affected node들은 이전 그래프가 가지고 있지 않은 새로운 패턴을 가지고 있으므로, Generative Model에 input으로 넣어 학습시킨 뒤에 다음 task부터 새로운 패턴을 반영해서 좋은 `replay buffer`를 만들도록 합니다.
추가로, 저자들은 간단한 filter를 추가해 generator가 생성한 node $$v_i$$가 affected node $$v_j$$와 **많이 비슷한 경우**, 패턴의 redundancy를 줄이기 위해 아래의 식처럼 필터링합니다.
$$p_{reject} = max(p_{sim} (v_i, v_j) , j \subset \mathcal{V}_ C^t) \times p_r$$
여기서 $$p_r$$은 disappearacne rate로 사전에 정의하고, similarity는 다음과 같이 정의됩니다.
$$p_{sim} (v_i, v_j) = \sigma (- \lVert F_ {\theta^{t-1}}(v_i, \mathcal{G}^{t-1}) - F_ {\theta^{t-1}}(v_j, \mathcal{G}^{t-1})  \rVert)$$
이때 $$\sigma$$는 sigmoid function이고, 위 식도 직관적으로 두 node의 representation의 차이가 적으면 비슷하다고 보는 겁니다.
이 filter를 통해 저자들은 중복되는 지식은 점차 잊혀지고 바뀌는 distribution이 안정적으로 학습될 것이라 했습니다.
아래의 알고리즘을 통해 지금까지 설명했던 내용들을 확인할 수 있습니다.
![image](https://user-images.githubusercontent.com/99710438/194888070-5da986d2-1702-4cd5-b77e-cfa3d76a0467.png)
## **4. Experiment**
> 본 논문에서 저자들은 다양한 dataset을 통해 baseline들과 `SGNN-GR`을 비교했습니다.
### **Experiment setup**
* Dataset
  * Cora
  * Citeseer
  * Elliptic (bitcoin transaction)
  * DBLP
* baseline
  * SkipGram models
    1. LINE
    2. DNE
  * GNNs (Retrained)
    1. GraphSAGE
    2. GCN
  * GNNs (Incremental)
    1. PretrainedGNN (첫 time step때만 학습되고 이후로는 학습하지 않음)
    2. SingleGNN (각 time step마다 한 번씩 학습)
    3. OnlineGNN (Continual Learning setting, without knowledge consolidation)
    4. GNN-EWC
    5. GNN-ER
    6. DiCGRL
    7. TWP
    8. ContinualGNN
  * `SGNN-GR`
여기서 Retrained `GNN`은 각 time step마다 Graph **전체**를 학습시킨 것으로, Continual Learning model 성능의 upper bound라고 생각하면 됩니다. Incremental `GNN`이 Continual Learning model들이라고 생각하시면 됩니다.
### **Result**
* Overall Results
위의 data를 사용한 실험의 결과는 아래와 같습니다. 저자들은 average Macro/Micro-F1를 성능 평가 지표로 사용했습니다.
![image](https://user-images.githubusercontent.com/99710438/195345047-bd69d686-e6d3-4ea6-ab81-4baff5f95e1e.png)
말씀드린대로, `LINE`, `RetrainedGCN`, `RetrainedSAGE`는 각 task에서 그래프 **전부**를 사용해서 Continual Learning setting의 성능을 상회합니다. 하지만 저자들의 `SGNN-GR`의 성능 또한 Retrained model과 유사한 것으로 보아 generator가 꼭 필요한 sample들만 생성해줬음을 알 수 있습니다.
* Analysis of Catastrophic Forgetting
앞서 `catastrophic forgetting`을 방지하는 것이 Continual Learning에서 가장 중요한 포인트 중 하나라고 말씀드렸는데, 저자들의 모델은 얼마나 이전의 정보를 잘 기억했는지 보겠습니다.
![image](https://user-images.githubusercontent.com/99710438/195346345-51daec92-bc57-4c36-a6d5-a4b883a6aeb2.png)
왼쪽 (a) 그림은 Cora dataset에서 모델이 14 step을 가는동안 0번째 task를 얼마나 잘 기억하는지 보여주는 그래프이고, 오른쪽 (b) 그림은 6번째 task를 얼마나 잘 기억하는지 보여주느 그래프입니다.
`OnlineGNN`은 이전 task의 정보를 거의 저장하지 못하는 것을 확인할 수 있고, 저자들의 방법론이 `GNN-ER`보다 더 이전 task의 지식을 잘 보존하는 것을 볼 수 있습니다.
* Anaylsis of Generative Model
그렇다면 과연 저자들이 `replay buffer`를 Generative Model로 생성한 것은 옳은 선택이었을까요?
![image](https://user-images.githubusercontent.com/99710438/195347882-15c5016a-3f55-4799-892a-4e73935493b6.png)
그림 (a) 는 실제 그래프의 label당 node 개수(파란색)와 Generative Model로 생성된 label당 node 개수(빨간색)을 보여줍니다. Generative Model이 실제 그래프의 label 분포와 굉장히 유사하게 node를 생성하고 있음을 보여줍니다.
또한 오른쪽 그림 (b) 는 generated 된 데이터를 보여주는데, 다양한 topological 정보를 담고 있음을 볼 수 있습니다.
* Ablation Study
![image](https://user-images.githubusercontent.com/99710438/195348924-c5e2fe7f-5238-4acb-a127-ba4bd18bdfbc.png)
마지막으로, 저자들은 `SGNN-GR`의 두 part들이 얼마나 성능 향상에 도움을 주는지 ablation study를 통해 Cora, Citeseer에서 확인했습니다.
여기서 Non-Affected는 새롭게 추가된 node들만 고려하고, 그로 인한 affected node들은 고려하지 않은 모델입니다. 또한 Non-Generator는 모든 affected node를 찾아 다시 학습시키지만, generator는 쓰지 않은 모델입니다.
당연하게 `SGNN-GR`이 가장 좋은 성능을 보이는 것을 확인할 수 있습니다.
## **5. Conclusion**
> **Summary**
이 논문에서는 지속적으로 들어오는 Graph 데이터를 학습하는 데, Generative Model을 사용해 이전에 학습했던 그래프와 비슷한 그래프를 계속 생성해 새로운 데이터와 함께 학습시킵니다.
기존 replay based Continual Learning은 task가 진행됨에 따라 `replay buffer`에 그래프의 일부를 저장하고, task가 많이 늘어나면 그에 따라 요구되는 메모리도 커지는데 비에, Generative Model로 그때그때 `replay buffer`를 생성해서 메모리 효율을 높였습니다.
단순히 메모리 효율을 높인 것에 그치지 않고, 새롭게 등장하는 패턴은 적극적으로 학습하면서 불필요해 보이는 패턴은 줄이도록 학습해서 단순한 Continual Learning을 보완했습니다.
그 사이사이에 `Random Walk`가 아니라 `Random Walk with Restart`를 쓴 것과 같은 디테일, 본인들이 주장하는 모델의 장점을 잘 보여주는 알찬 실험들까지, 좋은 연구인 습니다.
이 논문 뿐만 아니라 Continual leanring에서 Generative Model은 중대한 역할을 할 것으로 보이며 관련 연구들이 꼭 필요할 것으로 보입니다.
> **개인적인 생각**
**올게 왔구나**
본 논문은 Graph Neural Network에서의 Continual Learning에 Generative Model을 접목시킨 방법입니다. 사실 이 논문이 나오는 것은 시간문제라고 생각하던 찰나에 역시나 등장했습니다.
이미 Continual Learning에 Generative Model을 접목시킨 연구는 꽤 오래전에(AI 연구의 속도가 매우 빠른 것을 감안하면) 등장했지만, GNN에 접목된 것은 없었기 때문이죠.
관련 연구를 하시는 분들은 아시겠지만, 이 논문이 novelty가 엄청 높다거나, 기존의 상식을 깨는 굉장한 발견을 한 논문이라기 보단.. (**분명히 좋은** 논문입니다, 오해금지)
가장 큰 contribution은 특정 분야에서 처음 시도된 연구, 적절한 시기에 등장한 연구인 것 같습니다. Novelty만을 좇는게 아니라, trend에 맞는 연구를 하는 능력도 필요해 보입니다.
우리도 최신 논문을 잘 follow up 하는 '트렌디한' 연구자가 되도록 합시다.
***
## **Author Information**
* Wonjoong Kim
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Nerual Network, Continual Learning
  * Contact: wjkim@kaist.ac.kr
## **Reference & Additional materials**
* Github Implementation
  * None
* Reference
  * [[AAAI-21] Overcoming catastrophic forgetting in graph neural networks with experience replay](https://ojs.aaai.org/index.php/AAAI/article/view/16602)
  * [[NIPS-17] Continual learning with deep generative replay](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)
