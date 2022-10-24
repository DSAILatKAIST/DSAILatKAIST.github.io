---
title:  "[ICLR 2020] Graph Information Bottleneck for Subgraph Recognition"
permalink: Graph_Information_Bottleneck_for_Subgraph_Recognition.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Graph Information Bottleneck for Subgraph Recognition
## **1. Problem Definition**
> **Classification 작업에 중요한 역할을 하는 압축된 데이터를 추출한다.**  


레이블을 기반으로 분류하는 작업은 다양한 분야에서 적용될 수 있고 `deep learning` 학습에서 근본적인 문제라고 할 수 있습니다. 그러나 실제 데이터에서는 분류작업에 관계없는 노이즈 정보가 포함되어 있을 가능성이 높으며, 이것은 원 데이터에서는 추가적인 정보를 제공하여 고유한 특성을 보유하도록 하지만, 실제로 분류작업을 하는데 있어 부정적인 영향을 미칩니다. 이러한 문제에 기인하여 분류 작업에 결정적인 역할을 하는 압축된 정보를 인식하도록 하는 문제가 제안되었습니다. 예를 들어 원자를 node로 정의하고 원자간 결합을 edge로 정의한 분자 그래프에서 분자의 `functional group`를 나타내는 subgraph를 추출하는 것을 목표로 할 수 있습니다. 
  
![image](https://distill.pub/2021/gnn-intro/graph_xai.bce4532f.png)

 <br/> <br/>
## **2. Motivation**  

> **Graph 데이터를 Information Bottleneck의 관점에서 접근한다.**   

subgraph 인식이라는 문제가 중요한 과제로 대두되면서, 그래프의 label을 예측하는데 있어서, 정보 손실을 최소화하면서 압축된 subgraph를 어떻게 추출할 수 있을지에 대해서 개발하는 작업이 지속되고 있습니다. 최근 `Information Bottleneck(IB)`이라는 정보 이론 분야에서 이러한 문제에 대해서 다루고 있습니다. 이는 그래프 label의 정보를 가장 잘 담고 있는 정보를 유지한 채로 원본 데이터에서 압축된 방식으로 추출하는 것을 목표로 하고 있습니다. deep learning으로 강화된 IB는 컴퓨터 비전, 강화학습, 자연어 처리 분야에 적용되어, 다양한 분야에서 적절한 feature를 학습할 수 있도록 하는데 성공하였습니다.   
그러나 현재의 IB 방법은 상호간에 관계적인 정보와 이산적인 데이터를 담고 있는 그래프 데이터를 처리하는데 어려움이 있습니다. IB가 정보 손실을 최소화하면서 원본 그래프의 하위 그래프와 같은 불규칙한 그래프 데이터를 압축하는 것은 여전히 어려운 일이라고 할 수 있습니다.   
따라서 본 모델에서는 앞서 언급된 subgraph graph 인식 문제를 해결하기 위해서 그래프 데이터에 IB 원리를 발전시켜서, 새로운 원리인 `Graph Information Bottleneck(GIB)` 방법을 제안합니다. 기존의 IB는 숨겨진 임베딩 공간에서 최적인 representation을 학습하여 main task에 유용한 정보를 추출하는 한편, GIB에서는 graph level에서 중요한 subgraph를 추출하도록 합니다.  

 <br/> <br/>
## **3. Method**
> **Preliminaries**
  
논문에서 제안한 방법론을 이해하기 위해서 몇 가지 `Notation`과 `GCN`의 개념을 소개하겠습니다.

$N$개의 그래프로 구성된 집합 $$\lbrace ( \mathcal{G}_ 1, Y_ 1),\dots,(\mathcal{G}_ N, Y_N) \rbrace$$에서 $$\\mathcal{G}_ n$$은 n번째 그래프를 나타내고, $$Y_n$$는 n번째 그래프에 해당하는 레이블을 나타냅니다.   
$$\mathcal{G}_ n=(\mathbb{V},\mathbb{E}, A, X)$$에서 해당 그래프는 속하는 노드 집합 $$\mathbb{v}=\lbrace V_i|i=1,\dots, M_n \rbrace$$, edge 집합 $$\mathbb{E}=\lbrace (V_i, V_j)|i>j; V_i,V_j \text{ is connected} \rbrace$$, 인접행렬 $$A\in \lbrace 0,1 \rbrace^\lbrace M_n\times M_n \rbrace$$, feature 행렬 $$X\in \mathbb{R}^{ M_n\times d}$$로 구성되어 있습니다.   
$$\mathcal{G}_ {sub}$$는 특정 subgraph를 나타내고,  $$\overline{\mathcal{G}}_ {sub}$$는 $$\mathcal{G}_ {sub}$$를 제외한 나머지 부분을 의미합니다. $$f:\mathbb{G} \rightarrow \mathbb{R} / [0,1,\cdots,n] $$는 그래프에서 실수값으로 mapping하는 함수를 의미하고, 여기서 $$\mathbb{G}$$는 input graph의 도메인입니다. 

<br/> <br/>

> **Graph Convolutional Network** 
  
Graph Convolutional network(GCN)은 그래프 분류 작업에 널리 사용됩니다. node feature $X$와 인접행렬 $A$를 가지는 그래프 $\mathcal{G} = (\mathbb{V},\mathbb{E})$ 를 가정할 때, GCN은 다음의 과정을 통해 노드 임베딩 $X^{'}$을 도출합니다.

$ X^{'} = \mathrm{GCN}(A,X;W) = \mathrm{ReLU}(D^{-\frac{1}{2}}\hat{A}D^{-\frac{1}{2}}W) $

여기서 $D$는 노드의 차수를 담은 대각 행렬이고, $W$는 모델 파라미터를 의미합니다.

최근에는 그래프의 계층적 구조를 활용하여 다양한 graph pooling 방법을 시도하고 있습니다. 결과에 영향을 미치는 서로 다른 노드의 중요성을 활용하기 위해 self-attention 메커니즘을 통해 graph pooling 방식을 향상시킵니다. 마지막으로, 그래프 임베딩 $E$은 앞에서 얻은 노드 임베딩 $X^{'}$에 normalize된 attention score를 곱하여 얻을 수 있습니다.

$ E = \mathrm{Att}(X^{'}) =\mathrm{ softmax}(\Phi_{2}\mathrm{tanh}(\Phi_{1} X^{'T})) X^{'} $

여기서 $\Phi_{1}$와 $\Phi_{1}$은 self-attention의 모델 파라미터입니다.

<br/> <br/>
> **Graph Information Bottleneck**

이제 본격적으로 Graph Information Bottleneck(GIB)의 과정에 대해서 살펴보겠습니다. 먼저 Graph Information Bottleneck 현상과 IB subgraph를 정의합니다.  

![image](http://snap.stanford.edu/gib/venn.png)  

Information Bottleneck 현상의 원리를 일반화하여 불규칙하고 복합적인 그래프의 정보 표현을 학습하며, 이를 통해 그래프 정보 병목 현상(GIB)을 도출합니다.  
그래프 $\mathcal{G}$와 레이블 $Y$가 있을 때, GIB는 가장 유용하지만 압축된 representation $Z$를 탐색합니다. 탐색 과정은 아래와 같습니다. 

$ \max_{Z}{I(Y,Z)}  \text{ s.t. } I(\mathcal{G} ,Z) \leq I_{c} $

여기서 $I_c$는 $\mathcal{G}$와 $Z$ 사이의 정보량의 제한을 나타내기 위해 사용됩니다. 즉, 압축된 형태로 나타내는 것을 목적으로 하기 때문에 $\mathcal{G}$와 $Z$ 사이의 Mutual Information을 최소화하는 방식으로 최적화가 진행됩니다. 여기에서 Lagrange multiplier $\beta$를 도입하여, 제약 조건을 제거합니다.

$ \max_{Z}{ I(Y,Z) - \beta I(\mathcal{G},Z) } $

위의 식은 Graph Information Bottleneck의 핵심 과정을 나타냅니다. 여기서, subgraph 인식에서는 그래프 속성 측면에서 중요한 정보 손실을 최소화하면서 정보를 최대한 압축하는데에 집중하고 있습니다. 왼쪽의 항은 `prediction`과 관련된 항이고, 오른쪽의 항은 `compression`과 관련된 항입니다.
최종적으로 도출된 핵심 subgraph(IB-subgraph)는 유용하면서도 최소한의 그래프를 담아야 하고, 그 결과는 다음과 같이 표현될 수 있습니다.  

$ \max_{\mathcal{G}_ {sub} \in \mathbb{G}_ {sub}} I(Y,\mathcal{G}_{sub}) - \beta  I(\mathcal{G}, \mathcal{G} _{sub})$

이와 같이 도출된 IB-subgraph는 `그래프 분류 개선`, `그래프 해석`, `그래프 잡음 제거` 등 여러 그래프 학습 과제에 적용할 수 있습니다. 그러나 위의 식의 GIB 목적함수는 mutual information과 그래프의 이산적인 특성으로 인해서 최적화하기가 어렵습니다. 그래서 이러한 목적함수를 최적화하기 위해 subgraph를 도출하는 방법에 대한 접근방식이 추가적으로 필요합니다.

<br/> <br/>  
> **Graph Information Bottleneck의 목적함수 최적화 과정**  
* Maximization of $I(Y,\mathcal{G}_ {sub})$  
위의 GIB 목적함수는 2개의 부분으로 구성됩니다. 먼저 식의 첫 번째 항 $I(Y,\mathcal{G}_ {sub})$를 살펴봅니다. 이 항은 $\mathcal{G}_ {sub}$ 와 $Y$ 간의 연관성을 측정하는 부분입니다. 다시 말해, 타겟 레이블인 $Y$와 $\mathcal{G}_ {sub}$ 간의 mutual information을 최대화함으로써 $\mathcal{G}_ {sub}$가 $Y$에 관한 정보를 최대한 보존하도록 하고 있습니다. $I(Y; \mathcal{G}_ {sub})$는 다음과 같이 확장할 수 있습니다.  

$ I(Y,\mathcal{G}) = \int p(y,\mathcal{G}_ {sub}) \log{p(y \vert\mathcal{G}_ {sub})} dy d \mathcal{G}_ {sub} + \mathrm{H}(Y)  $  

$H(Y)$는 $Y$의 엔트로피으로 고정된 값이므로 무시합니다. 또한 
$$p(y,\mathcal{G}_ {sub}) \approx \frac{1}{N} \sum_{i=1}^{N} \delta_ {y_ i}(y) \delta_ {\mathcal{G}_ {sub, i}}(\mathcal{G}_ {sub})$$와 같이 
$p(y,\mathcal{G}_ {sub})$를 근사할 수 있습니다. 여기서 $$\mathcal{G}_ {sub}$$는 출력 하위 그래프이고 $$Y$$는 그래프 레이블입니다. 실제 사후 확률 $$p(y|\mathcal{G}_ {sub})$$를 variational approximation $$q_{\phi_{1}}(y|\mathcal{G}_ {sub})$$으로 대체함으로써, 우리는 위의 식에서 첫 번째 항의 tractable한 최소값을 다음과 같이 얻을 수 있습니다.  

$ 
I(Y,\mathcal{G}_ {sub}) \geq \int p(y,\mathcal{G}_ {sub}) \log{q_{\phi_{1}}(y|\mathcal{G}_ {sub})} dy \ d\mathcal{G}_ {sub} \\
\approx \frac{1}{N} \sum_{i=1}^{N} q_{\phi_{1}}(y_{i}|\mathcal{G}_ {sub_{i}}) =: -\mathcal{L}_ {cls}(q_{\phi_{1}}(y|\mathcal{G}_ {sub}),y_{gt}) 
$


여기서 $$y_{gt}$$는 그래프의 groundtruth 레이블입니다. 위의 식은 Y와 Gsub 사이의 classification loss를 $\mathcal{L}_ {cls}$로 최소화함으로써 $I(Y;\mathcal{G}_ {sub})$를 최대화한다는 것을 나타냅니다. 직관적으로 살펴보아도 $\mathcal{L}_ {cls}$를 최소화하여 분류 정확도를 높인다면, subgraph가 그래프 레이블을 정확하게 예측할 수 있다는 것을 의미합니다. 실제로 classification $Y$에 대해서는 cross entropy loss를 사용하고, regression Y에 대해서는 mean square error(MSE)를 선택합니다.

<br/> <br/>
*  Minimization of $I(\mathcal{G},\mathcal{G}_ {sub})$   
이제 두 번째 항 $I(\mathcal{G},\mathcal{G}_ {sub})$에 대해서 알아보겠습니다. 이는 기존의 입력 그래프 $\mathcal{G}$와 subgraph $\mathcal{G}_ {sub}$ 간의 mutual information을 최소화함으로써, $\mathcal{G}_ {sub}$이 $\mathcal{G}$의 관한 정보를 너무 많이 포함하지 않도록 합니다. 이는 $\mathcal{G}_ {sub}$에 prediction의 정확도를 높이기 위해 너무 많은 정보 만을 담고 있으면, subgraph를 인식하는 작업에 대해 무의미해지기 때문에, 어느 정도 샘플링을 통해서 적절한 subgraph를 추출하는 작업이 필수적이라고 할 수 있습니다.     
최적화 과정을 진행하기 위해 KL-divergence의 Donsker-Varadhan representation을 적용함으로써,  $I(\mathcal{G},\mathcal{G}_ {sub})$을 다음과 같이 나타낼 수 있습니다.  

$
I(\mathcal{G},\mathcal{G}_ {sub}) = \sup \limits_{f_{\phi_2}:\mathbb{G} \times \mathbb{G} \rightarrow \mathbb{R}} \mathbb{E}_ {\mathcal{G},\mathcal{G}_ {sub}\in p(\mathcal{G},\mathcal{G}_ {sub})}f_{\phi_{2}}(\mathcal{G},\mathcal{G}_ {sub})-\log{\mathbb{E}_ {\mathcal{G} \in p(\mathcal{G}),\mathcal{G}_ {sub}\in p(\mathcal{G}_ {sub})}e^{f_{\phi_{2}}(\mathcal{G},\mathcal{G}_ {sub})}}
$

여기서 $f_{\phi_{2}}$는 앞에서 말씀드렸다시피 그래프 집합에서 실수 집합으로 매핑되는 네트워크입니다. 이 식의 유도 과정에 집중하기보다는 결과 해석과 모델 설계 방향에 대해 말씀드리고자 합니다.  
앞서 나온 식을 바탕으로 설계된 모델의 아키텍처는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/67723054/196002229-221dee3c-a226-4541-a28f-d65f5232380a.PNG)

먼저 GNN을 사용하여 $\mathcal{G}$와 $\mathcal{G}_ sub$ 로부터 임베딩을 추출하고, $\mathcal{G}$와 $\mathcal{G}$ 임베딩을 concat시켜서, MLP의 입력으로 설정합니다. 그 결과 output은 subgraph $\mathcal{G}$와 나머지 부분 $\overline{\mathcal{G}}_ {sub}$에 해당하는지의 여부를 결정하는 노드 할당 행렬을 도출하게 됩니다. $p(\mathcal{G}_ {sub})$에 대한 샘플링 방법과 함께 $I(Y,\mathcal{G}_ {sub})$에 대한 다음과 같은 최적화 문제에 도달하게 됩니다.   
  
$
\max_{\phi_{2}} \mathcal{L}_ {\mathrm{MI}}(\phi_{2},\mathcal{G}_ {sub}) =  \frac{1}{N}\sum_ {i=1}^{N}f_{\phi_{2}}(\mathcal{G}_ {i},\mathcal{G}_ {sub,i})-\log{\frac{1}{N}\sum_{i=1,j\neq i}^{N}e^{f_{\phi_{2}}(\\mathcal{G}_ {i},\mathcal{G}_ {sub,j})}}
$  
  
그래프 데이터의 Mutual Information 근사과정을 통해서 $I(Y,\mathcal{G}_ {sub})$의 최대화 과정과 $I(\mathcal{G},\mathcal{G}_ {sub})$의 최소화 과정을 결합하여 최종적으로 다음과 같은 최적화 과정을 설계할 수 있습니다.  inner loop에서는 IB-subgraph를 통해 나온 임베딩을 활용하여 $I(\mathcal{G},\mathcal{G}_ {sub})$를 최소화하는 과정을 거칩니다. $I(\mathcal{G},\mathcal{G}_ {sub})$에 대한 추정이 이루어지면, outer loop에서 mutual information, classification loss, connectivity loss를 사용하여 GIB의 목적함수를 최적화합니다.

$
\min \limits_{\mathcal{G}_ {sub},\phi_ {1}} \mathcal{L}(\mathcal{G}_ {sub},\phi_{1},\phi_{2}^{* }) = \mathcal{L}_ {cls}(q_ {\phi_ {1}}(y|\mathcal{G}_ {sub}),y_ {gt}) + \beta \mathcal{L}_ {MI}(\phi_ {2}^{* },\mathcal{G}_ {sub}) 
$ 

$
\text{ s.t. }  \phi_{2}^{*} = \arg\max_ {\phi_ {2}}\mathcal{L}_ {\mathrm{MI}}(\phi_{2},\mathcal{G}_ {sub}) 
$ 


먼저 inner loop에서 $\phi_{2}$를 $\phi_{2}^{* }$로 최적화하고, 이후에 outer loop에서 $\phi_{2}^{* }$를 활용하여 $$I(\mathcal{G},\mathcal{G}_ {sub})$$에 대한 minimization 작업을 진행하고 classification loss $\mathcal{L}_ {cls}$를 기반으로 $Y$와 $\mathcal{G}$간의 mutual information을 최대화시킵니다. 이 과정에서 $\phi_{1}$과 $\mathcal{G}_ {sub}$가 IB-subgraph를 생성하도록 최적화하게 됩니다.

<br/> <br/>
> **Subgraph Generator**  

입력 그래프 $\mathcal{G}$에 대해 노드가 $\mathcal{G}_ {sub}$나  $\overline{\mathcal{G}}_ {sub}$ 중 어디에 속할지 나타내는 노드 할당 행렬  $\textbf{S}$를 생성하여 subgraph를 발생시킵니다. 그런 다음 $\mathcal{G}_ {sub}$ 또는  $\overline{\mathcal{G}}_ {sub}$에 속할 확률을 
예를 들어, $\textbf{S}$의 i번째 행은 2차원 벡터 $\textbf{[} p(V_{i}\in \mathcal{G}_ {sub}|V_{i}), p(V_{i} \in \overline{\mathcal{G}}_ {sub}|V_{i})\textbf{]}$로 구성됩니다. $l$- layer GNN을 사용하여 각각의 노드 임베딩을 얻는데, 그림에서 각 노드의 임베딩이 파란색과 초록색으로 구성되어 있습니다. 이후에 '노드 할당 행렬 $\textbf{S}$'를 도출하기 위해서 MLP를 사용합니다. 따라서 `subgraph generator`의 작동 알고리즘은 아래와 같습니다.

$
X^{l} = \mathrm{GNN}(A,X^{l-1};\theta_{1}), \quad 
S = \mathrm{MLP}(X^{l};\theta_{2})
$

$\textbf{S}$는 $n\times 2$ 행렬이고, 여기서 $n$은 노드의 수입니다. $\textbf{S}$가 학습되면, 노드 할당 행렬의 구성요소가 0과 1로 구성되게 되고, 첫번째 열은 그래프 레이블을 예측하는데 사용되는 $\mathcal{G}_ {sub}$의 representation에 해당하는 초록색 노드 임베딩이고, 두번째 열은 그 나머지 부분 $\overline{\mathcal{G}}_ {sub}$의 representation에 해당하는 파란색 노드 임베딩입니다. 최종적으로 $\textbf{S}^{T}X^{l}$의 첫 번째 열을 가져옴으로써 $\mathcal{G}_ {sub}$의 임베딩을 얻을 수 있습니다.

<br/> <br/>
> **Connectivity Loss**  

위에서 살펴본 목적함수의 최적화 과정으로는 모델이 모든 노드를 $\mathcal{G}_ {sub}$나 $\overline{\mathcal{G}}_ {sub}$에 할당하거나, $\mathcal{G}_ {sub}$의 representation에 중복된 노드로부터 불필요한 정보를 포함할 수 있습니다. 이를 예방하기 위해서 `connectivity loss` $\mathcal{L}_ {con}$를 도입합니다.  

$ \mathcal{L}_ {con} = \vert \vert \mathrm{Norm}(S^{T}AS)- I_2\vert \vert _ F $

여기서 $$\mathrm{Norm}$$은 행방향의 normalization을 나타내고  $$\vert \vert \cdot \vert \vert_ F$$은 Frobenous norm을 나타내며, $I_2$는 $2\times 2$의 단위행렬을 나타냅니다. 이 식이 가지는 의미를 해석하기 위해서 예를 들어 설명하겠습니다. $S^{T}AS$의 (1,1)의 원소를 $a_11$, (1,2)의 원소를 $a_12$라고 할 때, 

$
a_{11} = \sum_{i,j} A_{ij}p(V_{i}\in \mathcal{G}_ {sub}|V_{i})p(V_{j}\in \\mathcal{G}_ {sub}|V_{j}),
$

$
a_{12} = \sum_{i,j} A_{ij}p(V_{i}\in \mathcal{G}_ {sub}|V_{i})p(V_{j}\in \overline{\mathcal{G}}_ {sub}|V_{j})
$

로 나타낼 수 있습니다. 즉, $\mathcal{G}$에 포함된 2개의 노드는 서로 연결되어야 한다는 것이고, $\mathcal{G}$에 포함된 노드와 $\overline{\mathcal{G}}$에 포함된 노드는 서로 연결되지 않아야 한다는 것입니다.  

구체적으로 살펴보겠습니다. $\mathcal{L}_ {con}$를 최소화시키면, $\mathrm{Norm}(S^{T}AS)$의 (1,1) 원소는 항등행렬 $I_2$의 (1,1) 원소인 1에 가까워져야 하기 때문에 $\frac{a_{11}}{a_{11}+a_{12}}$가 1로 수렴하게 됩니다. 이는 노드 $V_i$가 $\mathcal{G}_ {sub}$에 속한다면, $V_i$의 이웃 노드 $\mathcal{N}(V_{i})$도 $\mathcal{G}_ {sub}$에 속할 확률을 높이는 작업입니다. 또한  $\mathcal{L}_ {con}$를 최소화시키면, $\mathrm{Norm}(S^{T}AS)$의 (1,2) 원소는 항등행렬 $I_2$의 (1,2) 원소인 0에 가까워져야 하기 때문에 $\frac{a_{12}}{a_{11}+a_{12}}$가 0으로 수렴하게 됩니다. 이는 $\mathcal{G}_ {sub}$에 속하는 노드와 $\overline{\mathcal{G}}$에 속하는 노드 사이에 연결을 제거하는 작업입니다. 이를 통해서 $\mathcal{L}_ {con}$는 노드 샘플링을 적절히 수행할 뿐만 아니라, 훨씬 더 압축된 subgraph를 도출할 수 있습니다.  

결국 $\mathcal{L}_ {con}$을 반영한 최종적인 loss 함수는 아래와 같습니다.  

$
\min \limits_{\mathcal{G}_ {sub},\phi_ {1}} \mathcal{L}(\mathcal{G}_ {sub},\phi_{1},\phi_{2}^{* }) = \mathcal{L}_ {con}(g(\mathcal{G};\theta))  + \mathcal{L}_ {cls}(q_ {\phi_ {1}}(y|\mathcal{G}_ {sub}),y_ {gt}) + \beta \mathcal{L}_ {\rm MI}(\phi_ {2}^{* },\mathcal{G}_ {sub}) 
$

$
\text{ s.t. }  \phi_{2}^{*} = \arg\max_ {\phi_ {2}}\mathcal{L}_ {\mathrm{Cancel changesMI}}(\phi_{2},\mathcal{G}_ {sub}) 
$


<br/> <br/>

## 4. Experiment  

실험은 총 3가지 실험을 진행하였고, `graph classification(분류)`과 `graph interpretation(해석)`, `graph denoising(노이즈 제거)`의 관점에서 본 논문에서 제안된 GIB(Graph Information Bottleneck)을 평가합니다. 

> **Graph Classification**  

GIB 알고리즘은 subgraph $\mathcal{G}_ {sub}$의 representation을 활용하여 $\mathcal{L}_ {cls}$을 기반으로 graph classification task를 진행합니다. 이 과정에서 GCN, GAT, GIN, GraphSAGE를 포함한 여러가지 backbone에 GIB를 연결합니다. 이 모델들을 다양한 aggregation 방식들의 모델과 비교합니다.  
* `pooling 기반 aggregation` : SortPool, ASAPool, DiffPool, EdgePool, AttPool
* `mean/sum 기반 aggregation` : GCN, GraphSAGE, GIN, GAT

`MUTAG`, `PROTEINS`, `IMDB-BINARY`, `DD` 총 4가지의 데이터셋으로 graph classification을 진행합니다. 그 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/67723054/196009252-9bdf3ea0-80a4-459a-8aff-1e567e1eacce.PNG)  

위의 결과에서 논문에서 제안한 방법과 baseline을 종합적으로 평가하였습니다. 다양한 backbone에서 GIB를 학습시키고, subgraph에서 representation을 aggregation하여 성능을 측정한 결과, 1개의 데이터셋(MUTAG)을 제외하고 나머지 데이터셋에서 가장 좋은 성능을 낸다는 것을 확인할 수 있습니다. 이는 GIB가 그래프 분류의 핵심적인 구조를 추출할 수 있으며 이것이 그래프 분류 작업에 있어서 결정적인 역할을 할 수 있다는 것을 의미합니다.

<br/> <br/>

> **Graph Interpretation**

GIB의 가장 두드러진 특징은 `interpretation` 과정을 진행할 수 있다는 것입니다. 이는 그래프의 핵심 특성을 반영할 수 있는 하위 구조를 찾는 작업입니다. 이 interpretation 작업과 classification 작업을 동시에 진행함으로써 많은량의 유용한 정보를 도출해낼 수 있다는 점에서 큰 의미가 있습니다.  

본 실험에서는 `GIB`와 `attention` 메커니즘을 비교합니다. 즉, graph prediction에 대한 node의 정보를 aggregation합니다. attention score를 `50%(Att05)`, `70%(Att07)`로 설정하여 해석가능한 subgraph를 추출합니다. 그래프와 그에 해당하는 subgraph 사이에 나타나는 예측 차이에 대한 평균과 분산을 측정합니다. 공정한 비교를 위해서 모든 방법에 대해 GCN backbone을 사용하였고, GIB에서 사용된 여러 Loss 함수가 성능 향상에 기여하는지의 여부를 알아보기 위하여 $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 각각 제외하여 추가적인 학습을 진행하였습니다.  

250000여개의 분자를 포함하는 `ZINC dataset`을 사용하고, 4가지 분자의 특성에 대해서 그래프 해석 작업을 진행합니다. QED는 0에서 1까지의 범위로 약물의 유사성을 측정합니다. DRD2는 0에서 1까지의 범위로 분자가 도파민 수용체에 대해서 활성일 확률을 나타냅니다. HLM-CLint와 MLM-CLint는 시험관 내에서 인간과 쥐의 간 마이크로솜의 대사 안정성을 추정한 값입니다. 각 작업에 대해서 training set, validation set, test set의 비율을 85%, 5%, 10%로 분할합니다.  

![image](https://user-images.githubusercontent.com/67723054/196010188-1a19c246-6835-4238-a857-5ac9a21ed0bf.PNG)

위의 결과는 graph interpretation 작업에 대한 정량적인 성능을 보여줍니다. GIB에 의해 도출된 subgraph에 대해서 전체 모델 중에서 가장 작은 차이를 보여주고 있기 때문에, IB-subgraph가 입력 그래프와 가장 유사한 특성을 가지기 때문에 GIB가 높은 그래프 해석 능력을 가진다는 것을 확인할 수 있습니다. 또한 $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 각각 제거한 방법의 결과도 함께 제시하고 있습니다. GIB가  $\mathcal{L}_ {con}$와 $\mathcal{L}_ {MI}$를 제외한 성능보다 우수하기 때문에 모델의 모든 부분이 성능 개선에 기여한다는 것을 알 수 있습니다.  

![image](https://user-images.githubusercontent.com/67723054/196010588-594f8ace-fb01-4038-808d-abd7a439aba4.PNG)  

위의 그림은 graph interpretation 작업에 대해서 한눈에 보기 쉽게 나타나는 정성적인 결과입니다. GIB가 HLM-CLint와 LML-CLint의 관점에서 왼쪽의 원래 그래프와 가장 유사한 특성을 보여준다는 것을 알 수 있습니다. 게다가 어떤 subgraph가 그러한 결과에 기인하였는지 붉은색의 부분으로 표시되어있습니다. 이를 통해서 어떤 subgraph가 모델의 전체 특성을 지배하는지 비교적 정확하게 알 수 있습니다.  

<br/> <br/>

> **Graph Denoising**  

여기에서는 앞에서 사용한 MUTAG dataset에서 각 그래프에 대해 30%의 추가적인 edge(노이즈)를 추가하여 `synthetic dataset`을 생성합니다. 이 synthetic dataset을 활용하여 GIB의 classification accuracy를 GCN, DiffPool과 함께 비교합니다. 여기에서 training set, validation set, test set을 각각 70%, 5%, 25%로 설정하였습니다.

![image](https://user-images.githubusercontent.com/67723054/196011179-304f2b12-e172-40ae-950e-47ce09ac5d04.PNG)

위의 결과는 노이즈가 있는 그래프에서 classification 성능을 보여주고 있습니다. GIB는 IB-subgraph의 denoising 능력은 다른 baseling보다 훨씬 뛰어나다는 것을 알 수 있습니다. 

<br/> <br/>

## **5. Conclusion**
> **Summary**

본 논문에서는 최대한의 정보를 제공하면서도 압축된 부분 그래프를 도출하기 위해 subgraph 인식 문제를 연구하였습니다. 이러한 subgraph를 IB-subgraph로 정의하고 IB-subgraph를 효과적으로 발견하기 위한 `GIB(Graph Information Bottleneck)` 프레임워크를 제안하였습니다. 정보이론 분야에서 연구되었던 Information Bottleneck 분야를 Graph Neural Network에 처음으로 접목시켜서 Graph 데이터에서 functional group과 같은 중요한 하부구조를 인식하는데 도움을 주고, 이는 분류 작업 뿐만 아니라 해석능력을 제공하는데 큰 장점을 가집니다. sampling 방법을 최적화시키기 위해서 목적함수를 `prediction` 항과 `compression` 항으로 구성시켜서 subgraph의 예측 정확도와 일정 정도 이상의 압축성을 보장하여 최적의 subgraph를 도출하도록 합니다. graph classification, graph interpretation, graph denoising 3가지 작업에 있어 GIB의 성능을 평가하였고, 그 결과 IB-subgraph의 우수한 성능을 확인할 수 있었습니다.

<br/> <br/>

> **생각 및 발전 방향**

Information Bottleneck이라는 정보이론 방식을 그래프 데이터로 가져와서 적용했다는 점에서 상당히 인상 깊었는데, 사실 그래프 데이터는 정보가 상호 연결되어 있다는 복합적인 특성으로 인해서 다루기가 까다로움에도 불구하고 최적화 과정을 적절히 활용하여 성능 향상을 이끌어내었습니다.   

사실 기존의 Explainable AI의 경우에는 모델이 작동한 이후에 해석 작업을 거치기 때문에 시간이 더욱 오래 걸릴 뿐만 아니라, 학습된 모델에 대해서 후속적인 작업으로 해석이 들어가기 때문에 좀더 본질적인 해석이 어려울 수 있습니다. 하지만, GIB의 경우에는 분류 작업과 동시에 해석 작업을 진행하기 때문에 상당히 개선적인 모델이라고 할 수 있습니다. 이러한 발전방향은 학습 시간, 해석 능력의 측면에서 실용적인 모델로 발전할 수 있다는 점에서 큰 의미를 가진다고 생각합니다.    

논문의 두드러진 특징은 GIB를 도입함으로써 진행되는 과정과 여러가지 과정에서 발생할 수 있는 문제점들을 논리적으로 풀어나간다는 점에서 흥미로웠고, 이 영향력은 실험 결과에서 그대로 드러났습니다. 제안된 모델을 3가지의 측면으로 성능을 측정하였는데, 여기에서 각 실험 방향에 맞는 데이터셋과 실험방법을 선택하였습니다. classification 부분에서는 널리 사용되는 benchmark dataset을 사용하였고, interpretation은 여러가지 특성을 담고 있고 있어서 해석력을 측정하기 용이한 ZINC dataset을 사용하였으며, graph denoising에서는 실험 의도를 잘 반영할 수 있는 Synthetic dataset을 설계하여 실험을 수행하였습니다. 모델 아키텍처를 구성하고 있는 부분들과 실험 설계 방향에 주목하여 논문을 읽는다면 더욱 유익할 수 있을 것이라고 생각합니다.  

본 모델에서 graph information bottleneck을 샘플링 방법에 좀 더 다양하게 접근한다면 더 높은 성능 향상을 기대할 수 있을 것이라 생각합니다. 최근에 GIB에 noise를 도입하여 추가적인 연구가 진행되었는데, 이 외에도 더 넓은 feature 탐색을 통해서 임베딩을 샘플링하는데 있어서 효과적인 탐색과정을 진행한다면 prediction 과정에 큰 도움이 될 수 있을 것입니다.


<br/> <br/>

## **Author Information**
* Sangwoo Seo
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Nerual Network, Information Bottleneck
  * Contact: tkddn8974@kaist.ac.kr
## **Reference & Additional materials**
* Github Implementation
  * [Code for the paper](https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition)
* Reference
  * [[ICLR-21] Graph Information Bottleneck for Subgraph Recognition](https://arxiv.org/abs/2010.05563)
  
