---
title:  "[ICML 2022] Structure-Aware Transformer for Graph Representation Learning"
permalink: Structure_Aware_Transformer_for_Graph_Representation_Learning.html
tags: [reviews]
---

# ****Structure-Aware Transformer for Graph Representation Learning****

_**Background before reading this review.**_

*Notation

$G = (V, E, \mathbf X)$

-   node $u \in V$
-   node attribute $x_u \in \mathcal X \subset \mathbb R^d$
-   $\mathbf X \in \mathbb R^{n \times d}$




1.  Transformers on Graph
    
    GNN에서는 그래프 구조를 explicit하게 활용하는 반면, transformer는 노드의 attribute를 활용하여 노드들 사이 relation을 나타내는데 활용
    
    Transformer 구성 요소
    
    1.  Self-attention module
        -   input node feature $\mathbf X$가 linear projection을 통해 Query($\mathbf Q$), Key($\mathbf K$), Value($\mathbf V$)로 투영되고, 이를 활용하여 self-attention을 계산할 수 있음
        -   multi-head attention
    2.  feed-forward NN
        -   self-attention의 output이 skipconnection이나 FFN등을 거치면 하나의 transforemer layer를 통과하게됨
2.  Absolute encoding
    
    그래프의 위치적/구조적인 representation을 input node feature에 더하거나 concatenate하여 Transformer의 input으로 사용 : Vanilla transformer의 PE
    
    Graph Transformer에서 자주 사용되는 Positional encoding method들
    
    -   Laplacian PE
    -   Random Walk PE
    
    (-) 노드와 그 이웃들 사이의 structural similarity를 반영하지는 않음
    
3.  Self-attention and kernel smoothing
    
    $\operatorname{Attn}\left(x_v\right)=\sum_ {u \in V} \frac{\kappa_ {\exp }\left(x_v, x_u\right)}{\sum_ {w \in V} \kappa_ {\exp }\left(x_v, x_w\right)} f\left(x_u\right), \forall v \in V$
    
    -   linear value function $f(x) = \mathbf W_ {\mathbf V}x$
    -   $\kappa_ {\exp }$ (non-symmetric) exponential kernel parameterized by $\mathbf W_ {\mathbf Q}, \mathbf W_ {\mathbf K}$
    
    
    $\kappa_ {\exp }\left(x, x^{\prime}\right):=\exp \left(\left\langle\mathbf{W}_ {\mathbf{Q}} x, \mathbf{W}_ {\mathbf{K}} x^{\prime}\right\rangle / \sqrt{d_ {\text {out }}}\right)$
    
    -   $\langle \cdot, \cdot\rangle$ : dotproduct
    -   학습가능한 exponential kernel
    -   (-) only position-aware, not structure-aware encoding

# **1. Problem Definition**

## _**Limitations of GNN**_

1.  limited expressiveness : 최대 1-WL test의 표현력을 가짐
2.  Over-smoothing problem : GNN layer의 수가 충분히 커지면 모든 node representation이 상수로 수렴하게됨
3.  Over-squashing problem : 그래프의 수많은 메세지들이 고정된 길이의 벡터 하나로 압축되어 발생하는 그래프 “bottleneck”으로 인해 멀리 위치한 노드의 메세지가 효율적으로 전파되지 않는 문제

⇒ Beyond neighborhood aggregation!

## _**Transformer**_

-   하나의 self-attention layer를 통해 그래프내의 어떤 노드쌍이든지 그 사이의 상호작용을 확인할 수 있음
-   GNN과 달리 중간 계층에서 structural inductive bias가 발생하지 않아 GNN의 표현력 한계를 해결
-   graph structure info를 얼마나 학습하는지 input node feature에만 structural, positional info를 encoding해 넣음
-   노드에 대한 structural, positional info만 input node feature로 encoding하지만, 그래프 구조 자체에서 학습할 수 있는 정보의 양이 제한됨

> 💡 Goal : 그래프 데이터에 Transformer를 적절히 변형해 적용하여 그래프 구조를 잘 반영하고 높은 표현력을 가지는 Achitecture를 디자인하는 것

# **2. Motivation**


## _**Message passing graph neural networks.**_

최대 1-WL test로 제한된 표현력, over-smoothing, over-quashing

## _**Limitations of existing approaches**_

-   노드들 사이 positional relationship만 encoding하고, strucutral relationship을 직접 encoding하지않음
    -   노드들 사이 structural similarity를 확인하기가 어렵고, 노드들 사이의 structural interaction을 모델링하는데 실패하게됨

ex.

![Untitled](https://user-images.githubusercontent.com/69068083/231114064-4063d3f0-f895-4d7c-a932-eddbeb77de34.png)

G1과 G2에서 최단거리를 활용한 positional encoding을 할경우 node u와 v가 다른 노드들에 대해 모두 같은 representation을 가지게되지만, 그래프의 실제 구조는 다름 → strucure aware에 실패

>💡 Message-passing GNN과 Transformer architecture 각각의 장점을 살려 local, global info를 모두 고려하는 transformer architecture를 제안

## _**Contribution of this paper**_
Q. Transformer구조에 structural info를 어떻게 인코딩할것인가?

A. Structure-aware self attention를 도입한 Structre-Aware Transformer(SAT)

1.  reformulate the self-attention mechanism
    -   kernel smoother
    -   원래 노드 feature에 적용하는 exponential 커널을 확장하여 각 노드가 중심인 subgraph representation을 추출하여 local structure에도 적용
2.  subgraph representation들을 자동적으로 만들어내는 방법론 제안
    -   이를 통해 kernel smoother가 구조적/특성적 유사성을 포착할 수 있게됨
3.  GNN으로 그래프의 subgraph info를 포함하는 node representation을 만들어 기존 GNN에 추가적인 구조 개선 없이도 더 높은 성능을 냄
4.  Transformer의 성능향상이 structure-aware한 측면에서 일어난 것을 증명하고 absolute encoding이 추가된 transfoemr보다 SAT가 얼마나 interpretable한지를 보여줌

# **3. Method**



## _**Structure-Aware Transformer**_

### _1. **Structure-Aware Self-attention**_

position-aware한 structural encoding에 노드들 사이 structural similarity를 포함하기 위해 각 노드의 local structure에 관한 generalized kernel을 추가

각 노드가 중심이되는 subgraph set을 추가함으로써 structure-aware attention은 다음과 같이 정의될 수 있음

$\operatorname{SA-Attn}\left(v\right):=\sum_ {u \in V} \frac{\kappa_ {\text{graph} }\left(S_G(v), S_G(u)\right)}{\sum_ {w \in V} \kappa_ {\text{graph}}\left(S_G(v), S_G(u)\right)} f\left(x_u\right)$

-   $S_G(v)$ : node feature $\mathbf X$와 연관된 $v$를 중심으로하는 subgraph
-   $\kappa_ {\text{graph} }$ : subgraph쌍을 비교하는 kernel

⇒ attribute & structural similarity 모두 표현 가능한 expressive node representation 생성 → table 1

⇒ 동일한 subgraph 구조를 가지는 경우에만 permutation equivariant한 성질을 갖게됨

$\kappa_ {\text {graph }}\left(S_G(v), S_G(u)\right)=\kappa_ {\exp }(\varphi(v, G), \varphi(u, G))$

-   $\varphi(v, G)$ : feature $\mathbf X$를 가지는 node $v$가 중심에 있는 subgraph의 vector representation을 만들어내는 structure extractor
    -   GNN이나 differentiable Graph kernel등 subgraph의 representation을 만들 수 있는 어느 모델이든 될 수 있음
    -   Task/data 특성에 따라 Edge attribute을 활용할 필요가 있는 경우 그에 맞는GNN을 선택하면 됨. 따로 edge attribute을 활용하지는 않고 subgraph extractor에서 활용

_**k-subtree GNN extractor.**_

$\varphi(u, G) = \operatorname{GNN}_G^{(k)}(u)$

-   node u에서 시작하는 k-subtree structure의 representation생성
-   at most 1-WL test
-   작은 k 값이더라도 over-smoothing, over-squashing issue없이 좋은 성능을 내는것을 확인

_**k-subgraph GNN extractor.**_

$\varphi(u, G) = \sum_ {v \in \mathcal N_k(u)} \operatorname{GNN}_G^{(k)}(v)$

-   node u의 representation만을 사용하는데서 나아가 node u가 중심이 되는 k-hop subgraph전체의 representation을 생성하고 활용
-   node u 의 k-hop이웃 $\mathcal N_k(u)$에 대해 각 노드에 GNN을 적용한 node representation을 pooling(논문에서는 summation)
-   More powerful than 1-WL test
-   original node representation과의 concatenation을 통ㅎ structural similarity뿐만 아니라 attributed similarity도 반영

_**Other structure extractors.**_

-   directly learn a number of “hidden graphs” as the “anchor subgraphs” to represent subgraphs
-   domain-specific GNNs
-   non-parametric graph-kernel

### _2. Structure-Aware Transformer_

![Untitled](https://user-images.githubusercontent.com/69068083/231114106-a71006e8-a9e5-44cb-b353-578ec4e09a80.png)

self-attention→ skipconnection → normalization layer → FFN → normalization layer

_**Augmentation on skip connection.**_

$x'_v = x_c +1/ \sqrt {d_v} \operatorname{SA-Attn}\left(v\right)$

-   $d_v$ : node $v$의 degree
-   degree factor를 포함하여 연결이 많은 graph component들이 압도적인 영향을 미치지 않도록함

*graph-level task를 진행해야 할 경우 input graph에 다른 노드와의 connectivity없이 virtual `[cls] `node를 추가하거나, node-level representation을 sum/average 등으로 aggregation

### _3. Combination with Absolute Encoding_

위의 structure aware self-attention에 추가로 absolute encoding을 추가하게 되면 postion-aware한 특성이 추가되어 기존의 정보를 보완하는 역할을 하게된다. 이러한 조합을 통해 성능향상을 확인할 수 있었다.

RandomWalk PE

Absolute PE만 사용할 경우 structural bias가 과도하게 발생하지 않아서 누개의 노드가 유사한 local structure를 갖고 있더라도 비슷한 node representation이 생성되는것을 보장하기 어렵다!

→ Structural, positional sign으로 주로 사용되는 distance나 Laplacian-based positional representation이 노드들 사이의 structural simialrity를 포함하지 않기때문

> 📌 Structural aware attenrion은 inductive bias가 더 강하더라도 노드의 strucutral similarity를 측정하는데 적합하여 유사한 subgraph구조를 가진 노드들이 비슷한 embedding을 갖게하고, expressivity가 향상되어 좋은 성능을 보임


### _4. Expressivity Analysis_

SAT에서는 각노드를 중심으로하는 k-subgraph GNN extractor가 도입되어 적어도 subgraph representation만큼은 expressive하다는 것을 보장


# **4. Experiment**



### _**Experiment setup**_

_**Dataset**_

-   ZINC
-   CLUSTER
-   PATTERN
-   OGBG-PPA
-   OGBG-CODE2

_**Baseline**_

-   _**GNNs**_
    -   GCN
    -   GraphSAGE
    -   GAT
    -   GIN
    -   PNA
    -   Deeper GCN
    -   ExpC
-   _**Transformers**_
    -   Original Transformer with RWPE
    -   Graph Transformer
    -   SAN
    -   Graphormer
    -   GraphTrans

### _**Results**_
**Table1.** SAT와 graph regression, classification task의 sota모델과 비교

- ZINC dataset의 경우 작을수록 더 좋은 성능을 의미하는 MAE(Mean Absolute Error), CLUSTER와 PATTERN의 경우 높을수록 더 좋은 성능을 의미하는 Acurracy가 평가지표로 사용되었음.

![Untitled](https://user-images.githubusercontent.com/69068083/231114155-056893f6-8d16-4a59-b43b-62c76fd482a3.png)

**Table2.** SAT와 OGB데이터셋에서의 sota모델 비교
- OGB dataset의 경우 높을수록 더 좋은 성능을 의미하는 Acurracy, F1 score가 평가지표로 사용되었음.

![Untitled](https://user-images.githubusercontent.com/69068083/231114185-23daa0d6-bc32-4838-93e8-0a6d09a17f7e.png)

**Table3.** structure extractor로 사용한 GNN과의 성능비교. Sparse GNN을 모든 경우에서 outperform하는 것을 확인할 수 있음

![Untitled](https://user-images.githubusercontent.com/69068083/231114223-e6e32dfd-039b-4caa-b123-14e72e9fc867.png)

**Fig3.** ZINC데이터셋에 SAT의 다양한 variant실험

- 평가지표 : MAE(더 작은 지표가 좋은 성능을 의미) 

![Untitled](https://user-images.githubusercontent.com/69068083/231114263-2ea26465-c8b3-4df8-b7d4-4d329d41d97b.png)

1.  structure extractor에서의 k의 영향 비교
    -   k=0일때, Absolute encoding만을 활용하는 vanilla transformer랑 같다고 볼 수 있음
    -   k=3일때, optimal performance를 보임을 확인
    -   k=4를 넘어서면 성능이 악화되는것을 확인할 수 있었는데, 이는 GNN에서의 알려진 사실인 더 적은 수의 layer를 가지는 network가 더 좋은 성능을 보이는 것과 마찬가지라고 할 수 있음
2.  Absolute encoding의 영향 비교
    -   RWPE vs. Laplacian PE
    -   Structure-aware attention의 도입으로 인한 성능향상보다는 그 정도가 낮았지만, RWPE를 도입할 경우 성능이 더 좋은것으로 보았을 때, 두가지 encoding이 상호보완적인 역할을 한다고 해석할 수 있음
3.  Readout method의 영향 비교
    -   node-level representation을 aggregate할 때 사용하기 위한 readoutd으로 mean과 sum을 비교하였음
    -   추가로 `[CLS]` 토큰을 통해 graph-level 정보를 pooling하는 방법도 같이 비교하여보았음
    -   GNN에서는 readout method의 영향이 매우 컸지만 SAT에서는 매우 약한 영향만을 확인함.

# **5. Conclusion**

_**Strong Points.**_

structural info를 graphormer에서처럼 휴리스틱하게 shortest path distance(SPD)를 활용하지 않고, 그러한 local info를 잘 배우는 GNN으로 대체한 점이 novel하다고 할 수 있음

Transformer의 global receptive field 특성과 GNN의 local structure특성이 상호보완적

encoding에 있어서도

1.  RWPE를 통한 positional encoding
2.  k-subtree/subgraph GNN을 통한 structure-aware attention

두가지가 상호보완적인 역할을 함

→ 각자가 잘 배우는 특성을 고려하여 상호보완적인 두가지 방법론을 잘 섞어서 좋은 성능을 내었고, 그 이유가 납득하기 쉬움



_**Weak Points.**_

그래프데이터에 Transformer를 적용한 다른 논문의 architecture인 Graphormer에서 사용한 SPD만의 장점 : 직접적으로 연결되어있지 않은, 아주 멀리에 위치한 노드쌍이더라도 shortest path상의 weighted edge aggregation을 하는 만큼 그러한 특성 반영되면 좋은 그래프 구조/ 데이터셋에서는 SAT가 capture하지 못하는 부분이 있을 것

***
# **Author Information**


-   Dexiong Chen
    
    -   Department of Biosystems Science and Engineering, ETH Zurich, Switzerland.
    -   SIB Swiss Institute of Bioinformatics, Switzerland.
-   Leslie O'Bray
    
    -   Department of Biosystems Science and Engineering, ETH Zurich, Switzerland.
    -   SIB Swiss Institute of Bioinformatics, Switzerland.
-   Karsten Borgwardt
    
    -   Department of Biosystems Science and Engineering, ETH Zurich, Switzerland.
    -   SIB Swiss Institute of Bioinformatics, Switzerland.

# **6. Reference & Additional materials**

-   Github Implementation : [](https://github.com/BorgwardtLab/SAT)[https://github.com/BorgwardtLab/SAT](https://github.com/BorgwardtLab/SAT)
-   Reference : [Structure-Aware Transformer for Graph Representation Learning](https://arxiv.org/abs/2202.03036)
