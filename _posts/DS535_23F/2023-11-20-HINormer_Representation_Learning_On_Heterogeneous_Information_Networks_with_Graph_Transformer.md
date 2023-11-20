---
title:  "[WWW 2023] HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer"
permalink: 2023-10-16-HINormer_Representation_Learning_On_Heterogeneous_Information_Networks_with_Graph_Transformer.html
tags: [reviews]
use_math: true
usemathjax: true
---


---
description : Qiheng Mao et al. / HINormer Representation Learning On Heterogeneous Information Networks with Graph Transformer / WWW-2023 (description)  
---

# **HINormer Representation Learning On Heterogeneous Information Networks with Graph Transformer (WWW-2023)** 


## **1. Problem Definition**  

&nbsp;&nbsp;&nbsp;&nbsp;본 연구는 현실에 존재하는 그래프 구조의 데이터는 노드(node)와 엣지(edge)의 타입이 다양하게 분포하는 heterogeneity를 띄고 있기 때문에 GNN 분야에서 주로 연구되던 homogeneous graph 기반의 신경망으로는 데이터의 특징(feature)과 구조(structure) 등의 정보를 충분히 학습하기 어려운 점을 기존 연구의 문제점으로 지적하였다. 또한, Transformer 구조는 자연어 처리(NLP)와 컴퓨터 비전(CV) 등 다양한 딥러닝 분야에서 모델의 성능을 향상시키는데 큰 역할을 하고 있지만, 노드 간의 연결성이 복잡한 그래프 구조의 특성을 고려하였을 때 이를 적용하는 과정에 많은 한계점이 존재하는 것을 문제로 제시하였다. 이에 따라, 본 연구가 정의한 문제는 다음과 같다.  
1. Local-view structure 정보를 활용하여 효율적인 node-level Transformer encoder를 어떻게 구성할 수 있는가?  
2. Heterogeneous Information Networks(HINs)에서 graph Transformer를 활용하여 노드 간의 heterogeneous semantic relation을 어떻게 효과적으로 학습할 수 있는가?

## **2. Motivation**  

&nbsp;&nbsp;&nbsp;&nbsp;이 연구는 기존의 HINs에서의 표현 학습 방법의 한계를 해결하고자 새로운 형태의 Transformer Mechanism를 활용하였다. HINs는 복잡한 heterogeneous semantic relation을 가진 노드와 엣지의 다양한 유형을 연결하는 heterogeneous 구조로, 웹 데이터와 같은 많은 실제 데이터는 heterogeneous networks 형태를 띄고 있다. <Fig 1>은 academic graph와 review graph에 대해 heterogeneous networks 예시를 보여주고 있다.  
<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/5ab0fd1a-6c16-4472-b378-f6b6500786d9" alt="heterogeneous_graph">
</div>
<center>

**Fig 1. Heterogeneous Networks Examples[12]**
</center>
&nbsp;&nbsp;&nbsp;&nbsp;Heterogeneous graph를 활용한 GNN은 구조와 heterogeneity을 동시에 학습함으로써 한 가지 타입의 노드와 엣지로 구성된 homogeneous graph 기반 GNN에 비해 downstream task에서 더욱 높은 성능을 달성함에 따라 관련 연구가 점차 증가하고 있다. 관련 연구는 크게 metapath-based approach[1,2,3]와 metapath-free approach[4,5,6]가 있다. 먼저 metapath-based approach는 두 노드의 연결 경로가 되는 노드 타입 기반의 metapath를 사용하여 노드간 관계성을 고려하여 학습하는 방식이고, metapath-free approach는 서로 다른 타입의 노드와 엣지의 정보를 타입별로 학습함으로써 metapath 없이 그래프의 heterogeneity를 학습하는 것이다. 그러나, 기본적으로 meassage passing을 기반으로 하는 GNN은 다음과 같이 몇 가지 한계점이 존재한다.  

&nbsp;&nbsp;&nbsp;&nbsp;먼저, GNN의 표현력은 Weisfeiler-Lehman isomorphism에 의해 더 미세한 그래프 구조를 학습하는 데 취약할 수 있다. 다음으로, over-smoothing 문제가 발생하여 서로 다른 노드가 유사한 표현을 가지게 되어 노드 간의 구별력이 감소하고, 그래프의 복잡한 패턴과 구조를 정확하게 학습하고 표현하는 능력이 저하된다. 세번째로, over-squashing이 발생하여 노드의 정보가 너무 많이 압축되어 모델이 먼 거리의 노드로 메시지를 효과적으로 전파하지 못하고 주로 단거리 신호만을 캡처한다.  
&nbsp;&nbsp;&nbsp;&nbsp;이러한 문제를 해결하기 위해 그동안 Transformer를 GNN에 활용하고자 하는 시도가 많이 있어 왔다. 대표적인 연구가 Graph Transformer Networks (GTN) [13] 이다. GTN은 복수의 Channel을 출력하는 1 $\times$ 1 CNN을 활용하여 하나의 그래프에서 다양한 인접행렬 (adjacency matrix) 후보군을 추출한다. 그중 2개의 인접행렬을 활용하여 새로운 형태의 인접행렬을 생성하여 기존 그래프로부터 재구성된 새로운 그래프를 생성한다. 이때, 새로운 인접행렬은 두 개의 후보 인접행렬의 행렬곱을 통해 생성되기에 multi-hop을 기반으로 한 새로운 meta-path가 형성된다. 이렇게 재구성된 그래프는 기존의 그래프에서는 포착하기 어려웠던 관계에 있는 노드의 연결성을 고려하여 node representation을 할 수 있게 된다. GTN의 구체적인 Architecture는 <Fig 2>와 같다. 그러나 Graph Transformer는 graph representation에 매우 긍정적으로 기여함에도 불구하고, 몇 가지 한계점이 존재한다.  Global attention mechanism은 그래프의 노드 수에 대해 제곱에 비례하는 복잡성을 초래하므로, 대규모 네트워크에 대한 확장성이 떨어진다. 이 문제로 인해 소규모 그래프 레벨 Transformer는 활발히 연구된 반면, 대규모 노드 레벨 Transformer는 그렇지 않다. 따라서, 본 연구에서 저자는 그러한 한계를 극복하며 large heterogeneous information networks에 transformer mechanism을 적용할 수 있는 모델을 소개한다.  
<div align="center">
  <img src="https://github.com/BBeeChu/-KAIST-data_science_and_machine_learning/assets/8716868/385b99dc-fe23-4a97-bba0-e247df5099a4" alt="graph_transformer">
</div>
<center>

**Fig 2. Graph Transformer Networks Architecture[13]**
</center>
연구의 contribution은 다음과 같이 세 가지로 구성된다.  

1) HINs에 대한 노드 embedding을 위해 기존과 다른 형태의 graph transformer를 적용한 새로운 표현 학습 패러다임 HINormer를 설계했다. 
2) local structure encoder와 hetero-relation encoding을 포함하는 새로운 모델 HINormer는 HINs의 graph structure와 heterogeneity를 모두 포착하여 노드를 효과적으로 학습한다. 
3) 네 가지 benchmark dataset에서의 실험을 통해 기존의 SOTA 모델을 능가하는 성능을 보여준다.  



## **3. Method**  

## *3.1. Preliminaries*
&nbsp;&nbsp;&nbsp;&nbsp;본 연구에서는 기존의 HINs의 효과적인 학습을 위해 선행 연구된 GNN 모델과 Transformer 모델을 기반으로 전체적인 model architecture를 구성한다. 먼저, 앞으로 사용될 notation을 정리하면 <Table 1>과 같다.  


<center>

| Notations   | Descriptions    |
| ------- | ------- | 
| G = {*V*, *E*, **X**, $\phi$, $\psi$}    | Heterogeneous Information Network  | 
| *V* | 노드 집합  | 
| *E*    | 엣지 집합  |
| **X** $\in R^{\vert V \vert \times d_{x}}$    | 노드 feature 행렬  |
| **x**$_{v} \in R^{d_x}$    | 노드 feature 벡터  |
| $\phi$    | 노드 타입 mapping 함수  |
| $\psi$    | 엣지 타입 mapping 함수  |
| $\phi(v)$    | 특정 노드 타입  |
| $\psi(e)$    | 특정 엣지 타입  |
| $T_{v}$ = {$\phi(v): \forall_{v} \in V$}    | 노드 타입 집합  |
| $T_{e}$ = {$\psi(e): \forall_{e} \in E$}    | 엣지 타입 집합  |
</center>

<center>

**<Table 1> Notations**
</center>


GNN은 layer에 따른 이웃 노드 aggregation을 통해 주변 노드의 feature를 학습하며 graph representation을 업데이트하는 모델이다. GNN의 학습 과정은 아래의 <Eq 1>식으로 표현된다.  

<center>

$\boldsymbol{h_ {v}^{l}} = AGGR(\boldsymbol{h}_ {v}^{l-1}, \{\boldsymbol{h}_ {i}^{l-1} : i \in N_ {v} \};\theta_{g}^{l})$,  

$N_ {v}$ : v노드의 이웃 노드 집합,  

$AGGR(\cdot ; \theta_{g}^{l})$ : 이웃 노드 aggregation function.  
</center>

<center>

**<Eq 1> Aggregation of GNN**
</center>

&nbsp;&nbsp;&nbsp;&nbsp;GNN은 위 식으로 표현되는 aggregation의 방법에 따라 mean-pooling을 활용하는 GCN[7], attention을 활용하는 GAT[8] 등의 다양한 방법이 있다. Transformer mechanism은 self-attention과 feed-forward로 구성되는데, multi-head self-attention을 통해 특정 벡터의 요소별 중요도를 계산한 후 feed-forward를 통해 그 중요도를 반영한 새로운 형태의 벡터를 추출하는 방법이다. 설명의 간략화를 위해 multi-head과정을 생략하고 <Eq 2>와 <Eq 3>으로 표현할 수 있으며 architecture에 대한 자세한 내용은 [9]에서 확인할 수 있다.
<center>

$Q = HW_{Q}, K = HW_{K}, V = HW_{V}$,  

$W_{Q} \in R^{d \times d_{Q}}, W_{K} \in R^{d \times d_{K}}, W_{V} \in R^{d \times d_{V}}$  

$MSA(H) = Softmax(\frac{\boldsymbol{QK}^{\top}}{\sqrt{d_{K}}}\boldsymbol{V})$  

**<Eq 2> Multi-head Self-Attention Mechanism**  
  
$\boldsymbol{\widetilde{H}^{l}} = Norm(MSA(\boldsymbol{H^{l-1}}) + \boldsymbol{H^{l-1}})$  

$\boldsymbol{H^{l}} = Norm(FFN(\widetilde{\boldsymbol{H^{l}}})+\boldsymbol{\widetilde{H}^{l}})$  

$Norm: Layer\ Normalization$  
  
**<Eq 3> Feed Forward Network**
</center>



## *3.2. Overall Architecture*
&nbsp;&nbsp;&nbsp;&nbsp;HINormer는 HINs의 정보를 학습하기 위한 새로운 방법으로, Graph Transformer를 사용하였다. 헷갈리지 말아야 할 것은 본 논문에서 언급하는 "Graph Transformer"라는 이름은 Multi-head Attention Mechanism을 활용하여 저자가 새롭게 제시하는 모델의 이름으로, 앞서 Motivation에서 언급한 GTN과는 다른 개념이다. HINormer는 local-structure encoder와 hetero-relation encoder로 구성이 되는데, local-structure encoder는 주변 노드의 feature를 aggregate하여 노드의 local structure을 학습하고, hetero-relation encoder는 주변 노드의 타입 정보를 aggregate하여 heterogeneity를 학습에 반영한다. 그리고, Graph Transformer는 두 encoder에서 얻은 정보를 aggregate하여 종합적인 노드 정보를 생성하는데 사용된다. 모델의 전체적인 architecture는 <Fig 3>과 같다.
<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/1753c58d-9f32-4d48-b812-53dd4268a6d5" alt="hinormer">
</div>
<center>

**Fig 3. Overall Framework of HINormer**

</center>


## *3.3. Node-level Heterogeneous Graph Transformer Architecture*
&nbsp;&nbsp;&nbsp;&nbsp;본 연구는 노드 embedding 과정에 Transformer Mechanism을 효율적으로 적용하고자 새로운 방안을 제안한다. 특히, Transformer를 GNN에 적용함에 따라 불가피하게 증가하는 parameter의 수로 인한 확장성 저하 및 과적합 문제를 해결하는 것에 집중하였다. 먼저, self-attention을 수행하기 위해서 target node를 중심으로 정해진 길이($S$) 만큼의 이웃 노드를 정해진 깊이($D$)만큼 탐색하며 추출한다. 이때 사용되는 $S$와 $D$는 하이퍼파라미터이다. 구체적인 예를 들어 설명하자면, 임의의 타겟 노드 $v$와 관련하여 추출된 이웃노드 집합을 $s(v) = {v, v_{1}, ..., v_{S-1}}$로 정의한다. 이들을 다시 Transformer 모델에 입력할 수 있는 embedding 형태의 집합으로 표현하면 $\boldsymbol{H}_ {v}^{s} = [\boldsymbol{h}_ {v}^{s}, \boldsymbol{h}_ {v_ {1}}^{s},...,\boldsymbol{h}_ {v_ {S-1}}^{s}]^{\top} \in R^{S\times d}$와 같다. 여기서, $d$는 노드 embedding 차원을 의미하고, $\boldsymbol{h}_ {v_ {i}}^{S}$는 3.4장에서 설명할 local structure encoder를 통해 학습되는 주변 노드 기반의 노드 feature이다. 유념해야 할 것은 본 장에서 설명하는 Transformer는 앞으로 설명할 local-structure encoder를 통해 학습한 노드 feature와 hetero-relation encoder를 통해 학습한 관계성 정보를 aggregate하는 방법이다.  
&nbsp;&nbsp;&nbsp;&nbsp;해당 연구에서는 Graph Transformer와 관련하여 앞서 설명한 문제들을 해결하기 위해서 Graph Attention Network 모델[8]에서 제안된 attention 기반의 aggregation 방식을 변형한 GATv2[10]를 Transformer model의 multi-head self-attention 방법으로 활용하였다. GATv2은 벡터간의 관계성 추출을 위해 내적(dot-product)이 아닌 concatenation을 활용함으로써 learnable parameter의 개수를 줄였다. 또한, 본 연구의 저자는 각 Transformer layer의 feed-forward network를 삭제함으로써 parameter의 개수를 현저히 줄였다. 먼저, 본 연구에서 활용된 Transformer의 작동 과정을 이해하기 위해서는 [8]에서 소개된 GAT의 attention mechanism을 이해할 필요가 있다. 본 페이퍼에서는 GAT의 과정에 대해 자세하게 설명되지 않았지만 앞으로 설명할 내용의 이해를 돕기 위해 [8]의 내용을 일부 활용하여 GAT를 설명하면 <Eq 4>와 같다. 
<center>

$s_{i}^{(k)} = \alpha_{i,i}^{(k)}\boldsymbol{W}^{(k)}\boldsymbol{h}_{i}^{(k-1)} + \sum_{j \in N_{i}}\alpha_{i,j}^{(k)}\boldsymbol{W}^{(k)}\boldsymbol{h}_{j}^{(k-1)}$,  

$\boldsymbol{h}_{i}^{(k)} = f_{k}(s_{i}^{(k)})$  

**<Eq 4> Graph Attention Networks Mechanism**
</center>

<Eq 4>에서 $N_{i}$는 $i$번째 노드에 대한 인접 노드들의 index이며, $\boldsymbol{W^{(k)}}$와 $f_{k}$는 각각 k번째 GAT의 가중치 행렬과 활성화 함수이다. attention score $\alpha_{i,j}^{(k)}$는 <Eq 5>에 의해 계산된다. 
<center>

$\alpha_{i,j}^{(k)} = \frac{exp(\phi^{(k)}(\boldsymbol{W}^{(k)}\boldsymbol{h}_{i}^{(k)}, \boldsymbol{W}^{(k)}\boldsymbol{h}_{j}^{(k)}))}{\sum_{r \in N_{i}}exp(\phi^{(k)}(\boldsymbol{W}^{(k)}\boldsymbol{h}_{i}^{(k)}, \boldsymbol{W}^{(k)}\boldsymbol{h}_{r}^{(k)}))}$  

**<Eq 5> Attention Score**
</center>

GATv2은 $exp(\phi^{(k)}(\boldsymbol{W}^{(k)}\boldsymbol{h}_ {i}^{(k)}, \boldsymbol{W}^{(k)}\boldsymbol{h}_ {j}^{(k)}))$ 함수에서 GAT과 차이가 있다. 두 방식을 나타내면 <Eq 6>과 같다. 
<center>

$GAT : \alpha(\boldsymbol{h_{i}}, \boldsymbol{h_{j}}) =  \sigma(\boldsymbol{a}^{\top}\cdot[\boldsymbol{Wh_{i}||Wh_{j}}])$   
<br>

$GATv2 : \alpha(\boldsymbol{h_{i}}, \boldsymbol{h_{j}}) = \boldsymbol{a}^{\top}\cdot \sigma(\boldsymbol{W}\cdot[\boldsymbol{h_{i}}||\boldsymbol{h_{j}}])$  

$\sigma: Leaky\ ReLU\ Activation\ Function$

**<Eq 6> GAT & GATv2**
</center>

앞서 설명한 바와 같이 본 연구에서는 GATv2방식을 활용하였다.


## *3.4. Local Structure Encoder*
&nbsp;&nbsp;&nbsp;&nbsp;먼저 local structure encoder는 노드와 엣지의 타입을 무시하고, GCN이나 GAT와 같은 모델을 활용하여 homogeneous graph 정보를 학습하는 encoder이다. 따라서, 서로 다른 타입의 노드 feature를 <Eq 7>과 같이 linear transformation을 통해서 동일한 공간에 projection해주는 것이 필요하다.
<center>

$\boldsymbol{h}_{v} = \boldsymbol{W}_{\phi(v)}\boldsymbol{x}_{v} + \boldsymbol{b}_{\phi(v)}$  

**<Eq 7> Node Features Projection**
</center>

&nbsp;&nbsp;&nbsp;&nbsp;다음으로, 사전에 정의된 $K_{s}$-hop만큼 기존의 GNN 모델(GCN, GAT, etc)을 활용하여 이웃 노드의 feature을 학습한다. 해당 부분의 작동 과정은 앞서 설명한 <Eq 1>과 동일하다. 다만, 본 연구에서는 모델의 성능을 세밀하게 분석하기 위해서 이웃 노드의 feature를 aggregate하는 과정에서 weight parameter가 곱해져서 학습되던 vanilla GNN과 달리 weight을 제거한 aggregation(<Eq 8>) 또한 실험을 진행하였다. 

<center>

$\boldsymbol{h}_{v}^{K_{s}} =(\boldsymbol{\hat{A}}^{K_{s}}\boldsymbol{H})[v,:]$  
$\boldsymbol{\hat{A}} = \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2} \rightarrow$ *Normalized Adjacency Matrix*  

**<Eq 8> Aggregation without Weight Parameters**
</center>

## *3.5. Heterogeneous Relation Encoder*
&nbsp;&nbsp;&nbsp;&nbsp;해당 encoder는 각 노드와 엣지의 타입을 고려한 정보인 heterogeneous semantic proximity를 학습하기 위한 encoder로서 각 노드를 타입별로 mapping하여 학습한다. 타입별 정보의 초기화를 위해서 one-hot vector를 각 노드의 초기 정보로 활용하였다. 즉, 노드 타입 행렬 $T \in R^{\vert T \vert \times \vert T \vert}$에서 노드 v의 타입 벡터 초깃값$(r^{0}_ {v})$은 $T[\phi(v), :]$이다. 타겟 노드의 이웃 노드 타입에 대한 정보를 aggregate 하기 위한 과정 또한 기존의 GNN layer(GCN, GAT, etc)를 활용한다. 단, 기존의 방식과 다른 점은 GNN aggregation 이후에 타입별 learnable parameter $w^{t-1}_ {\phi(u)}$를 가중합(weighted sum)함으로써 추가적인 aggregation을 진행하는 것이다. 이에 대한 수식은 <Eq 9>와 같다.
<center>

$\boldsymbol{r}^{t}_{v} = \sum_{u \in N(v)}w^{t-1}_{\phi(u)}f(\boldsymbol{r}_{u}^{t-1};\theta_{h}^{t}),$  

$f(\cdot ; \theta^{t}_{h})=$Aggregation function of GNN layers  


**<Eq 9> Aggregation of Neighbor Node Types**
</center>

&nbsp;&nbsp;&nbsp;&nbsp;Relation encoder에서 헷갈리지 말아야 할 것은 <Eq 9>에서 진행한 것은 하나의 노드를 기준으로 했을 때 이웃 노드의 타입 정보를 aggregate함으로써 노드 embedding을 하는 것이다. 본 연구에서는 여기에서 머무는 것이 아니라, 주변 노드와의 self-attention을 기반으로 한 Graph Transformer를 통해서 효과적인 학습을 기대한다. 따라서, 임의의 노드쌍 간의 관계성 또한 반영된 정보를 최종적으로 추출해야 한다. 해당 과정은 <Eq 10>에 자세히 나타나 있다.

<center>

$\boldsymbol{q}_{i}^{R} = \boldsymbol{W_{Q_R}}\boldsymbol{r}_{i}, \ \boldsymbol{k}_{j}^{R} = \boldsymbol{W}_{K_{R}}\boldsymbol{r_{j}}$,  

$\hat{\alpha}_{i,j} = \alpha_{i,j} + \beta\cdot\boldsymbol{q}_{i}^{R}\boldsymbol{k}_{j}^{R}$  

**<Eq 10> Consideration of Relations of Node Pairs**
</center>

<Eq 10>을 구체적으로 살펴보면 먼저 각 노드에 learnable parameter를 활용해서 각각 한번 더 projection을 해주고, 그 결과를 내적(inner-product)함으로써 두 노드간의 유사도를 계산한다. 다음으로, 하이퍼파라미터 $\beta$만큼 해당 유사도를 반영하여 이전 layer의 Transformer 과정에서 추출된 attention score와 합하여 positional encoding인 $\hat{\alpha}_{i,j}$를 추출한다. 최종적으로 local-structure encoder의 결과 벡터와 hetero-relation encoder의 결과 벡터에 multi-head self-attention을 적용하여 <Eq 11>과 같이 Transformer layer에 태워 최종 output을 출력한다.

<center> 

$\boldsymbol{H^{l}} = Norm(\boldsymbol{H^{l-1}} + MSA(\boldsymbol{H^{l-1}}, 
\boldsymbol{R})),$  

$Norm: Layer\ Normalization,$  
$MSA: Multi-head\ Self\ Attention$


**<Eq 11> Transformer Layer**
</center>

## *3.6. Training Objective*
&nbsp;&nbsp;&nbsp;&nbsp;본 연구에서는 대표적인 downstream task로 node classification을 진행하였다. Transformer를 통해 출력된 output값에 linear layer를 통해 1차원의 벡터를 추출한 후, 크로스 엔트로피(cross-entropy)를 통해 loss값을 계산하였다. 구체적인 과정은 <Eq 12>와 같다.

<center>

$\tilde{y}_{v} \in R^{C} = \phi_{Linear}(\boldsymbol{h_{v}};\theta_{pre})$  

$L = \sum_{v\in V_{tr}}CE(\tilde{y}_{v}, y_{v})$,  

$V_{tr}=training\ nodes$  
$CE: Cross\ Entropy\ Loss$

**<Eq 12> Objective Function**
</center>

## **4. Experiment**  

&nbsp;&nbsp;&nbsp;&nbsp;본 연구에서는 HINormer의 성능을 알아보기 위해 4가지 데이터셋에 대해 node classification을 수행하고, 다양한 baseline model과 비교하였다. 또한, transformer방법 및 하이퍼파라미터에 따른 ablation study를 추가적으로 진행하였다. 데이터셋은 $24:6:70$으로 random split을 진행하였다.  

### ***4.1. Experiment setup***  
* **Dataset**  
  총 네 개의 실제 HIN 벤치마크 데이터셋이 사용되었는데, 두 개의 academic citation 데이터셋과 두 개의 movie-related 데이터셋이 해당된다. 데이터셋에 대한 자세한 설명은 다음과 같다.  
1) **DBLP**: 컴퓨터 과학 분야의 문헌 데이터셋으로, 1994년부터 2014년까지 20개의 컨퍼런스에서 발표된 논문을 포함하고 있으며 4개의 연구 분야에 걸쳐 있다. 저자(A), 논문(P), 용어(T), 장소(V) 네 종류의 노드가 있으며, 본 연구에서는 HGB[6]에서 분할한 데이터셋을 사용하였다.
2) **AMiner**: 학술 네트워크 데이터셋이며, 본 연구에서는 원래 데이터셋의 서브그래프를 사용하였다. 데이터셋의 label로는 논문의 계열을 나타내는 네 개의 클래스가 사용되며, 논문(P), 저자(A), 참조(R) 세 종류의 노드가 있다.
3) **IMDB**: 영화 및 관련 정보에 대한 웹사이트로, 액션, 코미디, 드라마, 로맨스, 스릴러 클래스의 영화를 포함하는 multi-label 데이터셋이다. 영화(M), 감독(D), 배우(A), 키워드(K) 네 종류의 노드가 있으며, 본 연구에서는 HGB[6]에서 분할 데이터셋을 직접 사용합니다.
4) **Freebase**: huge knowledge graph이며, 본 연구에서는 영화(M), 배우(A), 감독(D), 작가(W)의 4가지 장르의 엔터티를 포함하는 서브그래프를 사용하였다.  
   
* **Baseline**  
  제안된 HINormer를 종합적으로 평가하기 위해 본 연구에서는 Homogeneous GNNs, Meta-path based HGNNs, 그리고 Meta-path free HGNNs의 주요 카테고리에 해당하는 모델을 baseline으로 설정하였다.  

1) **Homogeneous GCN**  
(a) **GCN**: GCN[7]은 이웃 노드로부터 메시지를 aggregate하여 노드 representation을 형성하기 위해 mean-pooling을 aggregation function으로 활용한다.  
(b) **GAT**: GAT[8]은 이웃 노드 aggregation에 multi-head attention mechanism을 사용하는 모델이다.  
(c) **Transformer**: Transformer [9]는 standard Transformer 아키텍처와 함께 node-level representation learning을 수행하기 위해 동일한 컨텍스트 샘플링 전략을 사용하였다.  

2) **Meta-path based HGNNs**  
(a) **RGCN**: RGCN [11]은 GCN을 여러 엣지 유형 그래프로 확장하고, heterogeneous graph convolution을 두 단계로 분해하였다. 각 노드의 첫 번째 단계는 특정 엣지 유형 그래프에서 mean aggregation을 수행하고 두 번째 단계는 모든 엣지 유형에서 표현을 aggregate한다.    
(b) **HetGNN**: HetGNN[12]은 먼저 타입 별 random walk로 생성된 각 이웃 노드의 feature를 Bi-LStM으로 aggregate하여 type-specific embedding을 추출한다. 다음으로, 모든 type-specific embedding에 attention mechanism을 사용하여 각 노드의 최종 embedding vector를 추출한다.  
(c) **HAN**: HAN[3]은 meta-path based neighbor aggregation을 위한 hierarchical node-level attention과 meta-path based semantic aggregation을 위한 semantic-level attention을 사용한다. 사용되는 attention mechasim은 GAT의 구조를 활용하였다.  
(d) **GTN**: GTN[13]은 learnable GCN을 사용한 서브 그래프 선택과 행렬 곱셈을 활용하여 meta-path의 수동 선택을 자동 학습 프로세스로 대체한 모델이다.  
(e) **MAGNN**: MAGNN[14]은 HAN을 기반으로하며 두 엔드 포인트의 노드만이 아닌 meta-path instance의 경로가 되는 모든 노드를 사용하는 모델이다. 

3) **Meta-path Free HGNNs**  
(a) **RSHN**: RSHN [15]은 먼저 골격화된 라인 그래프를 구축하여 다양한 엣지의 타입별 embedding을 얻고, message passing networks를 사용하여 노드와 엣지 정보를 모두 전파한다.  
(b) **HetSAN**N: HetSANN[16]은 heterogeneous information을 캡처하기 위해 타입 별 GAT 레이어를 사용하여 로컬 정보를 aggregate하는 모델이다.  
(c) **HGT**: HGT[4]은 각 엣지에 대한 heterogeneous attention을 특징으로 하는 heterogeneous한 Transformer-like attention mechanism을 제안하며, 대규모 HIN을 처리하기 위한 type-aware sampling method를 제안한다.  
(d) **SimpleHGN**: SimpleHGN[6]은 엣지 타입 임베딩과 노드 임베딩을 모두 고려하는 간단하면서도 강력한 GAT 기반 모델을 제안하며, HIN representaion learning을 표준화하기 위해 Heterogeneous Graph Benchmark(HGB)를 구축하였다.  
  
* **Hyperparameters Settings**   
본 연구에서는 baseline 하이퍼파라미터 설정을 위해 각 논문의 실험셋팅을 최대한 유지하였다. 또한, 제안하는 HINormer의 실험 셋팅을 <Table 2>와 같이 설정하여 진행하였다.  

<center>

| Settings   | Value    |
| ------- | ------- | 
| Optimizer    | Adam  | 
| Learning Rate | 0.0001  | 
| Dropout Rate    | Freebase: 0, Others: 0.5  |
| Sequence Length (S)   | [10, 200]  |
| Hidden Dimension    | 256  |
| Num of Head for MSA    | 2  |

**\<Table 2\> Hyperparameter Settings**
</center>

* **Evaluation Metric**  
모델의 분류 성능을 평가하기 위해 Micro-F1과 Macro-F1이 지표로 사용되었고, 모든 실험은 5번 반복되어, 표준 편차와 함께 평균 결과를 각 모델의 성능으로 설정하였다.

### ***4.2. Result***  
**Performance Evaluation**  
&nbsp;&nbsp;&nbsp;&nbsp;HINormer는 GCN으로 local structure encoder를 구성하였고, 다른 backbone 모델과의 성능을 함께 비교하였다. 그 결과는 <Fig 4>와 같다.
<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/ec4ebd70-4e03-472f-97fb-cf0f3f3eadcb" alt="performance_1">
</div>
<center>

**Fig 4. Performance Evaluation**

</center>

성능 비교의 내용은 다음과 같다. 먼저, standard Transformer 모델의 성능이 HGNNs 관련 모델을 대부분의 경우 능가함으로써, HINs의 representation learning에 context-based global attention mechanism의 높은 효과를 확인할 수 있었다. 다음으로, HINormer가 standard Transformer와 GAT의 성능을 높은 차이로 능가하였다. 이것은 단순한 형태의 standard Transformer는 그래프의 context structure와 heterogeneity를 충분히 학습하지 못하고, GAT은 Homogenous 기반 모델이기에 heterogeneity를 충분히 학습하지 못한다는 것을 보여준다. 세번째로, meta-path based HGNNs 모델들과 meta-path free HGNNs 모델들의 성능에 큰 차이가 없었으나 HINormer는 이들의 성능을 모두 능가하였다. 이것을 통해서, 현실적인 관점에서 깊은 도메인 지식을 접목하여 meta-path를 고려하는 것보다 meta-path를 고려하지 않고 heterogeneity를 반영하는 것이 HGNNs에서 경제적임을 알 수 있었고, HINormer는 heterogeneous 그래프에서 각 노드의 정보와 관계성을 동시에 학습하여 2-hop 이상의 노드 정보를 담는 high-order relational information을 효과적으로 파악하는 모델임을 확인할 수 있었다. 끝으로, GCN이나 GAT와 같은 homogeneous GNN 모델이 여전히 높은 성능을 보인 점을 고려하였을 때 high-order semantics 보다 주변 노드의 정보를 효과적으로 aggregate하는 것이 중요함을 확인할 수 있었다.  
&nbsp;&nbsp;&nbsp;&nbsp;다음으로, HINormer의 유연성을 확인하고자 learnable parameter weight을 적용하지 않는 GNN layer를 포함하여 다양한 backbone을 적용하여 비교하였다. 그 결과는 <Fig 5>와 같다.

<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/096cbdbe-a322-488d-9b63-5c5bf0423d29" alt="performance_2">
</div>
<center>

**Fig 5. Performance of HINormer with Different Backbones**

</center>

<Fig 5>에서 나타나는 것과 같이 HINormer는 다양한 GNNs backbone을 사용해도 standard Transformer 모델을 능가하였다. 또한, learnable parameter를 사용하지 않는 \'HINormer-Adj\'가 다른 backbone을 기반으로 한 HINormer와 성능 면에서 큰 차이를 보이지 않은 것으로 보아 local-structure encoder와 hetero-relation encoder 구조의 효과성을 다시 확인하였다.  

**Ablation Study**  
&nbsp;&nbsp;&nbsp;&nbsp;HINormer의 각 구성요소가 모델의 성능에 미치는 역할을 알아보고자 몇 가지 ablation study를 진행하였다. 결과는 <Fig 6>과 같다. 
<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/09a289d9-948f-4ce3-8acd-c8d54817a4a2" alt="ablation_1">
</div>
<center>

**Fig 6. Impact of Different Global Attention Mechanisms**

</center>

먼저, 세 가지 유형의 attention을 활용한 HINormer 모두가 일반적으로 GAT를 능가하는 것을 관찰하였으며, 이는 neighborhood-level attenion 보다 global attention mechanism이 뛰어난 성능을 달성하는 데 도움이 될 수 있음을 보여준다. 둘째로, 세 가지 유형의 attention을 활용한 HINormer 모두 유사한 성능을 달성할 수 있지만, HINormer-GATv2가 더 높은 안정성을 보이며 더 적은 수의 parameter로 네 가지 데이터셋에서 가장 높은 성능을 보였다.  
&nbsp;&nbsp;&nbsp;&nbsp;다음으로, HINormer를 구성하는 다양한 모듈의 영향력을 확인하고자 각 모듈에 대한 ablation study를 진행하였고, 그 결과는 <Fig 7>과 같다. 

<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/eb63e32b-b56e-49f5-8860-c456d7b62ab3" alt="ablation_2">
</div>
<center>

**Fig 7. Ablation Study & Scalability Study**
</center>

먼저, local structure encoder를 제외한 상황(no LSE)과 heterogeneous relation encoder를 제외한 상황(no HRE), 그리고 두 모듈을 모두 제외한 상황 (no LSE & HRE) 모두 성능이 감소하였다. 눈에 띄는 것은 \'no LSE\'인 경우가 \'no HRE\'인 경우에 비해 성능이 비교적 더 크게 감소하였다. 이것은 HINormer의 node representation 과정에서 local structural information이 더욱 의미있게 반영된다는 것을 나타낸다. 따라서, 노드 간 관계성의 효과적인 학습전략에 대한 후속연구가 필요해 보인다. 또한, GATv2 방식의 attention을 사용한 HINormer가 standard Transformer(회색)에 비해 더 높은 성능을 보이는 것으로 보아 parameter의 수를 줄이는 것이 모델로 하여금 과적합의 가능성을 낮추도록 함을 확인할 수 있었다. 추가적으로, 몇 가지 모델과 training time을 비교하며 scalability를 분석하였는데, HINormer가 다른 모델에 비해 큰 차이가 없는 training time이 소요되며 심지어 HGT보다 높은 폭으로 감소됨을 관찰하였다. 이로써, global attention을 적용하더라도 parameter의 수를 최소화 함으로써 HINormer가 높은 scalability를 가짐을 확인할 수 있었다.  
&nbsp;&nbsp;&nbsp;&nbsp;끝으로, parameter의 수치에 따른 모델의 성능을 확인하였고, 결과는 <Fig 8>과 같다.

<div align="center">
  <img src="https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/assets/8716868/1511d17f-14f3-42b5-8ece-4373c6af72f3" alt="hyperparameter">
</div>
<center>

**Fig 8. Parameters Sensitivity**
</center>

HINormer의 Transformer 레이어의 수($L$)는 더 작은 경우 전반적으로 더욱 높은 성능을 보였고, context의 사이즈를 나타내는 sequence 길이($S$)의 변화는 성능에 크게 영향을 주지 않았다. Transformer 레이어의 hidden dimension은 그래프의 heterogeneity로부터 더욱 다양한 의미정보를 학습할 수 있기 때문에 그 크기가 클수록 일반적으로 더 좋은 성능을 보였다.

## **5. Conclusion**  

&nbsp;&nbsp;&nbsp;&nbsp;본 논문에서는 node representation learning을 위해 HINs에 Graph Transformer를 효과적으로 활용할 수 있는 방안을 제시하였다. 기존에 Graph Transformer가 가지고 있던 문제를 해결하며 이를 달성하기 위해, local structure encoder와 heterogeneous relation encoder를 활용하여 target node의 neighborhood information과 high-order relation information을 추출하고 이를 GATv2를 활용한 multi-head self-attention 기반의 Transformer를 통해 학습하여 효과적인 node representation을 수행하는 HINormer 모델을 제안하였다. 이 모델은 heterogeneous graph의 structural information과 heterogeneity를 모두 포착함으로써 네 가지 벤치마크 데이터셋에서 기존의 SOTA를 능가하는 성능을 보임을 extensive experiment를 통해 확인하였다.  
&nbsp;&nbsp;&nbsp;&nbsp;개인적으로 현재 meta-path based GNNs을 활용한 온라인 학습 플랫폼 MOOC 콘텐츠를 추천하는 시스템을 연구하고 있기에 본 논문은 많은 insight를 주었다. 특히, 필자는 그동안 meta-path를 활용할 때 단순히 관계성만을 고려하는 것에서 나아가 그 관계의 경로가 되는 노드의 내용이 함께 활용될 수 있는 방안을 고민했는데, 본 논문에서 local-structure encoder를 통해서 그 문제를 해결하였고 심지어 모델의 성능에서 중요한 역할을 함을 ablation study를 통해서 입증하였다. 필자 또한 해당 structure를 활용할 수 있는 방안을 모색해야겠다. 다만, 한 가지 아쉬운 것은 heterogeneous relation encoder의 역할이 기대보다 크지 않았고, <Fig 4>의 성능지표에서 homogeneous 기반의 GNN 모델의 성능과 큰 차이가 없는 것을 보이며 meta-path의 효용성이 크지 않다고 언급한 점이다. 왜냐하면, 데이터셋의 성격과 그 데이터셋이 어떠한 도메인에서 수집된 것이냐에 따라 매우 다양한 특성을 보이기 때문이다. 대표적인 예로, MOOC dataset은 학생이 특정 교육콘텐츠를 선택함에 있어서 explicit factor와 implicit factor에 의해 영향을 받는다. 학생이 개인적으로 특정 강사를 좋아해서 그 강사가 가르치는 콘텐츠를 선택하는 경우(explicit factor)도 있지만, 강사를 인지하지 못한 채 내용 혹은 주제가 좋아서 몇 개의 콘텐츠를 선택했는데 공교롭게도 그 콘텐츠가 모두 동일한 강사에 의해 지도되는 경우(implicit factor)도 있기 때문이다.  
&nbsp;&nbsp;&nbsp;&nbsp;이러한 점을 고려하였을 때, graph 구조의 데이터에서 meta-path 기반의 heterogeneity가 큰 의미를 갖는 경우도 있기 때문에 long-length range에 있는 노드의 정보를 aggregate하여 효과적으로 high-order information을 학습할 수 있는 모델에 대한 고민이 필요해보인다. 시각을 달리해보면, 그만큼 본 연구의 후속연구로 수행할 수 있는 내용이 풍부하다는 것을 시사하기 때문에 다양한 측면에서 본 논문은 필자에게 매우 의미있는 연구였다.

---  
## **Author Information**  

* 추성엽 (SeongYeub Chu) 
    * Graduate School of Data Science, KAIST.  
    * AIED, Graph Neural Networks, Automated Essay Scoring, LLM Prompt Engineering

## **6. Reference & Additional materials**  


* Github Implementation  
  Original Github Repository: https://github.com/Ffffffffire/HINormer.git
  \\

* Reference  
  [1] Jie Liu, Lingyun Song, Guangtao Wang, and Xuequn Shang. 2022. Meta-HGT : Metapath-aware Hypergraph Transformer for heterogeneous information network embedding. Neural Networks (2022). https://doi.org/10.1016/j.neunet.2022.  
  [2] Yizhou Sun, Jiawei Han, Xifeng Yan, Philip S Yu, and Tianyi Wu. 2011. Pathsim: Meta path-based top-k similarity search in heterogeneous information networks. Proceedings of the VLDB Endowment 4, 11 (2011), 992–1003.  
  [3] XiaoWang, Houye Ji, Chuan Shi, BaiWang, Yanfang Ye, Peng Cui, and Philip S Yu. 2019. Heterogeneous graph attention network. In The world wide web conference. 2022–2032.
  [4] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous graph transformer. In Proceedings of The Web Conference 2020. 2704–2710.  
  [5] Zemin Liu, VincentWZheng, Zhou Zhao, Fanwei Zhu, Kevin Chen-Chuan Chang, Minghui Wu, and Jing Ying. 2017. Semantic proximity search on heterogeneous graph by proximity embedding. In Proceedings of the AAAI Conference on Artifcial Intelligence, Vol. 31.  
  [6] Qingsong Lv, Ming Ding, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, and Jie Tang. 2021. Are we really making much progress? Revisiting, benchmarking and refning heterogeneous graph neural networks. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 1150–1160.  
  [7] Thomas N Kipf and MaxWelling. 2017. Semi-supervised classifcation with graph convolutional networks. In ICLR.  
  [8] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2018. Graph attention networks. In ICLR.  
  [9] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).  
  [10] Shaked Brody, Uri Alon, and Eran Yahav. 2021. How attentive are graph attention
networks? arXiv preprint arXiv:2105.14491 (2021).  
  [11] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan
Titov, and Max Welling. 2018. Modeling relational data with graph convolutional
networks. In European semantic web conference. Springer, 593–607.  
  [12] Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V
Chawla. 2019. Heterogeneous graph neural network. In Proceedings of the 25th
ACM SIGKDD international conference on knowledge discovery & data mining.
793–803.  
  [13] Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J Kim. 2019. Graph transformer networks. Advances in neural information processing systems 32 (2019).  
  [14] Xinyu Fu, Jiani Zhang, Ziqiao Meng, and Irwin King. 2020. Magnn: Metapath
aggregated graph neural network for heterogeneous graph embedding. In
Proceedings of The Web Conference 2020. 2331–2341.  
  [15] Shichao Zhu, Chuan Zhou, Shirui Pan, Xingquan Zhu, and Bin Wang. 2019.
Relation structure-aware heterogeneous graph neural network. In 2019 IEEE
international conference on data mining (ICDM). IEEE, 1534–1539.  
  [16] Huiting Hong, Hantao Guo, Yucheng Lin, Xiaoqing Yang, Zang Li, and Jieping Ye. 2020. An attention-based graph neural network for heterogeneous structural learning. In Proceedings of the AAAI conference on arti￿cial intelligence, Vol. 34. 4132–4139.

  