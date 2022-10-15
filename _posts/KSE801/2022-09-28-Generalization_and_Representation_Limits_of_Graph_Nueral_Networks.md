---
title:  "[ICML 2020] Generalization and Representation Limits of Graph Nueral Networks"
permalink: Generalization_and_Representation_Limits_of_Graph_Nueral_Networks.html
tags: [reviews]
---

Write your comments
# Generalization and Representational Limits of Graph Neural Networks

### 1. Introduction
Graph Neural Network (GNN) 은 graph-structured data 를 학습하기 위한 모델로 등장하여, molecular structures, knowledge graph, social networks 등 다양한 domain 에서 사용되고 있다.

본 논문에서는 GNN 의 한계와 generalization properites 에 대하여 깊게 탐구하였다. 저자는 간단한 구조의 graph 라도 GNN 이 구분하지 못할 것이라는 가정하에, ~~
저자가 GNN 을 분석한 내용은 크게 두 가지로 나눌 수 있다.
a) GNN 모델들이 특정한 graph property 에 대하여 graph 를 구분할 수 있는가?
b) GNN 모델들이 graph 의 label 을 얼마나 잘 구분해낼 수 있는가?

a) 의 경우 아주 간단한 graph 들이 주어진 상황에서 graph의 성질 (i.e., longest or shortest cycle, diameter, clique information) 을 구분하지 못하는 경우가 있음을 보여준다.
b) 의 경우 간단한 binary prediction 으로 graph 모델의 performance limitation 을 설명하고자 한다.

본 논문의 contribution 은 다음과 같다.
1. 간단한 graph 지만, GNN 이 local information 만 사용하여 만든 embedding 으로는 구분할 수 없을 수 있음을 보인다. 또한 powerful 한 node information 을 얻어내기 위하여 "port numbering" 을 적용한 "CPNGNN" 과 "geometric information"을 사용하는 "DimeNet" 역시 해당 graph 들을 구분할 수 없음을 보인다.
2. CPNGNN 을 graph theoretic 관점에서 분석하여, GNN 의 효과에 대한 insight 를 얻는다.
3. GNN 의 message passing 에 관한 data dependent generalization bounds 를 제시한다. 또한, 기존 연구보다 더 tight 한 bound 임을 입증한다.

### 2. Related Work

### 3. Preliminaries
- Locally Unordered GNNs (LU-GNNs): spatial information 을 사용하지 않고 각 node 의 neighbors 에서 오는 message 로 node embedding 을 updata 하는 model (e.g., GraphSAGE, GCN, GIN and GAT).
LU-GNNs 에서 aggregation 과 conbine operation 은 다음과 같이 표기한다.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/formula_1.png"></p>

여기서, N(v) 는 node v 의 neighbor set 을 의미한다.

- Consistent port numbering GNNs (CPNGNNs): node 의 neighbors 에 port number 를 부여함으로써 local structure information 을 더 잘 뽑아내게 하는 model.
- DimeNet: molecular graphs 에서 directional message passing alogirhtm 을 사용한 model. Message passing 을 node 간의 angle 정보를 바탕으로 transform 하여 directional information 을 전달하는 model.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/formula_2.png"></p>

여기서 e^(uv) 는 노드 u 에서 노드 v 까지의 distance 를 의미하며, a^(wu,uv) 는 w 에서 u 까지의 거리에 wuv 의 각도를 합친 정보를 의미한다.
- Graph Property (P): Q 라는 GNN model과 서로 다른 property (P) 를 보이는 graph G_1 과 G_2 가 있을 때, 만약 f(g_Q(G_1) != f(g_Q(G_2)) 라면 model Q 는 P 를 분별할 수 있다.
또한, P 즉 graph property 의 종류로 저자는 1) grith (length of shortest cycle), 2) circumference (length of longest cycle), 3) diameter (maximum distance between any pair of nodes in graph), 4) radius (minimum node eccentricity; eccentricity: eccentricity of u is maximum distrance from u to other node in graph), 5) conjoint cycle (two cycles that share an edge), 6) total number of cycles, 7) k-clique (a subgraph of at least k >=3 vertices s.t. each vertex in the subgraph is connected by edge to any other vertex in the subgraph).

### 4. Representation limits of GNNs
#### Limitation of LU-GNNs
저자는 LU-GNNs 을 CPNGNNs 과 비교하여, LU-GNNs 의 한계를 보여준다.<br>
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Proposition_1.png"></p>
위의 Proposition 1. 을 조금 더 자세히 설명하면, LU-GNN 이 특정 property 에 대하여 구분할 수 없는 두 graph 가 존재한다고 할 때, CPNGNN 은 port numbering 덕분에 구분할 수 있다.<br> 구분하고자하는 property 를 "Isomorphic" 으로 가정하고, 이를 그림으로 표현하면 아래의 그림처럼 표현할 수 있다.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Figure_1.png"></p>
여기서 같은 색깔로 표현된 node 는 같은 feature vector 를 가지고 있다. 또한, edge 에 있는 숫자는 각 node 의 port number 를 나타낸다.<br>
Graph G 와 _G_ 는 isomorphic 관점에서 서로 다름을 알 수 있다. Graph G에서 structure 정보는 (B1 - C1 - D1 - B1 - C1 - D1 - B1 - ...), (B2 - C2 - D2 - B2 - C2 - D2 - B2 - ...) 으로 이루어져 있으며, Graph _G_ 의 경우 (_B1_ - _C1_ - _D1_ - _B2_ - _C2_ - _D2_ - _B1_ - ... ) 으로 이루어져있다.
하지만 LU-GNN 은 단순히 feature vector 만 사용하기 때문에, (보라색 - 빨간색 - 파란색 - 보라색 - 빨간색 - ... ) 과 같은 순서로 node 가 연결되어있다는 정보만을 알 수 있다. 즉, Graph G를 (보라색 - 빨간색 - 파란색 - 보라색 - 빨간색 - ...) 으로 인식하게 되며, Graph _G_ 의 경우도 (보라색 - 빨간색 - 파란색 - 보라색 - 빨간색 - ... ) 으로 인식하게 된다. 따라서, LU-GNN 의 경우 위의 그림과 같은 Graph 들의 isomorphism 을 분별하지 못한다. <br>
CPNGNN 의 경우, port number 를 사용하기에, Graph G 에서 D2 는 port 2 를 사용하여 B2 와 연결되었다는 정보와, Graph _G_ 에서는 _D2_ 가 port 1 을 사용하여 _B1_ 과 연결되었음을 알 수 있다. 따라서, CPNGNN 은 Graph G 와 _G_ 를 isomorphism property 에서 구별할 수 있다. <br>


#### Limitations of CPNGNNs
CPNGNN 이 LU-GNN 이 구분하지 못하는 Graph 들을 구별할 수 있음을 보여주었다. 하지만, CPNGNN 역시 구분할 수 없는 Graph 들이 존재하며, 저자는 LU-GNN 의 limitation 을 보여주었던 것과 같은 방식으로, CPNGNN 의 limitation 을 보여준다.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Proposition_2.png"></p>
Proposition 1. 과 동일하게, 색깔은 node 의 feature vector 를 edge 의 숫자는 port number 를 나타낸다. <br>
Graph 의 isomorphism property 를 구분하는 task 에서 다음 그림과 같은 Graph 들이 주어졌을 때, CPNGNN 은 isomorphism 을 구분할 수 없다.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Figure_2.png"></p>
그림의 Graph 들은 Proposition 1. 에서 보인 예시에서 Port number 를 변경한 Graphs 이다.<br>
CPNGNN 을 사용할 때, Graph G 의 D1과 D2 와 Graph _G_ 의 _D2_ 가 보라색, 빨간색 feature vector 와 연결된 port number 가 같음을 알 수 있다. 따라서, CPNGNN 은 해당 그래프에서 해당 node 들을 구분할 수 없다. 따라서, 두 그래프의 isomorphism 을 구분하는데 실패하게 된다.<br>
<br>
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Proposition_4.png"></p>
저자는 Proposition 4. 에서 isomorphism task 이외에 다른 task 에 대하여 일반화하였다.
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Figure_3.png"></p>
Graph S4 와 S8 은 girth, circumference, diameter, radius 를 가지고 있음에도 CPNGNN 이나 LU-GNN 으로 구분할 수 없다. 하지만, DimeNets 의 경우 node 간의 angle 을 사용하므로, Graph S4 의 A1-B1-C1 이 이루는 angle 과 Graph S8 의 _A1_-_B1_-_C1_ 의 angle 이 다르기에 두 graph 를 여러 properties 에 대하여 구분 할 수 있다.<br>
Graph G1, G2 의 경우 역시, CPNGNN 과 LU-GNN 으로 구분 할 수 없으나, DimeNets 의 경우 node 의 angle 정보를 사용함으로 graph 를 구분할 수 있다.
<br>

#### Limitations of DimeNets
DimeNets 이 CPNGNN 과 LU-GNN 이 구분할 수 없는 Graphs 들을 분별할 수 있을지라도, DimeNets 역시 구분 할 수 없는 상황이 존재한다.

### 5. Generalization bounds for GNNs
<p align="center"><img src="/images/Generalization_and_Representational_Limits_of_Graph_Neural_Networks/Figure_4.png"></p>


