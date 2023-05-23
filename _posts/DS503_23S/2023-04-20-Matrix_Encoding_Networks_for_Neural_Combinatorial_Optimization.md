---
title:  "[NIPS 2021] Matrix Encoding Networks for Neural Combinatorial Optimization"
permalink: Matrix_Encoding_Networks_for_Neural_Combinatorial_Optimization.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Title  
Matrix Encoding Networks for Neural Combinatorial Optimization

# 1. Problem Definition  
최근 몇 년간 **combinatorial optimization (CO) problem**을 해결하기 위한 approch로 **machine learning (ML)** 이 많이 활용되고 있다. 가장 대표적인 접근 방법은, CO problem의 input data의 global information을 problem의 각 entity에 대한 representation으로 encoding하는 **"front-end" neural net**을 활용하는 것이다.  
"front-end" model을 통해 각 entity의 representation을 도출하고, 이를 순차적으로 선택하는 selection strategy를 통하여 CO problem에 대한 solution을 얻을 수 있고, optimal selection strategy를 학습하기 위해 **end-to-end reinforcement learning (RL)** 을 활용하는 연구가 많이 진행되고 있다.  
본 연구는 CO problem의 **matrix-type data**를 효과적으로 처리하는 "front-end" model인 **Matrix Encoding Network (MatNet)** 을 개발한다. 그리고 _NP_-hard class에 속하는 **asymmetric traveling salesman problem (ATSP)** 과 **flexible flow shop problem (FFSP)** 에 대하여 효과를 검증하였다.

# 2. Motivation  
**Matrix-type data**는 CO problem에서 흔히 다루어진다. 예를 들어, 각 도시를 반드시 한번씩 방문하는 최단 tour를 찾는 문제인 traveling salesman problem (TSP)의 경우, Euclidean distance의 가정을 풀게되면, 두 도시간 임의의 거리가 설정되는 **distance matrix**가 정의되게 된다. (이러한 문제를 ATSP라 부른다.) 또한 job-shop/flow-shop scheduling problem이나 linear/quadratic assignment problem의 경우에도 matrix-type data가 정의된다. 따라서 일반적인 CO problem을 해결하는 neural solver 개발을 위해 matrix-type data의 효과적인 처리는 필수적이다.  
CO problem의 matrix-type data를 encoding하는 model은 본 연구 이전에 개발되지 않았으며, CO problem의 matrix-type data는 stacked vector-lists나 2D image를 표현하는 matrix와는 **permutaion invariance**라는 차이가 존재한다.  
따라서 본 연구는 matrix-type data를 가지는 CO problem 해결을 위해 새로운 구조의 model을 제안한다.

# 3. MatNet  
## 3.1 Complete bipartite graph with weighted edges  
MatNet은 **weighted edge**를 가지는 **complete bipartite graph**에 대한 graph neural network (GNN)으로 볼 수 있다 (**Figure 1**).  
해당 graph는 아래와 같이 정의된다.  
* $A=\{a_ 1,...,a_ M\}$
* $B=\{b_ 1,...,b_ N\}$
* $e(a_ i,b_ j)=D_ {ij}$

$A,B$는 서로 다른 node set, $D_ {ij}$는 $a_ i$와 $b_ j$ 간 edge의 weight이다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232305747-4ba77b6f-dcb2-4ea4-b340-44053028e30f.png" width="250px"></p>

## 3.2 Dual graph attention layer  
### 3.2.1 Graph attention layer in Kool _et al._, 2019  
MatNet은 [Kool _et al._, 2019][Attention_TSP]의 **Attention Model (AM)** 의 encoder architecture를 확장한 model이다.  
AM은 **Transformer**를 활용한 "front-end" model로, TSP와 vehicle routing problem (VRP)를 효과적으로 풀었다.  
AM의 encoder는 여러 층의 **graph attention networks (GATs)** 를 가지는 node-embedding framework를 따른다.  
Node $v$의 representation $\hat{h}_ v$를 $\hat{h}'_ v$로 업데이트하는 하나의 GATs layer는 아래와 같다.  
* $\hat{h}'_ v=\mathcal{F}(\hat{h}_ v,\{\hat{h}_ w\vert w\in \mathcal{N}_ v\})$ 

$\mathcal{N}_ v$는 node $v$의 neighboring node의 set이며, $\mathcal{F}$는 multiple attention head와 aggregation process로 구성된다.  
### 3.2.2 Dual graph attention layer in MatNet  
MatNet도 AM과 마찬가지로 GATs framework를 따른다.  
하지만, AM은 하나의 node set에 대한 **self-attention** 구조인 반면, MatNet은 각 node set에 대하여 구별되는 function을 구성하여 **cross-attention**을 수행하였고, 또한 edge weight에 대한 처리도 추가하였다.  
MatNet의 한 layer에 대한 update function은 아래와 같다.  
* $\hat{h}'_ {a_ i}=\mathcal{F}_ A(\hat{h}_ {a_ i},\{(\hat{h}_ {b_ j},e(a_ i,b_ j))\vert b_ j\in B\})$ for all $a_ i\in A$
* $\hat{h}'_ {b_ j}=\mathcal{F}_ B(\hat{h}_ {b_ j},\{(\hat{h}_ {a_ i},e(a_ i,b_ j))\vert a_ i\in A\})$ for all $b_ j\in B$

$\mathcal{F}_ A$와 $\mathcal{F}_ B$는 parameter를 공유하지 않으며, node 쌍 $(a_ i,b_ j)$의 attention score 계산 시 $\hat{h}_ {a_ i}$와 $\hat{h}_ {b_ j}$ 뿐 아니라 $e(a_ i,b_ j)$도 함께 사용된다.  
MatNet은 **Figure 2(a)** 와 같이 $\mathcal{F}_ A$와 $\mathcal{F}_ B$로 구성된 $L$개의 **dual graph attention layer**로 구성된다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232307968-80ed726b-47b8-4dfd-a2e2-af4aa53d9b89.png" width="800px"></p>

## 3.3 Mixed-score attention  
**Figure (a)** 의 "Multi-Head Mixed-Score Attention"은 **Transformer**의 "Multi-Head Attention"에서 scaled dot-product attention이 mixed-score attention (**Figure (b)**)로 대체된 것이다.  
Scaled dot-product attention은 query-key 쌍에 대한 attention score를 바탕으로 value의 weighted sum을 구하는 것으로, mixed-score attention에서는 attention score와 edge weight를 처리하는 **"Trainable Element-wise Function"** 이 추가되었다.  
이는 두 개의 input node와 하나의 output node를 가지는 **multilayer perceptron (MLP)** 으로, attention score와 edge weight를 input으로 받아 하나의 값을 도출한다.  
도출된 값을 바탕으로 value에 대한 weighted sum은 기존과 동일하게 진행된다.

## 3.4 The initial node representation  
Set $A$의 node에 대하여는 **zero-vector**를, set $B$의 node에 대하여는 **one-hot vector**를 initial representation으로 설정한다.  
Zero-vector를 사용한 $A$에 대하여는, node 수가 변하더라도 MatNet이 이를 처리할 수 있으나, $B$에 대하여는 그렇지 않다.  
논문에서는 $B$의 node 수에 대한 상한을 설정할 수 있다면, 상한에 해당하는 $N_ {max}$개의 one-hot vector를 구성하고, 각 instance가 $N_ o(\le N_ {max})$개의 $B$의 node를 가진다면 $N_ {max}$개의 one-hot vector 중 $N_ o$개를 sampling하여 사용할 수 있다고 한다.  
만약 적절한 $N_ {max}$를 설정할 수 없다면 $B$의 node의 initial representation을 random number로 구성된 vector로 대체할 수 있으며, 이는 size-agnostic이다. (하지만 **Appendix A.3**의 실험을 보면 약간의 성능 저하가 발생함을 확인할 수 있다.)    
B의 node의 initial representation을 distinct하게 구성함으로써, 두 가지 장점을 얻을 수 있다.  
첫 번째는 training에 사용할 instance를 augmentation할 수 있다는 것으로, B의 node에 대하여 서로 다른 조합의 one-hot vector를 할당함으로써 가능하다.  
두 번째는 test 시 B의 node에 대하여 서로 다른 조합의 one-hot vector를 할당하여 여러 개의 다른 solution을 얻을 수 있고, 그 중 좋은 solution을 취함으로써 더 좋은 성능을 보일 수 있다.

# 4. Experiment  
## 4.1 Asymmetric traveling salesman problem  
### Instance generation  
"tmat"-class ATSP instance를 사용하였고, 자세한 생성 방식은 [Cirasella _et al._, 2001][ATSP_generation]에서 확인할 수 있다.  
도시 수 $N$은 20, 50, 100의 3가지 problem size에 대하여 실험을 진행하였다.  
"from" city $a_ i$와 "to" city $b_ j$에 대한 **distance $d(a_ i,b_ j)$** 를 **edge weight**로 설정하였고, 해당 matrix는 $N$-by-$N$이다.  
### MatNet configuration  
5개의 encoding layer를 사용하였고 ($L=5$), embedding dimension $d_ {model}$은 256으로 설정하였다. $\mathcal{F_ A}$와 $\mathcal{F_ B}$는 동일하게 16개의 attention head를 가지며 score-mixing MLP는 16개 node의 하나의 hidden layer를 설정하였다.  
**Figure 2(a)** 에서 "Feed-forward" block은 dimmension 516의 하나의 hidden layer로 구성하였고, "Add & Norm"은 instance normalization이다.  
### Decoder  
Decoder는 [Kool _et al._, 2019][Attention_TSP]와 동일하게 한번에 하나의 방문할 city를 **autoregressive**하게 생성하며, model 구조는 **Figure 3**에서 확인할 수 있다.  
이때, [Kool _et al._, 2019][Attention_TSP]와 다르게 "QUERY" token은 현재 city와 처음 방문한 city의 representation을 concatenate하여 구성하였다.  
"to" city에 대한 representation이 decoder에 들어가면, 다음으로 방문할 **probability**가 각 city에 대하여 산출된다.  
이미 방문한 city는 mask 처리 된다.  
Probability를 기반으로 다음 방문 city를 선택하고, "QUERY" token을 수정하여 해당 과정을 반복함으로써 tour를 산출할 수 
있다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232313360-aaabd003-646c-4f0d-a8c9-012db21fdcad.png" width="250px"></p>

### Training  
[Kwon _et al._, 2020][POMO]의 **POMO training algorithm**을 활용한 RL로 MatNet model과 decoder를 학습하였다.  
POMO는 REINFORCE algorithm을 기반으로 하며 TSP 등 CO problem에 좋은 성능을 가짐이 검증되었다.  
$4\times 10^{-4}$의 learning rate와 200의 batch size를 설정하였다.  

### Result  
**Table 4.1**은 mixed-integer program (MIP)를 CPLEX를 통해 풀어 구한 optimal solution 대비 각 알고리즘들의 성능 저하 비율(Gap)을 보여준다.  
Nearest Neighbor, Nearest Insertion, Furthest Insertion은 ATSP의 대표적인 greedy rule이며, LKH3은 매우 효과적인 heuristic이다.  
MatNet은 single POMO rollout을 통해 하나의 solution을 도출한 결과를 보여주며, MatNet($\times$ 128)은 instance augmentaion을 통해 128개의 서로 다른 solution을 도출한 후 가장 좋은 것을 채택한 결과이다.  
MatNet을 활용한 알고리즘이 optimal solution과 거의 차이가 없는 성능을 보여줌을 확인할 수 있다. **(1% 미만의 optimality gap.)**  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232314604-dec97568-e0d6-441c-89e2-11b902cdd68f.png" width="800px"></p>

## Flexible flow shop problem  
### Problem definition  
FFSP는 순차적인 stage가 존재하고, 각 job이 각 stage에서 작업되어야 하는 문제이다. 각 stage는 여러 machine을 가지며, 각 job은 각 stage에서 반드시 하나의 machine에서 작업되어야 한다. 이때 machine은 한번에 하나의 job만 작업할 수 있으며, job과 machine pair에 대해 하나의 작업시간이 주어진다.  
본 실험에서는 3개의 stage와 stage 별 4개의 machine으로 구성된 문제를 다루었고, 20, 50, 100개의 서로 다른 job size에 대하여 실험하였다.  
Matrix-type data는 **machine-job pair에 대한 작업시간**이며, stage 별 하나의 matrix가 주어진다. 20개의 job에 대한 data 예시는 **Figure 4**에서 확인할 수 있다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232315222-a0bd2576-ea29-42e0-a622-adee3b208602.png" width="800px"></p>

### Encoding & decoding  
MatNet model과 decoder는 각 stage에 대하여 별도로 구성되었고, solution을 생성하는 과정은 본문에 자세히 설명되어있다.

### Training  
ATSP와 동일하게 POMO training algorithm을 활용하여 RL로 MatNet model과 decoder를 학습하였다.

### Result  
**Table 5.2**는 MatNet($\times$ 128) 대비 알고리즘들의 Gap(%)을 보여준다.  
CPLEX의 경우 합리적인 시간 내 solution을 도출하지 못하였다.  
Random과 Shortest Job First는 greedy rule, Genetic Algorithm과 Particle Swarm Optimization은 meta-heuristic으로 모두 FFSP에 많이 활용되는 알고리즘이다.  
MatNet을 활용한 알고리즘이 다른 **모든 알고리즘들 보다 좋은 solution**을 도출하였고, **매우 짧은 computation time**을 보임을 확인할 수 있다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232315487-eb2eed0f-e5bd-4aff-b926-7e0bf669b5cd.png" width="800px"></p>

# 5. Conclusion  
CO problem에서 많이 다뤄지는 matrix-type data를 처리할 수 있는 neural solver 개발을 위해 "front-end" model인 MatNet을 제안하였다. MatNet은 ATSP에 대해 optimal에 근접한 solution을 도출하였고, FFSP에 대해 전통적인 알고리즘들 보다 매우 우수한 성능을 보임이 확인되었다.  
Matrix-type data를 처리하기 적합한 dual attention layer를 구성한 것이 novelty를 가지긴 하지만, **Appendix A.4**의 **Table A.1**을 보면 attention layer를 dual로 구성하는 것이 single attention layer 대비 **실제로는 큰 이점이 되지 않음**을 확인할 수 있다.  
CO problem은 서로 다른 node set을 가지는 경우가 많으며, 각 node set에 대해 별도의 처리가 필요함은 분명하다. 따라서 matrix-type data를 더 효과적으로 처리할 수 있는 model 개발이 필요할 것으로 보인다.  
추가적으로 CO problem의 input data는 2D matrix를 포함하여, **3D, 4D matrix 형태의 data**를 가지는 경우도 많이 존재한다. 더 높은 차원의 data에 대하여도 MatNet과 같은 연구가 진행되어야 할 것으로 기대된다.  
<p align="center"><img src="https://user-images.githubusercontent.com/79552432/232316188-2d76e00c-90d0-4b0c-a85a-b7fc9c674eec.png" width="700px"></p>

# Author Information  
* Yeong-Dae Kwon, Jinho Choo, Iljoo Yoon, Minah Park, Duwon Park, and Youngjune Gwon
	* Samsung SDS

# 6. Reference & Additional materials  
Kool _et al._, 2019: https://arxiv.org/abs/1803.08475  
Cirasella _et al._, 2001: https://link.springer.com/chapter/10.1007/3-540-44808-X_3  
Kwon _et al._, 2020: https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html  

[Attention_TSP]: https://arxiv.org/abs/1803.08475
[ATSP_generation]: https://link.springer.com/chapter/10.1007/3-540-44808-X_3
[POMO]: https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html
