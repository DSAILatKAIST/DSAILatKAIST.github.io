---
title:  "[KDD 2021] A Multi-Graph Attributed Reinforcement Learning based Optimization Algorithm for Large-scale Hybrid Flow Shop Scheduling Problem"
permalink: A_Multi_Graph_Attributed_Reinforcement_Learning_based_Optimization_Algorithm_for_Large_scale_Hybrid_Flow_Shop_Scheduling_Problem.html
tags: [reviews]
use_math: true
usemathjax: true
---
# Title
A Multi-Graph Attributed Reinforcement Learning based Optimization Algorithm for Large-scale Hybrid Flow Shop Scheduling Problem

# 1. Introduction
IT 기술이 발달하면서, 제조시스템에서도 자동화 창고(automated warehouse) 시스템을 구축하는 것이 가장 큰 화두이다. 일반적으로 고객의 주문(order)은 다양한 제품들로 구성되어 있는데, 이들은 창고의 정보시스템에 의해 다양한 종류 및 크기의 상자들로 포장된다. 이 과정에서 각 제품들은 picking, packing, 그리고 weighting의 세가지 스테이지를 거치고 난 뒤에 비로소 창고에서 출하된다. 이를 효율적으로 수행하기 위해 각 스테이지는 여러개의 프로세서 혹은 기계들을 가지며, 각각의 제품은 해당 스테이지에서 그 중 한 개의 기계에서 작업이 수행된다. 이 과정을 효율적으로 스케쥴링하여 각 제품의 작업 완료시간을 줄이고 창고의 처리량(throughput)을 극대화 하는 것이 매우 중요하다.

이러한 자동화 창고의 주문 스케쥴링은 Hybrid Flow Shopt Scheduling Problem(HFSP)로 모델링이 가능하며, 해당 문제는 NP-hard임이 증명되어있다. 이를 극복하기 위해 제시된 해법들은 크게 exact algorithm과 휴리스틱 기반 알고리즘으로 구분된다. 다만, exact algorithm은 해당 문제의 높은 복잡도와 큰 규모로 인해 거의 사용되지 않는다. 휴리스틱 기반 접근법은 크게 일반적인 휴리스틱과 메타휴리스틱으로 나뉘며, 일반적인 휴리스틱은 빠르지만 좋은 해를 보장하지 못하고, 메타 휴리스틱은 도메인 지식에 크게 의존해야하는 문제점이 있다. 이를 극복하기 위해 몇몇 연구들이 learning 메커니즘을 도입하였지만, 모두 improving procedure없이 end-to-end 방식으로 해를 직접 구하고자 하여 한계가 존재한다.

이를 개선하기 위해 저자들은 Multi-Graph attributed Reinforcement learning-based Optimization algorithm(MGRO)하는 새로운 방법론을 제시한다.


# 2. Problem definition

HFSP 문제는 $n$개의 job이 있고, 이들이 거쳐야 하는 $M$개의 스테이지들이 있으며, 각 스테이지 $s$에는 $I_s$개의 기계 혹은 프로세서들이 할당되어있다. 각 스테이지는 1개 이상의 기계가 할당되어 있으며, 최소한 한개 이상의 스테이지는 2개 이상의 기계가 할당되어 있다. 즉, 모든 스테이지가 1개의 기계가 할당되어있는 경우는 가정하지 않는다.

모든 job들은 순차적으로 스테이지를 거쳐야 하며, 해당 스테이지의 기계들 중 하나의 기계에서 처리되면 다음 스테이지로 옮겨지게 된다. 즉, 같은 스테이지의 기계들은 모두 동일하다. 기계들은 한번에 하나의 job만 처리할 수 있으며, 처리 중에 멈추고 다른 job을 처리하는 것이 불가능하다. 또한, 스테이지 $s$에서 job $j$의 처리시간(processing time)은 $p_{sj}$로 나타내며 이는 다음 스테이지로 job이 옮겨지는 시간까지 포함한다.

이러한 조건에서 주어진 job을 효율적으로 처리할 수 있는 스케쥴링(각 스테이지별 job의 배정 및 job 처리 순서 결정)이 본 논문의 목표이며 이를 측정하는 지표는 makespan(공정이 완전히 끝날때까지 걸리는 시간)의 최소화이다.

# 3. Method

## 3.1 Multi Graph Formulation
저자들은 간트 차트를 directed multi-graph(다수의 disconnected graph로 구성된 그래프)로 표현하여 주어진 해를 효과적으로 표현한다. 간트 차트에서 각 스테이지 별로 multi-graph를 구성하는데, 각 multi-graph는 해당 스테이지에 할당된 기계의 개수 만큼 disconneted된 graph를 구성한다. 각각의 multi-graph의 노드는 job을 의미하고, 스테이지 s에서 각각의 기계들의 그래프는 해당 그래프에서 처리되는 job의 순서를 나타낸다. 즉, edge $e_{ij}$는 노드 $v_i$가 나타내는 job 다음으로 노드 $v_j$가 나타내는 job이 처리됨을 의미한다. 아래는 그 예시이다.

![image](https://user-images.githubusercontent.com/67723054/233354353-80c061bd-935e-44f6-81a1-b2835f658aa7.jpg)

스테이지 $k$의 multi-graph를 $G_k=(V,E_k)$라고 할때, 각 job 노드 $v_i$가 스테이지 $k$에서 가지는 6가지 feature를 다음과 같이 디자인 할 수 있다. $I_{ik}=$ 스테이지 $k$에서의 idle time, $W_{ik}=$ 스테이지 $k$에서의 waiting time, $A_{ik}=$ 스테이지 $k$에서의 arrival time, $S_{ik}=$ 스테이지 $k$에서의 starting time, $C_{ik}=$ 스테이지 $k$에서의 completion time, $P_{ik}=$ 스테이지 $k$에서의 processing time. 이 여섯가지를 한번에 나타내기 위해 다음과 같이 feature 벡터를 구성한다. $h_i^k=(I_{ik},W_{ik},A_{ik},S_{ik},C_{ik},P_{ik})$.

이렇게 인코딩 된 간트차트를 가장 널리 쓰이는 그래프 신경망인 Graph Convolutional Network(GCN)에 적용하는데, multi-layer 노드 임베딩은 다음과 같이 업데이트된다. 먼저, 노드 임베딩의 첫번째 layer는 $(h_i^k)^0=h_i^k$로 초기화 된다. 다음 $l$번째 layer번째 노드 임베딩은 $(h_i^k)^{l+1}=\sigma(W^l((m_i^k)^{l+1}+(h_i^k)^l))$로 업데이트 되는데, $(m_i^k)^{l+1}$는 노드 $v_i$와 연결된 job들의 aggregated representation이고, $W^l$은 GCN의 모델 파라미터, $\sigma()$는 activation function이다. 이때, $(m_i^k)^{l+1}$는 다음과 같이 계산된다. $(m_i^k)^{l+1}=\sum_{u\in N^k(i)}{(h_u^k)^l}$ where $N^k(i)=$노드 $v_i$와 연결된 job들의 집합.

이를 통해 스테이지 k의 그래프 feature 임베딩 $\hat{h_k}$을 다음과 같이 구할 수 있게 된다.
$$
\hat{h_k}=Readout(\{(h_i^k)^{l+1}:i\in|V|\})
$$

## 3.2 Attention-based Weighted Pooling
위의 과정으로 구해진 그래프 임베딩 $H=[\hat{h_1},...,\hat{h_M}]$은 스테이지 개수 $M$에 따라 길이가 변동된다. 때문에 고정된 임베딩이 필요한데 이를 위해 저자들을 Attention-based Weighted Pooling(ABWP)를 제안한다. 이는 self-attention 메커니즘과 adaptive weighted pooling을 결합한 방식으로, 크게 Multi-self Attention Layer 부분과 Adaptive Weighted Pooling과정으로 나눌 수 있다. 전체적인 과정은 다음과 같다.

![image](https://user-images.githubusercontent.com/67723054/233353280-c19cb0f9-f846-4c02-84f9-bca4db3c55c3.png)

#### 3.2.1 Multi Self-Attention Layer

그래프 임베딩 $H=[\hat{h_1},...,\hat{h_M}]$를 이용해 다음과 같이 $d$차원의 분포를 구할 수 있다.
$$F=softmax(\frac{(HW^Q)(HW^K)^T}{\sqrt(d)})(HW^V)$$
이를 MLP에 적용 후 ReLU 함수를 이용해 nonlinearity를 더하는데, 이 과정에서 오버피팅을 막기 위해 dropout을 이용한다. 그리고 self-attention operation에서 나타나는 transmission loss를 보정하기 위해 $F$를 추가로 더해준다.
$$E=F+ReLU(MLP(F))=SAN(H)$$
이 과정을 $L$번 반복하여 policy learning을 위한 더 좋은 feature representation을 얻는다. 
$$E^{(L)}=SAN(E^{(L-1)})$$

#### 3.2.2 Adaptive Weighted Pooling
위에서 구해진 multi-layer attention network를 MLP에 통과시킨 후 다시 한번 softmax 분포 $W$를 구한다. 이때 $W$는 trainable weight parameter이다.
$$W=Softmax(MLP(E^{(L)}))$$
마지막으로 final 임베딩 $\hat{H}$을 구한 뒤 이를 policy network에 입력한다.
$$
\hat{H}=W^TE^{(L)}\\
P=softmax(MLP(\hat{H}))
$$

이를 이용하여 action을 선택하게 된다.

## 3.3 Reward Shaping
각 액션에 대해 즉시 reward를 주게 되면 단기적으론 불리하지만 장기적으로는 local optima에서 빠져나오도록 하는 액션들을 무시하게 된다. 이를 해결하기 위해 저자들은 매 스텝마다 나타나는 makespan의 trajectory를 이용하는데, trajectory 상에 나타나는 local minima를 이용하여 전체 trajectory를 sub-segment로 나누는 것이다. 다만 실질적으로 local optima가 너무 많기 때문에 이를 해결하기 위해 local optima 중 monotonically decreasing local optima들만 이용하여 sub-segment를 구하여 reward를 구성하게 된다. 해당 그림은 다음과 같다.

![image](https://user-images.githubusercontent.com/67723054/233353476-3da663ad-84a0-4d37-affb-cfdf758751fa.png)
</p>
Local optima point(local optima가 나타나는 step)의 집합을 $S$, 그중 monotonically decreasing한 point들을 $\hat{S}$이라고 한다면, 각각의 스텝 $i$마다 shaped reward는 다음과 같이 계산된다.

$r_ i=\frac{O_ T-O_ 1}{T-1}+\frac{O_ {I_ {p-1}}-O_ {I_p}}{I_ p-I_ {p-1}}\textit{ for } i \in [I_ {p-1},I_ p)$

첫째항은 전체 trajectory의 global reward를, 둘째항은 sub-segment의 local reward를 나타낸다.

# 4. Experiment

실험에 사용되는 데이터는 empirically generated testbed와 화웨이의 Supply Chain Business Unit의 실제 데이터 두 가지로 구분된다. Empirically generated testbed는 480개의 instance로 구성되어 있는데, job과 스테이지의 개수에 따라 서로 다른 48개의 instance set으로 나뉜다. 즉, 각각의 instance set은 동일한 job 및 스테이지의 개수를 가지는 10가지 서로 다른 instance들을 가진다. job의 개수의 범위는 10개에서 240개 까지이며, 스테이지의 개수의 범위는 5개에서 20개까지이다. 화웨이의 데이터는 주문량으로 구분이 되는데, 주문량이 50개에서 500개가지 있으며, 각 주문에 할당되는 job의 개수는 달라질 수 있기 때문에 주문량이 적어도 전체 job의 개수는 많아질 수 있다. 주문량이 동일한 instance들을 50개씩 모아서 하나의 instance set을 구성한다.

비교 알고리즘으로는 NEH, IG, 그리고 BDS를 선택하였고 이는 각각, 휴리스틱, improved 알고리즘, 그리고 다른 learning-based 알고리즘을 대표한다. 그 외에도  480개의 instance들의 현재까지 알려진 best known 값과 비교한다. 화웨이 데이터에는 lower bound를 구하여 비교한다. RL 알고리즘은 Proximal Policy Optimization(PPO) 알고리즘을 사용한다.

## 4.1 Results

Empirically generated testbed에서의 실험 결과는 아래 두 표에 나와있다. 해당 표는 각 instance set의 알고리즘별 평균 makespan을 보여준다. 모든 instance set에서 MGRO가 가장 좋은 결과를 보여준다. Best-known과의 average gap를 비교한다면, MGRO의 gap은 2.42%이고 IG의 gap은 4.04%이다. 이는 MGRO가 기존 gap을 절반가까이 줄인 것으로 해석할 수 있다.

![image](https://user-images.githubusercontent.com/67723054/233353485-c8efcef4-72e8-4e9c-9fe0-9b32b09a1079.png)
![image](https://user-images.githubusercontent.com/67723054/233353408-eef669ef-5be8-42e0-80bd-014c61fd558a.png)

실제 산업 데이터로 실시한 실험 결과는 아래 표에 나와있다. 해당 표는 동일하게 각 instance set의 알고리즘별 평균 makespan을 보여준다. 역시나 모든 instance set에서 MGRO가 가장 좋은 결과를 보여준다. Lower bound와의 average gap를 비교한다면, MGRO의 gap은 3.38%이고 IG의 gap은 7.30%이다. 이는 MGRO가 기존 gap을 절반이상 줄인 것으로 해석할 수 있다. 또한, 알고리즘 별 평균 계산시간은 NEH는 6230s, BDS는 1025s, IG는 2102s, MGRO는 2078s 였으며, NEH는 정해진 시간 내에 해를 도출하는데 실패하는 경우가 있었다. end-to-end 알고리즘 특유의 빠른 속도를 가지는 BDS를 제외하고 IG와 MGRO가 실제 산업현장에서 요구하는 시간을 맞출 수 있었으며, 그 중 MGRO가 정해진 시간 내에 가장 좋은 solution을 찾아내었다. 

![image](https://user-images.githubusercontent.com/67723054/233353470-678c3438-7cba-4b8f-95b5-f2c19b6485d8.png)

## 4.2 Ablation Study

Ablation study란 제시된 알고리즘의 개별 구성 요소의 중요도를 알기 위해, 각 요소를 제거하고 제거 전후의 성능차이를 비교하는 방법이다. 해당 논문에서는 policy network, attention-based weighted pooling(ABWP), reward shaping, 그리고 generalization 네 가지 요소의 효과성을 실험한다.

##### Policy Network

논문에서는 solution을 개선시키는 몇가지 basic operator를 제시하고 MDP모델의 action으로 정의한다. 이때 policy network가 제대로 학습되는지 알아보기 위해 policy network없이 임의의 operator를 선택하도록 하여 실험을 진행하였고, 실험결과 성능이 확연히 차이가 나는 것을 확인하였다. 아래 그림은 job의 개수가 240개인 네 개의 instance set에서 실험한 결과이다.

![image](https://user-images.githubusercontent.com/67723054/233353449-20124558-ce43-4868-8c87-f305724b3802.png)

##### Attention-based weighted pooling(ABWP) and reward shaping

논문에서 제시한 pooling 방법과 reward shaping의 효과를 알아보기 위해 이를 기존 vanilla mean-pooling방식 및 vanilla immediate reward 방식으로 각각 대체한 뒤 실험을 진행하였다. 두 가지 실험 모두 동일하게 실제 산업 데이터 중 주문량이 500개인 instance set을 실험 대상으로 하였다. 실험 결과는 아래와 같다.

![image](https://user-images.githubusercontent.com/67723054/233353421-96b9f00d-c30e-4b9f-82fe-0e087d7d8845.png)

##### Generalization ability

Generalization ability란 training data에서 벗어난 data에서도 좋은 성능을 보여주는 지표이다. 저자들은 제시된 알고리즘의 generalization ability를 검증하기 위해 testbed 실험 instance 중에서 small-size instance에 training한 후 large-size instance에서 테스트하거나, testbed에서 training한 후 실제 산업데이터에 테스트한다. 실험 결과는 아래와 같다.

![image](https://user-images.githubusercontent.com/67723054/233353414-5a4b1ea7-cf86-4f28-a6a1-32af612d486c.png)

각각의 테스트들은 비교군으로 해당 test instance에 맞춰서 training 시킨 모델의 성능도 포함하였다. 실험 결과, 전체적으로 generalization이 잘 이뤄진 것을 확인할 수 있었으며, testbed와 실제 산업데이터 간의 gap이 약간 더 컸지만 비교 알고리즘과는 여전히 좋은 성능을 보여줬다.

# 5. Conclusion
해당 논문에서는 대규모 hybrid-flow shop scheduling problem(HFSP) 문제를 풀기 위해 learning 기반 알고리즘인 multi-graph attributed reinforcement learning-based optimization(MGRO)를 제안하였다. Instace solution이 가지는 간트차트를 멀티그래프로 재구성한 뒤 graph neural network(GNN)을 적용해 solution과 instance의 feature를 캡쳐한다. 알고리즘의 성능을 높이기 위해 기존 pooling 및 reward 방식이 아닌 attention-based weighted pooling 및 reward shaping 방식을 제안하였다.

Testbed 및 실제 산업데이터에서 실험결과 MGRO는 제한시간내에 다른 알고리즘 대비 매우 뛰어난 성능을 보여주었으며, 복잡하고 대규모 문제에서도 효과적으로 solution을 탐색하는 것을 확인하였다.