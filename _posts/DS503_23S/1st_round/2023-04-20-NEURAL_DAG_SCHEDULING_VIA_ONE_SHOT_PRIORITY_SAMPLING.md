---
title:  "[ICLR 2023] NEURAL DAG SCHEDULING VIA ONE-SHOT PRIORITY SAMPLING"
permalink: NEURAL_DAG_SCHEDULING_VIA_ONE_SHOT_PRIORITY_SAMPLING.html
tags: [reviews]
use_math: true
usemathjax: true
---

# NEURAL DAG SCHEDULING VIA ONE-SHOT PRIORITY SAMPLING

## 1. Problem Definition
본 논문에서는, 방향성이 있는 비순환 그래프(Directed Acyclic Graph, DAG)로 특징화된 작업/노드의 스케줄링 문제을 다룬다. 일반적으로 DAG와 관련된 스케줄링 문제는 DAG의 선/후행 조건을 만족하는 동시에 목적식을 최적화하는 것으로 수식화되며 job shop scheduling problem(JSSP) 등이 해당구조로 표현가능하고 많은 연구들이 진행되고 있다. 

## 2. Motivation
<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241284-b645ac36-0ea1-43cb-98b8-c08c12826e6b.png"  width="80%" height="30%">
</p>

리스트 스케줄링(List scheduling)은 simplicity로 실무에서 널리 채택되고 있는 우선순위 기반 스케줄 알고리즘로, 초기에 노드의 priority값들을 할당한 뒤 의사결정시점마다 prioirty로 순위를 결정한다 [1]. 

최근에 neural network를 활용하여 JSSP 등의 스케줄링 문제를 해결하고자 하는 시도들이 이어지고 있다. 하지만 대부분이 한번의 의사결정(어떤 작업을 먼저 시작하는 지 결정)에 encoding과 decoding을 한번씩 반복하는 구조를 가지고 있어 계산의 효율성이 떨어지며, 노드의 개수가 커졌을 때 실행시간도 함께 커진다는 단점이 있다. 이를 해결하고자 본 논문에서는 LIST scheduling 방법을 모사하여, instance가 한번의 encoder만을 통해 스케줄을 얻을 수 있는 모델을 만들고자 하였다. 

## 3. Method 
<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241287-b4028c01-f454-4c0d-8072-b338aedf504e.png"  width="80%" height="30%">
</p>

본 논문에서는 위 그림에서 볼 수 있듯, DAG 가 input으로 들어왔을 때 1) GNN 모델을 통해 logits을 구하고 2) 해당 logits에 perturbation을 주어 priority 값으로 활용한다. 이는 기존에 반복적으로 softmax 함수를 활용하는 것과 다르며, 이를 효과적으로 수행하기 위해서 Topoformer, Gumbel-Top-k trick, Reinforcement leanring 방법을 문제에 맞게 사용하였다. 특히 Reinforcement leanring에서는 logit norm regularization과 cost standardization을 통해 기존 논문보다 높은 성능을 보였다.

각 방법론들에 대해 하나씩 설명하도록 하겠다. 

### 3.1. Topoformer: Topologically Masked Attention
<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241242-0c8173e7-14be-43ce-95b7-61a81f226474.png"  width="80%" height="30%">
</p>
해당 구조는 [2] 논문에서 제시한 attention 구조로서, DAG 그래프를 인코딩 할 때에 original DAG만을 사용하는 것이 아닌 주어진 DAG에서 파생될 수 있는 다양한 형태의 그래프 구조를 만들고 multi-head-attention(MHT)에 적용하는 구조이다. 위 그림에서 볼 수 있듯이 총 4가지의 구조를 가지고 있으며 그 중에 3가지는 edge의 방향을 바꿔주어 총 7가지의 그래프에 대해 MHT을 적용하였다. 각 그래프에 대한 자세한 내용은 해당 논문에서 확인할 수 있다. 

### 3.2. Gumbel max logic 
Gumbel-Max trick 은 reparametrization tricks 중 하나로, 이 트릭을 통해 x1, ... , xn ∈ R 에 있을 때,  Gumbel(0, 1)에서 무작위추출한 g1, ... , gn 을 더해주었을 때 아래 식을 만족한다 [3].
<p align="left">
<img src="https://user-images.githubusercontent.com/83407412/232241281-5af4d8bf-3045-4fbf-8cb2-27beb761ef16.png"  width="30%" height="30%">
</p>

해당 방식을 조금 변형한 Gumbel-top k trick 이용하면 softmax 함수를 사용하지 않고 argsort만으로 같은 효과를 줄 수 있기 때문에 LIST scheduling 방법을 따라하고 싶은 저자의 입장에서 one-shot-encoding을 진행하기 위한 좋은 선택이라고 느껴졌습니다 [4]. 
<p align="left">
<img src="https://user-images.githubusercontent.com/83407412/232241269-65e8ee4d-7891-4fd8-8e0a-e03f189e0ba1.png"  width="30%" height="30%">
</p>
Gumbel-max 트릭이 argmax 만으로 softmax를 모사한다는 것에 대해서 자세한 증명 및 내용은 해당 자료에서 확인할 수 있다. 
[link](https://homes.cs.washington.edu/~ewein//blog/2022/03/04/gumbel-max/).


### 3.3.Reinforcement leanring 

많은 조합 최적화 문제는 NP-hard의 특징으로 optimal한 solution을 구하기 어렵기 때문에(label을 얻기 어려운 상황), 강화학습을 활용한 연구가 활발히 진행되고 있다. 특히 최근에는 REINFORCE 알고리즘이 많이 활용되고 있다 [6]. 
저자는 이전 논문([2])에서도 DAG 구조를 가진 문제(peak memory minimization problems in DAG)에 REINFORCE 알고리즘을 활용하여 강화학습을 진행하였는데, 이 과정에서 경험적으로 두가지 문제점을 발견하였으며 이를 해결하고자 하였다. 

### 3.3.1. Norm regularization 
[2] 에서 저자는 logits의 범위를 bound 하기 위해 각 logits를을 본인들의 평균과 분산을 이용해서 standardization 해주어 사용하였다. 
<p align="left">
<img src="https://user-images.githubusercontent.com/83407412/232241249-adfa79c9-f75d-4a3b-9461-6d1ad91e1523.png"  width="30%" height="30%">
</p>                                                           
하지만 본 논문의 저자는 standardization이 모델의 representation 능력을 제한하는 것을 경험적으로 경험했다. 특히, 하나의 간단한 아래의 예제를 통해 해당 방식의 성능 한계를 살펴보았는데, X ∈ {0, 1}, 인 경우 해당 standardization을 활용할 경우 고정된 확률값만 나타낼 수 있는 것을 쉽게 볼 수 있었다(자세한 증명은 본 논문의 Appendix A에서 확인할 수 있다.) 

본 논문에서는 위와 같이 간단한 예제를 통해 해당 standardization이 좋은 representation 능력을 가지지는 않는다는 것을 보이며, 아래와 같은 norm regularization 방법을 제시했는데, norm regularizer은 모델이 logits을 origin 주변에 위치하도록 하여, logits을 무한이 커지는 경우에 발생할 수 있는오류를 방지할 수 있다고 설명하였다. 또한, 아래와 같이 loss에 norm regularizer부분을 하이퍼 파라메터인 c_logits에 곱해서 넣어줌으로 충분한 유연성을 유지하면서 성능을 올릴 수 있었다고 주장한다.

<p align="left">
<img src="https://user-images.githubusercontent.com/83407412/232241289-a193edb3-d5d7-42d0-b512-2717f9708f80.png"  width="30%" height="30%">
</p>                                                           


### 3.3.2. Cost standardization

이 방법 역시 [2] 에서 경험적으로 성능의 악화의 원인중 하나로 생각이 되던 baseline 부분을 개선한 내용이다. [2] 에서는 많은 강화학습 알고리즘에서 사용되는 best-performing model을 저장하고 baseline으로 사용하였다. 하지만, 만약 작업 완료 시간(makespan)의 규모(scale)가 서로 다른 여러 개의 훈련 그래프(training graphs)에서 크게 차이가 나는 경우, 탐욕 기준선(greedy baseline)으로 훈련된 모델은 훈련 그래프의 작은 일부분에 대해 과적합(overfit)되기 쉽다.

따라서, 본 논문에서는 policy-gradient algorithms에서 많이 사용되는 cost standardization 과정을 통해 이 문제를 간단히 해결했는데, 여러개의 node priorities를 samping한 뒤, 아래와 같이 제일 makespan의 mean 과 std 값을 활용하여 standardization을 진행하는 것이다.(std 값의 경우 clipping 진행) 해당 알고리즘에서 한번 sampling을 진행할 때에 1000, 2000개와 같이 큰 수의 데이터를 sampling 하여 학습하기 때문에 standardization이 더 잘 적용되는 것으로 파악된다. (한번의 encoding으로 전체 trajectory를 구할 수 있기 때문에 큰 수를 샘플링 하는 게 계산적인 면에서도 부담이 되지 않은 것이라고 생각된다.)
<p align="left">
<img src="https://user-images.githubusercontent.com/83407412/232241292-a7c0da47-3c04-4b1b-84b4-cb715bdcacd5.png"  width="50%" height="30%">
</p>                                                           


## 4. Experiment 
본 논문에서는 DAG 구조를 가진 세가지 스케줄링 문제(JSSP, DAG scheduling on TPC-H dataset, scheduling on computation graphs)에 대해 다양한 노드 개수에 대해서, node 100개의 문제로 학습을 진행한 뒤, 50개의 문제에 대해서 test를 진행하는 방식으로 실험을 실험을 진행하였다. 

아래 table 에서 볼 수 있듯, 모든 문제에 대해서 성능 기존 NCO 보다 좋았으며, 저자가 처음 LIST 스케줄링을 모사하며 주장했듯 computing 시간 역시 기존 NCO 보다 훨씬 빠른 것을 볼 수 있다. 

하지만 최근에 많은 NCO 알고리즘들이 나온대 비해 SOTA 알고리즘과 비교를 하지 않고 초기 연구와 비교를 했다는 점에서 성능적으로 가장 뛰어난 NCO 알고리즘이라고 보기에는 것은 어려울 것 같다. (JSSP 의 경우 [5] 와 같이 GNN과 RL을 처음으로 적용한 초기 논문과만 성능을 비교함)

<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241261-d6915eef-cf28-48e3-89d9-0dadb9014c0c.png"  width="60%" height="30%">
</p>

### 4.1 Ablation Study
저자는 강화학습 부분에서 본인들이 norm regularization와 cost standardization 방식이 좋은 성능을 보이는 것을 주장하기 위해, norm regularization와 cost standardization 사용했을 때(본 연구)와 logits 별 standardization을 진행하고 greedy baseline을 사용했을 때([2]에서 사용한 방법)의 성능을 비교하였다.
아래 table에서 볼 수 있듯, norm regularization와 cost standardization을 사용했을 때에(본 연구) 더 좋은 결과를 얻을 수 있다는 것을 보였다.
<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241253-fc75367b-5f88-4a1f-833d-f24799eedd07.png"  width="60%" height="30%">
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/83407412/232241257-c897df1d-261d-4346-b5c5-60abd33e8ded.png"  width="60%" height="30%">
</p>

## 5. Conclusion 
본 논문에서는, DAG 스케줄링 문제를 위해, 여러 방법론들을 통합해 빠르고 좋은 성능을 보이는 모델을 제시하였다. 또한 기존 많은 neural network 기반의 스케줄링 알고리즘과 다르게 노드를 선택하는 데 있어 encoding-decoding 부분을 반복해서 진행하지 one-shot으로 진행시켜 computataional 시간을 NCO에 비해 크게 낮추었다. 하지만 SOTA 알고리즘들과 비교가 없었다는 점이 아쉬웠다.


## Reference 
[1] Ronald L. Graham. Bounds on multiprocessing timing anomalies. SIAM journal on Applied Mathematics, 17(2):416–429, 1969.

[2] Mukul Gagrani, Corrado Rainone, Yang Yang, Harris Teague, Wonseok Jeon, Roberto Bondesan, Herke van Hoof, Christopher Lott, Weiliang Will Zeng, and Piero Zappi. Neural topological ordering for computation graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[3] Emil Julius Gumbel. Statistical theory of extreme values and some practical applications: a series of lectures, volume 33. 1954.

[4] Vieira, Tim. "Gumbel-max trick and weighted reservoir sampling." (2014).

[5] Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, and Xu Chi. Learning to dispatch for job shop scheduling via deep reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), pp. 1621–1632, 2020.

[6] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.
