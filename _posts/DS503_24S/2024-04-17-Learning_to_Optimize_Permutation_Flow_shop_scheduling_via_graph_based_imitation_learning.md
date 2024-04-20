---
title:  "[AAAI 2024] Learning to Optimize Permutation Flow Shop Scheduling via Graph-based Imitation Learning"
permalink: Learning_to_Optimize_Permutation_Flow_shop_scheduling_via_graph_based_imitation_learning.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [2024 AAAI] Learning to Optimize Permutation Flow Shop Scheduling via Graph-based Imitation Learning

## 1. Introduction

- 스케줄링은 제조 시스템에서 job의 순서와 양을 배열, 제어 및 최적화하는 방법입니다. 해당 논문에서는 자주 발생하는 최적화 문제인 Permutation Flow Shop Scheduling(PFSS)을 해결하기 위해 머신 러닝을 활용합니다. PFSS는 여러 작업을 일련의 기계에서 순서대로 생산하는 것으로 가장 포괄적으로 연구된 스케줄링 문제 중 하나입니다. 일반적으로 PFSS의 목표는 작업의 최적 순열을 찾는 것으로, 총 처리 시간을 최소화하는 것을 목표로 합니다. 단일 기계의 경우, 이 PFSS 문제는 쉽게 해결할 수 있습니다. 그러나 n개의 작업과 m개의 기계(n ≥ 2)가 있는 PFSS 문제의 경우, 가능한 해는 n!개입니다. 따라서 PFSS는 NP-hard인 조합 최적화(combinational optimization, CO) 문제입니다.

- job의 갯수가 많은 large-scale PFSS 문제를 해결하기 위한 전통적인 최적화 알고리즘(branch & bound, branch & cut, local search 등)은 정확도와 계산 효율성을 모두 만족하기 어렵습니다. 그리고 최신 강화학습 방법론은 100개 미만의 job에 초점을 맞추고, 대규모 문제에서는 성능이 저하됩니다. 혹은 무거운 네트워크로 학습 시간이 오래걸리며 정확도는 개선이 필요합니다.

- 따라서 해당 논문에서는 Imitation Learning(IL)을 활용하는 것을 제안하며 이 기법을 활용해 PFSS 문제를 해결하는 최초의 연구입니다. job의 수를 1000개까지 확장하고, 그래프 구조로 더 나은 표현 능력을 보여주며 빠르고 효율적인 수렴을 가능하게 합니다.

## 3. Problem descriptions

- PFSS의 목표는 job의 makespan을 최소화하는 최적의 permutation을 찾는 것입니다.

- Input : Processing time $X_{mxn}$
- Output : optimal permutation $τ^* = [τ^*_0, τ^*_1, ... ,τ^*_{n-1}]$

- Assumption
   1) 시작시간은 0이라고 가정
   2) 하나의 machine은 하나의 job만 처리 가능
   3) 모든 작업은 각 기계에서 한 번만 처리
   4) 모든 작업은 동일한 작업 순서를 공유
   5) 작업이 시작되면 중단 될 수 없음

- Mixed-integer Programming Model
<img width="440" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/formulation.png">

- 제약 조건은 각 작업이 하나의 선후행 관계를 가지고, 한 machine은 한 번에 하나의 job만 처리할 수 있습니다. job j가 job k보다 앞서면 machine i에서 job k의 시작 시간이 작업 j의 완료 시간을 넘지 않아야 함을 나타냅니다. makespan이 마지막 machine에서 모든 작업의 오나료 시간보다 크거나 같아야 합니다. 또한, job이 이전 machine에서 처리를 되지 못했으면 다음 machine에 할당이 불가능합니다.

## 4. Proposed method

- 해당 논문에서는 PFSS을 Markov Decision Process(MDP)로 정의하고, encoder-decoder policy network를 보여줍니다. policy network 학습은 Imitation Learning(IL)을 통해 진행됩니다.

### 4.1 Markov decision process for PFSS

- PFSS 문제는 n개의 job을 스케줄링 하는 것으로 하나의 Job을 스케줄하는 것이 한 단계의 decision이라면, 하나의 n-job PFSS를 스케줄링하기 위해 만들어진 연속적인 n단계의 결정은 MDP로 볼 수 있습니다. PFSS 해결 과정을 환경으로 보고, 스케줄링 방법을 에이전드로 설정하면, state, action, policy는 다음과 같습니다.

- $state$ $s_t$ : 스케줄링의 현재 상태로 이미 스케줄된 작업과 남은 스케줄링 작업을 $[V_t, U_t]$로 표현합니다. $t \in {0,1,2,...,n}$
- $action$ $a_t$ : 스케줄이 아직 할당되지 않은 job set $U_t$에서 다음에 처리할 job index를 선택합니다. 이 action은 마마스킹 메커니즘을 포함한 policy network에 의해 결정됩니다. 마스킹 메커니즘은 이미 스케줄된 작업을 다시 선택하는 것을 방지하는데 도움이 됩니다. $t \in {0,1,...,n-1}$
- $Policy$ $π_ {\theta}(a\vert s_ {t})$ : policy π는 action이 어떻게 진행될지 결정하며, θ는 network의 weight를 나타냅니다. a는 action 집합입니다.

### 4.2 Policy network

- 해당 논문에서는 policy network로 encoder-decoder 구조를 사용하며 encoder로 Gated Graph ConvNets(GGCN)을 decoder는 attention 메커니즘을 사용합니다.

1) 그래프 인코더 사용의 동기
- 모든 작업이 동일한 특성을 가지고 있지 않으면 각 순열은 다른 makespan이 도출됩니다. 따라서 해당 논문에서는 그래프 구조를 도입하여, 한 작업의 특성을 하나의 노드로, 두 작업 간의 차이를 엣지로 사용합니다. 초기 엣지는 인접 작업 유클리드 거리의 임베딩으로 설정됩니다.
2) 그래프 인코더
  <img width="1073" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/network.png">

- n개 작업 m개 기계의 PFSS 문제에서, 입력은 모든 처리 시간을 포함하는 행렬 $X_{m×n}$입니다. 여기서 $x_{ij} ∈ X_{m×n}$은 기계 i에서 작업 j의 처리 시간을 나타냅니다. 우리는 $x_{Tj}=[x_{1j}, x_{2j}, ..., x_{mj}]$를 작업 j의 입력 특성으로 사용합니다. 그림 3에서 보이듯이, 인코더는 fully connected 그래프로 시작하여 희소화 기술을 사용해 sparse 그래프를 도출합니다. 이 과정은 고정된 그래프 직경을 사용하며 각 노드를 n×20% 최근접 이웃과 기본적으로 연결합니다

<img width="504" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/encoder.png">

- 해당 논문에서는 encoder로 GGCN을 사용합니다. GGCN을 사용하는 그래프 인코더는 각 작업 j와 간선 jk에 대한 노드와 간선의 특성을 각각 $h_{ℓj}$ 및 $e_{ℓjk}$로 나타내며, 여기서 ℓ은 해당 레이어를 나타냅니다.
- $h_{o,j} = W_hx_h와 h_{0,k} = W_hx_k$는 각각 작업 j와 k의 초기 노드 특성입니다. $e_{0,jk} = W_e​⋅∥x_j-x_k∥_2$는 초기 edge 특성입니다. $N_m(⋅)$은 정규화 방법으로 LayerNorm 또는 BatchNorm을 의미합니다. σ는 시그모이드 함수, ⊙은 하다마르 곱(요소별 곱)입니다. $N_j$는 노드 j의 이웃 노드 집합이며, $B_ℓ, C_ℓ, D_ℓ, E_ℓ, F_ℓ$은 학습 가능한 파라미터입니다.
- 위 식을 통해 각 layerd에서 node와 edge의 특성을 업데이트하며, 각 노드와 이웃 간의 관계를 반영하여 특성을 추출합니다.

3) attention 디코더 사용 동기

- PFSS는 feasibility 제약을 포함하고 있습니다. 디코딩 중에, 이미 스케줄된 작업은 다음 의사결정에서 제외돼야 합니다. 따라서 초기에 스케줄된 작업이 makespan에 더 큰 영향을 미칩니다. 이는 디코더에서 단순한 마스킹을 사용하는 것은 기술적으로 충분하지 않다는 것을 의미합니다. 따라서 전체적인 makespan 성능 향상을 위해 attention을 도입하고, 앞에 있는 Job에 집중합니다.

4) attention 디코더
- 디코더는 입력된 작업들의 집합과 이미 스케줄된 작업들의 상태를 고려하여 다음에 할당할 작업을 결정합니다. attention 메커니즘은 이미 할당된 작업에 집중하며 전체 makespan에 미치는 영향을 최적화합니다.
<img width="449" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/decoder2.png">

4-1) 초기 상태 및 컨텍스트 임베딩 설정 : 초기 상태 $s_0$는 모든 작업이 할당되지 않은 상태입니다. 각 시간 단계 t에서 디코더는 현재까지의 스케줄링 정보를 통합하여 컨텍스트 임베딩 $h_(c)$를 구성합니다. 이는 인코더에서 생성된 그래프 임베딩 $h_{(g)}$, 첫 작업의 $h_L^{τt-1}$을 포함하며 이 정보들은 향후 작업 선택에 필요한 컨텍스트를 제공합니다.
<img width="488" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/decoder3.png">

4-2) Multi-Head Attention(MHA) : MHA는 다수의 attention head를 사용하며, 다양한 정보 조각에서 중요한 정보를 추출합니다. 각 head는 다른 관점에서 정보를 처리해서 결합된 결과는 더 풍부한 정보를 반영할 수 있습니다. MHA는 쿼리(Q), 키(K), 값(V)으로 구성됩니다. MHA는 입력된 쿼리에 대해 모든 키와의 유사성을 계산하고, 이 유사성 점수를 사용하여 값들의 가중합을 계산하며 결과는 다음 작업 선택에 사용되는 컨텍스트 임베딩 $h_{(c)}$를 생성합니다.

<img width="168" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/decoder.png">

4-3) 작업 선택 및 확률 계산 : 디코더는 각 작업에 대한 logit u_{(c)j}를 계산합니다. 로짓은 특정 작업이 다음 작업으로 선택될 확률에 대한 점수를 나타내며 이 점수는 attention 메커니즘을 통해 계산됩니다. 모든 로짓에 소프트맥스 함수를 적용하여 확률 분포를 얻고, 이 분포에서 가장 확률이 높은 작업이 다음 작업으로 선정됩니다.

4-4) 일반화 : 해당 논문에서는 인코딩과 디코딩 과정을 $H_ {d×n} = W^{en}_ {d×m}X_ {m×n}$ 및 $O_ {1×n} = W^{de}_ {1×d} H_ {d×n}$으로 표현합니다. 여기서 X, H, O, d는 각각 입력, 숨겨진 표현, 출력 및 숨겨진 차원을 나타냅니다. $W^{en}$과 $W^{de}$는 각각 인코더와 디코더에 대한 네트워크 가중치를 나타냅니다. $W^{en}_ {d×m}$과 $W^{de}_ {1×d}$는 m과 d에만 관련이 있으며 n과는 무관합니다. 이는 모델이 n에 독립적이어서 다양한 작업 크기에 대해 일반화될 수 있습니다.

5) Imitation Learning
<img width="488" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/limitation_learning.png">

- 해당 논문에서는 최적화된 휴리스틱 알고리즘을 통해 얻어진 솔루션을 전문가 정책으로 설정합니다. 이 정책은 다양한 휴리스틱 방법론을 결합하여, PFSS에 대한 최적의 해결책을 제공합니다. 이후, 모방 학습은 행동 클로닝 방식을 통해 구현됩니다. 이 방법은 정책 네트워크가 주어진 전문가의 행동을 정확하게 모방하도록 학습합니다. 모방 학습에서 손실 함수는 그림과 같으며 모델이 각 시간 단계에서 전문가의 선택을 최대한 재현할 수 있도록 만드는 것입니다. 손실 함수를 최소화 함으로써, 모델은 주어진 상태에서 전문가의 행동을 선택할 확률을 극대화합니다. 결과적으로 모델은 전문가의 행동 패턴을 정확하게 학습하고, 복잡한 스케줄링 문제에 높은 성능을 발휘할 수 있도록 돕습니다.

## 6. Experiments
- 실험은 tesnorflow를 사용하여 구현되었고, 모든 실험은 동일한 하드웨어 조건에서 수행되었으며 각 알고리즘은 동일한 조건 하에서 최적의 성능을 낼 수 있도록 조정되었습니다. 데이터셋은 벤치마크 문제들로 구성되어 있으며, 다양한 크기와 복잡성을 가진 PFSS 문제를 포함하고 있습니다. 데이터셋은 작업 수가 20개 ~ 500개까지 다양하며, 각 작업은 5개 ~ 20개의 기계에서 처리됩니다. 성능 평가는 makespan을 최소화하는 값을 기준으로 수행했으며 학습 시간, 수렴 속도 및 안정성 등이 평가 지표로 사용되었습니다. 

<img width="1079" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/experiment.png">

<img width="1099" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/experiment2.png">

<img width="1054" alt="image" src="../../images/DS503_24S/Learning_to Optimize_Permutation_Flow_shop_scheduling_via_graph_based imitation_learning/experiment3.png">

- 해당 논문에서 제안된 모델은 강화 학습 방법과 비교할 때, 네트워크 파라미터를 기존의 37% 수준으로 줄이면서 optimal makespan과 모델의 makespan 간격이 6.8%에서 1.3%로 대폭 개선하며 최적해에 가까워졌습니다. 해당 모델은 학습 기반 방법 중에서 가장 좋은 성능을 보였으며 1000개의 작업이 있는 문제에서 기존 강화 학습 방법 대비 makespan을 평균 1.3% 향상시키며 대규모 문제에서도 해결이 가능한 것이 증명되었습니다. 또한, 해당 모델은 학습 과정에서 안정적이며, 다양ㅇ한 규모의 데이터셋에서 빠르게 수렴합니다. 

## 7. Conclusion
- Permutation Flow Shop Scheduling(PFSS) 문제는 광범위한 적용 가능성을 가진 중요한 조합 최적화 문제입니다. 최근 몇 년간 학습 기반 방법이 점점 더 많은 관심을 받고 있지만, 기존 연구는 주로 강화 학습 방법에 의존하고 있습니다. 이 논문에서는 그래프 기반 모방 학습 방법을 사용하여 PFSS 문제를 해결하는 방법과 그 이유를 소개하며, 이 방법은 최신 강화 학습 방법에 비해 네트워크가 가벼우면서 수렴 속도가 빠르고 안정적이며 makespan 간격이 낮습니다. 생성된 데이터셋과 벤치마크 데이터셋에서 실시한 실험은 제안된 방법의 경쟁력을 분명히 보여줍니다.
