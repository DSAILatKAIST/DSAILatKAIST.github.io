---
title:  "[KDD 2021] Learning Process-consistent Knowledge Tracing"
permalink: Learning_Process-consistent_Knowledge_Tracing.html
tags: [reviews]
---

# Learning Process-consistent Knowledge Tracing


# 1.INTRODUCTION

Knowledge Tracing은 학생의 문제(exercise) 혹은 문제의 개념(Knowledge concept)의 지식의 정도(knowledge state)가 시간에 따라 어떻게 변화하는지, 다음 문제를 맞출 수 있을지 없을지 예측하는 Task입니다. 예를 들어 산타토익과 같은 문제 풀이 앱을 활용할 때 기존 사용자가 푼 문제 1~6를 학습해 사용자의 학습 수준을 유추하고 이를 활용하여 문제 7,8,9,10의 정오답을 예측하는 것입니다. 



# 2.MOTIVATION 

본 논문에서 한계점으로 지적하는 부분은 이전 논문들은 모두 성능을 향상시키는 데만 집중하고, 높은 성능을 보이는 곳은 문제의 개념(Knowledge concept)의 지식의 정도(knowledge state)를 잘 추정할 수 있다는 가정을 가지고 있습니다. 하지만 모델의 성능을 향상시키는 데에만 치중하면 학생의 지식과 학습 과정 사이의 불일치를 초래할 수 있습니다. 
학생의 지식과 학습 과정 사이의 불일치는 두 가지가 존재합니다. 이해를 돕기 위해 Figure 1을 보겠습니다.

![figure1](https://user-images.githubusercontent.com/59594139/232191765-79b081ca-142e-42ec-9c2a-94f652a96030.png)

\* 학생이 문제를 틀리면 해당 개념에 대한 지식이 없다고 판단하여knowledge state가 감소합니다. 예를 들면, KC2에 대한 knowledge state가 e4 문제를 풀 때 0.74이었는데, e5 문제를 틀렸을 때 0.69으로 감소하는 것을 볼 수 있습니다. 그러나 실제로는 문제를 틀렸다고 해서 해당 학생의 지식이 반드시 감소하는 것은 아닙니다. 


\* KC와 관련이 없는 문제를 풀었지만, KC의 knowledge state가 감소한 경우도 있습니다. 예를 들면, e1~e4는 모두 KC1과 관련된 문제이지만 KC2과 KC3의 knowledge state가 업데이트됩니다. 그러나 실제로는 Absolute Value와 관련된 문제를 푼다고 해서 Ordering Integers의 지식이 학습되지는 않습니다. 


본 논문은 지식 추적(Knowledge Tracing) 문제를 해결하기 위해 학생들의 학습 과정을 직접 모델링하는 새로운 패러다임을 제시합니다. 이를 위해서는 학생들의 학습 과정을 정의하고 모델링하기 위한 적절한 형태로 변환하는 방법, 학습 효과를 측정하는 방법, 지식 상태의 감소도 고려하는 방법 등 다양한 문제가 존재합니다. 따라서 LPKT라는 새로운 방법론을 제시하고 이를 통해 기존 방법들보다 더 합리적이고 해석 가능한 지식 상태 추적 결과를 얻을 수 있다는 것을 실험을 통해 입증하였습니다.



# 3.METHOD

## 3.1 PRELIMINARY
 
LPKT의 몇 가지 중요한 임베딩을 제시합니다. 학습 프로세스는 연속된 문제 풀이 행동을 반복하며, 연속된 행동 간에는 간격이 있습니다. 따라서 학생의 학습 과정은 
$x = {(e_1,~at_1,~a_1),it_1,(e_2,~at_2,a_2),it_2,\ldots,(e_t,~at_t,~a_t),~it_t}$ 와 같은 학습 순서로 표시됩니다. 여기서 $e_t,~at_t,~a_t$ 튜플은 기본 학습 셀을 나타낸다.  

\* $e_t$: 연습 문항

\* $at_t$: 학생이 $e_t$를 대답하는 데 사용한 시간 (문제 푸는 시간)

\* $a_t$: 이진 정확성 레이블 (1은 맞고 0은 틀린 것)

\* $it_t$: 학습 셀 간의 간격 시간 (다음 문제를 풀기까지 걸린 시간)

Knowledge Tracing은 학생의 학습 과정에서 지식 상태의 변화를 모니터링하고, 다음 학습 단계 t + 1에서 학생의 미래 성과를 예측하는 것을 목표로합니다. 이는 개별화된 학습 방식을 적용하고 학습 효율성을 극대화하는 데 사용될 수 있습니다. 

## 3.2 Embedding

### 3.2.1 Time Embedding 
	
시간 임베딩은 학생들의 학습 과정에서 중요한 역할을 하는 답변 시간과 간격 시간을 임베딩하는 것을 의미합니다. LPKT에서는 간격 시간이 답변 시간보다 훨씬 길기 때문에 전자는 분 단위로 후자는 초 단위로 이산화하고, 1개월 이상의 간격 시간은 1개월로 설정합니다. 

### 3.2.2. Learning Embedding 
	
연습 문항 집합을 임베딩 행렬로 표현하고, 각각의 연습 문항과 답변 시간, 답변을 합쳐서 학습 임베딩을 만드는 방법을 제안하고 있습니다.

\* $l_t = W_1 [e_t \oplus \text{at}_t \oplus a_t] + b_1$
		
### 3.2.3 Knowledge Embedding 
	
LPKT에서는 학생의 학습 과정에서 학생의 지식 상태를 저장하고 업데이트하는 지식 임베딩이 사용됩니다. 지식 임베딩은 지식 개념의 수에 해당하는 M x dk 차원의 임베딩 행렬 h로 초기화됩니다. 학습 상호작용마다 LPKT가 모델링하는 각 지식 개념에 대한 학습 이득이 지식 임베딩에 업데이트되며, 동시에 지식 상태의 잊어버림 효과도 포함됩니다. Q-행렬은 연습 문제와 지식 개념 간의 관계를 나타내며, 해당 연습 문제를 푼 후 지식 임베딩의 해당 행이 업데이트됩니다. 오류와 주관적 편향성이 불가피하므로, 새로운 행렬 q를 도입하여 이를 보완합니다.


## 3.3 Module 
![figure2](https://user-images.githubusercontent.com/59594139/232191902-9bec2e62-5b12-4bf5-be26-2d5cfe556010.png)


Figure 2와 같이, LPKT는 각 학습 단계에서 세 가지 모듈로 구성됩니다: 

\* Learning Module : 이전 학습 상호작용과 비교하여 학습 효과를 모델링합니다.

\* Forget Module :시간이 지남에 따라 얼마나 많은 지식이 잊혀질지를 측정하는 데 사용됩니다. 이후 학습 효과와 망각된 지식은 학생의 이전 지식 상태를 업데이트하기 위해 활용됩니다.

\* Prdect Module : 학생의 최신 지식 상태에 따라 다음 연습문제에서의 성적을 예측하기 위해 사용됩니다.


### 3.3.1 Learning Module 
![figure3](https://user-images.githubusercontent.com/59594139/232191949-03b4bcbc-c350-45c3-a2e1-50a2837db5e4.png)


Learning gain modeling을 위해 학생의 이전 학습 임베딩 l_{t−1}과 현재 학습 임베딩 l_{t}을 연결하는 방식을 채택합니다. 그러나 두 연속적인 학습 임베딩으로 학생들의 성능 차이를 포착할 수 있지만, 학습 과정에서 학생들의 Learning gain 다양성을 포착할 수 없습니다. 예를 들어, 동일한 연속적인 학습 임베딩(즉, 겹치는 학습 시퀀스의 일부에서 동일한 성능을 가진)를 가진 학생들도 모두 동일한 Learning gain을 가지는 것은 아닙니다. 따라서, LPKT에서 Learning gain의 두 가지 영향 요소(문제 간격 시간과 학생의 이전 지식 상태)입니다. 두 학습 셀 간의 간격 시간은 학습 과정에서 중요한 요소이며, 학생들은 일반적으로 간격 시간이 짧을수록 더 많은 지식을 습득하므로, 그들의 학습 과정은 밀접하고 연속적이다는 것을 의미합니다.


\* $l_{t} = \tanh\left(W_{2}^T \left[l_{t-1} \oplus it_{t} \oplus l_{t} \oplus \hat{h}_{t-1} \right] + b{2}\right)$

\* $\Gamma_{t}^{l} = \sigma\left(W_{3}^T \left[l_{t-1} \oplus it_{t} \oplus l_{t} \oplus \hat{h}_{t-1} \right] + b{3}\right)$

\* $LG_{t} = \Gamma_{t}^{l} \cdot \left(\frac{lg_{t} + 1}{2}\right),\qquad LG_{t} = qe_{t} \cdot LG_{t}$

### 3.3.2 Forgetting Module 

![figure4](https://user-images.githubusercontent.com/59594139/232192028-44031dfa-9a99-4801-8f34-fa405f2027c3.png)


$LG_t$를 계산한 후, 학생들의 지식 상태에 추가된 역할을 하는 것으로, 시간이 지남에 따라 잊혀지는 지식의 영향을 반영합니다. Forget curve theory에 따르면, 학습한 자료 중 기억하는 양은 시간이 지남에 따라 지수 함수적으로 감소합니다. 그러나 단순한  지수 감소 함수는 지식 상태와 간격 시간 간의 복잡한 관계를 고려하기에는 충분하지 않습니다. 복잡한 시간의 망각을 모델링하기 위해, 본 논문은 LPKT에서 forget gate $Γ^f_t$를 설계했는데, 이는 학생들의 세 가지 요인을 기반으로 지식 매트릭스에서 상실 정보의 정도를 학습하기 위해 MLP를 적용합니다:

(1) 학생들의 이전 지식 상태 $h_{t−1}$.

(2) 학생들의 현재 학습 획득 $LG_t$.

(3) 간격 시간 $it_t$이다.

\* $\Gamma_{t}^{f} = \sigma\left(W_{4}^T \left[h_{t-1} \oplus LG_{t} \oplus it_{t} \right] + b{4}\right)$

\* $h_t = LG_t + \Gamma^{f}_{t} · h_{t−1}$


### 3.3.3 Predicting Module

![figure5](https://user-images.githubusercontent.com/59594139/232192054-fd5ad4f4-0fdd-4ce1-902d-2834bac10598.png)

앞선 두개의 module을 통해 파악 할 수 있는 학생의 learning gain(지식 상태)와 망각의 효과를 사용하여 학생의 지식 상태 $h_t$를 계산합니다. 계산된 $h_t$를 통해서 학생의 다음연습문제 $e_{t+1}$에 대한 풀이 여부를 예측 할 수 있습니다. 	

\* $y_{t+1} = \sigma\left(W_{5}^T \left[e_{t+1} \oplus h_t \right] + b{5}\right)$
	

# 4.EXPERIMENT

## 4.1 RQ1 Does our proposed LPKT model keep the consistency of students’ changing knowledge state to their learning process? 

![figure6](https://user-images.githubusercontent.com/59594139/232192074-ab59100a-2e81-4e0a-b5bd-b6c42c22a7de.png)

LPKT는 학생의 학습 과정을 추적하여 학생의 지식 상태 변화를 모델링하는 방법입니다. LPKT를 사용하면 학생들이 틀리거나 맞은 학습 상호작용에서 얻은 학습 지식을 습득할 수 있으며, 학생이 특정 개념을 풀지 않으면 해당 개념에 대한 지식 상태가 점차적으로 감소한다는 것을 확인할 수 있습니다. 또한 학생의 지식 상태 변화 과정은 그의 학습 과정과 일관성이 있으며, 학생이 새로운 지식을 습득함에 따라 지식 상태가 증가하고, 마지막 학습 상호작용에서는 일정 수준의 감소가 나타난다는 것을 알 수 있습니다. Figure 1과 Figure 3를 통해 LPKT가 학생의 지식 상태 변화를 추적하는 능력을 보여줍니다.

## 4.2 RQ2 Does our proposed LPKT model outperform the state-ofthe-art knowledge model on student performance prediction?

![table1](https://user-images.githubusercontent.com/59594139/232192149-9d15f58d-f3b8-4ecc-9392-30773da34bc1.png)


Table 2에서는 모든 벤치마크 모델과 LPKT를 학생 성적 예측에서 비교하고 5개의 test fold 에서의 평균 결과를 보여줍니다. 모든 데이터셋에서 실험을 수행하여 모든 모델의 성능을 종합적으로 평가하기 위해 RMSE, AUC, ACC 및 Pearson 상관계수(r2) 제곱으로 성능을 평가합니다.


## 4.3 RQ3 How does the learning module, forgetting module, and time information in LPKT impact the knowledge tracing result?

![table2](https://user-images.githubusercontent.com/59594139/232192198-31ca4e48-e143-41a0-a21b-d16e2309c3c7.png)

LPKT에는 3가지 변형이 있으며, 각각은 LPKT의 하나의 모듈을 제거하여 만들어졌습니다. 실험 결과, Forgetting 모듈을 고려하지 않는다면 예측 결과가 가장 크게 감소하며, Learning gain을 고려하는 것이 Learning outcome만 고려하는 것보다 더 좋은 예측 결과를 보입니다. Answer time과 Interval time 정보는 빠뜨리면 학습 과정을 정확하게 모델링하기 어렵습니다. 또한, LPKT는 AKT보다 학습 시퀀스가 짧을 때 더 나은 결과를 보이며, 이는 LPKT가 학습 과정을 더 잘 모델링하기 때문입니다.


## 4.4 RQ4 Can LPKT learn meaningful representations of exercises? 

![table3](https://user-images.githubusercontent.com/59594139/232192238-38881888-684e-48e4-bc80-9a40272e0442.png)


LPKT는 exercise embeddings를 무작위로 초기화하고 학습합니다. 그 결과로, 학습된 exercise embeddings는 의미 있는 cluster로 나뉘며 이는 교육 전문가들에게 유용한 정보를 제공할 수 있습니다. Figure 5에서는 ASSISTchall 데이터셋에서 100개의 exercise embeddings을 T-SNE 방법을 이용하여 시각화한 결과를 보여줍니다. 이를 통해 비슷한 KC를 가진 문제가 유사한 부분에 위치 한것을 알 수 있습니다. 따라서 LPKT는 exercise embeddings을 어느정도 잘하고 있다고 판단 할 수 있습니다


# 5.CONCULSION

본 논문에서는 학생들의 학습 과정을 모델링하는 새로운 지식 추적 패러다임을 탐구하고 Learning Process-consistent Knowledge tracing (LPKT)이라는 새로운 모델을 제안하였다. 학생들의 학습 과정을 효과적으로 모델링하기 위해 learning cell, interval time, learning gate,  forgetting gate 등을 도입하였고, 이를 바탕으로 다양한 실험을 진행하여 LPKT이 다른 KT 방법보다 더 높은 정확도와 해석 가능성을 보이며 학습 과정을 모델링하는 새로운 연구 방향을 제시하였다.

## AUTHOR INFORMATION
M.S. Kim,JonWwoo
Graduate School of Data Science, KAIST
Knowledge Innovation Research Center

# 6.REFERENCE
Shen, S., Liu, Q., Chen, E., Huang, Z., Huang, W., Yin, Y., ... & Wang, S. (2021, August). Learning process-consistent knowledge tracing. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 1452-1460).

