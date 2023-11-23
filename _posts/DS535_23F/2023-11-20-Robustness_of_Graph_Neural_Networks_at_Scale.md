---
title:  "[NIPS 2021] Robustness of Graph Neural Networks at Scale"
permalink: 2023-11-20-Robustness_of_Graph_Neural_Networks_at_Scale.html
tags: [reviews]
use_math: true
usemathjax: true
---


## **0. Background**

-   Adversarial Attack on GNN : Adversarial Pertubation을 적용해 기존 분류기 또는 GNN 모델의 성능을 낮추는 것을 말한다.
-   Testing Phase Attack : 분류 모델에 대해서 간섭을 주진 않지만 모델이 정확하게 동작하지 않도록 오동작을 유발하는 공격이다. 공격을 수행할 시 사용 가능한 지식의 양에 따라 White Box, Black Box Attack으로 분류한다.
    -   White Box Attacks : 분류에 사용되는 model에 대한 모든 지식을 가지고 있는 공격이다.
    -   Black Box Attacks : model에 대한 정보를 모르는 공격자가 행하는 공격으로 input과 이것이 주는 output을 관찰함으로써 수행될 수 있다.

논문에서는 `testing phase` + `white box` 조건에서 `Adversarial Robustness`를 달성하고자 아래 방법론들을 제시한다.

## **1. Problem Definition**

Graph Neural Networks가 Adversarial Perturbations에 Robust하지 않다는 점은 발견이 되나, 이를 중점으로 한 연구들은 거의 없다. 예를 들어 PubMed Graph Dataset의 경우, 인접 행렬을 기반으로 한 공격을 위해선 약 20GB의 메모리가 필요하다. 이런 메모리 요구는 GNN의 Robustness를 확인해보고자 하는 실질적인 연구를 어렵게 한다.

## **2. Motivation**

해당 논문은 GNN의 Adversarial robustness에 대한 기존 연구의 한계점들을 다음과 같이 명시했다 : 
	(1) 기존의 Loss는 Global Attack에 적합하지 않음
	(2) GNN Attack에 소요되는 비용이 $O(n^2)$이상으로 매우 큼
	(3) 기존의 Robust GNN은 scalable하지 않음.

따라서, 위의 한계점들을 (1) **Surrogate loss**를 제시함으로써, (2) R-BCD를 기반으로 하여 **시간복잡도를 $O(\Delta)$로 줄이는 방법을 제시함**으로써, 마지막으로 (3) **Soft Median으로 효과적으로 GNN을 방어하는 방법을 관찰함**으로써 해결하고자 했다.

## **3. Method and Experiment**

### 3.1 Surrogate Losses for global attacks

GNN의 Global Attacks들은 평균 Cross Entropy Loss(CE)를 증가시킨다. 그러나 Node가 많은 Large Graph들의 경우 CE loss는 효율적이지 않다. Accuracy가 낮아질 때, CE loss 는 증가할 것이다. 그러나 CE의 경우, Accuracy가 감소하지 않아도 증가할 때가 있다.

![Untitled.png](https://i.ibb.co/w7fSMqc/Untitled.png)
$\psi$는 Classification Margin으로 값이 클수록 Classification을 Confident하게 수행하는 것을, 값이 작을수록 Confident하지 않게 수행하는 것을 의미한다. 즉, 이 값이 작을 수록 misclassified될 가능성이 크다.
 위의 그림은 1% of nodes를 가지고 feature perturbation을 준 모습이다. CE Loss는 Tanh margin에 비해서 Classification margin이 낮은 node들에 대해 Attack을 가하며 Budget $\Delta$를 사용하는 모습을 볼 수 있다. ($\Delta$ : Perturbed된 인접행렬의 변경된 Entry의 수)
 ![Untitled-99.png](https://i.ibb.co/TLW11Mw/Untitled-99.png)
위 그래프는 각 데이터셋에 대한 CE, CW, Tanh margin loss에 대한 실험결과이다. CE, CW는 tanh margin loss에 비해서 loss, accuracy의 그래프의 결과가 안정적이지 못하고 그 성능이 덜한 것을 확인할 수 있다. 


그래서 이들은 `Surrogate loss`를 새롭게 정의한다. 

**Definition of Surrogate Loss:** Global Attack에 대한 Surrogate Loss $L^\prime$는

(1) 옳게 분류된 perturbed node들에 대해서만 incentive를 주고, $\frac{\partial L^\prime}{\partial z_ c^*} \vert \psi_ 0 = 0$

(2) Decision Boundary 근처에 있는 node들을 선호하는 방식으로 Loss를 구성한다.

$\frac{\partial L^\prime}{\partial z_ c^{\*}} \vert \psi_ 1 < \frac{\partial L^\prime}{\partial z_ c^{\*}} \vert \psi_ 2 \; for \, any \,0<\psi_ 1<\psi_ 2$

위 정의에서 도출된 정의에 따르면, Cross Entropy Loss는 (1)을 위반하여 Global Optimum을 가질 수 없기에 Surrogate loss가 될 수 없다. 또한, Carlini-Wagner(CW) Loss는 Decision boundary에 있는 node들을 고려하지 못하기에 (2)을 위반한다.


논문은 Surrogate loss에 맞는 loss로써 `Masked Cross Entropy(MCE)`를 제안한다.

$MCE = \frac{1}{\vert V^+ \vert} \sum_{i \in V^+} -\log(p^{(i)}_ {c^*})$

MCE는 Projected Gradient Descent attack(PR-BCD)에 대해서는 Cross Entropy와 큰 차이를 보이지 않지만, Greedy Gradient-Based attack에 있어서는 강점을 보이는 것을 확인할 수 있다. 따라서, 이후 실험에서는 Greedy Attack에선 MCE를 사용하고 다른 경우는 Tanh margin loss를 사용한다. 



### 3.2 Scalable Attacks

Gradient-based attacks들은 인접행렬 $A$에 대해서 모두 최적화하기에 $\Theta(n^2)$의 공간 복잡도를 보여 Large Graph에 대한 robustness는 대개 측정하기 어려웠다.

Large-scale Optimization을 위해 R-BCD(Randomized Block Coordinate Descent)를 사용함으로써 변수들의 부분집합에 대해서만 gradients를 구하기 때문에 사용되는 메모리와 수행 시간을 줄일 수 있다.

Perturbation $P\in{0,1}^{n*n}$ 는 아래와 같이 모델되었다.

$\max_{P\; \text{s.t.}\; P \in \{0, 1\}^{n \times n},\; \sum P \leq \Delta} L(f_\theta(A \oplus P, X))$
여기서 $\oplus$는 element-wise XOR을 나타내고 $\Delta$는 Edge budget을 의미한다.

그러나 $n^2$만큼의 parameter를 위 식에서 저장해야 하기때문에$O(n^2)$ 만큼의 공간 복잡도를 가진다. 이에 논문은 `Projected Randomized Block Coordinate Descent(PR-BCD)`를 제안한다.

![Untitled-1](https://i.ibb.co/Qb9hygL/Untitled-1.png)

**Explanation of PR-BCD :** P는 이산 Edge 행렬로 element p는 edge를 뒤집을 확률을 나타낸다. 우선, epoch마다 P에서 무작위로 추출된 Block을 바탕으로 특정 부분의 edge들만을 변경한다. 업데이트 후 p에 대한 확률 질량함수를 수정하여 베르누이 분포에 대한 기댓값이 Budget을 넘지 않도록 한다. 
 PR-BCD에서는 block size로 공간 복잡도를 줄이고 효율적인 공격을 가능하게한다. 효율적인 공격을 위해서 전체 가능한 edge들을 탐색할 필요가 없다고 생각했다. Figure 3-(a)를 보면 상대적으로 작은 Dataset인 Cora ML에서는 작은 block size($b$)로도 낮은 Accuracy를 달성할 수 있었다. 그리고 Figure 3-(b)를 통해 여러 block size에 대해서 실험해본 결과 향상된 Attack을 위해서는 Epoch의 수를 늘리는 것이 중요한 것을 확인할 수 있었다. 
 이로써 PR-BCD로 이전에는 $\mathcal{O}(n^2)$가 필요했던 것을 $\mathcal{O}(b)$로 줄일 수 있어 효과적인 Attack을 시행할 수 있었다. 

![Utitled-98.png](https://i.ibb.co/P5hzD8b/Utitled-98.png)

그리고 논문은 PR-BCD에 대한 또 다른 대안으로 위와 같이 `GR-BCD`를 함께 제안한다. PR-BCD에서 Block 추출 시 가장 큰 gradient를 가진 entry만 **greedy**하게 변경하는 것으로 E번의 epoch 후에 budget이 충족되도록 하는 방법이다. 

그러나 위 방법들은 실제 최적화 문제를 얼마나 효과적으로 근사하는지에 대한 보장은 제공하지 않고, 공격의 효과의 Upper bound만 보여준다는 점에서 한계가 있다.

위 방법론에서 확장하여 더 큰 그래프들을 사용하기 위해, `PPRGo`를 사용했다. 이는 Personalized Page Rank(PPR) Matrix($\Pi$)를 사용함으로써 explicit message passing steps 수를 1로 줄여 constant complexity를 가질 수 있게 한다.

$p = softmax \big[\text{AGG}\{\Pi_{uv}(A_{uv}, f_\text{enc}(x_u)\big), \; \forall u \in \mathbb{N}^\prime(v)\}\big]$

### 3.3 Scalable Defense

GNN의 메세지 패싱 프레임워크를 다음과 같이 표현할 수 있다.

$h^{(l)}_ v = \phi^{(l)} \big[( \text{AGG}^{(l)}\{\big(A_ {uv}, h^{(l-1)}_ u W^{(l)}\big), \quad \forall u \in \mathbb{N}^\prime(v)\}\big]$  where $\text{neighborhood} \; \mathbb{N}^\prime(v) = \mathbb{N}(v) \cup v$ 
and $AGG = \text{l-th layer message passing aggregation}$ 
and $h^{(l)}_ v = \text{embedding},\; \sigma^{(l)} \text{activation function}$


이전 논문인 Geisler et al. (2020)에서는 Aggregate Function으로 Soft Medoid를 다음과 같이 제안했다 : $\tilde{f}_ {\text{WSM}}(X, a) = c(s \circ a)X \quad where  \quad s_ i = \frac{\exp \left( -\frac{1}{T} \sum_ {j=1}^{n} a_ j ||x_ j - x_ i|| \right)}{\sum_ {q=1}^{n} \exp \left( -\frac{1}{T} \sum_ {j=1}^{n} a_ j ||x_ j - x_ q|| \right)}$
 Soft Medoid는 $s_i$로 표현되는 이웃 노드들의 embedding에 대한 distance matrix에 대해 행/열 summation을 요구하기에 neighborhood size에 대해 이차 복잡도를 지닌다. 

이를 개선하기 위해 논문은 `Soft Median`을 제시한다.

$\mu_\text{SoftMedian}(X) \\= \text{softmax}(\frac{-c}{T\sqrt{d}} )^\intercal \cdot \mathbf{X} = \mathbf{s}^\intercal \cdot \mathbf{X} \approx \arg \min_{x^\prime \in \mathbf{X}} \|x_{\bar{}} - x^\prime\|$

이로써 Dimension($d$)에만 의존함으로써 계산을 효율적으로 할 수 있다. $d$를 충분히 작게 한다면 Soft Median 스케일은 $\mathbb{N}$에 linear한 scale을 보인다.

![Untitled-97.png](https://i.ibb.co/g6v6dk2/Untitled-97.png)

위의 그래프는 원본 그래프와 Pertubed된 그래프를 첫 번째 message passing을 한 후의 latent space의 $L_2$ Distance를 나타낸 그래프이다. Soft Median이 Weighted Sum보다 20% 가까이 낮은 error를 보인다. 반면, 그래프에는 Soft Medoid가 Soft Median보다 Robust한 결과를 가지는 것으로 보인다. 그러나 오른쪽 표가 나타내듯 Adversarial accuracy에서는 Soft Medoid가 Soft Median보다 좋은 성능을 보이지 못했다.

이처럼 Soft Median은 (1) 차원에 대한 고려 없이 단순히 Summation 한다면 computation 비용이 많이 든다는 점과 (2) Soft Median에 비해 robust하지 않은 결과와 마찬가지로 잘못된 robustness의 결과를 가져올 수 있다는 점을 한계점으로 삼을 수 있다. 그러나 PPRGo와 결합하여 사용한다면 위 한계를 완화할 수 있다고 설명한다.

### 3.4 Empirical Evaluation

#### Surrogate Loss

![Untitled-2](https://i.ibb.co/Tr2CFsN/Untitled-2.png)
Figure 5는 logit의 classification margin을 가지고 loss들을 분류한 결과이다. CE, margin이 low margin에 Incentive를 주는 경향을 보이고, CW, NCE, elu margin들은 high confidence nodes들에 집중하는 경향을 보인다. 그리고 Surrogate loss에서 강조했듯 Decision boundary에 있는 node들에 집중하는 MCE, Tanh margin을 확인할 수 있다.
Figure 6는 Pubmed Dataset에 Loss들을 달리하여 Attack 후 Accuracy를 측정한 결과이다. 위에서 살펴본 바와 같이 MCE와 Tanh margin의 경우 Adversarial accuracy를 낮추는데 효과적이었고, 특히 MCE는 Greedy attack을 시행했을 때 더욱 효과적임을 볼 수 있다. 

#### Robustness w.r.t global attacks.

![Untitled-3](https://i.ibb.co/bFrdktP/Untitled-3.png)

![Untitled-4](https://i.ibb.co/b6P7Xwd/Untitled-4.png)

위 실험 결과는 Pubmed, arXiv, Products Dataset 각각에 대해 $\text{budget} = \Delta$만큼의 공격 후에 adversarial accuracy를 측정한 결과이다. 약 2%가량 Pertubation을 가했을 때, 대략 60%의 정확도로 떨어지는 모습을 확인할 수 있었다고 밝혔다. 각 경우에서 Soft Median GDC, Soft Median PPRGo, PPRGo Defense가 타 모델보다 비교적 좋은 결과가 보임을 확인할 수 있다.

#### Robustness w.r.t local attacks.

Cora ML, Citeseer, arXiv에서 PR-BCD가 SGA보다 효과적으로 공격을 하는 것을 볼 수 있었다. 그리고 아래 그래프가 보이듯 Soft Median PPRGo가 위 공격에 대해서 Vanila PPRGo와 Vanila GCN보다 대체로 잘 견디는 것을 확인할 수 있었다. (b)의 Papers100M에서 $\Delta_i = 0.25$일 때, Soft Median PPRGo는 공격자의 성공률을 90%에서 30%로 줄인 것을 그래프에서 볼 수 있다.

![Untitled-5](https://i.ibb.co/m0cRXCv/Untitled-5.png)

![Untitled-6](https://i.ibb.co/b5xBzvx/Untitled-6.png)

## 4. Conclusion

규모가 있는 GNN에서의 Adversarial Robustness를 살펴보았다. 논문은 이전까지 데이터셋의 규모로 인해 잘 다뤄지지 않았던 문제를 확인하기 위해 Attack과 Defense에 대해 직접 방법을 제시하고 실험한 결과를 보였다.

따라서, Complexity를 줄이기 위해 PPRGo와 Soft Median이라는 방법을 도입하고 Attack과 Defense에 반영하여 실제로 큰 데이터셋에 대한 실험결과를 낸 것을 보며 문제상황에 대해 여러 모델을 활용하여 다각도로 실험해본 것을 확인할 수 있었다. 그러나, 그들이 강조한 PPRGo에 대한 모델 설명이 타 모델에 비해서 부족한 점과 white box과 attack budget에 대한 가정이 실제 적용될 수 있는 부분인지에 대한 의문이 남아있다.


----------

**Author Information**

-   Sumin Lee
-   Affiliation: Dept. of Data Science, KAIST
-   Research Topic: Machine learning
-   Contact : [sumlee@kaist.ac.kr](mailto:sumlee@kaist.ac.kr)

**Reference**

-   Geisler, Simon, et al. "Robustness of graph neural networks at scale." _Advances in Neural Information Processing Systems_ 34 (2021): 7637-7649.
-   Bojchevski, Aleksandar, et al. "Scaling graph neural networks with approximate pagerank." _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_. 2020.
-   Geisler, Simon, et al. "Reliable graph neural networks via robust aggregation" _In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS'20)_