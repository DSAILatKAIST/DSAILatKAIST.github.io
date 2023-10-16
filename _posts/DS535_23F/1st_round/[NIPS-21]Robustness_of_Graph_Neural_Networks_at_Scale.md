
#  Robustness of GNN at Scale

  

#  0. Background

  

-  Adversarial Attack on GNN?

	-  Adversarial Pertubation을 적용해 기존 분류기 또는 GNN 모델의 성능을 낮추는 것을 말함.

	-  이는 현실에 존재할 수 있는 이미지에 대해서는 정상적으로 동작하도록 설계되었지만 사람이 일부 변화를 취한 Adversarial Examples(Perturbed된 결과물들)에 대해서는 취약할 수 있음.

-  Testing Phase Attack : 분류 모델에 대해서 간섭을 주진 않지만 모델이 정확하게 동작하지 않도록 오동작을 유발하는 공격. 공격을 수행할 시 사용 가능한 지식의 양에 따라 공격을 분류

	-  White Box Attacks : 분류에 사용되는 model에 대한 모든 지식을 가지고 있는 공격

	-  Black Box Attacks : model에 대한 정보 모르는 공격자가 행하는 공격. 조작된 input과 이것이 주는 output 관찰함으로써 이용될 수 있음.

  

#  **1. Problem Definition**

  

  Graph Neural Networks가 Adversarial Perturbations에 Robust하지 않다는 점은 발견이 되나, 이를 실험하고 해결하고자 한 연구들은 거의 없다. 예를 들어 PubMed라는 GNN Dataset의 경우, 인접 행렬을 기반으로 한 공격에 있어서 20GB의 메모리가 필요하며 이러한 경우 공격을 실행하기 어렵다. 이는 GNN의 Robustness를 확인해보고자 하는 연구가 진전되는 것을 어렵게 한다.

  

#  **2. Motivation**

  

해당 논문은 GNN의 Adversarial robustness에 대한 기존 연구의 한계점들을 다음과 같이 명시했다.: (1) 기존의 Loss는 Global Attack에 적합하지 않다는 점과 (2) GNN Attack에 소요되는 비용이 $O(n^2)$이상으로 매우 큰 점 그리고 (3) 기존의 Robust GNN은 scalable하지 않다는 점이다.

따라서, 위의 한계점들을 (1) **Surrogate loss**를 제시함으로써, (2) R-BCD를 기반으로 하여 **시간복잡도를 $O(\Delta)$로 줄이는 방법을 제시함**으로써, 마지막으로 (3) **Soft Median으로 효과적으로 GNN을 방어하는 방법을 관찰함**으로써 해결하고자 했다.

  

#  **3. Method and Experiment**

##  3.1 Surrogate Losses

**Why previous loss invalid?**

많은 GNN의 Global Attacks들은 평균 Cross Entropy Loss를 증가시킨다. 그러나 node가 많은 Large Graph들의 경우 위의 loss는 효율적이지 않다. 

  

###  Surrogate loss

  

**(정의 1)** Global Attack에 대한 Surrogate Loss $L^\prime$은

  

(1) 옳게 분류된 pertubing node들에 대해서만 incentive를 주고,

  

$\frac{\partial L^\prime}{\partial z_c^*}|\psi_0 = 0$

  

(2) Decision Boundary 근처에 있는 node들을 선호하는 방식으로 Loss를 구성한다.

  

$\frac{\partial L^\prime}{\partial z_c^*}|\psi_1 < \frac{\partial L^\prime}{\partial z_c^*}|\psi_2 \; for \, any \,0<\psi_1<\psi_2$

  

위 정의에서 도출된 명제에 따르면, Cross Entropy Loss는 (1)을 위반하여 Global Optimum을 가질 수 없기에 Surrogate loss가 될 수 없다. 또한, Carlini-Wagner(CW) Loss는 Decision boundary에 있는 node들을 고려하지 못하기에 (2)을 위반한다.

  

논문은 Surrogate loss에 맞는 loss로써 Masked Cross Entropy(MCE)를 제안한다.

  

$MCE = \frac{1}{|V^+|}  \sum_{i \in V^+}  -\log(p^{(i)}_{c^*})$

  

MCE는 Projected Gradient Descent attack(PR-BCD)에 대해서는 Cross Entropy와 큰 차이를 보이지 않지만, Greedy Gradient-Based attack에 있어서는 강점을 보이는 것을 확인할 수 있다.

  

![2023-10-15-17-52-14.png](https://i.ibb.co/gJFYDfd/2023-10-15-17-52-14.png)

  

**(정의 2)** 정의 1을 더 확장하여 Surrogate Loss $L^\prime$는

  

(1) 확실하게 잘못 분류된 노드($\psi$가 -1에 가까운 노드)들에 대해서 확실하게 구분하고,

  

$\lim_{\psi  \to  -1^+} L^\prime < \infty$

  

(2) 정의 1과 마찬가지로 Decision Boundary 근처에 있는 point들을 선호하는 방식으로 Loss를 구성한다.

  

$\frac{\partial L^\prime}{\partial z_c^*}|\psi_1 < \frac{\partial L^\prime}{\partial z_c^*}|\psi_2<0 \; \\ for \, any \; 0<\psi_1<\psi_2<1 \; \\ or \; -1<\psi_2<\psi_1<0$

  

##  3.2 Scalable Attacks

  

Gradient-based attacks들은 인접행렬 $A$에 대해서 모두 최적화하기에 $\Theta(n^2)$의 시간 복잡도를 보여 Large Graph에 대한 robustness는 대개 측정하기 어려웠다.

  

Large-scale Optimization을 위해 R-BCD(Randomized Block Coordinate Descent)를 사용함으로써 변수들의 부분집합에 대해서만 gradients를 구하기 때문에 사용되는 메모리와 수행 시간을 줄일 수 있다.

  

Perturbation $P\in{0,1}^{n*n}$ 는 아래와 같이 모델되었다.

  

$\max_{P\; \text{s.t.}\; P \in \{0, 1\}^{n \times n},\; \sum P \leq  \Delta} L(f_\theta(A \otimes P, X)) \\ \Delta = \text{edge budget}$

  

그러나 $O(n^2)$ 만큼의 공간복잡도이기에 논문은 Projected Randomized Block Coordinate Descent(PR-BCD)를 제안한다.

  
![img2.png](https://i.ibb.co/XsZ1v5K/img2.png)

P는 이산 Edge 행렬로, 각 element(p)는 edge를 뒤집을 확률을 나타낸다. 우선, epoch마다 P에서 무작위로 추출된 Block을 바탕으로 특정 부분의 edge들만을 변경한다. 업데이트 후 p에 대한 확률 질량함수를 수정하여 베르누이 분포에 대한 기댓값이 Budget을 넘지 않도록 한다. 그리고 논문은 PR-BCD에 대한 또 다른 대안으로 **GR-BCD**를 함께 제안한다. PR-BCD에서 Block 추출 시 가장 큰 gradient를 가진 entry만 **greedy**하게 변경하는 것으로 E번의 epoch 후에 budget이 충족되도록 하는 방법이다. 그러나 위 방법들은 실제 최적화 문제를 얼마나 효과적으로 근사하는지에 대한 보장은 제공하지 않고, 공격의 효과의 Upper bound만 보여준다는 점에서 한계가 있다.

  

위 방법론에서 확장하여 더 큰 그래프들을 사용하기 위해, PPRGo를 사용했다. 이는 Personalized Page Rank(PPR) Matrix($\Pi$)를 사용함으로써 explicit message passing steps 수를 1로 줄여 constant complexity를 가질 수 있게 한다.

  

$p = softmax \big[\text{AGG}\{\Pi_{uv}(A_{uv}, f_\text{enc}(x_u)\big), \;  \forall u \in  \mathbb{N}^\prime(v)\}\big]$

![img3.png](https://i.ibb.co/s6chCC0/img3.png)

  

##  3.3 Scalable Defense

  

GNN의 메세지 패싱 프레임워크를 다음과 같이 표현할 수 있다.

  

$h^{(l)}_v = \phi^{(l)}  \big[(  \text{AGG}^{(l)}\{\big(A_{uv}, h^{(l-1)}_uW^{(l)}\big), \quad  \forall u \in  \mathbb{N}^\prime(v)\}\big] \\ \\ \text{where} \; \text{neighborhood} \; \mathbb{N}^\prime(v) = \mathbb{N}(v)  \cup v \\\\\text{and} \; AGG = \text{l-th layer message passing aggregation} \\ \text{and} \; h^{(l)}_v = \text{embedding}, \sigma^{(l)}  \text{activation function}$

  

Geisler et al. (2020)에서는 Aggregate Function으로 Soft Medoid를 제안했다. Soft Medoid는 이웃 노드들의 embedding에 대한 distance matrix에 대해 행/열 summation을 요구하기에 neighborhood size에 대해 이차 복잡도를 지닌다.

  

이를 개선하기 위해 논문은 Soft Median을 제시한다.

  

$\mu_\text{SoftMedian}(X) = \text{softmax}(\frac{-c}{T\sqrt{d}}  )^\intercal  \cdot  \mathbf{X} = \mathbf{s}^\intercal  \cdot  \mathbf{X}  \approx  \arg  \min_{x^\prime  \in  \mathbf{X}} \|x_{\bar{}}  - x^\prime\|$

  

이로써 Dimension($d$)에만 의존함으로써 계산을 효율적으로 할 수 있다. $d$를 충분히 작게 한다면 Soft Median 스케일은 $\mathbb{N}$에 linear한 scale을 보인다.

![img4.png](https://i.ibb.co/5vY3Fdk/img4.png)

  

위 왼편의 그래프는 원본 그래프와 Pertubed된 그래프를 첫 번째 message passing을 한 후의 latent space의 $L_2$ Distance를 나타낸 그래프이다. Soft Median이 Weighted Sum보다 20% 가까이 낮은 error를 보인다. 반면, 그래프에는 Soft Medoid가 Soft Median보다 Robust한 결과를 가지는 것으로 보인다. 그러나 오른쪽 표가 나타내듯 Adversarial accuracy에서는 Soft Medoid가 Soft Median보다 좋은 성능을 보이지 못했다.

  

이처럼 Soft Median은 (1)차원에 대한 고려 없이 단순히 Summation 한다면 computation 비용이 많이 든다는 점과 (2)Soft Median에 비해 robust하지 않은 결과와 마찬가지로 잘못된 robustness의 결과를 가져올 수 있다는 점을 한계점으로 삼을 수 있다. 그러나 PPRGo와 결합하여 사용한다면 위 한계를 완화할 수 있다고 설명한다.

  

##  3.4 Empirical Evaluation

  

###  Robustness w.r.t global attacks.

  

아래 실험 결과는 Pubmed, arXiv, Products Dataset 각각에 대해 $\text{budget} = \Delta$만큼의 공격 후에 adversarial accuracy를 측정한 결과이다. 약 2%가량 Pertubation을 가했을 때, 대략 60%의 정확도로 떨어지는 모습을 확인할 수 있었다고 밝혔다. 각 경우에서 Soft Median GDC, Soft Median PPRGo, PPRGo Defense가 타 모델보다 비교적 좋은 결과가 보임을 확인할 수 있다.

![img5.png](https://i.ibb.co/x18qbsH/img5.png)

  

###  Robustness w.r.t local attacks.

![img6.png](https://i.ibb.co/Y2pV6Lp/img6.png)

  

Cora ML, Citeseer, arXiv에서 PR-BCD가 SGA보다 효과적으로 공격을 하는 것을 볼 수 있었다. 그리고 아래 그래프가 보이듯 Soft Median PPRGo가 위 공격에 대해서 Vanila PPRGo와 Vanila GCN보다 대체로 잘 견디는 것을 확인할 수 있었다. (b)의 Papers100M에서 $\Delta_i = 0.25$일 때, Soft Median PPRGo는 공격자의 성공률을 90%에서 30%로 줄인 것을 그래프에서 볼 수 있다.

  

#  4. Conclusion

  

규모가 있는 GNN에서의 Adversarial Robustness를 살펴보았다. 논문은 이전까지 데이터셋의 규모로 인해 잘 다뤄지지 않았던 문제를 확인하기 위해 Attack과 Defense에 대해 직접 방법을 제시하고 실험한 결과를 보였다.

  

따라서, Complexity를 줄이기 위해 PPRGo와 Soft Median이라는 방법을 도입하고 Attack과 Defense에 반영하여 실제로 큰 데이터셋에 대한 실험결과를 낸 것을 보며 문제상황에 대한 해결 의지를 느낄 수 있었다. 공격자가 Model을 다 안다는 가정(White-box attack)을 하고 Budget($\Delta$)를 중점으로 Complexity를 해결했던 것이 현실에서 어떻게 적용이 될 수 있을지를 찾아보고 싶어졌다.

  

---

 
-  Geisler, Simon, et al. "Robustness of graph neural networks at scale." *Advances in Neural Information Processing Systems* 34 (2021): 7637-7649.

-  Bojchevski, Aleksandar, et al. "Scaling graph neural networks with approximate pagerank." *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2020.