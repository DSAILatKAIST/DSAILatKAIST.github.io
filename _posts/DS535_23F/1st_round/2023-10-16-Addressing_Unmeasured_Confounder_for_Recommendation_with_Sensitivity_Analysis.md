---
title:  "[KDD 2022] Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis"
permalink: 2023-10-16-Addressing_Unmeasured_Confounder_for_Recommendation_with_Sensitivity_Analysis.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [KDD 2022] [Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis](https://dl.acm.org/doi/pdf/10.1145/3534678.3539240)

## **1. Introduction**

추천 시스템은 social media나 e-commerce 등 다양한 영역에서 중요한 역할을 맡고 있다. 이들은 아래와 같이 

> __“if recommending an item to a user, what would the feedback be”__

이라는 질문에 에 답을 할 수 있어야 하는데, 이를 causal inference의 언어로 해석해보자면 system exposure( 사람들에게 노출되는 정도 )라는 treatment가 user feedback이라는 outcome에 어떠한 영향을 주는지 ( 엄밀하게는 causal effect )에 대한 것으로 생각해볼 수 있다. 하지만 causal effect를 historical data로부터 직접적으로 얻는 것은 confounding bias를 겪을 수 있는데, 이는 treatment와 output에 둘 다 영향을 주는 confounder 때문에 둘 사이의 causal effect를 정확하게 측정할 수 없음을 의미한다. 깊은 이해를 위해 예시를 들자면, 아이템의 인기도는 system exposure(treatment) 줄 수 있고(인기도의 불균형에 따른 아이템의 접근성 조절), 동시에 user feedback(outcome)에도 영향을 주게 되는 경우(군중 심리)를 추천시스템에서 대표적으로 볼 수 있다. 이러한 confounder의 영향을 무시하게 될 경우, 인기 있는 아이템만을 과하게 추천하게 되는 popularity bias를 야기할 수 있기에 이러한 ( __confounding__ ) __bias__ 를 잡는 것이 추천 시스템에서 중요하다고 볼 수 있다.

하지만 앞서 말한 confounder의 경우 기술적인 어려움이나 프라이버시 정책으로 인해 데이터를 구하기 어려운 상황에 자주 놓이게 되는데, 이로 인해 unmeasured confounder를 잘 다루는 것이 현업에서 중요하다고 볼 수 있다. 위 논문에서는 Causal Inference의 대표적인 방법론인 propensity-based method를 깊이 있게 활용하여, 효과적으로 unmeasured confounder를 다룸으로써 추천시스템의 약점을 보완하고자 한다 ( 좀 더 자세한 내용은 뒤에서 다룰 것이다 ).

---

## **2. Problem Formulation**

U와 I는 set of users and items를 의미한다. Key components의 경우,

> - Unit : a user-item pair $(u,i)$
> - Target population : the set of all user-item pairs $D=U\times I$
> - Feature : the feature $x_ {u,i}$ describes user-item pair $(u,i)$
> - Treatment : $o_ {u,i} \in {1, 0}$. It is the exposure status of $(u,i)$, where $o_ {u,i}=1$ or $0$ denotes item $i$ is exposed to user $u$ or not
> - Outcome : the feedback $r_ {u,i}$ of user-item pair $(u,i)$
> - Potential outcome : $r_ {u,i}(o) \; \text{for}\; o \in {0,1}$. It is the outcome that would be observed if $o_ {u,i}$ had been set to $o$

* 여기서 Potential outcome이 무엇인지 궁금한 독자들이 있을 것이다. 이는 Causal Inference의 잘 정돈된 세팅인 potential outcome framework를 구성하는 성분이다. 이러한 개념의 등장 배경을 살펴보자면, 현실 세계에서 실제로 일어난(observed) 사건의 경우, (여기서는 특정 item의 exposure를 생각해볼 수 있다) 하나밖에 관찰되지 않고, 우리는 그 사건이 다르게 진행되었다면 어떻게 결과가 바뀔지에 대해 관심이 있는 것이기에 (일어나지 않은) 잠재적인 상황에 대한 결과를 표현하기 위해 potential outcome이라는 개념을 만들어 표현하고자 한 것이다. 이에 대해 좀 더 자세히 알고 싶은 경우 (https://www.youtube.com/watch?v=C15mZUnN7Ng)를 참고하기를 바란다.

P와 E를 target population (이를 random variable로써 간주할 것이다)에 대한 distribution과 expectation으로 두고, exposed units을 모아둔 set $O = { (u,i) : (u,i) \in D, o_ {u,i}=1 }$을 생각하자. 추천에서는 $E(r_{u,i}(1) \vert x_ {u,i})$ (given feature $x_ {u,i}$에 대하여, item $i$가 user $u$에게 노출되었을 때 feedback에 대한 기댓값)을 구하고자 하는 것이다. 이때 unmeasured confounders 또한 고려해주어야 하기에, measured confounders $x_ {u,i}$ ( feature는 우리가 알고 있기에 measured confounders로 활용해볼 수 있다 )와 unmeasured confounders $h_ {u,i}$를 고려해주면 다음과 같은 관계식을 얻어볼 수 있다.

> $o_{u,I} \perp r_ {u,i}(1) \vert (x_ {u,i}, h_ {u,i}), \; o_ {u,I} \not\perp r_ {u,i}(1) \vert x_ {u,i}$

이는 아래의 causal graph에서 그 의미를 확인해볼 수 있다. ( 여기서 causal graph는 Random Variable 간의 causal relation을 directed graph의 형태로 나타낸 것으로써, 좀 더 자세히 알고 싶은 독자는 (https://www.youtube.com/watch?v=rbZ4ebZCHMY)를 참고해보기를 바란다. )

![Causal Graph](https://i.ibb.co/Hq9fwMJ/Figure-2.png)

즉, $x_ {u,i}$ 만 알고 있을 경우 confounding effect로 인해 $o_ {u,i}$ 와 $r_ {u,i}(1)$의 causal effect 관계를 정확히 파악할 수 없지만, $h_ {u,i}$도 알고 있다면 둘의 causal effect 관계를 파악할 수 있다는 의미로 해석해볼 수 있다.

이러한 상황 하에서 결국 우리는 $E(r_ {u,i}(1) \vert x_ {u,i})$를 예측하는 recommender model $f_ {\phi}$을 학습하고자 하는 것이다. 이때 만약 모든 potential outcomes ${ r_ {u,i}(1) : (u,i) \in D }$ 을 관측했다면, 이상적인 loss function은

> $ L_ {ideal}(\phi) = \frac{1}{D}\sum_{(u,i)\in D}e_ {u,i} $

이 되고, 이때 $e_{u,i} = (\hat{r}_ {u,i}(1) – r_ {u,i}(1))^2$이다. 하지만 우리는 우리는 $o_ {u,i}=1$일 때만 (실제로 user $u$에게 item $i$가 추천된 경우 ) $r_ {u,i}$를 접근할 수 있기에, robust estimator를 만들어 $L_ {ideal}(\phi)$를 잘 근사하는 것이 최종 목표가 될 것이다.

---

## **3. Problem Definition**

먼저 위 논문에서는 현존하는 propensity-based methods의 경우 unmeasured confounders가 존재할 경우 여전히 bias 문제가 존재함을 아래의 Motivation에서 밝힌다.

### **3.1 Motivation**

Propensity based model인 IPS와 doubly robust (DR) learning의 경우 confounding bias를 완화하는 방법론은 맞지만, unmeasured confounders가 존재할 때는 이야기가 달라진다.
우선 nominal propensity score $p_ {u,i}=P(o_ {u,i}=1| x_ {u,i} )$ 를 정의하자. 이는 given feature에 대하여, user $u$에게 item $i$가 expose될 확률이다. 이때 $p_ {u,i}$는 정확히 알 수 없으므로 예측이 되어야 하기에 이를 $\hat{p}_ {u,i}$라 하고, prediction error $e_{u,i}$ 또한 마찬가지로 이에 대한 estimator $\hat{e}_ {u,i}$라고 하자. 그러면 IPS와 DR의 estimator를 구하는 상황에서, ( 좀 더 자세한 설명과 증명은 논문을 참고하기를 바란다. )

> __Theorem 3.1__
> Unmeasured confounders $h$가 ( Causal Graph 상에서 ) 있을 때,
> (a) $\hat{p}_ {u,i}$과 $\hat{e}_ {u,i}$ 이 정확히 estimate와 일치한다고 하더라도, IPS와 DR estimator가 biased.
> (b) 만약 우리가 true propensity score를 $\tilde{p}_ {u,i} = P(o_ {u,i}=1 \vert x_ {u,i}, h_ {u,i})$
로 정의하고 accurate estimate of $\tilde{p}_ {u,i}$를 $\hat{p}_ {u,i}$ 라고 하면, IPS와 DR estimator가 unbiased.

위 Theorem은 결국 기존의 model이 정의한 propensity score의 경우 measured confounders $x$에 의해 생기는 confounding bias만 통제할 수 있기에, 모든 confounding bias를 제거하기 위해서는 unmeasured confounders $h$도 propensity score에 고려해줘야 한다는 점을 시사하고 있다.

### **3.2 Robust Deconfounder Framework** ###

unmeasured confounders의 경우 우리가 접근할 수 없기에, strong assumption을 도입하여 $\tilde{p}_ {u,i}$을 estimate하여야 한다. 이를 위해서 propensity score model에 sensitivity analysis를 적용한 것에서 영감을 받아서, treatment에 가해지는 unmeasured confounding의 strength를 제한함으로써 $\tilde{p}_ {u,i}$의 uncertainty set은 얻어볼 수 있다. 여기서 nominal propensity score(measured confounder만을 고려한 score)를 arbitrary function $m$을 통해 정의하면 다음과 같다.

> $p_ {u,i}=P(o_ {u,i}=1 \vert x_ {u,i})=\frac{exp(m(x_ {u,i})) }{1+ exp(m(x_ {u,i})) }$

이때 주어진 bound $\Gamma \geq 1$에 대하여 additive model을 기반으로 arbitrary function $\phi$를 통해 true propensity score를 정의해볼 수 있다.

> $ \tilde{p}_ {u,i}=P(o_ {u,i}=1 \vert x_ {u,i}, h_ {u,i} )=\frac{exp(m(x_ {u,i}) + \phi(h_ {u,i}) ) }{1+ exp(m(x_ {u,i}) + \phi(h_ {u,i}) )  }$

아까 언급했듯이 unmeasured confounders의 strength에 대한 제약을 $ \vert \phi(h) \vert \leq \log{\Gamma}$로 잡아준다면 아래와 같은 부등식을 얻게 된다.

> $ \frac{1}{\Gamma} \leq \frac{(1-p_ {u,i})\tilde{p}_ {u,i} }{p_ {u,i}(1-\tilde{p}_ {u,i}) } \leq \Gamma $

여기서 $\hat{w}_ {u,i} = \frac{1}{\tilde{p}_ {u,i}}$로 잡으면 $a_ {u,i} = 1 + (1/p_ {u,i} -1 )/\Gamma, 
b_ {u,i} = 1 + (1/p_ {u,i} -1 )\Gamma$에 대하여 

> $a_ {u,i} \leq \tilde{w}_ {u,i} \leq b_ {u,i}$

가 성립한다. 이때 $\Gamma = 1$ 인 경우 $p_ {u,i} = \tilde{p}_ {u,i}$가 되어 unmeasured confounders가 없는 상황으로 볼 수 있고, 이 값을 키울수록 unmeasured strength를 크게한다고 볼 수 있다.

여기서 Uncertainty set $\mathbb{W} = W \in \mathbb{R}_ +^{\vert D \vert} : \hat{a}_ {u,i} \leq w_ {u,i} \leq \hat{b}_ {u,i}$ (여기서 $W = w_ {u,i} : (u,i) \in D$이고 vector 형태로 생각하면 된다. 그리고 $\hat{a}_ {u,i}$과 $\hat{b}_ {u,i}$는 $a_ {u,i}$와 $b_ {u,i}$의 estimator이다. ) 이 uncertainty set $\mathbb{W}$이 이들이 제안한 framework의 핵심이라고 볼 수 있는데, 바로 이 $\mathbb{W}$ 내에서 inverse of estimated nominal propensity를 변화시키면서 adversarial learning을 진행할 수 있기 때문이다. 기존의 Model인 IPS와 RD의 estimator가 다음과 같이 표현된다.

> $L_ {RD-IPS}(\phi) = \underset{W \in \mathbb{W}}{\max}{\frac{1}{\vert D \vert}\sum_ {(u,i) \in D}o_ {u,i}e_ {u,i}w_ {u,i}}$

> $L_ {RD-DR}(\phi, \theta) = \underset{W \in \mathbb{W}}{\max}{\frac{1}{D}\sum_ {(u,i)\in D}[ \hat{e}_ {u,i} + o_ {u,i}(e_ {u,i}-\hat{e}_ {u,i})w_ {u,i} ]}$

이렇게 adversarial procedure를 적용해줌으로써 좀 더 robust한 estimator를 찾을 수 있다. PseudoCode의 형태로 나타내면 다음과 같다.

![PseudoCode](https://i.ibb.co/hcvc5BT/Pseudo-Code.png)

### **3.3 Benchmarked RD Framework** ###

이들이 제안한 RD framework로는 기존의 propensity-based method의 robustness를 향상할 수 있지만 이론적 근거가 부족한 모습을 보인다. 따라서 이를 위해 benchmark estimator를 도입하여 예측 정확도 향상을 보장하는 Benchmarked RD framework (BRD)를 제안한다.

$\phi$라는 parameter를 바탕으로 prediction error를 $e_ {u,i}(\phi)$라 하자. 이때 $\phi$의 estimator를 얻어낼 수 있다고 하고 이를 $\hat{\phi}^{(0)}$라고 하자. 그러면

> $L_ {BRD-IPS}(\phi) = \underset{W \in \mathbb{W}}{\max}{\frac{1}{\vert D \vert}\sum_ {(u,i) \in D}o_ {u,i} { e_ {u,i}(\phi) – e_ {u,i}(\hat{\phi}^(0)) } w_ {u,i}} $

가 된다. 기존의 RD-IPS estimator에서 $\hat{e}_ {u,i}$ 부분이 $e_ {u,i}(\hat{\phi}^{(0)} )$으로 바뀐 것을 볼 수 있다. DR의 경우에도 똑같이 적용해볼 수 있고, 이러한 framework 우수성을 아래의 Theorem 3.2에서 확인해볼 수 있다.
여기서 $\phi^\dagger = \text{arg}\underset{\phi}\min{L_ {BRD-IPS}(\phi)}, \; \phi^\ddagger = \text{arg}\underset{\phi}\min{L_ {BRD-DR}(\phi)}$를 정의하면,

> __Theorem 3.2__ (“No-Harm” Property). $\phi^{(0)}$를 $\phi$의 benchmark estimator라 하고 $\vert D \vert $가 충분히 크다고 하자. 그러면
> (a) Theorem 4.1의 조건 하에서, 만약 $L_ {BRD-IPS}(\phi^{\dagger}) < 0$ 이라면, 적어도 $1-\eta$의 확률로 $L_ {ideal}(\phi^{\dagger}) < L_ {ideal}(\hat{\phi}^{(0)} )$ 이다.
> (b) Theorem 4.2의 조건 하에서, 만약 $L_ {BRD-DR}(\phi^{\ddagger}) < 0$ 이라면, 적어도 $1-\eta$의 확률로 $L_ {ideal}(\phi^{\ddagger}) < L_{ideal}(\hat{\phi}^{(0)} )$ 이다.

이 Theorem은 BRD framework로 구하는 parameter의 경우 benchmark estimator( IPS나 DR로 pre-train된 estimator라고 할 수 있다 )에 적지 않은 확률로 좋다 ( 낮은 Loss를 갖고 있다 ) 는 것을 의미한다.

### **3.4 RD and BRD AutoDebias** ###

AutoDebias는 역시 propensity-based method 로써 uniform data로부터 propensity score를 학습하고 training phase에서 unobserved data의 rating을 채워줌으로써 예측 성능을 향상한다. 이 경우에도 RD와 BRD framework를 도입해볼 수 있고, 자세한 수식은 논문을 직접 참고해보기를 바란다.

---

## **4. Theoretical Analysis**

보통 모델의 우수성을 보이기 위해서는 제안된 method의 generalization bounds를 보여준다. 이러한 bound는 prediction model class의 complexity에 의존하는데, 여기서는 Rademacher complexity를 이용하여 설명한다. $F$를 함수의 class of functions라고 하고 $f_ {\phi} \in F$라고 한다면, Rademacher complexity는 아래와 같다.

> $R(F) = \mathbb{E}_ {\sigma \sim {-1,+1}^{\vert D \vert} } \underset{f_ {\phi} \in F}{\sup}{\left[ \frac{1}{\vert D \vert}\sum_ {(u,i) \in D}\sigma_ {u,i}e_{u,i} \right] }$

이때 $\sigma = \sigma_ {u,i} : (u,i) \in D $는 Rademacher sequence이다. 여기서 $\vert D \vert \rightarrow \infty$ 임에 따라 $R(F) \rightarrow 0$라고 하자. ( 이는 matrix factorization과 같은 모델도 성립하는 매우 약한 가정으로, vanishing complexities로도 불린다 ) 그러면 앞서 언급했던 Theorem 4.1을 유도해볼 수 있다.

> __Theorem 4.1__ (Generalization bound of RD-IPS and BRD-IPS )
> 모든 $(u,i)$ 쌍에 대하여 $\tilde{w}_ {u,i} \in [ \hat{a}_ {u,i}, \hat{b}_ {u,i} ], e_{u,i} \leq C_1, \tilde{w}_ {u,i} \leq C_ 2$ 를 만족한다고 가정하자. 그러면 임의의 $f_ {\phi} \in F$ 와 $\eta > 0$ 에 대하여 적어도 $1-\eta$의 확률로 아래의 부등식이 성립한다.

> $L_ {ideal}(\phi) \leq L_{RD-IPS(\phi)} + B(\eta, D, F)$

> $L_ {ideal}(\phi) – L_ {ideal}(\hat{\phi}^{(0)} ) \leq L_ {BRD-IPS(\phi)} + B(\eta, D, F)$

여기서 $\phi^{(0)}$는 $\phi$의 pre-trained IPS estimator이고, $B(\eta, D, F) = 2(C_2 +1)R(F) + C_1(C_2+1)\sqrt{\frac{18\log{2/\eta} }{\vert D \vert} }$이다. 위의 부등식에서 결국 vanishing complexities에 의해 RD-IPDS의 estimator가 ideal estimator의 asymptotically upper bound가 된다. 또한 이를 바탕으로 아까 Theorem 3.2에서 언급했던 내용이 자연스럽게 따라오게 된다. Theorem 4.2는 Theorem 4.1과 매우 유사하니, 직접 논문을 참고해보기를 바란다.

---

## **5. Experiment**

모델의 우수성을 보이기 위해 이들은 아래 3가지의 질문에 답하고자 한다.

> - RQ1 : 제안된 RD와 BRD가 기존의 propensity-based method의 성능 향상을 일으키는가?
> - RQ2 : unmeasured confounder의 strength 정도가 다른 상황에서도 성능이 안정적인가?
> - RQ3 : 어떤 인자가 우리의 효과성에 영향을 주는가?

### **5.1 Experimental Settings**

Dataset의 경우 1) Yahoo!R3, 2) Coat, 3) Product를 이용하였고 이는 각각 music, coat, micro-video 추천을 담고 있다. 이것들 모두 set of biased data와 Randomized controlled trial로부터 얻어진 set of unbiased data를 포함하고 있다. 기본적으로 biased data를 training으로 쓰고 unbiased data의 일부를 추출하여 training으로 활용한다. 나머지 unbiased data를 validation set과 test set으로 활용한다. Dataset의 statistic의 경우 다음과 같다.

Method의 경우 Base model과 기존의 propensity-based model을 활용한다. Base model의 경우 단순히 Matrix Factorization(MF)을 이용한 것이고, DCF의 경우 MF를 unobserved confounder에 좀 더 robust하게 만들고자 하는 방법론이다. 이들 각각에 대한 자세한 내용은 링크된 논문을 읽어 보기를 바란다.

Evaluation metric의 경우 UAUC와 NDCG@K를 활용하는데, unbiased testing data에 대하여 각 user마다 AUC와 NDCG@K를 구하고 이를 평균 낸 score를 이용한다.

### **5.2 Performance Comparison**

먼저 Main table의 경우 다음과 같다.

![Table2](https://i.ibb.co/XLZ4hgp/Table-2.png)

몇 가지 중요한 observation을 보자면,
- RD와 BRD 모두 기존 propensity-based 모델의 성능을 향상시켰다.
- 특히 BRD의 경우 앞선 Theorem 3.2에서 증명된 “no-harm” property에 의해 기본 모델보다 더 좋은 성능을 보장받을 수 있었기에 좋은 성능으로 이어졌다.
- BRD의 경우 대부분의 case에서 RD보다 좋은 모습을 보였는데, 이는 BRD 모델이 pre-trained propensity-based model을 benchmark로 활용했기 때문으로 본다.
- DCF의 경우 대부분 RD와 BRD보다 좋지 않은 모습을 보인다. 이는 DCF가 부정확한 system exposure의 확률과 모든 confounder가 연관되어 있다고 보기에 이것이 성능을 하락하였다고 분석한다.

### **5.3 In-depth Analysis**

### __Study on Confounding Strength__

앞서 언급하였던 Confounding Strength에 대한 실험을 진행하였는데, 즉 RD와 BRD가 unmeasured confounder의 영향력이 점점 커지는 상황에서 어떻게 성능이 변화하는지를 관찰하고자 한 것이다. 이 영향력은 training data에서 positive feedback의 일부를 선택적으로 masking하여 진행해볼 수 있다. 왜냐하면, 이러한 masking은 unmeasured confounder가 treatment (item exposure)와 outcome (positive feedback 에 영향을 준다는 사실에 기대어 간접적으로 그 영향력을 조정하는 행위로 볼 수 있기 때문이다. 또한 비교군을 형성하기 위해 label에 상관없이 랜덤하게 masking한 것도 dataset으로 만들어 두었다. 

아래의 그림을 보면 (분량 상 IPS만 삽입하였다) 여러 observation을 얻을 수 있다. 

![Figure3](https://i.ibb.co/zhCxdzH/Figure-3.png)

먼저 mask ratio가 커질수록 모델의 성능이 하락하는 것을 볼 수 있는데 이는 당연히 data가 noise해지는 것이므로 당연한 결과라고 볼 수 있다. 그리고는 Random하게 masking한 것보다 선택적으로 masking한 것이 더 좋은 성능을 내는데, 이는 기존의 propensity-based model이 결국 unmeasured confounders의 영향력을 무시하였기 때문에 성능이 좋지 못했음을 간접적으로 알 수 있는 결과이다. 또한 adversarial learning을 도입하여 unmeasured confounders의 영향력이 큰 상황에서도 좋은 성능을 낼 수 있었다.

### Effects on Propensities

그런 다음으로는 RD의 효과를 좀 더 심층적으로 분석하고자 한 실험인데, 바로 inverse of nominal propensity와 RD에 의해 얻어진 final propensity의 absolute gap을 계산하는 것이다. 아래의 그림을 살펴보자.

![Figure4](https://i.ibb.co/NFhLk9Y/Figure-4.png)

이 그림에서 average gap과 item의 frequency가 positive correlation을 가짐을 볼 수 있는데 이는 Theorem 3.1에서도 증명했듯이 기존의 모델은 nominal propensity를 이용하기에 bias된 상황으로 볼 수 있고, frequent item이 bias에 더 치명적인 이유로는 data의 개수가 많기에 bias된 상황에서 더 fit하게 되었기 때문이다. 즉 이러한 이유로 인해 RD와 BRD의 framework을 이용하는 것에 대한 타당성을 인정받을 수 있다.

### Ablation Study

Ablation study로 adversarial learning의 우수성을 보이기 위해 첫 번째로는 uncertainty set $\mathbb{W}$에서 inverse propensities를 random하게 뽑는 것과 nominal inverse propensity score $w_{u,i}$에 Gaussian white noise를 주는 2가지의 경우와 제안된 pure 모델의 성능을 비교하는 실험을 진행하였다. 아래의 그림에서 UAUC와 NDCG metric 전부에 대하여 본 모델이 우월한 성능을 내었음을 보였다. 

![Figure5](https://i.ibb.co/JCVzkvW/Figure-5.png)

---

## **6. Conclusion & Limitation**

위 연구에서 이들은 현존하는 propensity-based methods를 이론적으로 분석하여 unmeasured  confounders가 존재하는 상황에서 모델이 추정하고자 하는 parameter가 bias 되어 있음을 밝혀내었다. 또한 이러한 사실을 바탕으로 true propensity를 재정의하고 이것의 uncertainty set을 추정하여 sensitivity analysis를 적용한 Robust Deconfounder를 제안하여 기존의 propensity-based model의 성능을 끌어올렸다. 더 나아가 pre-trained propensity-based model을 benchmark로 삼아 robustness와 accuracy 사이에서 생기는 trade-off 이슈를 완화하였고, 이론적으로도 기존의 method보다 좋은 성능을 낼 수밖에 없음을 보장하여 real-world dataset에 대해서도 좋은 성능을 이끌어 내었다.

하지만 필자가 보기에는 약간 아쉬운 점이 있다. 그것은 바로 propensity-based method가 아닌 모델과는 성능 비교를 하지 않았다는 점이다. 물론 기존의 propensity-based method의 성능을 끌어올린 것은 좋은 contribution이지만, 이를 이용하지 않은 다른 method의 성능이 잘 나올 수 있기에 이러한 비교가 필요하다고 본다.
그러나 Rademacher complexity라는 개념을 이용하여 이론적으로 모델의 우수성을 밝혀낸 점이 좋았으며, 꼭 추천시스템이 아니더라도 이러한 개념을 활용하여 모델의 성능을 보장할 수 있는 합리적인 결과를 추후 연구에서도 적용해볼 수 있다는 점에서 좋은 논문으로 생각하게 되었다.

---

## **Author**

__이준모 ( Junmo Lee )__

- Affiliation : KAIST ISysE
- Research Topic : Causal Inference, NLP, Anomaly Detection

---

## **Reference & Implementation**

- [Paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539240)

- [Implementation](https://github.com/Dingseewhole/Robust_Deconfounder_master/)