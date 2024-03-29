---
title:  "[SIGIR 2020] Neural Interactive Collaborative Filtering"
permalink: 2023-11-20-Neural_Interactive_Collaborative_Filtering.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **Review of "Neural Interactive Collaborative Filtering (Zou et al., 2020)"** 

### Zou, L., Xia, L., Gu, Y., Zhao, X., Liu, W., Huang, J. X., & Yin, D. (2020, July). Neural interactive collaborative filtering. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 749-758).

## **1. Problem Definition**  

본 논문에서는 Interactive setting에서의 collaborative filtering 상황에서 문제점을 찾고, 해결하기 위한 방안을 제시합니다. Interactive setting은 recommender agent가 추천을 하고, 그 추천을 바탕으로 feedback이 오는 연속적인 상황을 말합니다. 이러한 종류의 Interactive recommendation은 실생활에서 많이 찾아볼 수 있는데, Spotity, Amazon, Pinterests 등에서 사용하고 있습니다.

Interactive recommendation에서 어려운 점은 사용자에 대한 정보가 충분하지 않을 때 발생합니다. 예를 들어 크게 두 가지를 생각할 수 있는데, 1) cold-start user, 애초에 정보가 충분하지 않은 사용자와 2) warm-start user with taste drifting, 기존의 데이터가 있지만, 취향이 변했기 때문에 바뀐 취향에 대한 정보가 없는 사용자 두 가지로 생각을 할 수 있습니다. 그렇기 때문에 이 상황에서 중요한 문제는 사용자의 추천 경험을 손상시키지 않으면서 사용자의 관심사를 신속하게 알아낼 수 있어야 한다는 것입니다. 즉 사용자 프로필 학습함과 동시에 정확한 추천을 수행하는 두 목표 사이의 균형을 유지하는 방법에 대해 알아가야 하는 중요한 문제입니다.


## **2. Motivation**  

위에서 제시한 문제를 해결하기 위해 기존에 접근하던 방식은 크게 두 가지로 나눌 수 있는데, 1) MAB (Multi Armed Bandits) 방식과 2) Meta-Learning 방법입니다.

MAB 접근법은 문제를 MAB 또는 contextual bandits으로 formulate하고 GLM-UCB 및 톰슨 샘플링과 같은 복잡한 탐색 전략으로 해결합니다. 그러나 provable 한 lower bound를 달성하기 위해 최악의 상황에서의 추천을 최적화하게 되고, 그 결과로 비관적인 추천을 하게 되어 전반적인 최적의 성능을 달성하지 못할 수 있습니다. 또한 일반적으로 비선형 모델에 대해서는 computationally intractable하다는 단점도 있습니다.

Meta-learning 접근법은 최근 cold-start recommendation을 해결하기 위해 새로 들어오는 피드백에 대해 모델을 빠르게 적응시킬 수 있도록 하기 위해 활용되고 있습니다. Meta-learning을 활용한 기존의 방법은 다양한 사용자를 위한 항목을 다른 작업으로 취급하고, 적은 수의 recommendation (support set)을 관찰 후 빠르게 사용자의 선호를 알아내는 방법으로 접근했습니다. 그러나 이러한 접근법은 support set을 구성하는 단계에서 아예 관련없는 item을 추천할 수 있고, 또 사용자 경험 측면에서도 좋지 않습니다. 또한 휴리스틱하게 선택된 support set에 대해 과도한 의존성을 가지고 있으면서, 사용자의 취향이 바뀌거나 support set의 quality가 안좋을 경우 사용자의 관심사를 적극적으로 exploration하는 데 부족하다는 문제가 있습니다.

본 논문에서는 위와 같은 문제를 해결하기 위해 Interactive Collaborative Filtering을 Meta-Learning 문제로 간주하고 다양한 사용자에 대한 exploration과 exploitation의 균형을 목적으로 추천을 adaptive하게 선택할 수 있는 neural exploration policy 학습을 진행하는 NICF(neural interactive collaborative filtering)라는 Framework를 제안했습니다. 

<!-- NICF는 효율적인 강화 학습(Reinforcement Learning, RL) 알고리즘으로 추천 과정 전반에 걸쳐 전체 사용자의 만족도를 극대화하여 exploration policy의 가중치를 직접 최적화합니다. 이는 두 가지 측면에서 의미가 있습니다. 1) exploration/exploitation을 통해 interactive recommendation 과정에서 사용자들의 전반적인 참여를 maximize 시킬 수 있습니다. 2) 강화 학습의 관점에서는 exploration에 의해 나온 만족스러운 추천을 사용자 프로필의 품질 향상에 기여한 탐색 보너스(지연 보상)로 볼 수 있기 때문에 RL 적용에 의미가 있습니다. 따라서 즉각적인 보상과 지연된 보상의 합을 최적화하는 것은 exploration, exploitation의 균형에 의한 reward를 maximize하는 것으로 볼 수 있으며, 이는 RL에 의해 효과적으로 해결할 수 있습니다. 후에 더 설명이 나오지만, 이렇게 함으로써 학습된 exploration policy는 interactive 한 추천에 대한 학습 프로세스 역할을 하며 cold-start 또는 warm-start 추천에 대한 전략을 계속 조정할 수 있습니다. -->

NICF는 다음과 같은 장점들이 있습니다. 1) Interactive collaborative filtering을 위한 기존의 exploration policy의 문제점인 비관적인 선택을 방지하고, 지나치게 복잡해지는 것을 방지합니다. 2)  훨씬 더 많은 비선형 user-item interaction을 포착할 수 있는 어떤 추천 deep model에도 적용할 수 있습니다. 3) exploration/exploitation 균형을 맞추어 기존 meta-learning 의 완전한 
exploitation으로 인해 발생하는 사용자의 좋지 않은 경험을 완화합니다. 마지막으로 세 가지 벤치마크 데이터 세트에 대한 광범위한 실험과 분석을 통해 좋은 성능을 보인다는 점을 실험적으로 보였습니다.


## **3. Background**  

일반적인 추천시스템에서 우리는 

$N$ users set, $U=\{1,...,N\}$ 과 $M$ items set, $I=\{1,...,M\}$

을 가지고 있습니다. user의 피드백은 $N \times M$ preference matrix $R$로 나타납니다. 여기서 $r_ {u,i}$는 item $i$에 대한 user $u$의 선호도를 나타냅니다.

일반성을 잃지 않고, discrete 한 timestep에 대해서 생각해보자면, 각 timestep $t \in [0,1,2,...,T]$에 대해 system이 target user $u$에게 item $i_ {t}$를 추천하면, user은 그에 대한 피드백 $r_ {u,i_ {t}}$을 시스템에 주게 됩니다. 즉, $r_ {u,i_ {t}}$는 target user에 대한 reward라고 생각할 수 있습니다. 피드백을 받은 이후, system은 reward를 기반으로 model을 update하고 다음 item으로 어떤 것을 추천할 지 결정하게 됩니다. 여기서, 

$s_ {t}=\{i_ {1}, r_ {u,i_ {1}},...,i_ {t-1}, r_ {u,i_ {t-1}}\}$

를 system이 time $t$에서 확인할 수 있는 information (support set)이라고 하겠습니다. 그럼, item은 policy $\pi : s_ {t}\to I$에 의해 선택되게 됩니다. 여기서 policy $\pi$는 현재 support set에서 selected item $i_ {t}\sim\pi(s_ {t})$으로 mapping 되는 function입니다.

Interactive recommendation process에서 $\pi$의 전체 T-trial의 payoff는 $\sum_ {i=1}^{T} r_ {u,i_ {t}}$ 로 정의됩니다. 임의의 user $u$에 대해, 우리의 목표는 expected total payoff $G_ {\pi}(T)$를 maximize하는 policy를 만드는 것입니다. 여기서 

$G_ {\pi}(T)=\mathbb{E}_ {i_ {t}\sim\pi(s_ {t})} [\sum_ {i=1}^{T} r_ {u,i_ {t}}]$ 

로 정의됩니다. 비슷하게 우리는 optimal expected T-trial payoff를 

$G^{\*}_ {\pi}(T) = \mathbb{E}[\sum_ {i=1}^{T} r_ {u,i^{\*}_ {t}}]$ 

과 같이 정의할 수 있습니다. 여기서 $i^{*}_ {t}$는 timestep $t$에서 expected reward를 최대화시키는 최적의 item입니다.

### **3.1 Multi-Armed Bandit Based Approaches**

![MAB](https://i.ibb.co/YWg6Qds/1.png)

위 그림은 MAB Based approach 방법을 나타냅니다. 현재, interactive follaborative filtering에서의 exploration은 probabilistic matrix factorization (PMF)을 기반으로 합니다. PMF는 rating의 conditional probability가 gaussian, 즉 

$Pr(r_ {u,i} \vert \mathbf{p}_ {u}^{\top}\mathbf{q}_ {i}, \sigma ^{2}) = {N}(r_ {u,i} \vert \mathbf{p}_ {u}^{\top}\mathbf{q}_ {i}, \sigma ^{2})$ 

을 가정합니다. 여기서 $\mathbf{p}_ {u}$와 $\mathbf{q}_ {i}$는 각각 zero mean Gaussian prior인 user, item feature vector이고, $\sigma$는 prior variance입니다. 

기존의 MAB 접근에서 learning procedure은 위의 그림과 같이 

1) $t-1$ 번의 interaction이 끝나고 $Pr(\mathbf{p}_ {u})={N}(\mathbf{p}_ {u,t}\vert \mathbf{\mu}_ {u,t}, \Sigma_ {u,t})$과, $Pr(\mathbf{q}_ {i})={N}(\mathbf{q}_ {i,t}\vert \mathbf{\nu}_ {i,t}, \Psi_ {u,t})$
로 정의되는 posterior distribution 을 얻습니다. 각각의 mean, variance term은 MCMC-Gibbs 등으로 얻을 수 있습니다. 

2) cumulative reward를 최대화시키는 $t$ 번째 추천을 휴리스틱하게 선택합니다. 

선택하는 방법으로는 대표적으로 2가지가 있는데, 각각 Thomson Sampling, Upper Confidence Bound 방법입니다. 

Thomson Sampling은 $i_ {t} = \underset{i}{\text{arg max}} \,\tilde{\mathbf{p}}^{\top}_ {u,t}\tilde{\mathbf{q}}_ {i,t}$인 item을 선택하는 방법입니다. 여기서 $\tilde{\mathbf{p}}_ {u,t}\sim {N}(\mathbf{\mu}_ {u,t}, \Sigma_ {u,t})$과, $\tilde{\mathbf{q}}_ {i,t}\sim {N}(\mathbf{\nu}_ {i,t}, \Psi_ {u,t})$은 user, item feature vectors의 posterior distribution에서 sampling 됩니다. 

Upper Confidence Bound은 사용자가 신뢰할 수 있을 정도로 좋아하는 item을 선택하는 것으로,  upper confindence bound와 PMF를 합친 Generalized Linear Model Bandit-Upper Confidence Bound(GLM-UCB) 방법이 주로 사용됩니다. 자세한 식은 다음과 같습니다.

$i_ {t} = \underset{i}{\text{arg max}} \,(\rho(\mathbf{\mu}^{\top}_ {u,t}\mathbf{\nu}_ {i,t}) + c\sqrt{\log t}\vert\vert \mathbf{\nu}_ {i,t} \vert\vert_ {2, \Sigma_ {u,t}})$ 

여기서 $\rho$는 sigmoid function이고, c는 상수입니다. $\vert\vert \mathbf{\nu}_ {i,t} \vert\vert_ {2, \Sigma_ {u,t}}$는 $\sqrt{\mathbf{\nu}_ {i,t}^{\top} \Sigma_ {u,t} \mathbf{\nu}_ {i,t}}$로 정의됩니다. 이는 $r_ {u,i}$의 $t$ 번째 interaction의 불확실성을 의미합니다.

위와 같은 접근방식들로 인해 MAB 기반 방법은 몇 가지 한계를 가지게 됩니다. 1) 비선형 모델에 대한 posterior distribution 업데이트의 어려움으로 인해 linear user-item interaction model에만 적용되며, 이는 neural network 기반 모델에서의 사용을 크게 제한합니다. 2) 많은 중요한 hyperparameter(ex. prior distribution의 variance 및 exploration hyperparameter c)가 사용되기 때문에 최적의 추천을 찾는 데 어려움이 생깁니다. 3) Thomson Sampling과 GLM-UCB은 보통 최악의 추천 상황을 최적화하기 때문에 잠재적으로 지나치게 비관적입니다.

### **3.2 Meta-Learning Based Approachs**

![MAML](https://i.ibb.co/VmqFNvC/2.png)

위 그림은 Meta-Learning Based Approach 방법을 나타냅니다. 이 방법은 적은 support set을 관측한 후 user들의 관심사를 빠르게 알아내는 방법을 배우는 것을 목적으로 합니다. 위의 그림은 대표적인 방법 중 하나인 MELU입니다. MELU는 Model-Agnostic Meta-Learning (MAML)을 기반으로 해 cold start user들에 대한 빠른 model 학습을 가능하도록 합니다.

위 방법은 우선 recommender agent가 parameter로 $\theta$를 가지는 neural network로 modeling되어있다 가정합니다. MELU는 작은 support set $D$로 user의 관심사를 반영한 $\theta_ {1}$을 얻을 수 있는 initialization $\theta_ {0}$을 알아내는 것을 목적으로 합니다. $\theta_ {0}$는 $\theta_ {1}$이 update 된 후, support set $D$에 대한 특정한 loss $l$을 minimizing하는 방향으로 학습되어 갑니다. 구체적으로는 

$\theta_ {1}=\theta_ {0}-\alpha l(\pi_ {\theta_ {0}},D)$

로 update 되고, 

$\theta_ {0}\leftarrow\theta_ {0}-\alpha l(\pi_ {\theta_ {1}},D)$

로 loss를 minimize하는 방향으로 학습됩니다. 여기서 $\pi_ {\theta}$는 $\theta$로 parametrize 되어있는recommendation policy 입니다. $l$은 accuracy measure인데, 보통 MSE나 Cross entropy를 사용합니다.

Meta-Learning approachs는 exploration policy을 수작업으로 만들지 않아도 되고, deep neural network의 이점을 활용할 수 있다는 장점이 있습니다. 하지만 기존 Meta-Learning approachs 에서는 사용자에게 안좋은 경험을 발생시키지 않고 support set을 선택하는 방법에 대한 부분은 고려되지 않았습니다. 그래서 몇 가지 단점이 있는데, 위에서도 언급했듯 1) support set을 구성하는 단계에서 관련성이 매우 낮은 아이템의 추천으로 인한 사용자 경험 품질 저하, 2) 이 방법들은 완전히 exploitation의 전략이기 때문에 사용자의 관심사를 exploration 하는 데 부족하고, 이로 인해 사용자의 취향이 바뀌거나 품질이 떨어지는 support set이 주어지게 되면 성능이 좋지 않다는 단점이 있습니다.


## **4. Method**  

![NICF](https://i.ibb.co/XYh4BFX/3.png)

위 그림은 전체 procedure을 나타낸 그림입니다. 본 방법은 각 user에 대한 exploration policy를 직접 설계하는 것이 아닌, recommender agent가 다양한 사용자 각각에 대해 사용자의 관심사를 빠르게 포착하여 시스템에 대한 누적 사용자의 reward를 maximize할 수 있는 neural network 기반 exploration policy를 학습하는 것을 목표로 합니다. 즉, 모든 사용자 history의 item의 집합을 입력으로 받아 새로운 테스트 항목에 적용할 수 있는 scoring function을 생성하고 사용자 프로파일 학습과 정확한 추천 사이의 목표 균형을 맞추는 sequential neural network를 학습하고자 합니다.

### 4.1. Self-Attentive Neural Policy

Exploration policy는 두 부분으로 구성되어 있습니다. 1) 사용자 피드백의 정보를 별도로 캡처하기 위해 과거의 추천과 사용자의 피드백을 multi-channel stacked self-attention blocks에 공급하여 사용자 프로파일을 embedding 시킵니다. 2) policy layer은 multi-layer perceptron으로 추천을 생성합니다. 따라서 sequential neural network 구조로 과거의 추천을 기반으로 사용자 프로파일을 업데이트할 수 있습니다. 아래 그림은 exploration policy의 neural architecture 입니다. 

![Exploration Network](https://i.ibb.co/j8gwmP5/4.png)

각 부분에 대한 설명은 다음과 같습니다.

**Embedding layer**

$s_ {t}=\{i_ {1}, r_ {u,i_ {1}} , . . . , i_ {t−1}, r_ {u,i_ {t−1}}\}$ 이 주어진 상황에서 전체 $\{i_ {t}\}$ 를 item embedding vectors $\{\mathbf{i}_ {t}\}$ 로 변환시킵니다. 

**Self-Attntion Layer**

observation $s_ {t}$를 잘 나타내기 위해 다르게 rated된 item은 구분해 multi-channel stacked self-attentive neural network에 넣습니다. score $z$로 rated 된 아이템을 embedding matrix 

$E_ {t}^{z}=[...,\mathbf{i}_ {m},...]^{\top}(\forall r_ {u,i_ {m}}=z, m<t)$ 

로 나타냅니다. self-attention operation은 embedding $E_ {t}$를 input으로 받고, linear projection을 통해 3개의 matrix로 변환시킨 후 attention layer에 넣습니다. 

$S_ {t}^{z}=SA(E_ {t}^{z})=Attention(E_ {t}^{z}W^{z,c},E_ {t}^{z}W^{z,k},E_ {t}^{z}W^{z,v})$

Recommendation은 sequential하게 들어오기 때문에, attention layer은 $t$-th policy를 만들 때 첫 $t-1$개의 item만 고려해야 합니다. 

**Point-Wise Feed-Forward Layer**

모델에 비선형성을 부여하고, 서로 다른 latent dimension간의 상호작용을 고려하기 위해 $S_ {t\,\,m}^z$에 (self-attention layer $S_ {t}^z$의 $m$-th row) point-wise two-layer feed-forward network를 적용했습니다.

$F_ {t\,\,m}^{z}=\text{FFN}(S_ {t\,\,m}^{z})=\text{ReLU}(S_ {t\,\,m}^{z}W^{(1)}+b^{(1)})W^{(2)}+b^{(2)}$

**Stacking Self-Attention Block**

Self-attention layer, point-wise feed-forward layer은 더 복잡한 item transition을 학습하기 위해 더 쌓을 수 있습니다. 구체적으로 $b$-th ($b>1$) block은 다음과 같이 정의됩니다.

$S_ {t}^{z,b}=SA(F_ {t}^{z,b-1}),\; F_ {t\quad m}^{z,b}=\text{FFN}(S_ {t\quad m}^{z,b})$

**Policy Layer**

$b$ self-attention block이 쌓여서 계층적으로 정보를 추출할 수 있도록 만들어지면 next item의 score을 ${ F_ {t}^{z,b}}_ {z=1}^{R_ {max}}$를 기반으로 예측합니다. 여기서 $R_ {max}$는 maximum reward 입니다. 추천된 item들의 예측된 cumulative reward을 

$\mathbf{Q}_ {\theta}(s_ {t}, \cdot)=[Q_ {\theta}(s_ {t},i_ {1}),..., Q_ {\theta}(s_ {t},i_ {N})]^{\top}$

이라고 하겠습니다. Policy layer은 다음과 같은 두개의 feed-forwad layer들에 의해 계산됩니다 : 

$u_ {t}=\text{concat}\left[  F_ {t}^{1,b^{\top}},F_ {t}^{2,b^{\top}},...,F_ {t}^{R_ {max},b^{\top}} \right]^{\top}$

$\mathbf{Q}_ {\theta}(s_ {t}, \cdot)=\text{ReLU}(u_ {t}W^{(1)}+b^{(1)})W^{(2)}+b^{(2)}$

추정된 $\mathbf{Q}_ {\theta}(s_ {t}, \cdot)$으로 maximal Q-value 를 가지는 item으로 recommendation이 다음과 같이 생성됩니다. 

$\pi_ {\theta}(s_ {t})=\underset{i}{\text{arg max}} \,\mathbf{Q}_ {\theta}(s_ {t}, i)$ 

### 4.2. Policy Learning

 exploration policy parameter $\theta$를 update하기 위해 본 논문에서는 Q-Learning을 사용했습니다. $t$-th 시도에서 recommender agent는 support set $s_ {t}$를 관찰하고, $\mathbf{Q}_ {\theta}(s_ {t}, \cdot)$를 사용해 $\epsilon$-greedy policy로 item $i_ {t}$를 선택합니다. 이는 $1-\epsilon$의 확률로 max Q-value action을 선택하고, $\epsilon$의 확률로 random action을 선택합니다. agent는 user로 부터 response $r_ {u, i_ {t}}$를 받고 observed set을 $s_ {t+1}$로 update 합니다. 최종적으로 experience $(s_ {t}, a_ {t}, r_ {u, i_ {t}}, s_ {t+1})$을 large replay buffer $M$ 에 저장합니다. Value function $\mathbf{Q}_ {\theta}(s_ {t}, i_ {t})$는 다음과 같이 정의되는 loss-function을 최소화하는 방향으로 update됩니다.  
 
 ${l}(\theta)=\mathbb{E}_ {(s_ {t}, i_ {t}, r_ {i, i_ {t}}, s_ {t+1})} \sim {M}[(y_ {t}-Q_ {\theta}(s_ {t},i_ {t}))^{2}]$, $y_ {t}=r_ {u, i_ {t}}+\gamma \,\underset{i_ {t+1} \in {I}}{\text{max}}Q_ {\theta}(s_ {t+1}, i_ {t+1})$. 
 
 여기서 $y_ {t}$는 optimal Bellman Equation을 기반으로 한 target value입니다.

## **5. Experiment**

NICF의 효과를 평가하기 위해 본 논문에서는 세 가지 벤치마크 데이터셋에 대해 광범위한 실험을 진행했습니다. 실험을 통해 주로 다음과 같은 research question에 대해 답하는 데 중점을 두었습니다.
1. NICF가 기존 interactive follaborative filtering algorithms 보다 cold-start user들에 대해 더 잘 작동할 수 있는 방법은 무엇인가?
2. 취향이 바뀌는 warm-start user들에 대해 NICF를 적용할 수 있는가?
3. NICF에서 다양한 구성요소의 역할은 무엇인가?
4. cold-start recommendations를 위해 어떤 종류의 지식을 NICF가 학습하는가?

실험에 사용한 데이터는 MovieLens 1M, EachMovie, 그리고 Netflix 입니다. 추천시스템의 interactive nature 때문에, 실제 사용자와 상호작용하는 온라인 실험은 항상 가능하지는 않습니다. 그래서 interactive collaborative filtering의 setting에 따라, 기록된 rating이 시스템이 제공하는 recommendation에 편향되지 않고 사용자의 본능적인 행동이라고 가정합니다. 아래 표에는 각 데이터에 대한 통계적 정보가 담겨있습니다.

![Summary Statistics of Datasets](https://i.ibb.co/L5gvNdB/5.png)

NICF의 성능을 확인하기 위해 논문에서는 state-of-the-art methods를 포함해 다음과 같은 approach들과 성능을 비교했습니다.
* Random : 최악의 성능을 추정하는 데 사용되는 기준선
* Pop : 등급을 매기는 횟수로 측정한 인기도에 따라 순위를 매김
* MF : 대상 사용자와 유사한 rating을 매긴 다른 사용자의 rating을 기반
* MLP : non-linear collaborative filtering의 일반적인 관행
* BPR : MF 모델을 pairwise ranking loss를 통해 optimize. Item recommendation에서의 SOTA
* ICF : Probabilistic matrix factorization을 사용한 Interactive collaborative filtering
* MeLU : MAML 기반 Meta-Learning 방법. cold-start problem SOTA
* NICF : 본 논문에서 제안하는 방법

Evaluation Metrics는 다음과 같습니다. 추천의 정확성, 다양성 모두에 대해 모델을 평가하기 위해 가장 널리 사용되는 세 가지 metric을 사용해 측정합니다.

* Cumulative Precision@T. : 전체 T timestep interaction 중 positive interaction의 갯수. 본 실험에서는 positive interaction은 $r_ {u, i_ {t}}\geq4$일 때이고, 이때 $b_ {t}=1$. negative interaction 일 땐 $b_ {t}=0$
    * $\text{precision@}T=\frac{1}{ \sharp \, \text{users}}\sum_ {\text{users}}\sum_ {t=1}^{T} b_ {t}$ 

* Cumulative Recall@T. : T timestep 동안의 전체 만족 item set 중 positive interaction을 받은 갯수, recall
    * $\text{recall@}T=\frac{1}{ \sharp \, \text{users}}\sum_ {\text{users}}\sum_ {t=1}^{T} \frac{b_ {t}}{ \sharp \,\text{satisfied items}}$

* Cumulative $\alpha$-$NDCG@T$. : Diversity를 평가하는 지표 
    * $\frac{1}{Z}\sum_ {t=1}^{T}\frac{G@t}{\log(1+t)}$
    * $G@t= \sum_ {\forall i\in C}(1-\alpha)^{c_ {i,t}-1}$ : $t$ timestep까지 추천 목록의 순위에 topic i가 등장한 횟수. topic i는 item 혹은 user의 속성.


### 5.1. Performance comparison on cold-start cases (RQ1)

![Cold-start recommendation](https://i.ibb.co/kDDML7Q/6.png)

위 표는 cold-start case들의 40 trial recommendation 을 통해 볼 수 있는 precision, recall 지표를 보여줍니다. 결과를 해석해보자면 직관적으로 상당히 이해할 수 있는 결과가 나왔습니다.

우선 NICF는 3가지 데이터 셋 전부에서 최고의 precision, recall 지표를 보여줍니다. 이 결과를 통해 cold-start recommendation 상황에서 사용자의 관심사를 빠르게 파악하고 새로운 사용자를 만족시키기 위한 전략을 조정할 수 있다는 것을 의미합니다.

### 5.2. Performance comparison on warm-start cases with taste drift (RQ2)

![Warm-start recommendation](https://i.ibb.co/qNRZwXc/7.png)

위 표는 취향이 바뀌는 Warm-start user들에 적용이 가능한지 알아본 실험의 결과입니다. 

우선 취향이 바뀐다는 것의 기준을 정하고 가야 하는데, 우선 rating을 80개 이상 매긴 user들의 rating 기록들을 두개의 set으로 나누었습니다. set 1(첫 20개 기록)은 historical interaction으로 사용하고 set 2는 취향 변화 simulaton으로 사용했습니다. 그 다음 항목의 장르정보를 사용해 user의 interest를 나타내고, cosine similarity를 구합니다. Similarity가 가장 작은 user들을 선정했습니다. EachMovie 데이터에서는 장르 정보를 알 수 없어서 이번 실험에서는 MovieLens, Netflix 데이터만 사용했습니다. 

이렇게 선정된 데이터들에 대해 Precision, Recall 지표 측면에서 좋은 수치를 보임을 확인할 수 있습니다. 즉, 기존 방법들보다 취향이 바뀌는 warm-start user에 대해 바뀌는 취향을 추적할 수 있음을 보여줍니다. 

### 5.3. Ablation Study (RQ3)

![Ablation analysis](https://i.ibb.co/Gs2tjdg/8.png )

NICF framework에는 많은 구성요소가 있기 때문에 ablation study를 통해 각 구성요소의 영향을 분석합니다. 위의 표는 3개의 dataset에 대한 기본 setting 및 비교에 대한 precision 지표입니다.
1. LSTM : self-attension block을 2-layer, 30 hidden dimension의 LSTM으로 바꿔서 나온 결과입니다. 결과는 stacked self-attention block 사용이 더 도움이 된다는 것을 의미합니다.
2. $\gamma=0$ :  RL을 사용하지 않고 학습하는 것, 즉 delayed reward에 대한 사항을 고려하지 않았을 때의 결과입니다.
3. Number of blocks : block 수가 증가함에 따라 더 복잡한 item transition을 학습하는 데 도움이 된다는 것을 의미합니다. 
4. Multi-head : Transformer 논문에서 Multi-head attension을 사용하는 것이 유용하다는 것을 발견했는데, 본 연구에서는 Single-head보다 성능이 조금 떨어지는 결과가 나와있습니다. transformer에 비해 작은 dimension 때문이 아닐까 추측하고 있습니다.

### 5.4. Analysis on Diversity (RQ4)

직관적으로, 다양한 추천은 user의 관심사, 혹은 아이템 속성에 대한 정보를 더 많이 가져다 준다는 것을 알 수 있습니다. 아래의 표에서 NICF가 추천 다양성을 향상시킬 수 있는지 확인할 수 있습니다.

![The recommendation diversity on cold-start phase.](https://i.ibb.co/b3QN6m4/9.png)

처음 40 timestep 까지에 걸친 accumulative $\alpha$-$NDCG@T$ 결과입니다. RQ2에서와 마찬가지로, 장르 정보가 있는 두 데이터셋에 대해 실험을 진행했습니다. Explore 을 직접 학습한 결과보다 훨씬 다양한 item을 추천해주는 것을 알 수 있습니다. 즉 사용자의 관심사를 exploring하는 것이 추천 다양성을 높일 수 있는 결과를 가지고 왔다고 생각할 수 있습니다. 

![10](https://i.ibb.co/KFLghqh/10.png)

위 그림은 MovieLens dataset 상에서 NICF의 squential decision tree를 visualization한 것입니다. 장르 정보를 사용하지 않고 NICF는 사용자가 이 영화를 좋아하면 몇 가지 다른 주제를 가진 유사한 영화를 추천하거나, 영화의 장르를 변경하여 사용자의 관심사를 탐색합니다. 즉 NICF가 사용자의 관심사를 효과적으로 추적하고, cold-start 상황에서 exploration, exploitation balance를 맞추고 조절할 수 있음을 보여줍니다.

## **6. Conclusion**  

본 논문은 interactive setting 에서 cold-start, warm-start user w/ taste drifting 상황에서의 만족스러운 recommendation 을 수행하는 것을 주 문제상황으로 두었습니다. 여기서의 핵심 insight는 exploration에 의해 나온 recommendation에서의 positive interaction reward를 dalayed reward으로 봤기 때문에 이를 통해 RL로 전반적인 만족도를 높이는 과정이 exploration strategy가 user profile을 만들면서 정확한 recommendation을 할 수 있도록 했다는 점입니다.
또한 효과를 검증하기 위해 세 가지 벤치마크 collaborative filtering datasets에 대해 수행된 광범위한 실험과 분석에서 SOTA를 달성했습니다.

## **7. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation [https://github.com/zoulixin93/NICF]

---
## **Reviewer Information**  

* Name : Minsang Park  
    * Affiliation : AAILAB, KAIST
    * Research Topic : Deep Generative Model