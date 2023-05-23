---
title:  "[ICML 2020] Diffusion Models for Adversarial Purification"
permalink: Diffusion_Models_for_Adversarial_Purification.html
tags: [reviews]
---

# Title

Diffusion Models for Adversarial Purification

# 0. Simple terminology summary

본 논문 리뷰는 한국어로 쓰였고, 최대한 영어를 덜 쓰도록 노력하였습니다. 따라서 본 리뷰에서 사용하는 용어에 대해 먼저 간단히 정리를 해 보았습니다.

- Noise(잡음) : 일반적으로 예측할 수 없는 변동.
- Adversarial pertubation(적대적 교란 또는 적대적 변동) : 데이터에 적용할 수 있는 미세한 잡음.
- Adversarial attack(적대적 공격) : 데이터에 적대적 교란을 더하여 기계 학습 알고리즘을 공격하는 기법.
- Adversarial examples or Adversarial images(적대적 예시 또는 적대적 이미지) : 적대적 공격을 가한 데이터. 적대적 공격에 대항하지 않은 기계 학습 알고리즘은 이를 제대로 분류할 수 없다.
- Adversarial training(적대적 학습 또는 적대적 훈련) : 적대적 예시를 만든 후 모델 훈련에 통합하여 적대적 공격에 대해 대항하는 기법.
- Adversarial purification(적대적 정화) : 데이터의 적대적 교란을 정화하기 위해 생성 모형에 의존하는 기법.
- Unseen threats(보이지 않는 위협) : 적대적 훈련 기법이 학습하지 못한 적대적 공격
- Diffusion model(확산 모델) : 다음의 두 가지 과정을 통해 데이터를 생성하는 생성 모형.
    - Forward diffusion process(순방향 과정) : 데이터에 잡음을 점진적으로 더하는 과정. 이 과정을 계속 반복하다 보면 결국에는 데이터가 순수한 잡음으로 변한다.
    - Reverse generative process(역방향 과정) : 잡음을 한 단계씩 제거하는 과정. 이 과정을 통해 순수한 잡음으로부터 데이터를 생성한다.
- Adaptive attack(적응성 공격) : 특정 방어 모델을 목표로 설계된 공격
    - Strong adaptive attack(강력한 적응성 공격) : 방어 모델의 정확도를 기존보다 더 저하시킬 수 있는 공격
- Adjoint(수반 행렬) : 어떤 행렬에서 각 성분의 여인수 행렬로 이루어진 전치 행렬.
- SDE or Stochastic differential equation(확률 미분 방정식)

# 1. Introduction

## Motivation

1. 신경망은 적대적 공격에 취약하다. **데이터에 감지하기 어려운 변동을 주면 신경망이 잘못된 예측을 하도록 이끌 수 있다**. 
2. 적대적 공격에 대항하는 현재의 표준적인 형태는 적대적 훈련(adversarial training)이다. 
    1. 그러나 **대부분 적대적 훈련 기법은 그 모델이 훈련한 특정한 공격에만 대항**할 수 있다. 즉, **보이지 않는 위협에 대해 성능이 크게 저하**한다.
    2. 또한, **적대적 훈련의 계산 복잡도가 일반적인 훈련보다 보통 높다**.
3. 다른 방어 기법, 적대적 정화는 이미지의 적대적 교란을 정화하기 위해 생성 모형에 의존한다.
    1. 적대적 정화는 **모델을 다시 훈련하지 않고도 보이지 않는 위협에 대항**할 수 있다.
    2. 이러한 이점에도 불구하고 **성능은 적대적 훈련에 비해 떨어진다**. 특히 공격자가 방어 기법에 대한 완벽한 지식을 갖고 있을 때 성능은 더 크게 떨어진다. 이는 적대적 정화 모델에 사용되는 현재 생성 모델의 단점 때문이다.
4. 확산 모델(diffusion model)은 최근 강력한 생성 모델로 떠오르고 있다.
    1. **확산 모델의 생성 과정은 적대적 정화 모델의 역할과 유사**하다.
    2. 확산 모델의 **좋은 생성 능력과 다양성은 정화된 이미지가 원래의 이미지와 가까울 수 있도록 보장**한다.
    3. 확산 모델의 확률성(stochasticity)은 강력한 확률적 방어(stochastic defense)를 만들 수 있다.
5. 이러한 특성 덕분에 **확산 모형을 적대적 정화 기법의 이상적인 후보**로 생각할 수 있다.

## Main contributions

- ***DiffPure*** 모델을 제시한다. 이는 **사전 훈련된 확산 모델의 순방향, 역방향 과정을 통해 적대적 이미지를 정화**하는 첫 적대적 정화 기법이다.
- 레이블의 의미를 파괴하지 않고 적대적 교란을 제거할 수 있도록 순방향 과정에서 추가되는 잡음의 크기에 대한 이론적 분석을 제공한다.
- 강력한 적응 공격에 대한 평가를 위한 새로운 방법, ***adjoint method***를 제시한다. 이는 역방향 과정의 그래디언트 전체를 효율적으로 계산한다.

![image](https://user-images.githubusercontent.com/54873618/232223127-036b6b7a-f9d7-4c55-ad8f-7223cef1b1e8.png)

위 그림은 *DiffPure*를 설명한다. 사전 훈련된 확산 모델이 순방향 확산 과정(Forward diffusion process)을 통해 적대적 예시에 잡음을 더한다. $t=t^ *$에서 확산된 이미지를 얻을 수 있고, 이를 역방향 과정을 통해 깨끗한 이미지로 복원한다. 이 이미지를 분류에 사용한다. 

# 2. Background

이 섹션에서는 연속 시간 확산 모델(continuous-time diffusion model)에 대해 간단하게 리뷰한다.

- $p(x)$ : 알려지지 않은 데이터 분포. 생성 모델의 목표는 이를 추정하는 것이다.
    - 각 데이터 $x \in R^ d$가 이 분포로부터 추출된다.
    - 확산 모델은 $p(x)$를 잡음 분포(noise distribution)로 확산한다.
- 순방향 확산 과정(Forward diffusion process) $\{ x(t) \}_ {t \in [0, 1]}$는 SDE에 의해 다음과 같이 정의한다.
$dx = f(x, t)dt + g(t)dw\quad (1)$
    - 초깃값(initial value) : $x(0) := x \sim p(x)$
    - 드리프트 계수(drift coefficient) : $f : R^ d \times R \rightarrow R^ d$
    - 확산 계수(diffusion coefficient) : $g : R \rightarrow R$
    - 표준 위너 확률 과정(standard Wiener process) : $w(t) \in R^d$
    - $p_t(x)$ : $x(t)$의 주변 분포(marginal distribution)
        - $p_ 0(x) := p(x)$
    - $f(x,t)$와 $g(t)$는 순방향 확산 과정의 끝, 즉 $x(1)$이 표준 정규 분포(standard Gaussian distribution)를 따르도록 설계한다. 다시 말해서, $p_ 1(x) \approx N(0, I_ d)$
    - VP-SDE를 확산 모델로 사용한다.
        - $f(x, t) := -{1 \over 2}\beta(t)x$
        - $g(t) := \sqrt{ \beta(t) }$
            - $\beta(t)$ : 시간 의존적인 잡음 스케일(time-dependent noise scale)
            - 기본적으로 선형 잡음 스케줄(linear noise schedule)을 사용한다. 즉, $\beta(t) := \beta_ {\min} + (\beta_ {\max} - \beta_ {\min})t$
- 표본 생성은 역-시간 SDE(reverse-time SDE)를 통해서 가능하다.
    - $d \hat x = [f_ {(\hat x, t)} - g(t)^ 2 \nabla {\hat x} \log{p t(\hat x)}]dt + g(t)d\bar w \quad (2)$
    - $dt$ : 미소한 음의 시간 단계(An infinitesimal negative time step)
    - $\bar w (t)$  : 표준 역-시간 위너 확률 과정(standard reverse-time Wiener process)
    - $\hat x(1) \sim N(0, I_ d)$를 초깃값으로 설정한 채 위의 역-시간 SDE를 $t=1$에서 $t=0$까지 해결하는 것은 **데이터 분포에서 표본을 추출할 때까지(다시 말해서 $\hat x(0) \sim p_ {0}(x)$ ) 잡음이 덜한 데이터 $\hat x(t)$를 점진적으로 생성**한다.
    - 이상적으로, **식 (2)에서 나온 잡음 제거 과정(denoising process)의 결과 $\{ \hat x(t) \}_ {t \in [0, 1]}$는 식 (1)에서 얻은 순방향 확산 과정 $\{ x(t) \}_ {t \in [0,1]}$와 같은 분포**를 갖는다.
- 식 (2)의 역-시간 SDE는 시간에 의존적인 스코어 함수(time-dependent score function) $\nabla_ x \log{ p_ t(x) }$의 지식을 필요로 한다. 자주 쓰이는 접근 방법 중 하나는 매개 변수화된 신경망(parameterized neural network) $s_ \theta(x, t)$를 이용해 $\nabla_ x \log {p_ t(x)}$를 추정하는 것이다.
    - 확산 모델은 여러 시간 단계에 걸쳐 잡음 제거 스코어 매칭(denoising score matching)의 가중 조합으로 훈련된다.
    ![image](https://user-images.githubusercontent.com/54873618/232224743-6d6b4b70-f204-4078-ae7c-56728bf8c5ee.png)
        - $\lambda(t)$ : 가중치 계수(weighting coefficient)
        - $x(0) := x$에서부터 $x(t) := \tilde x$까지 전이 확률(transition probability) : $p_ {0t}(\tilde x \vert x)$
            - 식 (1)의 순방향 SDE를 통해 닫힌 식을 갖는다.

# 3. Method

본 논문에서는 다음 두 가지 기법을 제시한다.

1. ***Diffusion purification*** (또는 ***DiffPure***) : 확산 모델을 사용하여 적대적 이미지를 깨끗한 이미지로 복원하는 기법
2. ***Adjoint method*** : 강력한 적응 공격에 대해 효율적인 gradient evaluation을 하기 위해 SDE를 역전파하기 위한 기법

## 3.1 확산 정화(Diffusion Purification)

*DiffPure*는 다음과 같은 가정에서 시작한다. 주어진 적대적 예시 $x_ a$에 대해 $x(0) = x_ a$에서부터 **순방향 확산 과정**을 시작하면 데이터에 더해진 작은 지역적 구조인 **적대적 변동은 점진적으로 부드러워질(gradually smoothed) 것**이다. 

다음 정리는 깨끗한 데이터 분포 $p(x)$와 적대적 변동이 가해진 데이터 분포 $q(x)$가 순방향 확산 과정을 진행하면서 점점 가까워진다는 사실을 나타낸다. 이는 **적대적 변동이 잡음을 더하면서 결국에는 사라질 것임을 암시**한다.

### 정리 3.1

$\{x(t)\}_ {t \in [0,1]}$를 식 (1)의 순방향 SDE에 의해 정의된 확산 과정이라고 한다. 만약 $p_ {t}$와 $q_ {t}$를 $x(0) \sim p(x)$(즉, 깨끗한 데이터 분포) 와 $x(0) \sim q(x)$(적대적 표본 분포)일 때의 각각의(respective) 분포로 정의한다면 다음과 같은 식을 얻을 수 있다.

${\partial D_ {KL}(p_ t \vert \vert q_ t) \over \partial t} \le 0$

위의 부등식은 $p_ t = q_ t$일 때 등식이 된다. 즉, $p_ t$와 $q_ t$의 쿨백-라이블러 발산(KL divergence)은 순방향 SDE에서 $t=0$에서부터 $t=1$로 움직일 때 단조 감소(monotonically decreases)한다.

위 정리에 따라 $D_ {KL}(p_ {t^ {\*}} \vert \vert q_ {t^ {\*}}) \le \epsilon$ 을 만족하는 최소 시간 단계(timestep) $t^ {\*} \in [0, 1]$가 존재한다. 그러나 시간 단계 $t = t^ {\*}$에서 확산된 적대적 표본 $x(t^ {\*}) \sim q_ {t^ {\*}}$는 추가적인 잡음을 포함하고 있고 직접적으로 분류될 수 없다. 그러므로 식 (2)의 SDE를 통해, **$x(t^ {\*})$에서부터 시작하여 확률적으로 $t = 0$에서의 깨끗한 데이터를 복원**할 수 있다.

### 확산 정화

확산 모델을 이용하는 두 단계로 이루어진 적대적 정화 기법을 제시한다.

1. $t = 0$에서 주어진 적대적 예시 $x_ {a}$에 대해 식 (1)의 순방향 SDE를 $t=0$에서부터 $t = t^ {\*}$까지 해결하여(solve) 이를 확산한다. VP-SDE에 대해, 확산 시간 단계 $t^ {\*} \in [0,1]$에서 확산된 적대적 표본은 다음 식을 사용해 효율적으로 추출할 수 있다.
$x(t^ {\*}) = \sqrt{\alpha(t^ {\*})}x_ {a} + \sqrt{1 - \alpha(t^ {\*})}\epsilon\quad (3) \\ \text{ where } \alpha(t) = e^ {- \int^ {t_ {0}} \beta(s)ds} \text{ and } \epsilon \sim N(0, I_ d).$
2. 시간 단계 $t = t^ {\*}$에서 식 (2)의 역-시간 SDE를 식 (3)에서 주어진 확산된 적대적 표본 $x(t^ {\*})$를 사용해서 해결한다. 이 적대적 표본은 식 (2)의 SDE의 마지막 솔루션 $\hat x(0)$을 얻기 위한 초깃값이다. $\hat x(0)$은 닫힌 식 솔루션이 없기 때문에 sdeint로 불리는 SDE 솔버(solver)를 이용한다.
$\hat x(0) = \text{sdeint}( x(t^ {\*}), f_ {rev}, g_ {rev}, \bar w, t^ {\*}, 0)$
sdeint는 여섯 개의 입력(초깃값, 드리프트 계수, 확산 계수, 위너 확률 과정, 초기 시간, 끝 시간)을 받는다. 또한, 드리프트와 확산 계수는 다음과 같이 주어진다.
$f_ {rev}(x,t) := -{1 \over 2} \beta(t)[x + 2s_\theta(x,t)] \\ g_ {rev}(t) := \sqrt{\beta(t)}$
결과로 얻은 정화된 데이터 $\hat x(0)$은 예측을 위해 외부의 표준 분류기(standard classifier)로 전달된다. 

### 확산 시간 단계 $t^ *$ 정하기

정리 3.1로부터 $t^ *$는 지역적(local) 적대적 변동을 제거하기 위해 충분히 커야 한다는 사실을 알 수 있다. 그러나 이는 무작정 커서는 안 되는데, 왜냐하면 전역적(global) 레이블 의미(label semantics)가 $t^ *$가 증가하면서 확산 과정 속에서 같이 제거될 수 있기 때문이다. 이는 정화된 표본 $\hat x(0)$가 정확하게 분류될 수 없는 결과로 이어질 수 있다.

다음 정리는 어떻게 확산 시간 단계 $t^ *$가 깨끗한 이미지 $x_ 0$과 정화된 이미지 $\hat x(0)$의 차이에 영향을 끼치는지 나타낸다.

### 정리 3.2

만약 스코어 함수가 $\vert \vert s_ \theta(x,t) \vert \vert \le {1 \over 2}C_ s$를 만족한다면 식 (4)로부터 주어진 깨끗한 데이터 $x$와 정화된 데이터 $\hat x(0)$의 $L2$ 거리는 적어도 $1 - \delta$의 확률로 다음을 만족한다. 

$\vert \vert \hat x(0) - x \vert \vert \le \vert \vert \epsilon_ a \vert \vert + \sqrt{ e^ {2\gamma(t^ *)} -1 }C_\delta + \gamma(t^ *)C_s$

$\epsilon_ a$는 $x_ a= x + \epsilon_a$, $\gamma(t^ *) := \int^ {t^ *}_ 0 {1 \over 2}\beta(s)ds$ 그리고 상수 $C_ \delta := \sqrt{ 2d + 4\sqrt{d \log{1 \over \delta}}+4\log{1 \over \delta}}$를 만족하는 적대적 변동을 뜻한다.

$\gamma(t^ *)$는 $t^ *$ 에 따라 단조적으로 증가하고 $\forall t^ *,\gamma(r^ *) \ge 0$ 이기 때문에 위 정리의 상한의 마지막 두 항은 $t^ *$ 에 따라 증가한다. 그러므로, $\vert \vert \hat x(0) - x \vert \vert $를 최대한 낮게 하기 위해서 $t^ *$는 충분히 작아야 한다. 극단적으로, $t^ *= 0$인 상황에서 우리는 $\vert \vert \hat x (0) - x \vert \vert = \vert \vert \epsilon_a\vert \vert$를 갖는다. 이는 우리가 확산 정화를 수행하지 않으면 $\hat x(0)$이 $x_a$로 감소한다는 사실을 의미한다.

적대적 예시의 지역 변동을 정화하는 것(큰 $t^ *)$과 전역 구조를 보존하는 것(작은 $t^ *$)의 상충 관계 때문에 **높고 강건한 분류 정확도(high robust classification accuracy)를 얻을 수 있는 좋은 확산 시간 단계(diffsuion timestep) $t^ *$가 존재**한다. 적대적 교란은 보통 작기 때문에 작은 $t^ *$로 제거될 수 있고, 대부분 최적의 $t^ *$는 상대적으로 작다. 

## 3.2 확산 정화에 대한 적응성 공격

강력한 적응성 공격은 방어 체계의 전체 그래디언트를 계산해야 한다. 그러나 식 (4)에 있는 SDE 솔버를 통해 간단하게 역전파를 하는 것은 계산 메모리(computational memory)에서 제대로 확장되지 않는다. 특히 SDE를 해결하는 데에 쓰이는 함수 평가의 수 N개를 나타내면 필요한 메모리는 O(N) 만큼 증가한다. 이러한 문제 때문에 강력한 적응 공격에 대항하는 본 논문의 기법을 효율적으로 평가하기가 어렵다. 

이전의 적대적 정화 기법은 강력한 적응 공격에 대한 메모리 문제 때문에 애를 먹었다. 그래서 그들은 블랙박스 공격에 대해서만 평가를 하거나 전체 그래디언트 계산을 피하도록 평가 전략을 변경했다(예를 들면 그래디언트를 근사해서). 이는 더 표준적인 평가 프로토콜 하에서 적대적 훈련 기법의 비교를 어렵게 했다. 이를 극복하기 위해 이 논문에서는 *adjoint method*를 제안한다. 이는 SDE의 전체 그래디언트를 메모리 문제 없이 효율적으로 계산할 수 있는 방법이다. 또다른 증가된(augmented) SDE를 해결하는 것으로 SDE의 그래디언트를 얻을 수 있다고 직관적으로 생각할 수 있다.

이어지는 명제는 식 (4)에 있는 SDE의 입력 $x(t^ *)$에 대해 목적 함수 $L$의 그래디언트를 계산하는 데에 필요한 증가된 SDE를 제시한다. 

### 명제 3.3

식 (4)에 있는 SDE에 대해, 역전파의 그래디언트 $\partial L \over \partial x(t^ *)$를 계산하는 증가된 SDE는 다음과 같이 주어진다.

![image](https://user-images.githubusercontent.com/54873618/232223138-6b6bbd62-92b7-448c-b140-e439af30562a.png)

$\partial L \over \partial \hat x(0)$은 식 (4)의 SDE의 결과 $\hat x(0)$에 대한 목적 함수 $L$의 그래디언트다. 그리고,

![image](https://user-images.githubusercontent.com/54873618/232223141-a8421042-540d-46ab-99d9-489bbb475876.png)

$1_d,0_d$는 각각 전부 1, 혹은 0인 d차원 벡터를 뜻한다.

이상적으로 만약 SDE 솔버가 작은 수치적 오차를 갖는다면 이 명제로부터 얻은 그래디언트는 참값과 매우 가깝다. 그래디언트 계산이 식 (6)의 증가된 SDE를 해결하는 것으로 변환되었으므로, 우리는 중간 작업을 저장할 필요가 없다. 그러므로 **메모리 비용이 O(1)** 에 그친다. 즉, *adjoint method*는 식 (4)의 역-시간 SDE를 미분 가능한 연산으로 변환한다(메모리 문제 없이). 식 (3)의 순방향 확산 과정 또한 재매개변수화(reparametrization) 기법을 써서 미분이 가능하기 때문에 **강력한 적응성 공격에 대한 적대적 이미지에 관한 손실 함수의 전체 그래디언트를 쉽게 계산**할 수 있다. 

# 4. Related Work

## Adversarial training

모든 가중치 업데이트 중에 만들어지는 적대적 예시를 훈련하는 것으로 강건한 분류기를 얻을 수 있다. 적대적 훈련은 적대적 공격에 대항하는 성공적인 신경망 기법이 되었다. 방어 형태의 차이에도 불구하고 몇 개의 적대적 훈련은 DiffPure와 유사한 점을 공유한다. Gowal et al. (2021)은 적대적 훈련을 개선하기 위해 생성 모형을 이용하여 데이터를 증강하였고, 확산 모형이 가장 좋은 성능을 보였다.

## Adversarial purification

분류 전 적대적 이미지를 정화하기 위해 생성 모형을 사용하는 적대적 정화는 적대적 훈련의 유망한 호적수가 되었다. 특히 Samangouei는 GAN을 정화 모형으로 사용하는 defense-GAN을 제시하였고 Song은 PixelDefense 모형을 제시하였는데 이는 autoregressive 생성 모형에 의존한다. 최근 Du & Mordatch (2019); Grathwohl et al. (2020); Hill et al. (2021)는 공격을 당한 이미지를 정화하기 위한 개선된 robustness of using EBM을 보였다. 우리의 기법과 유사하게, Yoon은 denoising score-based model을 정화를 위해 사용하였으나 그 모형의 표본 추출 기법은 Langevin dynamics(LD)의 다른 한 가지 방법에 지나지 않았다. DiffPure는 예전 모델과 경험적으로 비교되었고 승리를 거두었다. 

## Image editing with diffusion models

비지도 모델링을 위한 확률 생성 모델로서 확산 모델은 좋은 표본 추출 성능과 이미지 합성에서의 다양성을 보였다. 그 이후로 이미지-to-이미지 번역(translation)이나 텍스트-기반 이미지 수정(text-guided image editing)과 같은 많은 이미지 수정 작업에 쓰였다. 비록 적대적 정화 역시 특정 이미지 수정 도구로 고려되고, 특히 DiffPure은 SDEdit와 유사한 절차를 공유하지만 이 모든 것 중 어떤 것도 모형의 강건성을 개선하기 위해 확산 모형을 적용하지는 않았다. 게다가, 강력한 적응 공격에 대한 DiffPure을 평가하는 것은 이전 연구가 다루지 않았던 잡음 제거 과정을 통해 역전파하는 새로운 과제다. 

# 5. Experiment

## 5.1 Experimental settings

### 데이터와 네트워크 구조

평가를 위해 세 가지 데이터를 고려한다.

1. CIFAR-10
2. CelebA-HQ
3. ImageNet

특히, CIFAR-10 및 ImageNet에 대해 표준화된 벤치마크 RobustBench에 의해 보고된 SOTA 방어 기법과 비교한다. 동시에 CIFAR-10 및 CelebA-HQ에 대해서 다른 적대적 정화 방법과도 비교한다.

분류에 대해서는 널리 쓰이는 세 가지 구조를 고려한다.

1. ResNet
2. WideResNet
3. ViT

### 적대적 공격

강력한 적응성 공격에 대해서도 기법을 평가한다. 공통적으로 쓰이는 AutoAttack $l_ {\infty}$과 $l_ {2}$위협 모델을 사용하여 적대적 훈련 기법과 비교한다. $l_ p$-norm 공격뿐 아니라 다양하게 적용될 수 있는 DiffPure의 능력을 보이기 위해 부분적으로 변환된 적대적 예시에 대해서도 또한 평가한다. 확산과 잡음 제거 과정에 있는 확률성(stochasticity) 덕분에 이러한 적응성 공격에 대해 Expectation Over Time(EOT)을 적용할 수 있다. EOT는 20을 사용한다. 게다가 다른 적대적 정화 기법과의 공정한 비교를 위해 BPDA+EOT 공격도 적용한다. 

### 평가 지표

방어 기법의 성능을 평가하기 위해 두 가지 지표를 고려한다.

1. 표준 정확도(*standard accuracy*) : 깨끗한 데이터에 대해 방어 기법의 성능을 측정한다. 이는 각 데이터 세트에 있는 모든 테스트 세트를 평가한다.
2. 강건한 정확도(*robust accuracy*) : 적대적 공격으로부터 생성된 적대적 예시에 대한 성능을 측정한다. DiffPure에 적응성 공격을 가하는 데에는 많은 비용이 소모된다. 따라서 강건한 정확도를 측정하기 위해 테스트 데이터 세트에서 임의로 추출한 512개의 고정된 이미지에 대해서만 평가를 진행한다(DiffPure와 이전 모델 모두). 한 가지 알아야 할 사실은, 테스트 데이터 세트 전체를 적용했을 때와 그것에서 추출한 일부 데이터 세트에만 적용했을 때 모두 대부분의 베이스 라인의 강건한 정확도는 달라지지 않는다는 것이다. 

## Result

## 5.2 SOTA와 비교

### CIFAR-10

![image](https://user-images.githubusercontent.com/54873618/232223147-d70df401-11ec-4145-a725-831233757ee2.png)

위 테이블은 CIFAR-10 데이터를 공격한 $l_ {\infty}$ 위협 모델($\epsilon = 8/255$)에 대한 DiffPure의 강건성을 나타낸 것이다. 해당 기법이 **표준 정확도와 강건한 정확도에 있어 모두 이전의 SOTA 모델보다 나은 성능**을 보였다. 

![image](https://user-images.githubusercontent.com/54873618/232223151-66c706b8-cf1f-4dfd-b07b-9ebda2fd76e4.png)

위 테이블은 CIFAR-10 데이터를 공격한 $l_ 2$ 위협 모델($\epsilon = 0.5$)에 대한 강건성을 나타낸 것이다. DiffPure가 다른 방어 기법보다 **표준 정확도와 강건한 정확도에 있어 우수한 성능**을 보였다. DiffPure와 추가 데이터로 훈련한 기법 사이의 격차는 존재한다. 하지만 아래 그림과 같이 DiffPure에서 표준 분류기를 적대적으로 훈련된 분류기로 대체하여 수준을 높일 수 있다.

![image](https://user-images.githubusercontent.com/54873618/232223154-975157ac-0bca-4238-91d8-1d904bc53bff.png)

이러한 결과는 $l_ {\infty}$과 $l_ 2$ 위협 모델 모두를 방어하는 데에 있어 DiffPure의 효과를 나타낸다. 평가에 사용되는 특정 $l_ p$-norm 공격에 대해 훈련된 다른 기법과 달리 **DiffPure는 위협 모델에 구애를 받지 않는다**.

### ImageNet

![image](https://user-images.githubusercontent.com/54873618/232223161-e81a06fb-8124-4075-b57a-656acf577a64.png)

위 테이블은 ImageNet을 공격한 $l_ {\infty}$ 위협 모델($\epsilon = 4/255$)에 대한 강건성을 나타낸 것이다. DiffPure을 두 개의 CNN 구조 ResNet-50과 WideResNet-50-2, 그리고 한 개의 ViT 구조 DeiT-S로 평가하였다. DiffPure은 **SOTA보다 표준 정확도, 강건한 정확도 모두에서 더 좋은 성능**을 보인다. 이 기법의 이점은 ViT 구조에서 더욱 두드러진다. 

이러한 결과는 ImageNet을 공격한 $l_ {\infty}$ 위협 모델을 방어하는 데에 있어 해당 기법의 효과를 나타낸다. 또한, **DiffPure는 분류기의 구조의 구애를 받지 않는다**.

## 5.3 보이지 않는 위협에 대한 방어

적대적 훈련의 약점은 보이지 않는 공격에 대해 취약하다는 점이다. 모델이 특정 위협 모델에 강건하다고 하더라도 다른 위협 모델에 취약하다. 이를 확인하기 위해 세 가지 공격, $l _ \infty, l_ 2$ 그리고 StAdv를 이용하여 평가하였다. 적대적 훈련 기법은 특정 공격만 훈련하고 다른 공격에 대해서는 훈련하지 않았다. 예를 들어 아래 테이블에서 첫 번째 기법은 $l_ \infty$ 공격에 대해서만 훈련하였고, 따라서 나머지는 보이지 않는 위협과 같다. 아래 테이블에서 보이는 위협(즉, 이미 훈련한 공격)은 회색 글자로 하였다.

![image](https://user-images.githubusercontent.com/54873618/232223175-47ccbaca-7c3f-4bbd-beae-f5b0b98c897d.png)

DiffPure은 **세 가지 보이지 않는 위협 모두에 있어 적대적 훈련보다 강건한 성능**을 보였다. 보이지 않는 위협에 대한 SOTA 방어 모델과 비교했을 때, DiffPure는 **표준 정확도와 강건한 정확도 모두에 있어 우수한 성능**을 보였다. 

## 5.4 다른 정화 기법과 비교

대부분의 적대적 정화 기법은 최적화(optimization) 또는 표본 추출 루프(sampling loop)를 방어 과정에 포함한다. 때문에 이러한 기법들은 AutoAttack과 같은 가장 강력한 화이트-박스 적응 공격(strongest white-box adaptive attacks)에 대해 직접적으로 평가되지 못한다. 이를 해결하기 위해 우리는 BPDA+EOT 공격을 사용하였다. 이는 특정 정화 기법에 대해 설계된 적응성 공격이고 DiffPure과 베이스라인을 공정하게 비교할 것이다. 

### CelebA-HQ

적대적 예시를 정화할 수 있는 NVAE나 Style-GAN2 같은 다른 강력한 생성 모델과 DiffPure을 비교하였다. 기본적인 아이디어는 적대적 이미지를 먼저 잠재 코드(latent code)로 변환한 뒤 디코더를 이용하여 정화된 이미지를 합성하는 것이다. 비교를 위해 CelebA-HQ를 택하였는데, 왜냐하면 앞서 말한 두 모델이 해당 데이터에 대해 잘 작동하기 때문이다. 

![image](https://user-images.githubusercontent.com/54873618/232223192-1b832aea-a7ad-4798-855d-2ca1b61d3eda.png)

DiffPure의 강건성을 보이기 위해 *eyeglasses* 속성을 사용하였다. **DiffPure는 베이스 라인에서 제일 우수한 성능보다 무려 15%나 향상된 강건한 정확도와 더불어 뒤떨어지지 않는 표준 정확도**를 보였다.

이러한 결과는 **적대적 강건성에 있어 다른 생성 모델보다 확산 모델의 우수한 성능**을 설명한다. 

### CIFAR-10

DiffPure을 CIFAR-10 데이터를 통해 다른 적대적 정화 기법과 비교하였다. 

![image](https://user-images.githubusercontent.com/54873618/232223199-0510db60-ce4e-431a-bf6f-db306e95c7a8.png)

DiffPure는 BPDA+EOT 공격에 대한 이전 모델의 방어 성능보다 강건한 정확도에 있어 11.31%나 향상한 우수한 성능을 보였다. 표준 정확도 역시 크게 뒤떨어지지 않는다. 

위의 결과는 **DiffPure가 적대적 정화 필드에서 새로운 SOTA 모델**이 되었다는 사실을 보인다.

# 6. Conclusion

본 논문의 결론은 다음과 같다. 

1. 새로운 기법 ***DiffPure***을 선보였다. 이는 확산 모델을 사용하여 적대적 예시를 정화한다. 이는 이전의 다른 모델에 비해 우수한 성능을 보였다. 
2. ***Adjoint method***를 제시하였다. 이는 강력한 화이트-박스 적응 공격을 평가하기 위해 SDE 솔버의 전체 그래디언트를 계산한다. 
3. DiffPure의 강건성을 보이기 위해 다양한 데이터(CIFAR-10, ImageNet 등)를 통해 이를 SOTA 모델과 비교하여 평가하였다. AutoAttack, StAdv와 같은 강력한 적응성 공격을 방어하는 데에 DiffPure는 이전의 다른 접근과 비교하여 우수한 성능을 보였다. 

위와 같은 많은 개선에도 불구하고 DiffPure는 다음과 같은 두 가지 중요한 한계를 갖는다.

1. 정화 과정은 많은 시간을 소요한다. 이는 해당 기법이 실시간 작업에는 적합하지 않다는 사실을 뜻한다.
2. 확산 모델은 이미지의 색에 민감하다. 이는 해당 기법이 색에 연관된 오염을 제대로 방어하지 못한다는 사실을 나타낸다. 

이러한 두 가지 한계를 극복하기 위해 확산 모델 가속화(accelerating)에 대한 최근 연구를 적용하거나 모델의 강건성을 위해 새로운 확산 모델을 설계하는 등의 관심을 기울일 필요가 있다. 

---

# Author Information

- Name : SeongWon Kim(김성원)
- Master student in StatHT, Graduate School of Data Science, KAIST
- Email : ksw888@kaist.ac.kr
- github : https://github.com/Won-Seong/
