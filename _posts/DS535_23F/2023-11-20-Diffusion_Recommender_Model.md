---
title:  "[SIGIR 2023] Diffusion Recommender Model"
permalink: 2023-10-16-Diffusion_Recommender_Model.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Diffusion Recommender Model

# Title

**Diffusion Recommender Model**

## 1. Introduction

## 1.1 Motivation

**생성 모형은 개인 맞춤형 추천(*personalized recommendation*)에 널리 사용**이 되어 왔다. 일반적으로, **생성 추천 모형(*generative recommender models*)은 상호 작용(*interaction*)을 하지 않는 모든 아이템에 대해 사용자의 상호 작용 확률을 추론하는 생성 과정을 학습**한다. 이러한 생성 추천 모형은 실제 세계의 상호 작용 생성 절차와 일치하여 상당한 성공을 거두었다.

생성 추천 모형은 주로 두 그룹으로 나뉜다.

- GAN 기반 모형: **사용자의 상호 작용 확률을 추정하고 적대적 훈련(*adversarial training*)을 활용하여 최적의 매개변수**를 찾는다. 그러나, **적대적 훈련은 일반적으로 불안정**하며 이는 만족스럽지 않은 성능으로 이어진다.
- VAE 기반 모형: **인코더를 사용하여 잠재 요인(*latent factors*)에 대한 사후 분포를 근사하고 관측된 상호 작용의 가능도를 최대화**한다. VAE 기반 모형은 추천 시스템에서 일반적으로 GAN 기반 모형보다 나은 성능을 보이지만 **계산의 용이성(*tractability*)과 표현 능력(*representation ability*) 사이에 상충 관계**로 어려움을 겪는다.

### 1.2 Diffusion Models

확산 모형(*Diffusion Models*, DMs)은 이미지 합성 작업에서 SOTA를 달성한 또 하나의 생성 모형이다. 이는 다음과 같은 특징을 갖는다.

- VAE가 갖는 상충 관계를, **계산 가능한 순방향 과정을 통해 이미지를 점진적으로 오염시키고 이를 반복적으로 복원하는 방법을 학습**하는 것으로 완화하였다.
- 그림 1(b)에서 볼 수 있듯, 확산 모형은 먼저 $x_0$를 임의의 잡음으로 점진적으로 오염시킨다. 이후, 오염된 $x_ T$로부터 $x_ 0$를 복원시킨다.
- 이러한 순방향 과정은 계산 가능한 사후 분포로 이어지며, 역방향 과정에서 유연한(*flexible*) 신경망을 통해 복잡한 분포를 반복적으로 모델링하는 방법을 제시하였다.

**추천 모형의 목적은 확산 모형과 잘 어울**린다. 왜냐하면 **추천 모형은 본질적으로 손상(*corrupted*)된 과거의 상호 작용을 기반으로 미래의 상호 작용 확률을 추론**하기 때문이다. 그림 1(c)를 보면 더욱 쉽게 이해할 수 있다. 추천 모형은 사용자의 과거의 상호 작용(*histrocial interactions*)에 기반하여 미래의 상호 작용(*future interactions*)을 예측하고자 한다. 과거의 상호 작용이 깨끗한 데이터라면 이러한 접근은 매우 쉽다. 그러나, 일반적으로 과거의 상호 작용은 손상이 되어 있다. 이 경우 **손상은 거짓 양성(*false-positive*)과 거짓-음성(*false-negative*)에 의해 오염된 상호 작용을 암시**한다. 예를 들어, 어떤 아이템에 대해 사용자는 긍정적인 상호 작용을 하였지만 실은 그다지 마음에 들지 않았을 수도 있다. 혹은, 부정적인 상호 작용을 하였으나 실제로는 괜찮았을 수도 있다. 이렇게 **잡음이 섞인 과거의 상호 작용을 통해 미래의 상호 작용을 만들어 내는 게 추천 모형의 목적이고, 이는 잡음에서 깨끗한 데이터를 만들어 내는 확산 모형과 매우 어울**린다. 따라서, 추천을 위해 확산 모형을 이용하는 건 자연스러운 일이라고 생각할 수 있다. 확산 모형에게는 강력한 표현 능력으로 복잡한 상호 작용을 더 정확하게 포착할 수 있는 잠재력이 있다. 

![Untitled](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/6b46ce56-ccd1-4a65-8811-60d9c41c6046)


## 1.3 DiffRec

이 논문에서 그들은 **확산 추천 모형(*Diffusion Recommender Model*, DiffRec)** 을 제시한다. 이 모형은 다음과 같은 특징을 갖는다. 

- **잡음을 제거하는(*denoising*) 관점에서 사용자의 상호 작용 확률을 추론**한다.
- 기술적으로, DiffRec은 사용자의 상호 작용 과거(*interaction histroies*)를 **순방향 과정에서 사전에 주어진 가우시안 잡음(*scheduled Gaussian noises*)을 주입하는 것으로 오염**시킨다.
- 역방향 과정에서, 매개 변수화된(*parameterized*) 신경망을 통해 **오염된 상호 작용으로부터 원래의 상호 작용을 복원**한다. 이는 반복적인 과정으로 이루어진다.
- 그러나 이미지 생성에서 사용했던 순방향 과정을 그대로 이용할 수는 없는데, 왜냐하면 개인 맞춤형 추천 생성의 필요성 때문이다. 이미지 생성에서는 순방향 과정에서 이미지가 순수한 잡음이 될 때까지 이를 오염시킨다. 그러나  추천 모형에서는 사용자의 손상된 상호 작용에서 개인화된 정보를 유지하기 위해서는 이를 피해야 한다. 따라서, **순방향 과정에서 추가되는 잡음의 크기를 줄일 필요**가 있다.

한걸음 더 나아가, 추천 시스템을 위한 생성 모형을 구축하는 데에 존재하는 두 가지 본질적인 어려움을 이 논문에서는 다룬다: **대규모 아이템 예측(*large-scale item prediction*)과 시간 모델링(*temporal modeling*)** 이다.

1. **생성 모형은 동시에 모든 아이템의 상호 작용 확률을 예측하는데, 이는 상당한 자원을 요구**한다. 따라서 대규모 아이템 추천을 위해 생성 모형을 응용하는 게 제한된다.
2. **생성 모형은 상호 작용 시퀀스에서 시간적 정보를 포착**해야 한다. 이러한 시간적 정보는 사용자의 선호도 변화를 처리하는 데에 중요한 역할을 한다. 

이러한 어려움을 해결하기 위해, 이 논문에서는 DiffRec을 *Latent DiffRec*(L-DiffRec)과 *Temporal DiffRec*(T-DiffRec)으로 확장한다. 

1. **L-DiffRec은 아이템을 그룹으로 클러스터링하고 각 그룹의 상호 작용 벡터를 *group-specific VAE*를 통해 저차원 잠재 벡터로 압축**한다. 이후, **잠재 공간에서 순방향 과정과 역방향 과정을 수행**한다. **클러스터링과 잠재 확산 덕분에 L-DiffRec은 모형의 모수와 메모리 비용을 상당히 줄이고, 대규모 아이템 예측의 능력을 향상**시킨다. 
2. **T-DiffRec은 간단하지만 효과적인 *time-aware reweighting* 전략을 통해 상호 작용 시퀀스를 모델링**한다. 직관적으로, **사용자의 더 나중의 상호 작용은 더 큰 가중치**를 부여 받고, 이는 이후 훈련과 추론을 위해 DiffRec에 입력된다.

이 논문에서 그들은 세 개의 대표적인 데이터로 실험을 하였으며 다양한 설정에서 DiffRec과 기존에 존재하는 모형을 비교하였다. 

## 1.4 Summary of introduction

- 그들은 확산 추천 모형을 제시하였으며, 이는 생성 추천 모형의 방향성을 가리키는 완전히 새로운 추천 패러다임이다.
- 그들은 기존의 확산 모형을 확장하여 고-차원 범주형 예측을 하기 위해 쓰이는 비용을 줄였고 상호 작용 시퀀스의 *time-sensitive* 모델링을 가능하게 하였다.
- 그들은 다양한 설정에서 세 개의 데이터에 대해 실험을 수행하였으며 기존에 존재하는 모형과 DiffRec을 비교하여 놀라운 성능 향상을 보였다.

# 2. Preliminary

확산 모형은 주로 순방향과 역방향 과정으로 구성되어 있다.

## 2.1 Forward process

주어진 입력 데이터 표본 $x_0 \sim q(x_0)$에 대해, 순방향 과정은 $T$ 단계 동안 점진적으로 가우시안 잡음을 더하는 것으로 마르코프 연쇄 내에서 잠재 변수 $x_ {1:T}$를 구축한다. 구체적으로 말하면, 확산 모형은 순방향 전이(*forward transition*) $x_ {t-1} \rightarrow x_ {t}$를 $q(x_ t \vert x_ {t-1}) = N(x_ t; \sqrt{1 - \beta_ t} x_ {t-1}, \beta_ t I)$로 정의한다.

- $t \in \{1, \dots, T\}$는 확산 단계(*diffusion step*)를 나타낸다.
- $N$은 가우시안 분포를 지칭한다.
- $\beta_ t \in (0, 1)$은 단계 $t$에서 더해지는 잡음의 크기를 조절한다.
- $T \rightarrow \infty$일 때, $x_ T$는 표준 가우시안 잡음으로 접근한다.

## 2.2 Reverse process

확산 모형은 역 과정에서 $x_ {t-1}$을 복원하기 위해 $x_ t$로부터 더해진 잡음을 제거하는 방법을 학습한다. 이는 복잡한 생성 과정의 작은 변화를 포착하는 것을 목표로 한다. 더 자세히 말하자면, 확산 모형은 $x_ T$를 초기 상태로 받아 $x_ t \rightarrow x_ {t-1}$와 같은 잡음 제거 과정을 반복적으로 학습한다. 이 잡음 제거 과정의 전이는 $p_ \theta ( x_ {t-1} \vert x_ t) = N( x_ {t-1} ; \mu_ \theta (x_ t, t), \Sigma_ \theta(x_ t, t))$으로 정의된다. $\mu_ \theta(x_ t, t)$와 $\Sigma_ \theta (x_ t, t)$는 가우시안 잡음의 평균과 공분산으로, $\theta$로 매개 변수화된 신경망에 의해 예측된다.

## 2.3 Optimization

확산 모형은 관측된 입력 데이터 $x_ 0$의 가능도의 증거 하한(*Evidence Lower BOund*, ELBO)을 최대화하는 것으로 최적화된다.

![Untitled 1](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/cabd7b53-e67a-4c25-9e3a-3efa4cbf8ac8)

- 복원 항(*reconstruction term*)은 $x_0$에 대한 음의 복원 오차(*negative reconstruction error*)를 의미한다.
- 사전 매칭 항(*prior matching term*)은 상수이므로 최적화 과정에서 무시할 수 있다.
- 잡음 제거 매칭 항(*denoising matching term*)은 $p_ \theta(x_ {t-1} \vert x_ t)$를 계산 가능한 참(*ground-truth*) 전이 단계 $q( x_ {t-1} \vert x_ t, x_ 0)$과 최대한 같아지도록 조정한다.
- $\theta$는 $x_ t$로부터 $x_ {t-1}$을 반복적으로 복원하기 위해 최적화된다.
- 잡음 제거 매칭 항은 $\sum {E_ {t, \epsilon} {[ \lVert \epsilon - \epsilon_ \theta (x_ t, t) \rVert ^ 2 _ 2 ]}}$으로 간단하게 쓸 수 있다.
    - $\epsilon \sim N(0 , I)$
    - $\epsilon_ \theta(x_ t, t)$는 $\epsilon$을 예측하기 위해 신경망에 의해 매개 변수화된다. 이 $\epsilon$은 순방향 과정에서 $x_ 0$에서 $x_ t$를 결정하기 위해 쓰인다.

## 2.4 Inference

$\theta$를 학습한 이후, 확산 모형은 $x_T \sim N(0, I)$를 추출하고 $p_ \theta ( x_ {t-1} \vert x_ t)$를 이용하여 생성 과정을 반복한다. 이러한 생성 과정은 $x_ T \rightarrow x_ {T-1} \rightarrow \cdots \rightarrow x_ 0$으로 나타낼 수 있다.

# 3 Diffusion Recommender Model

확산 모형의 강한 생성 능력의 이점을 활용하기 위해 그들은 새로운 **DiffRec 모형**을 제안한다. 이는 다음과 같은 특징을 가진다.

- 이는 **잡음이 섞인 상호 작용으로부터 사용자의 미래 상호 작용 확률을 예측**한다.
- 주어진 사용자의 과거의 상호 작용에 대해, DiffRec은 순방향 과정에서 잡음을 더해 점진적으로 이를 오염시킨다. 그 후, 역방향 과정에서 반복적으로 원래의 상호 작용을 복원하는 방법을 학습한다.
- 이러한 반복적인 잡음 제거 훈련에 의해, DiffRec은 **복잡한 상호 작용 생성 과정을 모델링할 수 있고 잡음이 섞인 상호 작용의 효과를 완화**할 수 있다.
- 결국, **복원된 상호 작용 확률은 상호 작용이 되지 않은 아이템(*non-interacted items*)에 대해 순위를 매기고 추천을 하는 데에 사용**된다.

추가적으로, 그들은 다음의 두 가지 작업을 위한 DiffRec의 확장을 제시한다. 

- 대규모 아이템 예측
- 시간적 모델링

이러한 확장은 실제 업무에서 DiffRec의 사용을 촉진한다.

## 3.1 Forward and Reverse Processes

![Untitled 2](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/482ce1ae-ac1c-402e-979c-d70b3c298523)

그림 2에서 볼 수 있듯, DiffRec은 두 가지 중요한 과정을 가진다.

1. 가우시안 잡음을 단계마다 더하면서 사용자의 상호 작용 과거를 오염시키는 순방향 과정
2. 점진적으로 잡음을 제거하는 방법을 학습하고 상호 작용 확률을 출력하는 역방향 과정

### Forward process

- 사용자 $u$
- 아이템 집합 $\mathcal I$에 대한 상호 작용 과거 $x_ u = [ x_ u^ 1, x_ u^ 2, \dots, x_ u^ {\vert \mathcal I \vert} ]$
    - $x_ u^ i=1\ or\ 0$은 사용자 $u$가 아이템 $i$와 상호 작용을 했는지, 아닌지를 나타낸다.
- $x_ 0 = x_ u$를 초기 상태로 두고 전이 확률을 다음과 같이 매개 변수화할 수 있다.

![Untitled 3](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/44ed4fc7-2431-4046-ba41-d2ee0eae8303)

- $\beta_ t \in (0, 1)$은 각 시간 단계 $t$에서 더해지는 가우시안 잡음의 정도를 조절한다.
- 재매개 변수화 기법(*reparameterization trick*)을 이용하면 $x_ 0$로부터 $x_ t$를 직접적으로 얻을 수 있다.

![Untitled 4](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/1a50342a-29ae-4004-a262-4f0ce5d453f7)

- $\alpha_ t = 1 - \beta_ t, \bar{ \alpha }_ t = \prod {\alpha_ t'}$
- $x_ {t} = \sqrt{ \bar{ \alpha } }_ t x_ 0 + \sqrt{ 1 - \bar \alpha_ t }\epsilon,\ \epsilon \sim N(0, I)$
- $x_ {1:T}$에서 더해지는 잡음을 제한하기 위해, 다음과 같이 선형 잡음 스케줄을 설계한다.

![Untitled 5](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/10a216a3-0a67-4085-ad3f-7ab0b7310456)

- 하이퍼 파라미터 $s \in [0, 1]$은 잡음의 정도를 조절한다.
- 하이퍼 파라미터 $\alpha_ {\min} < \alpha_ {\max} \in (0, 1)$은 더해지는 잡음의 상한과 하한을 지시한다.

### Reverse process

- $x_ T$에서부터 시작하여, 역방향 과정은 잡음 제거 전이 과정을 통해 점진적으로 사용자의 상호 작용을 복원한다.

![Untitled 6](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/233f6736-6bd6-4ef7-9306-dae3354806d3)

- $\mu_ \theta(x_ t, t), \Sigma_ \theta(x_ t, t)$는 학습 가능한 $\theta$를 포함하는 신경망에 의해 출력되는 가우시안 모수다.

## 3.2 DiffRec Training

$\theta$를 학습하기 위해, DiffRec은 관측된 사용자의 상호 작용 $x_ 0$의 ELBO를 최대화하는 것을 목표로 한다. 

![Untitled 7](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/68b49d76-ab61-4569-9b5d-2b07a6bc2f08)

- 식 (1)에 존재하는 사전 매칭 항은 상수이기 때문에 위 식에서 제거되었다.
- 복원 항은 $x_ 0$의 복원 확률을 측정한다.
- 잡음 제거 항은 역방향 과정에서 $t$가 $2$에서 $T$까지 변하는 동안 $x_ {t-1}$의 복원을 제한한다.
- 최적화는 복원 항과 잡음 제거 매칭 항을 최대화하는 방식으로 이루어진다.

### Estimation of denoising matching terms

잡음 제거 매칭 항은 KL 발산을 이용해 $p_ \theta(x_ {t-1} \vert x_ t)$가 계산 가능한 분포 $q( x_ {t-1} \vert x_ t, x_ 0)$와 근사적으로 같아지도록 한다. 베이즈 규칙에 의해, $q( x_ {t-1} \vert x_ t, x_ 0)$는 다음과 같이 닫힌 식으로 다시 쓸 수 있다.

![Untitled 8](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/7fd1a2b2-ac5f-4317-89fd-208992b8e539)

- $\tilde{ \mu } (x_ t,x_ 0, t)$와 $\sigma^ 2 (t) I$는 식 (2)와 (3)으로부터 유도된 $q( x_ {t-1} \vert x_ t, x_ 0)$의 평균과 공분산이다.
- 안정적인 훈련과 계산을 간단하게 만들기 위해 $\Sigma_ \theta (x_ t, t) = \sigma^ 2(t) I$로 직접적으로 설정한다.

시간 $t$에서 잡음 제거 매칭 항 $L_t$은 다음과 같이 계산된다.

![Untitled 9](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/58f6b950-9336-45ac-89ef-635119966892)

- 이는 $\mu_ \theta (x_ t, t)$가 $\tilde{ \mu } (x_ t,x_ 0,t )$와 같아지도록 한다.

식 (8)에 의해, $\mu_ \theta (x_ t, t)$를 다음과 같이 분해할 수 있다. 

![Untitled 10](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/09f42ed7-fa98-489d-abae-dc9db3fa6597)

- $\hat{ x }_ \theta(x_ t, t)$는 $x_ t, t$를 기반으로 예측한 $x_0$이다.

게다가, 식 (10)과 (8), (9)를 이용하면 다음을 얻을 수 있다.

![Untitled 11](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/bd9d8dbf-c799-43ce-9ffa-7c2adb597dec)

- 이는 $\hat{ x }_ \theta (x_ t, t)$가 $x_ 0$를 정확하게 예측하도록 한다.

요약하자면, 잡음 제거 매칭 항을 추정하기 위해 신경망으로 $\hat{ x }_ \theta (x_ t, t)$를 구현하고 식 (11)을 계산할 필요가 있다. 

### Estimation of the reconstruction term

$\mathcal {L_ 1}$을 식 (6)에 있는 복원 항의 음의 값이라고 정의하고, 이를 다음과 같이 계산할 수 있다.

![Untitled 12](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/de0d6854-a68a-423f-abd3-83bdc695438f)

- 가우시안 로그-가능도 $\log{ p(x_ 0 \vert x_ 1) }$를 $-\lVert \hat{ x }_ \theta (x_ 1,1 )-x_ 0 \rVert^ 2_ 2$로 추정한다.

### Optimization

식 (11)과 식 (12)에 의하면, 식 (6)에 있는 ELBO는 $\mathcal{ L }_ 1 - \sum_ {\mathcal L_ t}$로 쓸 수 있다. 그러므로, ELBO를 최대화하기 위해서 $\sum \mathcal{ L }_ t$를 최소화하는 것으로 $\hat{ x }_ {\theta} (x_ {t}, t)$ 안의 $\theta$를 최적화할 수 있다. 실제 구현에서는 균등하게 시간 단계 $t \sim U(1, T)$를 추출하여 기댓값 $\mathcal { L }(x_ 0, \theta)$를 최적화한다. 

![Untitled 13](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/711dc29e-6725-4ea9-9ef0-309a61cdf08e)

DiffRec의 훈련 과정은 알고리즘 1에 다음과 같이 제시되어 있다.

![Untitled 14](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/e1272155-da3f-4bf7-b0da-60b5cc1f5ea6)

### Importance sampling

최적화 문제는 다른 시간 단계마다 다양할 수 있다. 따라서 그들은 큰 손실 값 $\mathcal L _t$를 갖는 시간 단계에 대해 학습을 강조하기 위해 중요도 샘플링(*importance sampling*)을 고려한다. 형식적으로, 그들은 $t$에 대한 새로운 표본 추출 전략을 사용한다.

![Untitled 15](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/dc49a104-3cd7-4b7f-92f6-254664ae4804)

- $p_ t \propto \sqrt{ E[ \mathcal L_ {t^ 2}] }  / \sqrt{ \sum E[ \mathcal {L} _{t'}^ {2}] }$는 표본 추출 확률을 나타낸다.
- $\sum p_t = 1$

$E[ \mathcal {L}_ {t^ {2}}]$를 계산하기 위해, 훈련 중 열 개의 $\mathcal {L}_ {t}$를 모으고 평균을 취한다. 충분한 $\mathcal {L}_ {t}$를 얻기 전까지는 균등 표본 추출을 이용한다. 직관적으로, 큰 $\mathcal L _t$ 값을 갖는 시간 단계는 더 쉽게 추출될 것이다.

## 3.3 DiffRec Inference

이미지 합성 작업에서, 확산 모형은 임의의 가우시안 잡음을 추출하여 역방향 과정에 통과시킨다. 그러나, 상호 작용을 순수 잡음으로 오염시키는 건 추천에서 개인화된 사용자의 선호를 망가뜨린다. 따라서, 그들은 간단한 추론 전략을 제시한다. 이는 DiffRec의 훈련을 상호 작용 예측에 맞추어 조정한다. 

더 자세히 말하자면, DiffRec은 우선 순방향 과정에서 $x_0$을 $x_ 0 \rightarrow x_ 1 \rightarrow \cdots \rightarrow x_ {T'}$와 같은 과정을 통해 오염시킨다. 이후, $\hat x_ T = x_ {T'}$와 같이 설정하고 $\hat x_ T \rightarrow \hat x_ {T-1} \rightarrow \cdots \rightarrow \hat x_ 0$와 같은 역방향 잡음 제거 과정을 수행한다. 이러한 역방향 과정은 분산을 무시하고 결정론적 추론을 위해 $\hat x_ {t-1} = \mu_ \theta(\hat x_ t, t)$ 식 (10)을 통해 활용한다. 특히, 다음의 두 사항을 고려하여, 그들은 순방향 과정에서 $T' < T$와 같이 설정하여 더하는 잡음을 줄인다. 

1. 사용자의 상호 작용은 *false-positive*와 *false-negative* 상호 작용 때문에 자연스럽게 잡음이 섞여 있다.
2. 개인화된 정보를 함유하고 있다. 

마지막으로, 그들은 아이템의 순위를 정하고 가장 높은 순위의 아이템을 추천하기 위해 $\hat x_ 0$을 사용한다. 추론 과정은 다음과 같이 알고리즘 2에 요약이 되어 있다. 

![Untitled 16](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/6af592eb-91cf-4ffb-8213-4d5f40a7b3a8)

## 3.4 Discussion

그들은 DiffRec의 두 가지 특별한 포인트를 강조한다.

### Personalized recommendation

1. 훈련 중, DiffRec은 개인화된 정보를 보유하기 위해 사용자의 상호 작용을 순수한 잡음으로 오염시키지 않는다. 즉, 잠재 변수 $x_ T$는 광범위한 개인화된 특성을 잃는 표준 가우시안 잡음으로 접근하지 않는다. 
2. 그들은 추론을 위해 $T' < T$를 조절하는 것으로 더하는 잡음을 줄인다. 

### x0-ELBO

DiffRec은 $\epsilon$ 대신 $x_ 0$을 예측하는 것으로 최적화된다. 그 이유는 다음과 같다.

1. 추천의 목적이 아이템 순위를 정하기 위해 $\hat x_ 0$을 예측하는 것이고, 따라서 $x_ 0$-ELBO는 직관적으로 이 상황에서 더 적합하다. 
2. 임의로 추출한 $\epsilon \sim N(0, I)$은 불안정하고 MLP가 $\epsilon$을 추정하는 것을 더욱 어렵게 한다. 

## 3.5 Latent Diffusion

DiffRec과 같은 생성 모형은 모든 아이템에 대해 동시에 상호 작용 확률 $\hat{ x }_ 0$을 예측하는데, 이는 방대한 자원을 요구하고 따라서 대규모 아이템 예측과 같은 작업에서는 제한이 있다. 이러한 비용을 줄이기 위해 그들은 L-DiffRec을 제시한다. 이는 다중 VAE를 통해 아이템을 클러스터링하여 차원을 압축하고 잠재 공간에서 확산 과정을 수행한다. 

### Encoding for compression

- 주어진 아이템 세트 $\mathcal I$에 대해, L-DiffRec은 먼저 k-평균 알고리즘을 적용하여 아이템을 $C$개의 범주로 클러스터링한다. 이는 $[ \mathcal{ I }_ 1,\mathcal{ I }_ 2, \dots, \mathcal{ I }_ C]$와 같이 나타난다.
- L-DiffRec은 그 다음 사용자의 상호 작용 벡터 $x_ 0$를 클러스터에 따라 $C$개의 부분으로 나눈다. 이는 $x_ 0 \rightarrow [x_ 0^ c] ^ C_ {c=1}$와 같이 나타난다.
    - $x_ 0^ c$는 $\mathcal{ I }_ c$에 대해 사용자 $u$의 상호 작용을 나타낸다.
- 이후, $\phi_ c$로 매개 변수화된 변분 인코더를 사용하여 각 $x_ 0^ c$를 저-차원 벡터 $z_0 ^c$로 압축한다. 이때, 인코더는 $\mu_ {\phi_ c}$와 $\sigma^2_ {\phi_ c}I$를 변분 분포의 평균과 공분산으로써 예측한다.

클러스터링은 비용을 줄일 수 있다. 그 이유는 다음과 같다.

- 다른 범주에 대해 계산을 병행할 수 있다.
- 바닐라 VAE와 비교하여, 이는 여러 인코더 사이의 전체적인 연결을 끊어 모수의 수를 줄일 수 있다.

### Latent diffusion

$[ z_ 0^ c ]^ C_ {c=1}$을 합치는 것으로 압축된 $z_ 0$을 얻을 수 있고 이를 확산 모형에 활용할 수 있다. DiffRec를 훈련할 때처럼, $x_ 0$를 $z_ 0$로 대체하여 잠재 공간에서 순방향과 역방향 과정을 수행한다. 식 (13)과 유사하게, 이 경우에도 최적화 손실 함수는 $\mathcal L(z_ 0, \theta) = E_ {t \sim U(1, T)} \mathcal L_ t$와 같다. $\theta$는 잡음 제거 MLP의 모수를 나타낸다.

### Decoding

그림 3에서 나타나 있듯, 역방향 과정으로부터 복원된 $\hat {z}_ 0$을 아이템 클러스터에 따라 $[\hat {z}_ 0^ c ]^ C_ {c=1}$와 같이 분해한다. 각 $\hat {z}_ 0^ c$ 는 이후 $\psi_ c$로 매개 변수화된 디코더를 통과하여 $p_ {\psi_ c} ( \hat { x }_ 0^ c \vert \hat { z }_ 0^ c )$를 통해 $\hat x_ 0$을 예측한다. 

![Untitled 17](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/e47fa6a5-f9a0-4555-a24c-a28428d2bec8)

### Training

직관적으로, 인코더 $q_ {\phi_ c}$와 디코더 $p_ {\psi_ c}$는 결합적으로 VAE를 구축한다. 이는 상호 작용 공간과 잠재 공간을 연결한다. $\phi = [ \phi_ c ]_ {c=1}^ C$와 $\psi = [ \psi_ c ]_ {c=1}^ C$를 가진 VAE의 집합은 다음의 손실 함수를 통해 최적화될 수 있다.

![Untitled 18](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/fb80ff7c-f766-49f2-b998-6460bf209afd)

- $\gamma$는 KL 규제(*regularization*)의 강도를 조절한다.

이후, VAE와 확산 모형의 손실 함수를 조합하여 L-DiffRec의 최적화를 위한 손실 함수 $\mathcal L_ v (x_ 0, \phi, \psi) + \lambda \cdot \mathcal L(z_ 0, \theta)$를 얻을 수 있다. 하이퍼 파라미터 $\lambda$는 두 항이 동일한 영향력을 갖도록 보장한다. 

### Inference

1. 추론을 위해, L-DiffRec은 우선 $x_ 0$을 $x_ 0 \rightarrow [x_ 0^ c] ^ C_ {c=1}$로 분해한다. 
2. 각 $x_ 0^ c$를 분산을 고려하지 않은 결정론적 변수 $z_ 0^ c = \mu_ {\phi_ c}(x_ 0^ c)$로 압축한다. 
3. L-DiffRec은 DiffRec이 하는 것처럼 $[z_ 0^ c] ^ C_ {c=1}$를 $z_ 0$로 합친다.
4. 복원된 $\hat z_ 0$을 디코더에 넣는 것으로, 아이템의 순위를 정하고 추천을 생성하기 위한 $\hat { x }_ 0$을 얻을 수 있다.

## Temporal Diffusion

사용자의 선호도는 시간에 따라 바뀔 수 있기 때문에 DiffRec의 훈련 중에 시간적인 정보를 포착하는 것은 중요하다. 최근의 상호 작용이 사용자의 현재 선호도를 더 잘 반영한다고 가정한다면, 시간-인지(*time-aware*) *reweighting* 전략을 통해 더 나중의 사용자의 상호 작용에 더 많은 가중을 둘 수 있다. 

형식적으로는 다음과 같이 기술한다.

- $M$개의 상호 작용 아이템과 사용자 $u$에 대해, 상호 작용 시퀀스는 $S = \{i_ 1, \dots, i_ M \}$과 같이 표현할 수 있다.
    - $i_ m$은 $m$번째 상호 작용된 아이템의 ID를 나타낸다.
- 상호 작용된 아이템의 가중치를 $w = [ w_ 1, \dots, w_ M]$과 같이 정의한다.
- 시간-인지 선형 스케줄은 $w_ m = w_ {\min} + {m - 1 \over M - 1 }(w_ {\max} - w_ {\min})$과 같다.
    - 두 개의 하이퍼-파라미터 $w_ {\min} < w_ {\max} \in (0, 1]$은 상호 작용 가중의 하한과 상한을 나타낸다.
- 그 후, 사용자 $u$의 상호 작용 과거 $x_ 0$는 $\bar x_ 0 = x_ 0 \odot \bar w$와 같이 다시 가중치를 부여 받는다.
    - $\bar w \in R^ {\lvert I \rvert}$는 $w$에 의해 계산된 가중치 벡터이다. 즉, 다음과 같다.
        
        ![Untitled 19](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/d23dfaa2-53ff-4c2b-a2eb-831bf7eb31f8)

    - Idx(i)는 사용자 $u$의 상호 작용 시퀀스 $S$ 내에서 아이템 $i$의 인덱스를 나타낸다.
- 다시 가중된 상호 작용 과거 $\bar x_ 0$을 DiffRec과 L-DiffRec에 넣는 것으로 T-DiffRec과 LT-DiffRec을 얻을 수 있다. 이들은 상대적으로 시간적 정보를 사용한다.

# 4. Experiments

그들은 다음과 같은 질문에 답하기 위해 세 가지 실제 데이터를 통해 실험을 수행하였다. 

- 연구 질문 1: DiffRec은 다양한 실험 설정에서 기존 모델과 비교했을 때 얼마나 잘 작동하는가? 또한, DiffRec의 설계가 어떻게 성능에 영향을 미치는가?
- 연구 질문 2: L-DiffRec은 추천 정확도와 비용에 대해 어떻게 작동하는가?
- 연구 질문 3: T-DiffRec은 훈련 중에 상호 작용 타임 스탬프를 이용할 수 있을 때, 순차적 추천 모형(*sequential recommender models*)을 능가할 수 있는가?

## 4.1 Experimental Settings

### Datasets

그들은 다음과 같은 세 개의 공개되어 있는 데이터에 대해 실험을 수행했다.

- **Amazon-book**: 아마존 리뷰 데이터에서 온 이 데이터는 많은 책에 대한 풍부한 사용자의 상호 작용을 제공한다.
- **Yelp**: 대표적인 비즈니스 데이터로, 다른 식당에 대한 사용자의 리뷰를 포함한다.
- **ML-1M**: 영화에 대한 사용자의 평가를 포함하는 벤치마크 데이터이다.

### Training settings

모든 데이터에 대해, 그들은 세 가지 다른 훈련 설정을 고려했다.

1. **Clean training**: 평가가 4번 미만인 사용자의 상호 작용은 버린다.
2. **Noisy training**: *clean training*과 같은 설정에서, 테스트 데이터 세트에 잡음이 섞인 상호 작용을 더한다. 
3. **Temporal training**: 훈련에 추가적으로 타임 스탬프를 고려한다. 즉, 순차적 추천 모형과 같이 사용자의 상호 작용 시퀀스를 모델링한다. 이는 시간적 모델링의 효과를 평가하기 위함이다. 

### Table 1

![Untitled](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/ac33fc4b-f66c-4814-88ed-41341899cfb8)

테이블 1은 두 가지 세 가지 데이터에 대한 통계량을 나타낸다.

- **C**, **N**은 각각 *Clean training*과 *natural noise training*을 나타낸다.
- **Int**는 상호 작용(*interactions*)을 나타낸다.

### Baselines

그들은 DiffRec을 뛰어난 성능을 보이는 기존 모형과 비교하였다. 

- **MF**: 행렬 분해(*matrix factorization*)를 기반으로 하는 대표적인 협업 필터링 모형.
- **LightGCN**: 그래프 합성 네트워크(GCN)의 *linear neighborhood aggregation*을 통해 사용자와 아이템의 표현을 학습한다.
- **CDAE**: 오염된 상호 작용으로부터 원래의 상호 작용을 복원하기 위해 오토인코더를 훈련한다.
- **MultiDAE**: 드롭 아웃을 이용해 상호 작용을 오염시키고 오토인코더를 통해 이를 복원한다.
- **MultiDAE++:** DiffRec과 유사한 방식으로 상호 작용을 오염시키고 multiDAE를 통해 상호 작용을 복원한다. DiffRec과 다른 점은, 잡음 제거(*denoise*)를 한 번에 한다는 점이다.
- **MultiVAE**: 상호 작용 생성 과정을 모델링하기 위해 VAE를 활용한다.
- **CODIGEM**: 확산 과정을 이용하는 생성 모형이다. 역 생성 과정을 모델링하기 위해 많은 AE를 활용하지만 오로지 하나의 AE만을 이용하여 상호 작용을 예측한다.

### Evaluation

각 사용자에 대해, 상호 작용이 되지 않은(*non-interacted*) 모든 아이템에 대해 순위를 정하는 *full-ranking protocol*을 따라 평가한다. 성능 비교를 위해 자주 쓰이는 두 가지의 지표 Recall@K와 NDCG@K를 이용한다. K는 10 혹은 20으로 설정한다. 

## 4.2 Analysis of DiffRec (RQ1)

### Table 2

![Untitled 1](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/f23fef66-2158-44c6-8bdc-25a853a3df92)

*Clean training*에 대해, 기존 모형과 DiffRec의 전체적인 성능을 나타낸 테이블이다.

- 최고의 결과는 볼드체로, 두 번째로 좋은 결과는 밑줄로 표시하였다.
- **%Improve**는 기존 모형과 비교하였을 때 DiffRec의 상대적인 성능 개선을 나타낸다.
- *는 단일 표본 t 검정($p <0.05$) 아래, 기존 모형과 비교했을 때 통계적으로 유의한 성능 향상을 보였다는 사실을 암시한다.

### Clean training

- 대부분의 생성 모형은 MF나 LightGCN보다 좋은 성능을 보였다. 그 이유는 생성 모형과 실제 세상의 상호 작용의 생성 과정이 일치하기 때문일 수 있다. 특히, MultiVAE가 좋은 성능을 보였는데, 이는 변분 추론(*variational inference*)과 다항 가능도(*multinomial likelihood*)의 활용이 더 강한 생성 모델링으로 이어졌기 때문이라 생각할 수 있다.
- 모든 경우에서 MultiDAE++는 MultiDAE보다 성능이 좋았다. CODIGEM은 LightGCN보다도 성능이 떨어졌다. 이는 CODIGEM이 하나의 AE만을 추론에 이용했기 때문이라고 생각할 수 있다. 이런 식의 접근은 본질적으로 MultiDAE를 적은 잡음으로 훈련하는 것과 같다.
- **DiffRec은 세 개의 데이터 모두에 대해 유의한 성능 향상**을 보였다. 이유는 다음과 같이 생각해 볼 수 있다.
    - DiffRec은 잡음 제거 과정을 점진적으로 학습하는 것으로 더 복잡한 분포를 모델링할 수 있다.
    - DiffRec은 간단한 순방향 과정을 채용하여 VAE 기반 모형이 갖는 상충 관계를 완화한다.
    - 순방향 과정에서 쓰이는 잡음 스케줄이 개인화 선호 모델링을 보장한다.

### Table 3

![Untitled 2](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/74d9ce45-701f-4035-a38e-5e236964705a)

DiffRec과 가장 성능이 좋은 기존 생성 모형 MultiVAE, 그리고 성능이 가장 좋으면서 생성 모형이 아닌 기존 모형을 *noisy training with natrual noises* 설정 아래 비교하여 그 성능을 나타낸 테이블이다. 

### Figure 4

![Untitled 3](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/5a8375a1-b52d-4be0-b67b-7e438c2aa8e7)

Amazon-book 데이터에 대해 *noisy training with random noises* 설정 아래 비교한 성능을 나타낸 그림이다. 

### Noisy training

실제 추천에서는 상호 작용이 거짓-양성과 거짓-음성 아이템을 포함한다. 이러한 상황에서 DiffRec의 성능을 분석하기 위해 두 가지 설정 아래 실험을 수행하였다.

1. **natural noises**: 거짓-양성 상호 작용을 양성으로 하여 임의로 훈련 데이터와 검증 데이터에 추가한다. 테이블 3에 실험 결과를 나타냈다. 
2. **random noises**: 상호 작용이 되지 않은 아이템을 양성 상호 작용으로 추가한다. 그림 4에 잡음을 10%에서부터 50%까지 증가시키면서 비교한 결과를 나타냈다. 

테이블 3에서 **DiffRec이 MultiVAE와 LightGCN을 능가**한다는 사실을 관찰할 수 있다. 이는 DiffRec이 *natural noises*에 강건하다는 사실을 나타낸다. 이는 합리적인데, 왜냐하면 거짓-양성 상호 작용은 본질적으로 오염된 상호 작용이기 때문이다. DiffRec은 그러한 오염에서부터 깨끗한 상호 작용을 복원하기 위해 최적화된다. 

그림 4의 결과는 다음과 같이 요약할 수 있다.

1. 10%에서부터 50%까지 잡음을 증가시키면서 세 모형의 성능은 점진적으로 감소한다.
2. 그럼에도 불구하고, DiffRec은 다른 두 모형을 능가한다. DiffRec은 각 단계에서 서로 다른 크기의 잡음으로 훈련되기 때문에 크게 손상된 상호 작용으로부터도 깨끗한 상호 작용을 잘 복원할 수 있다. 

### Figure 5

![Untitled 4](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/31130f2c-2503-435f-9453-b762ef6a093f)

중요도 표본 추출(*importance sampling*)과 균등 표본 추출(*uniform sampling*)의 효과를 나타낸 그림이다. 

### Figure 6

![Untitled 5](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/fc320285-cd66-4221-b40f-be7cb191c3e0)

잡음 크기 $s$와 확산 단계 $T$에 대한 효과를 나타낸 그림이다. 

### Table 5

![Untitled 6](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/869eec07-97c6-4a3d-a6c9-5a9f9a285129)

$\epsilon$-ELBO의 성능을 나타낸 테이블이다. 

### In-depth Analysis

DiffRec의 설계를 분석하기 위해 그들은 추가적인 실험을 수행하였다. 결과를 다음과 같이 간단하게 요약할 수 있다.

- **Importance sampling**: 그림 5의 (a)에서 중요도 표본 추출과 균등 표본 추출의 비교 결과를 확인할 수 있다. 오른쪽의 균등 표본 추출은 성능이 보다 떨어지는데, 이는 중요도 표본 추출의 효과를 입증한다.
- **Effect of inference step**: 추론 단계에서 $T'$를 변화시킨 결과를 그림 5의 (b)에 나타냈다. $T'=0$이 가장 나은 성능을 보였다. $T'=T/4$에서부터는 결과를 비교하기가 어려운데, 이는 잡음의 크기가 상대적으로 작기 때문이다.
- **Effects of noise scale and step**: 두 가지 중요한 하이퍼-파라미터에 대해 분석한 결과를 그림 (6)에 나타내었다.
    - 잡음의 크기가 증가하면서 처음에는 성능이 증가하지만 특정 포인트에서 다시 성능이 감소한다. 따라서, 상대적으로 작은 크기의 잡음을 선택할 필요가 있다.
    - 성능이 $T$에 따라 요동치는데, 이는 확산 단계의 증가가 정확도에 적은 영향을 끼친다는 사실을 나타낸다. 너무 큰 $T$는 자원을 크게 요구하기 때문에, 그들은 $T=5$를 선택하였다.
- **$x_0$-ELBO vs. $\epsilon$-ELBO**: 테이블 5에서 $x_0$와 $\epsilon$를 예측을 비교한 결과를 나타냈다. $\epsilon$-ELBO가 더 성능이 좋지 않은데, 그 이유를 MLP를 통해 임의의 잡음을 예측하는 게 어렵기 때문이라고 생각할 수 있다.

## 4.3 Analysis of L-DiffRec (RQ2)

### Table 6

![Untitled 7](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/ab868961-03b2-4a9a-81ca-684472d0b678)

L-DiffRec, DiffRec과 MultiVAE의 성능을 clean training 설정 아래 비교한 것이다. 

- **par**는 모수(*parameter*)의 수를 나타낸다.

### Clean training

결과는 테이블 6에서 확인할 수 있다.

- L-DiffRec은 더 적은 자원으로 MultiVAE보다 우수한 성능을 보였다.
- 동시에, L-DiffRec은 정확도를 유지하면서 DiffRec의 비용을 크게 낮추었다.

즉, **L-DiffRec은 기존의 모델과 경쟁할 수 있는 정확도를 유지하면서 소모한 비용은 훨씬 적었다**. 이러한 정확도의 유지는 클러스터링을 통한 차원 축소가 중복된 정보는 줄이면서 중요한 정보는 잃지 않기 때문이라고 생각할 수 있다.

또한, 이러한 소모 비용의 절감은 다음과 같은 이유에서 온다고 생각할 수 있다.

1. 아이템 클러스터링이 인코더와 디코더의 모수를 감소시킨다.
2. 잠재 확산 과정이 denoising MLP의 모수의 수를 줄인다.

이러한 결과가 말하는 건, **L-DiffRec이 산업 현장에서 대규모 아이템 예측을 가능케 할 수 있다는 것**이다.

### Table 4

![Untitled 8](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/a82dc950-5e74-4409-b90c-6228fa2d58c9)

세 개의 데이터에 대해, *natural noise training* 설정 아래 L-DiffRec과 DiffRec을 비교하여 그 성능을 나타낸 테이블이다. 

### Noisy training

요구되는 자원은 *noisy training*과 *clean training*이 같은 반면, L-DiffRec이 DiffRec보다 *noisy training*에서 더 나은 성능을 보인다. 테이블 4에서 이 결과를 확인할 수 있다. 이러한 현상이 관찰되는 한 가지 가능한 이유를 클러스터링된 범주가 거짓-양성 상호 작용일 가능성이 있는 상호 작용을 더 적게 갖고 있기 때문이라고 생각할 수 있다. 아이템 클러스터링을 통한 표현 압축은 이러한 잡음의 효과를 약하게 한다. 

### Figure 7

![Untitled 9](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/251ab59c-2111-428c-a360-e9c37345d05e)

L-DiffRec의 클러스터링 범주의 수의 효과를 *clean training* 설정 아래 비교하여 나타낸 그림이다.

### Effect of category number

그들은 L-DiffRec에서 범주의 수의 효과를 조사하기 위해 클러스터링 범주의 수를 1에서부터 5까지 변경하면서 실험을 수행하였다. 결과는 그림 7에서 확인할 수 있고, 다음과 같이 요약할 수 있다.

1. 그림 7의 (a)와 (b)에서 볼 수 있듯, Recall, NDCG, GPU 사용량과 모수는 범주의 수에 증가함에 따라 감소한다. 
2. 범주의 수가 설령 1이더라도 L-DiffRec의 비용은 다른 두 모형과 비교했을 때 매우 적다.
3. L-DiffRec은 C가 적을 때 다른 두 모형보다 떨어지지 않는 성능을 보인다. 즉, **범주의 수를 잘 조정하면 적은 비용으로도 뛰어난 성능을 갖는 모형을 기대**할 수 있다. 

## 4.4 Analysis of T-DiffRec (RQ3)

T-DiffRec의 시간적 모델링에서의 효과를 증명하기 위해 T-DiffRec과 LT-DiffRec, 그리고 SOTA 순차적 추천 모형인 ACVAE를 비교하였다. 그 결과는 테이블 7에서 확인할 수 있고, 다음과 같이 요약할 수 있다.

1. **T-DiffRec과 LT-DIffRec은 시간적 모델링 상황에서 DiffRec과 L-DiffRec보다 나은 성능**을 보인다.
2. 또한, 현재 SOTA 모델인 ACVAE보다 T-DiffRec과 LT-DiffRec은 더 정확하고 강건한 성능을 보인다. 
3. 또한, 더 많은 모수에도 불구하고 DiffRec 기반 기법들은 ACVAE보다 더 적은 GPU 메모리를 사용하였다. 즉, DiffRec 기반 기법들은 ACVAE보다 더 적은 자원을 요구한다. 
4. **LT-DiffRec은 T-DiffRec보다 더 적은 모수를 갖고 있음에도 우수한 성능**을 보였다. 

### Table 7

![Untitled 10](https://github.com/Won-Seong/Review-Diffusion-Recommender-Model/assets/54873618/20f52d17-5e36-4c2b-8029-9361919d1601)

DiffRec과 SOTA를 달성한 순차적 기존 모형 ACVAE를 비교한 성능을 나타낸 테이블이다. 각 모형을 타임 스탬프를 이용하여 훈련하였다. 

- **par**는 모수의 수를 나타낸다.

# 5. Related Work

### Generative recommendation

생성 모형은 아이템 사이 협업 신호(*collaborative signals*)를 잘 학습하는데, 이는 모든 아이템에 대한 예측을 동시에 모델링하기 때문이다. 게다가, 생성 모형은 사용자의 선호와 상호 작용 사이의 복잡한 관계를 잘 포착한다. 

기존의 생성 추천 모형은 크게 두 가지로 나눌 수 있다.

1. GAN 기반 기법: 적대적 훈련(*adversarial training*)을 활용하여 생성기를 최적화하고 사용자의 상호 작용을 예측한다.
2. VAE 기반 기법: 사후 분포를 추정하여 인코더와 디코더를 학습하고 모든 아이템에 대해 상호 작용 확률을 예측한다. 

기존의 생성 추천 모형 역시 성공적이었으나, 확산 모형은 그보다 더한 장점을 갖는다. 예를 들면, 확산 모형은 더 안정적이면서 다양한 작업에 대해 높은 품질을 보장한다. 

### Diffusion models

확산 모형은 조건부 생성과 무-조건부 생성 모두에서 높은 품질의 생성 능력을 보유하고 있다. 그러나 이러한 확산 모형의 성공에도 불구하고, 이를 추천 모형에 활용하는 데에는 약간의 조사가 필요했다. 예를 들어, CODIGEM은 확산 모형을 이용해 추천을 생성했다고 주장하였지만 이는 본질적으로 MultiDAE 기법과 크게 다르지 않았고, 저조한 성능을 보였다. 특히 CODIGEM은 각 단계에서의 예측을 위해 많은 AE를 활용하지만 정작 상호 작용 확률을 추정할 때에는 하나의 AE만을 사용하였고, 나머지 AE는 쓸모가 없었다. DiffRec은 이런 점에서 다른데, 이 모형은 다중-단계 예측을 위해 공유된 MLP를 이용하고 추론을 위해 다중-단계의 잡음 제거를 고려한다. 

# 6. Conclusion and Future Work

## Conclusion

- 이 논문에서 그들은 DiffRec이라는 모형을 제시하였다. 이는 생성 추천 모형에 대한 새로운 추천 패러다임이다.
- 개인화 추천을 보장하기 위해, 그들은 순방향 과정에서 잡음의 크기와 추론 단계를 줄여 사용자의 상호 작용을 오염시켰다.
- 그들은 또한 전통적인 생성 모형을 두 가지 확장을 통해 확장하였다.
    - 첫 번째 확장은 L-DiffRec에 관한 것으로, 이는 차원 압축을 위해 아이템을 클러스터링하고 잠재 공간에서 확산 과정을 수행한다. 결과적으로 이는 대규모 아이템 예측을 상대적으로 저렴한 비용으로 수행할 수 있다.
    - 두 번째 확장은 T-DiffRec에 관한 것으로, 이는 시간-인지 *reweighting* 전략을 활용하여 사용자의 상호 작용에서 시간적 패턴을 포착한다. 결과적으로 이는 상호 작용 시퀀스의 시간적 모델링을 할 수 있게 한다.

## Future work

- 이 연구는 생성 추천 모형에 대한 새로운 연구 방향을 제시한다.
    - L-DiffRec과 T-DiffRec은 간단하고 효과적이다. 그렇지만 더 나은 압축과 시간 정보를 인코딩하기 위해 더 나은 전략을 고안할 수 있다.
    - DiffRec에 기반한 조건부 추천을 연구하는 것 역시 의미가 있다.
    - 더 나은 사전 가정, 예를 들어 가우시안 잡음 외의 다른 잡음과 같은 가정과 다양한 모형의 효과를 탐구하는 것 역시 흥미로운 일이다.

# 7. My Reivew
 
먼저, 추천 모형의 목적을 확산 모형과 나란히 놓은 저자들의 주장이 흥미롭다. 그들의 말에 따르면, 추천 모형은 본질적으로 잡음이 섞인 과거의 상호 작용을 통해 미래의 상호 작용 확률을 예측한다. 비록 완전히 일치하지는 않지만, 확산 모형을 떠올리기에 충분한 설명이다. 그들의 말대로 현재 상승 가도를 달리는 확산 모형을 추천 모형에 통합하고자 하는 시도는 자연스럽고, 그들은 그 논리와 아이디어를 잘 발전시키지 않았나 싶다.

다만, 추천 모형이 아직 익숙하지 않은 나에게 있어 이러한 접근이 많이 직관적이지는 않다. 잘 아는 건 아니지만, 확산 모형의 본질은 결국 조금씩 잡음을 제거하는 그 과정에 있다고 생각한다. Gaussian에서 한 번에 데이터 분포를 포착하라는, 어찌 보면 가혹한 일을 우리는 VAE에게 시켰었고, 확산 모형에게는 이를 조금씩 나누어서 하라고 시켰다. 확산 모형의 구조는 우리에게 많은 혜택을 갖다 주었고, 결과적으로 이렇게 조금씩 잡음을 제거하여 데이터를 추출하는 작업은 그리 무리한 게 아니게 되었다. 확산 모형과 스코어 기반 모형과의 연결은 우리에게 더 많은 직관을 갖다 주기도 하였다. 

확산 모형의 입장에서 보면 할 수 있는 건 과거의 데이터를 통해 사용자의 상호 작용을 생성하는 것이다. 잡음을 더하고 제거하는 과정이 실제 세계의 상호 작용 생성 과정과 일치하는지는 잘 모르겠으니까 넘어가자. 어떻게 모든 아이템에 대해 특정 사용자의 상호 작용을 잘 생성할 수 있는 건지 잘 모르겠다. 일견 그래프 문제처럼 굉장히 복잡해 보이는데... 왜 그들은 조건부 생성 모형을 쓰지 않았을까? 나에게는 사용자의 특성을 조건으로 주고 그에 따라 아이템에 대한 상호 작용을 생성하는 모형이 조금은 더 자연스러워 보인다.

물론, 나는 추천 모형에 익숙하지 않기 때문에 이러한 의견이 그리 말이 되지 않을 수도 있다. 아무튼, 내 지식으로는 확산 모형과 추천 모형을 결합하는 게 아직은 그렇게 직관적이지는 않다. 이를 더 잘 이해하려면 추천 모형을 통계적 관점에서 조금 더 공부를 해 보아야 할 듯하다.

---

# Author Information

- Name : SeongWon Kim(김성원)
- Master student in StatHT, Graduate School of Data Science, KAIST
- Email : ksw888@kaist.ac.kr
- github : https://github.com/Won-Seong/
