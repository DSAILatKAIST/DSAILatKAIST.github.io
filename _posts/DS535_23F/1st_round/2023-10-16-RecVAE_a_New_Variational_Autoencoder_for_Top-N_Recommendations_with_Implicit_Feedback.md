---
title:  "[WSDM 2020] RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback"
permalink: 2023-10-16-RecVAE_a_New_Variational_Autoencoder_for_Top-N_Recommendations_with_Implicit_Feedback.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback** 

## **1. Introduction**  

행렬 분해 (Matrix Factorization)은 협업 필터링 (Collaborative Filtering)을 기반으로 한 추천시스템의 기본적인 방법 중 하나이다. 하지만 이는 다음과 같은 문제점을 가지고 있다. 

- 행렬의 크기가 사용자 (User)와 항목 (Item)의 수에 선형적으로 비례하기 때문에 매우 많은 파라미터를 필요로 한다.
- 콜드 스타트 (Cold Start) 문제가 발생한다. 즉, 새로운 사용자나 항목이 추가될 때, 정확한 추천을 하기가 어렵다.
- 몇몇 사용자나 항목에 대해서 주어진 데이터가 매우 적을 수 있다. 이는 과적합을 초래할 수 있으며 일반적으로 강력한 정규화가 필요하다.

최근에 이러한 극복하기 위해 오토인코더 기반의 접근이 지속적으로 연구되고 있다. Collaborative Denoising Autoencoder (CDAE)[^1]는 사용자 피드백을 임베딩 (Embedding)하여 콜드 스타트 문제를 해결하였다. Variational Autoencoder (Mult-VAE)[^2]는 적절한 우도 (Likelihood)를 도입하여 향상된 결과를 보여주었다. 이 연구에서는 Mult-VAE의 확장으로서 암시적 피드백 (Implicit Feedback)을 활용하는 Recommender VAE (RecVAE) 를 제안하고 이것의 기여는 아래와 같다.

> - 사용자 임베딩을 향상시키는 인코더 네트워크 설계 제안
> - 추천 시스템에 알맞는 적절한 사전분포 제안
> - 암시적 피드백으로 인한 새로운 $\beta$-VAE 제안 

## **2. Preliminary**  

### **2.1 Variational autoencoders and their extensions**

변분 오토인코더는 복잡한 분포를 학습할 수 있는 잠재 변수 모델이다. 이에 대한 간략한 요약으로 시작하자. 주어진 데이터가 $p_ {true}(x)$를 따르고 모델을 $p_ {\theta}(x)$라고 하자. 잠재변수 $z$를 통해서 모델을 다시 표현 할 수 있다.

> $p_ {\theta}(x) = \int p_ {\theta}(x \vert z)p(z)dz$

이 적분을 계산하는 것은 어려우므로 이것의 하계 (Lower Bound)를 최대화 하는 방법으로 모델이 훈련된다. 이를 ELBO (Evidence Lower Bound)라고 부르며 다음과 같다. 

>$p_ {\theta}(x) \geq \mathcal{L}_ {\text{VAE}} = \mathbb{E}_ {q_ {\phi}(z \vert x)} \left[ \log p_ {\theta}(x \vert z) - \text{KL}({q_ {\phi}(z \vert x)} \parallel p(z) )\right],$

여기서 $\text{KL}$은 KL-divergence를 의미하고 $p(z), q_ {\phi}(z \vert x)$는 각각 사전분포와 변분 분포를 의미한다. $p_ {\theta}(z,x)$를 학습함으로써 변분 오토인코더는 생성모델로서 활용될 수 있다. 또한, $q_ {\phi}(z \vert x)$를 이용한다면 임베딩을 통한 재표현 (Representation)을 얻을 수 있다. $\beta$-VAE[^3]는 더욱 향상된 재표현을 얻기 위해 다음과 같이 변형된 손실 함수를 제안 하였다.

>$\mathcal{L}_ {\beta\text{-VAE}} = \mathbb{E}_ {q_ {\phi}(z \vert x)} \left[ \log p_ {\theta}(x \vert z) - \beta \text{KL}({q_ {\phi}(z \vert x)} \parallel p(z) )\right]$

Denoising variational autoencoders (DVAE)[^4]는 데이터에 노이즈를 강제로 주입하여 재표현을 학습하기 위한 방법이다. 노이즈 분포 $p(\tilde{x} \vert x)$에 대하여 (예를 들어, 가우시안 혹은 베르누이 분포) 변형된 손실 함수를 정의한다.

>$\mathcal{L}_ {\text{DVAE}} = \mathbb{E}_ {q_ {\phi}(z \vert x)} \mathbb{E}_ {p(\tilde{x} \vert x)} \left[ \log p_ {\theta}(x \vert z) - \text{KL}({q_ {\phi}(z \vert \tilde{x})} \parallel p(z) )\right]$,

그 결과로서 노이즈가 있는 데이터에 대해서도 의미있는 재표현을 얻을 수 있다. Conditional Variational Autoencoder (CVAE)[^5]는 변분 오토인코더의 또 다른 확장이다. 주어진 데이터가 $x$ 뿐만 아니라 $y$라는 레이블이 있다면 이것에 따른 조건부 확률 분포를 학습 할 수 있다. 

>$\mathcal{L}_ {\text{CVAE}} = \mathbb{E}_ {q_ {\phi}(z \vert x, y)}\left[ \log p_ {\theta}(x \vert z, y) - \text{KL}({q_ {\phi}(z \vert x, y)} \parallel p(z \vert y) )\right]$

마지막으로, VAE with Arbitrary Conditioning (VAEAC)[^6]는 결측값 예측을 위해 사용하는 모델이며 협업 필터링 문제와 상당히 비슷하다. 주어진 데이터 $x$에 대해서 결측된 특성을 $x_ {b}$ 나머지를 $x_ {1-b}$ 라고 하자. CVAE에서 $y$ 대신 $(x_ {1-b}, b)$를 사용하면 VAEAC의 손실 함수를 정의할 수 있다.

>$\mathcal{L}_ {\text{VAEAC}} = \mathbb{E}_ {q_ {\phi}(z \vert x, b)}\left[ \log p_ {\theta}(x_ {b} \vert z, x_ {1-b}, b) - \text{KL}({q_ {\phi}(z \vert x, b)} \parallel p(z \vert x_ {1-b}, b) )\right],$

여기서 $b$는 이진 마스킹 (Binary Masking)을 의미한다. 

### **2.2 Autoencoders and Regularization for Collaborative Filtering**

$U$, $I$를 유저와 항목의 집합으로 표기하고 $X$를 암시적 피드백 행렬이라고 하자. 즉, $x_ {ui} = 1$ 인 필요충분조건은 유저 $u$가 항목 $i$를 긍정적으로 작용했다는 것이다. $x_ {u}$를 피드백 벡터라고 하자. CDAE[^1]는 $x_ {u}$에 마스킹을 적용해서 복구하는 모델이므로 2.1 섹션의 DVAE를 협업 필터링에 적용한 것으로 볼 수 있다.

Mult-VAE[^2]는 협업 필터링에 적용하기 위해서 우도를 다항 분포로 가정한 변분 오토인코더 모델이다. $n_ {u}:= \sum_{j} (x_ {u})_ {j}$ 라고 하면 모델은 다음과 같이 정의된다.

> - $z_ {u} \sim N(0,I_ {k \times k})$
> - $f_ {\theta}: \mathbb{R}^{k} \rightarrow \mathbb{R}^{\vert I \vert}$ is a neural network.
> - $\pi(z_ {u}) \sim \text{softmax}(f_ {\theta}(z_{u}))$ 
> - $x_ {u} \sim \text{Multinomial}(n_ {u}, \pi(z_ {u}))$
> - (Objective) $\mathcal{L}_ {\text{Mult-VAE}} = \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})} \left[ \log p_ {\theta}(x_ {u} \vert z_ {u}) - \beta \text{KL}({q_ {\phi}(z_{u} \vert x_ {u})} \parallel p(z_ {u}) )\right]$

## **3. Method** 

기본적으로, 제안된 모델 RecVAE는 Mult-VAE의 확장이다. DAE 방법을 추가하여 생성모델을 정의한다.

 > - $p_ {\theta}(x_ {u} \vert z_ {u}) = \text{Multinomial}(x \vert n_ {u}, \pi(z_ {u}))$
 > - $\pi(z_ {u}) = \text{softmax}(f_ {\theta}(z_ {u}))$
 > - $f_{\theta}(z_ {u})$ is a neural network.
 > - $q_ {\phi}(z_ {u} \vert x_ {u}) = N(z_ {u} \vert \psi_ {\phi}(x_ {u}))$
 > - (Objective) $\mathcal{L} = \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})} \mathbb{E}_ {p(\tilde{x}_ {u} \vert x_ {u})}\left[ \log p_ {\theta}(x_ {u} \vert z_ {u}) - \beta \text{KL}({q_ {\phi}(z_ {u} \vert \tilde{x}_ {u})} \parallel p(z_ {u}) )\right]$

### **3.1 Model Architecture**

<p align="center">
<img src="https://i.ibb.co/xJQrsLz/2023-10-14-164239.png" width="50%" height="50%">
</p>

첫번째 변화는 dense CNNs[^7], swish activation functions[^8], layer normalization[^9]과 같은 아이디어를 결합해 협업 필터링에 알맞는 추론 네트워크를 제안하며 위 그림과 같은 구조를 가지고 있다.

### **3.2 Composite prior**

<p align="center">
<img src="https://i.ibb.co/tJDwC9P/2023-10-14-205240.png" width="50%" height="50%">
</p>

Mult-VAE 구조에서 데이터의 희소성 (Sparsity) 때문에 변분 분포의 파라미터 최적화가 어려움을 겪을 수 있다. 이는 강화학습에서 forgetting 효과라고 알려져 있으며 정책 기반 강화학습 논문에 많은 논의가 있었다[^10]. 이를 해결 하기 위한 방법중 하나는 학습된 파라미터를 기억해두는 방법이다. 즉, 새로운 파라미터를 찾는 학습은 좋은 결과를 주는 파라미터로 부터 크게 벗어나지 않게 정규화를 주어야 한다.

이는 오토인코더에 구조에서 $q_ {\phi}(z \vert x)$를 업데이트 할 때, 이전에 얻었던 $q_ {\phi_ {\text{old}}}(z \vert x)$을 적당히 유지하고 싶은 것과 같다 . 이를 수행할 수 있는 한 방법은 본래의 사전분포와 $q_ {\phi_ {\text{old}}}(z \vert x)$의 컨벡스 결합을 새로운 사전분포로 사용하는 것이다. 즉,

> $p(z \vert \phi_ {\text{old}},x) = \alpha N(z \vert 0,I) + (1-\alpha)q_ {\phi_ \text{old}}(z \vert x)$

최종적인 모델 설계는 위 사진과 같다.

### **3.3 Rescaling KL divergence**

$\beta$-VAE[^3]은 재표현을 학습하기 위한 좋은 방법이지만 파라미터를 선택하는 방법이 학습에 큰 영향을 미친다. 기존의 연구[^11]와는 다르게 협업 필터링 변분 오토인코더 모델에 알맞는 $\beta$ 선택 방법에 대한 연구가 필요하다. 

유저 피드백이 부분적으로 주어졌다고 하자. 부분적인 데이터에 대해서 $X_ {u}^{0}$를 유저 $u$가 긍정적으로 평가한 항목의 집합이라 하고 $X_ {u}^{f}$ 긍정적으로 평가한 모든 항목의 집합이라고 하자. 항목들이 원 핫 인코딩으로 주어졌다고 하고, 다음과 같이 기호를 적자.

- $x_ {u} = \sum_ {a \in X_ {u}^{0}}1_ {a}$
- $x_ {u}^{f} = \sum_ {a \in X_ {u}^{f}}1_ {a}$
- $\text{KL}_ {u} = \text{KL}(q_ {\phi}(z_ {u} \vert x_ {u}) \parallel p(z_ {u}))$ 
- $\text{KL}_ {u}^{f} = \text{KL}(q_ {\phi}(z_ {u} \vert x_ {u}^{f}) \parallel p(z_ {u}))$ 

여기서 $1_ {a}$는 항목 $a$에 대응되는 원 핫 인코딩된 벡터이다. 기존의 ELBO를 다음과 같이 정리 할 수 있다.

> $\mathcal{L} = \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u}^{f})}\left[ \log \text{Multinomial}(x_ {u}^{f} \vert \pi(z_ {u})) - \text{KL}_ {u}^{f}\right]$
> $= \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u}^{f})} \left[ \sum_ {a \in X_ {u}^{f}} \log \text{Cat}(1_ {a} \vert \pi(z_ {u})) - \text{KL}_ {u}^{f}\right] + C$
> $= \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u}^{f})}  \sum_ {a \in X_ {u}^{f}} \left[  \log \text{Cat}(1_ {a} \vert \pi(z_ {u})) - \frac{1}{\vert X_ {u}^{f}\vert} \text{KL}_ {u}^{f}\right] + C$,

여기서 $\text{Cat}$는 카테고리 분포이고 $C$는 최적화에 영향을 주지 않는 상수이다. (실제로 $\text{Multinomial}$의 정규화 상수이다.) 부분적 피드백에 대해 주어진 ELBO를 근사시키기 위해서 $q_ {\phi}(z_ {u} \vert x_{u}) \approx q_ {\phi}(z_ {u} \vert x_ {u}^{f})$ 그리고 $\text{KL}_ {u} \approx \text{KL} _{u}^{f}$를 가정하자. 위 마지막 식에서 급수의 범위 $X_ {u}^{f}$를 $X_ {u}^{0}$로 대체하고 추가적인 가정을 이용하면,

> $\approx \frac{X_ {u}^{f}}{X_ {u}^{o}} \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u}^{f})}  \sum_ {a \in X_ {u}^{0}} \left[  \log \text{Cat}(1_ {a} \vert \pi(z_ {u})) - \frac{1}{\vert X_ {u}^{f}\vert} \text{KL}_ {u}^{f}\right] + C$
> $\approx \frac{X_ {u}^{f}}{X_ {u}^{o}} \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})}  \sum_ {a \in X_ {u}^{0}} \left[  \log \text{Cat}(1_ {a} \vert \pi(z_ {u})) - \frac{1}{\vert X_ {u}^{f}\vert} \text{KL}_ {u}\right] + C$
> $= \frac{X_ {u}^{f}}{X_ {u}^{o}} \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})} \left[  \sum_ {a \in X_ {u}^{0}} \log \text{Cat}(1_ {a} \vert \pi(z_ {u})) - \frac{\vert X_ {u}^{0}\vert}{\vert X_ {u}^{f}\vert} \text{KL}_ {u}\right] + C$
> $= \frac{X_ {u}^{f}}{X_ {u}^{o}} \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})} \left[  \log \text{Multinomial}(x_ {u} \vert \pi(z_ {u})) - \frac{\vert X_ {u}^{0}\vert}{\vert X_ {u}^{f}\vert} \text{KL}_ {u}\right] + C$

만약 $u$ 마다 $\vert X_ {u}^{f} \vert$가 일정하다면 새로운 상수 $\gamma = \frac{1}{\vert X_ {u}^{f} \vert}$를 정의하여 최종적으로 다음을 얻는다. (기댓값의 계수는 제거 할 수 있다.)

> $\mathcal{L} \approx \mathbb{E}_ {q_ {\phi}(z_ {u} \vert x_ {u})} \left[  \log \text{Multinomial}(x_ {u} \vert \pi(z_ {u})) - \gamma \vert X_ {u}^{0}\vert \text{KL}_ {u}\right]$

이와 같은 방법으로 암시적인 피드백이 주어졌을 때 $\beta = \beta(x)$를 $\gamma \vert X_ {u}^{0}\vert$로 선택 할 수 있다. 

### **3.4 Summary**

섹션 3.1, 3.2, 3.3의 결과를 종합하여 개선 손실 함수를 제안한다.

> $\mathcal{L}_ {\text{RecVAE}} = \mathbb{E}_ {q_ {\phi}(z \vert x)} \mathbb{E}_ {p(\tilde{x} \vert x)}\left[ \log p_ {\theta}(x \vert z) - \beta(x) \text{KL}({q_ {\phi}(z \vert \tilde{x})} \parallel p(z \vert \phi_ {\text{old}}, x) )\right]$

모델 훈련을 마친 뒤, 새로운 사용자에 $x$에 대해서 $p_ {\theta}( x \vert q_ {\phi}(z \vert x))$은 항목 별 긍정적으로 평가할 확률을 준다. 이를 이용하여 상위 항목을 추천 해줄 수 있다.

  
## **4. Experiment**  

RecVAE는 Adam[^12]옵티마이저로 최적화 됐으며 $\text{lr} = 5*10^{-4}$와 $500$의 배치 크기가 사용되었다. 노이즈는 평균이 $0.5$인 베르누이 분포로 주입됐고 성능을 향상시키위해 $N(0,10I)$을 복합 사전분포에 추가했다. 즉, $p(z)$, $q_ {\phi_ {\text{old}}}$, $N(0,10I)$가 복합 사전분포로 사용됐고 각각의 비율은 3:15:2가 적합했다. $\gamma$는 교차검증 (Cross-validation)을 통해 데이터마다 다른 값을 선택했다.

### **4.1 Datasets**  

||데이터 차원|평가된 항목 수|$\gamma$|
|---|---|---|---|
|MovieLens-20M|(136677, 20720)|9,990,682|0.005|
|Netflix Prize Dataset|(463435, 17769)|56,880,037|0.0035|
|Million Songs Dataset|(571355, 41140)|33,633,450|0.01|

RecVAE는 MovieLens-20M[^13], Netflix Prize Dataset[^14], Million Songs Dataset[^15]에서 평가되었으며 위 표는 각 데이터의 정보와 사용된 $\gamma$를 나타낸다. 각 데이터는 8:2의 비율로 훈련데이터와 평가데이터로 분리됐다.

### **4.2 Baselines**  

모델을 비교하기 위해서 3가지 유형의 모델들을 비교할 것이다.

- Linear models from classical collaborative filtering
	- Weighted Matrix Factorization (WMF)[^16]
	- Sparse LInear Method (SLIM)[^17]
	- Embarrassingly Shallow Autoencoder (EASE)[^18]
- Rank method
	- WARP[^19]
	- LambdaNet[^20]
- Autoencoder-based method
	- CDAE[^1]
	- Mult-DAE & Mult-VAE[^2]
	- Ranking-Critical Training (RaCT)[^21]

### **4.3 Evaluation Metrics**  

테스트 유저 $u$의 항목 $X_ {u}^{t}$와 모델의 (내림차순) 결과 $R_ {u}^{(n)}$에 대해서 $\text{Recall@}k(u)$와 $\text{NDCG@}(k(u)$가 평가 지표로서 사용될 것이다.

> $\text{Recall@}k(u) = \frac{1}{\min(\vert R_ {u}^{(n)} \vert, \vert X_ {u}^{t} \vert)} \sum_{n=1}^{k} 1\left[R_ {u}^{(n)} \in  X_ {u}^{t} \right]$ 
> $\text{DGG@}k(u) = \sum_{n=1}^{k}\frac{2^{1\left[R_ {u}^{(n)} \in  X_ {u}^{t} \right]}-1}{\log(n+1)}$
> $\text{NDCG@}(k(u) = \text{DCG@}k(u) / \left( \sum_{n=1}^{\vert X_ {u}^{t} \vert } \frac{1}{\log(n+1)} \right)$

### **4.4 Results**  

<p align="center">
<img src="https://i.ibb.co/ZdpKqMG/2023-10-15-001248.png" width="50%" height="50%">
</p>

RecVAE을 각 경쟁 모델과 비교한 결과이다. 볼드체는 가장 좋은 결과이며 밑줄은 두번째로 좋은 결과이다. Million Songs Dataset에서는 EASE가 좋은 성능을 보이지만 나머지 결과에선 RecVAE가 좋은 모습을 보여준다.

<p align="center">
<img src="https://i.ibb.co/yWsMDnT/2023-10-15-001902.png" width="50%" height="50%">
</p>

인코더 설계, 복합 사전분포, $\beta$ 조정, 교대훈련, 노이즈 주입에 대한 제거 연구 (Ablation Study)에 대한 결과이다. 교대훈련이란 인코더와 디코더를 동시에 훈련하지 않고 각각 훈련하는 것을 의미한다. 위 표에 따르면 각각의 기능은 성능 향상에 도움이 된다. 일부 기능은 개별적으로 적용되는 것보다 함께 사용되는 것이 효과적이다. (예를 들어, $\beta$ 조정과 교대훈련) 

<p align="center">
<img src="https://i.ibb.co/VJqBBFt/2023-10-15-002402.png" width="80%" height="80%">
</p>

위 그래프는 복합 사전분포의 용이함을 증명하기 위해 임의로 선택된 사용자의 에폭 (epoch)에 따른 NDCG@100의 변화량을 그린 것이다. 기존의 가우시안 사전분포 보다 복합 사전분포의 변동량이 더욱 안정적인 것을 확인 할 수 있다. 

## **5. Conclusion**  

이 논문에서는 Mult-VAE의 개선된 버전인 RecVAE를 제안한다. 이는 새로운 인코더 구조, 복합 사전분포, 협업필터링에 알맞은 $\beta$ 조정 방식을 포함하고 있으며, 여러가지 데이터에서 다른 모델의 성능을 능가했다. 향후 연구 방향으로서 주목되는 점은 (1) 이 방법을 유저와 항목을 뒤바꾸어 실험하면 어떻게 될지, (2) 사전분포를 더욱 복잡하게 만들면 어떻게 될지, (3) 컨벡스 결합이 아닌 다른 방법의 정규화를 이용하여 forgetting problem을 해결할 수 있는지와 같은 것이 고려된다.

---  
## **Additional materials & References**
Official Code Availability
>https://github.com/ilya-shenbin/RecVAE

(Review) Author information
* Gwangwoo Kim
    * Korea Advanced Institute of Science and Technology (KAIST), Graduate School of Data Science (GSDS)
    * urikokp@kaist.ac.kr
 
[^1]: Yao Wu, Christopher DuBois, Alice X Zheng, and Martin Ester. 2016. Collaborative denoising auto-encoders for top-n recommender systems. In Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 153–162.
[^2]: Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. 2018. Variational autoencoders for collaborative filtering. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 689–698.
[^3]: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. 2017. BetaVAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, Vol. 3.
[^4]: Daniel Im Jiwoong Im, Sungjin Ahn, Roland Memisevic, and Yoshua Bengio. 2017. Denoising criterion for variational auto-encoding framework. In Thirty-First AAAI Conference on Artificial Intelligence.
[^5]: Kihyuk Sohn, Honglak Lee, and Xinchen Yan. 2015. Learning structured output representation using deep conditional generative models. In Advances in neural information processing systems. 3483–3491.
[^6]: Oleg Ivanov, Michael Figurnov, and Dmitry P. Vetrov. 2019. Variational Autoencoder with Arbitrary Conditioning. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net. https://openreview.net/forum?id=SyxtJh0qYm.
[^7]: Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. 2016. Densely Connected Convolutional Networks. CoRR abs/1608.06993 (2016). arXiv:1608.06993 http: //arxiv.org/abs/1608.06993.
[^8]: Prajit Ramachandran, Barret Zoph, and Quoc V. Le. 2018. Searching for Activation Functions. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings. OpenReview.net. https://openreview.net/forum?id=Hkuq2EkPf.
[^9]: Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization. arXiv e-prints, Article arXiv:1607.06450 (Jul 2016), arXiv:1607.06450 pages. arXiv:stat.ML/1607.06450.
[^10]: Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. 2016. Vime: Variational information maximizing exploration. In Advances in Neural Information Processing Systems. 1109–1117.
[^11]: Samuel R Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. 2016. Generating Sentences from a Continuous Space. In Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning. 10–21.
[^12]: Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, Yoshua Bengio and Yann LeCun (Eds.). http://arxiv.org/abs/1412.6980
[^13]: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 5, 4, Article 19 (Dec. 2015), 19 pages. https://doi.org/10.1145/2827872.
[^14]: James Bennett, Stan Lanning, et al. 2007. The netflix prize. In Proceedings of KDD cup and workshop, Vol. 2007. New York, NY, USA., 35.
[^15]: Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 2011. The Million Song Dataset. In Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR 2011).
[^16]: Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE International Conference on Data Mining. Ieee, 263–272.
[^17]: Xia Ning and George Karypis. 2011. Slim: Sparse linear methods for top-n recommender systems. In 2011 IEEE 11th International Conference on Data Mining. IEEE, 497–506.
[^18]: Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data. In The World Wide Web Conference. ACM, 3251–3257.
[^19]: Jason Weston, Samy Bengio, and Nicolas Usunier. 2011. Wsabie: Scaling up to large vocabulary image annotation. In Twenty-Second International Joint Conference on Artificial Intelligence.
[^20]: Christopher J Burges, Robert Ragno, and Quoc V Le. 2007. Learning to rank with nonsmooth cost functions. In Advances in neural information processing systems. 193–200.
[^21]: Sam Lobel, Chunyuan Li, Jianfeng Gao, and Lawrence Carin. 2019. Towards Amortized Ranking-Critical Training for Collaborative Filtering. arXiv preprint arXiv:1906.04281 (2019).
