---
title:  "[KDD-24] Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias"
permalink: 2024-10-13-Popularity-Aware_Alignment_and_Contrast_for_Mitigating_Popularity_Bias.html
tags: [reviews]
use_math: true
usemathjax: true
---

## 1. Problem Definition

<img src="https://i.postimg.cc/CLLWnsSr/motivation.png" width="400" height="200" />

**[Figure 1] 아이템의 인기에 따른 추천 성능 차이 및 아이템 임베딩의 분리**

Collaborative Fitlering 기반의 방법들은 인기 없는 아이템의 경우, 사용자와 interaction이 적기 때문에 인기 있는 아이템의 supervisory로 학습이 된다. 이러한 Popularity bias로 인해 Figure1에 나타난 것처럼 인기 있는 아이템과 인기 없는 아이템 사이에서 추천 성능이 차이가 나며, 임베딩 표현도 벌어지게 된다. 

## 2. Motivation
 본 논문에서는 두 가지 모듈을 활용해 unpopoular item에 대한 표현성을 향상시키고자 하였다. 기존의 방법들은 contarstive learning을 통해 popularity bias를 완화시키고자 하였으나, 이러한 방법은 popular, unpopular item 사이의 representation sepration을 심화시킨다. 본 논문에서 제안한 PAAC의 경우는 contrastive loss에서 가중치를 조절하여 popular item과 unpopular item의 representation이 너무 분리되지 않도록 하였다.


## 3. Method

<img src="https://i.postimg.cc/HWw3GdS1/framework.png" width="650" height="300" />

**[Figure 2] Popularity-Aware-Alignment and Contrast 모델의 학습 과정**

 본 논문에서 제시한 프레임워크는 Collaborative Filtering을 기반으로 하여 Supervised Alignment Module, Re-weighting Contrast Module 두 가지 모듈을 추가한 형태로 구성되어 있다. 

### 3.1 Supervised Alignment Module
 Figure2에 그려진 대로, GCN encoder를 통과한 item embedding에 대하여 같은 user와 interaction이 있는 아이템들에 대해서 representation을 유사하게 만들어준다.

<img src="https://i.postimg.cc/NjKbyTxR/five.png" width="250" height="45" />

### 3.2 Re-weighting Contrast Module
 본 논문에서는 상위 50% item을 popular item으로 정의하고, 하위 50%의 item을 unpopular item으로 정의하였다. Contrastive loss를 계산할 때 popular item에 대해서 unpopular item이 negative sample로 뽑히거나, unpopular item에 대해서 popular item이 negative sample로 뽑히게 되면 representation separation이 심화된다. 따라서 본 논문에서는 아래의 식처럼 contrastive loss를 계산할 때 가중치를 주어 negative sample을 지나치게 밀어내는 것을 막도록 하였다.
<img src="https://i.postimg.cc/1XGVt6Bc/cl.png" width="300" height="40" />

<img src="https://i.postimg.cc/Gm6J83Kt/8.png" width="300" height="50" />

<img src="https://i.postimg.cc/Gm6J83Kt/8.png" width="300" height="50" />


## 4. Experiment

### Research Question
 - RQ1: PAAC가 기존의 모델과 어떻게 다른지?
 - RQ2: PAAC의 서로 다른 구성 요소가 어떻게 역할을 수행하고 있는지?
 - RQ3: PAAC가 어떻게 popularity bias를 완화하는지?
 - RQ4: Hyper-parameter가 PAAC의 추천 성능에 어떻게 영향을 주는지?

### Experiment setup
- Dataset: Amazon-book, Yelp2018, Gowalla
- baseline: IPS, $𝛾$-AdjNorm, MACR, InvCF, Adap-$t$, SimGCL
- Evaluation Metric: Recall@K, HR@K, NDCG@K

### Result

### Overall Performance(RQ1)
<img src="https://i.postimg.cc/SKJ9GBZ5/test.png" width="800" height="300" />

**[Table 1] Baseline과 PAAC의 성능 및 향상 정도**

 모든 베이스라인에 대하여 본 논문에서 제시한 PAAC가 popularity bias를 완화하며 성능 향상을 보였다. 특히, LightGCN을 베이스로 사용한 PAAC는 LightGCN의 NDCG@20 성능을 모든 데이터셋에 대해 크게 향상시켰다. 그러나 sparse한 Gowalla같은 데이터에 대해서는 작은 향상만을 보였다.

### Ablation Study(RQ2)
<img src="https://i.postimg.cc/SjFV0yPJ/ablation.png" width="670" height="150" />

**[Table 2] PAAC의 Ablation**
- PAAC-w/o P: popular item의 re-weighting contrastive loss가 사라진 경우
- PAAC-w/o U: unpopular item의 re-weighting contrastive loss가 사라진 경우
- PAAC-w/o A: popularity-aware supervised alignment loss가 사라진 경우

popular item의 re-weighting contrastive loss가 사라진 경우에 가장 큰 성능 하락이 있었으며, alignment loss가 없는 케이스도 SimGCL보다 좋은 성능을 보였는데 이는 popularity에 따라 구분된 contrastive loss가 적용되었기 때문이다.

### Debias Ability(RQ3)
<img src="https://i.postimg.cc/zvC5YffQ/pop.png" width="500" height="200" />

**[Figure 3] Popular/ Unpopular item 각각에서의 추천 성능**

 Gowalla와 Yelp2018 데이터셋에 대하여, 상위 20%의 item을 Popular item으로, 나머지를 Unpopular item으로 분류하여 성능을 측정하면 LightGCN 베이스의 PAAC가 Unpopular item에서 성능을 많이 향상시킨다는 것을 확인할 수 있다.

### Hyperparameter Sensitives(RQ4)
<img src="https://i.postimg.cc/FFyY0yb0/dd.png" width="500" height="200" />

**[Figure 4] $\lambda_ 1, \lambda_ 2$에 따른 성능 향상 정도**

$\lambda_ 2$이 증가할 때, 처음엔 성능이 향상되지만 어느순간 감소하며 $\lambda_ 1$이 증가할 때 역시 초반엔 성능이 향상되지만 어느순간 감소한다.


<img src="https://i.postimg.cc/8PvfsdF8/hyper.png" width="500" height="300" />

**[Figure 5] $\gamma, \beta$에 따른 PAAC의 성능**

Yelp2018에서 $\gamma = 0.8, \beta = 0.6$일 때, Gowalla에서 $\gamma = 0.2, \beta = 0.2$일 때가 최적의 값을 보이는데, 이는 item당 interaction이 상대적으로 많은 Yelp에서는 popular item을 positive sample로 쓰는데서 많은 이득을 보기 때문이다.

## 5. Conclusion

 본 논문에서는 popularity bias 해결을 위해 PAAC를 제안하였다. 같은 user를 공유하는 item들은 비슷한 특성을 가졌을 거라는 가정하에서, popularity-aware supervised alignment approach를 고안하고 contrastive learning 기반의 모델에서 representation separation을 방지하기 위하여 popularity level에 따라서 loss의 weight를 조절하였다. 이러한 방법으로 개선된 PAAC는 다양한 데이터셋에서 성능이 개선되는 것으로 증명되었다.
  창의적인 솔루션이 아니더라도, 문제의 존재를 명확히 밝히고 개선의 여지를 보일 수 있는 것 또한 좋은 연구라는 생각이 들었습니다.

- Author Information

  - Jimin Seo
  - Dept. of ISysE, KAIST
  - Research Topic: Recommender System
  
## 6. Reference & Additional materials
\- Miaomiao Cai, Lei Chen, Yifan Wang, Haoyue Bai, Peijie Sun, Le Wu. Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias. *KDD(2024)*.

\- Github Implementation : https://github.com/miaomiao-cai2/KDD2024-PAAC.

