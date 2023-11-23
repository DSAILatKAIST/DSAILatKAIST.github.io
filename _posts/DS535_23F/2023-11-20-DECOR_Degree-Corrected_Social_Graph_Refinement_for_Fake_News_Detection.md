---
title:  "[KDD 2023] DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection"
permalink: 2023-11-20-DECOR_Degree-Corrected_Social_Graph_Refinement_for_Fake_News_Detection.html
tags: [reviews]
use_math: true
usemathjax: true
---

# DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection

## **1. Problem Definition**  
> **GNN 방법론으로 가짜 뉴스를 탐지한다.**  

의도적으로 허위 정보를 포함하는 가짜 뉴스는 사회에 악영향을 끼칩니다.  이러한 가짜 뉴스를 진짜 뉴스와 구별해 내는 것을 **Fake News Detection** task라고 합니다. 이때 각각의 뉴스 기사와 그것을 읽는 유저 간의 관계를 graph 형태의 data로 나타내어 Graph Neural Network (GNN)로 학습해 가짜 뉴스를 탐지하는 형태의 연구가 활발히 이뤄지고 있습니다. 여기서 각 뉴스가 진짜인지 가짜인지 여부를 **veracity**라고 부릅니다.

![image](https://portal.fgv.br/sites/portal.fgv.br/files/styles/noticia_geral/public/noticias/mai/18/fake-news.jpg?itok=diJJLEtE&c=33313733cdad61e4bd51beabb4a84531)

<br/> <br/>

## **2. Motivation**  
> **Fake News Detection에서 noisy edge와 관련해 발생하는 문제를 해결한다.**   

GNN을 기반으로 한 fake news detection을 다룬 이전 연구들은 주로 structure (edge와 adjacency matrix 관련 정보)가 고정된 graph를 사용합니다. 하지만 이러한 형태는 graph가 element(뉴스, 유저 등) 간의 관계를 정확하게 드러낼 수 있어야만 효과적입니다. 즉, 구조적인 결함으로서 GNN의 message passing mechanism에 악영향을 끼치는 **noisy edge가 존재하는 경우 fake news detection이 원활하게 이루어지지 않을 수 있습니다.**

논문에서 저자들은 각 뉴스가 node인 graph에서, **진짜 뉴스와 가짜 뉴스 사이를 잇는 edge를 noisy edge라 칭합니다.** Veracity 관점에서 성격이 정반대인 두 node 사이의 edge가 있으면 둘 간의 정보 교환으로 인해 GNN 학습이 제대로 되지 않기 때문입니다. 따라서 fake news detection의 성능을 높이기 위해 이러한 noisy edge의 weight를 감소시킬 필요가 있습니다. 이렇게 edge weight에 변화를 주는 방식을 **social graph refinement**라 부릅니다.

저자들은 empirical한 분석을 통해 noisy edge가 각 node의 degree와 유의미한 연관이 있음을 보이고, 이를 바탕으로 **De**gree-**Cor**rected Social Graph Refinement **(DECOR)** framework을 제안합니다. DECOR는 **social degree correction mask**를 기반으로 **noisy edge downweighting**을 펼치는데, 이를 통해 fake news detection의 성능을 향상시킬 수 있습니다.  

<br/> <br/>

## **3. Preliminary Analysis**  
> **Problem Formulation**  

저자들의 empirical analysis를 설명하기에 앞서 논문에서 해결하고자 하는 task에 대해 소개하겠습니다.

$\mathcal{D} = (\mathcal{P}, \mathcal{U}, \mathcal{R})$는 $N$개의 sample을 포함하는 fake news detection dataset입니다. 여기서 $\mathcal{P} = \{p_ 1, p_ 2, ..., p_ N\}$는 뉴스 기사의 집합이고, $\mathcal{U} = \{u_ 1, u_ 2, ...\}$는 유저의 집합입니다. 이때 각 유저는 적어도 하나의 뉴스 기사를 SNS를 통해 공유한 적이 있어야 합니다. $\mathcal{R}$은 유저 engagement의 집합인데, 각 $r \in \mathcal{R}$은 $\{(u, p, k) \vert u \in \mathcal{U}, p \in \mathcal{P}\}$로써 정의됩니다. 즉 유저 $u$가 뉴스 기사 $p$를 $k$번 공유했다는 의미입니다.

여기서 **'fake news detection on social media'라는 task는 binary classification problem**입니다. 즉 주어진 $\mathcal{D}$에서 $\mathcal{P}$를 training set $\mathcal{P}_ {train}$과 test set $\mathcal{P}_ {test}$로 나눈 후, $\mathcal{P}_ {train}$에 있는 뉴스 기사에 대해서는 ground-truth veracity label $\mathcal{Y}_ {train}$ 이 제공된 상황에서, classifier $f$를 학습해 $\mathcal{P}_ {test}$에 있는 **뉴스 기사들의 veracity label $\mathcal{Y}_ {test}$ 를 예측하는 것**이 목적입니다. 이때 각 뉴스 기사에 대해서 **veracity label은 가짜 뉴스라면 1, 진짜 뉴스라면 0**입니다.

<br/> <br/>

> **News Engagement Graph**  

GNN을 활용하기 위해서는 위의 $\mathcal{D}$를 graph 형태의 data로 변환해야 합니다. 우선 활동적인 유저에 대해서만 보기 위해 3개 이상의 뉴스 기사 공유를 한 유저의 집합 $\mathcal{U}_ A \in \mathcal{U}$로 범위를 줄입니다. 여기서 **user engagement matrix** $\mathrm{E} \in \mathbb{R}^ {\vert \mathcal{U}_ A \vert \times N}$를 만드는데, 이때 $\mathrm{E}_ {ij}$에 해당하는 값은 유저 $u_ i$가 뉴스 기사 $p_ j$ 를 공유한 횟수, 즉 $(u_ i, p_ j, k_ {ij}) \in \mathcal{R}$에서 $k_ {ij}$입니다.

활동적인 유저들의 뉴스 소비 패턴에 대한 정보를 바탕으로 비슷한 유저의 흥미를 끄는 뉴스 기사를 서로 이어줄 수 있습니다. 이를 **news engagement graph** $\mathcal{G} = \{\mathcal{P}, \mathcal{E}\}$라 부르는데, weighted undirected graph입니다. 여기서 각 뉴스 기사가 node이고, $\mathcal{G}$의 adjacency matrix $\mathrm{A} \in \mathbb{R}^ {N \times N}$는 $\mathrm{A} = \mathrm{E}^ \top\mathrm{E}$입니다. 즉 $\mathrm{A}_ {nk}$는 두 뉴스 기사 $p_ n, p_ k$를 여러 유저가 공유한 횟수의 총합이고, 따라서 $\mathrm{A} _{nk}$를 edge $e_ {nk} \in \mathcal{E}$의 weight로 두면 이 값이 클수록 두 뉴스 기사 모두를 공유한 유저 집단의 관심사가 더 비슷하다고 볼 수 있습니다.

<br/> <br/>  

> **Empirical Observations**  

저자들은 두 dataset PolitiFact와 GossipCop에서 만든 news engagement graph를 분석한 결과, **weighted node degree**의 관점에서 가짜 뉴스와 진짜 뉴스가 다른 패턴을 보임을 밝힙니다. Weighted node degree란 각각의 node에 대해서 **그 node와 이어진 모든 edge의 weight의 총합**입니다.

**1. Degree-Veracity Correlations**  

여기서 저자들이 알아보고자 하는 것은 **가짜 뉴스 기사는 진짜 뉴스 기사에 비해 더 많은/적은 공유를 유도하는지**입니다. 아래 figure는 각 dataset에 대해 가짜 뉴스와 진짜 뉴스 각각의 degree distribution을 kernel distribution estimation (KDE)으로 시각화한 것 입니다. 

![image](https://i.ibb.co/0VF74rC/image.png)

여기서 **가짜 뉴스와 진짜 뉴스는 확연히 다른 degree distribution을 보인다**는 것을 알 수 있습니다. 이때, 둘 중 어느 쪽이 더 큰 degree를 가지는 경향인지는 dataset마다 다릅니다. 연예 뉴스를 다루는 GossipCop의 경우 진짜 뉴스가 더 많은 공유를 유도하고, 정치 뉴스를 다루는 PolitiFact는 반대의 패턴을 띠는 것을 확인할 수 있습니다.

**2. News Co-Engagement**  

다음으로 저자들은 degree에 따른 뉴스 기사 pair의 특성을 분석합니다. 우선 news engagement graph에서 두 뉴스 기사 $p_ i, p_ j$ 사이에 edge가 있을 경우, 이 edge는 **(1) 진짜 뉴스 pair, (2) 진짜-가짜 pair, 그리고 (3) 가짜 뉴스 pair**라는 3가지 종류 중 하나에 속합니다. 또한 두 뉴스 기사 모두 공유한 유저의 수를 다음과 같은 **co-engagement score**로 정의할 수 있습니다.

![image](https://i.ibb.co/cy60vLM/image.png)

여기서 **주어진 edge의 종류와 그 edge가 잇는 두 뉴스 기사의 co-engagement score 사이에 관계가 있는지** 확인할 수 있습니다. 아래 figure는 각 edge를 $d_ i \times d_ j$ (각 뉴스 기사의 degree의 곱) 값에 따라 여러 bucket으로 나눈 후, 세 종류 각각의 co-engagement score를 보여줍니다.

![image](https://i.ibb.co/pdcZ0MC/image.png)

두 dataset 모두에서 degree가 주어졌을 때, **가짜 뉴스 pair (파랑 사각형)는 $C_ {ij}$ 값이 높은 (즉 두 뉴스 모두 공유한 유저가 많은) 경향이 있다는 것,** 그리고 **진짜-가짜 pair (초록 삼각형)는 $C_ {ij}$ 값이 낮은 경향이 있다는 것**을 알 수 있습니다.

이 두 observation을 통해 news engagement graph에서 **node(뉴스 기사)와 edge(유저의 공유) 모두 degree와 관련이 있음**을 알 수 있습니다. 앞서 noisy edge를 진짜-가짜 pair로 정의했는데, 이 **degree-related 패턴을 이용해 noisy edge의 weight를 줄일 수 있다**는 것이 저자들의 요지입니다.

<br/> <br/>

## **4. Proposed Framework - DECOR**  
> **Framework**  

우선 더 자세한 설명에 앞서 DECOR의 학습 과정을 하나의 figure로 나타내면 다음과 같습니다.

![image](https://i.ibb.co/Y3ZS3nv/image.png)

<br/> <br/>

> **Connection with the DCSBM Model**  

논문에서 제안하는 DECOR framework은 [Degree-Corrected Stochastic Blockmodel (DCSBM)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.016107)이라는 모델을 fake news detection task에 맞게 extend한 것으로 볼 수 있습니다.

우선 intuition 상 same-class edges (= 진짜 뉴스 pair 또는 가짜 뉴스 pair를 잇는 edges)가 cross-class edges ( = 진짜-가짜 pair를 잇는 edges)보다 학습에 용이할 것입니다. 따라서 class가 서로 같은 ($z_ i = z_ j; z \in [0, 1]$는 각 node의 class label) edge를 판별하는 것이 매우 중요하고, 이를 위해 다음과 같이 co-engagement score $C_ {ij}$를 기반으로 **same-class likelihood ratio**를 정의할 수 있습니다.

![image](https://i.ibb.co/59nFPbX/image.png)

여기서 **likelihood ratio가 클수록 edge가 same-class일 가능성이 더 높다는 것을 의미**하고, 이에 따라 그러한 edge에는 더 큰 weight를 주게 됩니다.

이때 DCSBM 모델에서 정의된 $LR_ {ij}$의 **maximum likelihood estimate (MLE)**를 다음과 같이 구할 수 있는데,

![image](https://i.ibb.co/CKMQ2t6/image.png)

여기서 $m = \vert \mathcal{E} \vert$, $d_ i, d_ j$는 edge $e_ {ij}$가 잇는 두 뉴스 기사 node $p_ i, p_ j$ 각각의 degree, $p, q$는 parameter입니다. $m, p, q$를 고정값으로 두면 이 MLE가 다음과 같이 $C_ {ij}, d_ id_ j$에 대한 log-linear function임을 알 수 있습니다.

![image](https://i.ibb.co/3NRDRb0/image.png)

일반적인 DCSBM 모델에서는 $p, q$ 등의 정보가 pre-defined되어 있기 때문에 $LR_ {ij}$의 estimator로서 위의 $\Phi (C_ {ij}, d_ i, d_ j)$를 그대로 사용하지만, 이렇게 하면 learnable parameter가 너무 적어서 news engagement graph의 복잡한 degree-based pattern을 반영하기가 어려워집니다. 따라서 DECOR에서는 fixed and log-linear function인 $\Phi (C_ {ij}, d_ i, d_ j)$ 대신에 **learnable non-linear function**인 $\tilde{\Phi} (C_ {ij}, d_ i, d_ j)$를 사용합니다. 
 
<br/> <br/>

> **Social Degree Correction**  

앞서 empirical analysis에서 보였듯이, news engagement graph (이하 $\mathcal{G}$)에서 edge의 종류는 그 edge가 잇는 node의 degree와 co-engagement score와 관련이 있습니다. 이 분석 결과와 위의 DCSBM 모델에서 정의한 function을 기반으로, **social degree correction mask** $\mathrm{M} \in \mathbb{R}^ {N \times N}$을 학습해 $\mathcal{G}$의 edge weight를 조정하는 것이 DECOR의 목적입니다. 여기서 $\mathrm{M}_ {ij}$sms $(0, 1)$ 사이의 값을 취하고, 이 값은 두 뉴스 기사 $p_ i, p_ j$ 사이를 잇는 edge $e_ {ij}$의 degree correction score를 나타냅니다.

$p_ i, p_ j$의 degree $d_ i, d_ j$와 co-engagement score $C_ {ij}$를 바탕으로 $\mathrm{M}_ {ij}$를 예측할 수 있습니다. 우선 다음과 같이 구하는데, 

![image](https://i.ibb.co/HhX2Zvh/image.png)

여기서 $\tilde{\Phi} (\cdot)$는 MLP 기반 architecture이고 DCSBM 모델의 Eq. 8의 learnable extension으로 볼 수 있습니다. 이 $s_ {ij}$를 모든 edge에 대해 구한 후 (모든 node pair에 대해 하지 않기 때문에 computationally efficient), 다음과 같이 degree correction score를 구합니다.

![image](https://i.ibb.co/R6sd7mF/image.png)

여기서 $v_ {ij}$는 $s_ {ij}$를 softmax 함수로 normalize한 score입니다. $C_ {ij}$를 모두 포함하는 co-engagement matrix $\mathrm{C}$와 $\mathrm{M}$을 기반으로, 최종적으로 degree-corrected adjacency matrix $\mathrm{A}_ c$를 다음과 같이 구할 수 있습니다.

![image](https://i.ibb.co/SdkV1N3/image.png)

여기서 $\mathrm{I}$는 identity matrix이고 $\mathrm{D}$는 degree에 대한 $\hat{\mathrm{A}}$의 diagonal matrix입니다.

이 일련의 과정을 통해 $\mathcal{G}$의 noisy edge에 작은 weight를 줄 수 있는데, 이는 Eq. 9의  $\tilde{\Phi} (\cdot)$가 학습을 통해 낮은 degree correction score를 예측하기 때문입니다. 학습에 대해서는 다음 section에서 설명하겠습니다.

<br/> <br/>

> **Prediction on Degree-Corrected Graph**  

앞서 구한 degree-corrected adjacency matrix $\mathrm{A}_ c$를 기반으로 GNN architecture를 통해 뉴스 기사의 veracity label을 예측할 수 있습니다. 우선 뉴스 기사 $p \in \mathcal{P}$에 대해 intial news article feature는 다음과 같습니다.

![image](https://i.ibb.co/c6j7LdX/image.png)

여기서 $\mathrm{x}_ p$는 뉴스 기사의 내용을 pre-trained 언어 모델을 통해 feature로 만든 것입니다. 이를 시작으로 GNN의 $k$번째 layer에서의 news article feature는 다음과 같이 구할 수 있습니다.

![image](https://i.ibb.co/LpLyYLj/image.png)

여기서 $\mathcal{N}(p)$는 $\mathcal{G}$에서 $p$의 neighbor입니다. 최종적으로 $p$에 대해 이 GNN 기반 classifier의 output을 $\mathrm{h}_ p \in \mathbb{R}^ 2$라 할 때, $p$의 veracity label $\tilde{\mathrm{y}}_ p = \mathrm{softmax}(\mathrm{h}_ p)$를 구할 수 있습니다. 학습 때는 다음과 같이 cross entropy loss를 사용합니다.

![image](https://i.ibb.co/dWFck6K/image.png)

<br/> <br/>

> **Framework Recap**  

이해를 돕기 위해 앞서 설명한 DECOR의 학습 과정을 나타낸 figure 다시 한 번 보여드립니다. 정리하자면 **(1) news engagement graph $\mathcal{G}$를 만든 후**, **(2) 각 edge에 대해 co-engagement score $C_ {ij}$를 구합니다.** 이후 $C_ {ij}$와 각 node의 degree $d_ i, d_ j$를 기반으로 **(3) noisy edge에 더 작은 weight가 부여된 degree-corrected adjacency matrix $\mathrm{A}_ c$를 만들고** GNN 학습을 통해 **(4) 각 뉴스 기사에 대해 가짜인지 진짜인지 여부를 판별**합니다.

![image](https://i.ibb.co/Y3ZS3nv/image.png)

<br/> <br/>

## **5. Experiments**  

이 논문은 여러 종류의 실험과 ablation study를 통해 다음 5개의 research question에 답하고자 합니다:
* (Q1) Fake News Detection Performance: DECOR는 SOTA baseline method와 비교해 우위를 가지는가?
* (Q2) Ablation Study: Co-engagement와 degree pattern이 DECOR의 성능에 도움을 주는가?
* (Q3) Limited Training Data: Label이 sparse한 상황에서도 DECOR의 성능이 괜찮은가?
* (Q4) Computational Efficiency: 이전 method에 비해 DECOR는 얼마나 효율적인가?
* (Q5) Case Study: DECOR의 downweighting이 실제로 제대로 이루어지는가?

<br/> <br/> 

> **Experimental Setup**  

**Datasets**  

다음과 같이 2개의 real-world dataset을 실험에 사용합니다. 두 dataset 모두 요약된 뉴스 기사와 트위터 유저의 공유 정보를 포함합니다. 이때 현실의 상황과 유사하게 하기 위해, 가장 최근 20%의 뉴스 기사를 test set으로, 이전 80%를 training set으로 삼습니다.   

![image](https://i.ibb.co/GFC7bVS/image.png)

**Baselines**  

다음과 같이 12개의 baseline method와 비교합니다.
* **News content based methods (G1)** - 단순히 뉴스 기사의 내용을 feature로 사용: dEFEND\c, SAFE\v, SentGCN, BERT, DistilBERT
* **Social graph based methods (G2)** - 유저와의 관계를 graph structure로 만들어 GNN 기반으로 학습: GCNFN, FANG, GCN, GIN, GraphConv
* **Graph Structure Learning (GSL) based methods (G3)** - graph structure에 변형을 주어 최적시킨 후 학습: Pro-GNN, RS-GNN 

DECOR의 경우 model-agnostic합니다. 즉 다양한 GNN 모델과 결합해 사용될 수 있습니다. 논문에서는 GCN, GIN, GraphConv를 backbone architecture로 삼은 3가지 버전의 DECOR를 비교합니다.

**Evaluation Metrics**  

다음과 같이 5개의 metric으로 detection performance를 측정합니다.
* Accuracy (Acc.), Precision (Prec.), Recall (Rec.), F1 Score (F1)

이때 모든 method에서 20번의 실험 후 그 평균을 구합니다.

<br/> <br/>

> **Performance Comparison (Q1)**  

![image](https://i.ibb.co/wJcKhgB/image.png)

위 table에서 모든 경우에서 **DECOR가 baseline method보다 월등한 성능을 보임**을 알 수 있습니다. 또한 몇 가지 흥미로운 점을 관찰할 수 있는데, 우선 G2 method가 G1 method보다 성능이 좋은 것을 통해 **graph structure를 사용하는 것이 fake news detection에 유용함**을 알 수 있습니다. 또한 이전 GSL method (G3)의 성능이 많이 안 좋기 때문에 이 task에 적합하지 않다고 볼 수 있는데, 저자들은 이것이 Pro-GNN 등은 feature similarity 기반이기 때문이라고 추측합니다. 즉 두 node의 feature가 서로 많이 다르더라도 (두 뉴스 기사의 주제나 내용이 상이하더라도) 다른 관점 (co-engagement나 veracity type)에서는 비슷할 수 있기 때문에 **단순히 feature 간의 차이가 크다고 edge weight를 줄이는 방식보다는 DECOR의 방식이 더 뛰어나다**고 볼 수 있습니다.

<br/> <br/>

> **Ablation Study (Q2)**    

![image](https://i.ibb.co/yRQhpPn/image.png)

DECOR의 주요 compoment가 모델의 fake news detection 성능에 미치는 영향을 확인하기 위해 co-engagement와 degree 정보가 각각 배제된 DECOR-COE와 DECOR-Deg를 원본과 비교했습니다. 위의 figure에서 기존의 DECOR가 두 variant보다 높은 성능을 보임을 확인할 수 있는데, 이는 **co-engagement와 degree 정보 모두 DECOR의 성능에 큰 영향을 미친다**는 것을 의미합니다. 또한 degree 정보만 사용하는 DECOR-COE가 둘 모두 사용하지 않는 일반 GNN보다 높은 성능을 보인다는 것도 알 수 있는데, 이는 저자들이 계속 강조한 것처럼 degree 정보의 중요성을 더욱 부각시킵니다.

<br/> <br/>

> **Performance under Label Security (Q3)**    

![image](https://i.ibb.co/jLWrqRM/image.png)

Fake news detection을 real-world에 적용시킬 때 **label scarcity, 즉 각 뉴스 기사가 진짜인지 가짜인지 아는 비율이 적다는 문제가 발생**합니다. 실시간으로 생기는 뉴스 기사 모두에 대해 정확한 정보를 얻기가 어렵기 때문입니다. Label의 양이 적을 때 성능이 심하게 하락한다면 real-world에서의 적용이 힘들 것입니다. 이러한 상황에서 DECOR의 안정성을 확인하기 위해 training set의 비율을 줄여가며 실험한 결과, 위 figure와 같이 다른 method에 비해 성능이 높음을 확인할 수 있습니다. 저자들은 **DECOR가 사용하는 다른 정보 (co-engagement, degree)가 ground-truth label 정보의 부재를 완화**해주기 때문이라고 합니다.

<br/> <br/>

> **Computational Efficiency (Q4)**    

![image](https://i.ibb.co/k1Z0VNY/image.png)

일반적으로 GSL method는 다른 GNN method에 비해 computational time이 많이 듭니다. 하지만 위 figure에서와 같이 DECOR의 경우 성능 향상은 가져가되 computational cost가 적게 듭니다. 그 이유로는 우선 이전 GSL method는 dimension이 큰 뉴스 기사 내용 feature를 사용하는 반면 DECOR는 **degree와 co-engagement라는 low-dimensional feature를 사용**하기 때문입니다. 또한 모든 node pair에 대해 무언가 계산을 해야 하는 (quadratic complexity) 이전 GSL method와는 다르게 DECOR는 **edge가 연결된 경우에 대해서만 계산 (linear complexity)**하면 됩니다. 이러한 특징 때문에 DECOR는 리소스가 제한된 상황에서 더욱 유리합니다.

<br/> <br/>

> **Case Study (Q5)**    

![image](https://i.ibb.co/sq3tKT2/image.png)

위 figure에서 DECOR 적용 후 실제로 가짜 뉴스 pair (clean edge)의 weight는 증가한 반면 진짜-가짜 pair (noisy edge)의 weight는 감소한 것을 확인할 수 있습니다. 이 결과는 **DECOR가 news engagement graph를 효과적으로 refine함**을 보입니다.

<br/> <br/>

## **6. Conclusion**  
> **Summary**  

이 논문에서는 **Fake News Detection 분야에 social graph refinement, 즉 Graph Structure Learning (GSL)을 적용한 method**를 다루었습니다. 저자들은 news engagement graph에서 noisy edge가 node degree와 관련이 있음을 empirical하게 보인 후 이를 기반으로 edge downweighting을 하는 DECOR 모델을 소개합니다. 2개의 real-world dataset에 대해 여러 metric에서 12개의 baseline과의 비교를 통해 성능이 높음을 확인할 수 있습니다. 앞으로 Fake News Detection에서의 GSL의 적용과 발전이 기대가 됩니다.

<br/> <br/>

> **생각 및 발전 방향**  

GSL이라는 익숙한 방법론을 fake news detection에 적용한 점이 흥미로웠습니다. 다른 class의 node끼리 연결되어 있으면 학습이 좋지 않다는, 어쩌면 당연한 얘기를 empirical하게 보여주며 분석한 점도 인상 깊었습니다. 글도 읽기 쉽게 쓰여 있어 이해하기 어렵지 않은 논문이었습니다.   

다만 몇 가지 의문점 및 비판이 있습니다. 가장 먼저 co-engagement score를 구할 때는 왜 활동적인 유저 ($\mathcal{U}_ A$)가 아닌 전체 유저 ($\mathcal{U}$)를 보는지 의문입니다. 큰 이슈는 아니지만 그 이유에 대해 명확한 설명이 있으면 더 좋았을 것 같습니다.

또한 ablation study에서 DECOR-COE와 DECOR-Deg에 대한 설명이 너무 단순한 것은 흠으로 느껴집니다.  DECOR에서 새 adjacency matrix를 만드는 과정 상 co-engagement와 degree 정보 중 어느 하나라도 사용할 수 없다면 구조적으로 큰 변화가 생길 텐데, 이에 대한 언급이 없어 아쉽습니다.

마지막으로 label scarcity에 대한 실험에서 왜 GCN, GIN, GraphConv와만 비교하고 다른 baseline은 없는지 궁금합니다. 중요한 이슈라고 저자들이 언급한 만큼 더 광범위한 실험이 있으면 좋았을 것입니다.

위 의문점을 해소하면서 실험하는 것도 괜찮은 발전 방향인 것 같습니다.

Thank you for reading!

<br/> <br/>

## **Author Information**
* Junghoon Kim
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Learning, Anomaly Detection
  * Contact: jhkim611@kaist.ac.kr

## **Reference & Additional materials**
* Github Implementation
  * [Official codes for the paper](https://github.com/jiayingwu19/DECOR)
* Reference
  * [[KDD-23] DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection](https://arxiv.org/abs/2307.00077)
  * [DCSBM](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.016107)
