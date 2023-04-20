---
title:  "[ICLR 2022] Neural Link Prediction with Walk Pooling"
permalink: Neural_Link_Prediction_with_Walk_Pooling.html
tags: [reviews]
use_math: true
usemathjax: true
---


# [ICLR 2022]Neural Link Prediction with Walk Pooling

## 1. ****Motivation****

그래프는 coauthorship 네트워크 또는 human protein interactome와 같은 관계 데이터에 대한 자연스러운 모델입니다. 성공적인 Link prediction은 그래프 형성의 원리를 이해해야 합니다.그래프 신경망은 그래프 topology와 노드 feature들을 활용하여 Link prediction문제에서 높은 정확도를 달성합니다. 그러나 topology 정보는 간접적으로 표현되어 왔습니다. 서브 그래프 분류를 기반으로 하는 기존 state-of-the-art 방법들은 타겟 링크와 노드 거리에 따라 라벨을 붙여 topology 정보를 표현했습니다. 이 방법은 topology정보가 존재하지만, pooling에 의해 정보가 약해집니다.

본 논문에서는 이전 방법과는 두 가지 차이점을 둔 link prediction알고리즘을 제안합니다:

(1) 복잡한 네트워크를 통해 topology의 중요성을 보여줍니다.

(2) 기존의 GNN 기반 링크 예측 알고리즘은 topology 기능을 간접적으로만 인코딩하였습니다.Link prediction은 강력한 topology 테스크이기 때문에 topolgy정보를 직접적으로 표현 해야합니다. 이를위해 그래프 신경망(GNN)의 새로운 도메인을 활용하고, 최적의 feature를 학습합니다.

![https://user-images.githubusercontent.com/130838113/232212276-b9e0e247-fca0-4d56-a2df-9c0836a8d4e3.png](https://user-images.githubusercontent.com/130838113/232212276-b9e0e247-fca0-4d56-a2df-9c0836a8d4e3.png)

기존의 heuristics은 삼각형 또는 사각형 모양을 선호하는 것과 같은 특정 topological한 규칙을 가정합니다. 하지만 이러한 규칙을 구성하는 것은 그래프 전체에서 보편적이지 않으므로 학습해야 합니다. 본 논문에서는 topological한 heuristics의 학습 가능한 버전인 **WALKPOOL**이라는 새로운 랜덤 워크 기반 pooling메커니즘을 소개합니다.

## 2**. Problem Definition**

$N$개의 노드를 가진 그래프  $\mathcal{G^o =(V,E^o)}$에 대해 생각해보겠습니다. $\mathcal{V}$는 노드 집합이고,

$\mathcal{E^o}$는 관측된 에지의 집합입니다. $\mathcal{E^o}$는 전체 링크 집합 $\mathcal{ E^*}$의 부분 집합입니다. link prediction의 target은 $\mathcal{E^c}$(관측 되지 않은 에지)로 부터  link를 추론하는 것 입니다. 

문제를 다음과 같이 정의할 수 있습니다.

**Problem (Link predction)**

관찰된 그래프 $\mathcal{G^o} \subset\mathcal{G}$를 가지고 정확하게 $\mathcal{E^c}$의 link들을 예측하는 알고리즘 $LearnLP(\mathcal{G^o}) = \Pi :\mathcal{V\times V}\rightarrow \{True,False\}$ 를 설계하는 것.

## ****3. Method****

**WALKPOOL**을 소개하기 전에 단순화를 위해 몇 가지 notation에 대해 설명하겠습니다.

- 각 노드 =   $1,2,...,N$
- 인접 행렬 $A = (a_{ij})^N_{i,j=1}\in\{0,1\}^{N\times N}\text{ with }a_{ij} = 1 \text{ if } \{i,j\} \in \mathcal{E^o}$ and $a_{ij} = 0$  otherwise.
- 노드의 feature 벡터 =  $x_i \in \mathbb{R}^F, i \in \{1,....N\}$
- feature 벡터 행렬 $X = [x_1,...,x_n]^T \in \mathbb{R}^{N \times F}$
- $D = diag(d_1,...d_n)$ ,  $d_i = \sum_ja_{ij} = |N(i)|$
- $P = D^{-1}A$

**WALKPOOl**은 먼저 대상 링크를 포함하는 k-hop 서브 그래프(타겟 링크를 구성하는 노드들로 부터 거리가 k이내인 노드들이 이루는 그래프)를 샘플링 합니다, 타겟 링크를 포함한 서브 그래프, 포함하지 않은 서브 그래프에 대한 랜덤 워크 profiles을 각각 계산합니다.그 후 랜덤 워크 profiles이 링크 분류기에 입력됩니다. walk profiles의 계산은 다음과 같이 이루어집니다.

 1. Feature를  GNN을 통해 구합니다 $Z = f_\theta(A,X), where\text{ } f_\theta\text{ is a GNN}$

 2. Transition matrix P를 계산합니다.  $P = AttentionCoefficients_\theta(Z;\mathcal{G})$

 3.  $P^\tau$ 의 entries로 부터 Walk profiles를 구합니다.( $2\leq \tau \le\tau_c$   )

![https://user-images.githubusercontent.com/130838113/232212271-edfbc72d-e7d0-4ee3-ba7f-c1a00ee4f593.png](https://user-images.githubusercontent.com/130838113/232212271-edfbc72d-e7d0-4ee3-ba7f-c1a00ee4f593.png)

여기서  $Attentioncoefficients$를  랜덤 워크의 transition 확률로 해석합니다. 다음은 **WALKPOOL**의 각 단계에 대해서 설명하겠습니다.

### 3.1 Sampling the enclosing subgraphs

이전 연구에 따르면 링크의 존재는 작은 반경 k 내(k-hop 이내의)의 이웃에만 의존한다고 가정합니다. k(이 논문에서는 2정도로)를 작게 유지하는 것이 메모리를 절약하고 계산을 줄일 수 있어서 실용적입니다. 타겟 링크 $\{i,j\}$을 포함한 서브 그래프를 샘플링 한다고 가정하겠습니다.

- $d(x,y)$ : 노드 $x,y$의 거리중 가장 짧은 거리
- $\mathcal{V^k_{\{i,j\}}} = \{x\in\mathcal{V}:d(x,i) \le k\text{ or }d(x,j)\le k\}$
- $\mathcal{G_{\{i,j\}}^k}$ = $(\mathcal{V^k_{\{i,j\}}},\mathcal{E^k_{\{i,j\}}})$ : k- hop의 서브 그래프
- $\{x,y\} \in \mathcal{E^o}$ 이고 $x,y \in \mathcal{V^k_{\{i,j\}}}$일 때 $\{x,y\} \in \mathcal{E^k_{\{i,j\}}}$
- $A_{\{i,j\}}$ : $\mathcal{V_{\{i,j\}}}$에 있는 노드에 해당하는 인접 행렬
- $Z_{\{i,j\}}$( $Z_{\{i,j\}}$ $\subset$ $Z$)  : 서브 그래프 노드들의 feature matrix

간단히  $\mathcal{G_{\{i,j\}}^k}$ = $(\mathcal{V^k_{\{i,j\}}},\mathcal{E^k_{\{i,j\}}})$를  ${G_{\{i,j\}}}$ = $(\mathcal{V_{\{i,j\}}},\mathcal{E_{\{i,j\}}})$로 표시하겠습니다.

$\mathcal{V_{\{i,j\}}}$에 있는 노드 $i$와 노드$j$가 1과 2로 각각 레이블되어 타겟 링크 $\{i,j\}$가 서브 그래프에서 $\{1,2\}$에 mapping 될 수 있도록 합니다. $\mathcal{E^c}$에 대해 $\mathcal{S^c} = \{G_{\{i,j\}}: \{i,j\}\in \mathcal{E^c}\}$ 인 서브 그래프를 만들어서 link prediction문제를 k-hop 서브 그래프 분류 문제로 변환합니다. 학습을 위해 링크 존재 유무가 알려진 에지 $\mathcal{E^t}$를 샘플링하여 이에 해당하는 서브 그래프 set  $\mathcal{S^t}=\{G_{i,j}:(i,j)\in \mathcal{E^t}\}$ 를 구성합니다.

### 3.2 Random-walk profile

다음 단계는 인접 행렬 $A_{\{i,j\}}$ 및 노드 representations $Z_{\{i,j\}}$에서 샘플링된 서브 그래프를 분류하는 것입니다. 랜덤 워크를 사용하여 higher-order topological 정보를 추론합니다. 서브 그래프의 경우 노드 representations  $Z$를 에지 weights로 인코딩하여 해당 서브 그래프에서 랜덤 워크의 transition 확률을 계산하고 이를 사용해 타겟 링크에 profile을 계산합니다.

먼저 두 개의 노드의 상관 관계를 에지 가중치로 인코딩합니다.

$$
\omega_{x,y} = Q_\theta(z_x)^TK_\theta(z_y)/\sqrt{F^{\prime\prime}} \quad\quad\quad\quad\quad\quad\quad(1)
$$

여기서 $\{x,y\} \in \mathcal{E}$ 이고, $Q_\theta : \mathbb{R}^{F\prime}\rightarrow\mathbb{R}^{F\prime\prime}$ , $K_\theta : \mathbb{R}^{F\prime}\rightarrow\mathbb{R}^{F\prime\prime}$ 인  2층 구조인 퍼셉트론 입니다. $F^{\prime\prime}$은 퍼셉트론의 출력 차원과 같습니다.  $\{x,y\} \in \mathcal{E}$ 인 에지에 대해서 

$$
p_{x,y} = [softmas((\omega_{x,z})_{z\in \mathcal{N}(x)})] := \frac {exp(\omega_{x,y})}{\sum_{z\in \mathcal{N}(x)}exp(\omega_{x,z})} \quad\quad(2)
$$

랜덤워크 transition 확률 행렬 $P$ = $p(x,y)$ 를 계산 합니다.  $\mathcal{E}$에 포함되지 않는 에지에 대해선 $p_{x,y} = 0$ 입니다.  $\mathcal{N}(x)$는 서브 그래프에서  $x$의 이웃 노드 입니다.

행렬 $[P^\tau]_{ij}$의 성분들은 랜덤 워커가 $i$에서 $j$로 $\tau$ hops에 갈 확률로 해석할 수 있습니다. P는 타겟 링크와 관련된 노드 속성과 topogical한 정보들을  랜덤 워크의 형태로 나타냅니다. Topolgy 정보들은 GNN에서 노드 features $Z$를 추출 할때 간접적으로, P에 의해 직접적으로 포함됩니다. 입력 feature는  GNN이 노드의 features를 추출할 때 직접 포함되며, 키, 값 함수 $Q_\theta\text{ },K_\theta$에 의해 Topology정보와 결합됩니다.

행렬 $P$와 그 거듭 제곱으로부터 그래프 분류 문제에서 사용되는 features들을 계산 할 수 있습니다.노드 레벨, 링크 레벨, 그래프 레벨의 feature들을  아래와 같이 계산할 수 있습니다. 

$$
node^\tau = [P^\tau]_{1,1}+[P^\tau]_{2,2},\text{ }link^\tau = [P^\tau]_{1,2}+[P^\tau]_{2,1},\text{ }graph^\tau = tr[P^\tau].\quad(3)
$$

![https://user-images.githubusercontent.com/130838113/232212273-29035fca-878a-4508-a75a-62e32dd9cced.png](https://user-images.githubusercontent.com/130838113/232212273-29035fca-878a-4508-a75a-62e32dd9cced.png)

모든 feature들은 $\tau$가 2인 경우로 계산되었습니다.

특징들을 살펴보면, 노드 레벨의 feature는 주변 후보 링크들의 loop 구조로 나타나고,링크 레벨의 feature들은 대칭적입니다. 그래프 레벨의 feature는 모든 노드가  $\tau$의 크기만큼 loop를 돌 확률의 합과 같습니다. GNN에서 $A$의 거듭 제곱은 filter에 의해 가중치로 곱해지고 노드 feature에 가중치를  주는 데 사용되고 ,**WALKPOOL** 에서는 그래프 신호를 유효한 에지 가중치로 인코딩하고 $P^\tau$의 성분에서topological한 정보를 추출하는데 사용됩니다.

### 3.3 Perturbation extraction

정의에 따라 true 링크는 항상 서브 그래프에 존재하고, negative 링크는 존재하지 않습니다. 링크의 존재 유무가 walk prfiles에 많은 영향을 미치기 때문에 서브 그래프의 feature(3)을 직접 계산하면 overfitting으로 이어집니다. 이를 방지하기 위해 perturbation 접근 방식을 사용합니다. 샘플링된 서브 그래프 $\mathcal{G} =\mathcal{(V,E)}$ 에 대해 각각 타겟 링크를 존재하도록, 존재하지 않도록 변형한 그래프 $\mathcal{G^+} =\mathcal{(V,E\cup\{1,2\})}$  , $\mathcal{G^-} =\mathcal{(V,E\setminus\{1,2\} )}$ 를 정의합니다. 

그러면 각각의 그래프에 대해 노드 레벨 feature를 $node^{\tau,+}$ 와 $node^{\tau,-}$로 나타낼 수 있습니다. 유사하게 링크 레벨의 feature와 그래프 레벨의 feature도 나타낼 수 있습니다. 노드 레벨의 feature와 링크 레벨의 feature와는 다르게 그래프 레벨의 feature는 예측하고자 하는 링크와 관련된 정보가 그래프 전체 정보로 합쳐지게 되면서 흐려지게 만들기 때문에 링크 예측에 유용하지 않습니다. 필요 없는 전체 정보를 삭제하기 위해  perturbation = $\Delta graph^\tau = graph^{\tau,+} - graph^{\tau,-}$ 을 계산합니다.

모든 $\mathcal{G \in\text{ }}\{\mathcal{G_{\{i,j\}}:\{i,j\}\in E^c\} }$에 대해 WALKPOOL을 사용하여 아래와 같은 features의 리스트를 읽을 수 있습니다.

![https://user-images.githubusercontent.com/130838113/232212249-b56e1361-b613-41bb-a211-d69990d7ef21.png](https://user-images.githubusercontent.com/130838113/232212249-b56e1361-b613-41bb-a211-d69990d7ef21.png)

$\tau^c$는 walk 길이의 cutoff이고 하이퍼파라미터로 다룰 수 있습니다. 이러한 features는 최종적으로 시그모이드 함수를 포함한 MLP  $\Pi_\theta$를 통과합니다.

### 3.4 Training the mode

논문에 설명된 모델은 관측된 positive 및 negative 링크가 포함된 집합 $\mathcal{E^c}$와 그 주변 서브 그래프에 적합한 trainable한 매개 변수 $\theta$가 포함되어 있습니다. 본 논문에서는 MSE loss 이용하여 **WALKPOOL**을 train하였습니다.

![https://user-images.githubusercontent.com/130838113/232212255-4b2bbce8-1dcd-427a-8b65-c9eea6d0eb83.png](https://user-images.githubusercontent.com/130838113/232212255-4b2bbce8-1dcd-427a-8b65-c9eea6d0eb83.png)

$y_{\{i,j\}} = 1$   $\{i,j\} \in \mathcal{E^o}$ , $y_{\{i,j\}} = 0$  otherwise .

## 4. Experiment

### 4.1 Datasets

노드 속성이 없는 8개의 데이터 세트와 속성이 있는 7개의 데이터 세트로 실험합니다. 속성이 없는 그래프로는  USAir, NS, PB, Yeast , C.ele, Power, Router, E.coli 을 사용하였습니다. 

속성이 있는 그래프는 Cora , Citeseer, Pubmed , Chameleon, Cornell , Texas ,  Wisconsin을 사용하였습니다. 에지의 90%는 training 에지로 사용하고 나머지 10%는 test 에지로 사용합니다. 동일한 수의 negative sampling된 에지를 사용하였습니다.

### 4.2 Baselines

본 논문에서는 노드 속성이 없는 벤치마크에서 WalkPool을  walkbased heuristics인 AA, Katz ,PR과 subgraph-based heuristic인 Weisfeiler–Lehman graph kernel (WLK) 과 WLNM 및 feature method인 node2vec(N2V), spectral clustering (SPC), matrix factorization (MF), LINE과 GNN 기반의 SEAL 총 8가지 방법론과 비교하였습니다. 노드 속성이 있는 데이터 세트의 경우에는  VGAE, ARGVA 및 GIC(Graph InfoClust)의 세 가지 비지도 GNN 기반 모델과 WalkPool과 결합하여 사용하였습니다.

![https://user-images.githubusercontent.com/130838113/232212258-5739676b-1b3e-4ab7-ac3a-e5b4e2078862.png](https://user-images.githubusercontent.com/130838113/232212258-5739676b-1b3e-4ab7-ac3a-e5b4e2078862.png)

### 4.4 Results

**Synthetic datasets의 경우**

![https://user-images.githubusercontent.com/130838113/232212261-874e375e-baaf-4e6f-9d6d-9e14f23c0a63.png](https://user-images.githubusercontent.com/130838113/232212261-874e375e-baaf-4e6f-9d6d-9e14f23c0a63.png)

WalkPool은 모든 실험에서 동일한 하이퍼 파라미터를 사용하여 그래프 형성 규칙에 대한 사전 지식 없이 패턴을 잘 학습합니다. 실험에서 볼 수 있듯 topological한 organizing 규칙이 명시적으로 나타난 Synthetic datasets의 경우에 WALKPOOL은 매우 좋은 성능을 보입니다.

 

**Datasets without attributes인 경우**

![https://user-images.githubusercontent.com/130838113/232212264-0931b6d9-2e3a-40d5-96e2-fd5d8871a3df.png](https://user-images.githubusercontent.com/130838113/232212264-0931b6d9-2e3a-40d5-96e2-fd5d8871a3df.png)

WalkPool은 homophilic(높은 ACC) 데이터 세트와 heterophilic(낮은 ACC) 데이터 세트 모두 state-of-the-art를 달성합니다. AA 및 Katz와 같은 topology-based heuristics이 잘 작동하지 않았던 Power 및 Router 그래프에서도 WalkPool은 매우 좋은 성능을 내는 것을 볼 수 있습니다.  이를 통해 walk profiles이 로컬 네트워크의 패턴을 잘 학습하며, WalkPool이 사전에 topological한 가정하지 않고도 데이터에서 잘 학습할 수 있음을 볼 수 있습니다.

**Datasets with node attributes인 경우**

![https://user-images.githubusercontent.com/130838113/232212268-9cc5ec91-bc9f-43f4-b2d4-152dc7538593.png](https://user-images.githubusercontent.com/130838113/232212268-9cc5ec91-bc9f-43f4-b2d4-152dc7538593.png)

**WALKPOOL**을 사용했을 때와 사용하지 않았을 때의 모델의 결과를 볼 수 있습니다. **WALKPOOL**은 모든 비지도 학습 모델에 대해서 성능을 향상시킵니다. 특히 Pubmed 같은 topolgy정보에 대한 중요성이 큰 데이터에 대해서 많은 성능향상을 보입니다.

## Discussion

그래프의 topology는  링크 예측에서 중요한 역할을 합니다. topology는 링크에 의해 정의되기 때문에 링크 예측과 topology가 많은 관계가 있습니다.대부분의 GNN 기반 링크 예측 방법은 노드 representation과 함께 작동하며 topological한 정보를 적절하게 활용하지 않습니다. **WALKPOOL**은 노드representation 과 그래프 topology를 학습된 topological한 feature로 인코딩합니다. Supervised하거나unsupervised한 GNN을 **WALKPOOL**과 결합하면 다양한 구조적 특성을 가진 광범위한 벤치 마크에서 state-of-the art 를 달성할 수 있습니다. 

### Author

- Seonghyeon Jo
    - Graduate School of Data Science, KAIST, Daejeon, Korea