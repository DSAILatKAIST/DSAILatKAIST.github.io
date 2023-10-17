---
title:  "[ICML 2022] Rethinking Graph Neural Networks for Anomaly Detection"
permalink: 2023-10-16-Rethinking_Graph_Neural_Networks_for_Anomaly_Detection.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [ICML-22] Rethinking Graph Neural Networks for Anomaly Detection

# Title

---

Rethinking Graph Neural Networks for Anomaly Detection

| Author | Booktitle | Year |
| --- | --- | --- |
| Tang, Jianheng and Li, Jiajin and Gao, Ziqi and Li, Jia | International Conference on Machine Learning | 2022 |

## **1. Problem Definition**

### 1.1 Background

 이상치 (An anomaly or an outlier) 는 대부분의 개체에서 크게 벗어난 데이터 객체를 뜻하며, 이상치 탐지 (Anomaly Detection) 문제로는 사이버 보안, 사기 탐지, 장치 오류 탐지 등이 있습니다. 기술의 발전으로 인하여 그래프 데이터가 보편화되면서 structural data에 대한 분석으로 Graph Neural Networks (이하 “GNN”)이 각광받았고 자연스럽게 그래프 이상 탐지 작업 (Graph Anomaly Detection Task)에 적용되었습니다. 하지만, vanilla GNN은 지나친 확일화 문제로 인하여 이상치 탐지에 적합하지 않았고 이를 개선하기 위하여 attention 매커니즘을 적용하는 방법, resampling 전략을 사용하는 방법, 그리고 보조의 losses를 설계하는 방법이 제안되었습니다. 이 세 가지 방법은 모두 spatial domain에서의 분석이며, spectral domain에서의 분석은 거의 이루어지지 않았습니다. GNN을 설계할 때 알맞는 spectral filter을 적용하는 것 또한 중요하기에 해당 논문에서는 ‘이상치 탐지를 위한 GNN에서 적절한 spectral filter을 어떻게 고를 것인가?’ 에 대해 답변하고자 합니다.

### 1.2 Overview

 ‘이상치 탐지를 위한 GNN에서 적절한 spectral filter을 어떻게 고를 것인가?’에 대해 답변하기 위해 두 가지 과정을 거칩니다. 첫 번째로, 그래프 이상 탐지에서 spectral localized band-pass filters의 중요성을 확인하는 과정입니다. 논문의 저자는 이상의 정도(degree)가 커질수록, 저주파 에너지가 점진적으로 고주파 에너지로 전환됨을 확인하였고 이를 spectral 에너지 분포의 ‘오른쪽 편이 (right-shift)’ 현상으로 정의하였습니다. ‘오른쪽 편이’ 현상에 대한 수리적 증명 및 데이터를 통한 검증으로 적절한 spectral filter의 필요성을 보입니다. 두 번째로, 그래프 이상치에서의 ‘오른쪽 편이’ 현상을 잘 다루는 새로운 알고리즘, Beta Wavelet Graph Neural Network (이하 “BWGNN”)을 제안합니다. Hammond’s graph wavelet theory에서 착안하여 Heat kernal이 아닌 Beta kernal을 사용함으로써, 매우 유동적이고 spatial/spectral-localized 하며 band-pass한 filter을 통해 고주파 이상 현상을 해결합니다. 

### 1.3 Preliminaries

 속성 그래프 (Attributed graph)는  $G = \{ V,E,X \}$ 로 정의되며, $V$ 는 node, $E$는 edge, $X$는 node features을 의미합니다. $A$를 adjacency matrix, $D$를 degree matrix로 표현합니다. $V_ {a}$ 와 $V_ {n}$을 두 개의 분리된 하위집합이라 할때, $V_ {a}$ 는 모든 이상 노드를 나타내고, $V_ {n}$ 은 모든 정상 노드를 나타냅니다. 그래프 기반 이상 탐지는 주어진 그래프 구조 $E$, node features $X$, 그리고 부분적인 노드 라벨 $\{V_ {a}, V_ {n}\}$ 정보를 활용하여 $G$ 내의 라벨링 되지 않은 노드를 이상 또는 정상으로 분류하는 것입니다. 해당 논문은 node 이상치에 집중하며, 모든 edge는 신뢰된다고 가정합니다. 보통, 정상 노드가 이상 노드보다 훨씬 많기에 그래프 기반 이상 탐지는 불균형한 이진 노드 분류 문제로 여겨집니다.

## **2. Motivation**

 그래프 이상 탐지에서 spatial domain에 대한 분석은 이루어졌으나 spectral domain에 대한 분석이 거의 이루어지지 않았음이 해당 논문의 동기입니다. 그래프 이상 탐지를 위한 spectral domain 분석이 유효한지를 확인하기 위하여 ‘오른쪽 편이’ 현상을 정의하고 Gaussian anomaly model로 증명하며 인조적인 데이터와 실제 데이터에서 유효한지 확인하였습니다.

### 2.1 Theoretical insights of the ‘right-shift’ phenomenon

$G$ 에서 $x= (x_1, x_2, … , x_N)^T \in R^N$ 을 signal, $\hat{x}= (\hat{x}_ 1, \hat{x}_ 2, … , \hat{x}_ N)^T = U^T$ 를 $x$ 의 graph Fourier transform 이라 가정해봅시다. 
이때 $\hat{x}^2_ {k} / \sum _ {i=1} ^ N \hat{x} ^2 _i$ 를 $\lambda _k (1 \leq k \leq N)$ 에서의 spectral energy distribution 이라 합니다.  

논문의 저자는 이상치의 존재가 존재하면 spectral energy 에서의 ‘오른쪽 편이’ 현상이 나타남을 확인하였으며, 이는 spectral energy distribution이 낮은 주파수에는 적게 집중되어 있고 높은 주파수에는 많이 집중되어 있음을 의미합니다. 본문은 증명을 위해 $x$가 Gaussian distribution을 따른다고 가정합니다. (i.e. $x \sim N(\mu e_ N,\sigma^2 I_ N)$ 이때, $x$의 이상치 정도는 $\sigma \vert \mu \vert$로 표현할 수 있습니다. $x$의 이상치 정도에 따라 spectral energy distribution이 얼마나 바뀌는지를 energy ratio라 할때, 어떠한 $1 \leq k \leq N-1$에 대하여 k번째 낮은 주파수 energy ratio 를 $\eta_ k(x,L) = \frac{\sum_ {i=1}^k \hat{x}^2_ i } {\sum_ {i=1}^N \hat{x}^2_ i}$ 로 정의합니다. x의 이상치 정도가 바뀜에 따라 $\eta_k(x,L)$ 가 어떻게 바뀌는지를 알기 위하여 eigen-decomposition을 수행하면 시간이 많이 소요됩니다. 따라서 본문은 high-frequency area $S_ {high} = \frac{\sum_ {i=1}^k \lambda_k \hat{x}^2_ i } {\sum_ {i=1}^N \hat{x}^2_ i} = \frac {x^T Lx} {x^Tx}$ 를 정의내려 설명합니다. 이를 통해 모든 스펙트럼에서의 ‘오른쪽 편이’ 현상을 증명할 수 있습니다.


### 2.2 Validation on Datasets

 해당 논문에서는 $x$가 Gaussian distribution을 따르는 데이터셋과 따르지 않은 데이터셋 각각에 대하여 ‘오른쪽 편이’ 현상을 검증합니다. 첫 번째로, 인조적인 데이터셋인 Barabasi-Albert graph (Figure 1 (a)-(b))와 Minnesota road graph (Figure 1 (c)-(d)) 에서 이상치의 효과를 보입니다. 아래 그림에서 파란색 원은 spatial domain에서의 이상 노드를 나타내며 원의 크기가 클수록 이상치의 정도가 심함을 의미합니다. 그래프를 해석하면, 이상치의 정도, 즉, $\sigma$ 와 $\alpha$ 가$\alpha$ $\lambda$ $\lambda$ $\alpha$가 큼을 확인할 수 있으며 이는 2.1에서 설명한 ‘오른쪽 편이’ 현상이 보임을 의미합니다. 

![Figure1.png](https://i.ibb.co/hgYpfq9/Figure1.png)

 

두 번째로, node feature가 Gaussian distribution을 엄격하게는 따르지 않는 현실의 데이터셋에서의 ‘오른쪽 편이’ 현상을 입증합니다. 아래는 해당 4가지 데이터셋, Amazon, YelpChi, T-Finance, T-Social 의 특징과 이상치 효과에 대하여 정리한 도표입니다.

![Table1.png](https://i.ibb.co/njWJRtj/Table1.png)

아래의 표는 Amazon dataset에서 (1) 기존 그래프, (2) 모든 이상치를 없앤 그래프, (3) 임의의 같은 노드의 수를 없앤 그래프의 spectral energy를 비교한 표입니다. Figure 3의 왼쪽 그래프에서, 낮은 주파수일때, 즉, $\lambda$ 값이 작을 때, Drop-Anomaly 가 Drop-Random 보다 큰 spectral energy distribution을 가짐을 확인할 수 있으며 이는  ‘오른쪽 편이’ 현상이 있음을 나타냅니다.

![Figure3.png](https://i.ibb.co/9VCmrWV/Figure3.png)

## **3. Method**

 대부분의 해당 논문 이전의 GNN은 low-pass filter 또는 adaptive filter을 사용하였으며 이는 band-pass 와 spectral-localized 를 보장하지 못합니다. 이러한 단점을 극복하기 위하여 해당 논문에서는 Hammond’s graph wavelet theory를 기반으로 한 새로운 GNN architecture인 BWGNN를 제안합니다. Hammond’s Graph Wavelet 은 graph signal $x \in R^N$ 에 wavelets $W = (W_ {\psi_1}, W_ {\psi_2},…)$ 를 적용하여 변형시키는 것이며 이때, $\psi$ 는 “mother” wavelet 입니다. 

 Beta distribution은 종종 wavelet basis의 역할을 합니다. 이전에 Beta distribution을 사용한 기록이 없어 해당 논문에서는 Graph kernal function로 Beta distribution을 선택하여 Beta graph wavelet를 만들었고 특징을 분석하였습니다. Heat Wavelet과 Beta Wavelet 를 비교해보면, Figure 4의 왼쪽에서 볼 수 있듯이 Beta Wavelet은 low-pass filter만 있는 Heat Wavelet 과 달리 low-pass 와 band-pass를 포함한 다양한 filter type을 포함합니다. Figure 4의 오른쪽에서는 Beta Wavelet은 긍정의 반응만 보이는 Heat Wavelet 과 달리 서로 다른 채널에 대해 긍정과 부정의 효과를 둘다 보임을 확인할 수 있습니다.

![Figure4.png](https://i.ibb.co/S0FG2Nc/Figure4.png)

 위에서 설명한 Beta graph wavelet을 활용하여 BWGNN은 병렬적으로 서로 다른 wavelet kernel을 사용한 후 해당 filtering의 결과를 병합합니다. 구체적으로 BWGNN은 아래의 propagation 과정을 채택합니다.

$
Z_i = W_ {i,C-i} (MLP(X))
$

$
H = AGG([Z_0, Z_1, ..., Z_C])
$

## **4. Experiment**

### 4.1 **Experiment setup**

- 4 Dataset : T-Finance, T-Social , YelpChi, Amazon
    - T-Finance : 거래 네트워크에서의 이상 계좌를 찾는 것을 목적으로 하는 데이터셋
    - T-Social : 소셜 네트워크에서 이상 계정을 찾는 것을 목적으로 하는 데이터셋
    - YelpChi : [Yelp.com](http://Yelp.com) 에 올라온 악성 리뷰를 찾는 것을 목적으로 하는 데이터셋
    - Amazon : [Amazon.com](http://Amazon.com) 의 음악 악기 카테고리에 올라온 가짜 제품 리뷰를 찾는 것을 목적으로 하는 데이터셋
- Evaluation Metric : F1-macro, AUC
- Baselines
    - First group : MLP, SVM
    - Second group : GCN, ChebyNet, GAT, GIN, GraphSAGE, GWNN
    - Third group : GraphConsis, CAREGNN, PC-GNN
    - Fourth group : BWGNN (hetero), BWGNN (homo)

### 4.2 **Result**

아래의 표에서 확인할 수 있듯이 BWGNN은 Amazon을 제외한 dataset 에서의 최고의 성능을 보여줌을 확인할 수 있습니다.

![Table2.png](https://i.ibb.co/k6ZSy1T/Table2.png)

![Table3.png](https://i.ibb.co/yVjMqxg/Table3.png)

민감도 분석을 진행한 결과는 아래와 같습니다. 중요한 hyperparameter인 order C와 이상치 정도의 영향에 대하여 민감도 분석을 진행하였고 각각 왼쪽과 오른쪽의 결과를 보였습니다.

![Figure5.png](https://i.ibb.co/TRSBmsj/Figure5.png)

![Figure6.png](https://i.ibb.co/2kggTsk/Figure6.png)

## **5. Conclusion**

해당 논문은 그래프 이상 탐지에 대하여 설명한 후 ‘이상치 탐지를 위한 GNN에서 적절한 spectral filter을 어떻게 고를 것인가?’에 대해 답변합니다. 이를 위하여 핵심 특징인 ‘오른쪽 편이’에 대해 그래프로 증명하여 알고리즘의 필요성을 이야기한 후 Beta Wavelet Graph를 활용한 새로운 알고리즘인 BWGNN를 수식적으로 보여줍니다. 실험에서는 4가지 datasets을 활용하여 알고리즘 비교 실험을 진행하였고 BWGNN은 우수한 성능을 보였습니다. Future work로 noe anomalies에서 더 나아간 edge anomalies를 분석해볼 수 있다고 생각합니다. 수리적으로 energy distribution 을 표현한 것이 인상깊었습니다.

---

## **Author Information**

- 심윤주 (Yoonju Sim)
    - Master Student, Department of Industrial & Systems Engineering, KAIST
    - Computational Optimization, Reinforcement Learning, Transportation system

## **6. Reference & Additional materials**

- Github : https://github.com/squareRoot3/Rethinking-Anomaly-Detection
- Datasets :  [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing)