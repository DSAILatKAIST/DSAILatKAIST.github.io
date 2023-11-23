---
title:  "[ICML 2022] Rethinking Graph Neural Networks for Anomaly Detection"
permalink: 2023-11-20-Rethinking_Graph_Neural_Networks_for_Anomaly_Detection.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [ICML-22] Rethinking Graph Neural Networks for Anomaly Detection

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

Laplacian matrix L 을 $D-A$ 또는 $I-D^{-1/2}AD^{-1/2}$ 라 해봅시다. 이때, $I$는 Identity matrix 입니다. $L$ 은 $0=\lambda_1 \leq \lambda_2 \leq ... \leq \lambda_N$ 인 고유값을 갖는 대칭행렬이며, 이에 대응하는 고유벡터는 $U = (u_1,u_2,...,u_N)$ 입니다. 두 끝 점 $\lambda_1$과 $\lambda_N$을 제외하고 임의의 기준값 $\lambda_k$에 대하여 우리는 고유값을 두 개의 집합, 저주파 $\{\lambda_1, \lambda_2, ... \lambda_k\}$와 고주파 $\{\lambda_ {k+1}, \lambda_ {k+2}, ... \lambda_N\}$ 로 나눌 수 있습니다.

$G$ 에서 $x= (x_1, x_2, … , x_N)^T \in R^N$ 을 signal, $\hat{x}= (\hat{x}_ 1, \hat{x}_ 2, … , \hat{x}_ N)^T = U^Tx$ 를 $x$ 의 graph Fourier transform 이라 가정해봅시다. 
이때 $\hat{x}^2_ {k} / \sum _ {i=1} ^ N \hat{x} ^2 _i$ 를 $\lambda _k (1 \leq k \leq N)$ 에서의 spectral energy distribution 이라 합니다.  

논문의 저자는 이상치의 존재가 존재하면 spectral energy 에서의 ‘오른쪽 편이’ 현상이 나타남을 확인하였으며, 이는 spectral energy distribution이 낮은 주파수에는 적게 집중되어 있고 높은 주파수에는 많이 집중되어 있음을 의미합니다. 본문은 probabilistic anomaly model을 사용하여 이 현상을 증명합니다. 증명 과정은 다음과 같습니다. 그래프의 특징은 Gaussian distribution을 따르며 i.i.d 하다고  가정이 됩니다. (i.e. $x \sim N(\mu e_ N,\sigma^2 I_ N)$. 이때, $x$의 이상치 정도는 $\sigma / \vert \mu \vert$로 표현할 수 있습니다. $x$의 이상치 정도에 따라 spectral energy distribution이 얼마나 바뀌는지를 energy ratio라 할때, 어떠한 $1 \leq k \leq N-1$에 대하여 k번째 낮은 주파수 energy ratio 를 $\eta_ k(x,L) = \frac{\sum_ {i=1}^k \hat{x}^2_ i } {\sum_ {i=1}^N \hat{x}^2_ i}$ 로 정의합니다. $\eta_ k(x,L)$ 가 크다는 것은 에너지의 더 큰 부분이 처음 $k$개의 고유값으로 축소된다는 것을 의미합니다. 이때, 만약 $\vert \mu \vert \neq 0$ 이고 $L = D - A$ 라면, 저주파 energy ratio의 역의 기댓값 $E_x[1/\eta_k(x,L)]$ 는 이상치 정도 $\sigma / \vert \mu \vert$ 로 단조롭게 증가한다는 것을 수식으로 증명할 수 있습니다.(해당 논문의 Appendix A 참고) 아쉽게도, x의 이상치 정도가 바뀜에 따라 $\eta_k(x,L)$ 가 어떻게 바뀌는지를 알기 위하여 eigen-decomposition을 수행하여 계산하면 시간이 많이 소요됩니다. 따라서 본문은 계산이 쉽도록 고주파 영역 $S_ {high}$에 대한 정의를 내리는데 이것의 핵심은 고주파 영역이 위에서 표현한 $E_x[1/\eta_k(x,L)]$ 을 대체할 수 있다는 것입니다. 고주파 영역은 다음으로 계산할 수 있습니다: $S_ {high} = \frac{\sum_ {i=1}^k \lambda_k \hat{x}^2_ i } {\sum_ {i=1}^N \hat{x}^2_ i} = \frac {x^T Lx} {x^Tx}$. 낮은 주파수의 스펙트럼 에너지는 작은 고유값을 곱한 후엔 $S_ {high}$ 에 덜 기여하기에 우리는 $S_ {high}$의 변화를 이용하여 모든 스펙트럼에서의 '오른쪽 편이' 현상을 표현할 수 있습니다. 


### 2.2 Validation on Datasets

 해당 논문에서는 $x$가 Gaussian distribution을 따르는 데이터셋과 따르지 않은 데이터셋 각각에 대하여 ‘오른쪽 편이’ 현상을 검증합니다. 첫 번째로, 인조적인 데이터셋인 Barabasi-Albert graph (Figure 1 (a)-(b))와 Minnesota road graph (Figure 1 (c)-(d)) 에서 이상치의 효과를 보입니다. 저자는 이상 현상의 두 가지 변형을 분석합니다. (i) 이상 현상의 비율은 5%로 고정되고 이상 현상의 표준 편차는 변경되는 경우 (즉, $\sigma$ = 1, 2, 5, 20). (ii) 이상치의 표준편차는 5로 고정되고 이상치의 비율이 변경되는 경우 (즉, $\alpha$ = 0%, 1%, 5%, 20%).
 아래 그림의 상단에서 파란색 원은 spatial domain에서의 이상 노드를 나타내며 원의 크기가 클수록 이상치의 정도가 심함을 의미합니다. 그림의 하단은 $x$의 energy distribution 을 spectral domain과 이상치 정도에 따라 그린 그래프입니다. 이를 해석하면, 이상치의 정도, 즉, $\sigma$ 와 $\alpha$ 가 커질수록, $\lambda$ $\geq$ 0.5 일때의 spectral energy가 큼을 확인할 수 있으며 이는 2.1에서 설명한 ‘오른쪽 편이’ 현상이 보임을 의미합니다. 

![Figure1.png](https://i.ibb.co/hgYpfq9/Figure1.png)

두 번째로, node feature가 Gaussian distribution을 엄격하게는 따르지 않는 현실의 데이터셋에서의 ‘오른쪽 편이’ 현상을 입증합니다. 아래는 해당 4가지 데이터셋, Amazon, YelpChi, T-Finance, T-Social 의 특징과 이상치 효과에 대하여 정리한 도표입니다.

![Table1.png](https://i.ibb.co/njWJRtj/Table1.png)

아래의 표는 Amazon dataset에서 (1) 기존 그래프, (2) 모든 이상치를 없앤 그래프, (3) 임의의 같은 노드의 수를 없앤 그래프의 spectral energy를 비교한 표입니다. Figure 3의 왼쪽 그래프에서, 낮은 주파수일때, 즉, $\lambda$ 값이 작을 때, Drop-Anomaly 가 Drop-Random 보다 큰 spectral energy distribution을 가짐을 확인할 수 있으며 이는  ‘오른쪽 편이’ 현상이 있음을 나타냅니다.

![Figure3.png](https://i.ibb.co/9VCmrWV/Figure3.png)

## **3. Method**

 대부분의 해당 논문 이전의 GNN은 low-pass filter 또는 adaptive filter을 사용하였으며 이는 band-pass 와 spectral-localized 를 보장하지 못합니다. 이러한 단점을 극복하기 위하여 해당 논문에서는 Hammond’s graph wavelet theory를 기반으로 한 새로운 GNN architecture인 BWGNN를 제안합니다. Hammond’s Graph Wavelet 은 graph signal $x \in R^N$ 에 wavelets $W = (W_ {\psi_1}, W_ {\psi_2},…)$ 를 적용하여 변형시키는 것이며 이때, $\psi$ 는 “mother” wavelet 입니다. graph signal $x$ 에 $W_ {\psi_i}$를 적용하는 것은 다음과 같이 쓸 수 있습니다 : $W_ {\psi_i}(x) = Ug_i(\Lambda)U^Tx$. 이때, $g_i(\cdot)$ 은 $[0, \lambda_N]$ 에서 정의된 spectral domain의 kernal function이며, $g_i(\Lambda) = diag(g_i(\lambda))$ 입니다. 

 Beta distribution은 몇몇 논문에서 wavelet basis의 역할을 하였습니다. 하지만 이전에 Beta distribution을 그래프 데이터에 사용한 기록이 없어 해당 논문에서는 Graph kernal function로 Beta distribution을 선택하여 Beta graph wavelet를 만들었고 특징을 분석하였습니다. 해당 논문에서 제안하는 Beta wavelet transform $W_ {p,q}$ 는 다음과 같이 작성할 수 있습니다: $W_ {p,q} = U\beta^*_ {p,q}(\Lambda)U^T = \beta^*_ {p,q}(L) = \frac{(L/2)^p(I-L/2)^q}{2B(p+1,q+1)}$. 이때, $p+q = C$ 는 상수이며 Beta wavelet transform $W$ 는 $W = (W_ {0,C}, W_ {1,C-1}, ..., W_ {C,0})$ 로 총 $C+1$ 개의 Beta wavelets 으로 구성될 수 있습니다. $C$ 가 클수록 더 안좋은 공간적 집약성을 희생하여 더 나은 스펙트럼 집약성을 제공할 수 있습니다.
 Heat Wavelet과 Beta Wavelet 를 비교해보면, Figure 4의 왼쪽에서 볼 수 있듯이 Beta Wavelet은 low-pass filter만 있는 Heat Wavelet 과 달리 low-pass 와 band-pass를 포함한 다양한 filter type을 포함합니다. Figure 4의 오른쪽에서는 Beta Wavelet은 긍정의 반응만 보이는 Heat Wavelet 과 달리 서로 다른 채널에 대해 긍정과 부정의 효과를 둘다 보임을 확인할 수 있습니다.

![Figure4.png](https://i.ibb.co/S0FG2Nc/Figure4.png)

 위에서 설명한 Beta graph wavelet을 활용하여 만든 BWGNN은 병렬적으로 서로 다른 wavelet kernel을 사용한 후 해당 filtering의 결과를 병합합니다. 구체적으로 BWGNN은 아래의 propagation 과정을 채택합니다.

$
Z_i = W_ {i,C-i} (MLP(X))
$

$
H = AGG([Z_0, Z_1, ..., Z_C])
$

이때, MLP($\cdot$) 은 multi-layer perceptron, AGG($\cdot$)은 단순 집계 합수를 의미하며, $W_ {i,C-i}$ 는 우리의 wavelet kernel을 뜻합니다. BWGNN의  학습을 위하여 weighted cross-entropy loss가 사용되었으며 BWGNN의 시간복잡도는 $O(C \vert \epsilon \vert)$에 해당합니다. 

## **4. Experiment**

### 4.1 **Experiment setup**

- 4 Dataset : T-Finance, T-Social , YelpChi, Amazon
    - T-Finance : 거래 네트워크에서의 이상 계좌를 찾는 것을 목적으로 하는 데이터셋
    - T-Social : 소셜 네트워크에서 이상 계정을 찾는 것을 목적으로 하는 데이터셋
    - YelpChi : [Yelp.com](http://Yelp.com) 에 올라온 악성 리뷰를 찾는 것을 목적으로 하는 데이터셋
    - Amazon : [Amazon.com](http://Amazon.com) 의 음악 악기 카테고리에 올라온 가짜 제품 리뷰를 찾는 것을 목적으로 하는 데이터셋
- Evaluation Metric
    - F1-macro : 두 클래스의 F1 점수에 대한 비가중 평균으로, 정상 레이블과 이상 레이블 간의 불균형 비율을 무시함 
    - AUC : ROC 곡선 아래 영역
- Baselines
    - First group : 그래프 관계 없이 노드 기능만 고려
        - MLP : 활성화 함수가 있는 두 개의 선형 레이어로 구성된 Multi-layer Perceptron Network
        - SVM : RBF(Radial Basis Function) 커널을 갖춘 Support Vector Machine
    - Second group : 노드 분류를 위한 일반 GNN 모델
        - GCN : 그래프의 국지적 스펙트럼 필터의 1차 근사를 사용하는 Graph Convolutional Network
        - ChebyNet : Convolution kernal을 Chebyshev 다항식으로 제한하는 Graph Convolutional Network
        - GAT : 이웃 집계(Aggregation)을 위한 Attention 메커니즘을 사용하는 Graph Attention Network
        - GIN : Weisfeiler-Lehman(WL) 그래프 동형성 테스트에 연결되는 GNN 모델
        - GraphSAGE : 고정된 이웃 노드 샘플 수를 기반으로 하는 GNN 모델
        - GWNN : wavelet 변환을 생성하기 위해 heat kernal을 사용하는 Graph Wavelet Neural Network
    - Third group : 그래프 기반 이상 탐지를 위한 최신 기법
        - GraphConsis : 그래프 이상 탐지에서 문맥, 기능 및 관계 불일치 문제를 해결하는 heterogeneous GNN
        - CAREGNN : 위장 및 강화 학습에 대한 세 가지 고유 모듈을 통해 집계 프로세스를 향상하는 위장 방지 GNN
        - PC-GNN : 리샘플링을 통해 그래프 기반 사기 탐지의 클래스 불균형 문제를 해결하는 GNN 기반 불균형 학습 방법
    - Fourth group : 해당 논문에서 제안하는 모델
        - BWGNN (homo) : 모든 종류의 edges를 동일하게 취급하는 Beta Wavelet GNN
        - BWGNN (hetero) : 각 관계에 대해 개별적으로 그래프 전파를 수행하고 maximum pooling을 적용한 Beta Wavelet GNN

### 4.2 **Result**

첫 번째 표는 training 비율이 1%와 40%인 YelpChi와 Amazon에서 비교된 모든 알고리즘의 실험 결과입니다.
![Table2.png](https://i.ibb.co/k6ZSy1T/Table2.png)

두 번째 표는 training 비율이 다른 T-Finance 및 T-Social 데이터셋에 대한 실험 결과 및 전체 훈련 시간입니다.
![Table3.png](https://i.ibb.co/yVjMqxg/Table3.png)

결과를 분석하면, 일반적으로 BWGNN은 PC-GNN이 최고의 AUC 점수를 얻는 Amazon(1%)을 제외한 모든 데이터세트에서 최고의 성능을 보입니다. 다중 관계 그래프가 있는 두 데이터셋의 경우 BWGNN(Hetero)은 YelpChi에서 더 나은 성능을 발휘하고 BWGNN(Homo)은 Amazon에서 더 나은 성능을 발휘합니다. GraphConsis, CAREGNN 및 PC-GNN은 그래프 기반 이상 탐지를 위한 세 가지 최첨단 방법인 반면, BWGNN은 훨씬 짧은 훈련 시간으로 이들보다 훨씬 뛰어난 성능을 발휘합니다. 추가적으로, 그래프 구조가 무시되더라도 MLP와 SVM은 일부 데이터셋에서 비슷한 성능을 달성할 수 있음을 알 수 있습니다.

민감도 분석을 진행한 결과는 아래와 같습니다. 중요한 hyperparameter인 order C와 이상치 정도의 영향에 대하여 민감도 분석을 진행하였고 다음의 결과를 보였습니다.

![Figure5.png](https://i.ibb.co/TRSBmsj/Figure5.png)
Beta Wavelet 은 $L$의 $C$-order 다항식이고 각 노드의 $C$-hops에 국한되어 있으므로 C-order은 BWGNN에서 중요한 hyperparameter입니다. 그림 5는 C를 1에서 5로 변경할 때 두 데이터셋에 대한 BWGNN의 F1-macro 및 AUC 점수를 나타냅니다. T-Social에서는 C가 높을수록 성능이 향상되는 반면, T-Finance에서는 C $\geq$ 2에 대한 결과에 큰 차이가 없습니다. 

![Figure6.png](https://i.ibb.co/2kggTsk/Figure6.png)
그림 6은 T-Finance(1%)에서 BWGNN, ChebyNet 및 CAREGNN의 F1-macro 및 AUC 점수를 다양한 이상 수준으로 비교합니다. $\sigma$가 증가하면 이상 현상을 더 잘 구별할 수 있으므로 세 가지 모델 모두 더 나은 성능을 발휘합니다. 그 중에서 BWGNN은 가장 빠르게 성장하는 알고리즘이며 $\sigma$ = 4에서 99% F1-macro에 도달합니다. $\alpha$ 가 변화할 때 BWGNN은 일관되게 다른 방법보다 성능이 뛰어나며 다양한 이상 정도에 대해 견고합니다.

## **5. Conclusion**

해당 논문은 그래프 이상 탐지에 대하여 설명한 후 ‘이상치 탐지를 위한 GNN에서 적절한 spectral filter을 어떻게 고를 것인가?’에 대해 답변합니다. 이를 위하여 핵심 특징인 ‘오른쪽 편이’에 대해 여러 데이터셋으로 검증하여 알고리즘의 필요성을 이야기한 후 Beta Wavelet Graph를 활용한 새로운 알고리즘인 BWGNN를 수식적으로 보여줍니다. 알고리즘 비교 실험에서는 4가지 datasets을 활용하였고 BWGNN은 우수한 성능을 보였습니다. Future work로 node anomalies에서 더 나아간 edge anomalies를 분석해볼 수 있다고 생각합니다. 수리적으로 energy distribution 을 표현한 것이 인상깊었습니다.

## **Author Information**

- 심윤주 (Yoonju Sim)
    - Master Student, Department of Industrial & Systems Engineering, KAIST
    - Interest: Computational Optimization, Reinforcement Learning, Transportation system

## **6. Reference & Additional materials**

- Github : https://github.com/squareRoot3/Rethinking-Anomaly-Detection
- Datasets :  [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing)