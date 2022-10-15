---
title:  "[ICML 2022] How Powerful are Spectral Graph Neural Networks"
permalink: How_Powerful_are_Spectral_Graph_Neural_Networks.html
tags: [reviews]
---

# **How Powerful are Spectral Graph Neural Networks** 

이 리뷰에서 소개하는 논문 'How Powerful are Spectral Graph Neural Networks'는 [이번 ICML 2022에서 Spotlight로 선정된 논문](https://icml.cc/virtual/2022/spotlight/17796) 중 하나입니다. 이 논문에서는 Spectral GNN의 표현력에 대한 분석 및 이를 기반으로 한 새로운 Spectral GNN인 'JacobiConv'를 소개하고 있습니다.

<br/>

## **1. Introduction**  

Message Passing Framework를 활용하여 이웃한 node의 정보를 aggregate 함으로써 node들의 표현(representation)을 학습하는 Graph Neural Network(GNN)는, 그동안의 Graph Representation Learning 방법론들 가운데 여러 Downstream Task에서 State-of-the-art(SOTA) 성능을 보여줬습니다.

그 한 갈래인 Spectral GNN은, Spatial한 그래프 신호(graph signal)를 Graph Laplacian을 활용해 Spectral domain으로 변환하여 필터링하고  필터링된 신호를 다시 Spatial domain으로 가져와 prediction을 수행합니다. GCN[2], GAT[3]과 같이 Popular한 모델이 등장하기 이전부터도 ChebyNet[4]과 같은 Spectral GNN이 연구되었고, 그중 GCN의 경우 ChebyNet에서의 Spectral 필터를 단순화한 모델입니다.

이외에도 이 논문에서 언급되는 여러 Spectral GNN 모델들이 등장하지만, 저자들은 이러한 Spectral GNN 모델의 표현력(expressive power)에 대해 분석하고 연구한 논문이 없었음을 지적합니다. 저자들은 이 논문을 통해 Spectral GNN 모델의 표현력에 대해 이론적인 분석을 제시하고, 이를 바탕으로 'JacobiConv'라는  Spectral GNN 모델을 제안합니다.

이 논문의 Contribution은 아래와 같이 정리할 수 있습니다.
 1. 이 논문에서는 비선형성(non-linearlity)이 없는, 간단한 형태의 Linear Spectral GNN조차도 강력한 표현력이 있음(universal함)을 이론적으로 보이며, 그런 표현력을 갖추기 위한 조건을 제시하고 이에 대해 분석합니다.
 2. 또한, Linear Spectral GNN의 Universality 조건과 그래프 동형 테스트(Graph Isomorphism Test; GI Test)와의 연관성에 대해서도 분석합니다. 이런 GI Test를 활용한 GNN의 표현력 분석은 Spatial한 GNN에서 다뤄진 바 있습니다[5].
 3. 여러 Spectral GNN의 실험적인 성능 차이를 최적화 관점에서 분석하고, 이를 통해 그래프 신호 Density에 맞는 basis function으로 그래프 신호 필터를 구성하는 것이 중요함을 보여줍니다.
 4. 위의 분석을 기반으로 JacobiConv이라는 Spectral GNN 모델을 제시합니다. JacobiConv은 비선형성 없이도 synthetic 및 real-world dataset에서 다른 Spectral GNN baseline들을 상회하는 성능을 보여줍니다.

논문에서 내용을 서술하는 흐름이 자연스럽기 때문에, 이 리뷰는 논문의 내용 순서를 그대로 따라 서술되어 있습니다. 이 리뷰를 읽으시면서 좀 더 자세하고 엄밀한 부분이 필요하다면 논문을 참고하셔도 좋습니다.

<br/> 
   
 *(주) 본문에 들어가기에 앞서, 이 리뷰는 논문의 핵심적인 개념을 위주로 서술한 것임을 밝힙니다. 이 논문은 이론적인 분석이 주가 되는 논문이기에, 이 논문에 있는 모든 Theorem, Proposition 등을 충분히 이해하기 위해서는 Specral GNN에서 포괄하고 있는 많은 배경 지식을 필요로 합니다. 다만 이 리뷰를 작성하는 저도 그러한 배경 지식이 충분하지 않기에, 이 논문에서 말하고자 하는 essential한 부분에 대해서만 다루고자 합니다. 부족한 부분은 Revision 기간에 더욱 보완하도록 하겠으니, 그때까지 기다려 주시면 정말 감사드리겠습니다.*

<br/> 

## **2. Preliminaries**  

이 Section에서는 논문 본문에서 쓰인 Notation을 그대로 서술하도록 하겠습니다. 아래는 matrix의 행, 열에 대한 Notation입니다.

$$\forall M \in \mathbb{R}^{a\times b}: M_{i}=\mathrm{row_{i}}(M), M_{:i}=\mathrm{col_{i}}(M)$$

그리고, 주어진 node $i\in\mathbb{V}$에 대해서 그 이웃을 $N(i)$로 표기합니다.

아래는 matrix의 condition number의 정의입니다. 이 개념은 전술했던 Contribution 3번에서의 분석과 관련이 있습니다. 여기서 $\lambda_{max}$는 matrix의 Maximum Eigenvalue, $\lambda_{min}$은 matrix의 Minimum Eigenvalue를 의미합니다.
$$\kappa(M)=\frac{|\lambda_{max}|}{|\lambda_{min}|}$$

이때, 주어진 matrix $M$이 singular(=not invertible; inverse가 존재하지 않는 경우)라면 $\kappa(M)=+\infty$이고, 이는 matrix의 모든 eigenvalue가 non-zero 값을 갖는 것이 matrix의 invertiblility와 동치이기 때문입니다. [6]

 *(주) 다만 위 정의의 경우 오류가 있는 것 같습니다.* $|\lambda | _{max}$, $|\lambda | _{min}$ *가 맞는 표기이지 않을까 싶습니다.*

아래는 Graph와 관련된 Notation입니다. 기본적으로 주어진 Graph는 undirected입니다. $\mathcal{G}=(\mathbb{V}, \mathbb{E}, X)$는 주어진 Graph이고, 여기서 
$$\mathbb{V}=\{1,2,\cdots,n\},\ \mathbb{E}\subset \mathbb{V}\times\mathbb{V},\ X\in\mathbb{R}^{n\times d}$$

는 각각 Node set, Edge set, node feature matrix입니다.

$A, D$를 각각 Adjacency, Degree matrix라고 하면, normalized adjacency는 $\hat{A}=D^{-1/2}AD^{-1/2}$이고 symmetric normalized graph Laplacian은 $\hat{L}=I-\hat{A}$입니다. Graph Laplacian은 Real symmetric이기에 orthogonally diagonalizable하고, 따라서 아래와 같이 Eigen-decomposition할 수 있습니다.
$$\hat{L}=U\Lambda U^{T}$$

U는 $i^{\mathrm{th}}$ column이 $\hat{L}$의 $i^{\mathrm{th}}$ eigenvalue에 해당하는 eigenvector인 orthogonal matrix이고, $\Lambda$는 eigenvalue들을 diagonal entry들로 갖는 diagonal matrix입니다.


### **2.1. Graph Isomorphism**

이 Section에서는 Graph Isomorphism에 대해 간략하게 다룹니다.

Graph Isomorphism은 중요한 개념이긴 하나, 이 리뷰에서는 Theorem, proposition의 증명을 상세히 다루지 않고 그 안에 담긴 의미에 대해서만 다룰 예정이기에 논문 본문에서 서술한 것 대신, 널리 알려진 정의[7]에 대해서 서술하도록 하겠습니다.

두 graph $\mathcal{G_1}=(\mathbb{V_1}, \mathbb{E_1}, X_1),\ \mathcal{G_2}=(\mathbb{V_2}, \mathbb{E_2}, X_2)$에 대해 bijective(1 to 1 correspondence; 일대일대응) mapping $f:\mathbb{V_1}\rightarrow\mathbb{V_2}$가 존재해서, $(i,j)\in\mathbb{E_1}$인 임의의 두 node $i, j\in\mathbb{V_1}$의 mapped node $f(i),f(j)\in\mathbb{V_2}$가 $(f(i),f(j))\in\mathbb{E_2}$일 때 두 graph $\mathcal{G_1},\mathcal{G_2}$를 **isomorphic**하다고 하고, $f$를 **isomorphism**이라고 부릅니다.

간단하게 말하자면, 두 graph의 구조가 같은 것을 의미합니다.

### **2.2. Graph Signal Filter and Spectral GNNs**

이 Section에서는 Graph Signal Filter와 Spectral GNN의 개념, 그리고 논문에서 주로 다루는 Linear Spectral GNN(linear GNN in original paper)에 대해 서술합니다. 그리고 Filter의 표현력에 대한 개념인 _Polynomial-Filter-Most-Expressive_(PFME)와 _Filter-Most-Expressive_(FME)에 대해서도 소개하겠습니다.

**Graph Fourier Transform**의 정의는 논문에서 정의된 바와 같이, (Shuman et al., 2013)[8]의 정의를 따릅니다.
Signal $X\in\mathbb{R}^{n\times d}$의 Graph Fourier Transform은
$$\tilde{X}=U^{T}X\in\mathbb{R}^{n\times d}$$

로 정의하며, **inverse transform**은
$$X=U^{T}\tilde{X}$$

와 같이 정의합니다. 여기서 $U$의 $i^{\mathrm{th}}$ column은 eigenvalue $\lambda_{i}$에 해당하는 frequency component(eigenvector)입니다.

Eigenvalue $\lambda$에 해당하는 eigenvector를 $U_{:\lambda}^{T}$라고 하면, frequency $\lambda$에 해당하는 $X$의 frequency component를 $\tilde{X_{\lambda}}=U_{:\lambda}^{T}X$로 정의합니다.  
이때, $\tilde{X_{\lambda}}\neq\mathbb{0}$라면 $X$가 $\lambda$ frequency component를 갖고 있다고 정의합니다. 그렇지 않은 경우, $\lambda$ frequency component가 $X$에서 missing되었다고 정의합니다.

Graph Fourier Transform과 원래 Fourier Transform의 연관성은 주어진 Signal(Graph에서는 Node feature $X$)을 Frequency(Graph에서는 Laplacian $\hat{L}$의 Eigenvalue $\lambda$) domain으로 transform한다는 점에서 동일합니다.

또한 Fourier Transform의 경우 주어진 signal을 Function space에서의 orthonormal basis를 이용해 변환하는데, Graph Fourier Transform의 경우 주어진 signal을 vector space의 orthonormal basis인 eigenvector를 이용해 변환한다는 점에서 연관성이 있습니다.

이 이상의 Graph Fourier Transform에 대한 자세한 서술은 이 리뷰의 범위를 벗어나므로 생략하도록 하겠습니다.

 *(주) 이 리뷰에서 function space의 orthonormal basis에 대해서 자세히 다루는 것은 훨씬 심도깊은 논의가 필요하기 때문에 생략하도록 하겠습니다. 이와 관련하여 좀 더 알고 싶으신 분들은, Elias M. Stein and Rami Shakarchi의 Real Analysis: Measure Theory, Integration, and Hilbert Spaces (Princeton Lectures in Analysis)를 보시는 것이 좋을 것 같습니다. 또 Graph Fourier Transform에 대해서 더 자세히 알고 싶으시다면 (Shuman et al., 2013)[8]을 참고하시면 좋을 것 같습니다.*

이젠 Graph Signal Filter에 대해서 서술하도록 하겠습니다. Graph Signal Filter는 signal의 frequency component를 필터링하는 역할을 수행합니다.

Filter $g:[0,2]\rightarrow\mathbb{R}$는 $g(\lambda)$ 값을 각각의 frequency component에 곱해주는 방식으로 필터링을 수행합니다. Signal $X$에 spectral filter $g$를 적용하는 것은 다음과 같이 정의합니다.
$$Ug(\Lambda)U^{T}X$$

 *(주) filter의 정의역이 [0,2]인 것은 Normalized Graph Laplacian의 성질에 기인합니다.[9, Lemma 1.7.]*

여기서 filter $g$는 $\Lambda$에 element-wise하게 적용됩니다. Filter를 parametrize하기 위해, $g$는 아래와 같이 degree $K$의 polynomial로 설정합니다.
$$g(\lambda):=\sum_{k=0}^{K}{\alpha_{k}\lambda^{k}}$$

여기서 $g(\hat{L})$을
$$g(\hat{L})=\sum_{k=0}^{K}{\alpha_{k}\hat{L}^{k}}$$

로 정의하면, 필터링 과정은 아래와 같이 표현 가능합니다.
$$Ug(\Lambda)U^{T}X=\sum_{k=0}^{K}{\alpha_{k}U\Lambda^{k}U^{T}X}=\sum_{k=0}^{K}{\alpha_{k}\hat{L}^{k}X}=g(\hat{L})X$$

ChebyNet 등 여러 널리 알려진 spectral GNN의 filter form은 아래 표에 정리되어 있습니다.

<p align="center"><img width="700" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Table_5.png"></p>

일반적으로, spectral-based GNN은 아래와 같은 form으로 정리할 수 있습니다.
$$Z=\phi(g(\hat{L}))\psi(X)$$

여기서 $Z$는 prediction, $\phi, \psi$는 Multi-Layer Perceptron(MLP)와 같은 함수입니다.

이때, spectral GNN의 filter가 그 어떤 polynomial filter function이라도 근사할 수 있다면, 그 GNN이 **Polynomial-Filter-Most-Expressive(PFME)** 하다라고 정의하고, arbitrary한 real-valued filter function을 근사할 수 있다면 **Filter-Most-Expressive(FME)** 라고 정의합니다.

이러한 PFME, FME property는 spectral GNN의 표현력에 있어서 중요한 성질인 것으로 보입니다. Frequency component를 scaling 함으로써 말 그대로 필터링을 해주는 Filter의 역할을 생각해봤을 때, arbitrary한 filter을 학습할 수 있느냐(=FME)는 spectral GNN의 표현력(주어진 두 node를 구별하는 능력)에 분명 큰 역할을 할 것이라고 생각할 수 있습니다.

이 논문에서는 $\phi, \psi$가 linear한 경우에 초점을 두고 있기 때문에, 'Linear GNN', linear한 spectral GNN을 아래와 같이 정의합니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Definition_2_1.png"></p>

아래의 Proposition 2.2는 Linear GNN이 PFME, 즉 충분히 강한 표현력을 갖고 있고, General한 spectral GNN의 표현력의 Lower bound가 됨을 서술하고 있습니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Prop_2_2.png"></p>

비록 길었지만, 이 논문의 중요 개념을 이해하는데에 필요한 부분은 모두 짚어보았습니다. 나머지는 분석에 앞서, 이 논문에서 가정하고 있는 부분에 대한 서술입니다.

우선, 이 논문에서는 Fixed graph, fixed node features에서 오직 node property prediction task만 처리한다고 가정합니다.

위와 같은 Setting에서는 PFME=FME가 성립하게 됩니다. 왜냐하면 PFME한 GNN이 비록 polynomial filter function만 표현할 수 있지만, fixed graph setting에서는 eigenvalue $\lambda$가 discrete하기 때문에 arbitrary filter function을 충분히 근사할 수 있는 interpolation polynomial을 얻을 수 있고[10, Theorem 3.1., 3.3.], 이 polynomial은 PFME GNN으로 표현 가능하기 때문입니다. 이를 위해, 추가적으로 Linear GNN의 Polynomial Filter가 충분히 큰 degree K를 가지도록 설정합니다.

<br/> 

## **3. Analyses: The Expressive Power of Linear GNNs**  

이 Section에서는 세 가지 조건 아래에서 linear GNN이 Universal하다는 것을 증명합니다. 이어지는 3개의 sub-section에서는 세 가지 Universality 조건을 분석하여, spectral GNN이 얼마나 강력한 표현력을 가질 수 있는 지에 대해 다룹니다.

나머지 sub-section에서는 Graph Isomorphism과의 연관성(3.4.), spectral GNN에서 Non-linearlity의 역할(3.5.)에 대해 분석합니다.

본문에 들어가기에 앞서, Linear GNN $Z=g(\hat{L})XW$의 두 핵심 Component에 대해 다시 한 번 짚어보겠습니다.
 1. **Linear Transformation** $W$: $XW=U(\tilde{X}W)$라는 사실은 spatial domain에서의 선형 변환이 spectral domain에서의 선형 변환을 의미함을 보여줍니다.
 2. **Filter** $g(\hat{L})$: $g(\hat{L})X=U(g(\Lambda)\tilde{X})$이기에, 우리는 filter가 frequency component를 scaling해주는 역할이라는 것을 알 수 있습니다.

이 논문의 핵심인, Linear GNN의 Universal Theorem은 아래와 같습니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Thm_4_1.png"></p>

따라서, Universality를 얻기 위해서는 아래의 세 가지 조건이 필요합니다.
 1) 1-dimensional prediction
 2) Graph Laplacian has no multiple eigenvalues
 3) Node feature has no missing frequency components

이 조건은 linear GNN의 표현력의 소위 'Bottleneck'이라 할 수 있습니다. 따라서 아래 sub-section들에서는 이 세 가지 Bottleneck에 대해서 자세하게 분석합니다.

### **3.1. About Multi-dimensional Prediction**

아래 Proposition을 통해, 논문에서는 Linear GNN이 multi-dimensional prediction에 대해서는 Universal하지 않다는 것을 서술하고 있습니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Prop_4_2.png"></p>

Proposition 4.2.의 조건에서 $X$ is not a full row-rank matrix, 즉 $\mathrm{rank}(X) < n $이라는 조건은 자명합니다. 보통의 경우 output dimension 값은 node 갯수 $n$보다 작은 값을 갖기 때문입니다.

Universal Theorem을 보면 Linear GNN은 1-dimensional prediction만을 산출하는 경우에는 충분히 강력하지만, 위의 Propsition 때문에 Multiple channel을 갖는 prediction을 산출하기 위해서는 각기 다른 polynomial filter를 필요로 하게 됩니다.

이에 대해서는 Figure 1에 묘사되어 있는 Toy Example을 보도록 하겠습니다. (b), (c)를 보면, (a)에서 주어진 Node feature을 이용해 여러 dimension의 output을 만들기 위해서는 서로 다른(하나는 High-pass, 다른 하나는 Low-pass) filter가 필요하다는 것을 서술하고 있습니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Figure_1.png"></p>

이 Toy Example을 통해서 우리는 논문에서 서술하고 있는 위의 내용 이외에도, GNN의 표현력, Universality에 있어서 arbitary filter을 근사하는 능력인 FME property가 왜 중요한 지에 대해서 생각해볼 수 있습니다. 만약 Model에서 사용하는 filter가 특정 filter를 근사할 수 없다면, 이는 특정 prediction 값을 산출할 수 없다는 것이고 다시 말해 universal하지 못하게 된다는 것을 의미합니다.

이 논문에서는 이런 Multi-dimensional prediction 문제를 각 output channel마다 다른 polynomial coefficient parameter을 사용하는 것으로 해결할 수 있다고 서술하고 있습니다.

### **3.2. About Multiple Eigenvalue**

Graph Laplacian이 multiple eigenvalue을 갖는다는 것은 두 개의 frequency component가 같은 eigenvalue $\lambda$를 갖는 경우이며, 이는 다른 frequency component가 같은 scale $g(\lambda)$로 scaling 된다는 것을 의미합니다.

다시 말해, 서로 다른 두 frequency component에 대해서 Model이 다르게 필터링할 수 없다는 것입니다. 우린 이와 같은 경우가 Linear GNN의 표현력을 저해할 수 있다고 생각할 수 있습니다.

이러한 Multiple eigenvalue는 주어진 graph의 topology, 즉 구조와 연관되어 있습니다.

하지만 우리는 아래에서 node feature을 갖는 real-world graph의 경우 이런 multiple eigenvalue가 유의미하게 적은 구조를 갖고 있다는 것을 확인할 수 있습니다.

### **3.3. About Missing Frequency Components**

위에서 전술했듯이, Filter는 frequency component를 scaling해주는 역할만을 수행합니다. 만약 node feature의 어느 frequency component가 missing되었다면, prediction에 해당 frequency component가 반영되지 못하게 됩니다.

아래 Figure 2에는 missing frequency component가 생기는 Toy graph를 다루고 있습니다.  
 *(주) Figure 2에 있는 1-dim node feature와 graph structure을 이용해 계산해보면, 왼쪽의 node feature로는 frequency, 즉 eigenvalue=2에 해당하는 frequency component가 0이 됩니다.*

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Figure_2.png"></p>

이 Missing frequency component 문제는 Graph structure과 node feature 둘 다 영향을 끼치고, 그렇기에 다루기 어려운 문제입니다.

하지만 Multiple eigenvalue 문제처럼 이 문제 역시 node feature을 갖는 real-world graph에서는 보기 어렵습니다. 아래 Table은 10개의 benchmark dataset에서의 multiple eigenvalue 비율과 missing frequency component의 수를 정리한 것입니다.

<p align="center"><img width="700" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Table_7.png"></p>

각 output channel마다 다른 polynomial filter를 이용하는 방법과, 위와 같은 real-world dataset의 특성을 통해 우리는 Linear GNN의 Universality를 위한 세 가지 조건이 실전에서 쉽게 만족될 수 있음을 알 수 있습니다.

### **3.4. About the Connection to Graph Isomorphism**

Spatial GNN의 표현력에 대해 분석한 논문[5]에서는 GI test를 활용해 분석하였습니다. 이와 비슷하게 이 논문에서도 Universality 조건과 Graph Isomorphism의 연관성에 대해 분석합니다.

Graph Isomorphism Test 기법으로 언급이 되는 것이 바로 1-dimensional Weisfeiler-Lehman(1-WL) test입니다. 1-WL test는 주어진 두 graph가 isomorphic한지 판별하는 알고리즘으로, 웬만한 non-isomorphic graph들을 구별할 수 있습니다. 보다 자세한 내용은 이 [링크](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/)를 참조하시면 좋을 것 같습니다.

이 논문에서는 먼저, $K+1$ iteration 1-WL test가 구별할 수 없는 node pair는 degree $K$ polynomial filter를 갖는 Linear GNN도 구별할 수 없다는 것을 아래 Proposition을 통해 서술합니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Prop_4_3.png"></p>

Proposition 4.3.은 Linear GNN의 표현력 역시 Spatial GNN 처럼[5] 1-WL test에 의해 Bound된다는 것을 의미합니다.

하지만, 우리는 Universal Theorem을 통해 Linear GNN이 각기 다른 node들에 대해, 그 node들이 isomorphic한 지와 상관 없이 서로 다른 prediction을 산출할 수 있는 표현력을 갖고 있다는 것을 알고 있습니다. 또한, 1) 1-WL test는 몇몇 non-isomorphic한 node들을 구별하지 못하며, 2) 1-WL test의 경우 isomorphic한 node들에 대해 같은 label을 산출한다는 것 역시 알려져 있는 바입니다. 이렇듯 모순되어 보이는 두 사실은 Universality Condition 2와 3이 만족되면 1-WL test 역시 충분히 Powerful하다는 것(모든 non-isomorphic node를 구별할 수 있음)과 graph가 isomorphic한 node를 가질 수 없다는 것을 보여주는 결과릍 통해 해소되며, 그 결과는 아래 Corollary 4.4.와 Theorem 4.5., Theorem 4.6.에 정리되어 있습니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Corr_4_4.png"></p>

Corollary 4.4.는 Universal Theorem과 Proposition 4.3.을 통해 유도되는 결과입니다. 두 조건 아래에서 1-WL test 역시 충분히 Powerful하다는 것을 보여줍니다.

아래 두 theorem들은 두 조건이 Graph와 node feature을 제약한다는 것을 보여줍니다.

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Thm_4_5.png"></p>

<p align="center"><img width="500" src="/images/How_Powerful_are_Spectral_Graph_Neural_Networks/Thm_4_6.png"></p>

따라서, 우리는 Universal Theorem의 조건들이 Graph Topology와 Node feature이 제약되어, 1-WL test가 결국 linear GNN의 표현력을 Bound하고 있다는 것을 뒷받침합니다. 위의 결과들을 통해서 우리는 Universality 측면에서의 spectral GNN의 표현력과 1-WL test 측면에서의 spatial GNN의 표현력 간 연결고리를 얻게 됩니다.

### **3.5. About the Role of Non-linearlity**

우리는 앞선 결과들을 통해서 Linear GNN이 충분히 강력한 표현력을 갖고 있음을 알게 되었습니다. 그럼에도 non-linearlity는 SOTA 성능의 GNN에서 활용되고 있습니다. 이 sub-section에서는 non-linearlity가 spectral GNN에서 어떤 역할을 하는지 분석합니다.

Spectral GNN의 General form $Z=\phi(g(\hat{L}))\psi(X)$을 보면, Non-linearlity는 서로 다른 frequency component를 서로 transform하는 것이라고 정리할 수 있습니다.

$\sigma$를 spatial signal $X$에 element-wise하게 적용되는 non-linearlity activation이라고 하고, spectral signal $\tilde{X}$에의 영향 $\sigma '$을 보면, 
$$


아래 Figure 4를 통해, 이러한 해석을 뒷받침할 수 있습니다.



<br/> 

## **4. Methodology-JacobiConv**  

이 Section에서는 Polynomial Filter을 구성하는 Basis function 선택의 영향에 대해, Optimization 관점에서 분석합니다. 그리고 이를 바탕으로 논문에서 제안한 JacobiConv 모델에 대해 다룹니다.

<br/>

## **5. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

### **Result**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  

<br/> 

## **6. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

이 논문에서 가장 아쉬운 부분은 PFME/FME property에 대해 자세히 서술하지 않은 점입니다. 앞의 Section에서 전술했듯 Spectral GNN의 표현력은 spatial GNN에서 표현력 분석[5]에서 그랬던 것처럼 주어진 두 node를 구별할 수 있느냐 없느냐로 서술되는데(linear spectral GNN이 Universal하다는 것을 통해), 위에서 정의된 PFME, FME 성질들이 이러한 GNN의 표현력과 어떻게 연관되어 있는지에 대해서는 논문에서 직접적인 이론을 통해서 설명하지는 않았습니다. 다만, Polynomial Filter의 basis 선택이 Empirical한 성능에 중요하다는 부분을 지적하는 부분이나, [링크](https://icml.cc/virtual/2022/spotlight/17796)의 발표자료에 있는 'same expressive power'과 같은 맥락을 통해서 간접적으로는 PFME, FME property가 표현력에 영향을 미치지 않을까라고 추측해볼 수 있습니다. 그럼에도, 이 논문이 spectral GNN의 표현력을 분석하는 첫 논문이라는 점을 생각해보면 아쉬운 대목입니다. Non-PFME/non-FME spectral GNN의 표현력이 약하다와 같은 분석이 있었다면 논문의 컨텐츠가 더더욱 풍성했을 것 같아 더더욱 아쉬움이 남습니다. 

<br/> 

---  
## **Author Information**  

* Xiyuan Wang  
    * Institute for Artificial Intelligence, Peking University
* Muhan Zhang
    * Institute for Artificial Intelligence, Peking University
    * Beijing Institute for General Artificial Intelligence

<br/> 

## **7. Reference & Additional materials**   
The Official Implementation은 [여기](https://github.com/GraphPKU/JacobiConv)에서 확인 가능합니다.
 1. Xiyuan Wang and Muhan Zhang. [_How Powerful are Spectral Graph Neural Networks_](https://arxiv.org/abs/2205.11172). ICML, 2022.
 2. Thomas N. Kipf and Max Welling. _Semi-Supervised Classification with Graph Convolutional Networks_. ICLR, 2017.
 3. Petar Veličković et al. _Graph Attention Networks_. ICLR, 2018.
 4. Michaël Defferrard et al. _Convolutional neural networks on graphs with fast localized spectral filtering_. NeurIPS, 2016.
 5. Keyulu Xu et al. _How Powerful are Graph Neural Networks?_ ICLR, 2019.
 6. [_Eigenvalues and eigenvectors_](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors). Wikipedia, 2022.
 7. [_Graph Isomorphism_](https://en.wikipedia.org/wiki/Graph_isomorphism). Wikipedia, 2022.
 8. David I Shuman et al. [_The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains_](https://ieeexplore.ieee.org/document/6494675). IEEE Signal Process Magazine, 2013.
 9. Fan R. K. Chung. _Spectral Graph Theory_. Americal Mathematical Society, 1996.
 10. Richard Burden and J. Douglas Faires. _Numerical Analysis_. Cengage Learning, 2005.
 11. Stephen Boyd and Lieven Vandenberghe. _Convex Optimization_. Cambridge University Press, 2009.
 12. 

