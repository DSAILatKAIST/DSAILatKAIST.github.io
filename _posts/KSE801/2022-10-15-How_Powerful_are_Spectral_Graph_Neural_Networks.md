---
title:  "[ICML 2022] How Powerful are Spectral Graph Neural Networks"
permalink: How_Powerful_are_Spectral_Graph_Neural_Networks.html
tags: [reviews]
---

# **How Powerful are Spectral Graph Neural Networks** 
 

## **1. Introduction**  

Message Passing Framework를 활용하여 이웃한 node의 정보를 aggregate 함으로써 node들의 표현(representation)을 학습하는 Graph Neural Network(GNN)는, 그동안의 Graph Representation Learning 방법론들 가운데 여러 Downstream Task에서 State-of-the-art(SOTA) 성능을 보여줬습니다.

그 한 갈래인 Spectral GNN은, Spatial한 그래프 신호(graph signal)를 Graph Laplacian을 활용해 Spectral하게 필터링하고 필터링된 신호를 다시 Spatial domain으로 가져와 prediction을 수행합니다. GCN[2], GAT[3]과 같이 Popular한 모델이 등장하기 이전부터도 ChebyNet[4]과 같은 Spectral GNN이 연구되었고, 그중 GCN의 경우 ChebyNet에서의 Spectral 필터를 단순화한 모델입니다.

이외에도 이 논문에서 언급되는 여러 Spectral GNN 모델들이 등장하지만, 저자들은 이러한 Spectral GNN 모델의 표현력(expressive power)에 대해 분석하고 연구한 논문이 없었음을 지적합니다. 저자들은 이 논문을 통해 Spectral GNN 모델의 표현력에 대해 이론적인 분석을 제시하고, 이를 바탕으로 JacobiConv라는 Spectral GNN 모델을 제안합니다.

이 논문의 Contribution은 아래와 같이 정리할 수 있습니다.
 1. 이 논문에서는 비선형성(non-linearlity)이 없는, 간단한 형태의 Linear Spectral GNN조차도 강력한 표현력이 있음(universal함)을 이론적으로 보이며, 그런 표현력을 갖추기 위한 조건을 제시하고 이에 대해 분석합니다.
 2. 또한, Linear Spectral GNN의 Universality 조건과 그래프 동형 테스트(Graph Isomorphism Test; GI Test)와의 연관성에 대해서도 분석합니다. 이런 GI Test를 활용한 GNN의 표현력 분석은 Spatial한 GNN에서 다뤄진 바 있습니다[5].
 3. 여러 Spectral GNN의 실험적인 성능 차이를 최적화 관점에서 분석하고, 이를 통해 그래프 신호 Density에 맞는 basis function으로 그래프 신호 필터를 구성하는 것이 중요함을 보여줍니다.
 4. 위의 분석을 기반으로 JacobiConv이라는 Spectral GNN 모델을 제시합니다. JacobiConv은 비선형성 없이도 synthetic 및 real-world dataset에서 다른 Spectral GNN baseline들을 상회하는 성능을 보여줍니다.

논문의 서술 흐름이 좋기 때문에, 이 리뷰는 논문의 순서를 그대로 따라 서술되어 있습니다. 이 리뷰를 읽으시면서 좀 더 자세하고 엄밀한 부분이 필요하다면 논문을 참고하셔도 좋습니다.

<br/> 
   
*(주) 본문에 들어가기에 앞서, 이 리뷰는 논문의 핵심적인 개념을 위주로 서술한 것임을 밝힙니다. 이 논문은 이론적인 분석이 주가 되는 논문이기에, 이 논문에 있는 모든 Theorem, Proposition 등을 충분히 이해하기 위해서는 Specral GNN에서 포괄하고 있는 많은 배경 지식을 필요로 합니다. 다만 이 리뷰를 작성하는 저도 그러한 배경 지식이 충분하지 않기에, 이 논문에서 말하고자 하는 essential한 부분에 대해서만 다루고자 합니다. 부족한 부분은 Revision 기간에 더욱 보완하도록 하겠으니, 그때까지 기다려 주시면 정말 감사드리겠습니다.*

<br/> 

## **2. Preliminaries**  

이 섹션에서는 논문 본문에서 쓰인 Notation을 그대로 서술하도록 하겠습니다. 아래는 matrix의 행, 열에 대한 Notation입니다.

$$\forall M \in \mathbb{R}^{a\times b}: M_{i}=\mathrm{row_{i}}(M), M_{:i}=\mathrm{col_{i}}(M)$$

그리고, 주어진 node $i\in\mathbb{V}$에 대해서 그 이웃을 $N(i)$로 표기합니다.

아래는 matrix의 condition number의 정의입니다. 이 개념은 전술했던 Contribution 3번에서의 분석과 관련이 있습니다. 여기서 $\lambda_{max}$는 matrix의 Maximum Eigenvalue, $\lambda_{min}$은 matrix의 Minimum Eigenvalue를 의미합니다.
$$\kappa(M)=\frac{|\lambda_{max}|}{|\lambda_{min}|}$$

이때, 주어진 matrix $M$이 singular(=not invertible; inverse가 존재하지 않는 경우)라면 $\kappa(M)=+\infty$이고, 이는 matrix의 모든 eigenvalue가 non-zero 값을 갖는 것이 matrix의 invertiblility와 동치이기 때문입니다. [6]

*(주) 다만 위 정의의 경우 오류가 있는 것 같습니다. $| \lambda | _{max}$, $| \lambda | _{min}$이 맞는 표기이지 않을까 싶습니다.

아래는 Graph와 관련된 Notation입니다. 기본적으로 주어진 Graph는 undirected입니다. $\mathcal{G}=(\mathbb{V}, \mathbb{E}, X)$는 주어진 Graph이고, 여기서 
$$\mathbb{V}=\{1,2,\cdots,n\},\ \mathbb{E}\subset \mathbb{V}\times\mathbb{V},\ X\in\mathbb{R}^{n\times d}$$

는 각각 Node set, Edge set, node feature matrix입니다.

$A, D$를 각각 Adjacency, Degree matrix라고 하면, normalized adjacency는 $\hat{A}=D^{-1/2}AD^{-1/2}$이고 symmetric normalized graph Laplacian은 $\hat{L}=I-\hat{A}$입니다. Graph Laplacian은 Real symmetric이기에 orthogonally diagonalizable하고, 따라서 아래와 같이 Eigen-decomposition할 수 있습니다.
$$\hat{L}=U\Lambda U^{T}$$

U는 ith column이 $\hat{L}$의 ith eigenvalue에 해당하는 eigenvector인 orthogonal matrix이고, $\Lambda$는 eigenvalue들을 diagonal entry들로 갖는 diagonal matrix입니다.


### **2.1. Graph Isomorphism**

이 섹션에서는 Graph Isomorphism에 대해 간략하게 다룹니다.

Graph Isomorphism은 중요한 개념이긴 하나, 이 리뷰에서는 Theorem, proposition의 증명을 상세히 다루지 않고 그 안에 담긴 의미에 대해서만 다룰 예정이기에 논문 본문에서 서술한 것 대신, 널리 알려진 정의[7]에 대해서 서술하도록 하겠습니다.

두 graph $\mathcal{G_1}=(\mathbb{V_1}, \mathbb{E_1}, X_1),\ \mathcal{G_2}=(\mathbb{V_2}, \mathbb{E_2}, X_2)$에 대해 bijective(1 to 1 correspondence; 일대일대응) mapping $f:\mathbb{V_1}\rightarrow\mathbb{V_2}$가 존재해서, $(i,j)\in\mathbb{E_1}$인 임의의 두 node $i, j\in\mathbb{V_1}$의 mapped node $f(i),f(j)\in\mathbb{V_2}$가 $(f(i),f(j))\in\mathbb{E_2}$일 때 두 graph $\mathcal{G_1},\mathcal{G_2}$를 isomorphic하다고 하고, $f$를 isomorphism이라고 부릅니다.

간단하게 말하자면, 두 graph의 구조가 같은 것을 의미합니다.

### **2.2. Graph Signal Filter and Spectral GNNs**

이 섹션에서는 Graph Signal Filter와 Spectral GNN의 개념, 그리고 논문에서 주로 다루는 Linear Spectral GNN(linear GNN in original paper)에 대해 서술합니다. 그리고 Filter의 표현력에 대한 개념인 _Polynomial-Filter-Most-Expressive_(PFME)와 _Filter-Most-Expressive_(FME)에 대해서도 소개하겠습니다.

Graph Fourier Transform의 정의는 논문에서 정의된 바와 같이, (Shuman et al., 2013)[8]의 정의를 따릅니다.
Signal $X\in\mathbb{R}^{n\times d}$의 Graph Fourier Transform은
$$\tilde{X}=U^{T}X$$

로 정의하며, 



<br/> 

## **3. Analyses**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

<br/> 

## **4. Methodology-JacobiConv**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

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
 9. Stephen Boyd and Lieven Vandenberghe. _Convex Optimization_. Cambridge University Press, 2009.
 10. Richard Burden and J. Douglas Faires. _Numerical Analysis_. Cengage Learning, 2005.

