---
title:  "[NIPS 2022] Pure Transformers are Powerful Graph Learner"
permalink: Pure_Transformers_are_Powerful_Graph_Learner.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **Pure Transformers are Powerful Graph Learners**

## **0. Preliminaries**
- What is the Graph?
	- Graph 는 Vertex와 Edge의 집합으로, $G(V,E)$ 로 표현합니다. 
	- Wikipedia의 정의는 "a structure made of vertices and edges" 입니다.
	- Directed Graph는 Edge에 방향이 있는 경우를, Undirected Graph는 Edge의 방향이 없는 경우를 의미합니다. 해당 논문은 Undirected Setting입니다. 
- $D$ : 행렬의 (n,n) 요소가 nth-Vertex에 걸려 있는 Edge의 갯수로 표현된 Diagonal Matrix입니다. 
- $A$ : m번째 Vertex가 n번째 Vertex와 연결되어 있으면 행렬의 (m,n) 요소가 1이 되며 아닌 경우 0이 됩니다. Vertex간의 연결성 정보를 담고 있습니다. 
- Graph Laplacian: $L=D-A$ 행렬로, 각 행과 열은 합이 0이 됩니다. Real Symmetric Matrix이기에 항상 Eigendecomposable 합니다. 
- Normalized Graph Laplacian: Graph Laplacian은 각 행과 열의 합이 0이 되기는 하나, 각 행과 열의 스케일이 다릅니다. 
- $L' = D^{-1/2}LD^{-1/2}=I-D^{-1/2}AD^{-1/2}$ 와 같이 Laplacian Matrix에 Diagonal Matrix의 -0.5승을 양옆에 곱하여 각 행/열을 degree로 Normalize합니다. 

## **1. Introduction and Motivation**

 강력한 Natural Language Module로써의 역할을 수행하던 Attention Architecture가 Image 도메인으로 이식되어 Vision Transformer 가 탄생한 것과 같이, Graph Domain에서도 Transformer를 사용하려는 시도가 지속적으로 있어 왔습니다. Graph Attention Network 처럼 Message Passing GNN 의 Aggregate 과정에서 Attention을 활용하는 시도도 있었으며 Transformer 구조를 도입하여 Attention을 사용하려는 시도도 있었습니다. 

 Transformer 구조를 채용할 때에 그저 단순히 노드들을 Transformer의 Token으로 사용한다면 그래프 구조가 제대로 학습될 수 없는 문제가 발생합니다. Edge 정보와 그래프 연결에 대한 정보가 Token으로 전환되지 않았기 때문입니다. 그래서 Attention is All You Need 논문에서 각 Token에 Positional Encoding을 더해주듯이,  Edge 의 정보와 Vertex간의 관계 (i.e. shortest path)를 Encoding하여 그래프 구조를 Transformer Encoder에 학습시키는 기법들이 연구되어 왔으며 벤치마크 데이터셋에서 매우 우수한 성과를 거두었습니다. 아래는 Graph Transformer 관련하여 추가로 읽어 볼 만한 논문들입니다.  

- Self-Supervised Graph Transformer on Large-Scale Molecular Data. NeurIPS 2020
- Rethinking Graph Transformer with Spectral Attention? NIPS 2021
- [**Do Transformers Really Perform Bad for Graph Representation?**](https://dsailatkaist.github.io/Do_Transformers_Really_Perform_Bad_for_Graph_Representation.html)  NeurIPS 2021


볼드체로 표시된 Do Transformer Really Perform Bad for Graph Representation[1]에서는 각 노드간의 상관관계를 인코딩한 1)Centrality Encoding과, 두 노드 사이의 Shortest Path를 Parametrize한 2)Spatial Encoding 개념을 제시함과 동시에, 그래프의 Edge Feature 또한 Transformer에게 전달하기 위해 두 노드 사이의 shortest path들의 feature들을 Encoding 한 3)Edge Encoding을 Transformer에게 전달해 줍니다. Graphormer는 결국 Transformer에게 그래프 구조를 제대로 학습시키기 위해서 3가지의 Learnable Encoder를 Pure Transformer에 추가로 도입합니다. 아래 삽화를 통해 Graphormer에 도입된 세개의 추가적인 Encoding을 직관적으로 이해할 수 있습니다. Graphormer와 같은 경우에는 추가적인 Encoding들을 더함으로써 Pure Transformer 구조에 Graph Specific Modification이 가해진다는 사실을 알 수 있습니다. 

![](https://github.com/Jaewoopudding/DeepLearningPractice/blob/master/homer.PNG?raw=true)


Message Passing 구조의 GNN경우에는  Attention을 Local Node에만 적용시켜 Attention의 장점을 모두 활용하지 못 하는 문제와, 학습이 진행됨에 따라 모든 Node의 Feature가 비슷비슷해지는 graph oversmoothing 문제가 있으며 잘 학습이 되더라도 1-WL test의 표현력을 넘지 못한다고 알려져 있습니다[2][3].  

그래서   **Pure Transformers are Powerful Graph Learners**의 저자들은, 표현력에 제한이 있는 Message Pssing GNN을 사용하지 않으면서도, Transformer를 그래프에 맞게 변형하는 연구 흐름과는 반대로 기존 Transformer Encoder의 구조를 변형하지 않은 Pure Transformer를 사용하기로 결정하였습니다.  그 결과 논문 저자들은 Edge와 Vertex를 각각 Tokenize하여 추가적인 Learnable Encoder 도입 없이도 Vertex Feature와 Edge Feature, 그리고 그래프 구조까지도 Transformer에 효과적으로 전달 수 있는 새로운 기법인 **TokenGT** 를 개발했습니다.

![](https://github.com/Jaewoopudding/DeepLearningPractice/blob/master/tokenGT.PNG?raw=true)




## **2. Contributions**

1. Node Identifier와 Type Identifier의 도입을 통해 그래프 구조를 Vertex/Edge Feature에 통합하여 Pure Transformer를 사용할 수 있는 방법을 제시하였습니다. 
2. Token-wise Embedding과 그 사이의 Self Attention이 그래프 상의 Permutation Equivalent한 선형 연산자(IGN)를 근사할 수 있다는 사실을 밝힘으로써 Transformer 구조가 Message Passing GNN보다 더 우월한 표현력을 가짐을 보였습니다. 
3. Transformer Encoder 가 최소한 WL test 만큼의 강력한 표현력을 가짐을 증명하였습니다. 
4. 여러 그래프 벤치마크 데이터셋에 대해서 TokenGT 가 Transformer 구조에 많은 변형을 가한 모델들 만큼  경쟁력 있음을 보였습니다.  

## **3. Method**

- Node Identifier
	- n개의 node가 있는 그래프에 대해서, 서로 orthonormal한 $\vec{p_v} \in \mathcal{R}^{(d_p)}$ 가 Node Identifier 입니다. 
	- $v$의 Node Feature $X_v$는 다음과 같이 augmented 됩니다. $X_v = [X_v, \vec{p_v}, \vec{p_v}]$
	- $u$와 $v$사이의 Edge Feature $X_{(u,v)}$는 다음과 같이 augmented 됩니다. $X_{(u,v)} = [X_{(u,v)}, \vec{p_v}, \vec{p_u}]$
	- orthonormal한 $\vec{p_v} \in \mathcal{R}^{(d_p)}$을 얻는 방법
		-  Normalized Graph Laplacian $I-D^{1/2}AD^{1/2}$ 를 Eigen Decomposition 해서 얻는 Eigenvector들을 사용합니다.
		- 무작위로 생성된 행렬을 QR Decomposition해서 얻은 Orthogonal Random Features을 사용합니다.
		- Normalized Graph Laplacian Eigenvector들은 그래프 연결성에 대한 정보를 갖고 있기 때문에 Orthogonal Random Features 기법보다 더 강력한 node identifier를 제공합니다.
		- Orthonormal한 벡터를 Node Identifier로 사용함으로써, 서로 연결되어 있는 Node와 Edge의 내적 $[\vec{p_k}, \vec{p_k}] \cdot [\vec{p_v}, \vec{p_u}]$ 은,  $k\in {(v,u)}$가 아닌 이상 모두 0이 되어 Transformer는 연결 관계를 학습할 수 있게 됩니다. 


- Type Identifier
	- 학습 가능한 두개의 vector $\vec{E^\mathcal{V}}$와 $\vec{E^\mathcal{E}}$를 통해 최종 형태의 Token을 정의합니다.
	- Node Token:  $X_v = [X_v, \vec{p_v}, \vec{p_v}, \vec{E^\mathcal{V}}]$
	- Edge Token: $X_{(u,v)} = [X_{(u,v)}, \vec{p_v}, \vec{p_u}, \vec{E^\mathcal{E}}]$
	- 최종 Token을 통해 우리는 Transformer 구조의 변형 없이도 해당 토큰이 Node인지, Edge인지, 서로 연결되어 있는지의 여부를 학습 할 수 있습니다. 

- Transformer
	- ViT와 Bert에 사용된 Transformer와 똑같은 Multi Head Attention 을 활용합니다.

		
## **4. Theoretical Analysis**

IGN에 관련된 구체적인 Theorem들과 증명과정들은 생략하였습니다. 

- k-IGN은 k-WL test와 연계되어 있으며 k-IGN은 k-WL만큼 강력하다는 것이 증명되어 있습니다. [4]
- 2-IGN은 모든 Message Passing GNN보다 표현력이 높은 것이 증명 되어 있습니다. 
- 논문에서 IGN의 Equivariant Linear Layer를 Transformer로 근사할 수 있음을 증명했습니다.
- 2-IGN은 모든 Message Passing GNN 보다 강력하므로, Transformer로 구현된 TokenGT는 이론적으로 모든 Message Passing GNN보다 표현력이 높습니다. 
 

## **5. Experiment**

### **Experiment setup**

-   Dataset
	-Barabasi-Albert Random Graph: IGN Approximation 실험을 위한 그래프 실험 환경
	-PCQM4Mv2: 대규모 양자 화학 데이터셋 


### **Result**

 Theoretical Analysis에서 언급했다시피, Transformer는 IGN을 근사할 수 있고, IGN을 근사할 수 있다는 건 IGN의 Equivariant Basis를 근사할 수 있다는 의미입니다. 아래의 실험 결과는 unseen graph에 대해서 TokenGT가 Equivariant Basis를 얼마나 잘 근사할 수 있었는지 보여주고 있습니다. 또한 Node Identifier와 Type Identifier에 대한 Ablation Study를 통해서, 각각의 요소가 성능 향상에 기여했음을 보여줍니다.  

Figure2를 온전히 이해하기 위해서는 IGN의 핵심 내용을 알아야 하는데, 이는 수식적으로 전개하자면 크로네커 곱, Vectorize 등의 선형대수 연산에 더해 여러 수식과 증명에 대한 추가 설명이 필요합니다. 자세한 내용이 궁금하시다면 원문인 [Invariant and Equivariant Graph Networks](https://arxiv.org/pdf/1812.09902.pdf) 을 참고하시면 좋을 것 같습니다. 

간단하게 요약하자면, 그래프에 대해서 Permutation Invariance 한 Linear Operator, X의 Orthorgonal Basis $(\in \mathbb{R}^{n^2 \times n^2})$가 15개가 존재하며, 최우측의 Orthorgonal Basis의 Ground Truth를 Pure Transformer가 훌륭하게 학습해낸다 정도로 이해하시면 되겠습니다. Figure2 는 15개의 Basis 중 2개의 예시이며, Node Identifier 와 type Identifier가 둘다 제공될 때에서야 비로소 Equivariant Basis를 잘 학습한다는 결과를 볼 수 있습니다. 


![](https://github.com/Jaewoopudding/DeepLearningPractice/blob/master/GT2.PNG?raw=true)


아래의 Table 2를 통해 TokenGT가 Strong modification이 가해진 Transformer만큼의 성능을 가지고 있음을 보여줍니다. 또한 Normalized Graph Laplacian Eigenvector로 node identifier를 설정한 경우가 랜덤한 node identifier보다 성능이 높음을 알 수 있습니다. 

Normalized Graph Laplacian $(I-D^{1/2}AD^{1/2})$ 의 Eigenvector는 Diagonal Matrix로부터 각 Node의 차수(degree) 정보를 받고, Adjacency Matrix로부터 각 노드간의 연결성 정보를 받습니다. 그러므로, Node Identifier를 Normalized Graph Laplacian Eigenvector로 결정한 경우, Random Orthogonal Data보다 그래프에 대한 설명력을 더 크게 반영할 수 있습니다. 

![](https://github.com/Jaewoopudding/DeepLearningPractice/blob/master/GT1.PNG?raw=true)


## **6. Conclusion**

#### Good Points
1. Node Identifier와 Type Identifier의 도입으로 Pure Transformer가 Thoery와 Practice 모두에서 효과적임을 밝혔습니다.
	- Transformer가 최소한 k-IGN, k-WL 만큼 표현력이 강하다는 사실을 증명했습니다. 
	- TokenGT가 GNN보다는 우수한 성능을, Strongly Modified Transformer와는 비슷한 성능을 내는 것을 확인하였습니다. 
2. 그래프를 (n+m)개의 토큰으로 해석함으로써 Graph Learning 의 새로운 Paradigm을 열었습니다. 

#### Challenges
1. 토큰이 (n+m)개가 되어서 큰 그래프에 적용하기에는 Computational Cost가 지수적으로 증가합니다.
2. SOTA보다 낮은 성능을 기록했습니다.


>#### Transformer 구조의 가장 큰 특징중 하나는 weak inductive bias입니다.
>- CNN : Transformer와 GNN(MPNN) : Graph Transformer 간의 관계에 상사성이 존재합니다. 
>- CNN은 강력한 inductive bias를 도입하여 weight sharing을 통해 적은 parameter로 이미지 특징을 효과적으로 학습할 수 있었습니다. 
>- ViT는 induvtive bias를 도입하지 않기에, 많은 데이터가 주어져야 제대로 된 성능을 낼 수 있지만, image에 대한 global view를 가질 수 있어 CNN보다 강력한 성능을 발휘합니다.
>- Message Passing GNN 또한 k-hop node들의 정보를 aggregation함으로써 간단한 구조임에도 불구하고 강력한 성능을 내는 strong inductive bias를 가지고 있습니다. 그러나, 멀리 있지만 유의미한 정보를 가진 Node의 신호가 제대로 전달이 안 될 수도 있습니다.
>- Graph Transformer 계열은 weak inductive bias 를 가지고 있지만, Message Passing 과정이 없기에 Graph 전체의 신호를 한번에 파악할 수 있어 강력한 성능을 낼 수 있습니다. 




----------

## **Author Information**

-   Author name : Jaewoo Lee
    -   Affiliation : KAIST SILAB
    -   Research Topic : Offline Reinforcement Learning, Graph Representational Learning, Meta Learning.

## **7. Reference & Additional materials**

Please write the reference. If paper provides the public code or other materials, refer them.

-   [Github Implementation](https://github.com/jw9730/tokengt)
-   Reference
	- [Paper Link](https://arxiv.org/abs/2207.02505)
	- [1] Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., & Liu, T.-Y. (2021). _Do Transformers Really Perform Bad for Graph Representation?_ http://arxiv.org/abs/2106.05234
	- [2] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). _How Powerful are Graph Neural Networks?_ http://arxiv.org/abs/1810.00826
	- [3] Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). _Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks_. www.aaai.org
	- [4] Maron, H., Ben-Hamu, H., Shamir, N., & Lipman, Y. (2018). _Invariant and Equivariant Graph Networks_. http://arxiv.org/abs/1812.09902
	- [5] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). _Neural Message Passing for Quantum Chemistry_. http://arxiv.org/abs/1704.01212
	- k-IGN : https://harryjo97.github.io/paper%20review/Invariant-and-Equivariant-Graph-Networks/
	
