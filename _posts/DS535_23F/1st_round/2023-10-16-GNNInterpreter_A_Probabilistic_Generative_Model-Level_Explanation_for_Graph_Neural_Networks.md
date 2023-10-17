---
title:  "[ICLR 2023] GNNInterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks"
permalink: 2023-10-16-GNNInterpreter_A_Probabilistic_Generative_Model-Level_Explanation_for_Graph_Neural_Networks.html
tags: [reviews]
use_math: true
usemathjax: true
---

# GNNInterpreter: A Probabilitstic Generative Model-level Explanation for Graph Neural Networks
Writer: 20233497 이광현

본 리뷰는 ICLR 2023에 Accept된 'GNNInterpreter: A Probabilistic Generative Model-level Explanation for Graph Neural Networks' 논문에 대한 리뷰입니다.
한글로 번역하였을 때 의미나 이해에 오히려 혼동을 줄 수 있는 용어는 영어 원문 그대로 사용한 점 양해 부탁드립니다.

## 0. Preliminary: Explanation in AI
머신러닝 기술이 발달함에 따라 우리는 더 정확하고 정교한 결과값을 모델을 통해 얻을 수 있게 되었습니다. 그러나 이러한 모델이 실제 상황에 적용되기 위해서는 해당 모델의 신뢰도, 즉 모델이 도출한 결과값이 얼마나 믿을 수 있는지가 중요합니다. 이러한 신뢰도는 모델의 성능뿐만 아니라 모델의 결과값 도출 과정이 해당 업무에서 보편적으로 통용되는 상식, 즉 Domain Knowledge에 얼마나 부합되냐에 따라 높을 수도 낮을 수도 있습니다. 오늘날 대부분의 딥러닝/머신러닝 모델은 결과값 도출 프로세스가 사람이 직관적으로 이해하기 어려운 blackbox 모델의 형태를 하고 있습니다. 어떠한 이유로 결과값이 나왔는지 이해할 수 없는 모델은 금융, 의료, 법 등의 민감한 분야에 사용되기 어려울 것입니다. 그렇기에 많은 연구자들이 이러한 이러한 결과값 또는 모델 자체를 사람이 이해할 수 있는 레벨로 설명할 수 있도록 하는 방법을 연구하였으며, 이를 Explainable AI(XAI)라고도 합니다. 본 논문도 이러한 맥락에서 그래프를 학습한 딥러닝 모델의 결과를 사람에게 더 이해하기 쉬운 방법으로 설명하고자 하는 방향의 연구입니다.

아래의 그림은 XAI 시스템을 표현한 그림입니다.

![image](hhttps://i.ibb.co/sH5CRCs/1-d-C5-Ha-OQVUXn-VPDXWdop65-A.png)

(출처: https://towardsdatascience.com/knowledge-graphs-for-explainable-ai-dcd73c5c016)

위 그림에서 knowledge matching을 통해 기존의 딥러닝 모델의 hidden layer, feature를 사람에게 이해 가능한 설명을 하거나 결과값 도출에 대한 근거를 제공함으로써 모델의 신뢰도를 높일 수 있습니다.

## 1. Introduction
그래프 데이터 구조는 Node와 Edge를 이용하여 각 Object의 관계성을 표현하는데 매우 뛰어난 효과를 보이기 때문에 다양한 응용 분야에서 사용됩니다. 그러나 그래프 데이터 구조는 Non-Euclidean하기 때문에 이를 모델링하는 것에 어려움을 겪었었고 Graph Neural Network (GNN)이 등장하면서 이러한 문제는 어느 정도 해결되었습니다. 현재 GNN은 그래프 데이터의 특성을 표현하는데 의미가 있는 feature를 추출하는 Representation Learning에 강점을 보이나, Neural Network의 복잡도로 모델의 도출된 결과에 대한 설명력이 부족합니다.
현재 텍스트, 이미지 데이터를 다루는 딥러닝 모델을 설명하는 연구는 많이 진행되어 왔으나 GNN은 몇 가지 이유로 다른 데이터에 비해 연구가 덜 진행되어 왔습니다. 본 논문에서는 GNN을 설명하는 것이 어려운 이유로 3가지를 들었습니다.

* Adjacency matrix가 discrete한 값(0 또는 1)을 가지기 때문에 Gradient method를 바로 적용하기 어렵습니다.
* 특정 도메인에서는 도메인 지식에 의한 규칙을 만족해야만 그래프가 유효합니다. (Ex. 특정 물질의 화학식을 그래프로 표현할 때 화학에서 통용되는 원칙을 지켜야 유효한 모델이 됩니다.)
* 그래프 구조는 다양한 종류의 Node와 Edge feature를 가지고 있기 때문에 이를 일반적으로 설명하는 method를 제안하는 것은 굉장히 어렵습니다.

현재까지 GNN을 설명하는 연구는 방식에 따라 크게 2가지로 분류할 수 있습니다. Instance-level과 Model-level로 나뉘며, Instance-level explanation은 특정 그래프 인스턴스에 대한 모델의 예측을 설명하는 것에 초점을 맞춘다면, Model-level explanation은 모델의 전반적인 이동을 분석하여 표현하는데 초점을 맞춥니다. 예를 들어, 암을 예측하는 모델이 있다면 Instance-level로는 특정 환자의 특정 이상 건강 정보를 가지고 암이 걸렸음을 설명한다면, Model-level은 전반적으로 어떠한 feature가 암 진단 여부에 얼마나 영향을 미치는지를 설명한다고 생각할 수 있습니다. 그렇기에, Model-level explanation은 Black-Box GNN에 대해 이 모델이 어떠한 feature에 대해 어떻게 움직이고, 특정 feature에 편향된 결과값을 도출하는 지에 대한 여부를 확인할 수 있습니다. Model-level이 더 고차원적인 접근이라고 볼 수 있습니다.

본 논문이 제안한 GNNInterpreter는 기존 Model-level explanation method 중 state-of-art method인 XGNN과 비교하여 다음의 장점을 가지고 있습니다.

* GNNInterpreter는 XGNN에 비해 다양한 종류의 node와 edge feature에 대해 설명력을 가지기 때문에 더 general하게 적용될 수 있습니다. 반면에, XGNN은 continous node feature에 적용이 불가능합니다.
* 설정한 Objective function의 특징으로 GNNInterpreter는 task에 대한 도메인 지식에 더 유연하게 설명이 가능합니다. 반면에, XGNN은 강화학습을 위해 도메인에 따라 수동적으로 reward function을 설정해야 합니다.
* GNNInterpreter는 XGNN에 비하여 더 적은 시간에 학습이 가능합니다.
* GNNInterpreter는 GNN을 설명하기 위해 또 다른 딥러닝 모델을 학습해야 하는 XGNN과 다르게 수학적 최적해를 구하는 방식으로 접근하기 때문에 더 직관적인 설명력을 가집니다. (XGNN은 blackbox 모델을 사용하기 때문에 이 blackbox 모델로 인한 설명력 저하가 발생합니다.)

## 2. Background (Notation)
본 논문에서 사용하는 Notation은 아래와 같습니다.

### **Notation**

* 그래프 $G$는 $G = (V, E)$로 표현됩니다. 이때, $V, E$는 각각 그래프의 Node(vertex)와 Edge를 의미합니다.
* Node와 Edge의 숫자는 각각 $N, M$입니다.
* 노드와 노드의 연결 여부를 보여주는 adjacency matrix는 $A \in {\{ 0,1 \} } ^ {N \times N}, a_{ij} \in {\{ 0,1 \} }$
* 그리고 Node와 Edge의 feature matrix는 각각 $X\in{\mathbb {R}}^{N \times k_V} ,Z\in{\mathbb {R}}^{M \times k_E}$로 표현됩니다.

본 논문에서도, 일반적인 GNN 논문과 같이 Message passing 아이디어를 사용하며 다음과 같은 notation을 사용합니다.

* i번째 Hidden layer에 대한 Message passing은 다음과 같이 표현됩니다. $H^i = f(H^{i-1},A,Z), H^0 = X, where H^i \in {\mathbb {R}}^{N \times F^i}$

## 3. GNNInterpreter Implementation
간략하게 GNNInterpreter는 주어진 그래프에 대한 node, edge의 분포를 가정하여 parameter를 학습하고 이를 샘플링하는 일반적인 generative model의 구조를 활용합니다. (이미 학습된 GNN에 대해서 각 임베딩 ($Z, X$) 분포의 parameter를 학습하는 것이기 때문에 기존의 GNN에 영향을 주지는 않습니다.)

본 논문에서는 해당 GNN을 설명하는 과정을 explanation graph을 샘플링하는 것으로 보여줍니다.

GNNInterpreter의 학습을 위한 Objective Function은 아래와 같이 정의됩니다.
$$L(G)=L(A,Z,X)=\phi_c(A,Z,X)+\mu sim_{cos}(\psi(A,Z,X), \bar{\psi}_c)$$
그리고 모델 학습은 위의 Objective function을 maximize하는 $A,Z,X$를 찾는, learning objective를 가집니다.
![image](https://i.ibb.co/qNhkbLJ/image1.png)

여기서 $\phi_c, \psi, \bar{\psi}_c$는 각각 다음의 역할을 합니다.
* $\phi_c$: 특정 클래스 $c$에 대한 설명하고자 하는 GNN가 예측한 softmax 함수에 태우기 전의 scoring function 값입니다.
* $\psi$: 설명하고자 하는 GNN의 그래프 임베딩입니다.
* $\bar{\psi}_c$: 특정 클래스 $c$에 속하는 모든 그래프의 그래프 임베딩의 평균값입니다.
* $sim_{cos}, \mu$: cosine similarity와 weighting factor입니다.

위의 objective 식을 간단하게 설명하면, 
* First term의 경우 scoring function을 maximize함으로써 주어진 GNN embedding에 대한 가장 확률이 높은 $A,Z,X$를 찾는다는 뜻입니다.
* Second term은 속한 클래스의 average와 유사도를 높임으로써 클래스의 특성을 고려하는, 즉 도메인 지식을 고려한다는 뜻입니다.

위의 objective식을 maximize하여 그래프의 확률 분포를 학습하기 위해 논문에서는 2개의 가정을 합니다.

* 첫번째로, 모든 가능한 edge가 독립적인 0<p<1의 확률을 가지고 생성된다고 가정합니다. (이러한 edge의 확률을 정의한 그래프를 Gilbert random graph라고도 합니다.)
* 두번째로, node와 edge의 feature는 독립적으로 분포한다고 가정합니다.

이러한 2개의 independence 가정을 통해, 그래프 $G$의 확률 분포는 다음과 같이 factorize가 가능합니다. 

![image](https://i.ibb.co/NN282s3/image2.png)

즉, 이상적인 아이디어는 adjacency matrix $A$, node feature $X$, edge feature $Z$를 모두 확률 변수로 생각하고 다음의 분포를 가진다고 가정하는 것입니다.

* $a_{ij} ~ Bernoulli(\theta_{ij})$: Adjacency matrix의 $a_{ij}$는 $\theta_{ij}$의 확률을 가진 베르누이 분포라 가정합니다.
* Node와 Edge의 클래스 개수를 $k_V, k_E$라 하였을 때,
* $x_i ~ Categorical(p_i)$: 각각의 node feature는 normalized된 categorical distribution 확률 $p_i$를 따릅니다.
* $z_{ij} ~ Categorical(q_{ij})$: 각각의 edge feature 또한 normalized된 categorical distribution 확률 $q_{ij}$를 따릅니다

이러한 확률 분포를 가정하였을 때 learning objective를 아래와 같이 다시 쓸 수 있습니다.

![image](https://i.ibb.co/VvyYMMk/image3.png)

즉, generative model에서 자주 사용되는 (expected) learning objective를 maximize하는 각 variable의 parameter를 optimize하는 task가 됩니다.

그리고 gradient method를 사용하기 위해 discrete한 $a_{ij}, x_i, z_{ij}$를 아래와 같이 categorical distribtuion의 continuous 버전인 concrete distribution으로 relaxation합니다.

![image](https://i.ibb.co/Fh4j6gT/image4.png)

여기서 또 문제가 생기는 것이, 여전히 각 feature가 분포를 가정하고 샘플링을 하면 gradient 계산을 통한 backpropagation을 하기 어렵기 때문에 $\epsilon ~ Uniform(0,1)$를 이용한 reparametrization trick을 사용합니다. 이를 통해 gradient 계산 및 backpropagation이 가능해집니다.

![image](https://i.ibb.co/sqr0y06/image5.png)

위와 같이 continuous relaxation 및 reparametrization trick을 적용한 learning objective는 Monte Carlo 샘플링을 통해 근사가 가능해집니다.

![image](https://i.ibb.co/ZcRb84P/image6.png)

추가로 GNNInterpreter 구현에서 더 나은 최적화를 위해 논문에서는 3가지의 regularization을 주었습니다.

* 각 latent parameter에 L1, L2 regularization을 적용하여 학습 때 gradient saturation을 방지하였습니다.
* 너무 복잡한 표현은 되려 좋지 않기 때문에 간결성을 위해 budget penalty term을 추가하여 explanation graph의 사이즈를 제한하였습니다.
* 인접한 edge간의 correlation을 높이기 위한 incentive term을 추가하였습니다.

아래 Algorithm 1은 위에서 설명한 GNNInterpreter를 학습하고 explanation graph를 generating, 즉 샘플링하는 알고리즘입니다.

![image](https://i.ibb.co/x8Dk6BY/image7.png)

## 4. Experimental study

GNNInterpreter의 실험은 총 4개의 데이터셋에서 진행되었고, XGNN이 적용 가능한 데이터셋에 한하여 정량적, 정성적 비교를 하였습니다.

아래는 본 논문에서 사용한 데이터셋과 해당 데이터셋에서 학습한 GNN의 종류에 대한 정보입니다.

![image](https://i.ibb.co/KzMvtk7/image8.png)

4개의 데이터셋 중 Cyclicity, Motif, Shape는 Synthetic한 데이터셋으로 클래스에 따른 다양한 모양의 그래프가 있습니다.
MUTAG는 분자구조에 따른 돌연변이 여부와 관련된 데이터셋으로 화학적 특징을 고려한 현실 세계에 가까운 데이터셋입니다.

XGNN의 경우, multiple edge feature를 적용할 수 없기 때문에 Cyclicity 데이터셋에서는 비교할 수 없었습니다. 그리고 Motif와 Shape 데이터셋의 경우, 논문에서는 XGNN을 해당 GNN에 학습시켜 표현하는데 여러 시행착오를 거쳤지만, 수용되기 어려울 정도의 퀄리티를 가진 설명 결과가 나와 공정한 비교를 위해 MUTAG 데이터셋에 대해서만 XGNN과 GNNInterpreter를 비교했다고 합니다.

![image](https://i.ibb.co/ZS8YDRd/image9.png)

Table 2는 4개의 데이터셋에 대해서 1000개의 그래프 분류 평균 성능입니다. MUTAG 데이터셋에서는 XGNN보다 더 높은 확률과 더 적은 분산을 보이고 있으며, 평균적인 학습시간도 XGNN에 비해 작은 것을 확인할 수 있습니다.

![image](https://i.ibb.co/2qG1XS0/image10.png)

Figure 1은 4개 데이터셋에 대한 정량적인 결과를 보여줍니다. 그림에서 왼쪽은 Explanation으로, XGNN 또는 GNNInterpreter가 예측한 그래프의 Node, Edge를 표현한 것입니다. 오른쪽은 Example, Motif라 하여 Explanation으로 보여준 그래프의 원래 모습입니다. MUTAG 데이터셋에서 XGNN과 GNNInterpreter를 비교하면 GNNInterpreter가 만든 그래프가 조금 더 원형에 가까운 모습인 것을 볼 수 있고, Motif나 Shape 데이터셋에서는 일부 그래프가 원형과 매우 비슷한 것을 확인할 수 있습니다. (House, Complete-5 등)

또한 논문에서 Ablation Study로써, 많은 경우에 논문에서 제안한 Objective 식의 seconde term이 도메인 지식과 관련하여 의미있는 설명을 생성하는데 중요한 역할을 한다고 주장합니다. 아래의 Figure 3은 second term을 추가하지 않았을 때 도메인 지식을 얼마나 놓치는 지를 보여줍니다.
![image](https://i.ibb.co/qNhkbLJ/image1.png)

![image](https://i.ibb.co/kJDgJ0P/image11.png)


논문에서는 mutagen class의 경우 NO2를 feature로 가지는 그래프가 많은데 second term을 고려하지 않았을 때 N이나 O node를 전혀 표현하지 못하는 것으로 보아 second term이 유의미한 explanation graph를 생성하는데 큰 역할을 한다고 주장합니다.

## 5. Conclusion

본 논문에서 제안한 GNNInterpreter는 GNN의 explanation을 위해 확률적 생성 모델의 아이디어를 사용합니다. 각각의 feature의 분포를 가정하고 분포의 parameter를 학습하는 방식으로 그래프의 패턴을 파악합니다. 새로운 objective를 설정하여 node, edge feature에 대한 제약을 줄였으며, 이러한 objective는 도메인과 상관없이 적용될 수 있고, 또한 도메인 지식을 표현하는데 효과를 보였습니다. 실제로 GNNInterpreter는 4개의 데이터셋에서 대해서 유의미한 설명력을 보였으며, 이를 통해 현실 세계의 데이터셋에 대해 좋은 설명력을 보일 것으로 기대됩니다. 또한 latent distribution을 정의함으로써 기존 방법으로 찾지 못하는 새로운 그래프 패턴을 발견할 수 있다는 점에서 Model Explanation (Graph), XAI 분야에 상당한 기여를 했다고 생각됩니다.
