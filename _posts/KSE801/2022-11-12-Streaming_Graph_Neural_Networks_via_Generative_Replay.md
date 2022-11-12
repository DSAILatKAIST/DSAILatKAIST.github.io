---
title:  "[KDD 2022] Streaming Graph Neural Networks via Generative Replay"
permalink: Streaming_Graph_Neural_Networks_via_Generative_Replay.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Streaming Graph Neural Networks with Generative Replay

## **1. Problem Definition**

> **Continual Learning에서 사용할 Replay Buffer를 Generative Model을 활용시켜 생성한다!**

본 논문은 Continual Learning을 Graph Neural Network(`GNN`)과 접목시킴과 동시에 이에 사용할 `Replay buffer`를 Generative Model을 사용해 생성합니다.

이미 Continual Learning에 Generative Model을 사용한 연구는 예전에도 있었으나 [(여기)](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html), 이는 이미지 데이터를 주로 targeting한 반면에 지금 다루는 논문은 Graph domain의 데이터에 적용, 그에 맞는 특성(structure of graph)를 고려했다는 점에서 novelty가 있습니다.

> **Continual Learning이란?**

Continaul Learning은 Lifelong Learning, Incremental Learning이라고도 불리며, 과거의 정보를 최대한 유지하며 새로운 정보를 학습하는 방법론입니다.

예를 들어, 인간이 '강아지' 라는 지식을 알고 있는 상태로, '고양이'라는 지식을 새로 습득했을 때, '강아지'를 잊지 않고 '강아지'와 '고양이'를 구별해 낼 수 있는 것 처럼, 지속적으로 들어오는 새로운 데이터를 학습함과 동시에 이전에 학습되었던 데이터를 잊지 않도록 인공지능을 설계하는 것이 Continual Learning의 목적입니다.

> **Catastrophic Forgetting이란?**
> 
Continual Learning에서, 새로운 데이터가 들어옴에 따라 이전에 학습했던 데이터의 정보를 망각하는 현상을 `Catastrophic Forgetting`이라고 합니다. 아래 그림을 보시겠습니다. 

![image](https://user-images.githubusercontent.com/99710438/194873406-84f4a722-c562-4c39-adec-5ecc498a498f.png)

그림에서 볼 수 있듯이, Task 1에서는 파란색 node들을 구별하도록 학습합니다. Task 2에서는 새로운 보라색 node가 추가되면서 파란색과 보라색을 포함해 학습시키고, Task 3에서는 빨간색의 새로운 node가 추가되면서 새롭게 학습이 진행됩니다. 이 과정이 Continual Learning 입니다.

그리고 Task가 진행됨에 따라 이전 Task에서 학습했던 node들에 대한 예측 성능이 떨어지는 것을 볼 수 있습니다. 

예를 들어, Task 1에서 파란 node들을 분류하는 데에는 95%의 성능을 보였으나, Task 2에서는 55%로 줄었고, Task 2에서 학습했던 보라색 node들도 Task 3에서의 성능은 현저히 줄어든 것을 볼 수 있습니다. 이러한 현상이 앞서 말씀드린 `Catastrophic Forgetting`입니다. 

Continual Learning 중 Replay approach는 이전 Task에서 학습했던 데이터 중 **일부**만을 sampling하여 이후 Task에 사용합니다. 전체 데이터를 계속해서 누적하여 학습하면 효율이 좋지 않기 때문이죠. 이렇게 sampling되어 이후 Task에서 같이 학습될 data를 `Replay buffer`라고 부릅니다. 이 과정에서 `Catastrophic Forgetting`현상이 일어나게 되는데, 이 현상을 최소화 하도록 이전 Task에서 학습했던 데이터를 잘 대표하는 데이터를 sampling하는 것이 관건입니다.

## **2. Motivation**

> **기존 Replay based Continual Learning 방법들은 storage limitation을 가진다!**

앞서 언급한대로, Replay based Continual Learning은 이전에 학습했던 데이터 중 전체 데이터를 잘 represent하는 **일부**의 데이터를 sampling해서 이후 Task에서 등장하는 데이터와 같이 학습시킵니다. 

하지만 이럴 경우 Task가 계속해서 진행됨에 따라, 각 Task에서 아무리 적은 데이터를 sampling한다 해도 기억해야하는 데이터가 지속적으로 늘어나고, 결국에는 memory를 다 사용해버리는 일이 생길 것입니다. 

저자들은 이런 한계를 지적하며 Generative Model을 도입해 `Replay buffer`에 데이터를 누적시키는 것이 아니라, Task가 시작될 때 마다 필요한 만큼 Generative Model로 `Replay buffer`를 생성해 학습을 진행하고자 합니다. 

> **기존 Replay based Continual Learning 방법들은 온전한 graph distribution을 보존하지 못한다!**

Image 도메인과 달리, Graph 도메인의 데이터를 다룰 때는 각 데이터의 성질(feature)뿐 만 아니라 그래프의 전체적인 structure도 고려해야 합니다. 어떤 node가 어떤 node와 연결되어 있으며, 연결된 node들은 어떤 특성을 가지고 있는지까지 종합적으로 고려되어야 한다는 것이죠. 

저자들은 Continual Learning 중에서 `Replay buffer`를 생성할 때 각 node들의 feature만 고려될 뿐, 전체적인 그래프의 distribution(structure)이 보존되지 못한다고 주장합니다. 이는 `Graph Neural Network`가 학습될 때 성능 저하를 야기하는 가장 큰 문제 중 하나로, 이 논문에서는 Generative Model을 통해 이러한 topological information까지 저장하도록 하는 것을 목표로 합니다.


## **3. Method**

> **Preliminaries: `GNN`**

논문에서 제안한 방법론을 이해하기 위해서는 `GNN`의 개념을 알고 있어야 합니다.

본 리뷰에서는 간단하게 소개를 하겠습니다.

$$N$$개의 노드를 가진 그래프 $$\mathcal{G}= \lbrace \mathcal{V},\mathcal{E} \rbrace$$가 주어지고, $$X = \lbrace x_{1}, x_{2}, ..., x_{N} \rbrace$$ 을 node feature의 집합이라고 하고, $$A$$를 node들의 관계를 표현하는 adjacency matrix라고 하겠습니다.

$$l-th$$ hidden layer에서의 $$v_{i}$$의 hidden representation을 $$h_{i}^{(l)}$$ 이라고 할 때, 이 $$h_{i}^{(l)}$$는 다음과 같이 계산됩니다:

$h_{i}^{(l)} = \sigma(\sum_{j \subset \mathcal{N}(i)} \mathcal{A_{ij}}h_{j}^{(l-1)}W^{(l)})$

이 때, $$\mathcal{N}(i)$$ 는 $$v_{i}$$의 neighbors를 의미하고, $$\sigma ( \bullet )$$는 activation function, $$W^{(l)}$$은 $$l-th$$ layer의 transform matrix를 나타냅니다.

$$h_{i}^{(0)}$$은 node $$v_{i}$$의 input feature를 나타내고, $$\mathcal{A}$$는 neighbors의 aggregation strategy이며, `GNN`의 핵심 중 하나입니다.

본 논문에서는 다양한 `GNN`중 `GraphSAGE`라는 모델을 사용하는데, 이 `GraphSAGE`의 $$k$$번째 layer는 다음과 같이 정의됩니다:

$h_{v}^{k} = \sigma(W^k \cdot MEAN( \lbrace h_v^{k-1} \rbrace \cup \lbrace h_u^{k-1}, \forall u \in \mathcal{N}(v)\rbrace)$

> **Problem Definition**

Continual Learning setting에서, 데이터는 그래프의 형태를 띠고 연속적으로 들어옵니다. 이는 다음과 같이 표현이 가능합니다.

$\mathcal{G} = (\mathcal{G}^1, \mathcal{G}^2, ..., \mathcal{G}^T)$$

where $$\mathcal{G^t} = \mathcal{G}^{t-1}+\Delta \mathcal{G}^t$$

여기서 $$\mathcal{G} = (A^t, X^t)$$ 는 attributed graph at time $$t$$이고, $$\Delta \mathcal{G} = (\Delta A^t , \Delta X^t)$$는 time $$t$$에서의 node attribute와 network의 structure의 변화량을 나타냅니다.

이 때 Streaming `GNN`은 traditional `GNN`을 streaming setting으로 확장한 것이 됩니다. Streaming graph가 있을 때, Continual Learning의 목적은 $$(\theta^1, \theta^2, ..., \theta^T)$$ 를 배우는 것입니다. 이 때 $$\theta^t$$ 는 time $$t$$ 에서의 `GNN` parameter를 의미합니다. 


> **Model Framework**

저자들은 이 논문에서 `SGNN-GR`이라는 방법론을 제시합니다. 모델 구조는 아래 그림과 같습니다.

![image](https://user-images.githubusercontent.com/99710438/194887946-3f736cc4-1c2c-47ca-97aa-4516da0ae42e.png)

**이 그림을 참고해 방법론을 개괄적으로 설명하자면, 아래와 같습니다.**

* 새로운 task가 오면 `GAN`으로 sequence를 생성(이게 `replay buffer`가 되는 것이죠)해서 이번 task의 그래프와 **같이** `GNN`을 학습합니다. 
* 이러면 이 `GNN`은 **현재 그래프를 학습함과 동시에 이전의 정보까지 기억**하게 될 것입니다. 
* 또한 이번 task에서 새롭게 생성된 node들과 그것들로부터 영향받은 node들을 다시 `GAN`의 input으로 주어 학습시킵니다. 
* 이러면 다음 task에서는 `GAN`은 더 양질의 `replay buffer`를 만들어 낼 수 있을 것입니다. 

지금부터 `SGNN-GR`의 자세한 내용을 살펴보겠습니다. 위 그림을 잘 참고하면서 아래 설명을 따라오시기 바랍니다.

가장 먼저, Streaming GNN의 time $$t$$에서의 loss는 다음과 같습니다.

$\mathcal{L}(\theta^t ; \mathcal{G}^t) = \mathcal{L}(\theta^t ; \mathcal{G}_A^t) + \lambda \mathcal{R} (\theta^{t-1} ; \mathcal{G}_S^t)$

우변의 첫 항은 incremental learning에 관한 것이고, 두 번째 항은 historical knowledge에 관한 것입니다. 

본 논문에서 $$\mathcal{G}_A^t$$ 는 graph의 affected part, $$\mathcal{G}_S^t$$ 는 graph의 stable part로 정의합니다. 

> 여기서 affected part는 계속해서 새로운 data가 들어옴에 따라 영향을 받는(변화된) part, 즉 새롭게 학습해야하는 part라고 생각하면 되고, stable part는 이전에 학습했던 part, 즉 변하지는 않았지만 기존의 지식을 잊지 않기 위해(`Catastrophic forgetting`을 방지하기 위해) 지속적으로 학습시켜야하는 part라고 생각하면 됩니다.

이 때 $$\Delta \mathcal{G}^t \subset \mathcal{G}_A^t$$ 이고 $$\mathcal{G}_S^t \subset \mathcal{G}^{t-1}$$ 입니다. 몇몇 node들이 새롭게 바뀐 node들에 대해서 영향을 받는 것입니다.

각 time step에서 모델은 main model(`GNN`)과 Generative Model로 구성됩니다. 위 그림에서 확인할 수 있듯이, Generative Model은 $$\mathcal{G}_ A^t$$에서 바뀐 node들과 $$\mathcal{G}^{t-1}$$에서의 replayed node를 training data로 받습니다. 이 때 replayed node는 이전 time step의 Generative Model로부터 나옵니다. 

이 논문에서는 Generative Model로 `GAN`을 사용하였습니다. `GAN`에 대한 자세한 설명은 생략하며, 원 논문은 [여기](https://dl.acm.org/doi/abs/10.1145/3422622)를 참고하시기 바랍니다. 

`GNN` 모델도 changed node와 replayed node를 똑같이 input으로 받습니다. 

Main model의 loss function은 다음과 같습니다.

$\mathcal{L}_ {GNN} (\theta^t) = r \mathbf{E}_ {v \sim \mathcal{G}_ A^t } \[ l(F_{\theta^t}(\upsilon), y_{\upsilon} ) \] + (1-r) \mathbf{E}_ {v' \sim G_{\phi^{t-1}}} \[ l(F_{\theta^t}(\upsilon '), F_{\theta^{t-1}}(\upsilon ')\] $

여기서 $$v$$는 changed node, $$v'$$는 replayed node입니다. 즉, 이 모델은 새로 들어온 node와 이전에 학습했던 node(replayed)를 동시에 학습합니다.


> **Generative Model for Node Neighborhood**

앞서 언급한대로, 일반적인 Generative model(ex. `GAN`)은 주로 computer vision 분야에서 활발하게 연구되었으나, graph data는 structure에 dependent하기 때문에, edge의 생성은 independent한 event가 아니라 jointly structured 되어야 합니다. 

`NetGan`이나 `GraphRNN`같은 Graph Generative model들이 있지만, 이는 전체 그래프를 생성하기 위함이지 node의 neighborhood를 생성하기 위함이 아니어서, 저자들은 `ego network`라는 node neighborhood 생성모델을 제시합니다. 이 `ego network`는 `GAN`의 프레임워크와 유사하지만, 그래프 상에서의 random walks with restart, 즉 `RWRs`를 학습하는 방향으로 사용합니다. 

`RWRs`는 일반적인 `Random Walk`모델에서 일정 확률로 starting node로 돌아가고, 그렇지 않으면 neighborhood node로 넘어갑니다. 이는 기존 `RWRs`가 `Random Walk`보다 훨씬 적은 step으로 explore가 가능하게 한다고 합니다. 

---

> **어떻게 `Random Walk with Restart(RWR)`이 기존 `Random Walk`보다 적은 length로 graph를 explore할까?**

Graph $$\mathcal{G}= \lbrace \mathcal{V},\mathcal{E} \rbrace$$ 가 있다고 합시다. Starting node는 $$v_0$$이고 그 node의 degree는 $$m$$이고 neighborhood는 $$N(v_0)$$이라고 합니다. $$T_{RW}$$를 기존 `Random Walk`가 $$v_0$$의 ego network를 explore하는데 필요한 step이라고 하면, $$E[T_{RW}] = \frac{(m-1)}{c} \cdot \frac{\sum_{v \in \mathcal{V} \setminus \mathcal{E} (v_0) } deg(v)}{d_{max}}$$, where $$\mathcal{E} (v_0) = \lbrace v_0 \rbrace \cup N(v_0) $$, $$d_{max}$$ : maximum degree of nodes in $$v$$'s neighborhood, $$c$$: the size of cut set of cut $$(\mathcal{E} (v_0), \mathcal{V} \setminus \mathcal{E}(v_0) )$$ 이라고 합니다. 자세한 증명은 논문의 appendix를 참고하시기 바랍니다. 

실제 그래프에는 node가 많으므로, $$ \left| \sum_{v \in \mathcal{V} \setminus \mathcal{E} (v_0)} deg(v) \right| \gg \left| cd_{max} \right| $$ 이고, $$E[T_{RW}]$$는 굉장히 큰 수가 되게 됩니다.

하지만 $$\alpha$$의 확률로 restart하는 `RWR`의 expected length to explore는 $$E[T_{RWR}] < \frac{m(\ln m+1)}{\alpha (1-\alpha)}$$가 된다고 합니다. 역시 자세한 증명은 논문의 appendix를 참고하시기 바랍니다.

\alpha를 예를 들어 0.2로 설정하면, $$E[T_{RW}] = \frac{(m-1)}{c} \cdot \frac{\sum_{v \in \mathcal{V} \setminus \mathcal{E} (v_0) } deg(v)}{d_{max}} \gg \frac{m(\ln m+1)}{\alpha (1-\alpha)} > E[T_{RWR}]$$이므로, `RWR`를 사용하는 것이 기존 `Random Walk`를 사용하는 것 보다 훨씬 빠른것을 확인할 수 있습니다.

---

지금부터 Generative Model에 관한 설명을 보겠습니다.

저자들은 node간의 dependency를 capture하기 위해 **m**이라는 graph state를 정의합니다. 각 walk step에서 $$m_l$$과 $$v_l$$을 계산하는데, 이 때의 input은 last state $$m_{l-1}$$과 last input $$s_{l-1}$$입니다. 이 $$s_{l-1}$$은 node identity $$v_{l-1}$$과 node attribute $$x_{l-1}$$을 포함하고 있습니다. 

Current state $$m_ l$$은 neural network $$f$$로 계산됩니다. 

Generator의 update process는 다음과 같습니다.

$m_l = f(m_{l-1}, s_{l-1}),$

$v_l = softmax(m_l \cdot W_{up,adj}),$

$x_l = m_l \cdot W_{up,fea},$

$s_l = (v_l \oplus x_l) \cdot W_{down}$

여기서 $$W_{up}, W_{down}$$은 차원을 맞춰주기 위한 projection matrix라고 생각하시면 됩니다. 

저자들은 `WGAN` 프레임워크를 사용해 모델을 학습을 진행했고, 위의 그림에서 확인할 수 있듯이 이 generator는 새로운 그래프 $$\mathcal{G}_ t$$ 에서 `RWRs`로 생성된 Sequence들을 input으로 받아 학습을 진행하고, 다음 task에서 `replay buffer`에 넣을 sequence를 뱉어줍니다. `GNN`은 이 sequence까지 포함해 학습하여 `catastrophic forgetting`을 방지합니다. 

Discriminator는 역시 sequence W의 node identity와 그에 해당하는 attrribute를 받아서 sequence의 score를 output으로 반환합니다. 

$p_{score} (W) = q(\lbrace (v_l, x_l), l=1,...,L \rbrace)$

여기서 $$q$$는 일반적인 neural network입니다. 

다들 아시다시피, 이 discriminator는 sequence가 real인지 fake인지 판별하면서 generator의 성능을 높이게 되며 positive sample들은 real graph로부터 오는 `RWR`, negative sample들은 위에서 정의한 generator로부터 오게 됩니다.


> **Incremental Learning on Graphs**

지금부터는 Continual Learning이 어떻게 이루어지는지 보겠습니다.

먼저 저자들은 affected nodes를 정의합니다. 

그래프가 time step에 따라 변하면서, 새로운 node나 edge가 생성되면 주위 K(`GNN`의 layer 수)-hop 이내의 neighborhood만 change 됩니다. (`GNN`의 layer가 2개라면, 한 node가 변할 때 그 node와 edge 2개 이내로만 연결되어 있는 node들만 변한다는 의미입니다.) 

Changed node중에 **크게 변한 것들**이 있을 것이고, **유의미한 변화가 없는 것들**이 있을 것입니다. 이 **크게 변한 것들**이 전체적인 neighborhood의 패턴을 바꿀 가능성이 있는 node 들이라, 학습에 사용해야합니다. 

> 다시 말해서, node 중에 성질이(feature) 크게 변한 node들은 새로운 data의 패턴을 반영할 확률이 성질이 변하지 않은 node보다 높으므로, model을 학습할 때 train data에 포함시켜서 학습시켜야 한다는 것입니다. 성질이 변한 node를 제쳐두고 변하지 않은 node만을 사용해서 학습한다면 model은 새로 들어온 data를 충분히 반영하지 못하겠죠.

그렇다면 어떤 node가 크게 변했다는 것을 어떻게 확인할 수 있을까요? 

저자들은 아래와 같은 influenced degree를 정의하고 그 influence degree가 threshold $$\delta$$ 보다 크다면 affected node라고 취급합니다.

$ \mathcal{V}_ C^t = \lbrace v \lVert F_ {\theta^{t-1}} (v, \mathcal{G}^t) - F_ {\theta^{t-1}} (v, \mathcal{G}^{t-1}) \rVert > \delta \rbrace$

위 식을 해석해보면, 어떤 node $$v$$의 이전 그래프 $$\mathcal{G}^{t-1}$$에서의 representation와 현재 그래프 $$\mathcal{G}^t$$에서의 representation이 많이 차이난다면, 이 node는 이전 그래프에서 현재 그래프로 넘어오면서 영향을 받았다고 보는 겁니다. 꽤 직관적인 해석입니다.

이런 affected node들은 이전 그래프가 가지고 있지 않은 새로운 패턴을 가지고 있으므로, Generative Model에 input으로 넣어 학습시킨 뒤에 다음 task부터 새로운 패턴을 반영해서 좋은 `replay buffer`를 만들도록 합니다.

추가로, 저자들은 간단한 filter를 추가해 generator가 생성한 node $$v_i$$가 affected node $$v_j$$와 **많이 비슷한 경우**, 패턴의 redundancy를 줄이기 위해 아래의 식처럼 필터링합니다.

$p_{reject} = max(p_{sim} (v_i, v_j) , j \subset \mathcal{V}_ C^t) \times p_r$$

여기서 $$p_r$$은 disappearacne rate로 사전에 정의하고, similarity는 다음과 같이 정의됩니다. 

$p_{sim} (v_i, v_j) = \sigma (- \lVert F_ {\theta^{t-1}}(v_i, \mathcal{G}^{t-1}) - F_ {\theta^{t-1}}(v_j, \mathcal{G}^{t-1})  \rVert)$

이때 $$\sigma$$는 sigmoid function이고, 위 식도 직관적으로 두 node의 representation의 차이가 적으면 비슷하다고 보는 겁니다.

이 filter를 통해 저자들은 중복되는 지식은 점차 잊혀지고 바뀌는 distribution이 안정적으로 학습될 것이라 했습니다.



아래의 알고리즘을 통해 지금까지 설명했던 내용들을 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/99710438/194888070-5da986d2-1702-4cd5-b77e-cfa3d76a0467.png)




## **4. Experiment**

> 본 논문에서 저자들은 다양한 dataset을 통해 baseline들과 `SGNN-GR`을 비교했습니다.

### **Experiment setup**

* Dataset
  * Cora
  * Citeseer
  * Elliptic (bitcoin transaction)
  * DBLP
* baseline
  * SkipGram models
    1. LINE
    2. DNE
  * GNNs (Retrained)
    1. GraphSAGE
    2. GCN
  * GNNs (Incremental)
    1. PretrainedGNN (첫 time step때만 학습되고 이후로는 학습하지 않음)
    2. SingleGNN (각 time step마다 한 번씩 학습)
    3. OnlineGNN (Continual Learning setting, without knowledge consolidation)
    4. GNN-EWC
    5. GNN-ER
    6. DiCGRL
    7. TWP
    8. ContinualGNN
  * `SGNN-GR`

여기서 Retrained `GNN`은 각 time step마다 Graph **전체**를 학습시킨 것으로, Continual Learning model 성능의 upper bound라고 생각하면 됩니다. Incremental `GNN`이 Continual Learning model들이라고 생각하시면 됩니다.


### **Result**

* Overall Results

위의 data를 사용한 실험의 결과는 아래와 같습니다. 저자들은 average Macro/Micro-F1를 성능 평가 지표로 사용했습니다.

![image](https://user-images.githubusercontent.com/99710438/195345047-bd69d686-e6d3-4ea6-ab81-4baff5f95e1e.png)

말씀드린대로, `LINE`, `RetrainedGCN`, `RetrainedSAGE`는 각 task에서 그래프 **전부**를 사용해서 Continual Learning setting의 성능을 상회합니다. 하지만 저자들의 `SGNN-GR`의 성능 또한 Retrained model과 유사한 것으로 보아 generator가 꼭 필요한 sample들만 생성해줬음을 알 수 있습니다. 

* Analysis of Catastrophic Forgetting

앞서 `catastrophic forgetting`을 방지하는 것이 Continual Learning에서 가장 중요한 포인트 중 하나라고 말씀드렸는데, 저자들의 모델은 얼마나 이전의 정보를 잘 기억했는지 보겠습니다.

![image](https://user-images.githubusercontent.com/99710438/195346345-51daec92-bc57-4c36-a6d5-a4b883a6aeb2.png)

왼쪽 (a) 그림은 Cora dataset에서 모델이 14 step을 가는동안 0번째 task를 얼마나 잘 기억하는지 보여주는 그래프이고, 오른쪽 (b) 그림은 6번째 task를 얼마나 잘 기억하는지 보여주 그래프입니다.

`OnlineGNN`은 이전 task의 정보를 거의 저장하지 못하는 것을 확인할 수 있고, 저자들의 방법론이 `GNN-ER`보다 더 이전 task의 지식을 잘 보존하는 것을 볼 수 있습니다.


* Anaylsis of Generative Model

그렇다면 과연 저자들이 `replay buffer`를 Generative Model로 생성한 것은 옳은 선택이었을까요?

![image](https://user-images.githubusercontent.com/99710438/195347882-15c5016a-3f55-4799-892a-4e73935493b6.png)

그림 (a) 는 실제 그래프의 label당 node 개수(파란색)와 Generative Model로 생성된 label당 node 개수(빨간색)을 보여줍니다. Generative Model이 실제 그래프의 label 분포와 굉장히 유사하게 node를 생성하고 있음을 보여줍니다.

또한 오른쪽 그림 (b) 는 generated 된 데이터를 보여주는데, 다양한 topological 정보를 담고 있음을 볼 수 있습니다.


* Ablation Study

![image](https://user-images.githubusercontent.com/99710438/195348924-c5e2fe7f-5238-4acb-a127-ba4bd18bdfbc.png)

마지막으로, 저자들은 `SGNN-GR`의 두 part들이 얼마나 성능 향상에 도움을 주는지 ablation study를 통해 Cora, Citeseer에서 확인했습니다. 

여기서 Non-Affected는 새롭게 추가된 node들만 고려하고, 그로 인한 affected node들은 고려하지 않은 모델입니다. 또한 Non-Generator는 모든 affected node를 찾아 다시 학습시키지만, generator는 쓰지 않은 모델입니다.

당연하게 `SGNN-GR`이 가장 좋은 성능을 보이는 것을 확인할 수 있습니다.

## **5. Conclusion**

> **Summary**

이 논문에서는 지속적으로 들어오는 Graph 데이터를 학습하는 데, Generative Model을 사용해 이전에 학습했던 그래프와 비슷한 그래프를 계속 생성해 새로운 데이터와 함께 학습시킵니다.

저자들은 여러 Continual Learning 방법 중 regularization method는 optimal solution을 얻는 것이 어렵다고 주장하고 replay based Continual Learning은 task가 진행됨에 따라 `replay buffer`에 그래프의 일부를 저장하고, task가 많이 늘어나면 그에 따라 요구되는 메모리도 커진다고 주장하며 Generative Model로 그때그때 `replay buffer`를 생성해서 메모리 효율을 높이겠다고 했습니다.

본 논문은 단순히 메모리 효율을 높인 것에 그치지 않고, 새롭게 등장하는 패턴은 적극적으로 학습하면서 불필요해 보이는 패턴은 줄이도록 학습해서 단순한 Continual Learning을 보완했습니다.

그 사이사이에 `Random Walk`가 아니라 `Random Walk with Restart`를 씀과 동시에 그 효율을 증명으로 보인 것과 같은 디테일, 본인들이 주장하는 모델의 장점을 잘 보여주는 알찬 실험들까지, 좋은 연구인 것 같습니다.

이 논문 뿐만 아니라 Continual learning에서 Generative Model은 중대한 역할을 할 것으로 보이며 관련 연구들이 꼭 필요할 것으로 보입니다.

추가적으로, 본 논문은 task incremental setting에서 generative model을 활용하고 있습니다만, 조금 더 어려운 setting(e.g. class incremental)에서의 활용 방안도 고안할 필요가 있다고 생각합니다.

Class incremental setting에서는 task incremental setting과 달리 한 번 등장한 class는 이후 task에서 다시 등장하지 않기 때문에 `GAN`같은 생성 모델을 활용하는 데 추가적인 전략이 필요할 것으로 보이며, 그런 경우에 task간의 similarity를 측정해서 활용하는 것도 하나의 future work가 될 것 같습니다.


> **개인적인 생각**

**올게 왔구나**

본 논문은 Graph Neural Network에서의 Continual Learning에 Generative Model을 접목시킨 방법입니다. 사실 이 논문이 나오는 것은 시간문제라고 생각하던 찰나에 역시나 등장했습니다.

이미 Continual Learning에 Generative Model을 접목시킨 연구는 꽤 오래전에(AI 연구의 속도가 매우 빠른 것을 감안하면) 등장했지만, GNN에 접목된 것은 없었기 때문이죠.

관련 연구를 하시는 분들은 아시겠지만, 이 논문이 novelty가 엄청 높다거나, 기존의 상식을 깨는 굉장한 발견을 한 논문이라기 보단.. (**분명히 좋은** 논문입니다, 오해금지)

가장 큰 contribution은 특정 분야에서 처음 시도된 연구, 적절한 시기에 등장한 연구인 것 같습니다. Novelty만을 좇는게 아니라, trend에 맞는 연구를 하는 능력도 필요해 보입니다. 

우리도 최신 논문을 잘 follow up 하는 '트렌디한' 연구자가 되도록 합시다.


***


## **Author Information**

* Wonjoong Kim
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Neural Network, Continual Learning
  * Contact: wjkim@kaist.ac.kr

## **Reference & Additional materials**

* Github Implementation
  * None
* Reference
  * [[AAAI-21] Overcoming catastrophic forgetting in graph neural networks with experience replay](https://ojs.aaai.org/index.php/AAAI/article/view/16602)
  * [[NIPS-17] Continual learning with deep generative replay](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)
