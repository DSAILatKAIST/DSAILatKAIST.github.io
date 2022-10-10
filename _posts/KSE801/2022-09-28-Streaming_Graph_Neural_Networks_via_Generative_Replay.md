---
description: >-
  Wang, Junshan et al./ Streaming Graph Neural Networks with Generative Replay /
  KDD-2022
---

# Streaming Graph Neural Networks with Generative Replay

## **1. Problem Definition**

> **Continual learning에서 사용할 replay buffer를 Generative model을 활용시켜 생성한다!**

본 논문은 Continual learning을 `Graph Neural Network(GNN)`과 접목시킴과 동시에 이에 사용할 replay buffer를 Generative model을 사용해 생성합니다.

이미 Continual learning에 Generative model을 사용한 연구는 예전에도 있었으나 [(여기)](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html), 이는 이미지 데이터를 주로 targeting한 반면에 지금 다루는 논문은 Graph domain의 데이터에 적용, 그에 맞는 특성(structure of graph)를 고려했다는 점에서 novelty가 있습니다.

> **Continual learning이란?**

Continaul learning은 Lifelong learning, Incremental learning이라고도 불리며, 과거의 정보를 최대한 유지하며 새로운 정보를 학습하는 방법론입니다.

예를 들어, 인간이 '강아지' 라는 지식을 알고 있는 상태로, '고양이'라는 지식을 새로 습득했을 때, '강아지'를 잊지 않고 '강아지'와 '고양이'를 구별해 낼 수 있는 것 처럼, 지속적으로 들어오는 새로운 데이터를 학습함과 동시에 이전에 학습되었던 데이터를 잊지 않도록 인공지능을 설계하는 것이 Continual learning의 목적입니다.

> **Catastrophic Forgetting이란?**
> 
Continual learning에서, 새로운 데이터가 들어옴에 따라 이전에 학습했던 데이터의 정보를 망각하는 현상을 `Catastrophic Forgetting`이라고 합니다. 아래 그림을 보시겠습니다. 

![image](https://user-images.githubusercontent.com/99710438/194873406-84f4a722-c562-4c39-adec-5ecc498a498f.png)

그림에서 볼 수 있듯이, Task 1에서는 파란색 node들을 구별하도록 학습합니다. Task 2에서는 새로운 보라색 node가 추가되면서 파란색과 보라색을 포함해 학습시키고, Task 3에서는 빨간색의 새로운 node가 추가되면서 새롭게 학습이 진행됩니다. 이 과정이 Continual learning 입니다.

그리고 Task가 진행됨에 따라 이전 Task에서 학습했던 node들에 대한 예측 성능이 떨어지는 것을 볼 수 있습니다. 

예를 들어, Task 1에서 파란 node들을 분류하는 데에는 95%의 성능을 보였으나, Task 2에서는 55%로 줄었고, Task 2에서 학습했던 보라색 node들도 Task 3에서의 성능은 현저히 줄어든 것을 볼 수 있습니다. 이러한 현상이 앞서 말씀드린 `Catastrophic Forgetting`입니다. 

Continual learning 중 Replay approach는 이전 Task에서 학습했던 데이터 중 **일부**만을 sampling하여 이후 Task에 사용합니다. 전체 데이터를 계속해서 누적하여 학습하면 효율이 좋지 않기 때문이죠. 이렇게 sampling되어 이후 Task에서 같이 학습될 data를 `Replay buffer`라고 부릅니다. 이 과정에서 `Catastrophic Forgetting`현상이 일어나게 되는데, 이 현상을 최소화 하도록 이전 Task에서 학습했던 데이터를 잘 대표하는 데이터를 sampling하는 것이 관건입니다.

## **2. Motivation**

> **기존 Replay based Continual learning 방법들은 storage limitation을 가진다!**

앞서 언급한대로, Replay based Continual learning은 이전에 학습했던 데이터 중 전체 데이터를 잘 represent하는 **일부**의 데이터를 sampling해서 이후 Task에서 등장하는 데이터와 같이 학습시킵니다. 

하지만 이럴 경우 Task가 계속해서 진행됨에 따라, 각 Task에서 아무리 적은 데이터를 sampling한다 해도 기억해야하는 데이터가 지속적으로 늘어나고, 결국에는 memory를 다 사용해버리는 일이 생길 것입니다. 

저자들은 이런 한계를 지적하며 Generative model을 도입해 `Replay buffer`에 데이터를 누적시키는 것이 아니라, Task가 시작될 때 마다 필요한 만큼 Generative model로 `Replay buffer`를 생성해 학습을 진행하고자 합니다. 

> **기존 Replay based Continual learning 방법들은 온전한 graph distribution을 보존하지 못한다!**

Image 도메인과 달리, Grpah 도메인의 데이터를 다룰 때는 각 데이터의 성질(feature)뿐 만 아니라 그래프의 전체적인 structure도 고려해야 합니다. 어떤 node가 어떤 node와 연결되어 있으며, 연결된 node들은 어떤 특성을 가지고 있는지까지 종합적으로 고려되어야 한다는 것이죠. 

저자들은 Continual learning 중에서 `Replay buffer`를 생성할 때 각 node들의 feature만 고려될 뿐, 전체적인 그래프의 distribution(structure)이 보존되지 못한다고 주장합니다. 이는 Grapn Neural Network가 학습될 때 성능 저하를 야기하는 가장 큰 문제 중 하나로, 이 논문에서는 Generative model을 통해 이러한 topological information까지 저장하도록 하는 것을 목표로 합니다.


## **3. Method**

> **Preliminaries: `GNN`**

논문에서 제안한 방법론을 이해하기 위해서는 `GNN`의 개념을 알고 있어야 합니다.

본 리뷰에서는 간단하게 소개를 하겠습니다.

$$N$$개의 노드를 가진 그래프 $$\mathcal{G}= \lbrace \mathcal{V},\mathcal{E} \rbrace$$가 주어지고, $$X = \lbrace x_{1}, x_{2}, ..., x_{N} \rbrace$$ 을 node feature의 집합이라고 하고, $$A$$를 node들의 관계를 표현하는 adjacency matrix라고 하겠습니다.

$$l-th$$ hidden layer에서의 $$v_{i}$$의 hidden representation을 $$h_{i}^{(l)}$$ 이라고 할 때, 이 $$h_{i}^{(l)}$$는 다음과 같이 계산됩니다:

$$h_{i}^{(l)} = \sigma(\sum_{j \subset \mathcal{N}(i)} \mathcal{A_{ij}}h_{j}^{(l-1)}W^{(l)})$$

이 때, $$\mathcal{N}(i)$$ 는 $$v_{i}$$의 neighbors를 의미하고, $$\sigma ( \bullet )$$는 activation function, $$W^{(l)}$$은 $$l-th$$ layer의 transform matrix를 나타냅니다.

$$h_{i}^{(0)}$$은 node $$v_{i}$$의 input feature를 나타내고, $$\mathcal{A}$$는 neighbors의 aggregation strategy이며, `GNN`의 핵심 중 하나입니다.

본 논문에서는 다양한 `GNN`중 `GraphSAGE`라는 모델을 사용하는데, 이 `GraphSage`의 $$k$$번째 layer는 다음과 같이 정의됩니다:

$$h_{v}^{k} = \sigma(W^k \cdot MEAN( \lbrace h_v^{k-1} \rbrace \cup \lbrace h_u^{k-1}, \forall u \in \mathcal{N}(v)\rbrace)$$

> **Problem Definition**

Continual learning setting에서, 데이터는 그래프의 형태를 띠고 연속적으로 들어옵니다. 이는 다음과 같이 표현이 가능합니다.

$$\mathcal{G} = (\mathcal{G}^1, \mathcal{G}^2, ..., \mathcal{G}^T)$$ 

where $$\mathcal{G^t} = \mathcal{G}^{t-1}+\Delta \mathcal{G}^t$$

여기서 $$\mathcal{G} = (A^t, X^t)$$ 는 attributed graph at time $$t$$이고, $$\Delta \mathcal{G} = (\Delta A^t , \Delta X^t)$$는 time $$t$$에서의 node attribute와 network의 structure의 변화량을 나타냅니다.

이 때 Streaming GNN은 traditional GNN을 streaming setting으로 확장한 것이 됩니다. Streaming graph가 있을 때, continual learning의 목적은 $$(\theta^1, \theta^2, ..., \theta^T)$$ 를 배우는 것입니다. 이 때 $$\theta^t$$ 는 time $$t$$ 에서의 GNN parameter를 의미합니다. 


> **Model Framework**

저자들은 이 논문에서 `SGNN-GR`이라는 방법론을 제시합니다. 모델 구조는 아래 그림과 같습니다.

![image](https://user-images.githubusercontent.com/99710438/194887946-3f736cc4-1c2c-47ca-97aa-4516da0ae42e.png)

지금부터 `SGNN-GR`의 자세한 내용을 살펴보겠습니다. 

Streaming GNN의 time $$t$$에서의 loss는 다음과 같습니다.

$$\mathcal{L}(\theta^t ; \mathcal{G}^t) = \mathcal{L}(\theta^t ; \mathcal{G}_A^t) + \lambda \mathcal{R} (\theta^{t-1} ; \mathcal{G}_S^t)$$

우변의 첫 항은 incremental learning에 관한 것이고, 두 번째 항은 historical knowledge에 관한 것입니다. 

본 논문에서 $$\mathcal{G}_A^t$$ 는 graph의 affected part, $$\mathcal{G}_S^t$$ 는 grpah의 stable part로 정의합니다. 

이 때 $$\Delta \mathcal{G}^t \subset \mathcal{G}_A^t$$ 이고 $$\mathcal{G}_S^t \subset \mathcal{G}^{t-1}$$ 입니다. 몇몇 node들이 새롭게 바뀐 node들에 대해서 영향을 받는 것입니다.

각 time step에서 모델은 main model(`GNN`)과 generative model로 구성됩니다. 위 그림에서 확인할 수 있듯이, generative model은 $$\mathcal{G}_A^t$$에서 바뀐 node들과 $$\mathcal{G}^{t-1}$$에서의 replayed node를 training data로 받습니다. 이 때 replayed node는 이전 time step의 generative model로부터 나옵니다. 

이 논문에서는 generative model로 `GAN`을 사용하였습니다. `GAN`에 대한 자세한 설명은 생략하며, 원 논문은 [여기](https://dl.acm.org/doi/abs/10.1145/3422622)를 참고하시기 바랍니다. 

`GNN` 모델도 changed node와 replayed node를 똑같이 input으로 받습니다. 

Main model의 loss function은 다음과 같습니다.

$$\mathcal{L}_{GNN} (\theta^t) = r$$

> **Generative Model for Node Neighborhood**


> **Incremental Learning on Graphs**



아래의 알고리즘을 통해 지금까지 설명했던 내용들을 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/99710438/194888070-5da986d2-1702-4cd5-b77e-cfa3d76a0467.png)




## **4. Experiment**

> 본 논문에서 저자들은 다양한 baseline과 실험을 통해 `ODE-RNN`과 `Latent ODEs`를 비교했습니다.

### **Experiment setup**

* Dataset
  * Toy dataset (extrapolation)
  * MuJoCo (extrapolation, interpolation)
  * Physionet (time-series prediction)
  * Human Activity (time-series prediction)
* baseline
  * Autoregressive model
    1. **ODE-RNN**
    2. RNN
    3. RNN-Decay
    4. RNN-Impute (missing values imputed by weighted average of previous value)
    5. GRU-D (GRU-Decay)
  * Encoder-Decoder model
    1. **Latent ODE**
    2. RNN-VAE
    3. ODE-RNN
* Evaluation Metric
  * Mean squared error
  * AUC
  * Accuracy

### **Result**

* Toy dataset

저자들은 1000개의 periodic trajectories를로 toy dataset을 만들었습니다.

그리고 `RNN`을 encoder로 쓴 `Latent ODE`와 `ODE-RNN`을 encoder로 쓴 `Latent ODE`로 각 trajectory의 20%를 학습시킨 뒤, 다음을 trajectory를 예측하도록(extrapolation) 했습니다.

![Approximate posterior smaples](https://user-images.githubusercontent.com/99710438/164261107-8f595251-839d-4fd2-90a6-c2c71af14e24.png)

위 그림에서 확인할 수 있듯이, `ODE-RNN`을 encoder로 쓴 `Latent ODE`는 training data를 한참 넘는 구간을 periodic dynamics을 유지하면서 잘 extrapolate 합니다.

반면에, `RNN`을 encoder로 쓴 `Latent ODE`는 periodic dynamics를 잘 extrapolate 하지 못하는 것을 확인할 수 있습니다.

* MuJoco Physics Simulation

이 데이터는 어떤 물체가 껑충 뛰는 physical simulation으로 이루어져 있습니다. 각 hopper의 initial position과 velocity를 sampling 하고, 이 trajectory들은 initial state에 대한 function으로 이루어져 있습니다. 저자들은 이 데이터에 대해 interpolation과 extrapolation을 각각 진행하고, MSE를 측정했습니다.

![MSE(\*0.01) on the MuJoCo dataset](https://user-images.githubusercontent.com/99710438/164263996-b1907e81-c7e9-4848-9c7c-8bae5343434b.png)

위 표는 각각 10, 20, 30, 50%의 observation을 주고 autoregressive 모델과 Encoder-Decoder(Latent model) 모델로 interpolation과 extapolation을 한 결과입니다.

위 표에서 볼 수 있듯이, Interpolation에서는 Autoregressive 모델의 `ODE-RNN`이, Encoder-Decoder 모델의 `Latent ODE`(`ODE-RNN` encoder)가 성능이 가장 좋게 나왔습니다.

Extrapolation에는 Encoder-Decoder 모델은 같은 결과가 나왔으나 Autoregressive 모델에서는 `ODE-RNN` 모델의 성능이 좋지 않은 것을 확인할 수 있었습니다. 이는 autoregressive model은 one-step-ahead prediction을 위해 training 되었으므로 예견된 결과라고 합니다.

주목할 것은 `RNN`과 `ODE-RNN`의 성능 차이가 데이터가 sparse해 질수록(observation이 적어질수록) 커진다는 것입니다. 이를 통해 ODE 기반 모델이 sparse한 데이터에도 더 적합하다는 것을 확인할 수 있었습니다.

저자들은 또한 latent state의 norm이 trajectory에 따라 어떻게 변화하는지도 확인했습니다.

![Trajectory from MuJoCo dataset & Norm of the dynamic functions](https://user-images.githubusercontent.com/99710438/164266880-12d49223-d6fb-4e44-9187-580a754236ba.png)

위 그림에서 확인할 수 있듯이, `Latent ODE`는 data의 trajectory를 잘 따라가는 것을 확인할 수 있었습니다.

또한, `Latent ODE`의 norm은 trajectory가 급변할 때(hopper가 땅을 박차고 올라올 때) norm이 변하는 반면, `RNN`의 norm은 특별한 규칙 없이 변하는 것을 확인할 수 있었습니다.

이는 `Latent ODE`가 `RNN`보다 hidden state에 더 유의미한 정보를 담고있는 것을 의미합니다.

* Physionet

이 데이터는 8000개의 time-series 포인트로 구성되어 있고, irregular time step과 sparse한 것이 특징입니다. 여기서 저자들은 observation time에 Poisson Process likelihood를 포함시켜 Latent ODE 모델과 같이 학습시켰을 때의 성능도 확인해 봤습니다.

![MSE on PhysioNet, Autoregressive models](https://user-images.githubusercontent.com/99710438/164268642-c8f5bfd2-e176-41c9-a077-dfd5f93aaff0.png)

![MSE on PhysioNet, Encoder-Decoder models](https://user-images.githubusercontent.com/99710438/164268796-d70189f3-e74d-4224-b3be-2bb398bc736f.png)

위 테이블에서 확인할 수 있듯이, Autoregressive 모델과 Encoder-Decoder 모델에서 역시 저자들의 모델이 다른 baseline보다 좋은 성능을 내고 있습니다.

* Human Activity dataset

이 데이터에는 다섯가지 activity(걷기, 앉기, 눕기 등)에 대한 time series data가 포함되어 있습니다.

![Per-time-point classification, accuracy on Human Activity](https://user-images.githubusercontent.com/99710438/164271166-69bc6eb2-3159-46f3-aff4-1c48df1c9755.png)

이 데이터에서도 저자들의 모델의 성능이 다른 모델의 성능보다 좋은 것을 확인할 수 있었습니다.

## **5. Conclusion**

> **Summary**

이 논문에서는 hidden state dynamics를 `Neural ODE`로 구성한 `ODE-RNN`을 소개했습니다.

또한 이 모델을 `VAE`의 encoder로 사용한 `Latent ODE`도 제안했습니다.

이를 통해 지금까지 **discrete한 hidden layer**를 가졌던 모형들이 아닌, **continuous한 hidden layer**를 가진 모형으로서 기존 방법론들의 단점(irregular time step, sparse data에서 성능이 저하되는 현상)을 극복할 수 있었습니다.

`Latent ODE`는 비교적 hidden state에 대한 설명력을 가지며 **observation time에 구애받지도, 전처리 과정에 data를 impute 할 필요도 없습니다**.

이에 수많은 irregularly-sampled time series data에 적용 가능할 것으로 보입니다.

> **내 생각...**

본 논문은 2018년 NeurIPS에서 best paper를 받은 `Neural ODE`를 `RNN`과 `VAE`에 적용시킨 후속 연구입니다.

Neural ODE라는 새로운 방식을 여러 방면에 접목시킨 논문들이 우후죽순 생겨나고 있습니다.

처음 시도되는 방법론이다 보니 특별한 theoretical contribution이 없어도 접목만 잘 시키면 논문이 publish 되기가 용이한 것 같습니다.

우리도 지금 어떤 연구가 trend인지 잘 follow up하는 자세가 필요할 것입니다.

또한 연구도 융합의 시대인 것 같습니다. 분야를 가리지 않고 여러 방법론을 창의적으로 녹여내는 것이 새로운 연구의 창을 열 수 있을 것입니다.

***

## **Author Information**

* Wonjoong Kim
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Nerual Network, Continual Learning
  * Contact: wjkim@kaist.ac.kr

## **6. Reference & Additional materials**

* Github Implementation
  * None
* Reference
  * [[AAAI-21] Overcoming catastrophic forgetting in graph neural networks with experience replay](https://ojs.aaai.org/index.php/AAAI/article/view/16602)
  * [[NIPS-17] Continual learning with deep generative replay](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html)
