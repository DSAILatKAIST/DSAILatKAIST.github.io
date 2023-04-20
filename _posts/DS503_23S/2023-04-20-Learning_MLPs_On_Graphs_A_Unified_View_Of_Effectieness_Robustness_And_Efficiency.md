---
title:  "[ICLR 2023] LEARNING MLPS ON GRAPHS: A UNIFIED VIEW OF EFFECTIVENESS, ROBUSTNESS, AND EFFICIENCY"
permalink: Learning_MLPs_On_Graphs_A_Unified_View_Of_Effectieness_Robustness_And_Efficiency.html
tags: [reviews]
---

### Title
LEARNING MLPS ON GRAPHS: A UNIFIED VIEW OF EFFECTIVENESS, ROBUSTNESS, AND EFFICIENCY

#### 1. Motivation
Graph Machine Learning 은 non-Euclidea nsctructural data 인 graph 를 다루는 연구분야이다.<br>
기존 Machine Learning 과 달리, node 와 edge 로 이루어진 graph 를 학습하면서, node 들의 연결성과 관계에 대한 다양한 properties 를 발굴해내면서, Recommendation System, Social Netowrk, Traffic forcast 등 여러 분야에서 높은 Performance 를 보인다.<br>
하지만, structure data 를 사용하는데 발생하는 scalability 문제와 message passing 기법으로 인한 multi-hop data dependency 때문에, Graph Machine Learning 은 아직 현실세계에 적용하기에 많은 문제점들이 존재한다.<br>
Graph Machine Learning 의 scalability 문제를 해결하기 위하여, 최근에는 Graph based model 이 아닌 MLP 를 Graph Machine Learning 의 성능까지 끌어올리려는 연구가 진행되고 있다.<br>
주로 pretrain 된 GNN (Graph Neural Network) 를 MLP 로 knowledge distillation 하는 방식으로 진행되는 위 연구는, MLP 자체가 가지는 문제점 때문에 아직 효과적이지도, robust 하지도 못한 성과를 보이고 있다.<br>
그 이유로는 다음과 같다.<br>
- MLP 는 기본적으로 Euclidean space 의 data 를 다루는 것에 특화된 model 이다. 따라서, Graph 의 non-Euclidean 한 structure 특성을 MLP 에 반영할 수가 없다.
- 만약에, node label 이 graph structure 와 밀접한 연관이 있다면, MLP 의 embedding vector 와 label 의 space 가 다를 수도 있다.
- GNN based teacher network 에 MLP 를 매칭시키기 어렵다.
- Node feature 의 noise 에 대하여 너무 sensitive 하다.
따라서 본 연구에서는 다음과 같은 질문을 던진다.<br>
Can we learn MLPs that are graph structure-aware in both the feature and representation spaces, insentive to node feature noises, and have superior performance as well as fast inference speed ?<br>
MLP 가 GNN 을 따라갈 수 없는 이유와 본 연구에서 던지는 질문에서 확인할 수 있듯이, 가장 중요한 것은 MLP 가 structure 정보를 포착할 수 있게 만드는 것이다.<br>
본 연구는 이를 해결방안을 다음과 같이 제시한다.<br>
- Structure 정보를 포착하기 위하여, Graph 에서 node 의 position featurers 를 추출해 낸후, node feature 와 조합하여 MLP 의 input 으로 사용한다.
- GNN 에서 얻은 node similarity 정보를 MLP 로 전달한다.
- 마지막으로, MLP 의 robustness 를 증가시키기 위하여, adversarial feature augmentation 을 진행한다.
위와 같은 해결방안을 제시한 model 은 논문에서 NOise-robust Structure-aware MLPs On Graphs (NOSMOG) 로 부르게 된다.

#### 2. Preliminary
- Notation
Graph 는 다음과 같이 표기한다. $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \boldsymbol{\mathcal{C}})$ 여기서, $\mathcal{V}$ 는 node set 을, $\mathcal{E}$ 는 edge set 을, $\boldsymbol{\mathcal{C}} \in \mathbb{R}^{d _c}$ 는 node content attributes (i.e., node featuers)를 나타낸다.<br>
Node classification task 의 경우, 각 node $v \in \mathcal{V}$ 의 category (i.e., label) probability 를 예측하는 것으로, ground truth node label 은 $\boldsymbol{Y} \in \mathbb{R}^{K}$ 로 표기한다. 여기서 $K$ 는 category 의 개수를 의미한다.<br>
또한, ground truth label 을 가지고 있는 node set 을 $\mathcal{V}^{L}$ 로 표기하며, ground truth label 이 없는 node set 은 $\mathcal{V}^{U}$ 로 표기한다.

- Graph Neural Network
node $v \in \mathcal{V}$ 가 주어졌을 때, GNN 은 node $v$ 의 neighbor 인 $\mathcal{N}(v)$ 에서 message 를 aggregate 하여, node $v$ 의 embedding $\boldsymbol{h} \in \mathbb{R}^{d _n}$ 를 업데이트한다. <br>
$l$-th layer 의 node embedding 을 $\boldsymbol{h}^{(l-1)} _{v}$ 라고 할 때, neighbor 의 embedding 을 aggregate (denote AGG) 하고, 이전 layer 에서 얻은 node 의 embedding 과 combine (denote COM) 을 진행한다. 이는 $\boldsymbol{h}^{(l)} _{v} = COM(\boldsymbol{h}^{(l-1)} _{v}, AGG(\left\{\boldsymbol{h}^{(l-1)} _{u} : u \in \mathcal{N}(v)\right\}))$ 로 작성할 수 있다.

#### 3. Proposed Model
NOSMOG 는 GNN 을 MLP 로 distillation 하는 것을 기본으로 하여, 세개의 key components 가 존재한다. 우선 NOSMOG 의 overview 는 다음 그림과 같다.
![Figure1](https://user-images.githubusercontent.com/76777494/231966040-1986ce5e-aad7-4b43-8bab-b1cdd84045d8.png)
Distillation 은 figure 에서 $(a)$ 에 기술되어 있으며, 세가지 key components 는 figure 에서 $(b)$, $(c)$, $(d)$ 로 나타내져 있으며, $(b)$ 는 graph 의 position feature 를 얻어내는 과정이며, $(c)$ 는 representational similarity distillation, 마지막으로 $(d)$ 는 adversarial feature augmentation 을 나타낸다.<br>

- Training MLPs with GNNs Distillation
우선 figure 의 $(a)$ 에 해당하는 부분에 대하여 살펴보도록하겠다.<br>
GNN 을 MLP 로 knowledge distillation 하는 과정은 매우 간단하다. 우선적으로 GNN 을 pretrain 시킨다. 이 pretrain 된 GNN 을 teacher GNN 이라고 부른다. teacher GNN 으로부터 생성된 node 의 embedding vector 를 soft labels 이라고 부르며 $\boldsymbol{z} _{v}$ 로 표기한다.<br>
따라서, Knowledge distillation 은 다음과 같이 정의할 수 있다.<br>
$\mathcal{L} = \sum _{v \in \mathcal{V}^{L}} \mathcal{L} _{GT}(\hat{\boldsymbol{y}} _{v}, \boldsymbol{y} _{v}) + \lambda \sum _{v \in \mathcal{V}} \mathcal{L} _{SL}(\hat{\boldsymbol{y}} _{v}, \boldsymbol{z} _{v})$ (Eq 1).<br>
$\boldsymbol{y} _{v}$ 는 node $v$ 의 ground truth label 이며, $\hat{\boldsymbol{y}} _{v}$ 는 student 인 MLP 의 prediction 이며, 즉 node $v$ 에 대한 MLP 의 output 이다. $\mathcal{L} _{GT}$ 는 cross-entropy loss 이며, $\mathcal{L} _{SL}$ 은 KL-divergence loss, $\lambda$ 는 두 loss term 의 균형을 맞추기 위한 coefficient (hyper-parameter) 이다.

- Incorporating Node Position Features
이 부분에서는 figure 의 $(b)$ 에 해당하는 부분에 관하여 설명한다.<br>
Eq 1. 만을 사용하여, MLP 가 GNN 을 따라할 수 있도록 유도할 수는 있다. 하지만, message passing 기법을 사용하여, graph structure 정보를 사용할 수 있는 GNN 과는 달리, MLP 는 graph structure 를 다룰 수 없다. 이는 node content feature 와 label space 가 다를 경우 (e.g., node label 이 graph structure 와 큰 연관성이 있는 경우) 학습 결과에 큰 영향을 미칠 수 있다.<br>
이를 해결하기 위하여, MLP 가 node position 정보를 학습하게 만들어, graph structure 에서부터 나오는 information 을 다룰 수 있도록 만든다.<br>
간단하게, given graph $\mathcal{G}$ 에서 node $v \in \mathcal{V}$ 에 대하여 DeepWalk 를 진행하여, node $v$ 의 positional feature $\boldsymbol{P} _{v}$ 를 얻는다. 이 positional feature 는 graph 의 structure 정보를 내포하고 있는 data 로써, node content feature 를 사용하지 않고 DeepWalk 를 진행하여, node feature 의 정보가 전혀 반영되지 않고, graph structure 자체와 해당 grpah 에서 node positions 정보만들 사용한 positional information 이라고 볼 수 있다.<br>
모든 node 에 관하여, positional feature 를 얻어주고, 이를 node 의 기존 content feature 에 concatnate 하여, MLP 의 input 으로 사용한다.<br>
즉, Eq 1. 에서 사용된 $\hat{\boldsymbol{y}} _{v}$ 는 다음과 같이 표기할 수 있다.<br>
$\boldsymbol{X} _{v} = CONCAT(\boldsymbol{\mathcal{C}} _{v}, \boldsymbol{P} _{v}), \hat{\boldsymbol{y}} _{v} = MLP(\boldsymbol{X} _{v})$ (Eq 2).<br>

- Representational Similarity Distillation
이 부분은 figure 의 $(c)$ 에 해당되는 부분이다. <br>
Representational Similarity Distillation (denote RSD) 는 MLP 의 output 이 teacher GNN 의 output 에 strict 하게 matching 하는 것을 조금 완화시키고, MLP 가 soft structural node similarity 를 포착할 수 있도록 도와주는 loss 이다. RSD 는 node embedding 의 similarity 에 대한 정보를 담고 있기 때문에, MLP 가 GNN 의 representation space 를 학습할 수 있도록 도와주게 된다. Pretrain 되어있는 GNN (i.e., teacher GNN) 에서 생성한 모든 node 에 대한 representation 을 $\boldsymbol{H} _{G} \in \mathbb{R}^{N \times d _{G}}$, MLP (i.e., student) 에서 생성한 모든 node 에 대한 representation 을 $\boldsymbol{H} _{M} \in \mathbb{R}^{N \times d _{M}}$ 이라고 표기하면, 다음 식을 통하여, RSD 를 계산하기 위한 GNN, MLP 각각의 node similarity matrix 를 구할 수 있다.<br>
$S _{GNN} = \boldsymbol{H} _{G} \cdot (\boldsymbol{H} _{G})^{T} \text{ and } S _{MLP} = \boldsymbol{H'} _{M} \cdot (\boldsymbol{H'} _{M})^{T} \text{ , } \boldsymbol{H'} _{M} = \sigma (W _{M}\cdot \boldsymbol{H} _{M})$ (Eq 3).<br>
Eq 3. 에서, $W _{M} \in \mathbb{R}^{d _M \times d _M}$ 은 transformation matrix 이며, $\sigma$ 는 activation function (본 논문에서는 ReLu 를 사용하였다.), $\boldsymbol{H'} _{M}$ 은 MLP 의 representation 을 transformation 시킨 결과이다. Eq 3. 에서 얻은 $S _{GNN}, S _{MLP}$ 를 사용하여, RSD loss (denote $\mathcal{L} _{RSD}$) 를 다음 식과 같이 나타낼 수 있다. 본 논문에서는 RSD loss 에 Frobenius norm $\vert\vert\cdot\vert\vert _{F}$ 를 사용하였다.<br>
$\mathcal{L} _{RSD}(S _{GNN}, S _{MLP}) = \vert\vert S _{GNN} - S _{MLP} \vert\vert _{F}^2$ (Eq 4).<br>

- Adversarial Feature Augmentation
MLP 은 feature noise 에 대하여 굉장히 민감한 특징을 가지고 있다. 따라서, noise 에 대한 MLP 의 robustness 를 증가시키기 위해서, 본 논문에서는 adversarial feature augmentation 방법을 도입하였다. Node feaute 에 대한 (adversarial feature augmentation 에서 part 에서 언급하는 feature 는 node content feature 와 positional information 을 concatenate 한 feature 를 뜻함), 작은 fluctuations 에도 invariant 하고, 기타 다른 sample 들에 대하여 generalize 가 잘 될 수 있도록 한다.<br>
Node $v \in \mathcal{V}$ 에 대한 adversarial feature augmentation 과 해당 loss $\mathcal{L} _{ADV}$ 는 다음과 같이 정의한다.<br>
$\boldsymbol{X'} _{v} = \boldsymbol{X} _{v} + \delta, \hat{\boldsymbol{y'}} _{v} = MLP(\boldsymbol{X'} _{v})$<br>
$\mathcal{L} _{ADV} = \text{max} _{\delta \in \varepsilon} \left[- \sum _{v\in \mathcal{V}^{L}}\boldsymbol{y} _v \text{log}(\hat{\boldsymbol{y'}} _{v}) - \sum _{v\in \mathcal{V}}\boldsymbol{z} _{v} \text{log}(\hat{\boldsymbol{y'}} _{v})\right]$ (Eq 5).<br>
Eq 5. 에서 확인 할 수 있듯이, $\mathcal{L} _{ADV}$ 는, label 이 존재하는 node ($v \in \mathcal{V}^{L}$) 의 ground truth ($\boldsymbol{y} _v$) 와, graph 내 모든 node 들의 soft label ($\boldsymbol{z} _{v}$)를 사용하여, adversarial 를 학습하게 된다. 또한 $\text{max} _{\delta \in \varepsilon}$ 에서 확인 할 수 있듯이, 가능한 noise set 에서 가장 worst-case noise 를 선정하여 학습을 진행하게 된다. 이는 MLP 가 noise 에 대하여 더욱 robust 할 수 있도록 도와주게 된다.<br>
또한, 학습 과정에 있어서, 각 time step (epoch) 마다 worst-case noise 를 제공하기 위하여, noise $\delta$ 를 아래 식과 같이 학습하게 된다.<br>
$\delta _{t+1} = \prod _{\vert\vert\delta\vert\vert _{\infty} \leq \varepsilon} \left[ \delta _{t} + \boldsymbol{s}\cdot\text{sign}(\nabla _{\delta}(-\boldsymbol{Y}\text{log}(MLP(\boldsymbol{X} + \delta _t))))\right]$ (Eq 6).<br>
여기서 $\boldsymbol{s}$ 는 noise 의 step size 이며, $\nabla _{\delta}$ 는 $\delta$ 에 대하여 계산된 gradient 이다.

- Overall Loss
Final objective function $\mathcal{L}$ 은 Knowledge distillation ($\mathcal{L} _{GT}, \mathcal{L} _{SL}$), representational similarity distillation ($\mathcal{L} _{RSD}$), adversarial feature augmentation ($\mathcal{L} _{ADV}$) 의 합으로 아래 식과 같이 표현할 수 있다.<br>
$\mathcal{L} = \mathcal{L} _{GT} + \lambda \mathcal{L} _{SL} + \mu \mathcal{L} _{RSD} + \eta \mathcal{L} _ {ADV}$<br>
여기서, $\lambda, \mu, \eta$ 는 hyper-parameter 이다.

#### 4. Experiments
NOSMOG 에 대한 평가는 graph machine learning 에서 주로 사용하는 public benchmark dataset (Cora, Citeseer, Pubmed, A-computer, A-photo) 를 사용하였다.<br>
실험은 크게, **NOSMOG 에 대한 성능 평가**, **Inductive/Transductive setting 에서의 성능평가**, **Inference 속도에 대한 평가** 등으로 이루어져있다.

![Figure2](https://user-images.githubusercontent.com/76777494/232275417-98531f1c-abe0-4645-abb0-5ab9d6604ea2.png)
![Figure3](https://user-images.githubusercontent.com/76777494/232275443-f7a11c5c-b275-45e6-92db-db04bf6d5981.png)
Table 1, 2 에서 확인 할 수 있듯이, NOSMOG 는 GNN, MLP 와 비교하였을 때 전반적인 performance 뿐만 아니라, transductive, inductive setting 각각에서도 outperform 함을 확인할 수 있다.<br>
이는, graph 의 정보들을 적절한 방식으로 MLP 에 transfer 할 수 있다면, graph structure 정보를 explicit 하게 사용하는 graph based model 보다, graph structure 정보를 더 잘 포착할 수 있음을 보여준다. 또한, NOSMOG 와 같이 GNN-MLP 방식의 model 인 GLNN 은 large graph dataset 에서는 낮은 성능을 보이지만, NOSMOG 는 knowledge distillation 기반으로 GLNN 은 포착할 수 없는 추가적인 정보들을 사용하여, GLNN 보다 높은 성능을 보였다.

![Figure4](https://user-images.githubusercontent.com/76777494/232276249-fb56129b-fda3-463a-a948-48a98368df7e.png)
또한, Graph 의 가장 큰 단점인 inference 과정에서의 memory, time consuming issue 를 해결하였는지를 확인하기 위하여, 저자들은 accracy 와 inference time 에 관한 실험을 진행하였다. 위의 그림에서 확인 할 수 있듯이, NOSMOG 는 graph based model 보다 훨씬 빠른 inference time 을 보이면서도 높은 성능을 기록하는 모습을 보여주었다. 즉 NOSMOG 와 GLNN 에서 볼 수 있듯이, GNN-MLP model 은 graph based model 의 큰 한계점인 memory, time consuming issue 를 해결 할 수 있는 방법임을 증명하였다.

![Figure5](https://user-images.githubusercontent.com/76777494/232276741-77925b73-f928-454e-b62b-af5d79985f91.png)
마지막으로, ablation study 에 대한 분석을 진행하겠다. Ablation study 에서 확인 할 수 있듯이 NOSMOG 의 많은 component 들이 model performance 에 큰 기여할 하고 있음을 확인할 수 있다. 특히, graph 의 positional 정보들을 처리하는 **w/o POS** (i.e., Incorporating Node Position Features) 가 NOSMOG 의 performance 에 가장 큰 영향을 미치는 것을 확인할 수 있었다. 이는 결국 structure 구조로 이루어진 graph data 를 학습할 때에는, graph structure 정보를 포착하는 것이 그 무엇보다 중요함을 시사하는 바이다.

##### 5. Conclusion
본 논문에서는 MLP 를 사용하여, Graph data 를 학습하는 GNN-MLP 관련 연구에서 현재까지 한계점이었던, effectiveness, robustness, efficiency 문제를 해결하였다.<br>
특히, node position features 를 결합하여 graph structure 정보를 MLP 로 전달하고, representation similarity 에 대한 knowledge 를 distillation 함으로 structure-aware 를 성공적으로 이루었다. 또한, noise 에 취약한 MLP 를 adversarial feature augmentation method 로 noise 에 robust 한 model 을 성공적으로 구현하였다.