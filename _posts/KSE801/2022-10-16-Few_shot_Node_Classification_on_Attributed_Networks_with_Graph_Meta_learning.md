---
title:  "[SIGIR 2022] Few-shot Node Classification on Attributed Networks with Graph Meta-learning"
permalink: Few_shot_Node_Classification_on_Attributed_Networks_with_Graph_Meta_learning.html
tags: [reviews]
use_math: true
usemathjax: true
---

## **1. Introduction**  

### 1-1. Preliminaries  
 오늘 소개하고자 하는 논문의 제목에서 키워드를 뽑아보자면, `Few-shot`, `Attributed Networks`, `Meta-learning`정도로 추릴 수 있다. 결국 Graph 같은 ***Attributed Networks***(e.g. molecular graph, social network)***에 대해서 Few-shot learning task를 Meta-learning의 방식으로 해결***하고자 한다. 본격적으로 논문을 파헤치기 전에, 논문을 이해하는데 필수적으로 필요한 Few-shot learning과 Meta-learning의 개념에 대해 간략하게 설명하도록 하겠다. 두 개념에 대한 설명은 [Graph Meta Learning via Local Subgraphs (NIPS20)에 대한 리뷰포스트](https://dsail.gitbook.io/isyse-review/paper-review/2022-spring-paper-review/neurips-2020-g-meta)를 참조하였다.  
 
#### 1-1-1. Few-shot Learning   

Few-shot Learning은 적은 데이터를 가지고 효율적으로 학습하는 문제를 해결하기 위한 학습 방법이다. 

<div align="center">

<img width="564" src="https://user-images.githubusercontent.com/37684658/164231019-868292bd-9cbf-4d15-87cb-24d621ed78d6.png">
  
</div>

예를 들어, 위와 같이 사람에게 아르마딜로(Armadillo)와 천갑산(Pangolin)의 사진을 각각 2장씩 보여줬다고 생각해보자. 아마 대부분의 사람들은 아르마딜로와 천갑산이 생소할 것이다. 자, 이제 그 사람에게 다음의 사진을 한 장 더 보여주었다.  

<div align="center">

<img width="564" src="https://user-images.githubusercontent.com/37684658/164224487-822f266a-98db-4d2d-9c41-7303fdccf1ff.png">

</div>


위 사진의 동물이 아르마딜로인지, 천갑산인지 맞춰보라고 하면, 너무나 쉽게 천갑산임을 자신있게 외칠 수 있을 것이다. 사람들은 어떻게 이렇게 적은 양의 사진을 보고도, 두 동물을 구분할 수 있는 능력을 가지게 되었을까? 사람과는 달리 기존 머신러닝(Machine Learning)은 저 두 동물을 구분하기 위해 많은 양의 사진을 보고 학습하여야 할 것이다. 만약 모델이 아르마딜로와 천갑산을 잘 구분할 수 있게 되었다고 하자. 이제 갑자기 아래 두 동물을 구분하라고 하면 어떻게 될까?  

<br/> 

<div align="center">

<img width="564" src="https://user-images.githubusercontent.com/37684658/164231266-515ab539-110b-4835-971c-287fb759c44a.png">
<!-- ![image](https://user-images.githubusercontent.com/37684658/164231266-515ab539-110b-4835-971c-287fb759c44a.png) -->

</div>


두더지(Mole)는 모델이 처음 보는 동물이기 때문에 두 동물을 구분하려면 다시 두더지에 대한 사진을 학습을 해야할 것이다. 하지만 사람은 여전히 두 동물을 쉽게 구분할 수 있다. 사람과 같이 적은 양의 사진만 보고도 Class를 구분할 수 있는 능력을 학습하는 것이 Few-shot Learning이고, 이를 학습하기 위해 Meta-Learning의 학습 방법을 활용한다.  

Few-shot Learning에서 쓰이는 용어를 정리하고 넘어가면, 처음 모델에게 제시해주는 Class별 대표사진들을 `Support Set`이라고 한다. 2개의 Class로 구성되어 있다면 2-way라고 하며, Class별로 2장의 대표사진을 보여준다면 2-shot이라고 한다. 그리고 1장의 새로운 사진을 보여주는 데 이렇게 맞춰보라고 보여주는 사진들을 `Query Set`이라고 하며, 1번 맞춰보라고 주었으니 Query는 1개이다. Support Set과 Query Set을 합쳐서 하나의 `Task` 또는 `Episode`라고 지칭한다.  


#### 1-1-2. Meta Learning  

메타러닝(Meta Learning)은 새로운 task에 대한 데이터가 부족할 때, Prior Experiences 또는 Inductive Biases를 바탕으로 빠르게 새로운 task에 대하여 적응하도록 학습하는 방법을 말한다. 'Learning to Learn'이라는 용어로 많이 설명되곤 하는 데, 대표적인 접근 방법으로는 거리 기반 학습(Metric Based Learning), 모델 기반 학습 (Model-Based Approach), 그리고 최적화 학습 방식(Optimizer Learning)이 있다. 이 중, 오늘 소개하고자하는 Meta-GPS 논문을 제대로 이해하기 위해서는 최적화 학습 방식의 MAML[3]에 대한 이해가 선행되어야 한다.  

##### MAML (Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks)  
MAML은 최적화 학습 방식의 Meta Learning 방법론으로서 가장 대표적인 논문이라고 할 수 있다. 전체적인 개념은 어떤 Task에도 빠르게 적응(Fast Adaptation)할 수 있는 파라미터를 찾는 것이 이 모델의 목적이다. 일반적으로 딥러닝 모델은 기울기의 역전파를 통해 학습을 진행하나, 이런 학습 방법은 데이터가 충분히 많을 때 잘 작동하는 방식이다. 이 모델은 Task 하나하나에 대한 그래디언트를 업데이트 하는 inner loop와, 각 태스크에서 계산했던 그래디언트를 합산하여 모델을 업데이트하는 outer loop 단계로 이루어져있다(아래 그림에서의 실선). 공통 파라미터 <img width="564" src="https://user-images.githubusercontent.com/37684658/164229776-12b52e66-cf43-4b8e-ba97-ee0ccb723724.png">는 Task agnostic하게 빠르게 적응할 수 있는 파라미터이고, 다시 모델이 이 파라미터로부터 어떤 Task를 학습하게 되면 그 Task에 최적화된 파라미터를 빠르게 찾을 수 있게 된다.

<!-- ![image](https://user-images.githubusercontent.com/37684658/164229776-12b52e66-cf43-4b8e-ba97-ee0ccb723724.png) -->

  
<div align="center">

<img width="564" src="https://user-images.githubusercontent.com/37684658/164233736-dd00ab2f-adf4-42b9-a491-6def82a126d4.png">

<!-- ![image](https://user-images.githubusercontent.com/37684658/164233736-dd00ab2f-adf4-42b9-a491-6def82a126d4.png) -->

</div>

### 1-2. Problem Definition  
Citation networks, social media networks, traffic networks와 같은 attribute networks는 실생활에서 발생할 수 있는 수많은 문제들을 풀기 위해 활용되고 있다. Attribute networks들의 node classification 문제는 fundamental하면서도 굉장히 중요한 task이다. 이 문제를 풀기위해 많은 graph neural network(GNN) 모델들이 연구되었지만, 이들은 모두 labeled data가 충분한 상황을 가정하고 있다. Section 1-1에서 설명한 바와 같이, labeled data가 충분하지 않은 상황에서 node classification의 할 수 있는 방안을 모색하고자 GNN 분야에서도 Few-shot learning에 대한 연구가 점차 활발해지고 있고, 이를 풀기 위해 (graph) meta-learning 기반의 많은 방법들이 나오고 있다. 하지만 본 논문은 기존 Graph Few-shot Learning 모델에서 가지고 있는 3가지 limitation을 제시하고 해결하고자 한다.  

#### (I) Attributed networks가 homophilic이라는 가정은 real-world attributed networks에서 unreasonable하다.  
Attributed networks가 homophilic하다는 것은 같은 class의 nodes들은 연결되어 있는 경향이 높다는 것이다. 하지만 real-world attributed networks들은 단순히 두 node가 연결되어 있다고 해서 같은 라벨을 가지지 않을 수 있다. 예를 들어, dating graphs에서 gender class가 다른 서로 다른 이성들이 연결될 확률이 높은 것은 자명하다. 따라서 본 논문은 Attributed Networks를 homophilic이 아닌, heterophilic 가정하에 encoding하는 network encoder를 제안한다.  
#### (II) MAML기반의 meta-learning method(Meta-GNN, [G-Meta](https://dsail.gitbook.io/isyse-review/paper-review/2022-spring-paper-review/neurips-2020-g-meta))는 모든 data에 대해 하나의 initialization을 공유한다.  
하지만, attributed networks들의 data들은 independent and identically diistributed(*i.i.d*)가 아니기 때문에 모든 데이터들에 single initialization point를 공유하는 것은 적합하지 않다. 이를 해결하기 위해 sample된 data에 따라 MAML을 adaptive하게 initialization할 수 있는 방법을 제시한다.  
#### (III) Sample된 Task(episode)를 모두 equal하게 다루고 있는 문제가 있다.  
Few-shot learning의 task는 labeled data들로 구성된 support set, 그리고 model이 class를 prediction해야하는 query set로 이루어진다. 다시 말해, 모델은 몇 안되는 labeled data로 각 class에 대한 정보를 습득하고, 처음보는 data(query set)의 class를 맞추어야 한다. Task를 구성하는 방법은, 3way-3shot-5qry task인 경우, 3개의 class를 random으로 선택하고, class당 3개씩 data를 sample하여 support set을 구성하고, class당 5개씩 더 sample하여 query set을 구성한다. 즉, task에는 총 9개의 support set이 있고, 이 데이터를 통해 학습한 model은 총 15개의 query data의 class를 predict하여야 한다.  
하지만 이런 방법으로 만들어진 task들은 서로 큰 차이가 있을 수 있다. 예를 들어, social networks에서 user node의 종류가 유명인들으로 구성된 task와 일반인들로 구성된 task는 구조적인 패턴이 다르게 나타날 것이기 때문에 이런 차이를 인지하고 adaptive하게 다루어야 할 것이다. 본 논문은 task마다 learnable parameter를 adaptive하게 scaling하는 방법으로 이 문제를 다루고자 한다.  


### 1-3. Annotation    
<div align="center">
 
<img width="564" src="https://user-images.githubusercontent.com/37684658/195105466-015e14b8-3a97-4fea-af2b-ea93c69fe6cc.png">

<!-- ![image](https://user-images.githubusercontent.com/37684658/195105466-015e14b8-3a97-4fea-af2b-ea93c69fe6cc.png)   -->
 
</div>  
Few-shot learning에서는 train task와 test task 모두 support set과 query set으로 이루어져있다. 다시 말해, Train/Test 상관 없이 support set으로 query set의 label을 맞추는 task를 하는 것인데 주목해야할 점은 train task에 들어가는 데이터의 class와 test task에 들어가는 데이터 class가 겹치지 않는다는 점이다. 즉, test 때는 unseen class, unseen nodes들로 구성된 task를 풀어낸다. 결론적으로 하고자하는 것은 test 때의 상황을 train 때도 모방해서 학습하자는 것이다(episodic training).  
$\prod$개로 구성된 N-way K-shot meta-training tasks들은 다음과 같이 표현할 수 있다. 

$
\mathcal{T}_ {tr}=\lbrace \mathcal{T} \rbrace_ {i=1}^{\prod}, \mathcal{T}_ {i}=\{\mathcal{S}_ i, \mathcal{Q}_ i\} ,
$

$
\mathcal{S}_ i=\lbrace(v_{i,1},y_{i,1}),(v_{i,2},y_{i,2}),...,(v_{i,k},y_{i,k})\rbrace_ {k=1}^{N\times K},
$

$
\mathcal{Q}_ i=\lbrace(\bar{v}_ {i,1},\bar{y}_ {i,1}),(\bar{v}_ {i,2},\bar{y}_ {i,2}),...(\bar{v}_ {i,k},\bar{y}_ {i,k})\rbrace_ {k=1}^{N\times K},
$
 
## **3. Method**  
모델의 전체적인 구조는 다음과 같다.  

<div align="center">
 
<img width="812" alt="image" src="https://user-images.githubusercontent.com/37684658/195232172-ae904c09-613b-4aec-a211-31054e064943.png">

 </div>

3-1에서 Network Encoder에서는 attributed network를 heterophilic한 가정으로 인코딩하는 과정을 설명하고, 3-2에서는 샘플링 된 task에 속한 class에 따라 parameter를 initialization하는 과정을 설명한다. 그리고 3-3에서는 task-level에서 parameter를 scaling하고 shifting하는 방법을 설명한다. 마지막으로 3-4에서는 MAML 기반으로 parameter를 optimization하는 방법을 서술한다. 

### 3-1. Network Encoder  
대부분의 GNNs은 주변 노드들의 정보를 취합(aggregation)하여 본인의 정보를 업데이트하는 message passing mechanism을 따른다. 

$
s_v^l=AGGREGATE(\lbrace h_u^{l-1}:u\in\mathcal{N}_ v\rbrace)
$

$
h_v^l=COMBINE(h_v^{l-1}, s_v^l)
$

$\mathcal{N}$ is the set of neighbors of node $v$ (**including node $v$**)

여기서 주목할 점은 본인 노드와 이웃 노드들의 메시지를 모두 더해서(AGGREGATE) 평균을 내는 방식(COMBINE)으로 정보를 취합한다는 점이다. 즉, 본인 노드와 이웃 노드들은 같은 class임을 가정하에 메시지를 취합하기 때문에 이는 homophilc을 가정하고 취합한다고 볼 수 있다. Meta-GPS에서는 real-world attributed networks에서 발생할 수 있는 heterophilic을 다루기 위해서 자기 자신과 이웃 노드를 섞지 않고(더하지 않고) 따로 병합하는 방법을 사용한다. 즉, 본인의 임베딩이 이웃의 임베딩과 너무 비슷해지지 않도록 한 것이라고 보면된다. 더 자세히 설명하자면, $AGGREGATE$ 함수에서 이웃 노드 $\mathcal{N}$의 정의를 자기 자신을 제외한 $\mathcal{\tilde{N}}$로 바꾸고, $COMBINE$ 함수를 평균 또는 가중평균이 아니라 concatenation 함수로 재정의한다. 식으로 정리하면,  

<!-- $$
s_v^l=AGGREGATE(\textbraceleft h_u^{l-1}:u\in\mathcal{\tilde{N}}_ i(v)_ v\textbraceright)
$$ -->

$
\mathbf{F}=\sigma(\mathbf{XW}_ f), 　 \mathbf{H}_ 0 \equiv \mathbf{F}, 　 \mathbf{H}_ i = \tilde{\mathbf{A}_ i}\mathbf{F},
$

$
\mathbf{R}=\parallel_{i=0}^l \mathbf{H}_ i, 　 \eta=\sigma(\mathbf{RW}_ r), 　 \mathbf{Z} = \mathrm{Squ}(\mathrm{Res}(\eta)\mathbf{R})  
$

> $\mathbf{X} \in \mathbb{R}^{n \times d}$ : node features  
$\sigma(\cdot)$ : non-linear transformation  
$\tilde{\mathbf{A}}_ i=\bar{\mathbf{D}}_ i^ {-1/2} \bar{\mathbf{A}}_ i \bar{\mathbf{D}}_ i^ {-1/2}$ : $i$-hop neighbors' normalized symmetric adjacency matrix **without self-loops**  
$\parallel$ : concatenation operator  
$\mathbf{F}, \mathbf{H}_ i, \mathbf{Z} \in \mathbb{R}^{n \times d'}$  
$\mathbf{R} \in \mathbb{R}^{n \times(l+1)\times d'}$ : the concatenated embeddings  
$\eta \in \mathbb{R}^{n \times(l+1)\times 1}$ : the attention coefficient for different-hop neighbors  
$\mathrm{Squ},\mathrm{Res}$ : 'squeeze' and 'reshape' operations to match the matrix's dimensions  

위 식들을 정리해보면 $H_0$는 자기 자신을 포함한 ego-embeddings이고, $i$-hop의 정보를 취합할 때는 self-loop를 제외한 Adjacency matrix를 활용하여 자기 자신과 이웃들의 정보를 분리시킨다. 또한 합, 평균, 가중평균이 아닌 concatenation으로 정보를 합치는 방법으로 heterophilic graphs에 적합한 convolution layer를 설계하였다. 해당 layer는 $$\theta_e \ \lbrace \mathbf{W}_ f, \mathbf{W}_ r \rbrace$$의 파라미터를 가지고 있다.


### 3-2. Prototype-based Parameter Initialization  
이 세션에서는 class마다 prototype으 만들고 이를 기반으로 parameter를 intialization하는 방법을 소개한다. 이렇게하는 이유는 MAML은 general한 single initialzation을 사용하지만, attributed networks는 $i.i.d$ 가정으 따릊 않기 때문에 single intialization을 찾기 어렵다는 점에서 시작됐다. 따라서 prototype vector를 활용하여 class-specific initialized parameters를 찾아내어 task에서 다른 class를 샘플링하면서 발생하는 variance를 줄인다.

$
\mathbf{P}_ j= {1 \over {\| \mathcal{V}_ j \|}} \Sigma_ {k\in{\mathcal{V}_ j}} \mathbf{Z}_ k
$

$$\mathbf{Z}_ k$$는 node $$v_k$$의 feature이며, $$\| \mathcal{V}_ j \|$$는 class $$j$$에 속하는 node set이다. 각 class마다 node feature의 평균을 구해서 prototype을 만들고 이 prototyped을 MLP layer에 전달하여 class-specific initialized parameters를 만든다.  

$
\varphi_ j = \mathbf{MLP}(\mathbf{P}_ j;\theta_ p), j = 1, ..., N
$

> $\varphi_ j \in \mathbb{R}^{d'}$ for class $j$  
> $N$ is the number of categories of a task (i.e., $N$-way)  

class마다 initial parameter가 설정되면 task $\mathcal{T}_ i$의 support set $\mathcal{S}_ i$로 추가적인 adaptation을 진행한다. 

$
\varphi'_ i=\varphi-\alpha \nabla_\varphi\mathcal{L_{\mathcal{T}_ i}}(f(\mathcal{S}_ i ; \varphi, \Theta))
$

> $\Theta = \lbrace \theta_ e, \theta_ p \rbrace$ : prior parameters  
> $\mathcal{L_{\mathcal{T}_ i}}(\cdot)$ : cross-entropy loss function  
> $score$=softmax( $\mathbf{Z} \varphi^{T} + b), score \in \mathbb{R}^{N \times d'}$

### 3-3. $S^2$ Transformation for Different Tasks  
Task마다 구성된 class와 node가 다름으로 인해서, task간의 variance 차이로 발생한다. 따라서 inter-task간의 difference를 파악하여, parameters를 task-specific하게 바꿔주는 방법을 제시하는데, $S^2$ transformation을 이용한다. 먼저 task를 대표할 수 있는 representation vector $t_i$를 만든다(task의 prototype이라고 생각하면 된다). Task의 prototype은 task 내 포함되어 있는 모든 node embeddings을 mean해주는 방법으로 만든다. 이렇게 만든 prototype으로 scaling vector $\lambda_ i$와 shifting vector $\mu_ i$를 생성하는 데, task $\mathcal{T}_ i$의 성질을 인코딩하는 것이다.   

$
t_i = \frac{1}{\| \mathcal{V}_ {t_i} \|} \Sigma_ {k\in{\mathcal{V}_ {t_i}}}, \lambda_i=g(t_i;\psi_ \lambda), \mu_ i = g(t_ i; \psi_\mu)
$


>  $\mathcal{V}_ {t_i}$ : set of nodes involved in $\mathcal{T}_ i$  
>  $\lambda_ i, \mu_ i \in \mathbb{R}^{\| \Theta \|}$  
> $\psi_\lambda, \psi_\mu$ : paramters of two $MLPs$ with the neural network used in prototype-based parameter initialization.  

위에서 생성한 scaling/shifting vector로 다음과 같이 task's prior meta-parameters $\Theta$를 transformation해준다.  

$
\Theta_i = \lambda_i \odot \Theta + \mu_i
$

이를 통해 비슷한 task는 비슷하게 transformation하는 식으로 task-specific하게 parameter를 바꿔줄 수 있다.  

### 3-4. Meta-optimization    
마지막으로 model을 optimization하기 위해서 Meta-GPS는 meta-learning 방법을 활용한다. 그 중에서도 MAML의 방법을 따라가는데, 이는 Meta-training, Meta-testing phase로 나눌 수 있다.  

#### Meta-training  
특정 task $\mathcal{T}_ i$를 잘 맞추기 위해서 support set(labeled data)가 먼저 투입된다. support set의 label을 이용해서 cross-entropy로 task loss $\mathcal{L}_ {\mathcal{T}_ i}$을 계산한다.  

$
\mathcal{L}_ {\mathcal{T}_ i} (\mathcal{S}_ i, \psi'_ i, \Theta_ i)= -\sum_ {(v_ i, y_ i) \in \mathcal{S}_ i} (y_ i \log f(v_ i; \psi'_ i, \Theta_ i)+(1-y_ i)\log(1-f(v_ i;\psi'_ i, \Theta_ i))) 
$

이 loss를 task $\mathcal{T}_ i$에 adaptation하기 위해 $\Theta_ i$를 수 번의 gradient descent step을 통해 업데이트하게 된다.  

$
\Theta'_ i=\Theta-\alpha \nabla_\Theta\mathcal{L_{\mathcal{T}_ i}}(f(\mathcal{S}_ i ; \varphi'_ i, \Theta))
$ 

> $\alpha$ : meta-step size

support set으로 $\Theta'_ i$를 optimization을 하고 나서, 이제는 label을 모르는 query set을 이용해 loss를 구하고 이 loss를 minimize 시키는 것이 meta-objective function이다.  

$
\min_ {\Theta, \Psi} \mathcal{L}  ( f_ {\varphi'}, \Theta, g_{\psi})= \min_ {\Theta, \Psi} \sum_ {\mathcal{T}_ i ~ p(\mathcal{T})} \mathcal{L}_ {\mathcal{T}_ i}(f(\mathcal{Q}_ i; \varphi'_ i, \Theta'_ i))+\gamma \| \Psi \|^2_ 2
$

> $\Psi=\lbrace \psi_\lambda, \psi_\lambda \rbrace$  

이제 tasks 전반에 걸쳐 meta-optimization을 하게된다.  

$
\Theta=\Theta-\beta \nabla_ {\Theta} \mathcal{L}(f_ {\varphi'}, \Theta, g_{\psi}), \Psi = \Psi-\beta \nabla_\Psi\mathcal{L}  ( f_ {\varphi'}, \Theta, g_{\psi})
$

> $\beta$ : meta-learning rate  

#### Meta-testing  
meta-testing phase는 meta-training phase와 같은 과정을 거친다. 즉, meta-testing task $$\mathcal{T_ {te}}$$의 support set $$\mathcal{S}$$으로 prior parameter $$\Theta, \Psi$$를 수 번 optimization하고, query set $$\mathcal{Q}$$로 모델의 성능을 측정한다.  


## **4. Experiment**  
Meta-GPS에서는 6가지 데이터셋으로 실험을 진행하였다. Motivation에서 주장한 바와 같이, real-world의 heterophilic한 데이터셋에서도 모델이 잘 작동한다는 것을 증명하기 위해 데이터셋의 node homophily, $\mathbf{H}$를 정의하고 제시하였다.  

$
\mathbf{H}= {1\over{\vert\mathcal{V}\vert}}\sum_{v\in\mathcal{V}}\sum_{v\in\mathcal{V}} {\vert \lbrace  (u,v):u \in \mathcal{N}_v \wedge y_u = y_v \rbrace \vert \over{\textbar \mathcal{N}_ v \textbar}}
$

$$\mathbf{H}$$가 높을수록 homophlily가 높고, 낮을수록 heterophilic하다.


 
<img width="564" alt="스크린샷 2022-10-16 오후 4 53 48" src="https://user-images.githubusercontent.com/37684658/196024621-49ca06e5-7c33-4336-a3db-f3f9b62479fe.png">
 
<img width="1296" alt="image" src="https://user-images.githubusercontent.com/37684658/196022640-82257140-97d7-4603-ba72-da206326d920.png">  


6개의 5way-3shot, 5way-5shot, 10way-3shot, 10way-5shot 세팅에서 모두 SOTA의 성능을 보여주고 있다. 하지만 논문에서 주장하는대로, prototype-based parameer initialization, scalingand shifting vectors가 새로운 tasks를 맞추는데 더 효과적이고, tranferable knowledge를 축적할 수 있는 지 보여주지는 못하기 때문에, 이는 ablation study에서 보여준다. 위 실험에서 보여주는 것은, homophilic한 데이터셋은 물론 heterophilic attributed networks에서도 좋은 성능을 보여준다는 것이다. absolute improvement 정도를 보아도, homophilic한 가정으로 설계된 기존 baseline들과 heterophilic한 데이터셋에서 더 큰 성능 차이를 보여주고 있다.  
 
 
<img width="1355" alt="image" src="https://user-images.githubusercontent.com/37684658/196023310-e7da3ee6-401e-4a17-bd37-8cd169a93d67.png">  

또한 기존 메타러닝 기반의 baseline은 instance를 기준으로 모델을 학습하기 때문에 data noise에 취약하다. 위 실험으로 instance 기반의 학습에서 벗어나, class specific parameter, task prototypical parameter update를 통해 data noise에 robust한 모델임을 보여주고 있다.  
 
 
 <img width="761" alt="image" src="https://user-images.githubusercontent.com/37684658/196023600-4d93ab72-76e9-4670-aba6-351505c27e89.png">  

> (I) Meta-GPS-SGC : heterophilic convolution layer 대신 GCN으로 대체한 모델이다.  
> (II) Meta-GPS-PI : prototype-based initialization paramter를 제거하고 random initialization parameter를 사용한 모델이다.  
> (III) Meta-GPS-$S^2$ : $S^2$ transformation을 삭제하고 모든 task들을 동등하게 취급한 모델이다.  
 
 성능이 논문에서 제시하는 효과를 모두 증명하는 것은 아니지만, ablation study를 핵심 module 하나하나 잘 커버하면서 실행하였고, 그에 대한 결과도 바람직하게 보여주고 있다고 생각한다. 특히 Meta-GPS-SGC 같은 경우는 상대적으로 homophilic networks에서는 성능이 좋지만, heterophilic한 상황에서는 성능 하락이 더 크게 나타난다.  

## **5. Conclusion**  
Meta-GPS는 meta-learning 기반의 few-shot learning method이다. 기존 meta-learning 기반의 baseline들이 instance-based statistic을 기반으로 모델링 되었기 때문에, data outlier에 취약한 점을 포인트로 잘 잡아내었고, 이를 해해소하기 위해 prototype-based parameter initialization, $S^2$ transformation for suiting different tasks를 제시하였다. 또한 real-world의 attributed network의 heterophily한 데이터셋을 다루기위해 기존 convolution layer를 간단하면서도 효과적으로 수정하였다. 본 논문의 가장 큰 TAKEAWAY는 instance기반으로 다루는 방법에서 class-level, task-level에서 다루는 관점을 보여주었고, 그로 인한 효과로 few-shot learning의 핵심 문제인 outlier 문제를 효과적으로 다루었다는 점이라고 생각한다.  

 
## **Posting author information**

* **김성원 (Sungwon Kim)**
  * [Data Science & Artificial Intelligence Laboratory (DSAIL)](http://dsail.kaist.ac.kr) at KAIST
  * Graph Neural Network, Meta-Learning, Few-shot Learning
  * [github](https://github.com/sung-won-kim)
