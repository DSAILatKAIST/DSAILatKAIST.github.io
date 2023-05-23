---
title:  "[NIPS 2022] SHINE: SubHypergraph Inductive Neural nEtwork"
permalink: SHINE_SubHypergraph_Inductive_Neural_nEtwork.html
tags: [reviews]
use_math: true
usemathjax: true
---

# # SHINE: SubHypergraph Inductive Neural nEtwork
## **1. Problem Definition**
<img  src="https://user-images.githubusercontent.com/43372696/232283689-d231476b-740e-4938-9446-de7b4a07aa9b.png"  width="200px"  height="">

생명체는 다양한 단백질들을 활용하여 생명유지에 필요한 여러가지 반응과 작용들을 합니다. 단백질은 여러가지 gene의 정보를 바탕으로 만들어지기에, gene에 문제가 생기면 암을 포함한 여러가지 질병이 생길 수 있습니다. 단백질과 여러가지 물질들의 `기능적 단위`의 집합인 `pathway`를 활용하여 단백질에 대한 효과적인 representation을 구하고, 환자의 유전 정보의 변이를 바탕으로 subgraph를 형성하여 `환자들의 질병을 예측`하는 모델 SHINE을 제안합니다. 

 <br/> <br/>
## **2. Motivation**  
> **Preliminary - Graph Neural Network(GNN)**   

<img  src="https://user-images.githubusercontent.com/43372696/232258261-5b1b0adb-6428-447f-84cb-30974eee4367.PNG"  width="300px"  height="">

그래프는 여러가지 Entity 사이의 relation이나 Interaction을 나타내는 방법중 하나로, 보통 $\mathcal{G}=(\mathbb{V},\mathbb{E})$로 나타냅니다. 여기서 $\mathbb{V}$는 node set으로 vertex들을 모은 집합이며, $\mathbb{E}$는 edge set으로 각 edge는 node들을 연결해주는 선들로 표시됩니다. 두 node 또는 Entity가 edge로 연결된 경우에는 이 둘 사이에 유사한 부분이 있거나, interaction을 한다고 봅니다. 보통 각 node와 연결된 이웃 node들의 정보를 aggregation하여 각 node의 hidden representation을 구하는 방식으로 GNN이 작동합니다. 단순히 (node 관점에서) 자기 자신과 이웃을 mean aggregation(smoothing)하는 방식의 대표적인 예시로는 GCN이 있으며, 이웃 node들에 attention을 기반으로 하는 importance 차이를 두어 aggregation하는 GAT도 있습니다. 

<br/> <br/>
> **Preliminary - Protein-Protein Interaction graph(PPI)**  
 
<img  src="https://user-images.githubusercontent.com/43372696/232283637-9fe695d6-e7cb-4272-8b58-486c04d34cbe.png"  width="500px"  height="">

단백질에 대한 representation을 잘 뽑기 위하여 흔하게 사용되는 방식중 하나로 PPI network를 사용하는 방법이 있습니다. `protein-protein interaction network(PPI)`는 단백질들을 node로 두고 서로 interaction을 하는 단백질들을 edge로 연결하여 만들어집니다. 만들어진 network에서 GCN, GAT 그리고 heterogeneous GNN등 상황에 따라 적합한 GNN 모델을 사용하게 됩니다. 결과적으로 단백질을 나타내는 각 node는 edge로 연결되어있는 이웃 node들의 정보를 취합하여 단백질에 대한 더 좋은 representation을 얻습니다. 즉, node의 최종 representation에는 해당 단백질의 고유한 정보와 해당 단백질과 관련된 단백질의 정보가 적절히 반영되어있어서, 다양한 downstream task에 효과적으로 사용될 수 있습니다. 

<br/> <br/>
> **Preliminary - Pathway**   

<img  src="https://user-images.githubusercontent.com/43372696/232283770-0b7e3a2f-7d63-40a4-8a7d-881a54bdf3f8.png"  width="800px"  height="">

생명체 내부에서 나타나는 다양한 상호작용과 반응은 여러가지 단백질들이 복합적으로 작용하여 이루어집니다. 예를들면, 세포 소기관중 하나인 미토콘드리아에는 에너지를 만들어내기 위한 process가 있으며, 그 과정에서 다양한 단백질들이 사용됩니다. 이렇게 어떤 과정속에서 특정 기능을 하기 위해 사용되는 물질들과 그들의 반응을 `기능 단위`로 묶어둔 것을 `pathway`라고 합니다. 예를 들면, 위의 그림에는 apoptosis(세포사멸)에 관여하는 pathway와 각 pathway별 단백질들이 나타나있습니다. 생명체는 손상이 생기거나 오래된 세포를 스스로 죽도록 하여 관리를 하는데 이를 apoptosis라고 합니다. 이는 세포 외부에서 오는 여러가지 신호에 의해 조절됩니다. 세포 외부에서 신호가 오면, 이를 유전자가 저장되어있는 세포핵과 기타 조직들에 signaling pathway를 경유하여 전달합니다. 위의 그림에 있는 구체적인 예시로는 이미지 왼쪽의 MAPK signaling pathway가 있습니다. 세포 외부에서 survival factor와 관련된 신호가 오면, Ras, Raf MEK1/2 그리고 EPK1/2라는 단백질들을 순차적으로 거쳐서 세포핵까지 신호를 전달하게 되며, 세포핵에서는 해당 신호에 따른 gene regulation을 하게 됩니다. 이전에 설명한 PPI의 경우에는 단백질 사이의 pairwise interaction을 기준으로 만들었기에, 해당 단백질들이 interaction하는 맥락이 나타나지 않지만, pathway를 사용하게 된다면 각 단백질이 다른 단백질들과 상호작용하는 맥락을 활용할 수 있습니다. 

<br/> <br/>
> **Preliminary - Hypergraph**   

<img  src="https://user-images.githubusercontent.com/43372696/232283807-f608b997-5b74-4ba9-ae6d-024a534c73ca.PNG"  width="400px"  height="">

그래프는 Entity 사이의 pairwise relation을 나타내는 구조였습니다. 즉, edge가 2개의 node를 연결하는 구조였습니다. 이와 다르게 `Hypegraph`는 hyperedge가 2개 이상의 node(hypernode)를 연결합니다. 그래서 hyperedge를 3개 이상의 element(node)를 가질수 있는 set으로 해석하기도 합니다. 즉, hypergraph는 graph의 일반화된 형태로, 3개 이상의 entity가 모여서 상호작용하는 경우도 표현 가능한 구조입니다. (Hypernode의 경우 일반적으로 node라고 부르기도 합니다. 이번 리뷰에서는 편의상 node로 부르겠습니다.)

이렇게 3개 이상의 entity가 모여 interaction하는 경우에는 보통 해당 entity들이 모두 모여야지만 의미가 있다고 보며 이를 `high order correlation`이 있다고 표현합니다. 또한  hyperedge의 subset으로 이루어진 hyperedge는 존재할 수 없다고 보거나, 만약 있더라도 전혀 다른 의미를 갖는다고 해석합니다. 따라서 hyperedge는 pairwise relation(그래프의 edge)로 분해될 수 없다고 이야기합니다. 

<img  src="https://user-images.githubusercontent.com/43372696/232283841-20055381-7d85-448c-8036-af0224930c59.PNG"  width="300px"  height="">

실제로, 위에서 이야기한 high order correlation이 존재하는 경우가 많이 있습니다. 예를 들어서, 두 단백질이 상호작용하기 위해 보조인자가 필요한 경우가 있습니다. 보조인자가 없으면 두 단백질이 상호작용이 불가능하지만, 보조인자가 있는 경우 단백질의 3차원 구조가 변형되어 두 단백질이 상호작용 할 수 있게 됩니다. 이런 경우에는 3가지 물질이 모두 있는 경우에만 반응이 일어나기 때문에 hyperedge로 나타내는 것이 좋습니다. 해당 hyperedge를 분해하여 pairwise interaction(edge)로 나타내는 경우에는 high order correlation 특성을 나타내기 힘들며, 2개만 존재하는 경우에는 어떠한 interaction도 발생하지 않기 때문에 edge가 존재할 수 없습니다. 

이 논문에서 활용하는 pathway의 경우 여러가지 물질과 단백질들이 순차적으로 상호작용하지만, 수많은 pathway의 작용 순서가 아직 알려지지 않은 상태입니다. 따라서 일반적으로 각 pathway를 단백질들의 집합으로 나타내며, 순서는 나타내지 않습니다. 따라서 이 논문에서는 pathway를 활용하기 위해 hypergraph를 사용합니다. 

> **Preliminary - Hypergraph Neural Network**   

![image](https://user-images.githubusercontent.com/43372696/232259158-11cec002-cbcd-4502-98fb-e426e2530dfc.PNG)

그래프는 entity 사이의 pairwise relation을 나타내기에 adjacency matrix를 통하여 node 사이의 연결성을 나타내었습니다. 하지만 hypergraph의 경우에는 여러가지 node들이 hyperedge에 동시에 참여하기에 다른 방식으로 연결성을 표현합니다. 이를 `Incidence matrix`라고 하며, node와 hyperedge의 연결성을 나타냅니다. node가 특정hyperedge에 속할 경우에는 1의 값을 갖게되며, 속하지 않는 경우에는 0의 값을 갖게 됩니다. 이를 나타내면 아래와 같습니다.  

![image](https://user-images.githubusercontent.com/37684658/233369386-0b8eab9a-8ae8-4aaa-b1d3-72f9cebaa780.png)  

만약 $HH^ {\top}$를 하게 되면 결국 node-node matrix가 나오게 되며, 일반적인 그래프의 adjacency matrix와 같이 node 사이의 연결성을 나타내는 matrix가 나오게 됩니다. HGNN 모델은 이를 활용하여 hypergraph에서의 학습을 합니다.
![image](https://user-images.githubusercontent.com/43372696/232259205-ac05f9e6-85f3-4841-a352-ce305873cbfb.PNG)

HGNN 논문에서 제시하는 hypergraph laplacian은 아래의 식과 같습니다.

$\Theta = D_ {v}^ {-{1}\over{2}}HW_ {e}D_ {e}^ {-1}H^ {\top}D_ {v}^ {-{1}\over{2}}$


$W_ {e}$는 hyperedge의 가중치를 의미하며, 일반적으로 데이터에서 주어집니다. 만약 모든 hyperedge가 같은 가중치를 가지고 있다면 모두 1의 값을 갖게 되며, 더 중요한 hyperedge가 있는 경우에는 더 큰 값을 주게 됩니다. 일반적으로는 데이터셋에서 hyperedge의 값을 미리 정해두지 않은 경우에는 모두 동등한 가중치(1)를 갖는 것으로 설정합니다.  위의 식에서 $D_ {v}$는 node degree로 각 node가 속해있는 hyperedge의 숫자를 의미하며,  $D_ {e}$는 hyperedge degree로 각 hyperedge가 갖는 node의 수를 나타냅니다. $D_ {v}^ {-{1}\over{2}}$와 $D_ {e}^ {-1}$를 곱하여 node와 edge의 degree에 의해 좌우되지 않도록 normalize를 해줍니다. 이를 aggregation 관점에서 본다면 이웃의 정보들을 평균내는 것과 같습니다. 위의 hypergraph laplacian을 활용하여 각 layer는 $activation(\Theta X W)$형태로 구성됩니다. 여기서 $X$는 node feature나 hidden representation이며 W는 학습 가능한 weight matrix입니다. 위의 그림에 나타나있듯이, $H^ {\top}D_ {v}^ {-{1}\over{2}}X$ 부분에서 각 hyperedge는 가지고 있는 node의 정보들을 aggregation하게 됩니다. 그 이후에 $W_ {e}D_ {e}^ {-1}$ 부분을 거쳐 $W_ {e}D_ {e}^ {-1}H^ {\top}D_ {v}^ {-{1}\over{2}}X$가 되면서 hyperedge의 weight와 degree normalization이 진행됩니다. 마지막으로  $D_ {v}^ {-{1}\over{2}}H$ 부분을 거쳐 $D_ {v}^ {-{1}\over{2}}HW_ {e}D_ {e}^ {-1}H^ {\top}D_ {v}^ {-{1}\over{2}}X$가 되면서 hyperedge의 정보들이 각 node들로 모이게 됩니다. 즉, 2단계(node--> hyperedge, hyperedge-->node)로 정보 전달이 이루어집니다. 그리고 학습가능한 weight $W$를 곱하며 해당 layer가 완성됩니다. 
<br/> <br/>

> **Motivation**   

기존의 drug discovery나 drug-drug synergy prediction 등 protein을 기반으로하는 다양한 모델들은 PPI network에서 GNN을 학습하는 방식으로 진행이 되었습니다. 하지만 이는 단백질들의 interaction의 유무만 사용하기에 해당 interaction이 어떠한 맥락에서 사용되는지에 대한 정보를 반영할 수 없었습니다. 이를 해결하기 위해 pathway라는 `기능`위주의 정보를 hypergraph를 통하여 나타내고, dual attention 기반 hypergraph neural network를 학습하여 단백질에 대한 효과적인 representation을 학습합니다. 또한 환자의 gene 변이 정보를 subgraph로 나타내어 질병과 암의 종류를 예측하는데 성공합니다.
 <br/> <br/>
> **Contribution**   
* Hypergraph에서의 subgraph representation learning을 학습하는 방식을 최초로 제안합니다.
* Node와 hyperedge의 representation을 같이 학습하여 pathway 사이의 correlation에 대한 분석이 가능합니다.
* Genetic Medicine 분야에 적용한 최초의 논문입니다. (pathway 기반 방식으로 접근하는 모델이 bioinformatics 분야에 있기는 했으나 알고리즘 위주의 방식이며 hypergraph neural network를 이 분야에 사용한 것은 이 논문이 처음입니다.)
 <br/> <br/>
## **3. Method**
> **Notation**

![image](https://user-images.githubusercontent.com/43372696/232235555-f9e10fcf-af42-436c-bde9-183c6f90c840.PNG)

해당 논문에서 사용하는 표기는 위의 표와 같습니다. 또한 Hyperedge incidence matrix에 대한 정의는 2번 section의 preliminary에 있었듯이 다음과 같이 정의합니다.

![image](https://user-images.githubusercontent.com/37684658/233369386-0b8eab9a-8ae8-4aaa-b1d3-72f9cebaa780.png)

Node set($\mathcal{N}$)은 gene($g_ {i}$)들로 구성되어있습니다. ($\mathcal{N} = \{g_ {1}, ..., g_ {\vert \mathcal{N} \vert}\}$)
Hyperedge set($\mathcal{E}$)은 pathway($p_ {j}$)들로 구성되어있습니다. ($\mathcal{E} = \{p_ {1}, ..., p_ {\vert \mathcal{E} \vert}\}$) 
<br/> <br/>
> **Hypergraph - Strongly Dual Attention Message Passing**

![image](https://user-images.githubusercontent.com/43372696/232225281-0aeb8d7c-edf2-4929-8905-cceeb5930ca6.PNG)

앞서 preliminary에서 설명 드렸지만 hypergraph의 경우에는 2가지 단계를 거쳐 정보를 전달합니다. 

**node to hyperedge aggregation**

새로운 hyperedge representation은 다음과 같이 구합니다.

$h_ {E}^ {k}(p_ {j}) = \sigma(\sum_ {g_ {i} \in p_ {j}} a_ {E}( p_ {j}, g_ {i})h_ {N}^ {k-1}(g_ {i}))$

이 식의 의미는 결국 node (gene)의 feature를  attention score에 기반하여 가중치를 두고 aggregation하여 hyperedge (pathway)의 representation을 구한다는 것입니다. 여기서 attention score ($a_ {E}$)를 구하는 방법은  gat의 방법과 비슷하지만, `node 사이의 attention` 대신 `node (gene)와 hyperedge (pathway) 사이의 attention`을 구한다는 점에서 조금 다릅니다. 구체적인 식은 다음과 같습니다. 

$a_ {E}( p_ {j}, g_ {i}) = exp(c^ {\top} s(p_ {j}, g_ {i}) / (\sum_ {g_ {i^ {'}} \in p_ {j}} c^ {\top} s(p_ {j}, g_ {i^ {'}})))$

여기서 $s(p_ {j}, g_ {i})$는 아래의 식으로 구하며 $*$는 element wise product입니다. 

$s(p_ {j}, g_ {i}) = LeakyReLU((W_ {N}h_ {N}^ {k-1}(g_ {i}) + b_ {N})*(W_ {E}h_ {E}^ {k-1}(p_ {j}) + b_ {E}))$



**hyperedge to node aggregation**

새로운 node representation 학습은 전반적으로 위의 경우와 거의 동일합니다만 hyperedge와 node의 위치만 바뀝니다.

$h_ {N}^ {k}(g_ {i}) = \sigma(\sum_ {p_ {j} \owns g_ {i}} a_ {N}( g_ {i}, p_ {j})h_ {E}^ {k-1}(p_ {j}))$

여기서 $a_ {N}$는 node가 속해있는 hyperedge들에 대한 attention score로 다음과 같이 구합니다.

$a_ {N}(  g_ {i}, p_ {j}) = exp(c^ {\top} s(p_ {j}, g_ {i}) / (\sum_ {p_ {j^ {'}} \owns g_ {i}} c^ {\top} s(p_ {j^ {'}}, g_ {i})))$

여기서 $s(p_ {j}, g_ {i})$는 위에서(node to hyperedge aggregation) 구한것과 완벽히 동일합니다. 다른 hypergraph attention 방법들과는 다르게 두 과정에서 완전히 같은 attention matrix를 사용하며 이를 strongly dual attention이라고 저자들은 이야기합니다. 

**Regularization**

GNN과 hypergraph Neural Network에는 공통점이 있습니다. 바로 edge/hyperedge로 연결된 node의 representation이 비슷해진다는 점입니다. 이는 graph/hypergraph 자체가 '비슷한 entity끼리 연결된다'는 전제를 가지기 때문입니다. 따라서 aggregation을 통해 점점 비슷해지도록 학습을 하는 것입니다. 따라서 이웃한(연결된) node의 representation이 다를 경우 같아지도록 하기위해 penalize를 할 필요가 있습니다. preliminary에서 설명드린 hypergraph laplacian을 활용하여 저자는 연결된 node끼리 비슷해지도록 regularization term을 추가로 만들었으며 바로 다음과 같습니다.

$\mathcal{L_ {reg}} = \sum_ {i,j} ((X_ {i}- X_ {j})(X_ {i}- X_ {j})^ {\top}  * \Theta_ {i,j} ) =  \sum_ {i,j} ((X_ {i}X_ {i}^ {\top} -2X_ {i}X_ {j}^ {\top} + X_ {j}X_ {j}^ {\top}) * \Theta_ {i,j} )$
<br/> <br/>
> **Subgraph - Weighted Subgraph Attention**

Hypergraph Neural Network를 대략 만들었으니, 환자의 변이 gene정보로부터 subgraph를 만들어야합니다. 데이터셋으로부터 통계를 내어 $M$ matrix를 만듭니다. 해당 matrix의 row는 환자를 나타내며, column은 gene을 의미합니다. Matrix에 담겨있는 값들은 해당 환자의 특정 gene의 mutation rate을 의미합니다. 

데이터셋으로부터 얻은 환자의 mutation 관련 gene들을 묶어서 subhypergraph(또는 subgraph)를 만듭니다. $j$번째 환자의 subgraph를 $\mathcal{G}_ {j}$로 표기하고 $i$번째 gene을 $\mathcal{g}_ {i}$로 표기하겠습니다. 

각 환자(subgraph)의 representation은 다음과 같이 구합니다.

$h(\mathcal{G}_ {j}) = \sigma(\sum_ {\mathcal{g}_ {i} \in \mathcal{G}_ {j}} a(\mathcal{G}_ {j}, \mathcal{g}_ {i}) h_ {N}^ {K}(\mathcal{g}_ {i}))$

위 식의 의미는, 환자가 갖는 변이 gene들과 환자 사이의 attention score를 기반으로 하여 변이 gene들의 정보를 aggregation 하는 방식으로 환자를 나타낸다는 것입니다. 
여기서 $a(\mathcal{G}_ {j}, \mathcal{g}_ {i})$는 attention score이며 다음과 같이 구합니다.

$a(\mathcal{G}_ {j}, \mathcal{g}_ {i}) = exp(M_ {ji}b^ {\top}h_ {N}^ {K}(\mathcal{g}_ {i}) / (\sum_ {\mathcal{g}_ {i^ {'}} \in \mathcal{G}_ {j}} exp(M_ {ji^ {'}}b^ {\top}h_ {N}^ {K}(\mathcal{g}_ {i^ {'}})))$

이는 환자의 representation을 attention 기반으로 구하지만 여기에 환자의 gene mutation rate정보가 반영되도록 만든것입니다. 이렇게 여러 환자의 subgraph representation을 concatentation으로 합쳐서 $S$를 만듭니다.

$S = [h(\mathcal{G}_ {1})^ {\top} \vert h(\mathcal{G}_ {2})^ {\top} \vert ... \vert  h(\mathcal{G}_ {n})^ {\top}]^ {\top}$

각 subgraph를 분류하기 위해 다음과같이 classifier를 만듭니다.

$Z = softmax(W^ {(1)}(ReLU \circ FC)^ {(2)} (S)+W^ {(0)})$

Loss function으로는 cross-entropy loss를 사용합니다.

$\mathcal{L} = - \sum_ {j \in \mathcal{Y}_ {D}} {\sum_ {f =1}^ {F} Y_ {jf} \ln Z_ {jf}} + \mathcal{L}_ {reg}$

<br/> <br/>





## 4. Experiment  
> **Task and Dataset**  

이 연구에서는 3개의 데이터셋을 사용합니다 :  MSigDB, DisGeNet, TCGA-MC3.

 먼저 Molecular Signature Database(`MSigDB`)를 사용합니다. MsigDB에는 다양한 종류의 pathway와 각 pathway에 참여하는 gene들에 대한 정보가 있습니다. 해당 데이터셋의 경우에는 전문가들에 의해 curated되어있습니다. MSigDB의 경우 단백질의 학습보다는 각 task에 사용되는 데이터셋(DisGeNet, TCGA-MC3)를 전처리하기 위하여 사용합니다. 이 데이터셋을 활용하여 기능이 알려지지 않은 gene과 겹치는 hyperedge를 제거하는 전처리를 합니다. 
 
그 다음으로는 `DisGeNet` dataset을 사용합니다. 해당 데이터는 환자의 유전자 변이와 그로 인한 질병에 대한 정보를 담고 있습니다. 해당 데이터셋을 바탕으로 hypergraph와 subgraph를 구성하게 되며 `Disease Type Prediction` task를 위하여 사용됩니다. 해당 task는 변이된 gene 정보를 바탕으로 환자의 질병을 분류하는 것이며 `Multi class classification` 형태입니다. 질병의 종류는 MeSH code로 정해져있으며, 데이터에 대한 split과 통계는 아래 표에 정리되어있습니다. (위에서 설명하였듯이 DisGeNet 전처리를 위해 MSigDB를 사용합니다.)


![image](https://user-images.githubusercontent.com/43372696/232258276-11cf92fb-da27-491d-8bb7-523c886f4c70.PNG)


마지막으로 `TCGA-MC3` dataset입니다. 해당 데이터는 변이된 gene에 대한 정보와 그에 해당하는 암에 대한 정보를 담고있는 방대한 규모의 데이터입니다. Gene 정보를 바탕으로 `Cancer Type Prediction`을 하는 task를 진행하며 이는 마찬가지로 `Multi class classification` 형태입니다. TCGA-MC3 dataset은 전문가들에 의하여 7가지의 mutation-calling algorithm을 적용하여 pass 표시를 하였으며, 이 논문에서도 pass 표시가 되어있는 데이터만 활용했습니다. 전처리후 데이터에 대한 통계는 아래의 표와 같습니다. (위에서 설명하였듯이 TCGA-MC3 전처리를 위해 MSigDB를 사용합니다.)
![image](https://user-images.githubusercontent.com/43372696/232258277-a495971e-5c29-4643-88ff-0b2ec7365031.PNG)


아래는 두 task의 데이터셋마다 존재하는 hyperedge의 수와 class, subgraph 수에 대한 통계입니다. hyperedge는 pathway들이며 subgraph는 gene 변이 정보(환자 정보)를 바탕으로 하여 subgraph를 만든 것입니다. 
![image](https://user-images.githubusercontent.com/43372696/232258273-adebb37f-7283-417f-a886-d1ae42381c91.PNG)

<br/> <br/>  
> **Baseline Models**  
* `HGNN` : Hypergraph Neural Network 초기 논문중 하나로 preliminary에 설명한것과 같이 hypergraph laplacian을 제시하며 이를 활용하여 학습합니다.
* `HyperGCN` : HyperGCN의 경우 hypergraph를 graph로 변형시킨 다음에 GNN모델을 학습합니다. 매 iteration마다  hyperedge마다 가장 학습이 급한(feature difference가 큰) 두 node를 선별하여 edge로 연결하고 기존의 hyperedge를 삭제합니다. 이렇게 만들어진 graph위에서 GCN을 학습합니다.
* `HyperGAT` : text classification을 위해 만들어진 모델입니다. hyperedge를 하나의 가상의 node처럼 취급하여 GAT를 학습하였다고 생각하면 가장 간단합니다. HGNN과는 다르게 $HH^ {\top}$ 대신 hyperedge로 node들의 정보를 attention 기반으로 aggregation하고, node에서 hyperedge정보를 aggregation할때 attention을 사용합니다. SHINE 논문에서는 dual attention으로 같은 attention matrix를 사용한것과 다르게 HyperGAT에서는 두 단계에서 다른 attention matrix를 학습합니다.
* `AllSetTransformer` and `AllDeepSets` :  Hypergraph Neural Network를 가장 generalize하여 나타낸 모델로 DeepSet 논문으로부터 이야기를 시작합니다. AllDeepSet의 경우 hyperedge/ node의 representation을 얻기 위해 이웃한 node/hyperedge의 representation을 MLP를 거쳐서 aggregation하며, AllSetTransformer는 여기서 attention을 추가하기 위해 Transformer를 기반으로 하여 mean aggregation을 합니다.
* `SubGNN` : hyperedge를 node처럼 생각하여 hypergraph를 2가지 type이 존재하는 graph로 봅니다.
* `PRS` : polygenic risk score의 약자로 genetic medicine 분야에서 사용하는 방법입니다.(전문 지식)
* `NMF` : Non-negative matrix factorization의 약자로 Matrix를 latent vector들로 분해하는 방식입니다. 
* `XGBoost` : end-to-end tree boosting system

<br/> <br/>  
>  **Disease Type Prediction & Cancer Type Prediction**

![image](https://user-images.githubusercontent.com/43372696/232258270-faa1b12e-668a-43a5-92ea-3fd2c851ed6c.PNG)

저자는 제안한 모델인 SHINE과 다른 baseline들을 비교하며 SHINE이 다른 모델들에 비하여 확실하게 우월한 성능을 갖고 있음을 보여줍니다. 특히 attention을 사용한 hyperGAT와 AllSetTransformer와의 비교를 통해 node to hyperedge 정보전달 과정(aggregation)과 hyperedge to node 정보전달 과정에서 같은 attention matrix를 사용하는 dual attention 기법이 효과적임을 보여주었습니다. 
<br/> <br/>  
>  **Enrichment Analysis**

![image](https://user-images.githubusercontent.com/43372696/232273370-e309f525-7444-461c-82e9-3de1f286c22f.PNG)

저자는 SHINE이 node와 hyperedge의 representation을 같이 학습하기에 결과를 해석하거나 correlation을 분석하는데 있어서 장점이 있다고 주장했습니다. 이를 입증하기 위하여 암의 종류별로, attention 값을 기반으로 관련이 높은 pathway들을 추출했습니다. 위의 표에서 각 column이 암마다 관련있는 pathway들입니다. 첫 column은 breast cancer관련 pathway들이며 두번째 column은 lung cancer 관련 pathway들을 추출한 결과입니다. 글자의 색은 MSigDB를 구축하는 과정에서 사용한 데이터 출처이므로 실험 해석과 관련이 없습니다. 저자는 생명과학쪽 지식을 바탕으로 shine으로 추출한 암 종류별 pathway들이 실제로 해당 암과 관련이 높은것을 설명합니다. 예를 들어 breast cancer의 경우  HER2/4-1BB bispecific molecule(표의 제일 왼쪽 column, 위에서 3번째 row)의 경우 HER2-positive breast cancer의  치료 방법중 하나로 밝혀졌다고 합니다. 이외에도 appendix에서 각 pathway가 어떻게 해당 암과 관련이 있는지 생명과학/의학 지식을 활용하여 설명합니다. 이를 통해 SHINE 모델이 해석을 쉽게 할 수 있도록 도와준다는걸 보여주었습니다. 
<br/> <br/>  

## **5. Conclusion**
> **Summary**
해다 논문은 기존의 PPI 기반 학습보다 효과적으로 기능적인 정보를 반영하기 위해 pathway라는 정보로 구축된 hypergraph를 사용하였습니다. strong dual attention을 기반으로 효과적으로 학습하였으며, subhypergraph를 classification하는 첫 논문으로, 새로운 application으로 질병과 암을 예측하는 논문입니다. 

<br/> <br/>

> **Discussion & Further Development**
* 간략한 평가 :  새로운 application에 잘 적용하였고 효과적이지만 논문의 설명 방식이 명확하지 않으며 설명이 깔끔하지는 않았습니다. 
* 발전 방향 : pathway들이 여러 단백질의 기능적인 집합이듯이, 여러 pathway들이 모여 어떤 더 큰 기능과 작용을 구성하는것은 분명합니다. 이러한 정보를 활용하여 hypergraph in hypergraph 모델을 만들거나, pathway사이의 관계에 대한 데이터를 활용하면 더 좋은 representation을 뽑는것이 가능해보입니다. 

<br/> <br/>

## **Author Information**
* Yoonho Lee
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Nerual Network, Topological Deep Learning, Category Theory.
  * Contact: sml0399benbm@kaist.ac.kr
## **Reference & Additional materials**
* Github Implementation
  * [Code for the paper](https://github.com/luoyuanlab/shine)
* Reference
  * [[ICLR-17] Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
  * [[ICLR-18] Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)
  * [[AAAI-19] Hypergraph Neural Networks](https://arxiv.org/pdf/1809.09401.pdf)
  * [[EMNLP-20] Be More with Less: Hypergraph Attention Networks for Inductive Text Classification](https://arxiv.org/pdf/2011.00387.pdf)
  * [[ICLR-22] You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://arxiv.org/pdf/2106.13264.pdf)
  * [[NeurIPS-22] SHINE: SubHypergraph Inductive Neural nEtwork](https://arxiv.org/pdf/2210.07309.pdf)