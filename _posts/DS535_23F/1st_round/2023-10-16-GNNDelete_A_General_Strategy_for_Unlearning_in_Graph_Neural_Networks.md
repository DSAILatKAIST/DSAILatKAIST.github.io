---
title:  "[ICLR 23] GNNDelete : A General Strategy for Unlearning in Graph Neural Networks"
permalink: 2023-10-16-GNNDelete_A_General_Strategy_for_Unlearning_in_Graph_Neural_Networks.html
tags: [reviews]
use_math: true
usemathjax: true
---

# # GNNDelete : A General Strategy for Unlearning in Graph Neural Networks
## **1. Problem Definition**

현실의 문제들을 해결하기 위해 AI 모델들을 적용하다보면, 데이터를 삭제하는 경우가 있습니다. 예를 들어, 프라이버시 문제나 권한 때문에, 또는 시간이 지남에 따라 데이터가 정확하지 않아서 삭제하게 됩니다. 이럴 경우, 삭제된 데이터에 맞추어 모델 또한 변화해야하는데, 이를 `unlearning`이라고 합니다. 하지만 그래프의 경우에는 이 문제가 특히 까다롭습니다. 그래프는 여러 entity들이 서로 연결되어있고, 이를 활용하여 학습되기 때문에 특정 정보가 삭제되어도 삭제된 정보의 영향이 이웃한 node들에 반영되어있습니다. 따라서 모델은 이웃에 대한 영향까지 고려하여 '마치 해당 데이터가 존재한적 없던것처럼' 대응해야합니다. 이 논문에서는 `deleted edge consistency`와 `neighborhood influence`라는 두 가지 기준을 제시하며, 이에 맞추어 효과적인 Graph unlearning 방법을 제안합니다. 


 <br/> <br/>
 
## **2. Motivation**  
> **Notation**   
그래프는 $G = (\mathcal{V}, \mathcal{E}, X)$ 로 나타내며 각 구성요소는 node set, edge set, feature입니다. 삭제될 edge를 $\mathcal{E_ {d}}$로, 삭제될 node를 $\mathcal{V_ {d}}$로,  남아있는 그래프를 $G_ {r} = (\mathcal{V_ {r}}, \mathcal{E_ {r}}, X_ {r})$로 나타냅니다. 두 node의 k-hop 거리 이내에 있는 subgraph를 $\mathcal{S_ {uv}^ {k}} = (\mathcal{V_ {uv}^ {k}}, \mathcal{E_ {uv}^ {k}}, X_ {uv}^ {k})$로 나타냅니다. unlearning전 모델을 $m(G)$로, unlearning 후 모델을 $m'(G)$로 나타냅니다. 

<br/> <br/>

> **Motivation**   

기존의 graph unlearning 방법들은 대부분 그래프를 부분적으로 나누어 재학습하거나, 여러개의 submodel들을 학습하여 inference시 같이 활용하는 방법들을 사용합니다. 이 경우에는 그래프의 connectivity에 대한 정보가 손상되기 때문에 성능이 좋지 않습니다. 또한 여러개로 나누어 학습하거나 재학습하는 방식은 여전히 시간을 많이 쓰기 때문에 scalable하다고 할 수 없습니다. 최근 graph unlearing의 방법론은 linear GNN에만 가능하거나 approximation방법을 사용하기에 한계가 있다고 할 수 있습니다. 따라서 저자들은 2가지 기준을 제시하며 이에 따라 모델을 효율적으로 설계합니다. 

 <br/> <br/>

> **Graph Unlearning Property**   

Unlearning의 궁극적인 목표는 데이터 삭제전 모델 $m(G)$와 데이터 삭제후 모델 $m'(G)$의 결과를 최대한 비슷하게 유지하면서, 삭제된 데이터의 영향을 제거하여 삭제된 데이터가 존재하지 않았던것처럼 하는것이 목표입니다. 이를 통해 데이터의 프라이버시 등을 지킬 수 있습니다. 
저자들은 다음 2가지 기준을 제시했습니다. 
* `Deleted Edge Consistency` : 만약 어떤 데이터가 삭제되었다면 $m'(G)$는 해당 데이터의 유무와 상관없이 일관된 output이 나와야합니다.
* `Neighborhood Influence` : 데이터가 삭제되면, 이웃들이 영향을 받게 되는데, 이것이 최소화되어야합니다. 즉 $m(G)$와 $m'(G)$가 그래프의 다른 부분들에서도 거의 같아야합니다. 
이에 대한 조금 더 구체적인 내용은 다음 section에서 다룹니다.

 <br/> <br/>
 
> **Contribution**   
* Graph unlearning분야에서 반드시 충족해야할 기준 2가지를 제시합니다.
* 기존의 방법론들과 다르게 그래프의 connectivity를 무시하거나, 다시 학습해야하는 비효율 없이, 효율적으로 학습 가능한 모델을 제시합니다. 

 <br/> <br/>
## **3. Method**
> **Graph Unlearning Property in detail **

![image](https://i.ibb.co/GFRdPwq/main-fig.png)

전에 이야기한 2가지 기준을 수식으로 조금 더 명확하게 표현합니다.
* `Deleted Edge Consistency` : $\phi$를 readout function이라고 할때, $l$번째 layer에서는 삭제된 edge (node u,v)에 대한 representation과 랜덤으로 고른 두 node pair (p,q)에 대한 representation이 차이가 작아야합니다. 
	* $\underset{p,q \in \mathcal{V_ {r}}}{\mathbb{E}}[\phi(h_ {u}^ {l},h_ {v}^ {l}) - \phi(h_ {p}^ {l},h_ {q}^ {l})] = \delta$
* `Neighborhood Influence` : $\psi$를 subset representation function이라고 했을때, 삭제된 데이터의 k-hop이내의 subset이 unlearning 전과 후가 비슷해야합니다. unlearning전의 node representation을 $h_ {w}^ {l}$, 후를 $h_ {w}^ {'l}$로 나타내면 다음이 성립해야합니다.
	* $\psi(\{ h_ {w}^ {l} \vert w \in \mathcal{S_ {uv}} \}) - \psi(\{ h_ {w}^ {'l} \vert w \in \mathcal{S_ {uv/e_ {uv}}} \}) = \delta$

 <br/> <br/>
 
> **layer-wise deletion operator**

위의 그림을 보면 알 수 있듯이, 해당 모델은 재학습하거나 기존에 학습된 모델 파라미터를 변형시키기는 방법을 사용하지 않습니다. 대신 각 layer마다 추가적으로 학습가능한 deletion부분을 만들어 삭제된 데이터의 영향을 받는 node의 representation을 변형시키는 방법을 사용합니다. 
![image](https://i.ibb.co/8dKSVVx/del-operator.png)
위 식의 del operator를 사용하는데, 만약 삭제된 데이터의 영향을 받는 node라면 (k-hop이내) 데이터 변형을 시키고($\phi$) 아니면 node의 representation을 바꾸지 않습니다($\mathbb{1}$). 여기서 $\phi$는 MLP로 구현됩니다. 결과적으로 삭제된 데이터 주변에서만 del operator가 작동하면 되기에 추가적인 training이 최소화되며, 모델을 처음부터 다시 학습하는 일이 없어 효율적입니다. 

 <br/> <br/>
  
> **Loss function**

앞서 이야기한 두가지 graph unlearning 특성을 loss function으로 구현합니다.
* `Deleted Edge Consistency` : $\mathcal{L}_ {DEC}^ {l} = \mathcal{L}(\{ [h_ {u}^ {'l}; h_ {u}^ {'l}] \vert e_ {uv} \in \mathcal{E_ {d}} \},\{[h_ {u}^ {l}; h_ {u}^ {l}] \vert u,v \in \mathcal{V}_ {r}  \})$
* `Neighborhood Influence` : $\mathcal{L}_ {NI}^ {l} = \mathcal{L}({CONCAT\{  h_ {w}^ {'l} \vert w \in \mathcal{S_ {uv/e_ {uv}}} \}, CONCAT\{ h_ {w}^ {l} \vert w \in \mathcal{S_ {uv}} \}})$

최종 loss function은 이 둘의 convex combination으로 구성됩니다.
$\mathcal{L} = \lambda \mathcal{L}_ {DEC}^ {l} + (1-\lambda) \mathcal{L}_ {NI}^ {l}$

 <br/> <br/>


## 4. Experiment  
> **Task and Dataset**  

* Dataset : 5가지의 대표적인 homogeneous graph와 2가지 heterogeneous graph를 사용합니다 : Cora, PubMed, DBLP, CS, OGB-Collab / OGB-BioKG, WordNet18RR
* Task : edge deletion / Node deletion / Node feature deletion
* Setup : test-set과 멀리 떨어져있는 데이터 deletion(쉬움)과 test-set에 가까이 있는 데이터 deletion(어려움)으로 2가지 세팅이 가능합니다.

 <br/> <br/>

> **Result - Edge Deletion**  

![image](https://i.ibb.co/PMky8sT/table1.png)
table1 데이터 삭제 비율은 2.5%였으며 test-set과 가까이 있는 데이터들도 deletion이 가능하도록 한 어려운 세팅의 결과입니다. table 1은 DBLP와 wordnet18을 데이터셋으로 사용했습니다. link prediction task를 수행했으며 $\mathcal{E}_ {t}$는 testset에서의 AUROC 결과를,  $\mathcal{E}_ {d}$는 '삭제된 edge를 edge가 없는것으로 판단하고 남은 edge는 edge가 있는것으로 link prediction 했는지'에 대한 결과입니다. 다른 모델들과 비교했을때 압도적으로 좋은 점수를 보여줍니다.

아래는 다양한 데이터셋에 대하여 다른 edge deletion ratio에 대한 결과입니다. table 13,15,17,19는 test-set의 이웃과 관련없는 edge를 삭제한 경우이며, 14,16,18,20은 test-set의 이웃에 영향을 줄 수 있는 edge를 삭제한 어려운 경우입니다. 데이터셋이 무엇인지는 table 설명을 참고하시면 됩니다. 전반적으로 새로운 모델이 deletion의 영향을 잘 고려하여 설계되었기 때문에 좋은 성능을 보여주고 있습니다. 

![image](https://i.ibb.co/dGXg41G/table1314.png)

![image](https://i.ibb.co/tZZXjqx/table1516.png)
 
![image](https://i.ibb.co/M154XDH/table1718.png)

![image](https://i.ibb.co/SRSH3s2/table1920.png)

 <br/> <br/>

> **Result - Node Deletion**  

node 100개를 삭제하고 node classification(첫 3 column), linkprediction(마지막 column)을 DBLP 데이터셋에 적용한 결과입니다. 
![image](https://i.ibb.co/1XM7RWH/table8.png)

 <br/> <br/>

> **Result - Node feature Deletion**  

100개의 node의 feature를 바꾼후 node(DBLP)/link(DBLP)/graph classification(ogbg-molhiv)을 진행한 결과입니다.
![image](https://i.ibb.co/BtH7PcC/table9.png)


 <br/> <br/>

> **Result - Sequential Deletion**  

edge deletion 비율을 점진적으로 늘려나갔을때의 결과입니다. 기존의 모델들은 재학습등의 방식을 활용하기에 점진적으로 늘려나가는 상황에서 비효율적이지만, 이 논문에서 제안한 모델은 deletion operator만 삭제된 데이터 및 이웃에 적용하기에 효율적입니다. 
![image](https://i.ibb.co/T24x341/table10.png)


 <br/> <br/>

> **Result - Membership Inference(MI) attack** '

학습에 활용된 데이터는 학습에 활용되지 않은 데이터보다 일반적으로 잘 맞춥니다. 이를 활용하여 특정 데이터가 학습에 활용되었는지를 확인하는Membership inference attack이 있는데, 이는 데이터 프라이버시 문제가 있음을 의미합니다. 만약 데이터가 효과적으로 삭제되었다면, unlearning 모델은 삭제된 데이터에 대한 MI attack에도 잘 대응해야합니다. 측정 지표인 MI ratio는 edge가 존재한다고 판단할 확률에 대한 (edge deletion 전후) 비율로 클수록 삭제된 정보를 적게 담고 있으며, 작을수록 삭제된 정보를 많이 가지고 있음을 의미합니다. 아래 표에서 볼 수 있듯이 새 모델이 가장 큰 값을 가지며, 삭제된 정보를 가장 적게 가지고 있음을 알 수 있습니다. 
![image](https://i.ibb.co/ZzYCyhK/table2.png)


 <br/> <br/>


> **Result - Ablation study**  

두가지 기준에 대한 loss의 비율을 조절하였을때 결과를 보여줍니다. 둘다 필요함을 알 수 있습니다. 
![image](https://i.ibb.co/2kmV8GN/table3.png)

기존 방법론들과 execution time 을 비교하여 새로운 모델이 상당히 scalable한 모델임을 보여줍니다. 
![image](https://i.ibb.co/9nd2Yfz/fig2.png)


 <br/> <br/>


## **5. Conclusion**
> **Summary**
이 논문에서는 데이터가 삭제되어야하는 상황에서 데이터의 영향을 제거하며서 동시에 성능은 유지하는 unlearning task를 다룹니다. graph unlearning에서 지켜져야할 deleted edge consistency와 neighborhood influence라는 2가지 특성을 제시하고 이에 따라 효율적인 deletion operator를 만들어 좋은 모델을 설계하였습니다. 또한 이를 다양한 실험과 현실적인 세팅으로 검증하였습니다. 

<br/> <br/>

> **Discussion & Further Development**

* 간략한 평가 :  새로운 기준을 잘 정의하였고 이에 맞추어 간단한 방식이지만 효과적으로 만들었습니다.  evaluation을 다양한 task와 setting으로 했으며, 현실적인 문제들을 잘 고려하였습니다. MI attack과 같은 방식으로 장점을 잘 보여준 실험 설계도 인상적이었습니다. 
* 발전 방향 : real world application을 찾아서 좀더 현실적인 데이터에서 적용 가능한지 확인해보았으면 좋겠습니다. multi-relational graph로 확장하는것도 가능해 보입니다. 이 경우에는 특정 종류의 edge만 삭제될수도 있기에 조금 더 신경써서 설계할 필요가 있을것으로 예상됩니다. 또한 edge attributed graph의 경우 edge attribute이 영향에 관여하게 될텐데 이에대한 연구도 가능해보입니다.

<br/> <br/>

## **Author Information**
* Yoonho Lee
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: Graph Nerual Network, Topological Deep Learning, Category Theory.
  * Contact: sml0399benbm@kaist.ac.kr
## **Reference & Additional materials**
* Github Implementation
  * [Code for the paper](https://github.com/mims-harvard/GNNDelete)
* Reference
  * [[ICLR-23] GNNDelete: A General Strategy for Unlearning in Graph Neural Networks](https://openreview.net/pdf?id=X9yCkmT5Qrl)
