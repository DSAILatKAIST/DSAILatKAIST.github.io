---
title:  "[ICML 2023] Graph Ladling: Shockingly Simple Parallel GNN Training without Intermediate Communication"
permalink: 2023-11-20-Graph_Ladling_Shocikingly_Simple_Parallel_GNN_Training_without_Intermediate_Communication.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [ICML 2023] Graph Ladling: Shockingly Simple Parallel GNN Training without Intermediate Communication

## 0. Background
 본격적으로 Graph Ladling을 소개하기에 앞서, Graph Ladling에 research idea (inspiration)을 제공한 논문인,  2022 ICML에 발표된 `Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time` 을 먼저 간략히 말씀드리겠습니다. 최근들어 거대 언어 모형(LLM)과 ViT-G와 같은 큰 vision 모델 그리고 엄청 큰 데이터셋에서 Multi-modal하게 학습된 CLIP, ALIGN 등의 giant 모델들이 여러 domain에서 활약하고 있습니다. 이러한 giant 모델들을 원하는 downstream task에 적용하기 위해서는 `fine-tuning` 단계는 필수적입니다. 보통의 `fine-tuning` 단계는 2-step으로 이뤄집니다. 
 >**첫번째는 여러가지 hyper parameter configuration 조합으로 모델을 fine tuning 하는 것입니다.**

 >**두번째로 앞서 fine-tuned한 모델 중에서 validation dataset에 대해서 가장 좋은 성능을 보이는 모델 하나를 선택하고 나머지를 버리는 과정입니다.**
이런 보통의 2-step fine-tuning 과정은 2가지의 단점이 있습니다.  (1) 하나의 모델(the highest accuracy on the valid set)만을 택하지말고, 나머지 모델들을 모두 Inference 한 후 output들을 ensembling 한다면 하나의 모델의 성능을 능가할 수 있을 것입니다. 물론 이런 ensembling은 각 모델들의 inference 를 요구하기 때문에, 그에 따른 추가적인 computational cost가 드는 단점이 있습니다.
(2) fine-tuning을 통해서 얻은 하나의 모델은 OOD performance를 감소시킬 수 있습니다. 왜냐하면 target distribution에 대해서 fine-tuning을 진행하기 때문에, OOD distribution에는 떨어지는 성능을 보일 수 있습니다.
결국 큰 모델을 fine-tuning할 때, 어떻게 하면 more accurate하면서 more robust 모델을 만들 수 있을까? 에 대한 solution을 제시한 논문입니다. 
Large pre-trained 모델을 fine-tuning 할 때, 기존의 2-step의 fine-tuning 방식이 아니라 독립적으로 만들어진 fine-tuned된 모델들의 weights를 averaging 하는 model soup 방식을 제시합니다. fine-tuned 된 모데들의 parameter weights들은 `soup`을 위한 ingredient라고 생각한다면, averaging 과정은 이러한 ingredient를 적절하게 섞는 방식, `soup`을 끓이는 과정이 될 것입니다.
이러한 방식은 추가적인 training이 필요없고, inference cost도 들지 않으면서 ensembling과 같이 성능 향상을 보여주므로 large-pretrained model을 fine-tuning하기에 적합한 대안으로 보입니다.
위의 `Model Soup`을 graph domain 알맞게 적용한 논문이 지금부터 소개할 `Graph Ladling: Shockingly Simple Parallel GNN Training without Intermediate Communication` 입니다.


## 1. Motivation
Social network, knowledge graph, gene expression network까지 다양한 형태로 real-world의 다양한 domain에서 graph-structure를 찾아볼 수 있습니다.
이러한 graph data에 topology structure를 message passing을 활용해 잡아내고자 하는 모델이 GNNs이 대표적인 모델입니다. GNNs 방식이 graph data에 효과적인 방식으로 자리잡았지만, 여전히 message passing의 본질적인 특성으로인해 industrial-grade graph application(아주 큰 size의 graph)에 적용이 어렵다는 한계점이 있습니다. 그리고 GNNs 핛습시킬 때 발생되는 여러 문제들 (unhealthy gradient, over-smoothing, over squashing)을 해결하기 위해 deepening(layer를 더 쌓는것) 또는 widening(neighborhood coverage를 늘리는 것)방식을 활용하여 model capacity를 늘려 performance를 높이고자 합니다. 그러나 여전히 이런 방식(deepening or widening)은 좋은 성능을 보이지 못할수 있습니다. 추가적으로 Vision, NLP와 달리, 큰 데이터셋을 사용하여 model capacity를 올리는 것은 적절하지 않습니다. 그래서 보통은 모델 architectual change, regulaization 그리고 normalization technique에 의존합니다. 이러한 GNNs training의 문제점에 출발하여 다음과 같은 질문을 떠올릴 수 있을 것입니다. 
- "How to scale GNNs capacity without deepening or widening on small and large graphs"
`Graph Ladling`이 하고자 하는 것은 "Model soup의 개념을 이용하여 scalability를 가져가면서, deepening 또는 widening 없이 GNNs의 성능을 올려보겠다!" 의 한문장으로 표현될 수 있습니다.

이 논문의 contribution을 요약하자면 아래와 같습니다.

- Model soup을 적절하게 graph-structured data에 활용하여 다양한 scale graph data에서 GNNs의 performance benefit을 실험적으로 증명하였습니다. Candiatate model들의 orthogonal knowledge를 greedy interpolation 과정을 통해 포함 시켰습니다.
- Data centric 관점의 GNN tailored model soup 방식을 제시하였고 전체 그래프에 대한 비싼 computational resouce 없이 sampling과 partitioning을 이용해 단일 machine만으로 적용가능한 방식 또한 제시하였습니다.
- Small scale 부터 Large scale graph dataset에서 (Cora, Citeseer, Pubmed, Flickr, Reddit, OGBN-Arxiv, OGBN-products), 여러가지 GNNs 모델을 활용하여 `Graph Ladling`의 효과성을 보여주었습니다.

## 2. Methodology

### 2.1. Preliminaries
GNNs(Graph Nueral Networks) 반복적인 message passing을 거치고 이 과정을 통해서 node와 해당하는 neighborhood nodes간의 structural information을 활용할 수 있습니다.
GNN 보통 수식

### 2.2. GNNs and Issues with Model Explosion
GNNs의 성공에도 불구하고, Real-world의 많은 graph들이 큰 capacity를 가진 GNN을 사용하기를 원하지만, 현재의 많은 GNNs 모델들은 몇개 정도의 shallow layer와 제한된 neighboring nodes로 제한되어 있습니다. 또한 GNNs 모델 구성에서 더 깊은 layer를 쌓는다면 혹은 넓은 이웃 노드를 coverage하게 한다면 long-range 정보를 포함한 더 많은 이웃 노드들의 정보를 capturing할 수 있겠지만, `over-smoothing`, `over-squashing` 그리고 모델 학습에 방해가 되는 `unhealthy gradient back propagation` 등 의 trainiability challenges를 해결해야합니다.

![gl_1](https://i.ibb.co/v1Fkqqy/gl-1.png)  

위 그림은 SOTA GNNs을 `deepening` 과 `widening` 관점에서의 `scaling` 효과를 확인해본 그림입니다. (왼쪽) 그림은 GCN, GCN2 그리고 GPRGNN을 ogbn-arxiv에 대해서 layer 수를 늘려가며 실험한 결과입니다. 3개의 GNNs 모델 모두 Model capacity를 늘렸을 때 성능이 drop되는 것을 확인할 수 있스빈다. GCN2와 GPRGNN 모두 architecture modification, regularization, 그리고 normalization 을 사용하는 모델들이지만 이러한 결과를 보여줍니다. (오른쪽) 그림에서는 message passing의 breadth를 늘렸을때 GraphSAGE의 성능 변화를 나타내고 있습니다. 더 많은 neighborhood를 sampling 한다고 해서 성능에 도움을 주지 않는다는 결과를 알 수 있습니다.
이러한 결과들로부터 GNN 모델 capacity를 크게 키우는 것이 high-quality generalizable GNNs을 만든다는 것을 의미하지 않음을 알 수 있습니다. 본 논문에서는 이러한 model capacity를 직접적으로 키우는 방향과 대비하여 새로운 방식인 model soup개념(distirbuted and parallel training)을 GNN에 활용하고자 합니다. 즉, 어떠한 intermediate communication 없이 다수의 weaker 모델을 통해 강한 GNN 모델을 만들고자 합니다.

### 2.3. Model Soups and current state-of-the-art GNNs
이번 section에서는 0.background에서 소개했던 `model soup`의 개념이 `graph ladling`에 어떻게 녹아들었는지 살펴보겠습니다. `Model soup`은 large language pre-trained model(e.g, CLIP, ALGINN)의 parallel training에 대한 intuition을 제공하였지만, 그보다 훨씬 작은 GNNs을 scratch부터 training 시킬 때 적용시킬 수 있을지는 분명하지 않습니다. `Model soup`에서는 fine-tuned된 모델의 pre-trained weight를 intitialization point로 사용하고, 같은 initialization point로부터 만들어진 여러 모델(다른 configuration으로 fine-tuning 시킨 모델)들을 적절히 averaging시켜서 좋은 성능을 보여줍니다. 그렇지만 GNNs에서는 pre-trained된 모델을 사용하지 않기 때문에, `same random initialization`으로부터 학습된 여러 GNNs 모델을 사용합니다. `same random initialization`로 부터 학습된 independent 모델들은 same basin of error landscape을 가집니다. `Model soup`과 마찬가지로, same basin of error landscape가지기 때문에 weight의 linear interpolation 하였을 때 좋은 성능을 기대해볼 수 있습니다. 이 부분에 대한 자세한 설명은 `Model Soup`논문을 참고하신다면, 좀 더 자세한 설명을 보실 수 있습니다. Reference에 `model soup` 논문을 참고해주세요.
예를 들어서, K개의 GNNs가 있다고 했을 때, 이들은 같은 모델구조, same random intialization를 가지지만, 세부적인 configuration(learning rate, weight decay 등) 차이를 갖습니다. K개가 바로 `soup ingredients`가 됩니다. 이런 K개의 모델을 각각 single gpu에서 학습을 시켜 준비하고, `Algorithm 1`을 적용합니다.
`Algorithm 1`은 아래와 같습니다.
![gl_2](https://i.ibb.co/DYRTLRt/gl-2.png)  
위 `Algorithm 1`은 `graph ladling`의 가장 기본이 되는 방식입니다. 준비해놓은 K개의 모델들은 validation dataset에 대한 성능 순으로 정렬을 시킨 후, 성능이 제일 좋은 모델부터 soup에 포함시키고, $\alpha$ 를 0.01 으로 하여 soup에 있던 모델의 weight parameter와 새롭게 들어올 후보 모델의 weight paramter간의 linear interpolation을 진행합니다. 이때 후보 모델과 interpolation 시킨 모델의 성능이 기존보다 나이질때만 soup에 넣습니다. 즉, greedy하게 interpolation시켰을 때, validation set에 대해서 좋은 성능을 보이는 모델의 weight만 취하겠다는 방식입니다. 
이러한 간단한 방식으로, 여러 데이터셋에 대해서 좋은 성능을 보여주었습니다. 실험 결과에 대한 자세한 설명은 `3. Section`에서 말씀드리겠습니다.

### 2.4. Data-Centric Model soups and Large Graph Training Paradigms
Single gpu만 사용가능한 환경에서 large entire graph에서 message passing을 하기는 쉽지 않습니다.
(이 논문에서 single gpu만 사용가능한 환경은 memory가 큰 gpu는 사용할 수 없는, memory가 작은 gpu를 사용할 때를 말합니다. 실험에서는 여러대의 gpu를 parallel하게 사용하는데, 하나의 gpu만 있어도 같은 방식을 적용할 수있습니다. 물론 시간은 multiple-parallel 방식을 사용할 때 보다 부족한 gpu개수 배 만큼 걸리겠습니다.) 
왜냐하면 key bottleneck으로 $AX$ 의 계산이 매우 높은 computation cost를 요구하기 때문입니다. 그래서 `graph ladling`논문은 SOTA graph sampling 과 graph partitioning방식을 사용하여 single gpu resouce 상황에서 large graph를 다루고자합니다. 메모리 제약사항을 해결하기 위해서 sampling 방식과 partitioning방식은 full batch training이 아니라 sampling based batch training을 시행합니다. 이에 따른 GNNs 모델은 아래와 같이 표현될 수 있습니다.

>**$X_ {B_ {0}}^{(K)}[M_ {i}]=\tilde{A}_ {B_ {1}}^{(K-1)}\sigma (\tilde{A}_ {B_{2}}^{(K-2)}\sigma(\cdots \sigma(\tilde{A}_ {B_ {K}}^{(0)}X_ {B_{ K}}^{(0)}[M_ {i}]W^{(0)}[M_ {i}]))\cdots W^{K-2}[M_ {i}])W^{K-1}[M_ {i}]$**

$\tilde{A}^{l}$은 전체 batch $B_{l}$ 에서 sampled 된 graph의 $l$ 번째 layer에 대한 adjacency matrix이며, $M_ {i}$와 $W^{(l)}[M_ {i}]$는 $i$ th ingredient model과 해당하는 weight를 나타냅니다. $\sigma$ 는 Relu 함수와 같은 activation function을 의미합니다.

이렇게 mini-batch 방식과 sampling 및 partitioning방식이 결합된다면, time consumption과 memory usage를 줄일수 있을 것입니다. 

이제부터는 Graph sampling과 Graph partitioning을 적용한 `graph ladling` Algorithm을 살펴보겠습니다. 먼저, graph sampling방식에는 크게 3가지로 나뉩니다. `Node-wise sampling`, `Edge-wise sampling` 그리고 `Layer-wise sampling`입니다. 
Node sampling에서는 `GraphSAGE` 처럼 sampling distribution을 uniform distirubtion으로 두고 $Q$개의 노드를 sampling하는 방식입니다.
Edge sampling은 mini-batch training시에 node sampling처럼 uniform distirubiton을 sampling distribution으로 설정한 후, Q개의 정해진 edge만큼을 sampling합니다.
마지막으로 Layer sampling은 `FaseGCN`의 방식을 따라서 node sampling을 하는데, node의 importance에 따른 sampling distirbution을 가진다는 차이점이 있습니다.

아래는 `Model soup with Graph Sampling`에 대한 pseudocode입니다. Pseudocode속 greedy_weight_interpolation은 앞서 보신 `Algorithm 1`입니다. 위의 3가지 sampling 방식에 대해서 sampling ratio를 설정하여 gpu 수에 맞게 parallel training을 할 수 있습니다. 
![gl_3](https://i.ibb.co/jR9CB5B/gl-3.png)  

Graph sampling 방식은 여전히 neighborhood explosion issue로부터 자유롭지 않습니다. (node sampling 방식의 경우, 여전히 neighborhood explosion을 겪게 될 수 있습니다.)
이러한 neighborhood explosion은 GNNs 여전히 memory bottleneck을 가질 수 있습니다. 그래서 sampling방식에서 나아가서 graph partitioning 방식을 사용하는 model soup을 제안합니다.
전체 Algorithm은 아래와 같습니다. 

![gl_4](https://i.ibb.co/ggGrvst/gl-4.png)  

Graph Clustering방식 혹은 Metis Algorithm만을 통해서 전체 그래프 $G$ 를 $K$개의 cluster를 만들어 줍니다. Graph sampling 방식을 사용하여 model soup을 적용했던 `Algorithm 2`와 같은 방식으로 진행하지만, 추가적으로 $E$ 번의 iteration동안 $K$개의 partioned된(clustered) graph 중에서 $q$개를 뽑아줍니다. 이렇게 뽑힌 graph로 subgraph를 다시 만들어서 $M_{i}$를 학습신 후 model soup을 적용해줍니다. 

여기까지 `Greedy Interpolation Soup Procedure`, `Model Soup with Graph Sampling`, `Model Soup with Graph Partitioning` 까지 모든 `Algorithms`를 살펴봤습니다. 

## 3. Experiments and Analysis
논문이 실험한 결과와 그에 따른 분석을 살펴보겠습니다.

### 3.1. Dataset and Experimental Setup
본 논문에서는 2개의 GPU 서버를 활용해서 multiple gpu환경을 조성하였다고 합니다. Small graph에 대해서는 50개의 candidiate model을 구성하였고, Large graph에 대해서는 30개의 candidate model set을 구성하여 실험을 진행하였습니다. Candidate model을 구성할 때, batch size, learning rate, weight decay, dropout rate등을 다르게 하였습니다. Interpolation 계수인 $\alpha\$ 의 변화량(step)은 0.01로 놓고 실험을 진행하였습니다.

### 3.2. Model Soups and SOTA GNNs
![gl_6](https://i.ibb.co/tLz9zvN/gl-6.png)  

위의 표는 50개의 indepedent하게 학습된 candidate ingredients에 대한 model soup 결과입니다. SOTA GNNs와 모두 같은 모델 구성을 가졌습니다. `GCN`에 대해서 Cora의 경우 1.13%, `GCN2`의 경우 Pubmed에서 1.38%, `JKNet`,`DAGNN`,`SGC` 모델들 모두 OGBN-arxiv에서 성능이 올라간 결과를 볼 수 있습니다. 이러한 independent하게 학습된 모델을 적절히 model soup한다면 GNNs에서 또한 단일 모델을 능가하는 성능을 보여줄 수 있음을 실험적으로 확인하였습니다.

![gl_7](https://i.ibb.co/y0vF7VX/gl-7.png)  

위의 표는 여러가지 fancy architectual and regularization modification 방식을 적용했을 때와의 비교를 나타낸 성능표입니다. Vanilla-GCN과 50개의 모델들(ingredients)을 model soup한 결과가 여러 technique을 적용했을 때 보다 더 좋은 결과를 도출하는 것을 알 수 있습니다. Skip connection, Normalization 그리고 Random Dropping 3개의 category에서 여러 setting에 다양하게 실험을 진행하였고 model soup의 효과성을 또한 입증하였습니다.

### 3.3. Data-Centric Model Soup with Graph Sampling and Graph Partitioning
![gl_8](https://i.ibb.co/f8WTFpr/gl-8.png)  

위의 결과는 Model soup with graph sampling의 결과입니다. Flicklr, Reddit, OGBN-arxiv, OGBN-products 에서 모두 GraphSAGE, FastGCN, LADIES 그리고 GraphSAGE Ensemble방식들 보다도 model soup의 결과가 좋음을 확인할 수 있습니다. Graph Sampling 방식 중에서는 `Node sampling` 방식이 제일 좋은 성능을 보임을 알 수 있습니다. 다음으로는 Model soup with graph partitioning 일 때, Cluster GCN, GraphSAINT 그리고 ClusterGCN Ensembling 방식과의 비교를 한 결과입니다. 
![gl_9](https://i.ibb.co/YZYFHdr/gl-9.png)  

모든 데이터셋에 대해서 model soup 결과가 나음을 확인할 수 있습니다. 또한 ensembling방식보다도 더 나은 성능을 보이기 때문에, 추가적인 inference cost없이 좋은 성능을 보여줄 수 있는 방식이라는 것을 입증하였습니다.

## 4. Background Work
이번 section에서는 논문이 제시한 related work에 대해서 다루겠습니다. `Model Soup`논문으로부터 inspiration을 받았기 때문에, `Model Soup`논문은 중요한 background 논문입니다.(model soup에 대한 설명은 0.background와 논문 본문을 참고하시면 좋을 것 같습니다.) 또 다른 관련 work로는 (Li et al., 2022)가 제시한 branch-train-merge 방식입니다. 이 방식은 model combination과 distributed training을 모두 고려하였습니다. training data가 다른 domain으로 나눠질때, 각 domain 별로 expert model을 만들어서 후에 merging하는 방식을 채택합니다. merging 시에 weight averaging을 하거나 ensembling을 하는데, 이런 merging을 통해서 하나의 큰 단일 모델의 성능을 넘습니다. 이러한 여러 work들이 NLP, Vision 분야에서 다뤄졌지만, GNNs에 관해서는 underexplored 되었습니다. 이렇게 다른 data domain에 비해서 underexplored된 graph data에 대해서 논문은 graph data에 적합한 distributed training과 parallel training 방식을 접목시킨 `Graph Ladling` 방법을 제시하였습니다.

### 4.1. Comparison to Related Work and Concurrent Ideas in Distributed GNN Training
GNN ensembling 관점에서 `Graph Ladling`은 개별적으로 학습된 GNN들의 weight를 interpolationg 하여 accuracy를 올립니다. 이와 관련하여 몇가지의 관련 work가 존재합니다. (Ramezani et al., 2021)은 communication-efficient distributed GNN training technique (LLCG)를 제시합니다. 첫번째로 GNN을 local machine에서 학습을 시키고, server에서 Global Server Correction module을 통해서 locally trained 된 GNN을 periodic하게 averaging시켜줍니다. 하지만 server의 Global module에 많이 의존한다는 문제가 있습니다. 또 다른 work(Zhu et al., 2023)는 개별적으로 학습된 local 모델들을 time-based에 기반하여 aggregation하는 방식입니다. `Graph Ladling`은 개별 모델의 학습이 완료된 후에 merging한다는 점에서 다릅니다.

## 5. Conclusion
`Graph Ladling` 논문은 `Model Soup`을 GNNs에 이용하여, GNNs model을 deepening하거나 widening하는 방식 없이 여러 graph에서 scalability를 가져가면서 성능을 향상시킬 수 있는 방법을 제시한 논문입니다. Graph structrued data는 graph data만의 특성(topology structure, relation)이 있어서, NLP나 Vision에서 성공한 방식을 그대로 적용하기는 어렵습니다. 그러한 Data modality의 차이를 위해서 새로운 Design이 필요합니다. `Model Soup`을 적절히 graph data centric하게 응용해서, GNNs의 본질적인 문제점을 해결하려고 시도한점이 흥미로웠습니다. 기존 [ICML Model Soup]이 Model-centric 접근의 방식이라면, `Graph Ladling`은 data-centric하게 model soup을 활용한 것에서 새로운 관점을 확인할 수 있었습니다. 


## Author Information
-   Heewoong Noh
    -   Affiliation:  [DSAIL@KAIST](http://dsail.kaist.ac.kr/)
    -   Research Topic: Deep Learning
    -   Contact: heewoongnoh@kaist.ac.kr

## Reference & Additional materials
-   Github Implementation
    -   논문에서 사용한 [Graph Ladling](https://github.com/VITA-Group/graph_ladling)

-   Reference
    -   [[ICML 2022] Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
    -   [[ICML 2023] Graph Ladling: Shockingly Simple Parallel GNN Training without Intermediate Communication](https://arxiv.org/abs/2306.10466)


