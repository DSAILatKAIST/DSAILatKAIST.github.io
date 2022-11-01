---
title:  "[KDD 2022] How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications"
permalink: How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications.html
tags: [reviews]
---

GNN의 Robustness 는 최근에 많은 관심을 받아오고 있습니다. GNN 의 입력으로 넣는 그래프의 구조를 아주 조금만 변형시켜도 전혀 다른 예측값을 하게 함으로써 모델을 크게 망칠 수 있게 되는데요.
본 논문에서는 GNN의 Robustness 와 Graph 의 Heterophilly 와의 연결점을 처음으로 언급하고, 이 연관성을 이론적, 또 경험적으로 보여주는 논문 입니다. 




##  1. Task Definition

우선 Robustness 를 확인하는 Task 를 간단히 소개하겠습니다.

GNN의 Robustness 는 Attack 이 포함된 Graph 로 그래프 태스크를 수행했을 때, 얼마나 정확히 예측하는가를 평가척도로 측정합니다.

여기서 Attacked 이 포함된 그래프란 다음과 같이 정의되며,

![1](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk1.PNG)

그래프 태스크란, Node Classification, Graph Classfication, Link Prediction 그리고 Node Clustering 과 같은 전형적인 태스크들을 이야기 합니다.





## 2. Motivation

그렇다면 현재 연구들을 살펴보았을 때, 본 연구의 Contribution이 무엇인지 알아보겠습니다. 

### GNN for Heterophilous Network.

기존 GNN의 연구 분야 중에서, Heterophily 그래프를 위해 특별히 디자인을 하는 연구분야가 존재합니다. Heterophilly 란, "나의 친구는 나와 유사할 것이다" = (Homophilly) 의 반대 개념인데요.
즉, 나와 친구임에도 나와 매우 다른 특성을 지닌 친구들을 많이 포함하고 있는 그래프를 Hetrophilly Network 라고 부릅니다. 논문의 표현을 빌려 좀 더 엄밀히 정의하면,

![2](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk2.PNG)

위와 같은 homophily ratio 를 계산할 수 있는데요. 나와 같은 레이블 y 로 연결된 edge가 많이 존재한다면 homophilly ratio가 높을 것 (homophilly network)이고, 반대로 지나치게 낮으면 homophilly ratio 가 낮을 것 (heterophilly) 입니다.

GNN은 smooth 한 정보를 많이 잡는 특성을 지녔기 때문에, 주변 정보가 되게 다를 경우에는 잘 작동하지 않았습니다. 그동안은 이러한 Heterophilly Network 를 위해 GNN 에 특별한 디자인을 추가하여, 이를 해결하고자 했습니다.


### Robustness of GNN 

GNN의 성능을 망치게 되는 Graph Edge 는 서로 비슷하지 않은 유저를 Edge 로 연결해주는 그래프를 넣어주는 것 입니다. 그렇게 되면 학습된 GNN 은 기존에 배웠던 패턴과 다른 연결에 의해 예측력이 급격히 떨어지게 됩니다.


### Paper Contribution

언뜻, Heterophilly 와 Robustness 는 비슷한 속성을 공유한 듯 보이지만, 한번도 이 둘에 대해 정확히 연결점을 찾는 연구는 이루어지지 않았습니다. 본 논문에서는 둘의 관계를 이론적, 실험적으로 명확히 보여주고, GNN 의 robustness 를 올리는데 한 걸음 나아갈 수 있는 Intuition 을 제공합니다.


## 3. Theoretical Analysis

본 논문에서는 우선 Homophilly Network 와 Heterophilly Network 일 때 Attack 의 경향이 달라지기 때문에, 각각 나누어서 Theorem 을 제시합니다.



### Attack for Homphilly Network

Theorem 1은 Homophilly Graph에 적용 되는 이야기인데, Theorem1을 살펴보면 다음과 같습니다.

![3](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk3.PNG)


이를 직관적으로 해석해보면, 잘 학습된 GNN 이 존재할 때, 이 GNN을 망치는 효과적인 Structure Attack 은 1) 다른 Feature를 가진 노드끼리 서로 연결을 해주는 것이고,

2) 간접적으로 (2-hop neighbor 이상으로) 연결해주는 것보단 직접적으로 (1-hop neighbor 로서) 연결해주는 것이 GNN 의 성능을 크게 망친다고 할 수 있습니다.

사실, 이 Theorem 은 지금까지 존재했던 연구들에서도 널리 알려져 있던 사실이라 특별하지는 않지만, 이론적으로 다시 한번 밝힌 것으로 보입니다 [1].



### Attack for Heterophilly Network

Theorem 2은 Heterophilly Graph에 적용 되는 이야기이고 Theorem1 보다 다소 상황에 따라 복잡한 결과를 보입니다. 


![4](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk4.PNG)


이를 정리해서 설명해보면, Heterophilly 의 경우 High-degree 노드와 Low-degree 노드를 다른 전략으로 공격해야 합니다. low-degree 의 노드의 경우, 기존과 동일하게 다른 유저를 연결해주면 서로에게 악영향을 크게 끼치게 됩니다. 이는 직관적입니다. 반면에, high-degree 노드의 경우, 1) 또 다른 유사한 특성을 지닌 high-degree 노드를 연결해줍니다. 개인적인 추측을 더해보면, Heterophilly Network 가 어차피 다른종류의 Feature 를 연결한 Neighbor를 많이 지니기 때문에, high-degree 노드 끼리의 연결이 간접적으로는 그래프 특성을 더 크게 망치는 방법이기 때문입니다.
2) 또 다른 유사하지 않은 특성을 지닌 low-degree 노드를 연결해주는 것도 heterophilly network 로 학습한 GNN을 망치는 전략이라고 합니다. 요약해보면, high-degree 노드끼리의 연결을 제외하고는 모두 다른 특성을 지닌 유저를 연결해주는 것이 모델에게 좋지 않음을 알 수 있습니다.


### Robust GNN Designs.

위와 같은 Attack 에 대한 분석을 바탕으로, 본 논문에서는 GNN의 디자인이 다음과 같아야 한다고 주장합니다.

![5](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk5.PNG)

즉, 각각의 layer 별로 얻은 representation 을 concat 해서 skip-connection의 효과를 누릴 수 있게 하자는 것인데요. 그렇게 하자고 하는 이유는 "ego-embedding (target node의 embedding) 이 aggregator 의 영향을 줄일 수 있고, 그렇게 하면 attack 의 영향을 줄일 수 있기 때문" 입니다. 사실 위와 같은 사실은 너무나도 당연하며,

Heterophilly GNN 과 관련된 기존 연구들에서 사용했던 전략입니다 [2, 3]. 따라서, 이 논문에서 하고자 하는 말은, "Heterophilly GNN 모델들이 Attack을 잘 방어할 것이다" 라는 것 입니다. 지금까지의 분석들은 조금 당연하지만, 이 사실 만큼은 조금 특별한 것 같고, 이것이 논문을 Accept 되게한 이유이지 않을까 싶습니다.




## 4. Experiment & Result

실험에서 검증하고자하는 Research Question (RQ) 는 다음과 같습니다.

**RQ 1) 그래프 어택은 대부분 Heterophillous 하다. 즉, 다른 Feature를 지닌 노드 사이에 Edge로 연결하는 경향이 있다.**

**RQ 2) Heterophillous Design 이 Robustness를 향상한다.**

**RQ 3) (RQ2와 다른 방법으로) Heterophilly 가 보정된 GNN 이 가장 Robust 하다.**

### Experimental Setup
GNN의 Robustness는 Poisoning 세팅 (training/testing 모두 같은 attacked 그래프를 사용)에서 실험 합니다. Attack의 종류에는 특정 노드만 공격하는 Targeted Attack과 그래프 전체적으로 공격하는 non-targeted attack이 있는데, Targeted Attack으로는 Nettack, Untargeted Attack으로는 Metattack을 채택하였습니다.
데이터는 citation network cora/cite/pubmed 와 snap 이라는 데이터를 사용하는데, 각각 homophilly ratio 가 매우 달라, 그에 따른 경향을 보여주기 위해 다양한 데이터에서의 실험을 진행합니다. Baseline은 GCN/GAT 와 같은 기본 모델과, H2GCN/GraphSAGE/FAGCN/APPNP 등 Heterophillous Design을 채택한 GNN 들을 사용하며 Heterophilly 보정을 위한 방법으로는 SVD와 Soft Medoid Aggregation (SMGDC) 를 사용합니다.

### RQ 1) 그래프 어택은 대부분 Heterophillous 하다.

아래 표는 Targeted Attack인 Nettack 알고리즘을 각각 그래프에 사용하여 그래프를 공격했을 때의 결과입니다. 기대하는 바와 같이, Heterophilly 한 Attack을 더해주는 방식으로 그래프를 공격해야 효과적으로 모델이 망가집니다.

![6](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk6.PNG)

그림은 앞선 Theorem 에서 언급한 degree 를 분석한 것으로 보이는데, homophilly ratio 가 높은 cora, citeseer, pubmed 의 경우 low-degree 사이들을 연결합니다.
반면에, heterophilly graph (FB100, Snap) 의 경우 앞서 말한대로 degree는 low-degree는 high-degree 와 연결하는 경향과 low-degree 사이들을 연결하는 2가지 경향 모두를 보여줍니다.


### RQ 2) Heterophillous Design 이 Robustness를 향상한다.

아래 표는 Targeted/Untargeted 상황에서 각 모델별 Accuracy를 나타낸 표입니다. 표를 아래부터 살펴보면, MLP의 경우 그래프 정보를 사용하지 않기 때문에 Poison/Clean 상황에서 성능이 동일합니다.
하지만, 실제로 우리는 얼마나 Poisoned 되어 있는지 상황을 모르기 때문에 MLP를 사용하는 것은 좋지 않습니다. (일반적으로는 그래프 정보를 사용하는게 더 도움이 되기 때문이죠)

그 위의 그룹은 RobustGNN 그룹입니다. GNNGuard / ProGNN 과 같이 Poisoned 세팅에서 잘되는 State-of-art 모델들과 비교를 진행하였습니다 [1].

그 위의 그룹이 바로 Heterophilly GNN 그룹입니다 [2, 3]. 놀라운 사실은 Heterophilly GNN 은 Attack을 Defense 하기 위한 방법이 전혀 아닌데도 untargeted attack 상황에서 상당히 좋은 performance 를 보이고 있습니다.
이는 heterophilly design 이 attack 의 dissimilar feature를 연결한 패턴을 어느정도 잡아낼수 있기 때문에, 정보를 aggregation 하는 과정에서 이를 반영할 수 있기 때문으로 보입니다. 이는 저자가 언급한 heterophilly design 이 robust 하다는 것을 뒷받침 해줍니다.
그럼에도 불구하고, nettack 과 같이 특정 노드를 타겟으로 하고 평가하는 방식에서는 그렇게 좋은 성능을 보이지는 않습니다. 대신에, SVD와 SMGDC 와 같은 방식을 Heterophilly GNN에 결합한 경우에는 Nettack 까지도 잘 막는 결과를 보여줍니다 [5]. 즉, 기존의 Heterophillous design 을 잘 조합하면, SOTA Defense 방법들을 다 이겨낼 수 있는 것 입니다.



![7](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk7.PNG)


### RQ 3) Heterophilly 가 보정된 GNN 이 가장 Certifiably Robust 하다.

기존 attacked 그래프에서 평가하는 것은 다른 그래프에서 까지 robust하고 할 수 없습니다. 따라서 accumulated certificablity (AC) 의 개념을 도입합니다.
이는, random perturbation 의 강도 (radius) 늘려가면서 모델이 예측을 변하지않는 최대 radius 를 측정합니다. 즉, radius 가 길 수록 더욱 robust 하게 예측을 유지한다고 해석할 수 있습니다.

![8](/images/How_does_Heterophily_Impact_the_Robustness_of_Graph_Neural_Networks_Theoretical_Connections_and_Practical_Implications/figk8.PNG)

아래 표와 같이 heterophilly design 들은 certifiable robust 함을 확인할 수 있습니다.



## 5. Conclusion

본 논문에서는 GNN의 Robustness 와 Heterophilly-level 과의 상관성을 파악하고, 그를 해결할 수 있는 Key Intuition을 제공하였습니다.
두 연구사이에는 기존의 접점이 있지 않았으며, Heterophilly designed GNN 을 사용하는 것 만으로도 Robust GNN 모델들을 이길수 있다는 것이 놀라웠습니다.

### Take home message 
Adversarially Attacked Graph 가 지닌 특성을 이용해서 우리는 이제 Attacked Edge들을 Heterophillous Neighbor로 간주할 수 있다.
따라서, Heterophilly GNN과 같이 보다 일반적인 모델들을 잘 이용하면 adversarial Attack을 방어할 수 있다.


### Author

**윤강훈 \(Kanghoon Yoon\)** 

* Affiliation \(KAIST Industrial Engineering Department\)
* \(optional\) ph.D students in DSAIL


## Reference & Additional materials

1. Wei Jin et al. Graph Structure Learning for Robust Graph Neural Networks. KDD'20
2. Zhiong Ju et al. Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs. NeurIPS'20
3. Deyu Bo et al. Beyond Low-frequency Information in Graph Convolutional Networks. AAAI'21
4. Simon Geisler et al. Reliable Graph Neural Networks via Robust Aggregation. NeurIPS'20



