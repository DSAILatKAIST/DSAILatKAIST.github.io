---
title:  "[NeurIPS 2021] Do Transformers Really Perform Bad for Graph Representation"
permalink: Do_Transformers_Really_Perform_Bad_for_Graph_Representation.html
tags: [reviews]
---

# **Do Transformers Really Perform Bad for Graph Representation** 


## **1. Motivation**

Graph neural network (GNN)은 Input Graph 를 적절히 표현 (represent) 하는 데 있어서 최적의 도구로 여겨졌다. Graph Convolutional Networt (GCN), Graph SAGE, Graph Isomorphism Network (GIN) 등의 대표적인 GNN 모델은 독자적으로 발전되어 왔다.

이와 별개로 Transformer 는 딥러닝 모델로써 매우 큰 성공을 거두었는데, BERT와 GPT-3 를 위시한 language model 뿐만 아니라, Vision Transformer (ViT) 까지 개발되면서, Computer Vision 에서도 Convolutional Neural Network (CNN)의 성능을 압도하였다. Transformer는 (1) Attention 구조를 통해 node 간의 relation 을 global sense에서 잘 모사할 수 있으며, (2) positional encoding 등의 구조를 통해 효과적인 inductive bias를 부여할 수 있다.

Transformer 의 압도적인 성능은 당연히 Graph Representation 영역에서도 연구적 관심을 끌기에 충분했다. 사실 이는 매우 합리적인 접근인 것이, Transformer는 사실 Graph Neural Network 구조와 매우 크게 닮아있다. Node의 관계성을 attention 으로 매핑하는 것은 fully connected graph 의 massage passing 과 굉장히 유사한 부분이 있다. 

이 논문은 Transformer를 적절히 변형하여 graph representation 을 위한 transformer인 Graphormer를 개발하였다. 또한 이전의 여러 popular한 GNN 모델이 Graphormer의 특수 케이스라는 것을 증명하였다. 

## **2. Method**

사실 해당 논문은 매우 복잡한 테크닉과 증명이 들어있다. 이를 일일히 리뷰 하는 것은 오히려 큰 그림을 보는 시야를 해칠 수 있으므로, 이 리뷰에서는 Method의 의도와 Sketch 정도를 다루도록 하겠다. 구체적인 Technique과 증명은 논문과 코드를 참고하는 것이 좋겠다. 

해당 방법의 핵심은 Graph 의 특징을 Transformer 구조에 얼마나 잘 녹여내느냐 이다. 크게 두가지 Novel 한 방법이 제시된다: (1) Centrality Encoding, (2) Spatial Encoding. 

### **2.1 Centrality Encoding**

Transformer의 Attention 구조는 각 노드의 correlation 을 mapping 한다. Graph 에서는 좀 더 inductive bias를 부여해 줄 수 있는데, 그것은 node가 얼마나 많이 연결되어 있는지 (연결성)의 정도를 그 노드의 중요도로 평가할 수 있는 지점을 이용하는 방법이다. 즉 node $v_i$ 가 input degree와 output degree가 얼마나 되는지를 compute 하여 이를 bias로 이용해 줄 수 있다. 식으로 나타내면 다음과 같다. 

$$
h_i^{(0)}=x_i+z_{\operatorname{deg}^{-}\left(v_i\right)}^{-}+z_{\operatorname{deg}^{+}\left(v_i\right)}^{+}
$$
