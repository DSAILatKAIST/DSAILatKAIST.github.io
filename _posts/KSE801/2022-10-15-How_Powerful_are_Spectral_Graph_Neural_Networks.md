---
title:  "[ICML 2022] How Powerful are Spectral Graph Neural Networks"
permalink: How_Powerful_are_Spectral_Graph_Neural_Networks.html
tags: [reviews]
---

# **How Powerful are Spectral Graph Neural Networks** 
 

## **1. Introduction**  

Message Passing Framework를 활용하여 이웃한 노드의 정보를 aggregate 함으로써 노드들의 표현(representation)을 학습하는 Graph Neural Network(GNN)는, 그동안의 Graph Representation Learning 방법론들 가운데 여러 Downstream Task에서 State-of-the-art(SOTA) 성능을 보여줬습니다.

그 한 갈래인 Spectral GNN은, Spatial한 그래프 신호(graph signal)를 Graph Laplacian을 활용해 Spectral하게 필터링하고 필터링된 신호를 다시 Spatial domain으로 가져와 prediction을 수행합니다. GCN[2], GAT[3]과 같이 Popular한 모델이 등장하기 이전부터도 ChebyNet[4]과 같은 Spectral GNN이 연구되었고, 그중 GCN의 경우 ChebyNet에서의 Spectral 필터를 단순화한 모델입니다.

이외에도 이 논문에서 언급되는 여러 Spectral GNN 모델들이 등장하지만, 저자들은 이러한 Spectral GNN 모델의 표현력(expressive power)에 대해 분석하고 연구한 논문이 없었음을 지적합니다. 저자들은 이 논문을 통해 Spectral GNN 모델의 표현력에 대해 이론적인 분석을 제시하고, 이를 바탕으로 JacobiConv라는 Spectral GNN 모델을 제안합니다.

이 논문의 Contribution은 아래와 같이 정리할 수 있습니다.
 1. 이 논문에서는 비선형성(non-linearlity)이 없는, 간단한 형태의 Linear Spectral GNN조차도 강력한 표현력이 있음(universal함)을 이론적으로 보이며, 그런 표현력을 갖추기 위한 조건을 제시하고 이에 대해 분석합니다.
 2. 또한, Linear Spectral GNN의 Universality 조건과 그래프 동형 테스트(Graph Isomorphism Test; GI Test)와의 연관성에 대해서도 분석합니다. 이런 GI Test를  분석은 Spatial한 GNN에서 다뤄진 바 있으며, 
 3. 



## **2. Preliminaries**  

Please write the motivation of paper. The paper would tackle the limitations or challenges in each fields.

After writing the motivation, please write the discriminative idea compared to existing works briefly.


## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

### **Result**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  



## **5. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

---  
## **Author Information**  

* Xiyuan Wang  
    * Institute for Artificial Intelligence, Peking University
* Muhan Zhang
    * Institute for Artificial Intelligence, Peking University
    * Beijing Institute for General Artificial Intelligence

## **6. Reference & Additional materials**   
The Official Implementation은 [여기](https://github.com/GraphPKU/JacobiConv)에서 확인 가능합니다.
 1. Xiyuan Wang and Muhan Zhang. _How Powerful are Spectral Graph Neural Networks_. ICML, 2022.
 2. Thomas N. Kipf and Max Welling. _Semi-Supervised Classification with Graph Convolutional Networks_. ICLR, 2017.
 3. Petar Veličković et al. _Graph Attention Networks_. ICLR, 2018.
 4. Michaël Defferrard et al. _Convolutional neural networks on graphs with fast localized spectral filtering_. NeurIPS, 2016.
 5. Keyulu Xu et al. _How Powerful are Graph Neural Networks?_ ICLR, 2019.
 6. Stephen Boyd and Lieven Vandenberghe. _Convex Optimization_. Cambridge University Press, 2009.
 7. Richard Burden and J. Douglas Faires. _Numerical Analysis_. Cengage Learning, 2005.

