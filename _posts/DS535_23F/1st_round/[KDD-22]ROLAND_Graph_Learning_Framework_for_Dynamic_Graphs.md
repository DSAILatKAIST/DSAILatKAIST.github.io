<!-- ---
title: ROLAND Reivew
sidebar: Introduction_sidebar
keywords: introduction
permalink: template.html
toc: true
folder: introduction
--- -->

# **ROLAND: Graph Learning Framework for Dynamic Graphs**

> [Paper Link](https://arxiv.org/abs/2208.07239)
> [Github Implementation](https://github.com/snap-stanford/roland)

이 논문은 기존 연구들에서 제시한 dynamic graph에서의 graph representation learning의 한계점을 제시하고, static GNN에서 사용하던 테크닉들을 dynamic graph에서 활용할 수 있는 새로운 방법론인 ROLAND를 제시하였다.

### **Before We Start**
- **Static Graph**: 시간에 따라 변하지 않는 graph를 의미한다.
  - $G = \{V, E, X\}$
    * $V$: node set
    * $E$: edge set
    * $X$: attribute matrix  
- **Dynamic Graph**: 시간에 따라 node, edge, attribute 등이 변화하는 graph를 의미한다.
  -  $G(t) = \{V(t), E(t), X_ {v}(t), X_ {e}(t)\}$
     * $V(t)$: node set in timestep $t$
     * $E(t)$: edge set in timestep $t$
     * $X_v(t)$: node attribute matrix in timestep $t$
     * $X_e(t)$: edge attribute matrix in timestep $t$
- **Graph Nerual Network (GNN)**: GNN의 목적은 local network neighborhood로 부터 반복적인 message aggregation을 통해 node embedding을 learn하는 것이다.
  - ![img_gnn](https://i.ibb.co/vV6W0th/2023-10-15-03-43-26.png)
    * $h^{(l)}$: $l$-th layer GNN을 apply 한 모든 node들의 embedding
    * $m^{(l)}$: $l$-th layer에서의 message embedding
    * $MSG^{(l)}()$: $l$-th layer message-passing function, 다양한 종류의 function이 있음
    * $AGG^{(l)}()$: $l$-th layer aggregation function, 다양한 종류의 function이 있음
---
## **Introduction**  

Dynamic graph를 training 하는 것은 fraud detection, anti-money laundering, recommender systems 등 다양한 도메인에서 활용될 수 있다. 그렇기에 다양한 연구들을 통해 dynamic graph를 위한 GNN들이 개발되었지만 크게 다음과 같은 한계점들이 존재했다.

#### Limitations
- <span style="color:red"> Model Design </span>
  - 기존 dynamic graph 모델들은 성능이 좋은 static GNN 아키텍쳐를 응용하는 것에 실패했다.
    * GNN을 feature encoder로 사용한 후 sequence 모델 얹기
  - Skip-connection, batch normalization, edge embedding과 같은 테크닉들은 static graph GNN message passing에 효과적인 성능을 보여줬지만, dynamic graph에서는 응용되지 못하고 있다.
- <span style="color:green"> Evaluation </span>
  - 기존 연구들에서는 dynamic graph를 training 시키기 위해 dataset을 training, validation, test dataset으로 나눌 때 단순히 앞에서 부터 6:2:2로 자르는 등의 방법을 사용하였다.
  - 이는 dynamic graph가 가지는 시간에 따른 dataset distribution이 변화할 수 있다는 특성을 고려하지 않은 dataset 분리 방법이다.
- <span style="color:blue"> Training </span>
  - 대부분의 dynamic GNN에서는 모든 timestep $t$에 대한 그래프 정보 $G(t)$를 GPU 메모리에 저장해야 하기 때문에 scalability issue가 있다.
  - 그렇기에 edge 개수 200만개 이하의 작은 graph들로 실험을 진행하는 등의 문제가 있었다. 

ROLAND는 dynamic graph의 snapshot-based representation을 바탕으로 위의 limitation들을 타개하여 static GNN의 state-of-the-art 아키텍쳐들을 활용할 수 있도록 만들어졌다.

<!-- <p align="center">
  <img src="[http://some_place.com/image.png](https://i.ibb.co/RbLYk5J/2023-10-15-03-18-58.png)" />
</p> -->

![img_fig1](https://i.ibb.co/RbLYk5J/2023-10-15-03-18-58.png)

#### Tackling the Limitations
- <span style="color:red"> Model Design </span>
  - Static GNN에서 다른 GNN layer에 있는 node embedding들을 *hierarchical node state*들로 보는 새로운 관점을 제시하였다.
  - Static GNN을 dynamic하게 generalize하기 위해서는 *hierarchical node state*들을 새롭게 관찰되는 node와 edge들을 어떻게 이용하여 update할 지 정해야 한다.
- <span style="color:green"> Evaluation </span>
  - Dynamic graph에서 live-update를 할 수 있는 evaluation setting을 제시하였다.
    * 일별이나 주별로 batch를 만들어 evaluation을 진행하고 update를 할 수 있도록 하였다.
    * Data distribution이 timestep $t$에 따라 변화하는 dynamic graph의 특성을 고려하여 model update를 할 수 있다.
- <span style="color:blue"> Training </span>
  - 시간에 따른 모든 graph를 저장하는 것이 아닌 training 할 때 새로운 graph snapshot과 과거의 node state 정보들만 GPU 메모리에 저장하였다.
    * 이로 인해 5600만개의 edge들을 가지는 큰 graph를 train 시킬 수 있었다.
  - Dynamic graph에서의 timestep $t$가 다른 prediction 문제들을 다른 task로 생각함으로써 문제를 meta-learning problem으로 formulation하였다.
    * 위의 Figure 1 참고 


---
## **Proposed ROLAND Framework**

이제 위의 Tackling the Limitations를 조금 더 자세히 살펴보고자 한다. 해당 논문에서 다루는 자세한 notation들은 [paper](https://arxiv.org/abs/2208.07239)을 직접 참고하면 된다.

<a name="model_design"></a>
#### <span style="color:red"> Model Design </span>

먼저 작가들은 Figure 2에 기존의 static GNN과 이를 어떻게 dynamic GNN에 응용할지를 그림으로 나타내었다. 아래 Figure 2-(b)에서 볼 수 있듯이, embedding update 라는 블록을 통해 *hierarchical node state* $H$를 update 시켜줌으로써 static GNN을 쉽게 dynamic GNN으로 확장시켰다. 이 방법론대로라면 skip connection, batch normalization 등 static GNN에서 사용되던 효과적인 테크닉들을 dynamic GNN에서도 그대로 활용할 수 있다는 장점이 있다.

![img_fig2](https://i.ibb.co/C6cjxnN/2023-10-15-13-39-08.png)

결국 이 모델의 핵심은 *hierarchical node state* $H$를 어떻게 update할 지인데, 이는 algorithm 1에 자세히 소개되어 있다.

![img_algo1](https://i.ibb.co/t4VbsPV/2023-10-15-13-40-33.png)

이 논문에서 $GNN^ {(l)}$ 내부에 message embedding의 정보를 합치는 $AGG$ 함수로 sum, max, mean을 제안하였고, node embedding을 update하는 $UPDATE^ {(l)}$ 함수로 기존에 사용되던 간단하지만 효과적인 Moving Average, MLP, GRU를 제안하였다. 이는 기존 static GNN에서 사용하던 테크닉을 그대로 가져온 것으로 보인다.
모든 update가 끝나면 node $u$에서 $v$로 edge가 생길 확률을 $y_ {t}$를 MLP prediction head 블록을 통해 예측한다. 여기서 예측한 값은 다음에 설명될 live-update evaluation에 사용되어 모델을 update하는 것에 사용된다.

#### <span style="color:green"> Evaluation </span>

기존 GNN 연구들에서는 통상적으로 train, test, validation dataset을 단순히 timestep 앞에서부터 6:2:2로 나누는 방식을 사용해왔다. 하지만, 이는 앞서 언급했듯이 변화하는 data distribution을 고려하지 못하게 되는 치명적인 단점이 있다. 이 paper에서는 해당 문제를 live-update evaluation을 제안함으로써 해결한다.

![img_fig3](https://i.ibb.co/c8B4z0C/2023-10-15-15-03-03.png)

위 Figure 3에서 볼 수 있듯이, 이전 시점의 node embedding과 현재 시점의 새로운 graph snapshot을 input으로 GNN에 넣어 prediction $\hat{y}_ {t}$를 얻은 후, 실제  $y_ {t}$와 비교하여 mean reciprocal rank (MRR)을 통해 평가하게 된다. 자세한 training 과정은 아래 algorithm 2에 소개되어 있다.

![img_algo2](https://i.ibb.co/jrSF50V/2023-10-15-15-02-46.png)

Update에 앞서서 먼저 live-update를 위한 link predicion label들을 각 timestep에 모아준 후, training set과 validation set으로 나누어준다. Training set $y^ {(train)}$은 GNN을 fine-tuning하는 것에 사용되고, validation set $y^ {(val)}$은 early stopping criterion으로써 활용된다 (Algorithm 2의 3, 5 ,6번째 줄). 이는 $y^ {(val)}$으로 계산된 $MRR^ {(val)}$이 증가하는 것이 멈출 때까지 반복된다 (Algorithm 2의 4, 7번째 줄).
$MRR$은 propose 된 결과들 중 실제 값이 몇 번째 rank에 있는지에 관련된 metric이다. 예를 들어, 단수형 단어가 주어졌을 때 복수형 단어를 맞춰야하는 task가 있다고 생각해보자. 이 때, apple을 input으로 주고 모델이 appl, apples, applet 를 뱉었다면 Rank는 $2$, Reciprocal Rank ($RR$)는 그의 역수인 $\frac{1}{2}$로 정의된다. $MRR$은 모든 $RR$의 평균으로 input query를 $Q$라 할 때 다음과 같이 정의된다.

$ MRR=\frac{1}{\vert Q \vert} \sum_ {i=1}^ {\vert Q \vert} \frac{1}{rank_ {i}} $ 

 Training이 모두 끝나면 각 timestep $t$에 계산된 $MRR_ {t}$의 평균을 performance metric으로 사용하여 모델을 평가한다.

#### <span style="color:blue"> Training </span>

위 Figure 3에 제시된 ROLAND의 아키텍처를 살펴보면, 각 시점에서 GNN을 update하는 것에 필요한 input은 이전 시점 node embedding $H_ {t-1}$과 새롭게 들어오는 graph snapshot $G_ {t}$임을 알 수 있다. 때문에, ROLAND의 memory complexity는 graph snapshot의 개수에 agnostic하고 scalable함을 알 수 있다.

![img_algo3](https://i.ibb.co/42gtLdr/2023-10-15-16-19-14.png)

ROLAND에서 제안된 또 다른 방법론 한 가지는 dynamic graph에서의 prediction task를 meta-learning 문제로 formulation하는 것이다. 예를 들어, daily graph snapshot이 input이라고 생각했을 때, 금요일의 모델과 토요일의 모델은 크게 다를 것이다. 때문에 금요일 모델을 단순히 fine-tuning 후 토요일 모델이라고 하는 것은 모델 성능에 좋지 않은 영향을 끼칠 수 있다. 이에 대해 작가들은 meta-model $GNN^ {(meta)}$를 찾는 Algorithm 3을 위와 같이 제시하였다.


---
## **Experiments**  

#### **Experiment setup**  
* **Datasets**
  * 아래 표는 이 paper에서 사용된 dataset의 종류와 각 dataset의 특성을 정리해 둔 표이다.
  * ![img_table1](https://i.ibb.co/3hZv9MX/2023-10-15-17-18-24.png)

* **Baselines**
  * EvolveGCN-H
  * EvolveGCN-O
  * T-GCN
  * GCRN-GRU
  * GCRN-LSTM
  * GCRN-Baseline

* **Evaluation Metric**
  * Mean Reciprocal Rank (MRR)
    * $ MRR=\frac{1}{\vert Q \vert} \sum_ {i=1}^ {\vert Q \vert} \frac{1}{rank_ {i}} $

#### **Results**  
실험은 크게 기존 dataset splitting 방법과 논문에서 제시한 live-update를 적용한 방법 두 가지로 진행되었다. 먼저 아래 Table 2는 기존의 dataset splitting 방법을 사용하여 dataset을 나눴을 때 각 baseline들과 ROLAND의 결과이다. ROALND는 앞에 [Model Design](#model_design)에서 제시한 것과 같이 $UPDATE$ 함수를 Moving Average, MLP, GRU 3가지를 사용하여 결과를 비교했다.
  
![img_table2](https://i.ibb.co/8b3P8q6/2023-10-15-17-19-54.png)

모든 dataset에서 ROLAND GRU가 가장 좋은 baseline 대비 $MRR$이 적게는 43.33%에서 많게는 73.74% 향상된 결과를 보여주었다.
Table 3에는 이 paper에서 제시한 live-update 방법을 활용하여 train 시켰을 때의 결과를 기재하였다. 마찬가지로, 한가지 dataset을 제외한 모든 dataset에서 baseline들 대비 큰 $MRR$ 향상을 보여주었다.

![img_table3](https://i.ibb.co/nPTtyWx/2023-10-15-17-53-45.png)

ROLAND에서 제시한 meta-model $GNN^ {(meta)}$를 사용하면 평균 performance가 적게는 2.84%에서 많게는 13.19% 증가하는 결과를 보여주었다. 더하여, $GNN^ {(meta)}$를 사용했을 때 $MRR$의 standard deviation이 줄어듦으로써 stability가 높아짐을 보였다.


---
## **Conclusion**  

이 논문은 기존 static GNN에서 사용하던 테크닉과 아키텍처들을 dynamic graph에 어떻게 적용할 수 있을지 제안하였다. Paper은 크게 model design, evaluation, training 3가지 부분으로 나누어 기존 연구들의 한계점과 ROLAND의 contribution을 정리하였다.
새로운 dynamic GNN 모델을 scratch부터 만들지 않고 static GNN과 dynamic GNN을 연결하는 다리를 제안했다는 점, 그리고 live-update를 통해 scalability issue를 해결했다는 점에서 이 논문의 contribution이 큰 것 같다.

---  
## **Author Information**  

> Author name: Haeun Jeon
> - Contact: haeun39@kaist.ac.kr
> - Affiliation: Financial Engineering Lab., KAIST
> - Research Topic: Stochastic Optimization, End-to-end learning, Portfolio Optimization

<!-- ## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

> [Github Implementation](https://github.com/snap-stanford/roland)
* Reference   -->