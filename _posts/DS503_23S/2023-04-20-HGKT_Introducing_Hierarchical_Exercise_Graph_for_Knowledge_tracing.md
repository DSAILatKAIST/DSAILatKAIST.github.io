---
title:  "[SIGIR 2021] HGKT: Introducing Hierarchical Exercise Graph for Knowledge Tracing"
permalink: HGKT_Introducing_Hierarchical_Exercise_Graph_for_Knowledge_tracing.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [Review] HGKT: Introducing Hierarchical Exercise Graph for Knowledge Tracing

## 1. INTRODUCTION

Knowledge Tracing은 컴퓨터를 기반으로 하여 디지털화된 교육이 점차 증가함에 따라 학생의 문제 풀이 sequence를 활용하여 지식상태를 추적하는 과업을 의미한다. 즉, 이전까지 학생이 풀이한 문제와 그 문제의 정오답 여부 및 기타 다양한 feature를 활용하여 아직 풀이하지 않은 문제를 풀 수 있는지 여부를 예측하는 것이다. Knowledge Tracing은 Hidden Markov Model을 사용하는 통계적 방식인 Bayesian Knowledge Tracing(BKT)을 시작으로 최근에는 DNNs(Deep Neural Networks)를 활용한 다양한 모델이 연구되고 있고, 성능 또한 매우 빠르게 발전하고 있다. 그러나, 기존 모델들이 공통적으로 가지고 있는 두 가지 한계점은 다음과 같다.

1. Information Loss: 문제 풀이 기록과 같은 표면적으로 드러난 데이터만을 활용함으로써 문제의 난이도 혹은 문제에 내재된 다양한 의미들을 고려하지 않아 충분한 정보를 고려하지 않았다.
2. Insufficient Diagnosis problem: 충분하지 않은 정보만으로 지식을 추적함에 따라 학생의 지식상태를 충분히 진단하지 못하여 학생의 학습수준을 정확하게 파악하지 못하였다.

본 연구는 위의 두 가지 한계점을 Hierarchical Graph Structure를 활용하여 보완함으로써 모델의 성능(performance prediction)과 설명력(interpretability)을 높이고자 하였다. HGKT의 기본 framework은 Figure 1과 같다.


![Figure1](https://user-images.githubusercontent.com/96431635/233821065-5eb62c9d-469e-4e40-a923-3426cfcd46ef.png) 

<center>

$Figure 1$

</center>


Figure 1의 Training Process에서 학생은 $e_ {1}$과 $e_ {3}$은 맞혔고, $e_ {2}$는 틀렸다. 이때, 'Coordinate Calculation'의 이해도를 동일하게 확인하는 $e_ {7}$과 $e_ {8}$의 문제 풀이 여부를 예측하는 경우, 'Pythagorean Theorem'의 이해도를 확인하는 $e_ {2}$는 틀렸고, $e_ {7}$과 개념적으로 관련있어 $e_ {2}$와 관련된 학생의 지식상태가 $e_ {7}$를 예측하는데 직접적으로 참고가 되고(direct support), $e_ {3}$은 $e_ {8}$과 개념적으로는 연결되지 않고 문제의 스키마(problem schema)만 관련있기 때문에 $e_ {8}$의 문제 풀이 여부를 예측하는데 간접적으로 참고가 된다(indirect support). 여기서 문제의 스키마는 그 문제의 풀이 방식 혹은 설명 방식을 나타내는 것으로 문제에서 묻고자 하는 수학적 개념과 다르다.

지금까지 설명된 내용을 바탕으로 본 연구의 contribution을 요약하면 다음과 같다.

$\bullet$ hierarchical graph를 사용하여 문제 간의 관계를 두 가지 유형(direct support relation, indirect support relation)으로 나타내어 문제 간의 관계를 기존의 Knowledge Tracing 연구보다 정교하게 활용하였다.  
$\bullet$ 각 문제에서 묻는 개념인 knowledge concept뿐만 아니라, problem schema를 새롭게 고려함으로써 문제를 더욱 효과적으로 representation하여 기존 모델의 한계점인 information loss를 극복하고자 하였다.  
$\bullet$ 모델의 architecture에서 두 개의 attention mechanism을 사용함으로써 hierarchical graph의 내용을 충분히 반영하고, 각 학생의 지식상태 또한 정확하게 반영하고자 시도하였다.  
$\bullet$ knowledge&schema (K&S) diagnosis matrix를 활용하여 knowledge concept과 problem schema의 습득 여부(mastery)를 동시에 고려함으로써, 기존 연구의 한계점인 insufficient diagnosis problem을 해결하고자 하였다. knowledge와 schema의 관계와 K&S diagnosis matrix는 Figure 2와 같다.


![Figure 2](https://user-images.githubusercontent.com/96431635/233821080-3e4580f7-bfcb-4826-90ae-ed704e63e8fa.png)  

<center>

$Figure 2$

</center>


<br>

## 2. RELATED WORK

Knowledge Tracing의 기존의 연구를 간단히 설명하면 다음과 같다.

$\bullet$ DKT(Deep Knowledge Tracing): 학생의 문제풀이 데이터에 sequnce가 존재하는 것을 고려하여 RNNs(recurrent neural networks)를 활용한 모델

$\bullet$ DKVMN(Dynamic Key-Value-Memory Network): 두 가지 별개의 memory matrix를 활용하여 각 knowledge concept에 대한 학생의 지식상태와 그 knowledge concept의 습득 여부를 RNNs에 반영한 모델  

$\bullet$ GKT(Graph-based Knowledge Tracing): 각 문제에 대한 학생의 지식상태 잠재변수(hidden knowledge state)를 그래프의 각 node에 임베딩한 후 그래프 구조를 활용한 모델

위의 모델들은 각 시점에서 SOTA의 성능을 보였지만 학생들이 풀이한 실제 문제의 텍스트 정보와 같은 좀 더 풍부한 정보를 담지 못함으로써 문제 간의 관계를 충분히 고려하지 못한 한계점이 존재한다. 물론, EKT(Exercise Enhanced Knowledge Tracing) 모델이 처음으로 문제의 텍스트를 분석하여 knowledge tracing에 활용하였지만 문제의 텍스트를 그대로 분석함에 따른 다양한 noise로 인해 그 정보를 충분히 사용하지 못함으로써 문제 간 관계를 고려하지 못하였다.

<br>

## 3. PROBLEM DEFINITION

논문에서 소개하는 HEG(hierarchical exercise graph)는 exercise 간의 'direct support relations'와 'indirect support relations'를 보여주는 두 가지의 그래프로 나타난다. 먼저, direct support relations는 하단 그래프로 나타나며 각 node는 exercise를 나타낸다. 그리고, indirect support relations는 상단 그래프로 나타나며 각 node는 problem schema를 나타낸다. 이것을 시각적으로 보면 Figure 3와 같다.


![Figure 3](https://user-images.githubusercontent.com/96431635/233821098-3fc3e0bb-90a5-4ce7-882e-d2340d57984e.png)  

<center>

$Figure 3$

</center>



HEG는 $(A, F, S_ {e})$로 표현되며, $A$ $\in$ ${0,1}^{\vert E\vert *\vert E\vert }$는 direct support relations graph의 adjacency matrix이며, $F$ $\in$ $R^{\vert E\vert *t}$는 각 노드의 feature matrix로 노드별로 t개의 feature를 표현하고, $S_ {e}^{\vert E\vert *\vert S\vert }$는 direct support relations graph와 indirect support relations graph의 연결관계를 나타낸다.

<br>

## 4. HGKT FRAMEWORK

<br>

### 4.1 Framework Overview

HGKT의 framework은 figure 4와 같다.


![Figure 4](https://user-images.githubusercontent.com/96431635/233821105-3b2f30eb-8d14-4b12-b068-b197a53c409c.png)  


<center>

$Figure 4$

</center>


System 1은 HGNN(hierarchical graph neural network)를 통해서 문제 간 hierarchical graph 구조를 학습하여 problem schema embedding을 생성하여 System 2에 전달한다. System 2는 전달받은 embedding value와 hierarchical graph 정보를 활용하여 exercise에 대한 학습자의 지식상태를 예측한다.

<br>

### 4.2 Direct Support Graph Construction

Direct support는 exercise 간의 개념과 풀이가 연관된 경우를 나타낸다. Hierarchical graph의 direct support graph는 support relation이 높은 경우와 낮은 경우로 나누어 다음과 같은 방법으로 생성되었다. $Sup$는 exercise 간 연관도를 나타내고, $ei$는 i번째 exercise를 나타내고, $R_ {ei}$와 $W_ {ei}$는 각각 학생이 정답 혹은 오답을 선택한 경우를 나타낸다.

1. $e_ {1}$과 $e_ {2}$의 support relation이 높은 경우 각 exercise의 조건부 정답률  
   $P(R_ {e1} \vert R_ {e2}) > P(R_ {e1} \vert R_ {e2}, W_ {e2}), P(W_ {e2} \vert W_ {e1}) > P(W_ {e2} \vert R_ {e1}, W_ {e1}), if Sup(e_ {1} \rightarrow e_ {2}) > 0$
2. $e_ {1}$과 $e_ {2}$의 support relation이 낮은 경우 각 exercise의 조건부 정답률  
   $P(R_ {e1} \vert R_ {e2}) = P(R_ {e1} \vert R_ {e2}, W_ {e2}), P(W_ {e2} \vert W_ {e1}) = P(W_ {e2} \vert R_ {e1}, W_ {e1}), if Sup(e_ {1} \rightarrow e_ {2}) = 0$

위의 방식에 의해 exercise 간 support value를 구성하면 다음과 같다. $Count$(($e_ {i}$, $e_ {j}$) = ($r_ {i}$, $r_ {j}$))는 학생이 $e_ {j}$에 $r_ {i}$를 답하기 전에 $e_ {i}$에 $r_ {j}$라고 답한 경우의 수를 의미한다. 분모가 지나치게 작아지는 것을 방지하기 위해 laplacian smoothing parameter로서 $\lambda_ {p}$ = 0.01을 분모에 더했다.

<center>

$P(R_ {e1} \vert R_ {e2})$ = $Count((e_ {2}, e_ {1}) = (1, 1)) + \lambda_ {p} \over \Sigma_ {r_ {1}=0}^1 Count((e_ {2}, e_ {1}) = (1, r_ {1})) + \lambda_ {p}$,  
$P(R_ {e1} \vert R_ {e2}, W_ {e2})$ = $\Sigma_ {r_ {2}=0}^1Count((e_ {2}, e_ {1}) = (r_ {2}, 1)) + \lambda_ {p} \over \Sigma_ {r_ {2}=0}^1\Sigma_ {r_ {1}=0}^1 Count((e_ {2}, e_ {1}) = (r_ {2}, r_ {1})) + \lambda_ {p}$

</center>

위 식에 의해 exercise 간 support value는 다음과 같다.

<center>

$Sup(e_ {1} \rightarrow e_ {2})$ = $max$ (0, ln $P(R_ {e1} \vert R_ {e2})  \over P(R_ {e1} \vert R_ {e2}, W_ {e2})$) + $max$ (0, ln $P(W_ {e2} \vert W_ {e1}) \over P(W_ {e2} \vert R_ {e1}, W_ {e1})$).

</center>

<br>

### 4.3 Problem Schema Representation Learning

Indirect support는 공통된 problem schema를 가지고 있는 exercise 간의 관계를 그래프로 representation하는 것으로, 이 또한 hierarchical graph로 표현된다. 방법은 다음과 같다.  
먼저, problem schema를 추출하기 위해 BERT[1]를 활용하여 exercise 질문의 keyword를 임베딩하고 이것을 hierarchical clustering[2]을 통해 representation한다. Hierarchical clustering은 각 데이터를 계층에 따라 순차적으로 클러스터링 하는 계층적 군집 분석(agglomerative hierarchical clustering)을 활용한 unsupervised cluster analysis method이다. 이것을 활용한 이유는 임계치(threshold) $\gamma$를 활용하여 그래프의 level 수를 정하고, 이것을 통해 서로 다른 수준의 problem schema를 계층화하여 각 schema에 해당하는 exercise를 군집화하기 위해서이다. 다음으로, 모든 exercise 간의 관계를 나타내는 direct support graph를 indirect support graph의 problem schema와 fusing하기 위해 DiffPool[3]에서 소개된 assignment matrix($S_ {e}$)로 두 그래프의 연결관계를 표현하였다. $S_ {e}$는 row에 direct support graph의 exercise node를 두고, column에 indirect support graph의 problem schema node를 두어 두 그래프의 연결관계에 대한 정보를 제공하는 matrix이다. 끝으로, exercise와 problem schema의 정보를 담고 있는 HEG = ($A, F, S_ {\gamma}$)를 HGNN을 활용하여 convolution layers와 pooling layers를 통해 direct support graph의 exercise 노드 정보를 공통된 problem schema로 합성곱하여 전파한다. 이때, HGNN은 두 개의 GNN을 통해 두 그래프를 모델링하는데, 이와 관련된 구체적인 annotation과 수식은 다음과 같다.  

<!-- $A_ e$ $\in$ ${0, 1}^{E*E}$= direct graph의 adjacency matrix  

$H_ e$ $\in$ $R^{E*t}$= direct graph의 exercise embedding matrix로 node별 feature 표현, $H_o$ = $F \in$ $R^{E*t}$  

$A_ s$ $\in$ {0, 1}$^{S*S}$= indirect graph의 adjacency matrix  

$H_ s$ $\in$ $R^{S*t}$ = indirect graph의 exercise embedding matrix로 node별 feature 표현   -->

<img width="775" alt="image" src="https://user-images.githubusercontent.com/37684658/233393548-f30e7ffb-c119-47e4-b9f3-84a3bf3ac0ec.png">


<center>

$H_ {e}^{(l+1)}$ = $GNN_ {exer}(A_ {e}, H_ {e}^{(l)})$,  
$A_ {s}$ = $S_ {\gamma}^TA_ {e}S_ {\gamma}$,  
$H_ {S}^{(l+1)} = S_ {\gamma}^TH_ {e}^{(l)}$,  
$H_ {S}^{(l+1)} = GNN_ {sche}(A_ {s}, H_ {s}^{(l)})$.

</center>

<br>

### 4.4 Sequence Modeling Process

HEG를 통해 exercise와 problem schema 관련 정보를 추출한 후 RNNs 기반인 LSTM을 통해 학생의 문제 풀이 sequence에 맞게 각 문제의 정답 여부를 예측한다. 정답 여부 예측의 process는 다음과 같다.

#### 4.4.1 Sequence Propagation

학생이 풀이한 exercise의 데이터는 'exercise의 개념($v_ {t}$)', 'exercise 풀이 결과($r_ {t}$)', 그리고 4.3에서 설명한 HEG process를 통해 얻어진 'problem schema($s_ {t}$)'가 joint embedding된 $x_ {t}$를 exercise interaction sequences로 LSTM에 입력하고 그 출력값에 활성화 함수를 적용하여 학생이 problem schema를 습득한 정도를 나타내는 $m_ {(t+1)}^{cur}$를 출력한다. 구체적인 수식은 아래와 같고, $W_ {1}과 b_ {1}$는 학습되는 parameter이다.

<center>

$h_ {t+1}, c_ {t+1}$ = $LSTM(x_ {t+1}, h_ {t}, c_ {t}; \theta_ {t+1})$,  
$m_ {(t+1)}^{cur} = ReLU(W_ {1} \cdot h_ {t+1} + b_ {1})$.

</center>

#### 4.4.2 Attention Mechanism

HGKT는 두 가지 종류의 attention mechanism(sequence attention, schema attention)을 활용한다.  
먼저, sequence attention은 이전까지 유사한 exercise를 풀이한 결과를 나타내는 정보로 다음과 같이 표현된다. $\gamma_ {\beta}$는 hyperparameter로 0시점부터 $\gamma_ {\beta}$시점까지의 sequence를 차단함으로써, computational cost를 줄이고, 교육심리학에서 주장하는 학습과정에서의 망각효과[4]를 반영하였다.

<center>

$m_ {t+1}^{att} = \Sigma_ {i=max(t-\gamma_ {\beta}, 0)}^t\beta_ {i}m_ {i}^{cur},$  
$\beta_ {i} = cos(s_ {t+1}, s_ {i})$.

</center>

다음으로, schema attention은 현재 예측하고자 하는 exercise와 이전까지의 problem schema의 연관성에 대한 정보로 앞서 설명한 indirect support graph process의 최종 출력값인 $M_ {sc} \in R^{k*\vert S\vert }$와 problem 간의 유사도를 나타내는 $\alpha_ {t} \in R^{\vert S\vert }$를 활용하여 다음과 같이 나타난다. 즉, 한 exercise에 대한 정답 여부의 정보는 유사한 problem schema를 공유하는 다른 exercise 풀이 과정에도 영향을 주도록 한다.

<center>

$m_ {t+1}^f = \alpha_ {t+1}^Tm_ {t+1}^{cur},$  
$\alpha_ {t+1} = Softmax(s_ {t+1}^TM_ {sc})$

</center>

정리하자면, 예측하고자 하는 exercise와 관련한 학생의 knowledge mastery 정보($m_ {t+1}^{cur}$), 학생의 모든 knowledge mastery 정보($m_ {t+1}^{att}$), 그리고 예측하고자 하는 exercise와 관련한 학생의 problem schema mastery 정보($m_ {t+1}^f$)를 활용하여 학생의 exsercise 풀이 여부를 예측하는데, 이 3가지 정보는 concat되어 최종 예측값을 출력한다. 이를 식으로 표현하면 다음과 같고, $W_ {2}과 b_ {2}$는 학습되는 parameter이다.

<center>

$\widetilde{r_ {t+1}} = \sigma(W_ {2}\cdot[m_ {t+1}^{att}, m_ {t+1}^{cur}, m_ {t+1}^{f}]+b_ {2})$

</center>

#### 4.4.3 Model Learning

본 연구에서는 negative log likelihood를 활용하여 model을 학습하였고, loss function은 예측값(predicted response)과 실제값(ground truth response)을 비교하는 cross entropy를 활용하였으며, Adam[5]을 optimizer로 활용하였다. cross entropy의 수식은 다음과 같다.

<center>

$loss = -\Sigma_ {t}(r_ {t}log\widetilde{r_ {t}}+(1-r_ {t})log(1-\widetilde{r_ {t}}))$.

</center>

#### 4.4.4 K&S Diagnosis Matrix

HGKT 모델은 knowledge($k_ {i}$)와 problem schema($s_ {i}$) 조합에 대한 t시점에서의 학생들의 지식상태($r_ {t}$)를 K&S Diagnosis Matrix를 통해 확인하였다. 이것을 활용하여 t 시점에서 knowledge와 problem schema별 학생의 능숙도를 구할 수 있다. $q_ {i,j}$는 각 problem schema과 knowledge를 보유한 exercise의 개수를 나타내고, $R_ {t}^k$와 $R_ {t}^s$는 시점 t에서 knowledge와 problem schema에 대한 능숙도를 나타낸다.

<center>

$q_ {i, j}$ = $\vert {(e_ {(k_ {i}, s_ {j})} \vert k_ {i} \in K, s_ {j} \in S)} \vert$  
$R_ {t, i}^k$ = $R_ {t, i}^{ks}d_ {i}^k$, $d_ {i, j}^k$ = $q_ {i, j} \over \Sigma_ {j}q_ {i, j}$  
$R_ {t, j}^s$ = $R_ {t, j}^{ks}d_ {j}^s$, $d_ {i, j}^s$ = $q_ {i, j} \over \Sigma_ {i}q_ {i, j}$  

</center>

#### 4.4.5 Interpretability of Problem Schema

Problem schema는 학생의 능력을 정확하게 측정할 수 있고 모델의 설명력을 높이는 요소로서 본 연구에서 소개하는 모델에서 핵심 역할을 한다. 그러나, 정해진 problem schema가 없는 exercise에서 해당 요소를 추출하는 것은 쉽지 않다. 이러한 문제를 해결하고자, HGKT 모델은 TextRank[6]로 각 문제의 텍스트에서 keyword를 추출함으로써 문제에 내재된 problem schema를 형성하고, 공통된 problem schema를 갖는 exercise의 cluster를 구성하여 문제를 해결하고자 시도하였다. 이를테면, Table 1의 "Exercise Cluster"에 있는 exercise는 "Condition keyphrases"에 있는 공통된 조건을 제시하고 있고, "Objective keyphrases"에 있는 "shortest side length"를 찾는 것을 공통적으로 목표하고 있다. 이를 통해서, 각 exercise의 problem schema는 "given ratio of lengths", "circumference", 그리고 "find the shortest side length"로 추출된다.  

<center>

$Table 1$

</center>  

![Figure 5](https://user-images.githubusercontent.com/96431635/233821126-61c22a74-bdc1-43c1-8979-70e402f004c9.png) 


<br>

## 5. EXPERIMENTS

<br>

### 5.1 Experiment Setup

1. Dataset: 본 연구를 진행한 기관(AIXUEXI Education Group)이 자체 구축한 온라인 수학 학습 시스템인 Aixuexi online system을 활용하여 수집한 Aixuexi dataset (2018)를 사용하였으며, dataset에 대한 자세한 내용은 Table 2와 같다.

<center>

$Table 2$  

</center>

![Figure 6](https://user-images.githubusercontent.com/96431635/233821133-5431ed98-f2d8-4c3f-a332-485c0c08b5d5.png)



2.  Parameter Setting  
    $\bullet$ BERT: no fine-tuning, embedding size = 768  
    $\bullet$ clustering threshold($\gamma$) = 9  
    $\bullet$ number of GNNs = 3 graph convolution layers for $GNN_ {exer}$, 1 graph convolution layer for $GNN_ {sche}$  
    $\bullet$ embedding size in HGNN  
    exercise = 64, problem schema = 30  
    $\bullet$ attention window size = 20  
    $\bullet$ LSTM hidden embedding size = 200  
    $\bullet$ learning rate = 0.01  
    $\bullet$ batch size = 32  
    $\bullet$ dropout rate = 0.5

3.  Evaluation Setting  
    $\bullet$ train-test ratio: 60%, 70%, 80%, 90%로 random split하여 실험  
    $\bullet$ evaluation metric: 문제별 학생의 learning state의 RMSE와 문제 풀이 여부의 classification에 대한 Accuracy의 5회 평균값 비교  
    $\bullet$ 8개의 Intel Xeon Skylake 6133 (2.5 GHz) CPUs, 4개의 Tesla V100 GPUs 기반 Linux server 환경에서 실험 진행

4.  Baselines  
    $\bullet$ Traditional educational models: BKT(Bayesian Knowledge Tracing)[7]  
    $\bullet$ Deep learning models: DKT(Deep knowledge Tracing)[8], DKVMN(Dynamic Ky-Value Memory Networks)[9], EKT(Exercise-aware Knowledge Tracing)[10], GKT(Graph Knowledge Tracing)[11]  
    $\bullet$ Variants of HGKT
    - HGKT-B: HEG를 활용한 problem schema를 embedding하지 않고, BERT를 활용한 direct graph의 text encoding을 직접 활용
    - HGKT-S: HEG를 활용하여 problem schema를 embedding하지 않고, hierarchical clustering 단계에서 one-hot 인코딩된 problem schema 활용  
      <br>

### 5.2 Comparison Results
Dataset을 활용하여 각 모델의 성능을 비교한 결과는 Figure 5와 같다.
![Figure 7](https://user-images.githubusercontent.com/96431635/233821148-59b59278-51ee-4e1d-801d-dbbce45f3621.png)  

<center>

$Figure 5$

</center>


$\bullet$ HGKT가 다른 model에 비해 우수한 성능을 보인다.  
$\bullet$ BERT로 text representation을 한 HGKT-B가 Bi-LSTM으로 text representation을 한 EKT(논문상에서는 EKTA로 오타)보다 높은 성능을 보이는 것으로 보아 문제의 text representation에서 BERT가 Bi-LSTM보다 우수함을 확인할 수 있다.  
$\bullet$ HGKT-S가 HGKT-B보다 우수한 성능을 보인 것으로 보아, text embedding을 직접 사용한 것보다 많은 exercise의 text embedding값을 합성곱 전파한 problem schema를 사용하는 것이 적은 noise를 일으키는 것으로 추론할 수 있다.  
$\bullet$ HGKT가 HGKT-S와 HGKT-B보다 우수한 성능을 보이는 것으로 보아, problem schema embedding을 representation하는 과정에서 aggregation을 기반으로 하는 HGNN이 우수한 효과를 보임을 추론할 수 있다.  
<br>

### 5.3 Analysis

#### 5.3.1 Ablation Study

support relations와 attention의 효과를 분석하기 위해 각각에 대한 ablation study를 진행한 결과는 Table 3와 같다.

<center>

$Table 3$  

</center>
![Table 3](https://user-images.githubusercontent.com/96431635/233821159-205e61dd-ea5f-4850-8cd4-48627d5101c7.png)  

결과를 요약하면, 두 가지 support relations를 고려한 경우가 그렇지 않은 경우보다 더욱 나은 성능을 보였고, attention을 사용한 경우가 그렇지 않은 경우보다 높은 성능을 보였다. 그 이유를 추론하면, HEG는 각 문제의 유사도를 기반으로 attention을 사용하여 문제 간의 관계를 효과적으로 representation하는 과정을 거친다. 그러나, support relations를 한 가지 종류만 사용하거나 병합하게 되면 각 문제의 noise가 modeling에 포함되어 성능을 떨어뜨릴 수 있다. 따라서, convolutional layers와 pooling layers를 활용하여 문제를 압축하고, attention을 사용함으로써 효과적으로 exercise representation이 가능하다.

#### 5.3.2 Graph Structure Analysis

HEG를 구성하는 방법에 따른 HGKT의 성능 차이를 분석하기 위해 exercise 간의 관계를 나타내는 graph structure analysis를 진행하였다. 먼저, 그래프를 구성하는 4가지 방법은 다음과 같다.

$\bullet$ Knowledge-based Method: 두 exercise가 연결되어 있으면 '1', 그렇지 않으면 '0'값을 갖는 adjacency matrix로 표현되는 densely conneted graph를 활용한 방법  
$\bullet$ Bert-Sim Method: BERT embedding의 cosine similarity를 활용하여 유사도가 hyperparameter $\omega$보다 크면 1, 그렇지 않으면 0값을 갖는 adjacency matrix로 표현되는 그래프를 활용한 방법  
$\bullet$ Exercise Transition Method: exercise 풀이 과정에서의 변화를 반영하는 adjacency matrix로 표현되는 그래프를 활용한 방법으로, exercise i를 응답한 후 exercise j를 응답하는 경우의 수를 $n_ {i, j}$로 두었을 때, $n_ {i, j} \over \Sigma_ {k=1}^{\vert E \vert}$ > $\omega$(hyperparameter)인 경우 1값을 갖고 그렇지 않은 경우 0값을 갖는 방법으로 adjacency matrix가 표현됨.  
$\bullet$ Exercise Support Method: bayesian statistical inference를 활용하여 그래프를 생성하는 방법으로, 두 exercise의 연관도를 나타내는 $Sup(e_ {1}, e_ {2})$가 $\omega$보다 크면 '1', 그렇지 않으면 '0'값을 갖는 adjacency matrix를 활용함.  
  
Figure 6는 그래프 구성 방법에 따른 AUC를 비교한 결과를 나타내고 있는데, bayesian statistical inference를 활용한 Exercise Support Method가 가장 높은 성능을 보였다. 이것은 해당 방법이 단순히 exercise의 feature를 사용하는 다른 방법과 달리 exercise간의 상호작용 정보가 exercise 풀이가 진행될 때마다 누적됨에 따른 효과로 확인되었다. 또한, edge-to-node의 비율이 3-4일 때 graph convolution이 가장 높은 성능을 보였다.


![Figure 8](https://user-images.githubusercontent.com/96431635/233821177-6364f2db-6242-42cd-a6d6-b33b1404a9f3.png)   

<center>

$Figure 6$

</center>

  
Problem schema의 clustering level은 연산력과 exercise representation에 영향을 미친다. 즉, 너무 높으면 연산의 효율성이 떨어지고, exercise를 지나치게 세밀하게 clustering 함으로써 오히려 성능을 저하시킬 수 있다. 이에 따라, 본 연구에서는 Figure 7에서와 같이 clustering level을 5~20 내에서 성능치를 비교하였고, 그림 (b)에 나타나는 것처럼 level이 9일 때 가장 높은 성능을 보였다.
![Figure 9](https://user-images.githubusercontent.com/96431635/233821189-07a995cc-6ea7-42aa-9f10-c45dc15d1caa.png)  

<center>

$Figure 7$

</center>

  
마지막으로, $GNN_ {exer}$과 $GNN_ {sche}$의 layer의 수에 따른 모델의 성능을 비교한 결과는 Table 4와 같고, $GNN_ {exer}$과 $GNN_ {sche}$의 layer가 각각 3개, 1개일 때 가장 좋은 성능을 보였다.
<center>

$Table 4$

</center>

![Figure 10](https://user-images.githubusercontent.com/96431635/233821196-397925f6-7a28-4357-8d13-f377c54c80b7.png)  



#### 5.3.3 Graph Structure Analysis

HGKT는 hierarchical graph 사용에 있어서 두 가지의 attention을 활용한 것이 또 하나의 contribution이다. 이에 따라, 두 가지 attention(sequence attention, schema attention)의 성능을 분석하였고, 그 결과는 Figure 8과 같다.


![Figure 11](https://user-images.githubusercontent.com/96431635/233821208-0fcef1ee-4722-4837-b89e-fc3140ec4bf9.png)  

<center>

$Figure 8$

</center>


두 가지 attention을 모두 사용한 경우가 모두 사용하지 않은 경우보다 높은 성능치를 보였고, 각 attention이 단일적으로 사용된 경우도 두 가지 모두 사용되지 않은 경우보다 높은 성능치를 보인 것으로 보아, 해당 모델의 memory를 강화하는데 두 attention이 모두 유의한 역할을 하는 것을 확인할 수 있었다. 또한, 일정 부분의 문제 풀이 정보를 차단함으로써 연산 성능과 학생의 망각효과를 고려하는 window size($\gamma_ {\beta}$)에 따른 성능을 비교한 결과, 20일 때 가장 높은 성능을 보였다. 이것은 학생의 능력을 측정하는 평가 문항의 적절한 개수를 정하는데 참고할 수 있다.  
<br>

## 6. CASE STUDY

지금까지 설명한 바와 같이 HGKT는 단순히 Knowledge Tracing의 성능을 높이는 것뿐만 아니라, 학생의 학습 과정에 대한 높은 설명력을 보여줌으로써 Learner Diagnosis Report와 Adaptive Question Recommendation과 같이 실제 교육 분야에서 학생의 학습을 지원하기 위해 다양하게 활용될 수 있다. 한 가지 예시를 보면 Figure 9와 같다.


![Figure 12](https://user-images.githubusercontent.com/96431635/233821218-665af127-5dc6-496d-a714-89c9a4e0e4be.png)   

<center>

$Figure 9$

</center>


학생이 $e_ {12}$문제를 틀렸을 때 해당 문제를 틀린 이유와 그러한 문제를 틀리지 않기 위해 어떠한 문제를 풀어야 하는지 정보를 제공함으로써 학생의 학습을 지원하는 것뿐만 아니라 학생에게 가장 적합한 문제들을 추천하는 데에도 활용될 수 있다.  
<br>

## 7. CONCLUSION

본 연구에서는 기존의 모델이 갖는 두 가지 한계점(information loss, insufficient diagnosis problem)을 극복하기 위해 다양한 방법을 시도하였다. 첫째, BERT를 활용하여 exercise text의 정보를 추출함으로써 exercise 간의 연관도를 direct support graph로 representation하고, direct support graph 내 연관된 exercise에서 공통된 problem schema 추출을 통해 indirect support graph로 representation하였다. 그리고, 이 두 그래프의 위계구조를 반영하기 위해 HEG를 활용하여 exercise에 내포된 다양한 정보를 충분히 활용하였다. 또한, 두 가지 attention(sequence attention, schema attention)으로 구성된 HGNN을 활용하여 noise를 최소화하며 HEG의 정보를 활용함으로써 knowledge tracing 과정에서 information loss를 극복하였다. 둘째, HEG와 K&S diagnosis matrix를 활용하여 학생의 knowledge mastery와 porblem schema mastery를 추적하여 설명력을 높임으로써, knowledge tracing에서 나아가 학생의 능력을 정확하게 진단하고 학생의 수준에 맞는 exercise 추천의 가능성을 높여 insufficent diagnosis problem을 극복하였다.  
<br>

## 8. IDEA PROPOSAL

본 연구에서 HEG와 HGNN을 활용하여 제시한 HGKT에 대해 future work을 위해 한 가지 추가적인 아이디어를 제안하자면 다음과 같다. 저자는 HGNN에서 HEG의 direct support graph에 있는 개개의 exercise 정보를 indirect support graph에 있는 problem schema로 전파할 때 exercise의 weight을 공유하며 전파하는 convolution layer를 사용하였다. 그러나, 이 과정에서 attention layer를 활용한다면 각 node에 있는 exercise embedding에 따라 problem schema에 exercise별 서로 다른 정보를 attention weight으로 전파함으로써 knowledge tracing 과정에서 information loss를 극복하는 가능성을 높일 수 있을 것이다.  
<br>
<br>

## REFERENCE

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert:
Pre-training of deep bidirectional transformers for language understanding. arXiv
preprint arXiv:1810.04805 (2018).

[2] Stephen C Johnson. 1967. Hierarchical clustering schemes. Psychometrika 32, 3
(1967), 241–254.

[3] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure
Leskovec. 2018. Hierarchical graph representation learning with differentiable
pooling. In Advances in neural information processing systems. 4800–4810.

[4] Hermann Ebbinghaus. 2013. Memory: A contribution to experimental psychology.
Annals of neurosciences 20, 4 (2013), 155.

[5] Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR).

[6] Rada Mihalcea and Paul Tarau. 2004. Textrank: Bringing order into text. In Proceedings of the 2004 conference on empirical methods in natural language processing.404–411.

[7] Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling the
acquisition of procedural knowledge. User modeling and user-adapted interaction
4, 4 (1994), 253–278.

[8] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami,
Leonidas J Guibas, and Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. In
Advances in neural information processing systems. 505–513.

[9] Jiani Zhang, Xingjian Shi, Irwin King, and Dit-Yan Yeung. 2017. Dynamic keyvalue memory networks for knowledge tracing. In Proceedings of the 26th international conference on World Wide Web. 765–774.

[10] Zhenya Huang, Yu Yin, Enhong Chen, Hui Xiong, Yu Su, Guoping Hu, et al. 2019.
EKT: Exercise-aware Knowledge Tracing for Student Performance Prediction.
IEEE Transactions on Knowledge and Data Engineering (2019).

[11] Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo. 2019. Graph-based
Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
In 2019 IEEE/WIC/ACM International Conference on Web Intelligence (WI). IEEE,
156–163.
