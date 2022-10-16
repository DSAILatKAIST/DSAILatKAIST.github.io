---
title:  "[ICLR 2022] Online Coreset Selection for Rehearsal-based Continual Learning"
permalink: Online_Coreset_Selection_for_Rehearsal_based_Continual_Learning.html
tags: [reviews]
---


# **Title** 

Online Coreset Selection for Rehearsal-based Continual Learning

## **1. Problem Definition**  

Static한 graph setting에 맞춰져 있는 현재의 Graph Neural Networks (GNNs)는 현실의 상황과는 거리가 멀다.  
* Sequence of tasks에 continuously 적용될 수 있는 GNN을 고안하는 것이 본 논문의 주된 목적이다. 
* Continual Learning에서 발생하는 주된 문제인 catastrophic forgetting 문제도 보완한다.

## **2. Motivation**  


### 2.1 Continual Learning과 Catastrophic Forgetting

Graph Neural Networks (GNNs)은 많은 관심을 받고 있는 연구 분야이며, 눈에 띌만한 성장세를 보이고 있다.
현재까지의 GNN은 static한 graph setting에 초점이 맞춰져 개발되었다. 하지만 현실에서의 setting은 graph가 고정되어 있지 않고, 새로운 node와 edge 등이 끊임없이 추가된다. 이러한 상황에서 model은 정확성을 지속적으로 유지할 수 있어야 한다. 그렇다면 이러한 setting에서 새로운 task까지 잘 해내는 모델을 학습해야 한다면 어떻게 해야할까?

당연히 모델을 retraining 시켜야한다. 
모델을 retraining 시키기 위해 아래 두 가지 방법을 쉽게 떠올려 볼 수 있다.

첫째, 기존 데이터에 새로운 데이터까지 추가해서 모델을 처음부터 다시 학습하는 방법이다. 이 방법이 직관적일 수 있지만, 새로운 데이터가 수집될 때마다 전체 데이터셋에 대하여 모델의 모든 가중치값들을 학습하는 것은 시간과 computational cost 측면에서 큰 손실이다. 

그렇다면, 모델을 새로운 데이터로만 retraining 시키면 어떻게 될까? 이전에 학습했던 데이터와 유사한 데이터셋을 학습하더라도 아래의 그림처럼 이전의 데이터셋에 대한 정보를 잊어버리게 될 것이다. 이 문제를 일컬어 **Catastrophic Forgetting** 이라고 부른다.
> Catastrophic Forgetting : Single task에 대해서 뛰어난 성능을 보인 모델을 활용하여 다른 task를 위해 학습했을 때 이전에 학습했던 task에 대한 성능이 현저하게 떨어지는 현상

<div align="center">

![CGL example](https://user-images.githubusercontent.com/89853986/171803616-6104ebdb-34e3-4cb8-903f-aa9148b5e0e8.PNG)

</div>

Catastrophic forgetting은 neural network의 더욱 general한 problem인 "stability-plasticity" dilema의 결과이다. 
이 때, stability는 previously acquired knowledge의 보존을 의미하고, plasticity는 new knowledge를 integrate하는 능력을 의미한다. 

### 2.2 Limitation

Graph domain에서는 continual learning에 대한 연구가 놀랍도록 얼마 없다.  
이는 몇가지 한계점이 존재하기 때문이다.  
1. graph (non-Euclidean data) is not independent and identically distributed data.  
2. graphs can be irregular, noisy and exhibit more complex relations among nodes.  
3. apart from the node feature information, the topological structure in graph plays a crucial role in addressing graph-related tasks.

### 2.3 Purpose

1. 새로운 task를 학습할 때 이전 task에 대한 catastrophic forgetting 방지.  
2. 새로운 task 학습을 용이하게 하기 위해 이전 task의 knowledge를 사용.  
3. Online Corset Selection (OCS) 방법론을 고안하여 representative하고 diverse한 subset을 선정하여 buffer에 저장하고 새로운 task 학습에 함께 사용.
4. Current task에 대해서도 모든 data를 사용하는 것이 아닌 이전 task의 buffer들과 high affinity를 갖는 data를 선정하여 함께 training 시킴.

### 2.4 Contributions

* Continual Graph Learning (CGL) paradigm을 제시하여 single task가 아닌 multiple consecutive task (continual) setting에서 node classification task를 수행할 수 있도록 함.
* Continual node classification task에 기존 GNN을 적용할 때 발생하는 catastrophic forgetting 문제를 해결함.
* 유명한 GNN model에 적용 가능한 ER-GNN model을 개발하고, 이는 buffer로 들어갈 replay node를 선정할 때 기존 방법과는 다르게 influence function을 사용함.


## **3. Method**  
  
### 3.1 Online Coreset Selection

이 부분에서는 주어진 task에서 어떠한 기준으로 replay 시킬 data를 선정하는지에 대해 설명합니다.  

크게 두가지의 기준을 적용하는데, "similarity"와 "diversity"입니다.

#### 3.1.1 Minibatch similarity

<img width="549" alt="스크린샷 2022-10-16 오후 5 54 49" src="https://user-images.githubusercontent.com/89853986/196027003-a953f49a-e062-4475-835e-ae90e3b4e4d1.png">  

$b_{t,n} = {x_{t,n}, y_{t,n}} \in B_t$는 data point의 n-th pair를 의미하고, 분모의 좌측에 있는 식은 해당 datapoint의 gradient를 의미한다. 또한, 분모의 우측에 있는 식은 집합 $B_t$내에 있는 data들의 gradient의 평균을 의미한다.  

즉, 이 식은 특정 data point의 gradient와 집합 $B_t$내의 data들의 gradient의 평균 간의 similarity를 나타낸 식이다.


#### 3.1.2 Sample diversity
  
<img width="782" alt="스크린샷 2022-10-16 오후 6 11 45" src="https://user-images.githubusercontent.com/89853986/196027577-52824602-f595-4d0a-93e5-6145d6b639cf.png">

본 식에서는 특별히 새롭게 설명할 notation은 없을 것이다. 본 식은 특정 data point $b_{t,n}$과 subset 내의 다른 datapoint $b_{t,p}$ 간의 dissimilarity의 평균이다. 따라서 값이 클수록 subset 내의 다른 data와 다른, 즉 다양성을 갖는 data point라는 것이다.  


### 3.2 Online Coreset Selection for Current Task Adaptation 

<img width="954" alt="스크린샷 2022-10-16 오후 6 12 43" src="https://user-images.githubusercontent.com/89853986/196027593-ced3a9f0-9dde-4996-8697-a2664bef41d6.png">

이제 위의 section 3.1에서 다룬 두 가지 기준 "similarity"와 "diversity"를 고려하여 replay 시킬 data를 뽑아야 할 것이다.  
Similarity와 diversity 값을 더하여 그 값이 가장 큰 top k개를 선정한 $u^{\*}$집합을 선정한다.  

그 이후 아래와 같이 replay할 data를 갖고 loss가 최소가 되도록 model을 training 시키는 간단한 방법론을 제시하였다.


<img width="680" alt="스크린샷 2022-10-16 오후 6 13 07" src="https://user-images.githubusercontent.com/89853986/196027612-158471e7-2a59-4981-91c3-74616208434d.png">


### 3.3 Online Coreset Selection for Continual Learning  

지금부터는 저자가 제시한 OCS (Online Coreset Selection) 방법론에 대해 구체적으로 다룰 것이다.  
OCS 방법론의 목적은 previous task의 지식을 앞서 다룬 similarity와 diversity의 관점에서 고려하여 현재 task에서 활용도가 높은 coreset을 찾는 거시다.  
더 직관적으로 설명하자면, 현재 task에 대해서는 모든 dataset을 사용할 수 있는 것 아닌가라는 의문이 들 수 있다. 하지만 늘 그렇듯 real-world dataset에는 noise가 있기도 하고, 틀리지 않은 data 이지만 이전 task가 지향하는 방향과는 방향성이 다를 수 있다. 이에, 저자는 현재 task 이더라도, continual한 세팅에서 sequential한 학습에 도움이 되는 data subset을 선정하여 그 data들에 대해서만 training을 진행한다.  


#### 3.3.1 Coreset Affinity

<img width="628" alt="스크린샷 2022-10-16 오후 6 13 53" src="https://user-images.githubusercontent.com/89853986/196027637-03f9f7eb-a93d-43ec-860e-9f611003029f.png">  

위의 similarity 수식과 굉장히 유사하다. 분모의 우측에 있는 식이 의미하는 것은 coreset C로부터 randomly sampled 된 subset $B_c$에 대한 gradient의 평균이다. 따라서 이는 현재 task의 data distribution만 고려하는 것이 아니라 이전 task의 coreset과의 similarity도 고려한다는 의미이다.  

그렇다면 새로운 data selection equation은 아래와 같이 구성된다.  

<img width="1151" alt="스크린샷 2022-10-16 오후 6 14 15" src="https://user-images.githubusercontent.com/89853986/196027645-e90bb248-d44c-431a-9cc5-e066dc4f4bc3.png">  

그리고, 마찬가지로 아래와 같은 수식을 통해 current task의 coreset과 이전 task들에서 replay된 data들의 loss를 최소화하는 parameter를 찾는 방향으로 model이 training된다.  

<img width="726" alt="스크린샷 2022-10-16 오후 6 14 42" src="https://user-images.githubusercontent.com/89853986/196027655-da4f626d-89e3-4d1e-bd97-87f7b85f8e8f.png">


### 3.4 Algorithm  

위의 방법론을 하나의 algorithm으로 정리하면 아래와 같다.  

<img width="1232" alt="스크린샷 2022-10-16 오후 6 15 09" src="https://user-images.githubusercontent.com/89853986/196027674-50e06f92-2355-4010-9e01-ed2adb793666.png">



## **4. Experiment**  

### **4.1 Experiment setup**  


#### 4.1.1 Dataset  

**Domain Incremental**  
Rotated MNIST

**Task Incremental**  
Split CIFAR-100
Multiple Datasets (a sequence of five datasets)

**Class Incremental**  
Balanced and ?Imbalanced Split CIFAR-100



#### 4.1.2 baseline  

OCS과의 비교를 위해 continual setting에서 아래의 모델들과 비교하였다.  

~~~
  - EWC
  - Stable SGD
  - A-GEM
  - ER-Reservior
  - Uniform Sampling & k-means features
  - k-means Embeddings
  - iCaRL
  - Grad Matching
  - GSS
  - ER-MIR
  - Bilevel Optim
~~~


#### 4.1.3 Evaluation Metric  

본 논문의 주된 목적은 continual learning에서 고질적으로 발생하는 문제인 catastrophic forgetting을 줄이기 위함이므로 이에 알맞은 evaluation metric을 저자는 제안한다.  

* Average Accuracy : 일반적인 accuracy value이다.

* Average Forgetting : 이후 task를 학습하고 난 뒤, task의 accuracy가 떨어지는 정도를 측정한 값이다.

### **4.2 Result**  


#### 4.2.1 Quantitative Analysis for Continual Learning

<div align="center">

<img width="773" alt="스크린샷 2022-10-16 오후 7 19 44" src="https://user-images.githubusercontent.com/89853986/196030031-5b891dbe-a690-4443-9a50-b37a6996469e.png">

</div>

* Baseline model 모두 일정 수준의 catastrophic forgetting은 발생하는 것을 관찰할 수 있다.
* Balanced continual learning setting에서 random replay based methods (A-GEM & ER-Reservoir)과 비교하면 OCS는 average accuracy 관점에서 약 19%의 gain이 있다.
* 마찬가지로, balanced continual learning setting에서 forgetting average도 다른 baseline보다 현저히 낮은 수치가 관찰된다. 
* OCS는 imbalance setting에서 balanced setting에서보다 더욱 큰 강점을 보였다.  
* Accuracy와 forgetting 측면에서 baseline model들보다 훨씬 좋은 성능을 보였고, 이는 baseline model에서는 imbalance 상황에서 current task에 대해 coreset을 select하는 과정이 없으므로 biased estimation이 진행되어 performance degenerate이 일어났다고 볼 수 있다.  


#### 4.2.2 Noisy Continual Learning

<div align="center">

<img width="847" alt="스크린샷 2022-10-16 오후 7 30 40" src="https://user-images.githubusercontent.com/89853986/196030490-c293e4c2-740d-40ee-8464-ae1f289c0852.png">

</div>

* Gaussian noise를 적용하여 Rotated MNIST dataset을 noise하게 setting하였다.  
* 위의 table을 보면, noise는 모든 baseline의 성능을 상당히 저하시키는 것을 관찰할 수 있다.  
* 하지만 저자가 제안한 OCS의 경우, noise rate이 증가함에 따라 accuracy와 forgetting이 심각하게 저하되지는 않는 것으로 보여진다. 이는 task 내에서 similarity와 diversity를 고려하여 coreset을 선정하는 과정이 noise data를 상당부분 제외시키는 것으로 해석 가능하다.  


#### 4.2.3 Ablation Studies

<img width="591" alt="스크린샷 2022-10-16 오후 7 36 47" src="https://user-images.githubusercontent.com/89853986/196030702-cf33ae30-28ab-4559-89bc-7ba82aeacc6e.png">


* 본 실험은 gradient 활용의 효과를 검증한 실험이다.
* Gradient를 활용하여 coreset selection을 한 경우와 raw input (Input-OCS), feature-representations (Feat-OCS)를 활용하여 coreset selection을 한 경우를 비교하였는데, balanced / imbalanced CL setting에서 모두 gradient가 다른 두 방법에 비해 좋은 성능을 보였다.


<img width="589" alt="스크린샷 2022-10-16 오후 7 36 55" src="https://user-images.githubusercontent.com/89853986/196030706-9da699fc-630a-4c11-909d-de476780342a.png">


* Coreset에 들어갈 top k개의 data를 고르는 과정에서 "Minibatch similarity", "Sample diversity", "Coreset affinity"라는 세가지 식이 적용된다. 
* 본 실험은 이러한 세가지 component의 효과를 관찰하기 위해 component를 제외해보며 abalation study를 진행하였다. 
* Similarity와 diversity를 혼자만 사용하는 것은 성능 저하가 상당했다. Similarity만 사용할 경우, 중복되는 data를 선정할 가능성이 있고, diversity만 고려할 경우 representative한 data point를 선정하는데에 한계가 있을 것이다. 


## **5. Conclusion**  

#### 5.1 Summary

* Vision 분야의 continual learning problem에서 catastrophic forgetting을 방지할 수 있는 framework를 제안함.
* Continual learning의 큰 줄기 중에서 replay 방식을 채택하였고, online coreset selection을 접목시켜 similarity (대표성), diversity (overfitting 방지)를 동시에 고려하여 가장 영향력이 높은 node를 buffer에 저장하도록 함.
* 다양한 실험을 통해 targeting problem의 예시를 실제로 보임.


#### 5.2 Discussion

* 본 논문의 가장 주요한 novelty는 replay를 할 때, similarity (대표성), diversity (overfitting 방지)를 함께 접목시킨 것이다.
* 또한, 저자는 current task 내에서도 이전 task의 replay buffer와의 관계를 고려하여 data를 select하기 때문에 sequential task의 순서가 바뀌더라도 robust하게 대응할 수 있는 order-robustness의 특성을 가진다고 주장한다.
* Method에 담은 algorithm에서의 task의 개수를 미리 알고 있다는 가정은 real-world setting에 맞지 않은 strong assumption일 수 있겠다.
* 이렇듯 replay 방식에서는 buffer에 넣을 node를 선택하는 방법론이 매우 주요할 것이고, experience selection strategy에 대하여 연구해보는 것도 좋은 future research topic이 될 것 같다.


---  
## **Author Information**  

* Seungyoon Choi  
    * Affiliation : [DSAIL@KAIST](https://dsail.kaist.ac.kr/)
    * Research Topic : GNN, Continual Learning, Active Learning

## **6. Reference & Additional materials**  


#### 6.1 Github Implementation  
* https://github.com/jaehong31/OCS

#### 6.2 Reference 
* Yoon, Madaan, Yang, Hwang. "Online Coreset Selection for Rehearsal-based Continual Learning."
* Yoon, Yang, Lee, Hwang. "Lifelong Learning with Dynamically Expandable Networks."

