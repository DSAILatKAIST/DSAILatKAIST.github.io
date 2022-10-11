---
title:  "[SIGIR 2022] Few-shot Node Classification on Attributed Networks with Graph Meta-learning"
permalink: Few_shot_Node_Classification_on_Attributed_Networks_with_Graph_Meta_learning.html
tags: [reviews]
---

# **Few-shot Node Classification on Attributed Networks with Graph Meta-learning**  


## **1. Introduction**  

### 1-1. Preliminaries  
 오늘 소개하고자 하는 논문의 제목에서 키워드를 뽑아보자면, `Few-shot`, `Attributed Networks`, `Meta-learning`정도로 추릴 수 있다. 결국 Graph 같은 ***Attributed Networks***(e.g. molecular graph, social network)***에 대해서 Few-shot learning task를 Meta-learning의 방식으로 해결***하고자 한다. 본격적으로 논문을 파헤치기 전에, 논문을 이해하는데 필수적으로 필요한 Few-shot learning과 Meta-learning의 개념에 대해 간략하게 설명하도록 하겠다. 두 개념에 대한 설명은 [Graph Meta Learning via Local Subgraphs (NIPS20)에 대한 리뷰포스트](https://dsail.gitbook.io/isyse-review/paper-review/2022-spring-paper-review/neurips-2020-g-meta)를 참조하였다.  
 
#### 1-1-1. Few-shot Learning   

Few-shot Learning은 적은 데이터를 가지고 효율적으로 학습하는 문제를 해결하기 위한 학습 방법이다. 

<div align="center">
  
![image](https://user-images.githubusercontent.com/37684658/164231019-868292bd-9cbf-4d15-87cb-24d621ed78d6.png)
  
</div>

예를 들어, 위와 같이 사람에게 아르마딜로(Armadillo)와 천갑산(Pangolin)의 사진을 각각 2장씩 보여줬다고 생각해보자. 아마 대부분의 사람들은 아르마딜로와 천갑산이 생소할 것이다. 자, 이제 그 사람에게 다음의 사진을 한 장 더 보여주었다.  

<div align="center">
  
![image](https://user-images.githubusercontent.com/37684658/164224487-822f266a-98db-4d2d-9c41-7303fdccf1ff.png)  

</div>


위 사진의 동물이 아르마딜로인지, 천갑산인지 맞춰보라고 하면, 너무나 쉽게 천갑산임을 자신있게 외칠 수 있을 것이다. 사람들은 어떻게 이렇게 적은 양의 사진을 보고도, 두 동물을 구분할 수 있는 능력을 가지게 되었을까? 사람과는 달리 기존 머신러닝(Machine Learning)은 저 두 동물을 구분하기 위해 많은 양의 사진을 보고 학습하여야 할 것이다. 만약 모델이 아르마딜로와 천갑산을 잘 구분할 수 있게 되었다고 하자. 이제 갑자기 아래 두 동물을 구분하라고 하면 어떻게 될까?  

<br/> 

<div align="center">
  
![image](https://user-images.githubusercontent.com/37684658/164231266-515ab539-110b-4835-971c-287fb759c44a.png)

</div>


두더지(Mole)는 모델이 처음 보는 동물이기 때문에 두 동물을 구분하려면 다시 두더지에 대한 사진을 학습을 해야할 것이다. 하지만 사람은 여전히 두 동물을 쉽게 구분할 수 있다. 사람과 같이 적은 양의 사진만 보고도 Class를 구분할 수 있는 능력을 학습하는 것이 Few-shot Learning이고, 이를 학습하기 위해 Meta-Learning의 학습 방법을 활용한다.  

Few-shot Learning에서 쓰이는 용어를 정리하고 넘어가면, 처음 모델에게 제시해주는 Class별 대표사진들을 `Support Set`이라고 한다. 2개의 Class로 구성되어 있다면 2-way라고 하며, Class별로 2장의 대표사진을 보여준다면 2-shot이라고 한다. 그리고 1장의 새로운 사진을 보여주는 데 이렇게 맞춰보라고 보여주는 사진들을 `Query Set`이라고 하며, 1번 맞춰보라고 주었으니 Query는 1개이다. Support Set과 Query Set을 합쳐서 하나의 `Task` 또는 `Episode`라고 지칭한다.  

#### 1-1-2. Meta Learning  

메타러닝(Meta Learning)은 새로운 task에 대한 데이터가 부족할 때, Prior Experiences 또는 Inductive Biases를 바탕으로 빠르게 새로운 task에 대하여 적응하도록 학습하는 방법을 말한다. 'Learning to Learn'이라는 용어로 많이 설명되곤 하는 데, 대표적인 접근 방법으로는 거리 기반 학습(Metric Based Learning), 모델 기반 학습 (Model-Based Approach), 그리고 최적화 학습 방식(Optimizer Learning)이 있다. 이 중, 오늘 소개하고자하는 Meta-GPS 논문을 제대로 이해하기 위해서는 최적화 학습 방식의 MAML[3]에 대한 이해가 선행되어야 한다.  

##### MAML (Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks)  
MAML은 최적화 학습 방식의 Meta Learning 방법론으로서 가장 대표적인 논문이라고 할 수 있다. 전체적인 개념은 어떤 Task에도 빠르게 적응(Fast Adaptation)할 수 있는 파라미터를 찾는 것이 이 모델의 목적이다. 일반적으로 딥러닝 모델은 기울기의 역전파를 통해 학습을 진행하나, 이런 학습 방법은 데이터가 충분히 많을 때 잘 작동하는 방식이다. 이 모델은 Task 하나하나에 대한 그래디언트를 업데이트 하는 inner loop와, 각 태스크에서 계산했던 그래디언트를 합산하여 모델을 업데이트하는 outer loop 단계로 이루어져있다(아래 그림에서의 실선). 공통 파라미터 ![image](https://user-images.githubusercontent.com/37684658/164229776-12b52e66-cf43-4b8e-ba97-ee0ccb723724.png)는 Task agnostic하게 빠르게 적응할 수 있는 파라미터이고, 다시 모델이 이 파라미터로부터 어떤 Task를 학습하게 되면 그 Task에 최적화된 파라미터를 빠르게 찾을 수 있게 된다.  

<div align="center">
  
![image](https://user-images.githubusercontent.com/37684658/164233736-dd00ab2f-adf4-42b9-a491-6def82a126d4.png)

</div>

### 1-2 Problem Definition  
Citation networks, social media networks, traffic networks와 같은 attribute networks는 실생활에서 발생할 수 있는 수많은 문제들을 풀기 위해 활용되고 있다. 이 중   

## **2. Motivation**  

## **3. Method**  

## **4. Experiment**  

## **5. Conclusion**  
