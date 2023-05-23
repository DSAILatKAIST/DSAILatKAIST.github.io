---
title:  "[ICLR 2022] How to Train Your MAML to Excel in Few-Shot Classification"
permalink: How_to_Train_Your_MAML_to_Excel_in_Few_Shot_Classification.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **How to Train Your MAML to Excel in Few-Shot Classification** 

이 리뷰에서 소개하는 논문, "How to Train Your MAML to Excel in Few-Shot Classification"은 Meta-learning에 대해 소개한다면 반드시 언급되는 Methodology인 Model-Agnostic Meta-Learning(MAML)[2]을 실험적으로 분석하고, 새로운 모델 Unicorn-MAML을 제시한 논문입니다. 참고로 몇몇 용어의 번역은 임의로 작성하였음을 알려드립니다.  

<br/>

## **1. Introduction**  

인공지능 모델이 처음 보는 Class(Novel Class)를 단 몇 개의 Sample만이 주어져있더라도 분류할 수 있도록 하는 것을 목표로 하는 퓨샷 학습(Few-shot learning)은 근 몇 년 간 메타 학습(Meta-learning)을 이용한 전략이 주목받아 왔습니다.  
메타 학습은 퓨샷 학습 문제를, 테스트 단계의 환경을 모방하여 mini-batch를 구성함으로써 모델을 학습시켜 해결하려 하며, 이것을 에피소드 학습(Episodic learning)[1]이라 부릅니다.  
이렇게 메타 학습이 퓨샷 학습 문제를 해결하는 전략으로 대두된 이래, 2017년도 ICML을 통해 발표된 Model-Agnostic Meta Learning(MAML)[2]은 그 단순성과 효율성 때문에 가장 주목받았던 Baseline중 하나이며 수많은 Variant를 파생시켰습니다. 하지만 이후 발전된 모델이 수없이 등장하면서 더이상 뛰어넘어야하는 모델로 여겨지지는 않았습니다.

이 논문에서는 실증적 분석을 통해 MAML의 성능을 Improve할 수 있는 실험적 발견과 함께 단순한 수정을 통해 경쟁력있는 모델, Unicorn-MAML을 제시합니다. 이 논문의 Contribution은 아래와 같이 정리할 수 있습니다.
1. 실험을 통해 MAML이 Inner-loop에서 기존에 널리 사용되는 값보다 더 많은 값의 Gradient step이 필요함을 조명합니다.
2. MAML이 Testing phase에서 Class 별 label 배정이 어떻게 이뤄지는지(Permutation of class label assignments)에 민감하다는 것을 보여주며, 이를 기반으로 MAML을 Testing phase에서 Permutation-invariant하게 만들 수 있는 여러 가지 방법과 그 효율성을 살펴봅니다.
3. 위의 발견과 실험을 기반으로, 경쟁력있는 모델인 Unicorn-MAML을 제시합니다.

이 논문에서 지적하는 가장 중요한 포인트인 2. Testing phase에서의 Class별 label 배정의 Permutation에 따른 문제는 아래 Figure을 통해 확인해볼 수 있습니다.

<p align="center"><img width="600" src="https://github.com/JhngJng/JhngJng/assets/63398447/2dd5f6de-3262-4719-8968-9b100c484be1"></p>

위 이미지를 보면 학습된 Classification head (Linear Classifier)를 새로운 Task에 Fine-tuning할 때, 기존의 MAML은 각 class에 해당하는 head가 다르게 initialize되었기 때문에 Testing Task의 label assignment가 어떻게 되냐에 따라 각 Permutation 별 성능에 큰 차이가 있을 수 있고 이러한 성능 편차는 논문/리뷰의 Section 4에서 확인할 수 있습니다. 그렇기 때문에 저자들은 Unicorn-MAML을 제시하여 이 문제를 해결하고자 합니다.

몇 가지 생소할 수 있는 용어나 개념, 모델에 대한 설명은 다음 Section의 Preliminary에서 하도록 하겠습니다.

<br/>

## **2. Preliminaries**  

이 Section에서는 논문을 이해하는데에 필요한 에피소드 학습, MAML 등을 비롯한 개념에 대해 소개합니다.

### **2.1. Problem Definition**

#### 2.1.1. 퓨샷 학습(Few-shot Learning)

퓨샷 학습(Few-shot Learning)은 전술했듯이 모델이 경험하지 못한 Class의 데이터도 적은 수의 Training sample만 가지고도 분류할 수 있도록, 기존에 갖고있는 데이터로 모델을 학습하는 것을 말합니다.

이 때, 학습에 이용하는 Class를 Base class($C_ b$), 분류하고자 하는 Class를 Target/Novel class($C_ t$)라고 흔히 부릅니다. 이 리뷰에서는 논문에서 언급한대로, Novel class라고 하겠습니다. Base class는 일반적으로 충분히 많은 숫자의 labeled 데이터가 있다고 가정하며, Novel class는 testing-phase까지는 unknown입니다.

Formal하게는 아래와 같이 정의 가능합니다.

> Definition(_Few-shot learning_): Given labeled data of ($X_ {C_ b}$, $Y_ {C_ b}$ ), few-shot learning is problem of making prediction of $x_ q$ $\in$ $X_ {C_ t}$ (i.e. query set samples) based on few-labeled samples ($x_ s$, $y_ s$) $\in$ ($X_ {C_ t}$, $Y_ {C_ t}$ ) (i.e. support set samples) with model $f_ \theta$ trained with ($X_ {C_ b}$, $Y_ {C_ b}$ ).

관습[1]을 따라서, 우리는 Few-shot 분류 문제 $\mathcal{T}$ 를 $N$-way $K$-shot Task로 정의하며, 이때 $N$은 주어진 Few-shot 분류 문제에서 주어진 Novel class의 수, $K$는 각 class 별로 주어진 labeled sample의 숫자입니다. 이 때 주어진 $N \times K$개의 labeled sample들을 묶어 Support set이라 부르며 편리상 $\mathcal{S}$로 쓰고, 맞춰야 하는 Novel class의 $N \times Q$개의 sample들을 묶어 Query set이라 부르며 $\mathcal{Q}$로 씁니다.

#### 2.1.2. 에피소드 학습(Episodic Learning)

에피소드 학습(Episodic Learning)은 MatchingNet[1]에서 처음 제시된 개념으로, Few-shot learning의 Downstream task의 형태를 모방하여 학습에 쓰이는 mini-batch를 구성하는 방식입니다. mini-batch 대신에, Base class의 sample들로 Downstream task와 똑같은 형태인 $N$-way $K$-shot Task로 mini-batch를 구성하여 모델을 학습시킵니다. 그리고 이렇게 구성한 mini-batch 하나를 Episode라고 부릅니다.

이러한 Episode 구성 방식 때문에, Training/Testing 두 phase 모두 뽑힌 $N$개의 class 각각에 대해 $1,\cdots,N$ 중 하나의 값으로 임의의 label이 랜덤하게 배정되게 됩니다. 이러한 무작위성은 이전의 연구[3]를 통해 MAML의 over-fitting을 방지하는데 큰 역할을 한다고 알려져 있습니다. 하지만 Testing-phase에서는 이 label 무작위 배정이 성능에 큰 영향을 주게 된다는 것이 이 논문에서 주장하고 보여주는 바 입니다.


### **2.2. About Model-Agnostic Meta Learning(MAML)**

MAML은 Few-shot 분류 문제를 해결하는 $N$-way Classifier의 initialization을 학습하기 위한 방법론입니다. MAML을 통해 initialization을 학습한 뒤, Downstream task의 support set을 이용해 Classifier을 튜닝해준 뒤 Query sample에 대해 prediction을 산출합니다. 전체적으로는 어떤 Task에 대해서도 빠르게 적응할 수 있는 Parameter initialization을 찾아 모델에 제공하는 것이 목적이라고 할 수 있습니다.

MAML의 파라미터 update는 2단계로 구성되어 있는데, 먼저 Inner-loop라고 부르는 단계에서는 support set sample들을 이용해 task-specific한 파라미터를 찾고, 이렇게 찾은 task 별 파라미터를 바탕으로 query set sample들을 이용해 loss를 계산, 이를 모아 Outer-loop 단계에서 파라미터를 최종적으로 update하여 좋은 파라미터 초기 값을 찾을 수 있도록 합니다.

Inner-loop는 formal하게 아래와 같이 나타낼 수 있습니다.

<p align="center"><img width="200" src="https://user-images.githubusercontent.com/63398447/232225605-36362eef-1fb4-409f-85ce-af3a9bdbf036.png"></p>

이 때 $\theta$는 Feature extractor $f_ {\phi}$의 파라미터 $\phi$ 와 $N$-way Classification head의 파라미터 $\mathbf{w}_ 1 ,\cdots \mathbf{w}_ N$이고, $M$은 Support set $\mathcal{S}$를 이용해 Inner-loop update를 수행하는 gradient step의 숫자, $\alpha$는 (inner) learning rate입니다. 대체로 $M$은 5 이하 값을 쓰는 것으로 알려져왔으나, 이 논문에서는 이 점을 지적하며 더 큰 값(10 이상)으로 가져가는 것이 좋다고 보여줍니다.

Outer-loop는 Inner-loop에서 찾은 각각의 Task-specific한 파라미터 $\theta$로 각각에 해당하는 Query set으로 Loss를 계산, 각각의 Gradient의 평균값으로 최종 Gradient를 계산하여 파라미터를 update하게 됩니다. 

Training Task의 distribution을 $p(\mathcal{T})$라 하고, Inner-loop에서 특정한 task $\mathcal{T}$에 대해 찾은 Task-specific 파라미터를 $\theta_ {\mathcal{T}}$라 하면 Outer-loop update는 아래와 같이 나타낼 수 있습니다.

> $\theta \leftarrow \theta - \beta \nabla_ {\theta} \sum_ {\mathcal{T} \sim p(\mathcal{T})} \mathcal{L}_ {\mathcal{T}} (\mathcal{Q_ {\mathcal{T}} }, \theta_ {\mathcal{T}})$

이 때 $\mathcal{Q_ {\mathcal{T}} }$는 task $\mathcal{T}$의 query set입니다.


### **2.3. Experimental Setup**

본문에서도 언급하듯이, 이 논문은 실험 위주의 논문이기 때문에 주요한 실험 세팅을 이 Section에서 소개합니다.

* **Dataset**  
이전에는 다양한 언어의 문자 이미지인 Omniglot dataset이 주요 벤치마크 dataset 중 하나였으나, 모델들이 성능 측면에서 급속도로 성장하면서 현재는 **MiniImageNet**과 **TieredImageNet**, 그리고 **CUB** dataset 이 세 가지가 주로 쓰입니다.  
MiniImageNet은 100개의 class에 각각 600개의 이미지가 있는 데이터셋으로, Training/Validation/Test 에 쓰이는 class 수는 각 64/16/20개 입니다. 다시 말해, 64개의 Base class와 target으로 하는 20개의 class가 있습니다.  
TieredImageNet은 608개의 class가 있고, Class split은 Training/Validation/Test 각 357/97/160개 입니다.  
CUB dataset은 200개의 class가 있고 split은 100/50/50개로 나눠집니다.

* **Training & Evaluation**  
Episodic learning framework을 따라, meta-training/validation/testing phase 각각 주어진 class와 sample들로 $N$-way $K$-shot task들을 sample하게 됩니다. 이 논문에서는 $N=5$, $K=1,5$인 경우를 다룹니다. 각 Task마다 query set은 class별 15개의 sample로 구성됩니다.  
이전에 언급했던 것처럼, 모든 sample된 task들은 랜덤하게 각각 $1,\cdots,N$ 중 하나의 값으로 label이 재배정됩니다.  
Testing-phase에서는 sample된 10,000개 task로 모델의 성능을 평가하게 됩니다.  

* **Model Architecture**  
최근 연구들을 따라, Feature extractor $f_ \phi$로는 ResNet-12를 사용하되, 기존 MAML이나 이전 연구[1]와 같은 4-layer ConvNet또한 같이 사용합니다.

이외의 자세한 세팅은 논문 본문을 참고 부탁드립니다.

<br/>

## **3. MAML requires Larger Number of Gradient Steps in Inner-loop**  

MAML이 발표될 당시, MAML은 'Fast-adaptation'을 강점으로 내세우면서, 적은 수($M\approx 1$)의 inner-loop step/fine-tuning step으로도 좋은 성능을 보일 수 있다 주장했었습니다. 그것이 지배적인 생각이 되어, MAML을 개선한 논문 MAML++[4]에서도 $M$값은 5 정도로 선택되었습니다.

이 논문은 이렇게 알려진 부분을 정면으로 반박하는 실험 결과를 제시합니다. 아래 Figure 2를 보면, Backbone이나 Dataset에 상관없이 15이상의 큰 $M$값에서 더 좋아진 성능을 보여줌을 확인할 수 있고, 후술될 Main table에 제시된 최신 baseline들과 비교했을 때도 꽤 경쟁력있는 성능입니다.

<p align="center"><img width="600" src="https://user-images.githubusercontent.com/63398447/232234960-e1f36ecc-b206-4d95-adfd-c8b8c6b28d07.png"></p>

이러한 결과를 이 논문에서는 학습된 초기 파라미터로, 따로 sample한 meta-training task와 meta-testing task 모두에 adapt한 실험 결과를 보여줌으로써 설명합니다.

<p align="center"><img width="600" src="https://user-images.githubusercontent.com/63398447/232235288-97e15b56-f1b8-462b-907f-daea8281d9fa.png"></p>

Figure 3을 보면, 아래 두 가지 주요한 결과를 확인할 수 있습니다. 이를 통해 저자들은 충분히 좋은 성능을 얻기 위해서는 initialized 모델이라도 충분한 수의 update가 필요하다고 주장합니다.
> 1) _Training/Testing task에 관계 없이 inner-loop update 수가 늘어날수록 accuracy가 증가한다_
> 2) _Inner-loop update 이전엔, training/testing task 관계 없이 chance level-100/N% 정도의 정확도, random prediction 수준-의 성능을 보인다._

두 번째 결과의 경우, 저자들은 few-shot task를 만들 때 각 class별로 random한 label 배정이 이유라고 해석합니다. 같은 class(개, 고양이, 새, 사자, 호랑이)를 공유하는 task라도, 새로이 배정되는 label의 permutation에 따라('개' class가 label 1을 가질 수도 있고, 5를 가질 수도 있음) task가 다른 task가 되기에 inner-loop update 없이는 random prediction이라는 설명입니다. 


<br/>

## **4. MAML is Sensitive to Label Permutations during Testing-phase**  

이 Section에서는 MAML이 "Meta-testing phase"에서 앞서 살펴보았던 label permutation에 굉장히 민감하다는 것을 보여주면서, Testing-phase에서 MAML을 permutation-invariant하게 만들어야한다고 주장합니다.  
논문 본문에도 거듭 언급되지만, 이전 연구를 통해 MAML의 meta-training 과정에서 task에 부여되는 random label은 과적합을 방지하여 training에 도움이 된다고 하며, 이 논문에서도 주장하는 것은 Testing-phase에서의 label permutation이 영향을 미친다는 것임을 다시 강조하도록 하겠습니다.

### **4.1. Experimental Results showing Permutation-sensitivity of MAML in Testing-phase**

학습된 classification head $\mathbf{w}_ 1, \cdots, \mathbf{w}_ N$이 각기 다른 Class에 배정되었을 때의 성능 추이를 알아보기 위해, 아래 Algorithm 1에 정리된 실험을 이 논문에서는 수행했습니다.

<p align="center"><img width="600" src="https://user-images.githubusercontent.com/63398447/232243439-3da7fa9e-ea70-4073-af37-23ba16ffc185.png"></p>

2,000개의 meta-testing task를 뽑은 이후, 각 Task별로 120개의 각기 다른 label permutation에 대해 성능을 구해준 뒤, 이를 내림차순으로 정렬한 뒤 task-level로 averaging을 해주는 알고리즘입니다.  

아래 Figure 4를 보면, permutation에 따른 성능 variance가 굉장히 큰 것을 확인할 수 있습니다.(~15%/~8% gap for 1/5 shot tasks)

<p align="center"><img width="600" src="https://user-images.githubusercontent.com/63398447/232243595-ea760682-7cc6-42a5-b4d3-60afb6775e68.png"></p>

이 실험 결과를 바탕으로, 논문에서는 test query sample 없이 좋은 permutation을 찾거나, MAML을 permutation-invariant하게 만든다면 MAML을 기술적으로 향상시킬 수 있을 것이라 주장하며, 그렇게 논문에서 제시한 여러 방법론을 다음 Section에서 소개하겠습니다.


### **4.2. Explored method to make MAML Permutation-invariant**

이 논문에서는 아래 3가지 간단한 방법론의 가능성에 대해 조사합니다. 직접적으로 permutation-invariant하게 만들고자 하는 방법론은 2, 3번 방법이 되겠습니다.
> 1) 각 Task 별 Best permutation 탐색 (with Support set samples)
> 2) Permutation 별 산출한 prediction을 ensemble
> 3) 학습한 파라미터 초기값 $\mathbf{w}_ 1, \cdots, \mathbf{w}_ N$을 averaging


1번 방법론은 Testing task의 Support set sample을 갖고, Accuracy/Loss value 등을 통해 best permutation을 탐색합니다. 하지만 아래 Table 1을 보면 그 어떤 방법론도 일관된 효과를 보이지 못했음을 확인할 수 있습니다.

<p align="center"><img width="500" src="https://user-images.githubusercontent.com/63398447/232245990-e27255ed-91a8-4b8b-b8b5-93e48dc42237.png"></p>

이에 대해 두 가지 분석을 내놓습니다. 첫번째는 initial support accuracy/loss를 활용하는 경우는 inner-loop update 이전에 MAML로 학습한 파라미터 초기값이 chance-level prediction만을 내놓기 때문이고, 두번째로는 support set accuracy가 굉장히 빠르게 100%에 도달하여 informative하지 못하기 때문이라고 분석합니다.

그렇기에 이 논문에서는 best permutation을 찾는 것이 아닌, 각 permutation의 ensemble(2번 방법)을 시도합니다. 이때, label이 아닌 classification head $\mathbf{w}_ 1, \cdots, \mathbf{w}_ N$를 permute 하는데, 이는 label을 permute하지 않고 이렇게 하는 것이 prediction을 합치기 편리하기 때문입니다.(둘은 Equivalent)
다만 이 방법론은 permutation-invariant를 충족시키게 해주나 수많은 permutation($N=5$일 때도 무려 120가지입니다.)가짓수 만큼 prediction을 계산해줘야 하기 때문에 cost가 큰 편입니다. 따라서 단순히 Rotate함으로써 $N$가지 permutation만 고려하여 ensemble하는 'Rotated' 방법론도 아래 Table 2에서 결과를 확인할 수 있습니다.

<p align="center"><img width="400" src="https://user-images.githubusercontent.com/63398447/232246040-08400f96-7be4-4eee-b80b-7e5189a404c8.png"></p>

확실히 모든 세팅에서 성능 gain이 있습니다. 또한 단순 Rotation permutation만 고려해도 충분히 좋은 성능을 보입니다. 다만 Rotation만 고려한다고 해도, $N$배나 prediction을 계산해주어야 하고, 이는 분명한 단점입니다.

마지막 3번 방법론은, 각 class $c$에 할당되는 classification head $\mathbf{w}_ c$를 모두, 학습된 initialization $\mathbf{w}_ 1, \cdots, \mathbf{w}_ N$의 평균값 $\sum \mathbf{w}_ i$로 다시 초기화 해주는 방법입니다. 이는 MAML의 permutation sensitivity가 결국 각기 다른 class에 대해 각기 다른 classification head 파라미터가 assign되기 때문에 기인한 것이기 때문입니다.

<p align="center"><img width="300" src="https://user-images.githubusercontent.com/63398447/232246055-bc0a1795-c11e-40b9-85c0-a29b54ad6f61.png"></p>

이렇게 해줄 경우, 별다른 계산량 증가 없이 4 세팅 중 3 세팅에서 성능 gain을 보여줍니다.


<br/>

## **5. Proposed: Unicorn-MAML**
  
앞의 Section의 마지막 방법론을 기반으로, 이 논문에서는 아래의 Research Question을 제기합니다.

> Classification Head $\mathbf{w}_ 1, \cdots, \mathbf{w}_ N$를 initialize 하기 위해 meta-training 과정에서 단 하나의 vector $\mathbf{w}$를 학습하여 meta-training/testing phase 두 phase를 일관되게, permutation-invariant 하게 만든다면 MAML의 성능을 더 끌어올릴 수 있을까?

정리하자면, MAML을 통해 학습하는 파라미터 $\theta$는 $\phi, \mathbf{w}$로 줄어들게 됩니다. 이렇게 정의된 Unicorn-MAML의 Inner-loop와 Outer-loop는 아래와 같이 수행합니다.

1) **Inner-loop optimization**  
Shared head $\mathbf{w}$를 복제하여 각 class $c$에 $\mathbf{w}_ c = \mathbf{w}$가 되도록 한 뒤, MAML과 똑같이 inner-loop optimization을 진행합니다.

2) **Outer-loop optimization**  
Update한 Classification head parameter $\theta'$을 $\phi', \mathbf{w'}_ 1, \cdots, \mathbf{w'}_ N$라 하면, 각 $\mathbf{w}_ c$에 대해 gradient를 계산해 그 average 값인 $\sum_ {c=1,\cdots N} \nabla_ {\mathbf{w}_ c} \mathcal{L}(\mathcal{Q}, \theta')$으로 task 별 gradient를 계산해줍니다. 나머지는 MAML과 같습니다.

아래 Table 4를 통해, 우리는 Unicorn-MAML이 여타 State-of-the-art baseline과 비교했을 때 경쟁력 있으며, 5shot에서는 SOTA 성능을 달성함을 확인할 수 있습니다.

<p align="center"><img width="600" src="https://user-images.githubusercontent.com/63398447/232250170-0ebb72ff-e2b4-413e-8763-7c05835ac38c.png"></p>


<br/>

## **6. Conclusion**  

이 논문은 광범위한 실험을 통해 MAML의 개선할 수 있는 부분(More gradient steps in inner-loop, Permutation-sensitivity in testing-phase)을 조명하며 이를 기반으로 자신들의 새로운 모델, Unicorn-MAML을 제시한 연구입니다. 문제를 발견하고, 이를 해결하기 위해 다양한 방법들을 시도하여 이를 바탕으로 자신들만의 Unique한 해결책을 제시한 점은 높게 평가받을 요소라고 생각합니다.

이 논문을 통해, 좋다고 알려진 것에 항상 의구심을 갖고 도전하는 자세가 중요하다는 것을 느낄 수 있었습니다. 또한 어떠한 문제를 해결하기 위해 다양한 방법들을 시도하며 이를 소개함으로써 자신들이 제시하고자 하는 것을 뒷받침하는 부분도 연구에서 중요한 부분이라는 교훈을 주는, 이것저것 배워갈 것이 많은 논문이라고 할 수 있겠습니다.

---  
### **Review Writer Information**  

* 정지형 (Jihyeong Jung)  
    * Master Student, Department of Industrial & Systems Engineering, KAIST  
    * Few-shot learning with Meta-learning, Graph Neural Networks, etc.

<br/>

## **7. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* [Github Implementation(Official)](https://github.com/Han-Jia/UNICORN-MAML)
* [논문 본문](https://arxiv.org/pdf/2106.16245.pdf)
1. Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. 2016. _Matching networks for one shot learning_. Advances in neural information processing systems, 29.
2. Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. _Model-agnostic meta-learning for fast adaptation of deep networks_. In International conference on machine learning. PMLR, 1126–1135.
3. Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, and Chelsea Finn. 2020. _Meta-learning without memorization_. In ICLR.
4. Antreas Antoniou, Harrison Edwards, and Amos J. Storkey. 2019. _How to train your MAML_. In ICLR.
