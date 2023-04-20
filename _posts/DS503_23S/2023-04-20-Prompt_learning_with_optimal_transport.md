---
title:  "[ICLR 2023] PLOT: PROMPT LEARNING WITH OPTIMAL TRANSPORT FOR VISION-LANGUAGE MODELS"
permalink: Prompt_learning_with_optimal_transport.html
tags: [reviews]
---

# **PLOT: PROMPT LEARNING WITH OPTIMAL TRANSPORT FOR VISION-LANGUAGE MODELS **

## **1. Problem Definition**

CLIP[1], ALIGN[2] 과 같은 Large-scaled vision-language pretrained model (VLP)이 발전하면서 noisy하지만 많은 양의 데이터셋만 모을 수 있다면 충분히 model의 generalization 성능을 올릴 수 있다라는 것이 밝혀졌습니다. 이를 down-stream task에 적용하기 위해선 model을 finetuning을 해야하지만 다음과 같은 문제가 발생하였습니다. 

1. 모델의 모든 Parameter에 대해 Full-finetuning을 하는 것은 complex하기에 현실에서의 Scalability가 문제가 됩니다.
2. 그리고, full-finetuning은 기존 model의  generalization 능력을 저하시킵니다. 기존에 , 모델들이 domain generalization과 같은 task에 대해 좋은 robustness를 갖고 있던 반면 naive한 full-finetuning은 in-distribution에 대해서는 잘 하지만, 이후 outofdistribution에 대해서는 성능이 매우 떨어지는 현상이 발생합니다. 

그래서 이를 해결하기 위해 parameter-efficient하면서 기존 모델 성질을 해치지 않는 fine-tuning 기법으로 Prompt Learning이라는 연구가 발생하였습니다.

> VLP에서 Prompt learning이란 vision&language encoder를 freezing한 상황에서 language encoder에 들어가는 context vector를 parameterized하여 이를 학습하는 데 이용하는 것을 의미합니다. 이 때 학습은 image feature와 text feature 간의 alignment를 maximize하는 방향으로 학습이 진행됩니다. 



<img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416153017754.png" alt="image-20230416153017754" style="zoom:50%;" />







## **2. Motivation**

<img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416154336824.png" alt="image-20230416154336824" style="zoom:50%;" />

Figure 1의 예시처럼 이미지 하나에는 사실 여러 context가 존재할 수 있습니다. 그렇다면 이를 위해서 prompt 수를 늘리는 것이 하나의 방법이 될 것입니다. 단순히, 각각의 prompt에 대해서 cross entropy 즉 alignment score를 늘리는 방향으로 학습하게 된다면 어떻게 될까요. Loss가 convex하다고 가정한다면 모든 prompts는 initialized에 관계 없이 하나의 점으로 Collapse되며 이는 결국 하나의 prompt를 사용하는 것과 같은 의미를 지니게 될 것입니다.

![image-20230416154636082](/Users/aailab/Library/Application Support/typora-user-images/image-20230416154636082.png)

각각의 prompt가 다른 의미를 지니게 하기 위해선 직접적으로 서로가 멀어지도록 regularization을 걸어주는 방법도 있지만 이 논문에서는 좀 더 sematic 관점에서 새로운 방법을 제시합니다.





## **3. Method**

각 Prompt가 하나의 image feature를 모두 설명하도록 하는 것이 아니라 image feature의 각각의 locality와 의 pair를 상정하고 그에 따른 distance를 정의함으로 여러 prompt가 다른 semantic을 가지도록 하는 것입니다.

이를 위해선 1. 어떻게 pair를 정할지? 2. distance를 어떻게 정의할지를 위 논문에서 Optimal transport 의 관점에서 제시합니다.

![image-20230416162611136](/Users/aailab/Library/Application Support/typora-user-images/image-20230416162611136.png)

____

Optimal transport는 흔히 distribution간의 거리를 정의할 때 사용됩니다. 즉, 현재 상황에서 vision feature와 language feature간의 거리를 정의하는 용도로 사용합니다.  즉, 두 거리를 정의하기 위해 일단 두 feature에 대한 distribution을 dirac measure $\delta$ 로 정의합니다.

여기서 U는 vision feature에 대한 distribution을 의미하며 m은 vision feature 에 대한 local feature를 의미하며 V는 Several prompts에 대한 distribution을 의미합니다. 이 때 n은 각 prompt에 대한 index입니다.



![image-20230416170151382](/Users/aailab/Library/Application Support/typora-user-images/image-20230416170151382.png)

Distribution을 정의했으니 우리는 Optimal transport에서의 distance 역시 정의할 수 있게 됩니다. 이 때 distance는 다음과 같이 정의할 수 있습니다.  (3)에서 ![image-20230416171456125](/Users/aailab/Library/Application Support/typora-user-images/image-20230416171456125.png)는 local image feature m 와 promp n 간의 disimilarity로 정의하였고 ![image-20230416171636471](/Users/aailab/Library/Application Support/typora-user-images/image-20230416171636471.png)는 각 pair간의 transport plan을 의미합니다. 두 distribution의 Distance를 정의 하기 위해 (3),(4)의 equation을 정의합니다.

![image-20230416171153075](/Users/aailab/Library/Application Support/typora-user-images/image-20230416171153075.png)

![image-20230416171227179](/Users/aailab/Library/Application Support/typora-user-images/image-20230416171227179.png)

(4)를 optimize하게 되면 두 distribution에 대한 거리를 정의할 수 있지만 적어도 MN개의 변수를 처리해야하기에 이를 해결하는 것은 생각보다 complex합니다. 그래서 이를 해결하기 위해 Sinkhorn algorithm을 이용해 새로운 optimization 식을 정의합니다. 

![image-20230416192111610](/Users/aailab/Library/Application Support/typora-user-images/image-20230416192111610.png)

이 때 h term은 Transport plan에 대한 Entropy로 정의하며 entropy 를 고려한 problem에서 다음과 같은 해를 찾을 수 있게 됩니다. 

![image-20230416192242602](/Users/aailab/Library/Application Support/typora-user-images/image-20230416192242602.png)



이 때 t는 optimization에서의 iteration step을 의미하며 <img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416192351406.png" alt="image-20230416192351406" style="zoom:33%;" /> 와 <img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416192416617.png" alt="image-20230416192416617" style="zoom:33%;" />
<img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416192443134.png" alt="image-20230416192443134" style="zoom:33%;" />로 iteration이 돌아갑니다. 



이렇게 Transport Plan을 정의하면 이후에 Vision~Language feature 간 distance를 정의할 수 있게 되고 이를 minimize하는 방향으로 Prompt를 학습합니다. 그 때의 outer loop에 대한 optimization 식은 다음과 같습니다. 



<img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416192811535.png" alt="image-20230416192811535" />![image-20230416192825141](/Users/aailab/Library/Application Support/typora-user-images/image-20230416192825141.png)

![image-20230416192927177](/Users/aailab/Library/Application Support/typora-user-images/image-20230416192927177.png)

Inner loop에서 (7)의 Distance를 정의하고 이후 distance를 이용한 output function을 정의하여 이를 eq (9)라는 objective function의 꼴로 정의 하여 이를 minimize하는 방향으로 Prompt learning이 진행됩니다. 



## **4. Experiment**

크게 이 논문에서 중점적으로 다룬 실험은 두 가지라고 볼 수 있습니다.

+ Few-shot Classification.
+ Domain generalization in ImageNet.



첫 번째 실험은 Downstream task를 얼마나 잘 수행하고 있는지를 평가하는 항목이고 두 번째 실험은 Domain shift에 robust한 지를 평가하여 기존 VLP 모델의 generalization 성질을 잘 보존하고 있는지를 평가하는 항목입니다. 

### **Experiment1 setup: Few-shot Classification**

- Dataset

  - Caltech101
  - EuroSAT
  - DTD
  - FGVC Aircraft
  - Oxford pets
  - Oxford flowers
  - Food101
  - ImageNet
  - Stanford cars
  - UCF101
  - SUN397

- baseline

  - COOP [3]

    COOP는 하나의 Prompt parameter를 상정합니다. prompt로 만들어진 language feature와 image feature간의 Distance를 minimize 하는 방향으로 prompt parameter를 학습합니다. 

    <img src="/Users/aailab/Library/Application Support/typora-user-images/image-20230416153017754.png" alt="image-20230416153017754" style="zoom:50%;" />

    

  - COCOOP [4]

    COCOOP는 각 image의 context가 다를 수 있음을 상정합니다. 이를 위해 하나의 prompt 에다가 Meta-network라는 image feature를 받아 prompt parameter로 mapping하는 function을 이용해 knowledge distillation을 진행합니다. 

    

    ![image-20230416194126561](/Users/aailab/Library/Application Support/typora-user-images/image-20230416194126561.png)

- Evaluation Metric

  - Accuracy


### **Result of Exp1**

평균적으로 Ours(PLOT)가 다른 baseline보다 잘하는 것처럼 보이지만 T-TEST에 따르면 그리 significant하진 않은 것으로 판단됩니다. (t>0.05). 이 논문이 ICLR Spotlight를 받았지만 그 이유는 성능보다는 좀 더 Idea의 Novelty가 강한 것으로 생각됩니다. 4개의 Prompt를 사용했지만 1개의 Prompt를 사용한 COOP보다 그리 높지 않은 점은 이 논문의 단점이라고 생각합니다. 



![image-20230416193721212](/Users/aailab/Library/Application Support/typora-user-images/image-20230416193721212.png)



![image-20230416193749924](/Users/aailab/Library/Application Support/typora-user-images/image-20230416193749924.png)



### **Experiment2 setup: Domain Generalization**

Dataset

- ImageNet: Source distribution

- ImageNetV2: Target distribution

- ImageNet-R: Target distribution

- ImageNet-A: Target distribution

- ImageNet-Sketch: Target distribution

  

### Result of EXP2

Prompt parameter를 4배를 더 사용함에도 그리 큰 Gain을 얻지 못하는 것이 이 논문의 한계점이라고 생각합니다. 여전히 Target Distribution에 대해서 그리 큰 성능을 만들고 있지 않습니다. 

![image-20230416200530743](/Users/aailab/Library/Application Support/typora-user-images/image-20230416200530743.png)



## **5. Conclusion**

Optimal transport를 이용해 각 prompt가 Image의 local 영역을 설명하도록 한다는 것의 아이디어에 대한 좋은 점수를 주고 싶습니다. Parameter를 4배 사용했음에도 성능 gain이 적다는 것은 실제 학습에 문제가 있다( 예를 들어 parameter collapse to one point)로 있다고 생각합니다.



## **6. Reference & Additional materials**



[1] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In *International conference on machine learning* (pp. 8748-8763). PMLR.

[2] Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Learning to prompt for vision-language models. *International Journal of Computer Vision*, *130*(9), 2337-2348.

[3] Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). Learning to prompt for vision-language models. *International Journal of Computer Vision*, *130*(9), 2337-2348.

[4] Jia, Chao, et al. "Scaling up visual and vision-language representation learning with noisy text supervision." *International Conference on Machine Learning*. PMLR, 2021.