---
title:  "[WWW 2021] SimGRACE: A Simple Framework for Graph Contrastive Learning without Data Augmentation"
permalink: SimGRACE_A_Simple_Framework_for_Graph_Contrastive_Learning_without_Data_Augmentation.html
tags: [reviews]
use_math: true
usemathjax: true
---
# **SimGRACE: A Simple Framework for Graph Contrastive Learning without Data Augmentation** 

## **1. Problem Definition**  

Please write the problem definition on here  

Graph Contrastive learning(GCL)에서 Graph augmentation이 사용되는데 Graph의 본질적인 의미를 훼손하지 않고 진행하기가 어렵습니다. 따라서 Graph augmentaion은 GCL의 일반적인 적용의 가능성이나 효율성을 제안한다는 한계가 있습니다. 본 논문에서는 Graph Contrastive IEarning을 위한 Simple 프레임워크인 SimGRACE를 제시합니다. 이 프레임워크는 Graph Augmentation이 필요하지 않습니다.

## **2. Motivation**  

GCL은 보통 augmentaion으로 4가지 방법을 주로 사용합니다(node dropping, edge perturbation, attribute masking and subgraph). 하지만 이 4가지 방법은 모든 경우에 사용되는 것이 아니라 graph에 따라 차이가 존재합니다. 예를 들어 소셜 네티워크 graph에서는 edge perturbation이 잘되지만 생화학 분자 구조 edge를 변경하는게 분자 구조를 바꾸어 성능이 좋지않다는 것 등이 있습니다. 

이러한 문제를 해결하기 위해 매뉴얼에따라 augmentaion을 선택하는 trail-and-error를 이용하는 방법들이 제시되었습니다. 하지만 trial-and-error를 이용하는 방식들은 많은 연산량을 요구하고 여전히 GCL의 일반적인 사용에도 한계가 존재합니다. JOAO의 경우는 자동적으로 GCL에서 augmentaion pairs를 선택하는 방법을 제시했으나 계산의 복잡도가 올라갔고 augmentaion pool을 구성한느데 인간의 사전 지식을 이용한다는 한계가 존재합니다. 따라서 본 논문에서는 **어떻게하면 manual trial-and-errors를 사용하지 않고 복잡한 계산이나 domain 지식 또한 사용하는 않고 GCL을 사용할 수있을까**라는 motivation을 제시합니다. 따라서 저자는 graph augumentaion을 사용하지 않고 semantic-preserved data augmentaion을 사용하여 이를 해결하려합니다. 

## **3. Contribution**  

- *Significance*: 기존의 GCL 방법들에 비해 일반적으로 적용가능하고 manual trail-and-errors를 사용하지 않는 새로운 GCL 방법을 제시합니다.
- *Framework*: 새롭고 효율적인 framework를 제시하고 SimGRACE가 잘 작동할 수 있는 이유를 설명합니다.
- *Algorithm*: GCL의 Robustness를 향상시키기 위해 AT-SimGRACE라는 새로운 알고리즘을 제시합니다. 약간의 computational overhead가 존재하지만 더 Robustness한 결과를 제시합니다.
- *Experiment*: 여러 종류의 dataset에 대해 state-of-the-art 방법들과 비교해 더 뛰어나거나 경쟁력있는 모습을 보여줍니다.

## **4. Graph Contrastive Learning**  

GCL은 2가지로 나눌 수 있습니다. 첫번째는 local과 global representation을 대조하여 encoding을 진행하는 방식입니다. DGI과 InfoGraph는 graph-level representaion과 substructure-level representaion의 차이를 최대화하여 graph나 node의 representaion를 encoding합니다. 보다 최근에 나온 MVGRL은 node diffusion을 수행하고 contrast learning을 이용해 graph-level과 node-level의 representaion을 얻는 것을 제안합니다다. 두번째는 data를 변환하는 방법으로 사용되는데 augment하고 이를 shared encoder과 projection head에 넣어 mutual information을 최대화합니다. GCA는 node-level task를 위해 제시되었고 DGCL은 false negative 문제를 해결하기 위해 제시되었습니다. Graph-level에서는 GraphCL이 4가지 방법의 augmentaion을 사용하여 제시되었습니다. JOAO는 GraphCL의 manual trail-and-error의 문제를 해결하기 위해 제시되었습니다.

## **5. Method**  
![SimGRACE](https://user-images.githubusercontent.com/101261577/232289501-42c61afb-f639-473e-b6e1-d8c9b8b5f164.png)


**(1) Encoder perturbation**

$h$ 와 $h^\prime$ 2개의 graph-level representaion을 추출합니다.

$h=f(G;\theta),h^\prime=f(G;\theta)$

$\theta$와 $\theta^\prime$은 GNN 인코더의 l번째 레이어의 weight tensor와 perturbed version입니다.
$\Delta\theta_l$ 는 평균이 0이고 분포가 $\sigma^2_l$ 인 가우시안 분포에서 sampling하는 perturbation term입니다. 여기서 SimGRACE는 기존의 모델들과 3가지 차별점이 있는데. (1) 모멘텀 업데이트 대신 무작위 가우시안 노이즈로 인코더를 perturbation 시킵니다. (2) data augmentaion을 필요로하지 않습니다. (3) graph-level representaion에 집중되어 있습니다.

$\theta^\prime_l=\theta_l + \eta \cdot \Delta\theta_l$

**(2) Projection head**

Projection head라는 non-linear transformation $g(\cdot)$을 이용하여 representaion을 다른 latent space에 매핑 시켜 성능을 향상 시킬 수 있습니다. SimGRACE는 two-layer perceptron(MLP)을 이용합니다.

$z=g(h), z^\prime = g(h^\prime)$

**(3) Contrastive loss**

SimGRACE에서는 normalized temperature-scaled cross entropy loss (NT-Xent)를 사용하여 postive pairs인 $z$와 $z^\prime$을 negative pairs와 비교하여 그 차이를 줄여갑니다. 본 논문에서 N graph를 randomly sampling하여 GNN인코더를 통해 perturbed version까지 만들었습니다 따라서 2N개의 representaion이 존재하는데 미니배치에서 n번째 graph를 $z_n$라고 표현합니다. Negative pairs는 자신을 제외한 나머지 N-1개의 pertubed representaion을 통해서 나오게됩니다. 따라서 n번째 graph에 대한contrasive loss는 다음같이 나옵니다.

$l_n = -log {exp(sim(z_n, z^\prime_n)\tau)\over \sum^N_{n^\prime=1,n^\prime\ne n} exp(sim(z_n,z_{n^\prime}))\tau)}$

sim은 cosine similarity이고, final loss는 모든 postive pairs에 대해서 계산됩니다.


### **AT-SimGRACE**  

GraphCL은 GNN framework를 사용하여 Robustness를 얻을 수 있음을 제시하지만 그 이유까지 제시하지는 않습니다. 또한 GraphCL은 random attack에 대해서는 Robust하지만 adversrial attack에서는 취약한 모습을 보입니다. AT-SimGRACE는 adversarial attack에 Robustness를 향상시켰습니다. 일반적인 AT Framework는 다음과 같습니다.

![AT Framework](https://user-images.githubusercontent.com/101261577/232289517-a039b362-723c-4517-8693-95a02562e4e0.png)

하지만 위의 framework는 graph contrastive learning에 바로 적용할 수 없습니다. 따라서 loss를 위에서 설명한 Contrastive loss로 대체합니다. 또한 효율성을 높히기 위해 다음과 같은 방법을 도입합니다.

$\Theta$를 GNN의 weight space라고 가정하면 $\theta$를 L2 norm을 이용해 다시 정의할 수 있습니다.

![AT Framework2](https://user-images.githubusercontent.com/101261577/232289522-0ec02f2f-3783-434b-a0b7-da8b33eba60b.png)

![AT Framework3](https://user-images.githubusercontent.com/101261577/232289527-23f9ed5a-23df-4bf0-a551-9c66d0b47ed9.png)

이제 AT-SimGRACE는 optimization problem을 다시 정의하는데 inner maximization를 하기위해 contrastive loss를 gradient ascent 방법으로 update합니다. 이를 통해 $\theta$를 미니배치 단위로 SGD를 통해 update합니다

## **6. Experiment**  

### **Research Question**  

- **RQ1.(Generalizability)**: SimGRACE는 unsupervised와 semi-supervised에서 다른 모델보다 우수한가?
- **RQ2.(Transferabilitry)**: SimGRACE로 pre-train된 GNN이 다른 모델보다 더 나은 transferability를 보여줄 수 있는가?
- **RQ3.(Robustness)**: AT-SimGRACE는 다양한 adversarial attack에 더 나은 성능을 발휘할 수 있는가?
- **RQ4.(Efficiency)**: SimGRACE의 효율성은 어떻고 다른 모델에 비해 효율적인가?
- **RQ5.(Hyperparameters Sensitivity)**: SimGRACE가 hyperparameter에 대해 얼마나 Sensitivity한가?

### **Unsupervised and semi-supervised learning (RQ1)**  

![Table2](https://user-images.githubusercontent.com/101261577/232289536-9924cc76-f692-4988-87a3-94246c84aa89.png)

![Table4](https://user-images.githubusercontent.com/101261577/232289546-53f6a789-57c9-4a0a-8933-d8bf2ee8fac4.png)

Table2를 보면 Unsuperviesd 경우 SimGRACE가 다른 baseline들을 능가하며 모든 dataset에서 상위 3위 안에 듭니다. 또한 Table 4를 보면 semi-superviesde task를 1%와 10%와 label에서 진행하였는데 SOTA 방법론들과 비교했을때 비슷한 성능을 보이거나 더 능가하는 모습을 보였습니다. 10% label에서 JOAO가 조금 더 나은 성능을 보이느데 JOAO의 비효율성을 생각해보면 SimGRACE의 성능 또한 우수하다 볼 수 있습니다.

### **Transferability (RQ2)**  

![Table3](https://user-images.githubusercontent.com/101261577/232289556-a1a9f31d-e33e-45e8-8bc3-ffde2c869d4d.png)

Pre-training의 transferability를 평가하기 위해 단백질 기능 예측에 대한 transfer learning에 대한 실험을 진행하였습니다. Table 3에 나와 있듯이 SimGRACE는 PPI dataset에서 다른 pre-training scheme에 따라 더 나은 Transferability에 대한 가능성을 보여줍니다.

### **Adversarial robustness (RQ3)**  

![Table5](https://user-images.githubusercontent.com/101261577/232289574-7fc89115-24ee-439b-8d43-813cc544a57f.png)

RandSampling, GradArgmax와 RL-S2V에 대해 AT-SimGRACE의 Robustness를 평가했습니다. Structure2vec를 GNN 인코더를 사용하여 진행했습니다. 3가지 evasion attack에서 AT-SimGRACE는 GNN의 Robustness를 눈에 띄게 향상시켰습니다.

### **Efficiency (RQ4)**  

![Table6](https://user-images.githubusercontent.com/101261577/232289577-68964294-fe00-4165-a094-3f504437c510.png)

훈련 시간과 메모리 overhaed 측면에서 결과를 비교해본 결과 SimGRACE는 JOAOv2보다 거의 40-90배 더 빠르고 GCL보다 2.5-4배 더 빠릅니다. GCL의 traial-and-error의 시간까지 고려하면 SimGRACE의 효율성은 더 뛰어나다고 볼 수 있습니다


### **Hyper-parameters sensitivity analysis (RQ5)**  

![Figure4](https://user-images.githubusercontent.com/101261577/232289581-5700dfd6-219f-4ac6-ae13-8f065ab8e54e.png)

**Magnitude of the pertubation**
Figure 4에서 볼 수 있듯이 weight pertubation는 SimGRACE에서 매우 중요합니다. $\eta$에 따라 변화하는 성능을 보면 늘 높다고 좋은 결과를 보이지는 않습니다. $\eta$가 0인 경우는 가장 낮은 성능을 내는데 이는 직관적으로 옳은 결과입니다. 적잘한 $\eta$를 설정하는게 중요하다고 할 수 있습니다.

![Figure5](https://user-images.githubusercontent.com/101261577/232289585-415c1f7a-42f9-425b-a461-72bf41cb269c.png)

**Batch-size and traing epochs**
Figure 5은 다양한 배치 크기와 epoch로 훈련된 결과를 나타냅니다. 일반적으로 더 큰 배치 크기와 epoch일때 좋은 성능이 보여집니다. 왜냐하면 배치 크기가 더 클 수록 더 많은 negative sample을 제공하기 때문입니다.

## **5. Conclusion**  

SimGRACE는 기존 Graph Contarsive Learning model의 data augmentaion의 한계를 계선 시키였고, General한 사용이 가능하게 하였습니다. 또한 AT-simGRACE를 통해 Robustness도 향상되었습니다. 향후에는 (1)인코더의 perubation이 컴퓨터 비전이나 자연어 처리 부분에서 잘 활용될 수 있는지 연구해볼 필요가 있습니다. 또한 (2) Pre-train된 GNN을 여러 real-world task에 적용해볼 필요가 있습니다.

---  
## **6. Github**  

* Github Implementation  : https://github.com/junxia97/SimGRACE
