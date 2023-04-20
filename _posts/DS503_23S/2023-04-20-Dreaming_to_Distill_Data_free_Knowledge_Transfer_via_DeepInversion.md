---
title:  "[CVPR 2020] Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion"
permalink: Dreaming_to_Distill_Data_free_Knowledge_Transfer_via_DeepInversion.html
tags: [reviews]
---

# **Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion  | Paper Review** 


## <Strong>1. Problem Definition</strong>  


Teacher network의 기존 학습 데이터셋 없이 Knowledge distilation을 이뤄내는 것을 목표로 한다. 

<Blockquote>
<em><strong>Knowledge Distilation</strong></em>: 미리 잘 학습한 큰 네트워크(teacher network)의 지식을 실제로 사용하고자 하는 작은 네트워크(student network)에 전달하는 것
</blockquote>


## **2. Motivation**  

Trained neural network를 통해 knowledge를 transfer 시키기 위한 다양한 시도들이 있었고,최근에는 knowledge distilation 개념을 기반으로 문제를 해결하기 위한 시도들이 많이 있었다. 

Knowledge distilation의 경우 Trained network의 데이터셋을 보존하여 활용하거나,  데이터셋의 분포를 나타내는 대표적인 Real image가 일부 활용되어야 한다는 단점들이 있다. 실제로 Teacher network에 활용된 데이터셋은 프라이버시나 보안 등의 문제로 접근하기 어려운 경우도 많고, 용량이 커서 저장하거나 관리하기 어려운 경우가 많아 이전에 연구된 다양한 knowledge distilation 방식들은 현실적으로 활용하는데 한계가 있다.

다양한 제약으로 인한 Prior data나 Metadata의 부재 속에서 knowledge transfer를 하기 위하여 본 논문에서는 학습된 모델이 그 자체적으로 Rich information을 가지고 있다고 가정하고, Neural Network Inversion 방식을 활용하여 Input training data를 복원한다.

<blockquote>
<em><strong>Neural network inversion</em></strong> : pre-trained 모델의 weight을 고정하여 noise한 input을 forwarding 시키고, backpropagation을 통해서 weight가 아닌 node의 output을 update하여 실제 training data와 유사한 input을 예측하는 방식
</blockquote> 


본 논문의 contribution과 application은 다음과 같다. 
1. Data-free network pruning
2. Data-free knowledge transfer
3. Data-free continual learning


## **3. Method**  

본 논문의 Overall Framework는 다음과 같다. 

![image](https://user-images.githubusercontent.com/47962184/232245723-c92844f8-379a-4878-92d3-355f012161de.png)

#
<strong>3.0. Knowledge Distilation</strong>

앞서도 간단히 언급했듯이 Knowledge distilation은 large model인 teacher network에서 knowldge를 smaller model인 student로 transfer 하는 방법론이다. 주어진 trained model인 $p_T$와 데이터셋 $X$, student model의 파라미터 $\bf W_{\it S}$ 에 대해서 KL divergence loss를 minimize하는 방식으로 student의 파라미터 weight을 업데이트하며 학습한다. 


>$\min_{\bf W_{S}} \displaystyle\sum_{x \in X}{}{\bf KL\it (p_{T}(x),p_{S}(x))}$

>$p_{S}(x) = p(x, {\bf W_{\it S}})$  


#
<strong>3.1. DeepDream</strong>

DeepDream은  Input으로 training data 대신 noise를 받아 synthesized image로 optimize하는 방식이다. 

>$\min_{\hat{x}}L(\hat{x}, y)+R(\hat{x})$

>$R_{prior}(\hat{x})=\alpha_{tv}R_{TV}(\hat{x})+\alpha_{l_2}R_{l_2}(\hat{x})$

Loss function의 첫번째 term $L(\hat{x}, y)$은 실제 실제 이미지의 target label y 와 inversion과정에서 생성된 synthesized image인 $\hat{x}$과의  **Classification loss** 이다. 

Loss function의 두번째 term $R(\hat{x})$은 synthesized image의 total variance($R_{TV}$)와 l2 norm($R_{l_{2}}$)로 구성된 **Regularization Term**이다. 

Regularization Term은 Synthesized image를 smoothing하여 실제 이미지에 수렴시키는 역할을 한다. 

#
<strong>3.2. DeepInversion(DI)</strong>

본 논문에서 소개하는 DeepInveersion에서는 DeepDream의 **Regularization Term**을 확장하여 synthesized image가 실제 이미지와 더 유사해지도록 한다. 

>$R_{feature}(\hat{x})=\displaystyle\sum_{l}{}{}\vert\vert\mu_l(\hat{x}-\mathbb{E}(x)\vert X) \vert\vert_2+\displaystyle\sum_{l}{}{}\vert\vert\sigma_l^2(\hat{x})-\mathbb{E}(\sigma_l^2(x)\vert X) \vert\vert_2$


Training dataset 이미지와 Synthesized Image를 Forwarding 시켰을 때 모든 feature level (layer)에서의 값을 유사하게 만들기 위해 feature map의 차이를 minimize시키는 **feature Regularization Term**이 추가되었다. 이때 Feature statistics로서 Mean과 Variance를 활용하였다. 

DeepInversion의 Regularization Term은 $R_{prior}$과 $R_{feature}$가 합쳐져서 아래와 같이 정의할 수 있다. 

>$R_{DI}(\hat{x})=R_{prior}(\hat{x})+\alpha_{f}R_{feature}(\hat{x})$

#
<strong>3.3. Adaptive DeepInversion(ADI)</strong>

DeepInversion 방법론을 통해 이미지의 feature level을 유사하게 만들면서 실제 이미지와 유사한 synthesized Image를 생성할 수는 있지만, 그로 인해 repeated image가 만들어질 수도 있다. 이를 개선하기 위해 DeepInversion의 **Regularization Term**을 확장한다. Teacher network와 Student network output 분포가 불일치하도록 하여 동일 클래스에 속하는 이미지더라도 다양한 이미지를 생성하도록 유도한다. 

>$R_{complete}(\hat{x})=1-\bf JS\it (p_{T}(x),p_{S}(x)),$

>$JS(p_{T}(x),p_{S}(x))=\frac{1}{2}(\bf KL\it (p_{T}(x),M)+\bf KL\it (p_{S}(x),M))$

>$M =\frac{1}{2}(p_{T}(\hat{x}),p_{S}(\hat{x}))$

Jensen-Shannon divergence는 KL divergence의 평균으로서 Regularization term에서는 JS를 1에서 빼주면서 Teacher과 Student 분포의 거리가 최대가 되도록 유도함을 확인할 수 있다. 


Adaptive DeepInversion의 Reqularization Term은 $R_{DI}$과 $R_{complete}$가 합쳐져서 아래와 같이 정의할 수 있다. 

>$R_{ADI}(\hat{x})=R_{DI}(\hat{x})+\alpha_{c}R_{complete}(\hat{x})$



## **4. Experiment**  


### **4.1. Results on CIFAR-10**
* Implementation details 
	* Used Networks(for pretraining teachers)
		* VGG-11-BN, ResNet-34 
	* Image Synthesis
		* optimizer: Adam (lr = 0.05)
		* 2k gradient updates for image batch
	* Parameters with simple grid search optimizing
		* $\alpha_{tv}=2.5*10^{-5}$
		* $\alpha_{l_{2}}=3*10^{-8}$
		* $\alpha_f =$ {1.0, 5.0, 10.0, 100.0} (for DI)
		* $\alpha_c = 10.0$ (for ADI)
	* Baselines
		* Noise & DeepDream

![image](https://user-images.githubusercontent.com/47962184/232246546-50664aec-8246-4bf6-a9da-c3eae708f857.png)

**4.1.A. Baseline**

Noise 자체는 knowledge distilation에 어떤 도움도 되지 않았음을 확인할 수 있다. Noise만이 Input으로 주어질 경우 input 분포가 급격하게 변화하여 teacher를 방해하고, transferred knowledge까지 영향을 주었다. DeepDream $R_{prior}$을 더해주면 student accuracy가 소폭 상승함을 확인할 수 있다. 

**4.1.B. Effective of DeepInversion & Adaptive DeepInversion**

DeepInversion $R_{feature}$를 더해주면 모든 시나리오에서 40% 이상의 accuracy 향상을 확인하였다. 또한 DeepDream에서의 synthesized image와 달리 DeepInversion는 형체를 이해할 수 있는 현실적인 image를 생성하였다. 
여기에 Adaptive DeepInversion의 $R_{complete}$를 더해주면 student accuracy가 증가하여 teacher accuracy와 거의 비슷해졌다. 

### **4.2. Application**

**4.2.A. Data-free Knowledge Transfer**

* Dataset
	* ImageNet
* Network 
	* ResNet50v1.5
* Parameters
	* temperature $\tau$=3
	* initial lr = 1.024
	* batch size = 1024
![image](https://user-images.githubusercontent.com/47962184/232249321-2b1efb64-ff7a-4899-a7e9-859302610a9b.png)

DI를 통한 Knowledge Transfer는 base teacher model에 대비하여 3% 정도만의 accuracy 차이를 보이며 좋은 성과를 보였다. 

**4.2.B. Data-free Pruning**

Pruning은 network의 성능이 크게 저하되지 않는 선에서 개별 weight이나 특정 필터(neurons)를 삭제하는 모델 경량화 방식이다. 본 논문에서는 DI, ADI 방식으로 생성된 inverted image를 통하여 filter importance를 계산하여 pruning을 진행하여 타 모델과 결과를 비교하였다. 

![image](https://user-images.githubusercontent.com/47962184/232248656-29de76d4-7b81-448f-9367-43a827685e09.png)

Top-1 accuracy에 있어서 GAN과 거의 동등한 결과를 보였다. 하지만 더 많은 필터들을 pruning 할 수록 Synthesized image를 사용하는 ADI, BigGAN과 Natural image를 사용하는 MS COCO와 ImageNet 간의 accuracy 차이가 커졌다. Synthesized image가 생성되는 과정 자체의 한계인 것으로 보인다. 


## **5. Conclusion**  

* 본 논문은 Prior Dataset이나 Data distribution이 주어지지 않는 상황에서의 Knowledge Transfer를 해결하기 위해 모델의 내재적인 information을 활용하였다.
* Synthesize image를 생성하기 위한 regularization term을 활용하였고, output variety를 유지하기 위하여 Teacher 과 Student 모델 간의 distance에 constraint를 적용하였다.
* 실험 결과를 통해 Regularization term을 구성하는 각각의 요소들이 유의미하게 성능향상에 도움을 주고 있음을 확인하였고, Synthesized image도 기타 모델에 비하여 높은 현실성을 가지고 있음을 볼 수 있었다.

* Knowledge transfer과 Pruning 등 다양한 application에 있어서도 타 모델과 유사한 performance를 보였다. 

* Image synthesis에 상당한 시간이 소요되며 image 수와 비례하고, image의 색과 배경들이 유사하다는 단점들이 있었지만, Model internal information을 간단한 constraint를 통해 활용하였다는 점에서 매우 신선했다. 




---  
## **Author Information**  

* Minseok Kim
	* KAIST Financial Engineering Lab
	* contact: hankkim77@kaist.ac.kr
	
* Paper Source & Github
	* [1912.08795.pdf (arxiv.org)](https://arxiv.org/pdf/1912.08795.pdf)
	* https://github.com/NVlabs/DeepInversion

