---
title:  "[Nat. Mach. Intell. 2023] Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data"
permalink: Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging_with_unpaired_data.html
tags: [reviews]
use_math: true
usemathjax: true
---

## **1. Problem Definition**  

이 논문에서 다루는 문제는 <b><span style="color:Blue">single-shot in-line hologram에서 complex amplitude와 object-to-sensor distance를 재구성</span></b>하는 것이다.

Challenge는 <b><span style="color:Blue">복잡한 진폭의 ground truth를 필요로하지 않고</span></b> hologram에서 complex amplitude와 distance 정보를 정확하게 재구성할 수 있는 방법을 개발하는 것이다.

이를 통해 생물학적 이미징, 계측, 광학 데이터 저장 등 다양한 응용 분야에서 <b><span style="color:Blue">효율적이고 정확한 digita holographic imaging이 가능</span></b>해진다.

---
## **2. Backgound Research**  

본 연구의 배경은 객체에 의해 산란된 광파의 진폭과 위상 정보를 모두 포착하는 강력한 영상 기술인 디지털 홀로그래피에 있다. 디지털 홀로그래피는 스캔이나 기계적 초점 조절 없이 고해상도 3D 영상을 제공할 수 있는 능력 때문에 현미경, 계측, 센싱과 같은 다양한 분야에서 적용되어 왔다.

그러나 single-shot in-line 홀로그램에서 <b>complex amplitude (amplitude + phase)</b>를 재구성하는 것은 complex amplitude의 ground truth가 없는 경우에는 특히 어려운 문제이다. 전통적인 방법들은 종종 multiple aquisition에 의존하는데, 이는 시간이 많이 소요되며 동적이거나 빠르게 성장하는 샘플에 적합하지 않을 수 있다.

최근 딥러닝의 발전은 홀로그래픽 재구성을 포함한 다양한 영상 문제를 해결하는데 매우 유망한 결과를 보여주었다. 이 논문의 저자들은 이러한 발전을 활용하여 single-shot in-line 홀로그램에서 comple amplitude와 object-to-sensor 거리를 재구성하는 새로운 딥러닝 기반 방법을 개발하려고 한다. 이를 통해 전통적인 방법의 한계를 극복하고 더 효율적이고 정확한 접근법을 제공할 수 있다.

---
## **3. Motivation**  

이 논문의 motivation은 생물학적 이미징, 계측, 광학 데이터 저장과 같은 다양한 분야에서 single-shot in-line hologram에서 complex amplitude와 object-to-sensor distance를 재구성하는 어려움 때문에 직면한 limitation과 challenge를 해결하려는 것이다.

digital holographic imaging의 기존 방법은 종종 complex amplitude의 ground truth를 필요로 하는데, 이는 실제 application에서 major limitation이 될 수 있다.

complex amplitude의 ground truth를 필요로 하지 않는 딥러닝 기반 방법을 개발함으로써 연구자들은 이러한 limitation을 극복하고 다양한 응용 분야에서 digital holographic imaging의 효율성과 정확도를 향상시키고자 한다.

기존의 연구들과 차별화되는 아이디어는 <b><span style="color:Red">parameterized physical forward model을 사용한 딥러닝 기반 방법을 활용하는 것</span></b>이다.

이 접근법은 complex amplitude의 ground truth를 필요로 하지 않고 single-shot in-line hologram에서 complex amplitude와 object-to-sensor distance를 동시에 재구성할 수 있게 한다.

<b>제안된 방법은 U-Net, CycleGAN 및 PhaseGAN과 같은 기타 최첨단 기술과 비교하여 다양한 시나리오에서 정확한 재구성 결과와 거리 추정을 제공하여 우수한 성능을 보여준다.</b>

<img src="https://user-images.githubusercontent.com/127107965/230815084-7dc6440b-a23b-426b-81e4-7c1597cc64b8.png" width="1000" height="1000" alt="Fig1"/>
<center><b>Fig.1 Overview of the proposed model</b></center>

---
## **4. Method**  

본 연구에서 저자들은 single-shot in-line hologram에서 complex amplitude 및 object-to-sensor distance를 재구성하기 위한 딥러닝 기반 방법을 제안한다. 이 방법은 비지도 학습에 기반하며, complex amplitude의 ground truth를 필요로 하지 않는다.

> <b>1. Sample Preparation</b>

저자들은 실험을 위해 polystyrene microscphere, 적혈구, 조직샘플을 사용했다. 샘플은 희석, 분산 및 절단 등 다양한 방법으로 준비되었다.

> <b>2. Experimental Setup and Data Aquisition</b>

cutom-built Mach-Zehnder 간섭계를 사용하여 complex amplitude 데이터를 획득했다. 이 setup에는 laser, beam spliter, objective lens 및 sCMOS camera가 사용되었다. 데이터는 푸리에 변환 기반 탈축(off-axis) 홀로그래피 재구성 알고리즘을 사용하여 처리되었다. 

> <b>3. Deep Learning Architecture</b>

저자들은 complex amplitude 생성기로 U-Net 기반 아키텍처와 distance 생성기로 Deep Neural Network (DNN)을 사용했다. disciminator network는 complex amplitude의 구조 정보와 미세한 세부 정보를 강조하기 위해 구축되었다.

<b><span style="color:Blue">$G_{\theta}$</span></b>는 인코더와 디코더로 이루어진 U-Net 기반의 아키텍처이다.

인코더는 diffraction intensity를 input으로 받고 인코더 내에서 3x3 convolution, group normalization, 기울기 0.2인 leaky ReLU와 2x2 stride 2인 maxpooling으로 구성된 convolution 블럭이 두 번 반복된다. 인코더에 의해 latent vector가 생성되고 이는 디코더의 input으로 주어진다.

디코더는 transpose convolution과 두 개의 convolution 블럭의 반복으로 구성된다. 디코더 내에서 첫 번째 convolution 블럭은 인코더의 마지막 convolution 블럭의 output을 input으로 취한다. 디코더의 output은 squeeze-to-excitation과 1x1 convolution에 대한 input으로 사용되어 complex amplitude map을 생성한다.

<b><span style="color:Blue">$G_{\psi}$</span></b>는 feature extraction layer와 distance regression layer의 조합으로 구성된다.

feature extraion layer들은 kxk convolution (stage1에 대해 k=7, stage2에 대해 k=5, stage3에 대해 k=3), group normalization, 기울기 0.1인 leaky ReLU와 2x2 stride 2인 maxpooling으로 구성된 convolution 블럭이 두 번 반복된다. 추출된 feature들은 global average pooling에 의해 1x1 feature map으로 축소된다. 이는 하나의 1x1 convolution layer로 이루어진 distance regression layer의 input으로 사용되어 결국 distance를 생성한다.

<b><span style="color:Blue">$D_{\eta}$</span></b>는 local feature와 global feature를 강조하기 위해 구축되었다.

low-pass filter와 high-pass filter를 통해 filtering된 이미지는 $D_{\eta}$의 input으로서 주어진다. low-pass filter와 high-pass filter는 각각 5x5 Gaussian blur kernl과 Laplace pyramid representation으로 구현된다. amplitude와 phase 이미지는 각 pass filter에서 filtering된 이미지와 concatenation되며 stride 2인 4x4 convolution, leaky ReLU로 구성된 convolution 블럭의 input으로 들어간다. output들은 concatenation되고 4x4 convolution, group normalization, leaky ReLU로 이루어진 convolution 블럭들의 시리즈 이후의 squeeze-to-excitation network의 input이 된다. 결국 global average pooling과 1x1 convolution을 거쳐 주어진 complex amplitude가 real인지 fake인지 결정할 수 있게 된다.

> <b>4. Loss Function</b>

제안된 모델은 cycle-consistency loss, Wasserstein GAN (WGAN) loss, gradient penalty loss와 structural similarity index loss의 조합으로 훈련되었다.

제안된 모델은 다음 loss function으로 훈련되었다.

$min_{\Theta,\psi} max_{\eta} l_{tot}(\Theta,\psi;\eta) :$   
$ = l_{cycle}(\Theta,\psi) + \lambda_{WGAN}l_{WGAN}(\Theta;\eta) + \lambda_{GP}l_{GP}(\eta) + \lambda_{SSIM}l_{SSIM}(\Theta)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Theta :$ complex amplitude 생성기  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\psi :$ distance 생성기  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\eta :$ distance 판별기  

<b><span style="color:Blue">cycle-consistency loss</span></b>는 다음과 같이 정의된다.

$l_{cycle}(\Theta,\psi) = \lambda_UE_{U\tilde{}P_U}[\vert G_{\theta}(F(U,d))-U\vert] + \lambda_dE_{d\tilde{}P_D}[\vert G_{\psi}(F(U,d))-d\vert] + \lambda_IE_{I\tilde{}P_J}[\vert I-F(G_{\theta}(I), G_{\psi}(I))\vert]$  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$P_U :$ probability distribution for U  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$P_D :$ probability distribution for d  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$P_J :$ probability distribution for I  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$E[\dot{}] :$ expectation  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\lambda_{U}, \lambda_{d}, \lambda_{I} :$ hyperparameters for each cycle-consistency loss

<b><span style="color:Blue">Wasserstein GAN loss</span></b>는 다음과 같이 정의된다.

$l_{WGAN}(\Theta;\eta) = E_{U \sim P_U}[D_{\eta}(U) - E_{I\tilde{}P_J}[D_{\eta}(G_{\theta}(I))]$  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\lambda_{WGAN} :$ hyperparameters for adversarial loss  

<b><span style="color:Blue">gradient penalty loss</span></b>는 다음과 같이 정의된다.

$l_ {GP}(\eta) = -E_ {U_{0} \sim P_ U}[(\nabla_ {\hat{U}}D_ {\eta}(U) _2 - 1)^2]$  

where $\hat{U} = tU + (1-t)G_ {\theta}(I)$  

<b><span style="color:Blue">structural similarity index loss</span></b>는 다음과 같이 정의된다.

$l_{SSIM}(\Theta) = E_{U \sim P_U}[1-SSIM(U,G_{\theta}(F(U,d)))]$  

> <b>5. Data Preprocessing and Training</b>

이미지는 patch로 자르고 증강되었다. 훈련과정에는 Adam optimizer, Xavier normalization 및 hyperparameters tuning이 사용되었다.

> <b>6. Evaluation Metrics</b>

complex amplitude 재구성 및 distance 예측 결과는 Pearson Corelation Coefficient (PCC), Feature Similarity Index (FSIM), Mean Absolute Error (MAE) 및 modified ordinarty least square ($R_0^2$)을 사용하여 평가되었다.

* PCC - 1에 가까울수록 재구성 결과가 더 우수함

* FSIM - 1에 가까우면 재구성 결과가 더 우수하고, 그렇지 않으면 0에 가까움

* MAE - 낮을수록 회귀 분석 결과가 우수함

* $R_0^2$ - 1에 가까우면 더 나은 회귀 분석 결과를 나타내고, 그렇지 않으면 $R_0^2$<<1

> <b>7. Compared Methods</b>

저자들은 제안된 모델은 U-Net, CycleGAN 및 PhaseGAN 세 가지 방법과 비교했다.

> <b>8. Statistics and Reproducibility</b>

저자들은 폴리스타이렌 마이크로스피어, 적혈구, 조직 샘플과 같은 다양한 샘플 유형에 대한 다양한 입력 패치에서 훈련된 네트워크를 테스트했다. 결과는 다양한 입력 패치에서 일관성을 유지했다.

---
## **5. Results**

<img src="https://user-images.githubusercontent.com/127107965/230815390-90a482f2-71e5-4d93-b206-66b73ca79a77.png" width="1000" height="500" alt="Fig2"/>
<center><b>Fig.2 Demonstration of simultaneous reconstruction of complex amplitude and object distance</b></center>

`제안된 방법은 complex amplitude와 object distance를 동시에 재구성할 수 있다. 본 연구에서는 제안된 방법을 U-Net, CycleGAN, PhaseGAN과 같은 다른 딥러닝 접근법과 비교한다. 제안된 방법은 모든 object-to-sensor 거리에 대해 정확한 재구성 결과를 보여준다.`

<img src="https://user-images.githubusercontent.com/127107965/230815391-6d67dde0-9daf-4ffb-88c9-a6f41449a8b2.png" width="1000" height="500" alt="Fig3"/>
<center><b>Fig.3 Demonstration of adaptive holographic imaging</b></center>

`적응형 홀로그래픽 이미징 접근법은 모델이 강한 섭동으로 인한 분포 이외의 데이터를 처리할 수 있게 한다. 제안된 방법은 complex amplitude를 재구성하고 object distance를 예측하는 데 최첨단 성능을 유지하지만, 다른 방법들은 정확한 재구성 결과를 제공하지 못한다.`

<img src="https://user-images.githubusercontent.com/127107965/230815394-f655ecc2-9e98-497e-a9ea-574a107e0a8e.png" width="1000" height="500" alt="Fig4"/>
<center><b>Fig.4 Demonstration of holographic imaging of RBCs in a dynamic environment</b></center>

`제안된 방법은 빠르게 변하는 환경에서 적혈구의 홀로그래픽 이미징에 적용되어 실용성을 입증한다. 방법은 개별 적혈구로부터의 회절 강도가 겹쳐져 있거나 하나의 적혈구가 다른 적혈구 위로 흐르는 상황에서도 phase profile을 믿을 만하게 재구성한다.`

<img src="https://user-images.githubusercontent.com/127107965/230815395-ef80e3f2-0426-43cc-bbce-f28723321be6.png" width="1000" height="500" alt="Fig5"/>
<center><b>Fig.5 Holographic imaging of history slides without ground truth</b></center>

`제안된 방법은 또한 ground truth가 없는 조직 슬라이드의 홀로그래픽 이미징에 적용되어 실제 병리학적 환경에서의 효과를 보여준다. 제안된 모델은 기존 방법에 비해 세포 몸체와 구조의 가시성 측면에서 우수한 재구성 결과를 제시한다.`

---
## **6. Conclusion**

본 연구에서는 parameterized physical forward model이 이미지 복구의 inverse problem을 해결하는 딥러닝 모델에 적응성과 신뢰성을 부여할 수 있음을 보여준다. 특히, 이 접근법은 홀로그래피 이미징에 적용되어 놀라운 일관성을 보였으며, 노이즈 대처 능력이 크게 향상되었다. 제안된 접근법은 critical한 노이즈를 적응적으로 처리하는 능력을 통해 홀로그래피 이미징에서 딥러닝 기반 접근법의 적용 가능성을 확장할 것으로 예상한다.

또한, 본 연구에서는 이미징에 집중했지만, 제안된 접근법이 critical한 노이즈가 Out-of-distribution (OOD)데이터를 발생시킬 수 있는 다양한 inverse problem에서 강력하고 적응적인 정규화를 구현하는 효율적인 방법을 제공할 것으로 생각된다. 이를 통해 광학 이미징 분야의 다양한 이미징 방식에도 적용될 수 있을 것이다.

---  
## **Author Information**  

* Chanseok Lee
    * Department of Bio and Brain Engineering, KAIST, Daejeon, South Korea
    * KAIST Institute for Health Science and Technology, KAIST, Daejeon, South Korea
  
  
* Gookho Song
    * Department of Bio and Brain Engineering, KAIST, Daejeon, South Korea  
    * KAIST Institute for Health Science and Technology, KAIST, Daejeon, South Korea
  
  
* Hyeonggeon Kim
    * Department of Bio and Brain Engineering, KAIST, Daejeon, South Korea  
    * KAIST Institute for Health Science and Technology, KAIST, Daejeon, South Korea
  
  
* Jong Chul Ye
    * Kim Jaechul Graduate School of AI, KAIST, Daejeon, South Korea
      
      
* Mooseok Jang
    * Department of Bio and Brain Engineering, KAIST, Daejeon, South Korea
    * KAIST Institute for Health Science and Technology, KAIST, Daejeon, South Korea

---  
## **Reference & Additional materials**  

> ### Data availability
https://doi.org/10.6084/m9.figshare.21378744

> ### Code availabilty
https://doi.org/10.5281/zenodo.7220717
https://github.com/csleemooo/Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging
