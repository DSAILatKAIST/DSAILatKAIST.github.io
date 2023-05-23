---
title:  "[Nature Machine Intelligence 2022] Super-resolution generative adversarial networks of randomly-seeded fields"
permalink: Super_resolution_generative_adversarial_networks_of_randomly_seeded_fields.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Paper title : Super-resolution generative adversarial networks of randomly-seeded fields

## 1.introdcution
### 1.1 Problem definition
    "기계"는 종종 삶을 좀 더 편리하고 효율적으로 살아갈 수 있도록 도와주는 방법 혹은 도구로 정의된다. 많은 공학 (engineering) 분야에서 이러한 도구를 최적화하기 위해 많은 연구들이 수행되고 있으며, 특히, 딥러닝 방법론이 공학 분야에 어떤 식으로 더욱 효율적인 도구를 개발하기 위해 적용되는지 본 리뷰 논문을 통해 소개하려고 한다.

    본 수업을 수강하는 대부분의 data scienctists의 peer reviewer들 위해, 공학 분야에 사용되는 물리적인 방정식에 대한 세부적인 내용은 최대한 줄이고, "딥러닝 알고리즘의 공학 분야에 적용 가능성"을 중점으로 본 리뷰를 이어가려고 한다. 우선, 현실적으로 가장 와닿는 예시로써, 아래 그림 1과 같이, 대부분의 사람들은 하루를 시작하기 전에 기상, 날씨 혹은 미세먼지가 얼마나 높은지 검색을 해보곤 한다. 이는 특정 지역에서 기계 장비 (sensor)의 측정 (measurement)을 통해 얻어진 풍량, 온도, 기압과 같은 데이터 정보들을 종합한 예보를 통해 인간의 삶을 자연 재해로 부터 대비할 수 있게 만들어 준다.

<center><img src="/images_DS503/figure_1.gif" alt="example image" width="500" height="500"></center>
    
<center>그림 1. 센서 장비에 의해 측정된 데이터 정보 예시 </center>
    
    하지만, 이러한 기계 장비는 정확도가 높고 정밀한 데이터를 얻기 위해 다양하고 복잡한 전자 장비를 동반하기 때문에 큰 설치 비용이 발생한다. 이에 따라, 관심 있는 지역에 공간적으로 거리가 먼 (Sparse) 센서 장비 설치를 하고, 특정 위치에서 측정된 값을 넓은 영역을 대표하는 방식을 채택하여 해상도가 적은 (low-resolution) 결과를 제공한다. 즉, 사람이 활동하고 있는 해상도가 높은 (high-resolution) 다양한 공간 내 정보를 조밀하게 배치된 센서 장비를 통해 얻는 것은 현실적으로 큰 비용 문제가 발생한다.
    
### 1.2 Neural network methodology
    이러한 비용-공간 해상도 사이 절충 (trade-off) 문제를 해결하기 위해, 인공신경망 (neural-network) 기반 방법론이 저해상도 데이터와 고해상도 데이터 사이 비선형 (nonlinear) 관계성을 연결하는데 효율적이고 성공적인 결과를 보이는 도구로 증명되고 있다. 딥러닝 방법론이 저해상도와 고해상도 사이 모델링에 어떻게 적용되는지 설명하기 위해 아래 그림 2을 통해 나타내고자 한다. 대표적으로, 딥러닝 모델의 입력은 관심 있는 지역에 공간적으로 거리가 먼 (Sparse) 센서 장비 정보이고 딥러닝 모델의 출력은 해상도가 높은 (high-resolution) 다양한 공간 내 정보로 설정하여 두 사이 간 비선형적 관계를 노드와 레이어를 갖는 복잡한 neural network 조합을 통해 학습하고 새로운 저해상도 정보로 부터 고해상도 센서 정보를 예측한다 [1].

<center><img src="/images_DS503/figure_2.jpeg" alt="example image" width="700" height="500"></center>
    
<center>그림 2. 인공신경망 기법을 기반한 저해상도</center>
    
    더욱 구체적으로, 저해상도-고해상도 관계를 모델링하기 위해, 그림과 같이 크게 (1) residual block와 (2) super-resolution block 를 갖는Deep learning architecture [2]를 제안한다. 저해상도 이미지를 deep learning model에 입력으로 사용하고, Residual block을 반복적으로 연결하여 입력 및 출력 사이 공간적 정보 손실을 최소화한다. 이 후, 학습 변수를 갖는 up-sampling convolutional operation을 residual block 을 구조 끝단 활용하여 고해상도 output 차원까지 복구하는 기법이 이용되고 있다. 

<center><img src="/images_DS503/figure_3.png" alt="example image" width="800" height="500"></center>
    
<center>그림 3. 저해상도-고해상도 맵핑을 위한 뉴럴네트워크 구조</center>
    
    이는 기존 저해상도에서 고해상도로 복구하는 일반적인 선형 interpolation 기법에 비교하여 convolution 연산을 통해 비선형성을 맵핑할 수 있기 때문에, 놀라운 고해상도 복구 결과를 아래 그림 3과 같이 보인다.

<center><img src="/images_DS503/figure_4.png" alt="example image" width="800" height="500"></center>
    
<center>그림 4. 일반적인 저해상도-고해상도 맵핑 딥러닝 알고리즘 및 기존 보간 알고리즘 예측 결과 비교</center>
    
### 1.3 Challenges of the exsiting deep learning approachs for engineering application
    
    센서 저해상도-고해상도 문제에 대해 딥러닝 방법론이 성공적으로 적용되고 있지만, 앞서 설명한 기상 예보와 같은 센서 어플리케이션을 위한 딥러닝 방법론을 적용하기엔 다음과 같은 두 가지 한계점이 여전히 존재한다.
    
     (1) 딥러닝 학습을 위한 고해상도 데이터 획득 문제: 다량의 데이터가 존재한다면, 앞서 설명한 딥러닝 방법론이 성공적으로 다양한 저해상도-고해상도 문제 (미세 현미경 정확도 보정, 해수면 온도 측정 및 유체 난류 해상도 문제 등)에 적용할 수 있지만, 공학 분야에 이용되고 있는 센서들을 통해 조밀한 고해상도 딥러닝 학습 데이터를 확보 하는 것은 매우 큰 비용이 발생한다.
   
     (2) 딥러닝 학습을 위한 고해상도 데이터 관리 문제: (1)의 문제를 해결할 수 있더라도, 고해상도로 구축된 센서들은 예기치 못한 다양한 환경 조건으로 인해 기기가 무작위로 On-off되거나 움직임이 필요한 센서로 인해 일관성 있고 정확한 고해상도 데이터를 확보하는 것이 매우 어려운 문제를 제공한다.

    
    
## 2. Deep learning framework to solve these limitations

    본 논문에서는 딥러닝 학습 과정에서 고해상도의 측정 데이터 필요없이 저해상도-고해상도 측정 데이터 관계를 모델링 가능한 획기적인 딥러닝 프레임 워크를 제안한다. 제안된 방법론의 주된 장점은 딥러닝 모델이 학습 과정에서 오직 공간에 무작위로 spase하게 분포한 센서만을 이용하여 저해상도-고해상도 간 관계를 모델링 할 수 있다는 것이다. 해당 프레임 워크는 무작위로 분포한 데이터만을 활용하여 앞서 설명한 데이터 관련 문제를 해결하기 때문에 randomly seeded super-resolution GAN (RaSeedGAN)이라고 명칭되고, 제안하는 프레임 워크의 우수성을 검증하기 위해 3가지 예측 task (유체 유속 시뮬레이션, 지구 해수면 온도 및 유체 유속 측정 문제)를 채택한다. 자세한 딥러닝 구조에 대한 설명, 메커니즘 및 예측 결과들은 아래와 같이 설명하고자 한다.
    
### 2.1 Deep learning architecture for RaSeedGAN

<center><img src="/images_DS503/figure_5.jpeg" alt="example image" width="1000" height="500"></center>
<br>

        
<center>그림 5. 본 리뷰 논문에서 제안된 딥러닝 구조 세부사항</center>

<br>

    
**1. Main acrchitecture and idea** : GANs은 "generator" 그리고 "discriminator"라는 두 개의 서로 다른 neural network를 구성하고 있다."generator"는 참값 (target data)을 모방하여 인공적인 예측 값을 (generated data) 생성하는 network이고, "discriminator"는 인공적으로 생성된 예측 값과 참값 사이 다른 점을 구별하는 network이다. 본 연구에서는 이전 연구들의 한계인 학습 데이터에 high-resolution full fields 획득에 대한 제한점을 해결하는 generator를 제안하여 GAN을 이용하여 보다 우수한 생성 능력을 갖도록 학습시킨다 (그림5)
<br>   
    
**2. Generator network** : 본 연구에서 이용되는 generator의 입력과 출력은 기존 저해상도-고해상도 관계를 해결하는 딥러닝 연구들과 다르게 할당하여 해당 문제를 해결한다. 더욱 구체적으로, 고해상도의 모든 공간 데이터를 이용하는 것이 아닌 고해상도의 센서 데이터에서 무작위로 추출한 sparse한 데이터를 **generator 출력**으로 설정하고 (그림 5a), 무작위로 추출된 sparse한 고해상도 데이터 (그림 5b 왼쪽)로 부터 특정 직사각형 크기 픽셀마다 평형하게 이동하여 10개의 센서를 평균화 작업을 통해 **저해상도의 generator 입력 feature**를 생성한다 (그림 a 왼쪽). 즉, 고가 센서 장비를 sparse하게 분포시켜 데이터를 얻고, 이를 convolutions stride처럼 이동하여 평균 값을 산정하여 low-resolution5 입력을 얻는다. 이러한 접근은 잡음 뿐만 아니라 공간적 해상도에 대한 균일성을 증가하는 장점을 갖는다. generator는 기존 baseline neural network architecture [2]에 비해 변형된 구조를 갖는데 이는 아래와 같다.
<br>
    
    (1) Initial layer : 먼저 low-resolution fields가 generator로 입력되는데, filter size 9 × 9 그리고 64 feature maps을 갖는 convolutional layer가 이용되고, parametric rectified linear unit (ReLU) 활성화 함수가 convolutional operation에 의해 추출된 정보를 포착한다.
<br>
   
    (2) Medium layer : 초기 레이어에 의해 추출된 정보는 16개의 residual blocks [2] 을 통해 보다 높은 비선형적 관계에 대해 모델링한다. 이떄 residual block은 3x3 kernel을 동반한 64개의 feature map 을 이용한다. 여기서, 저해상도를 증가시키기 전에, skip-connection sum 레이어가 residual blocks의 출력 부분 그리고 initialization layers의 출력 부분 사이에 적용된다. 이 후애, [2]에 제안된 subpixel convolution layer 가 generator 출력 만큼 증가 시키기 위해 사용된다. 최종적으로 생성되는 spase-high resolution ouput은 비선형 보간 기법을 이용하여 sparse sample과 sample 사이 데이터를 공간적으로 보간하여 관심 있는 최종적 고해상도 데이터를 복구할 수 있다.
    <br>
<br>
    
**3. Discriminator network** : spase-high resolution target 그리고 generated fields 사이 예측 정확도를 향상시키 위해, 아래와 같은 discriminator network를 이용하여 generated fields를 입력으로 연결한다 (그림 5b). Discriminator network는 초기에 filter size 3 × 3 그리고 64 feature maps를 이용하여 중요 공간적 정보를 추출하고, 7개의 discriminator blocks이 연속적으로 적용된다. 이때, 홀수 block 마다 stride 사이즈를 높여 차원을 줄이고, 줄여진 feature-map tensor가 하나의 vector로 변환된다. 이때 1,024개의 fully connected layer를 이용하여 변환된 vector를 discriminator 출력 값으로 예측하고, 그 출력값이 sigmoid function을 통해 확률적으로 참 (0s)인지 거짓 (1s)인지 적대적으로 학습시켜 generator의 생성 능력을 향상시킨다. discriminator의 손실 함수 (loss function)는 다음과 같이 정의된다.
<br>
$$ \mathcal{L}_{\text{D}} = -\mathbb{E}[\log D(H_R)] - \mathbb{E}[\log(1 - D(F_v  \odot G(L_R)))] $$
    여기서 𝔼[]는 mini-batch 내 평균에 대한 연산자이고, H_R 및 L_R은 각각 고해상도 및 저해상도 이미지를 나타내며, D()는 저해상도 입력으로 부터 생성된 고해상도 이미지에 대한 loss를 계산하는 discriminator network를 나타낸다. F_v 는 하나의 연산자 계수로써 고해상도 이미지 내 센서가 존재할때 1 그렇지 않은 경우 0으로 변환하는 역할을 한다. 여기서, generator network의 loss function은 다음과 같이 정의된다.
$$ \mathcal{L}_{\text{G}} = -\sum_{i=1}^{Nx}\sum_{i=1}^{Nz}| H_R - F_v \odot G(L_R)_i,_j |^2 + \lambda  \mathcal{L}_A $$
    여기서 G()는 generator network가 실제 참값과 생성하는 sparse high-resolutional 값 사이 격자별 오차를 나타낸다.
$$ \mathcal{L}_A  =  -\mathbb{E}[\log D(F_v \odot G(L_R))]$$
    GANs을 학습시키기 위해, 위의 adversarial loss가 생성된 spase-high resolution field를 binary cross-entropy로 레이블링한다. 여기서, discriminator는 생성된 spase-high resolution field가 ‘fake’ 인지 진위 여부를 해당 1값으로 할당한 binary cross-entropy를 통해 결정한다.
    
### 2.2 Prediction results

    본 논문에서 제안하는 값비싼 고해상도 데이터를 학습과정에서 요구하지 않는 RaSeedGAN의 우수성을 검증하기 위해, 비선형성과 변동성을 포함하고 있는 3가지 실험 및 시뮬레이션 데이터를 이용한다. 우수성 검증을 위해, 아래 그림과 같이 (1) Pinball flow numerical simulation, (2) NOAA sea surface temperature database (3) particle-image velocimetry (PIV) experiment 와 같은 세 가지 예시에 대해 검증하고, 보다 자세한 numerical simulation 조건은 해당 리뷰에서 딥러닝 알고리즘에 대한 high-quality 정보를 집중하기 위해 생략한다. 세 가지 예시는 서로 다른 image 차원을 갖는다. (1),(2),(3)은 각각, 512 x 512, 720 × 1,440, 128 × 128 크기로 서로 다른 이미지 사이즈를 갖고 해당 예측 task에 마다 딥러닝 모델을 독립적으로 학습하여 예측 성능에 대해 검증한다. 또한, 서로 다른 task이지만 본 알고리즘의 일반화 능력을 검증하기 위해 무작위로 추출된 sparse한 고해상도 데이터로 부터 특정 직사각형 크기 픽셀마다 평형하게 이동하여 직사각형 공간 내 10개의 센서를 평균화하는 작업을 통해 32 x 32 bin size를 갖도록 input feature를 가공한다. 

    그림 6은 세 가지 예측 task에 대해 딥러닝 모델이 고해상도 데이터를 학습에 이용하지 않고, 저해상도-고해상도 사이 관계를 예측할 수 있는지에 대해 비교한 결과이다. 그림 6의 첫번째 열에는 평균화 작업을 거친 입력 저해상도 필드, 두번째 열은 "Sparse HR reference"라고 하는 특징화된 고해상도 필드, 세번째 그리고 네번째 열은 RaSeed GAN이 이를 예측하고 변환환 결과와 타겟 필드들을 나타낸다. 여기서 주목할 점은, 전체 고해상도 데이터는 딥러닝 모델 능력 테스트를 위해 그림 6의 마지막 열에 포함되어 있지만 훈련 중 직접적으로 사용되지는 않았다는 점이다. 그림 6의 각 task에서 알 수 있듯이, RaSeedGAN은 원기둥 주변과 원기둥 부근의 난류 발달 지역을 정확하게 복구할 수 있을 뿐만 아니라, 높은 수준의 디테일로 온도 실험 데이터를 복구할 수 있다.
    
#### Task 1: Pinball flow numerical simulation 
<center><img src="/images_DS503/figure6_1.jpeg" alt="example image" width="1000" height="500"></center>
    
#### Task 2: NOAA sea surface temperature database
<center><img src="/images_DS503/figure6_2.jpeg" alt="example image" width="1000" height="500"></center>
    
#### Task 3: particle-image velocimetry (PIV) experiment
<center><img src="/images_DS503/figure6_3.jpeg" alt="example image" width="1000" height="500"></center>
        
<center>그림 6. RaSeedGAN 예측 결과 비교</center>    


### 3 Conclusion
    
    본 연구에서는 무작위로 공간에 배치된 센서 데이터로부터 관심 있는 영역의 고해상도 센서 측정 결과들을 추정하기 위한 RaSeedGAN 프레임워크를 제안한다. 해당 프레임 워크의 가장 큰 장점이자 방법론적 접근 요약은 (1) 무작위로 공간에 배치된 센서 데이터들을 특정 구간 센서 개수마다 평균화하여 저해상도 입력 feature를 만드는 것. (2) 저해상도 입력 feature로 부터 듬성듬성 배치된 해당 센서 데이터를 복구하는 것. (3) 예측된 결과를 보간하여 다시 공간적으로 매우 세부적으로 기술된 온도/유속 고해상도 데이터를 얻는 것에 있다. 여기서 GAN은 듬성듬성 배치된 해당 센서 데이터들을 예측하는 generator에 대해 실제 결과와 매우 유사한 생성 field를 예측할 수 있게 했다. 특히, 본 연구는 유체 흐름 시뮬레이션, 해양 표면 온도 분포 측정 및 입자-이미지 속도측정 데이터에서 검증되었다. 이러한 프레임 워크는 고해상도의 데이터를 훈련 자체에 필요하지 않기 때문에 비용-효율적으로 많은 다양한 분야에 적용될 수 있을 것으로 생각한다.
    
### 4 Reference
    
[1]Adversarial super-resolution of climatological wind and solar data, ***Proceedings of the National Academy of Sciences of the United States of America***,July 6, 2020,117 (29) 16805-16815
https://doi.org/10.1073/pnas.1918964117
<br>
[2]Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, ***arXiv***:1609.0480
