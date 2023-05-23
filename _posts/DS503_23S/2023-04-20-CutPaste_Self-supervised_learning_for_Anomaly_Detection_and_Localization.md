---
title:  "[CVPR 2021] CutPaste : Self-Supervised learning for Anomaly Detection and Localization"
permalink: CutPaste_Self-supervised_learning_for_Anomaly_Detection_and_Localization.html
tags: [reviews]
use_math: true
usemathjax: true
---

## CutPaste : self-Supervised learning for Anomaly Detection and Localization

### Introduction

- 전형적인 supervised classification 문제와는 달리, 이상치 탐색은 추가적인 어려움이 있다. 
  - 1) 이상치의 특성 상, Labeling 유무와 관련없이 다량의 이상치 데이터를 확보하기가 어렵다.

  - 2) 정상 / 이상치 데이터간의 차이는 종종 fine-grained 하여, 고해상도 이미지에서도 식별하기가 어렵다  

<br>

- 이상치 데이터(anomalous data)을 얻기 힘들기 때문에, 기존의 이상치 탐색 방법은 semi-supervised  또는 one-class classification 환경에서 정상 데이터만을 활용하곤 한다. 
  - 이때 주로 사용되는 방법은 Autoencoder,  Generative Model 등을 통해 복원오류, 확률 분포를 기반으로 이상치를 탐색한다. 

  - 하지만, 이러한 방법들은 **픽셀 단위의 오류, 또는 고차원의 의미론적 정보(semantic information)를 포착하지 못한다**.

<br>

- 하나의 대안으로써, high-level learned representation을 활용하는 방안이 효율적이었다. 

  - EX)- Deep one-class classifier : End-to-End 구조로 연결된 DL 모델을 통해 Representation vector 추출 

  - Ex)- Self-supervised representation learning : 위치 / 회전 등 변화를 인식하는 Task 추가

  - 하지만, 대부분의 기존의 연구들은 객체 중심(object-centric)의 자연스러운 이미지로부터 의미론적 이상치(ex- visual object from different classes)을 찾는데 초점을 두었을 뿐, **이미지의 일부만 다른 fine-grained 이상치를 탐색하는데에는 한계가 있다.**

- 따라서 본 연구는 **고해상도 이미지에 fine-grained 한 다양한 이상치 패턴을 식별하는 One-classification 기반의 이상치 탐색 방법을 제안**한다. 

---

### Method

- **2단계 프레임 워크 제안**
  - Jihoon {54} 의 연구에 따라, 1) 정상 데이터로부터 Representation 을 추출할 DL 모델, 2) Representation을 Input으로 받는 self-supervised learning(One-classifier) 구조를 채택한다.    

    - {54} - Csi : Novelty detection via contrative learning on distributionally shifted instances(2020). 

  - 비정상 데이터는 확보하기 어렵기 때문에 1) 단계에서는 정상 데이터만을 활용하여 deep representation을 학습할 것이다. 

    - 예시) Autoencoder 와 유사함 

  - 이후 2) 단계에서 실제 비정상 데이터와 유사한 형태를 띄는 새로운 Data augmentation인 Cut Paste 방법을 적용함으로써 부족한 비정상 데이터를 확보한다. 그리고 Cut paste 유무를 식별하는 Self-supervised learning 방법을 적용함으로써 성능 향상을 꾀한다.   

<br>

##### 1. Self-supervised learning

- **Self-supervised representation learning 간 좋은 pretest task를 정의하는 것이 필수이다.**

  - 기존 연구에서 sematic one-class classification의 맥락에서 Rotation prediction / Constrastive learning이 유명한 방법으로 알려졌으나, 일부만 다른 local 이상치를 탐색하는데 최적의 방안은 아니다.

    - 본 연구에서 위 방안들은 semantic concept의 Representation을 학습하기에 적합하나,

    - 이미지 내부의 연속성, 반복 구조에 대한 Representation을 학습하기엔 적절치 않다고 판단한다. 

  - 본 연구에서는 **local irregular pattern에 대한 representation을 학습할 수 있는 Data augmentation 방안을 제안**한다. 

- **Local irrugularity을 생성하는 Data augmentation 방안을 참고하여, 이미지가 변형된 discriminative 부분을 포착하도록 Task를 부여한다.** 

  <img width="645" alt="2" src="https://user-images.githubusercontent.com/16533475/233392403-60c9a594-a91d-462d-84c7-492b1e34c3d7.png">

  - 기존 방안들에서 주로 사용되는 Cutout은 랜덤한 크기의 이미지 영역을 지웠을 때에도 동일하게 Prediction 하도록 만든다.  

    - Invariancce를 향상시킴으로써 Multi-class classification에서의 예측 정확도를 향상시킨다. 

  - 본 연구에서는 반대로 접근하여, 데이터가 변형된 부분을 찾도록 Task를 부여한다.

    - 데이터를 빈칸으로 채울 경우, 모델이 data augmented 된 부분에 대해 Naive decision rule을 학습할 수 있다. 

    - 따라서 좀 더 Task를 어렵게 부여함으로써 모델이 irregular Point을 잘 찾으면서, decision rule을 잘 세우도록 할 것이다.  

<br>

- 본 연구에서는 CutPaste라는 새로운 Data augmentation 방법을 제안한다. 

  <img width="643" alt="1" src="https://user-images.githubusercontent.com/16533475/233392525-a394cbf6-bce2-4760-abeb-9f22595ea2e8.png">
  - 1) 정상 데이터로부터 다양한 크기와 비율의 직사각형 이미지(Patch)를 잘라낸다. 

  - 2) (선택)자른 이미지를 회전하거나, 색을 바꾼다. 

  - 3) 가공한 이미지를 정상 데이터의 아무 위치에 붙여넣는다.  

<br>

- self-supervised representation learning 의 목표을 달성하고자 학습 목표는 다음과 같이 정의한다. 

  <img width="269" alt="3" src="https://user-images.githubusercontent.com/16533475/233392572-f71e8236-7d92-4bb0-943c-52d8b2e62d8b.png">
  - $X :$ set of normal data

  - $CP(.)$ : CutPaste augmentation 

  - $g$ : binary classfier parametrerized by deep networks 

  - $CE(., .$) : Cross entropy loss

<br>

##### 2. Data augmentation - CutPaste 방식

- **CutPaste는 다양한 형태로 적용할 수 있다.**
  - 1) CutPaste -Scar  

    - Cutout 방법 중에서 긴 직사각형 형태의 random 색상으로 채우는 'Scar' 방식이 있음

    - 위 방식을 채택하여, CutPaste 또한 긴 직사각형 형태의 이미지를 덧붙이는 방법 

  - 2) Multi class classification 

    - 일반적인 직사각형 형태의 이미지 Patch와, Scar 형태의 이미지는 서로 각각 장단점을 가지고 있음

    - 각각의 장점을 활용하기 위해서 정상 / 일반 Cut Paste / Scar Cut Paste 로 클래스를 구분함  

  - 3) Similarity between CutPaste and real defects 

    - CutPaste 방법을 통한 성능 향상은  pseudo 이상치를 생성 관점에서 해석될 수 있다. 

    - CutPaste는 정상 데이터의 local structure을 보다 많이 보유하고 있는 예제를 만들어 낸다. 

      - 이로 인해 모델은 아예 클래스가 다른 경우보다, 더 어렵게 이상치 탐지를 해야한다. 

    - 또한 CutPaste 방법은 그 자체로 이상치와 유사성을 가지고 있다. 이는 t-SNE을 통해 가시화할 수 있다.

      - <img width="648" alt="4" src="https://user-images.githubusercontent.com/16533475/233392615-58ec48c1-55c1-4100-9f06-679bb08e95c1.png">

##### 3. Computing Anomaly Score

- 이상치 점수(Anomaly score)을 구하는 방법에는 다양한 방법이 있다.

- 본 연구에서는 Kernel density 와 Gaussian density estimator을 비교할 것이다. 

##### 4. Localization with Patch Representation

- 본 연구는 이미지의 전체 representation을 학습하기 위한 방법을 제한하나, 

- 각 이미지 영역(Patch) 별로 representation을 구해 defective region을 찾아내는 데 사용할 수 있다.

---

### Experiment

- Dataset : MVTec anomaly detection dataset

  - 용도 : 이상치 탐색 모델을 점검 - 정상 / 비정상 클래스 이미지로 구분함 

  - 종류 : 총 15개 카테고리의 데이터셋 제공(물품 10개, 텍스처 5개)

  - 학습 데이터 크기 : 카테고리에 따라 60개 ~ 391개 보유 (상대적으로 적은 수의 학습 데이터)  

  - 특징 : 이미지 크기는 256 x 256 pixel 임

- Baseline : one-class classification 방법 채택

  - 각 카테고리별 정상 데이터만을 활용하여 one-class classifier을 학습함 

  - CutPaste 방식을 통해 data Augmentation을 적용하였음 

  - ResNet-18 모델 구조 아래에 pretrained 없이(scratch) 학습함

  - 이상치 정도는 ResNet-18 마지막 층에 한 층을 추가하여 Gaussian density estimation을 진행함

  - Hyperparameter과 data augmentation 방법은 항상 통일하여 실험을 진행함. 

- Evaluation Metric 

  - 각 카테고리 별로 random seed를 적용하여 5번씩 실험을 한후, 평균 AUC와 Standard error을 계산함

  - Self-supervised learning 관점에서, Rotation / Cutout / CutPaste 각각의 Task의 성능을 확인함 

  - 기존의 one-classification 방법론(DOCC, Uniformed student, patch SVDD)들의 방법들 간의 성능을 비교함

##### Result

- **1) Main result**
  - <img width="648" alt="5" src="https://user-images.githubusercontent.com/16533475/233392691-6520deef-5d92-47e6-8ef6-826f77057b30.png">

  - Rotation 방법은 Segmentic abnormal detection(ex- 클래스가 다른 경우)에서는 유용했으나, fine-grained 한 이상치에는 효과가 크게 없음을 확인할 수 있다. 

    - 한편으로 rotation 방법은 이미지의 방향이 정해진 'toothbrush', 'bottle' 데이터에서는 좋은 성과를 내었으나, 물체의 방향이 랜덤한 다른 데이터셋(screw)에서는 성능 악화가 식별되었다.

  - 반면 CutPaste 방식은 대부분의 경우 뛰어난 성과를 내었다. 

    - 그 중에서 3 classification 방법으로 2가지 CutPaste 방법의 장점을 채택한 방법이 가장 성능이 좋았다. 

<br>

- **2) Defect Localization**

  <img width="640" alt="6" src="https://user-images.githubusercontent.com/16533475/233392751-c579d595-d9d9-4344-a29d-3fc075594b5c.png">

  > GT Mask : Ground truth anomal annotation 
  > 
  > > - 픽셀별로 Abnormal detection score을 계산한 다음, GradCAM 방법을 적용하여 defective 영역을 시각화하였다.  

  - 이때 Grad Cam은 256 x 256 크기의 이미지를 input으로 하였을 때 결과이며, Patch heatmap은 256 x 256 이미지로부터 64 x 64 크기의 Patch을 추출한 다음 모델을 학습한 결과이다. 

    - Patch 로 적용했을 때 defective area를 잘 잡아내는 것을 확인할 수 있다.  

  - Pixel-wise localizatoin에 대해 기존의 방법들(FCDD, P-SVDD)과 비교했을 때, CutPaste 방식이 대부분의 경우 성능이 좋았다. 

    - <img width="313" alt="7" src="https://user-images.githubusercontent.com/16533475/233392811-e1c46459-2da5-4d88-bcf0-4e9944dd9195.png">

<br>

- **3) Transfer learning with Pretrained models** 
  - 기존의 One-classification 방법(DOCC, Uninformed student) 연구에서 제시했던 데로, pretrained 모델을 사용하는 것이 보다 성능이 뛰어났다. 
    - <img width="322" alt="8" src="https://user-images.githubusercontent.com/16533475/233392832-e811194b-727a-4bcf-935b-def0b28c2c38.png">

<br>

##### 4. Ablation study

- 1) From Cutout to CutPaste
  - Cut Paste 방법은 Cutout 방식과 다르게 다양한 영역에 적용할 수 있다.
    - <img width="319" alt="9" src="https://user-images.githubusercontent.com/16533475/233392836-77e4cccc-3797-4045-836f-5e4b2777532d.png">

    - CutPaste 방식은 기존 이미지의 Local structure 와 정보를 가지고 있어, 데이터셋과 별개로 적용할 수 있는 여지가 크다.

      - 기존의 Cutout 방법에서는 삽입하는 부분은 원본 데이터의 Local 정보를 담고 있지 못하다. 

      - 즉, 합성된 데이터는 실제 비정상 데이터와는 차이가 커, 부족한 비정상 데이터를 보완하기란 제한된다. 

      - 반면, Cut paste는 원본 데이터의 일부를 가져와 붙히기 때문에, 실제 비정상 데이터와 큰 차이가 없게 된다. 

<br>

- 2) Binary v.s. Finer-Grained Classification 
  - Task에 대해 Labeling 할 때 크게 2가지 방법으로 나눠볼 수 있다.  
    - 1) Binary Classification : 정상 vs 생성 데이터(CutPaste + CutPaste-scar)

    - 2) 3 way classification : 정상 vs Cutpatse vs CutPaste-scar
  - 실험 결과 3 way classfiication으로 Labeling 했을 때 성능이 더 좋았음을 확인할 수 있었다. 

<br>

- 3) CutPaste on Synthetic anomaly detection 
  - Synthetic 데이터셋에서도 CutPaste 방식이 성능이 좋은 지 실험하였다. 

  - MNIST 데이터셋에 다가 사각형, 타원, 하트, 실제 이미지 등 다양한 모양을 다양한 색깔로 붙여 이상치 데이터를 만들었다. 

  - 이후 CutOut / CutPaste 간의 성능을 비교하였을 때, CutPaste 방식의 성능이 항상 뛰어났다. 

    - <img width="310" alt="10" src="https://user-images.githubusercontent.com/16533475/233392828-f5c367e5-5982-4cfe-aed4-062c71d0397b.png">

    - 이외에 CIFAR-10에도 적용했을 때에도 CutPaste(60.2 < 69.2 AUC)가 더욱 뛰어난 성능을 가졌다. 

    - 하지만 rotation 기반의 Data augmentation을 적용했을 때(91.3 AUC)에 비해 성능이 좋지 않았다. 

      - 이는 semantic anomaly detection와 defect detection 간의 차이로 보인다. 

<br>

### Conclusion

- 본 연구는 Data-driven 관점에서 이상치 탐색 및 Localization을 할 수 있는 방안을 제안한다. 

- 주요 성과는 "CutPaste" 와 병행한 Self-supervised learning 방안을 제안한 것이다. 

  - 본 방식은 Local irregularity을 포착하는데 효과가 좋다. 

- 본 방법은 실제 데이터 사이에서 이미지 단위의 이상치를 탐색하는데 있어 뛰어난 성과를 보였으며, 이미지 Patch 단위에서 Pixel-wise 이상치 localization 성능에 있어 SOTA를 기록했다



- 본 연구는 ''실제 비정상 데이터와 유사한'' data augmentation 방법을 고안했다는 데 기여점이 있다. 

  - 그럼 이 아이디어를 확장하여 의료계 데이터와 같이 고유의 데이터 특성을 가진 분야에도 적용할 수 있을 것이다.  

  - 즉, 실제 비정상 데이터와 유사한 data augmentation 방법을 새롭게 고안할 수 있다면, 추가 기여를 쌓을 수 있을 것으로 보인다.





### Author Information

- Author name : Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister
  - Affiliation : Google Cloud AI Research 
