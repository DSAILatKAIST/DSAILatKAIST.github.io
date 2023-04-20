---
title:  "[T-ITS 2021] Spatio-Temporal Knowledge Transfer for Urban Crowd Flow Prediction via Deep Attentive Adaptation Networks"
permalink: Spatio_Temporal_Knowledge_Transfer_for_Urban_Crowd_Flow_Prediction_via_Deep_Attentive_Adaptation_Networks.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [TITS-21]Spatio-Temporal_Knowledge_Transfer_for_Urban_Crowd_Flow_Prediction_via_Deep_Attentive_Adaptation_Networks

# 0. Overview

- Title : Spatio-Temporal Knowledge Transfer for Urban Crowd Flow Prediction via Deep Attentive Adaptation Networks
- Authors : Senzhang Wang, Hao Miao, Jiyue Li, Jiannong Cao
- Year : 2021
- Publish : TITS (IEEE Transactions on Intelligent Transportation Systems)

<aside>
💡 **How to transfer spatio-temporal knowledge well, between different two domains?**

</aside>

<aside>
💡 **We propose the ST-DAAN framework : ConvLSTM + DAN + Attention**

</aside>

# 1. Introduction

## 1) Why do we need it?

- Deep learning이 다양한 spatio-temporal(시공간) prediction task에 사용되고 있음
    - [ST-ResNet(2017, Cit. 1606)](https://ojs.aaai.org/index.php/AAAI/article/view/10735) : forecast crowds inflow & outflow in each region of a city
    - [STDN(2018, Cit. 521)](https://arxiv.org/abs/1803.01254) : road network based traffic prediction
    - predict passenger pickup/demand demands (Attention+ConvLSTM)
    - DeepTransport : predict the traffic data within a transport network (CNN+RNN)

<aside>
🥲 **하지만, 현실에서 시공간 정보는 그리 풍부하지 않음 → DL 쉽게 적용할 수 없음**

</aside>

<aside>
🥲 **더불어 앞서 언급한 모델들 = 다른 시공간 정보에도 적용할 만큼 General 하지 않음**

</aside>

- 최근에는 transfer learning을 사용해 상기 문제를 풀어보고자 했음
    - [RegionTrans(2019, Cit. 88)](https://arxiv.org/abs/1802.00386) : source, target city의 비슷한 지역을 매칭 → 이 작업 하려면 other service data가 또 필요 (data 관점 = region level)
    - [MetaST(2019, Cit. 166)](https://arxiv.org/abs/1901.08518) : 여러 도시의 장기적 추세를 뽑아내서 target city에 써보자 → 이걸 automatically 해주는 통합 모델은 없음

- 우리는 data 관점 = distribution 수정하고, unified framework를 만들어보겠다.

## 2) Related works & Core things

- Urban Crowd Flow Prediction : 도시/교통 분야의 큰 주제. 전통적으로는 ARIMA 같은 통계 based methods를 주로 사용했으나, 최근에는 DL methods가 많이 쓰이는 편
    - DNN, ST-ResNet, SeqST-GAN, ConvLSTM, MT-ASTN, DCRNN, RegionTrans, MetaST 등

- Transfer Learning : ML의 scarce labeled data problem을 해결하기 위해 제시된 방법론
    - TCA, TLDA, JAN, JMMD 등

- [DAN(2015, Cit. 4413)](https://arxiv.org/abs/1502.02791) : CNN을 domain adaptation task에 맞게 일반화, 컴퓨터 비전 분야에서 큰 성공
    - Neural Net이 general feature 잘 잡아내고 성능 좋다만, labeled data 별로 없는 target domain에 바로 CNN 쓰니 문제가 많음
    - 실제로 [Yosinski et al.(2014, Cit. 8740)](https://arxiv.org/abs/1411.1792) 보니 Conv 1-3까진 OK, Conv 4-5부터 이상해지더니, FC 6-8에선 완전히 메롱
    - DAN 저자들은 Conv 1-3은 그대로 두고(freeze), Conv 4-5 단계에 fine-tuning 적용, FC 6-8은 CNN parameter optimizing에 multi-kernel MMD를 regularizer로 넣는 식으로 개선
        - [Sejdinovic et al.(2013, Cit. 610)](https://arxiv.org/abs/1207.6076) : two samples의 distribution이 같은지 평가할 만한 통계량으로 MMD(Maximum Mean Discrepancies)를 제시한 바 있음
    - 요약하면 CNN parameter를 찾되, FC-layers 단에서 만들어지는 source와 target의 hidden representation이 비슷해지도록 추가 제한을 설정한 것

- [ConvLSTM(2015, Cit. 6876)](https://arxiv.org/abs/1506.04214) : 기존 Fully Connected LSTM은 1차원 time-series → 공간정보(row, column)을 넣어서 3차원 데이터를 다루도록 확장
    - 홍콩 기상청에서 radar echo images로 강수 예보를 하려니, 기존 LSTM으론 공간성을 담아낼 수 없어선지 성능이 안 좋더라 → image를 LSTM에 넣기 전 CNN으로 초벌구이하는 방식을 제안

## 3) Formulationss

- Spatio-Temporal Data : 2차원 공간 상에서 기록되는, 시간에 따라 변하는 feature를 말한다. 따라서 단일 feature라면 기본적으로 3차원 데이터.
- 본 논문에서는 서로 다른 지역에서 만들어진 데이터를 다루며, 이들을 같은 수의 grid cell로 나눠 작업한다.
    - 서울, 대전, 뉴욕, … 도시의 크기/형태는 제각각이지만 cell 수가 같도록 격자를 만들어준다.
    
    ![데이터가 cover하는 공간을 m*n개의 grid cell로 나눈다. each cell region이 t시점에 갖는 정보(교통량, 강수 등)가 있을 텐데, 이들이 어떤 값을 갖는지 표현한 게 spatio-temporal image (matrix)라 한다.](https://user-images.githubusercontent.com/67723054/233354355-c106f23c-6012-48d2-8204-c7e78d49f7cd.jpg)
    
    데이터가 cover하는 공간을 m*n개의 grid cell로 나눈다. each cell region이 t시점에 갖는 정보(교통량, 강수 등)가 있을 텐데, 이들이 어떤 값을 갖는지 표현한 게 spatio-temporal image (matrix)라 한다.
    
- 격자 형태 matrix를 image라 할 때, 매 시점마다 기록된 image들의 time-series를 모으면 3차원 tensor가 된다.
    - 서울의 따릉이 통행량(a feature)을 열두 시간쯤 관찰했다면, 해당 데이터는 아래와 같은 spatio-temporal tensor로 묘사할 수 있겠다.
    
    ![image는 시간에 따라 변하며, t시점 기준으로 과거 k개 image를 축적하면, 위와 같은 3차원 tensor를 얻을 수 있다. 이 tensor가 앞으로 전개할 논리의 기본 단위로 자주 쓰인다.](https://user-images.githubusercontent.com/67723054/233354358-d21b52f2-a3bc-4403-98c5-fd7aeaa93a10.jpg)
    
    image는 시간에 따라 변하며, t시점 기준으로 과거 k개 image를 축적하면, 위와 같은 3차원 tensor를 얻을 수 있다. 이 tensor가 앞으로 전개할 논리의 기본 단위로 자주 쓰인다.
    
- tensor들은 최상단(latest) image를 기준으로 추려낸 최근 k개 images인 셈인데, 이 같은 뭉치를 1-step after 마다 계속 뽑아낸다면, 해당 tensors로 어떤 4차원 리스트를 만들 수 있겠다.
    - List with parameters : Row(m) * Column(n) * Accumulation(k) * Time-stamp(t)
    - 이 리스트를 tensor set, 길이를 ‘L’이라 하자.
    - 데이터가 많은(장기간) domain에서는 집합이 길쭉하게, 반대로 데이터가 부족한 domain에서는 짤막한 집합이 나온다.
    
    ![tensor는 정보를 의미하며, domain에 따라 정보량은 다를 테다. 예컨대 여기선 서울의 택시 승객 데이터는 나흘(최종 업데이트 기준) 정도로 길지만, 따릉이 통행량 데이터는 기껏해야 반나절쯤 돼서, 다른 domain인 택시 정보를 어떻게 잘 가져올 수 있을까 고민하게 된다. 그게 이 논문의 핵심 주제.](https://user-images.githubusercontent.com/67723054/233354364-0c50754c-04c4-4625-92a9-8dd41f75118b.jpg)
    
    tensor는 정보를 의미하며, domain에 따라 정보량은 다를 테다. 예컨대 여기선 서울의 택시 승객 데이터는 나흘(최종 업데이트 기준) 정도로 길지만, 따릉이 통행량 데이터는 기껏해야 반나절쯤 돼서, 다른 domain인 택시 정보를 어떻게 잘 가져올 수 있을까 고민하게 된다. 그게 이 논문의 핵심 주제.
    

# 2. Main Architecture

- 기본적인 특징은 stacked ConvLSTM 으로 잡아내며, 만들어진 hidden state에 DAN(generalized CNN), 마지막엔 Global Attention 적용 & 기타 features 추가하는 구성이다

![논문의 main figure. 크게 1) ConvLSTM, 2) CNN with MMD (DAN), 3) Global spatial attention 구간으로 나뉜다.](https://user-images.githubusercontent.com/67723054/233354374-0e4af3ed-40d4-4893-afe7-c0818881f20c.jpg)

논문의 main figure. 크게 1) ConvLSTM, 2) CNN with MMD (DAN), 3) Global spatial attention 구간으로 나뉜다.

## 1) Representaion Learning (ConvLSTM)

![convLSTM(CNN+LSTM) 과정을 거쳐 spatio-temporal image tensor set이 4차원 hidden tensor set ‘H’로 변한다. H는 이후 3D Convolution with MMD을 통과해 feature tensor set ‘F’가 된다. 파란색, 살구색 tensor의 경우 CNN을 거쳐 나오는 차원의 수가 불명확해 ?로 적어두었다. (최종 output인 F에선 다시 3*4*12로 맞춰지는 듯하다.)](https://user-images.githubusercontent.com/67723054/233354368-a5edfec0-af04-4a55-9c56-b00429ccf303.jpg)

convLSTM(CNN+LSTM) 과정을 거쳐 spatio-temporal image tensor set이 4차원 hidden tensor set ‘H’로 변한다. H는 이후 3D Convolution with MMD을 통과해 feature tensor set ‘F’가 된다. 파란색, 살구색 tensor의 경우 CNN을 거쳐 나오는 차원의 수가 불명확해 ?로 적어두었다. (최종 output인 F에선 다시 3*4*12로 맞춰지는 듯하다.)

- Input = Tensor set(4D) 이지만, 작업은 매 image(2D) 마다 진행 → 한 장씩 CNN을 거쳐 새로운 tensor set을 만들어 낼 수 있음 → 다시 LSTM의 Input gate에 투입 + 이전 hidden state tensor set과 결합 + … (마찬가지로 2D 단위로 진행) → 반복
- 모든 stacked LSTM을 통과해 만들어진 최종 결과물을 ‘H’라 하자

## 2) Knowledge Transfer (DAN)

- two different domains’ distributions이 얼마나 다른지, distance로 평가한 것을 MMD라 한다.
- 도메인 별로 hidden state에 CNN을 적용하되, CNN layer 마다 mmd loss를 산출해 평균을 낸다.
- Parameter set **Θ** = argmin Loss Function of (GT vs ConvLSTM & CNN & mmd_loss & … )

## 3) Global Spatial Attention

- local spatial correlations는 CNN 단계에서 잡히지만, 보다 넓은 범위에서 geographical dependencies는 잘 포착되지 않는다.
    - 지리상으로는 멀리 떨어진 두 지역이 유사한 Point of Interest distribution을 가지는 경우가 많다
    - 이는 taxi-trip, crowd flow 같은 시공간 정보도 마찬가지
- source domain 데이터를 활용할 때, attention score를 곱해서 가져오면 global relation을 체크하는 효과를 낼 수 있지 않을까

![아침 홍대의 택시 승객(source)은, 같은 시각 홍대와 노원의 자전거 통행량(target)과 닮아있다. domain은 다르지만, ‘출퇴근/통학’ 이라는 요소가 저변에 깔려있음을 attention mechanism을 통해 파악하는 셈. 성수는 노원보다 홍대에 가까이 있지만, 주거/업무/학군 보단 ‘문화예술’ 지역이라 아침에 자전거 타는 사람이 적다고 해석할 수 있겠다.](https://user-images.githubusercontent.com/67723054/233354371-07961d2f-8a3e-4941-b542-c7b4a2d25b23.jpg)

아침 홍대의 택시 승객(source)은, 같은 시각 홍대와 노원의 자전거 통행량(target)과 닮아있다. domain은 다르지만, ‘출퇴근/통학’ 이라는 요소가 저변에 깔려있음을 attention mechanism을 통해 파악하는 셈. 성수는 노원보다 홍대에 가까이 있지만, 주거/업무/학군 보단 ‘문화예술’ 지역이라 아침에 자전거 타는 사람이 적다고 해석할 수 있겠다.

- 구체적으로는 source domain의 2D image의 특정 부분 Region (i, j)가, target domain의 모든 m*n개 region과 얼마나 닮아있는지 체크한다
    - 본 논문에서 다루는 image는 모두 같은 m*n 사이즈 grid cell로 나눠져 있으니 행렬 계산이 용이하다.
    - dot-product, softmax 취해서 attention matrix 만드는 등 널리 알려진 attention mechanism과 크게 다른 점은 보이지 않았다

# 3. Modeling

<aside>
😞 **아직 이해하지 못해서, 다음 Review에서 다뤄볼까 생각 중입니다**

</aside>

## 1) Algorithm

![algo 1.jpg](https://user-images.githubusercontent.com/67723054/233354356-aaeed10f-eb7a-40fd-83df-02f213efb054.jpg)

## 2) Real Code

[https://github.com/MiaoHaoSunny/ST-DAAN](https://github.com/MiaoHaoSunny/ST-DAAN)

# 4. Evaluation

<aside>
🤷‍♂️ **ST-DAAN is good enough?**

</aside>

<aside>
🤷‍♂️ **Global Spatial Attention → Performance**

</aside>

<aside>
🤷‍♂️ **Amount of available data in Target & Source domain → Performance**

</aside>

<aside>
🤷‍♂️ **Sensitivity to model structure & parameters**

</aside>

- 과거 Taxi, Bike 데이터로 Crowd flow prediction 하는 task로 ST-DAAN 성능을 평가해보자

![여러 도시에서 수집된 taxi, bike 데이터셋으로, 각각 GPS 경로, 출발/도착지, 시각, ID 등 다양한 variables로 구성돼있다. number of trips, time span을 비교하면 DIDI는 같은 택시 데이터셋인 TaxiNYC보다 data scarce 하다고 볼 수 있다.](https://user-images.githubusercontent.com/67723054/233354337-985678a7-39e6-4abb-9b23-525748e55d12.jpg)

여러 도시에서 수집된 taxi, bike 데이터셋으로, 각각 GPS 경로, 출발/도착지, 시각, ID 등 다양한 variables로 구성돼있다. number of trips, time span을 비교하면 DIDI는 같은 택시 데이터셋인 TaxiNYC보다 data scarce 하다고 볼 수 있다.

- Intra-city(TaxiNYC → BikeNYC), Cross-city(BikeChicago → BikeNYC, DIDI → TaxiBJ) transfer case를 모두 다뤄보았다
- Baseline model은 non-transfer learning, 최근의 transfer leaning based에서 고루 골랐다
    - non-transfer learning based : ARIMA, ConvLSTM, DCRNN, DeepST, ST-ResNet
    - transfer learning based : (위 모델들에 fine-tuning), RegionTrans, MetaST

## 1) Comparison With Baselines

- ARIMA < non-transfer < non-transfer with fine-tuning < transfer < ST-DAAN 순으로 성능 Good
    - ST-DAAN full version과 Attention & External features을 각각 빼본 variation을 비교해보니, 이들 역시 성능 향상에 도움이 됐음
        
        ![Intra-city, Cross-city 무관하게 ST-DAAN이 좋은 성능을 보임. nonAtt, nonExt는 각각 global spatial attention, inserting external feature을 없앤 버전의 ST-DAAN](https://user-images.githubusercontent.com/67723054/233354343-d00945bc-988a-4d10-814a-54d5daf71861.jpg)
        
        Intra-city, Cross-city 무관하게 ST-DAAN이 좋은 성능을 보임. nonAtt, nonExt는 각각 global spatial attention, inserting external feature을 없앤 버전의 ST-DAAN
        

## 2) Effect of Data Amount

- 데이터가 많을 수록 좋긴 하더라. Source/Target 둘 다 데이터가 많으면 성능 좋음

![대체로 데이터 length 길수록 예측 성능이 좋아짐. 역시 다다익선](https://user-images.githubusercontent.com/67723054/233356050-9f85199f-d270-4f08-a353-48a055454b34.PNG)

대체로 데이터 length 길수록 예측 성능이 좋아짐. 역시 다다익선

## 3) Parameter Sensitivity Analysis

- Scarce data 다루는 transfer learning, 신경망 깊게 쌓으면 오히려 overfitting 문제가 발생
- Domain discrepancy에 적당한 penalty 줘야 함. 작게 주면 common knowledge가 전달되지 않고, 너무 크게 주면 only domain-specific feature만 전달됨

![ConvLSTM, CNN 단계에서 number of layers 너무 많으면 문제, penalty hyper-parameter gamma도 적당히 설정할 필요](https://user-images.githubusercontent.com/67723054/233354353-80c061bd-935e-44f6-81a1-b2835f658aa7.jpg)

ConvLSTM, CNN 단계에서 number of layers 너무 많으면 문제, penalty hyper-parameter gamma도 적당히 설정할 필요

# 5. Others

- TaxiBJ의 crowd flows를 RegionTrans, ST-DAAN으로 예측해보았는데, 택시 많이 잡는 Rush hour에서 ST-DAAN이 RegionTrans 대비 우수 → 본 모델을 이해하는 데 도움될 만한 직관적 예시?
    - 기존 모델은 time invariant, 특질을 제대로 구분하지 못하지만, ST-DAAN은 일정 부분 GT에 다가서는 모습을 보였다는 식으로 이해함

![택시 많이 안 잡는 심야 시각에는 RegionTrans, ST-DAAN 둘 다 비슷하지만, Rush hour에선 꽤 비슷하게 capture](https://user-images.githubusercontent.com/67723054/233354351-b35fb7c7-ded5-4a75-9e53-31e43cb7e7ea.jpg)

택시 많이 안 잡는 심야 시각에는 RegionTrans, ST-DAAN 둘 다 비슷하지만, Rush hour에선 꽤 비슷하게 capture

