---
title:  "[RecSys 2023] Trending Now: Modeling Trend Recommendations"
permalink: 2023-10-16-Trending_Now_Modeling_Trend_Recommendations.html
tags: [reviews]
use_math: true
usemathjax: true
---

**[Recsys 2023] Trending Now: Modeling Trend Recommendations**

<br> 

> **Info**

▢   **Authors** : AWS AI Labs, Amazon USA

▢   **Research topic** : Trending now (현재 인기있는 상품을 추천하는 추천시스템) 

▢   [paper links](https://dl.acm.org/doi/10.1145/3604915.3608810) 


<br>

> **Research Motivation & 논문 선정 이유** 

최근 추천시스템에서는 일반적으로 "현재 인기있는 상품 (trending now)" 과 같은 별도의 추천 항목을 제시해주어 해당 상품의 인기를 높여 활성유저를 유치하고 있습니다. 그러나 일반적으로 "시간 간격 내 interaction 수" 와 같은 단순한 휴리스틱 방법을 기반으로 추천해주기 때문에 rich-get-richer (인기있는 상품만 더 노출되어 더 많은 인기를 얻는) 문제 등 개선의 여지가 많이 남아있으며, 추천시스템에서 trend 를 모델링하는 연구는 제한적으로 이루어져 있습니다. 따라서 해당 논문은 **시계열 예측** 이라는 새로운 관점에서 **trend 를 반영한 추천시스템 모델을 제안**합니다. item trendiness 에 대한 정의를 통해 trend recommendation task 를 **one-step time series forecasting** 문제로 공식화합니다. item 의 미래 trend 를 예측하고 추천 리스트를 생성하는 deep latent variable 모델인 **TrendRec** 을 제안합니다.
현재 시점에서의 trend 를 파악하고 유저에게 관련 item 을 추천하는 것을 시계열 예측으로 접근한 관점이 신선했고, "acceleration" 이라는 개념을 정의하여 인기가 "빠르게 상승" 하고 있고 가까운 미래에 인기를 얻을 가능성이 있는 item 을 추천한다는 아이디어가 참신한 것 같아 해당 논문을 선정하게 되었습니다. 


<br>
<br>

### 1. Introudction 

#### 1-①. Definition 

##### ▶ 용어정의 

|용어|내용| 
|------|---| 
|**Popularity**|특정 시간 간격 내에 발생한 interaction 수 | 
|**trend in the recommendation context** |인기도 (popularity) 또는 가속도 (acceleration)의 **변화율**| 
|**trending now**|현재 시점에서 점점 더 많은 interaction 이 발생하는 item list 이지만, 해당 item 들은 반드시 가장 인기있는 item 을 뜻하진 않는다. 아직 인기가 높지 않은 유망한 trending up item 에 대한 탐색이 가능하므로 popularity bias 없이 효과적으로 item 을 추천한다.| 




##### ▶ trending now 의 target item 예시 


|Example|
|------| 
|•  최근 출시된 좋은 품질의 cold items (ex. 왕좌의 게임 새로운 에피소드) = Recently released cold items of good quality|
|•  갑작스러운 변화가 발생하는 항목 (ex. 영화가 오스카상을 수상하여 갑자기 유행하는 경우) = Items experiencing sudden changes|
|•  주기적으로 유행하는 오래 지속되는 품목 (ex. 겨울의류와 같이 계절적인 영향을 받는 품목) = Long lasting items with periodic up-trend |


<br>

#### 1-②. Challenge 
 
▢  **Problem**: 현재 trend 를 안정적이고 신뢰성 있게 파악하기 위해서는 충분한 interaction 을 수집하는 데 일정 시간이 필요하나, trend 는 정의 특성상 역동적으로 변화하고 데이터 수집 기간 동안 변동이 있을 수 있다. 

▢  **Solution** : trend recommendation 을 one-step forecasting problem 으로 공식화한다. 

<br>

#### 1-③. Problem setting : One-step forecasting problem 

![image](https://github.com/hopebii/kaist_ds535/blob/main/fig1.png)

▢ (왼쪽 그래프) 과거 item 의 trend 변화가 주어지면, **다음 시간 단계에 어떤 item 이 유행할지 예측 (→ One-step forecasting)** 하는 것을 목표로 한다. 모델이 다음 시간 단계에서 유행하는 item 을 예측하면, 백엔드에서 데이터를 버퍼링하면서 다음 시간 단계 내 user 에게 해당 item 을 표시한다. 다음 시간 단계가 끝나면 모델은 새로 축적된 데이터를 기반으로 바로 다음 시간 단계에 대한 새로운 예측을 수행한다. 또 다른 데이터 수집의 주기를 시작하는 것으로 추천을 반복한다. 

▢ (오른쪽 그래프) 특정 use case 에 대한 최적화된 트렌드 추천을 설계하려면 기존의 시계열 예측 모델을 기반으로 구축할 수 있다. 더불어 시계열 예측 모델 외에도 추천 상황의 고유한 속성이 활용될 수 있다. 시계열에서 각 item 에 대한 대략적인 누적 interaction 수 외에도 더 세분화된 user-item 간 interaction 이 존재한다. 특정 item 에 대해 얼마나 많은 user 가 해당 item 과 상호작용 했는지 알 수 있을 뿐 아니라 이러한 user 가 정확히 누구인지도 알 수 있다. 이는 item 간 근본적인 상관관계를 파악하는 데 도움이 되는 추가적인 정보를 제공하며 **user 가 많이 겹치는 item 은 공통 trend 패턴을 공유 할 수 있으므로 trend 예측의 정확도를 높일** 수 있다. 


<br>
<br>



### 2. Preliminaries 


#### 2-①. Term Definition 

##### ▶ 용어정의 

|용어|내용|표기|
|------|---|---| 
|**Time Step**| 미리 정의한 시간 간격 (ex. 한시간) |**Δ𝑡**|
|**Velocity**|item j 에 대해서, time step t 동안 수집된 interaction 수를 time step t 에서의 velocity 로 정의 (unit time Δ𝑡 당 item j 의 popularity) |**W𝑗𝑡**|
|**Acceleration**|time step t 에서의 item j 에 대한 acceleration. item j 의 velocity 가 단위 시간 Δ𝑡당 ΔW𝑗𝑡씩 **변화**하고 있음을 나타낸다 (**변화량**). | **A𝑗𝑡** = **ΔW𝑗𝑡** = W𝑗(𝑡) - W𝑗(𝑡-1)|



##### ▶ **Acceleration = Trend**

- item 𝑗 의 시간 단계 𝑡 에서의  acceleration A𝑗𝑡 가 **모든 item 의 acceleration 중 가장 높으면** 해당 item 은 시간 단계 𝑡에서 **trendy 한 것으로 간주**한다.

<br>

#### 2-②. Problem Definition 

##### ▶ 적절한 Time interval (ex. 향후 1시간 trend item, 향후 하루동안 trend item) 이 관건이다. 

현재 trend 를 빠르게 감지하고 실시간으로 인기 있는 item 을 추천하는 것이 이상적이다. 이를 위해선 충분한 interaction 을 축적하는데 일정 시간이 필요하면서도, trend 는 동적으로 변화하면서 (dynamic variations) 데이터 수집 과정에서 시간적 변동 (temporal drift) 있는 것이 특징이기 때문에 적절한 time interval Δt 를 설정하는 것이 중요하다. Δt 가 너무 작다면 수집된 데이터가 불충분하여 noisy 가 발생할 수 있고, 너무 크다면 시간적 변동성이 발생해 예측 성능이 낮아질 수 있다. 따라서 데이터에 맞는 feasible 하고 short 한 적절한 time interval 을 찾는 것이 중요하다. 



##### ▶ 시간 단계 길이 (time step length) 와 작업 실행 가능성 (task feasibility) 간의 상관 관계에 대한 가설 

**bias-variance tradeoff** : 시간 단계 길이가 짧으면(예: 1시간) 데이터 희소성 (data sparsity) 으로 인해 variance 가 발생하고, 시간 단계 길이가 길면(예: 하루) 시간적 드리프트 (temporal drift 시간에 따른 변동) 로 인해 편향이 발생한다. 따라서 둘 사이의 균형을 잘 맞출 수 있는 sweet spot 을 찾아야 한다. 

![fig3](https://github.com/hopebii/kaist_ds535/blob/main/fig3.png)


##### ▶ one-step time series forecasting task 

trend recommendation task 를 one-step time series forecasting 문제로 정의한다. 각 item 대해 주어진 **historical acceleration** [A𝑗0, A𝑗1, . . . , A𝑗𝑡] := **A𝑗,0:𝑡** 과, covariates 와 같은 추가적인 **contextual information** [C𝑗0, C𝑗1, . . . , C𝑗𝑡] := **C𝑗,0:t** 가 주어졌을 때, 다음 step 인 (t+1) 에서의 acceleration 을 예측하기를 원한다. 그리고 trend prediction 을 기반으로 상위 k 개의 아이템을 추천한다. 

![fig2](https://github.com/hopebii/kaist_ds535/blob/main/fig2.png) 


<br> 


#### 2-③. Baseline model 

- 널리 채택된 두 가지 휴리스틱 모델을 설명한 다음 일반적인 형태의 딥러닝 기반 확률론적 시계열 예측 모델을 baseline model 로 사용한다. 

##### ▶ (1) Markov Heuristic Model

임의의 item j 의 next time step 의 acceleration A𝑗 (𝑡+1) 는 “오직” 현재 time step 의 acceleration A𝑗t 에 의존한다고 가정한다. 실제로 acceleration 는  짧은 시간 동안 동일하게 유지되는 경향이 있으므로 마르코프 휴리스틱 모델을 다음과 같이 정의한다. 

![fig4](https://github.com/hopebii/kaist_ds535/blob/main/fig4.png)


- Aˆ𝑗 (𝑡+1) = next time step 에서 예측된 acceleration 
- Auto regressive model (AR) 의 special case 로도 볼 수 있다. 


##### ▶ (2) Exponential Moving Average (EMA) Heuristic Model

마르코프 휴리스틱 모델의 가장 큰 단점은 다음 시간 단계의 item acceleration 이 현재 시간 단계의 영향을 받기 때문에 데이터 희소성 등의 문제로 인해 노이즈가 발생할 수 있다는 것이다. 따라서 여러 개의 최신 시간 단계를 고려하여 더 최근의 시간 단계에 더 많은 가중치를 할당하는 지수이동평균 휴리스틱 모델을 정의할 수 있다. 

![fig5](https://github.com/hopebii/kaist_ds535/blob/main/fig5.png)


- Aˆ𝑗 (𝑡+1) = next time step 에서 예측된 acceleration
- T : 모델이 고려하고 있는 최근 시간 단계 수
- wk : 현재 시간 step 에서 멀어질수록 기하급수적으로 감소하는 미리 정의된 가중치
- ARIMA model 의 special case 로 볼 수 있다. 



##### ▶ (3) Deep Learning based Time Series Forecasting Model

휴리스틱 모델은 일반적으로 다양한 시나리오에 적응할 수 있는 유연성이 부족한 일반적인 가정 (general assumptions) 을 인코딩한다. 그러나 **acceleration 패턴은 도메인(리테일, 미디어, 뉴스 등)에 따라 다르다**. 예를 들어, 매주 수요일마다 TV 시리즈의 새 에피소드가 공개되는 것과 같이 리테일과 미디어 영역 모두에서 다양한 주기(일별, 주별, 계절별 등)의 주기적 acceleration  패턴이 풍부하게 존재한다. 반대로 뉴스는 시간에 민감하고 사람들은 가장 최근 뉴스를 팔로우하는 경향이 있기 때문에 뉴스 영역에서는 이러한 규칙적인 acceleration  패턴이 거의 관찰되지 않는다. 또한 같은 도메인 내에서도 다양한 acceleration  패턴이 공존할 수 있다. 예를 들어, 특정 영화 플랫폼에서 새로 개봉한 액션 영화의 acceleration 곡선은 해당 플랫폼 사용자 커뮤니티의 선호도에 따라 새로 개봉한 다큐멘터리 영화의 acceleration 곡선에 비해 지속적으로 가파른 증가세를 보일 수 있다. 따라서 **트렌드 추천을 위한 보다 일반적인 솔루션은 다양한 시나리오에 적응할 수 있는 학습 가능한 딥러닝 기반 시계열 예측 모델을 설계**하는 것이다. 모델을 공식화하면 다음과 같다. 

![fig6](https://github.com/hopebii/kaist_ds535/blob/main/fig6.png)

- 𝑓seq (·) : 과거 acceleration  를 집계하고 다음 시간 단계에서 acceleration  의 확률적 분포를 예측하는 순차적 모델로 DeepAR, RNN, MQCNN, TFT 등이 있다. 



<br>
<br>


### 3. Model : collaborative time series forecasting model with user-item interactions (TRENDREC)

#### 3-①. Two-phase framework 

|phase|objective| 
|------|---| 
|**1. 다음 item 추천** | 보조적인 목표로 하여, TrendRec은 user-item 간 interactive signal 을 활용하여 item 간의 기본 상관관계를 감지하고 이러한 지식을 item embedding 으로 인코딩한다. | 
|**2. 다음 time step 의 trend (=acceleration) 예측** | 학습된 item embedding 을 사용하여 시계열 예측 목표를 위해 각 시계열에 대한 추가 context 를 제공한다. | 

<br>

#### 3-②. Model overview 

> TrendRec : RecSys + Time series forecasting 

- 추천 모델은 user-item interaction 을 통해 다음 아이템 추천 objective 를 학습한다.
  -  item feature 에 대한 representation learning 을 통해 dense latent item embedding 을 생성한다. 이를 통해 item 간 correlation 을 파악하여 시계열 예측에 대한 추가적인 context 를 제공한다. Item correlation 을 인코딩하는 shared latent item embeddings 을 통해 두 objectives 가 연결된다. 
- 시계열 예측 모델은 item의 accelerations 를 사용하여 다음 단계 acceleration 예측 objective 에 대해 학습한다. 

<br>

> probabilistic graphical model (PGM)

![fig7](https://github.com/hopebii/kaist_ds535/blob/main/fig7.png)


##### ▸ **노드**

|표기|내용| 
|------|---| 
| V𝑗t ∈ R(D) |∘  **item 𝑗’s properties** till time step t  both static properties and dynamic properties <br> ∘  𝐷 is the hidden dimension of the embedding <br> ∘ **latent item embedding** | 
|𝜆v|∘  **hyperparameters** related to distribution variance of latent item embedding | 
|A𝑗,0:t ∈ R(𝑁𝑗t)|∘  **item 𝑗’s historical acceleration** till time step 𝑡 which is [A𝑗0, A𝑗1, . . . ,A𝑗𝑡]| 
|A𝑗(𝑡+1) ∈ R|∘  the acceleration of item 𝑗 at the next time step 𝑡 + 1| 
|𝑁𝑗t |∘  the number of historical time steps of item 𝑗 till time step t| 
|U𝑖t ∈ R(D) |∘  **user 𝑖’s interests** till time step t <br> ∘  **latent user embedding** | 
|𝜆u|∘  hyperparameters related to distribution variance of latent user embedding | 
|S𝑖t ∈ R(N𝑖t x D) |∘  **user 𝑖’s historical interaction sequence** till time step 𝑡 <br> ∘  embedding matrix and each row of it represents an item embedding| 
| 𝑁𝑖t |∘  the number of interactions from user 𝑖 till time step t| 
|R𝑖𝑗t ∈ {0, 1}|∘  **interaction label** denoting whether user 𝑖 interacted with item 𝑗 at time step t| 

##### ▸ **엣지**

|표기|내용| 
|------|---| 
|Edge S𝑖𝑡 → U𝑖t |∘  user 의 이전 interactions 는 user 의 interest 를 표현하며, 이는 user의 다음 행동에 영향을 미친다. <br> ∘  ex. 휴대폰을 구매한 사용자가 다음에 휴대폰 액세서리를 구매할 수 있다.| 
|Edge {U𝑖𝑡, V𝑗𝑡} → R𝑖𝑗𝑡|∘  Interaction 은 user interests U𝑖𝑡 와 item properties V𝑗t 에 따라 달라진다.| 
|Edge {V𝑗𝑡, A𝑗,0:𝑡 } → A𝑗 (𝑡+1)| ∘ 다음 시간 단계 𝑡 +1에서 아이템 𝑖의 acceleration 는 item feature 와 item 의 과거 acceleration 에 영향을 받는다. <br> ∘ ex. 액션 영화는 특정 웹사이트의 사용자 커뮤니티에서 트렌드가 될 가능성이 높다. <br> ∘ ex. 주간 trend 패턴이 주기적으로 나타나는 item | 

##### ▸ **Generative process**

평균이 𝝁 그리고 분산이 diagonal covariance λ<sup>-1</sup>ⅠD 인 가우시안 분포에서 latent offset vector 를 설정하여 latent item embedding 과 latent user embedding 을 계산한다. 

![fig8](https://github.com/hopebii/kaist_ds535/blob/main/fig8.png)

R𝑖𝑗t 를 구하기 위해서 softmax function 을 latent user embedding 와 latent item embedding 을 내적한 값에 적용하여 recommendation score 를 계산한다.
- Y𝑖𝑗𝑡 = 𝑓softmax(U'𝑖𝑡•V𝑗𝑡)
- R𝑖∗𝑡 ~ 𝐶𝑎𝑡([Y𝑖𝑗𝑡]), j: 1,,..,J , 𝐶𝑎𝑡 is categorical distribution. 



<br>

#### 3-③. Training 

##### ▸ **Maximum a Posteriori (MAP) Estimation**

maximum a posteriori (MAP) estimation 

![fig11](https://github.com/hopebii/kaist_ds535/blob/main/fig11.png)


다음 item 을 추천하는 것에 있어서 interaction R𝑖𝑗t 에 대한 조건부 확률을 다음과 같이 정의한다. 

![fig9](https://github.com/hopebii/kaist_ds535/blob/main/fig9.png)

item accelerations A𝑗(𝑡+1) 에 대한 조건부 확률을 다음과 같이 정의한다. 

![fig10](https://github.com/hopebii/kaist_ds535/blob/main/fig10.png)

-  𝑓𝑡𝑠 (·) : 다음 시간 단계 𝑡에서 acceleration 의 확률적 분포를 예측하기 위해 item 의 과거 acceleration 와 latent item embedding 을 모두 사용하는 모든 유형의 확률론적 시계열 예측 모델


##### ▸ **Negative Log Likelihood (NLL)**

posterior probability 를 최대화 하는 것은 negative log likelihood 를 최소화하는 것과 같다. NLL 은 다음과 같이 계산할 수 있다. 

![fig12](https://github.com/hopebii/kaist_ds535/blob/main/fig12.png)

- (10) : Next Item Recommendation Loss → 이를 최소화하면 학습 세트에서 다음 항목 추천 성능이 향상된다. 
- (11) : Time Series Forecasting Loss → 이 term 을 최소화하면 훈련 세트에서 acceleration 예측이 향상된다.
- (12) : Regularizing Latent Item Embedding V𝑗t and Latent User Embedding U𝑖t  →  V𝑗t 를 zero-mean Gaussian prior 에 근접하게 정규화하고 U𝑖𝑡 를 유저의 과거이력이 유저의 흥미를 나타낸다고 가정하고, 계산된  𝑓seq (S𝑖𝑡) likelihood function adopted by the probabilistic time series forecasting model에 근접하게 정규화한다. 

<br>

#### 3-④. Inference 

![fig13](https://github.com/hopebii/kaist_ds535/blob/main/fig13.png)

- V*jt : the posterior of item j’s latent item embedding
- 𝑓∗ts (·) : the trained sequential time series forecasting model

<br>

#### 3-⑤. Model architecture  

![fig14](https://github.com/hopebii/kaist_ds535/blob/main/fig14.png)

- 왼쪽그림 : overview network structure
- 오른쪽그림 : figure visualizes the full details of the TrendRec implementation

  
- the model contains two principal components
  - (1) a sequential recommender :   **R𝑖𝑗t** ⇨  recommendation score 는 latent user embedding U𝑖t 과 latent item embedding V𝑗t 사이의 내적 곱을 기반으로 계산된다. 
  - (2) collaborative time series forecasting model :   **A𝑗(𝑡+1)** ⇨  다음 item 추천 (1) 과정에서 pre-trained 된 latent item embedding 을 가져와 item historical acceleration 와 함께 활용하여 다음 시간 단계에서의 acceleration 를 예측한다. 
  
  → 2가지 요소는 학습가능한 latent item embedding 을 통해 join 된다. 




<br>
<br>

### 4. Experiments

▢ **Research question**

|Research question|내용| 
|------|---| 
|**Q1**| task feasibility 와 time step length 의 correlations 에 대해 제안한 가설 (적절한 Δ𝑡 의 존재) 이 적용되는지, 각 dataset 의 time step length 는 어떻게 선택해야 하는지  | 
|**Q2**| TrendRec이 휴리스틱 모델과 기본적인 딥러닝 기반 시계열 예측 모델을 포함한 모든 기준 모델보다 성능이 우수한지 | 

<br>

#### 4-①. Datasets

![fig15](https://github.com/hopebii/kaist_ds535/blob/main/fig15.png)

- 리테일 (TaoBao), 미디어(Netflix), 뉴스(MIND)를 포함한 다양한 도메인의 데이터를 이용
  - TaoBao 의 경우  아이템 카테고리가 크기 때문에 , 3개의 구분된 데이터셋을 구조화하기 위해 인터랙션 수를 기반으로 상위 3개의 아이템 카테고리를 선택한다 → TaoBao Cat1, TaoBao Cat2, TaoBao Cat3
- 다음 item 추천 objective 와 시계열 예측 objective 사이에 시간적 누수 (temporal leakage) 가 발생하지 않도록 엄격한 실험설정을 적용 : 모든 training interactions 이 모든 testing interactions 보다 먼저 발생하도록 데이터를 시간적으로 분할하고, training 단계에서 두 objective 에 대해 정확하게 동일한 훈련 데이터셋을 사용한다. 

<br>

#### 4-②. Evaluated methods 

|Mtehods|설명| 
|------|---| 
|**Oracle**| ∘  다음시간 단계에서 실제 정답 (ground truth) 미래 acceleration 에 접근할 수 있다. <br> ∘  항상 acceleration 을 정확하게 예측하고 상위 k 개의 트렌드 아이템을 추천한다. | 
|**Random**| ∘  전체 아이템 카탈로그에서 replacement 없이 전체 아이템으로부터 random selection 을 하여 아이템을 추천해준다. | 
|**Exponential moving average (EMA)**| ∘  지수이동평균은 최근 m 시간 단계 (latest 𝑚 time steps) 의 acceleration 의 가중치 합을 기반으로 다음 시간 단계의 acceleration 을 예측하는 규칙기반 모델이다. (m=8) <br>  ∘  가중치는 현재 시간 단계로부터 멀어지는 시간 단계수가 증가함에 따라 0.75의 계수로 기하급수적으로 감소한다.  | 
|**DeepAR**|∘  auto-regressive RNN 에 기반한 SOTA 시계열 모델 중 하나이다. | 
|**TrendRec**|∘  본 연구에서 제안한 모델로 two-phase 로 이루어져있다. <br> ∘ 다음 아이템 추천을 위해 GRU4Rec 을 채택해 latent item embedding 을 학습한다. <br> ∘ 시계열 예측을 위해선 DeepAR 모델을 사용한다. 최신 시계열 예측 모델 중 하나이고, 널리 채택되고 있기 때문이다. | 

<br>

#### 4-③. Evaluation metrics 

시계열 예측 설정에서 RMSE와 같은 평가 지표를 채택하는 대신, 다음 시간 단계에서 트렌드 아이템을 추천하는 트렌드 추천 집합의 목표에 밀접하게 부합하는 평가 지표를 설계한다. 


##### ▸ (1) Acceleration metric

![fig16](https://github.com/hopebii/kaist_ds535/blob/main/fig16.png)

- 모델이 예측한 다음 단계 시간 t 의accelerations 에 기반하여 상위 k 개 item 을 선택
- 그런 다음 선택한 𝑘 아이템을 다음 시간 단계 𝑡에서 해당 ground truth acceleration 에 다음과 같이 맵핑
- acceleration 이 trend 의 정량적인 측정 (quantitative measurement) 이기 때문에 item 의 다음 시간 단계의 예측한 acceleration 의 총합 (sum) 으로 계산하고 모델의 trendiness score 로 사용
  - 값이 높을수록 모델은 다음 시간 단계에서의 트렌드한 아이템에 대한 예측을 더 잘한다.
- [0,1] 사이의 값으로 스케일링을 하기 위해 trendiness score 의 top 에 대해 min-max normalization 을 적용한다. trendiness score 의 upper bound 는 Oracle 모델에서, lower bound 는 Random 모델에서 온다. 
 
![fig18](https://github.com/hopebii/kaist_ds535/blob/main/fig18.png)




##### ▸ (2) TNDCG Metric


![fig17](https://github.com/hopebii/kaist_ds535/blob/main/fig17.png)

-  Trendiness-Normalized-DCG (TNDCG) metric : 아이템의 rank position 을 logarithmic reduction factor 로 고려한다. 

![fig19](https://github.com/hopebii/kaist_ds535/blob/main/fig19.png)


- r : index the rank position
- A<sup>p</sup><sub>r</sub> : acceleration of item ranked at position r based on order from **model prediction (표기 p)** 
- A<sup>O</sup><sub>r</sub> : acceleration of item ranked at position r based on order from **ground truth (표기 O)** 




##### ▸ (3) Evaluation protocol 

timestamp 를 기준으로 training 과 test step 을 나눈다. 그리고 testing 을 위해 가장 최근의 20% time span 을 남긴다. (예. eight hour training window, two-hour testing window) 

<br>

#### 4-④.  Hypothesis validation Q1 : 적절한 Δt 선택하기 

![fig20](https://github.com/hopebii/kaist_ds535/blob/main/fig20.png)

Markov heuristic model 을 활용해 성능을 평가한다. 간단하지만 generic 한 가정에 기반한 기초적인 모델이고, 따라서 해당 모델의 성능은 task feasibility 를 반영한다. 
결과를 보면, TaoBao 와 MIND 데이터 세트의 곡선은 데이터 희소성 완화로 인해 시간 간격이 길어질수록 acc 지표가 먼저 개선된 다음 temporal drift 로 인해 감소하는 Q1 가설과 일치하는 결과를 보인다. 반면 Netflix 데이터셋의 경우 곡선이 계속 감소하고 있는데, 이는 time stamp 단위가 하루로, 충분한 데이터를 수집할 수 있을 만큼 길지만 temporal drift 가 발생하기 때문이다. 전반적으로 위의 결과는 가설을 입증하고 있다. 각 데이터셋의 시간 간격 **Δ𝑡을 각 곡선의 peak 에 따라 선택**한다. 일관성을 위해 3개의 TaoBao dataset 은 모두 3시간, Netflix 는 하루, MIND 는 30분 시간간격으로 설정한다. 

<br>

#### 4-⑤.  Experimental results Q2 : TrendRec 모델의 우수함 증명 

TrendRec 모델을 3개 도메인의 데이터에 대한 다양한 베이스라인모델에 대해 평가한다. 

![fig21](https://github.com/hopebii/kaist_ds535/blob/main/fig21.png)


TrendRec 이 가장 좋은 performance 를 보인다. TrendRec 의 시계열 예측 부분이 DeepAR 로 구성되어 있는데, DeepAR 대비 TrendRec 의 성능 향상은, 다음 item 추천 파트에서 얻은 pre-trained 된 latent item embedding 을 활용한 것이 효과적이었음을 보여준다. 

<br>

#### 4-⑥.  Findings 

##### ▸ Deep learning based models significantly outperform heuristic models

딥러닝 기반의 모델은 휴리스틱 모델보다 성능이 높다. 특히 TaoBao 과 Netflix 데이터셋에 대해 DeepAR 과 TrendRec 과 같은 딥러닝 기반의 모델은 휴리스틱 모델을 큰 차이로 더 성능이 높게 나온다. 해당 결과는 trend 추천을 위해 학습가능한 모델을 채택하는 것의 중요성을 강조한다. 

##### ▸ The EMA model is worse than the Markov model in most cases

EMA 모델은 대부분의 경우에서 마르코프 모델보다 성능이 더 저하되는 결과를 보였다. 이는 trend 가 동적으로 변하고  이러한 결과는, recency bias 를 가진 간단한 가중치합 (weighted sum) 보다, 다음 시간 단계에서의 트렌드와 과거 트렌드 (historical trends) 사이에 의존적인 관계가 보다 더 복잡하다는 것을 의미한다. 


##### ▸ Performance gain from deep learning based models is relatively small in the News domain

뉴스도메인이 리테일이나 미디어 도메인과 비교했을 때, 딥러닝 기반의 모델과 휴리스틱 모델 사이의 성능 차이는 상대적으로 미미하다. (relatively marginal) 뉴스 도메인의 item 시계열을 분석해 보면 일반적으로 아이템이 출시되면 단기간에 최대 acceleration 에 도달한 후 급격히 하락하는 것으로 나타난다.  이는 주로 시간에 민감한 뉴스의 특성 때문이며, 딥러닝 기반 모델에 큰 도전 과제이다. 



<br>
<br>

### 5. Conclusion

#### 5-①. Summary 

- 이 연구에서는 추천 시스템에서 잘 다루어지지 않은 주제인 trend recommender 를 연구한다. 선행 연구가 제한적으로 이루어져 있기 때문에 trend 라는 개념을 공식적으로 정의하는 것으로 시작한다. 이후 적시에 안정적으로 trend 를 식별하는데 문제가 되는 bias-variance tradeoff 현상을 관찰하여 이를 바탕으로 trend recommendation 을 one-step time series forecasting 로 공식화한다.
- 방법론 측면에서 user-item interactive signal 을 활용하여 item 간 correlation 을 파악하고 이를 바탕으로 trend 예측을 용이하게 하는 TrendRec 이라는 two phase model 을 개발하였다.
- Recommendation context 에서 trend 의 개념을 공식적으로 정의하고 그에 맞는 평가지표와 평가 프로세스를 수립했다. 
- 리테일, 미디어, 뉴스 등 다양한 영역의 데이터셋에 대한 실험으로 통해 TrendRec 모델의 효과를 입증했다. 

<br>

#### 5-②. Opinion 

해당 논문은 시계열 예측 모델을 적용하여 trend 한 item set 을 추천해주는 방법론을 제안하고 있습니다. user-item 간 interaction 정보를 바탕으로 item 정보를 embedding 하여, 더 효과적으로 시계열 예측이 가능하도록 모델 구조를 구성하였으며 특히 trend 라는 맥락에서 발생할 수 있는 적절한 time interval 을 설정하는 데 있어 bias-variance tradeoff 문제를 명시하고 관련된 해결책을 제시하고 있습니다. interaction 수의 변화율 (acceleration) 을 기준으로 trend 를 감지하려고 한 시도가 신선하게 다가왔으며, 수업에서 배웠던 sequence 한 정보를 기반으로 추천해주는 추천시스템 모델들과는 또 다른 맥락의 추천 방법론인 것 같아 전반적으로 인상깊었던 논문이었습니다. 또한 dataset 마다 trend 가 발생하는 상이한 특징에 따라 optimal 한 time interval 을 설정하는 접근 방식이 논문에서 뉴스나 영화 예시를 들었던 것 처럼 domain-based 한 부분이라, 추천 메커니즘에 대한 해석이 더 흥미롭게 다가왔던 것 같습니다. 그러나 TrendRec 에서 시계열 예측 모델로 DeepAR 을 선택한 것, 다음 아이템 예측 모델에 임베딩 방식으로 GRU4Rec을 채택한 것 대한 근거가 조금 부족하다고 느꼈습니다. 추가적인 다른 모델 채택 구성방식의 실험결과도 비교해주었으면 좋을 것 같다는 생각이 들었습니다. 하지만 trend recommendation 이라는 분야에서 해당 논문이 가지고 있는 가치는 매우 크다고 생각하며 앞으로 해당 분야가 발전함에 있어서 중요한 연구가 될 것이라 생각합니다. 



<br>
<br>

#### 👩🏻 Review writer information 
- 이다현 (Lee Dahyeon) 
  - Master student, Department of Data science, KAIST 
  - contact : isdawell@kaist.ac.kr 
