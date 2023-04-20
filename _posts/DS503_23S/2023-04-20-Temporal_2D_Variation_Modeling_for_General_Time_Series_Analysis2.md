---
title:  "[ICLR 2023] Temporal 2D-Variation Modeling for General Time Series Analysis2"
permalink: Temporal_2D_Variation_Modeling_for_General_Time_Series_Analysis2.html
tags: [reviews]
---

**Introduction**

시계열 분석은 기상 예측 , 모니터링 데이터의 이상 감지 및 행동 인식을 위한 궤적 분 류 와 같은 다양한 실제 응용 분야에서 널리 사용되며, 이러한 넓은 실용적 가치 때문에 시계열 분석은 큰 관심을 받고 있다.

일반적으로 언어나 비디오와 같은 다른 유형의 순차 데이터와 달리 시계열은 연속적으 로 기록되며 각 시간 지점은 일부 스칼라만 저장한다. 그러나 하나의 단일 시간 지점으 로는 보통 분석에 충분한 의미적 정보를 제공할 수 없기 때문에, 많은 경우 더 많은 정 보를 제공하기 위해 시계열의 내재적인 특성인 연속성, 주기성, 추세 등을 반영할 수 있 는 시간적 변화에 초점을 맞추고 있다. 그러나 실제 시계열의 변동은 항상 복잡한 시간 패턴을 포함하며, 여러 변동(상승, 하강, 변동 등)이 서로 섞이고 중첩되어 모델링을 매우 어렵게 만든다. 특히 딥러닝 커뮤니티 에서는 이런 복잡한 시계열 Data 의 시간적 변화 를 포착하기 위해, (RNN), (TCN), Attention 메커니즘 을 사용하여 모델링을 진행하고 있 으나, 각 모델들의 한계로 인해 효율적인 모델링을 진행하는데 한계가 존재한다.

본 논문에서는, 복잡한 시간적 변동을 해결하기 위해 시계열 데이터를 다양한 주기성에 서 분석한다. 실제 현실세계의 시계열 데이터는 일일, 주간, 연간 같은 주기성을 관찰할 수 있으며, 이런 주기성이 서로 상호작용하여 변동 모델링을 어렵게 한다. 

` `이  관점에서  해당  논문의  핵심은  이런  시계열  데이터를  주기내에서의  변동성 (intraperiod-variation)  과  주기  간  변동성(interperiod-variation)  으로  각각  나눠,  기존 Temporal Variation  을  Intraperiod - , Interperiod – variation  으로  확장하여  Multi – periodicity 특성을 반영하는 것이다. 

이런 Multi – periodicity 특성을 반영하기 위해, 1D 시계열을 2D 공간으로 확장한다. 구 체적으로, 그림 1처럼, 1D 시계열을 각 열이 주기 내 시간 점을 포함하고 각 행이 서로 다른 주기에서 동일한 위상의 시간 점을 포함하는 2D 텐서로 재구성 한다. 따라서 1D 

시계열을 2D 텐서 집합으로 변환하여 원래의 1D 공간에서 표현 능력 병목 현상을 깨고 2D 공간에서 intra-period- 및 inter-period-variations를 효과적으로 통합하여 시간적 2D- variations을 얻을 수 있다.

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.001.png)

` `또한 기술적으로, 위와 같은 동기에 기반하여, 이전 백본(backbone)을 넘어서 시계열 분 석을 위한 새로운 태스크-일반 모델로 TimesNet을 제안한다. 

구체적으로, TimesBlock은 학습된 주기를 기반으로 1D 시계열을, 2D 텐서 SET 으로 변 환하고, 파라미터 효율적인 inception block 을 통해 2D 공간에서 intra-period 및 inter- period 변화를 포착할 수 있다. 실험적으로, TimesNet 은 단기 및 장기 예측, 대체, 분류 및 이상 탐지를 포함한 5 가지 주요 분석 작업에서 일관된 최고 성능을 달성했다.

**RELATED WORK**

` `시계열 분석의 주요 문제인 시간적 변화 모델링은, ARIMA(Anderson & Kendall, 1976), Holt-Winter(Hyndman & Athanasopoulos, 2018) 및 Prophet(Taylor & Letham, 2018) 같은 고전적 모델 기반의 연구가 진행되었으나, 시계열의 변화는 일반적으로 미리 정의된 패 턴으로는 충분히 설명하기 어렵기 때문에 실용성은 제한되는 한계가 존재했다. 

` `최근 에는 MLP, TCN, RNN 기반의 다양한 딥 모델들이 시계열 모델링에 적용되 왔다. 우선 MLP 기반의 방법들 (Oreshkin et al., 2019; Challu et al., 2022; Zeng et al., 2023; Zhang et al., 2022)은 MLP을 시간 축으로 사용하며, 시간 의존성을 MLP 레이어의 고정 된 파라미터로 인코딩 한다. TCN 기반의 방법 (2019)은 시간 축을 따라 이동하는 컨볼루 션 커널로 시간 변화를 캡쳐하며, RNN 기반의 방법 (Hochreiter & Schmidhuber, 1997; Lai et al., 2018; Gu et al., 2022)은 재귀 구조를 활용하며, 시간 단계 간 상태 전이를 통해 시간  변화를  캡쳐한다.  특히  최근  딥러닝  분야에서  전반적으로  높은  성능을  보이는 Transformer는 시계열 예측에서도 큰 성능을 보여주고 있으며 (Zhou et al., 2021; Liu et al., 2021a; Wu et al., 2021; Zhou et al., 2022), 어텐션 메커니즘을 사용하여 시간 포인트 간의 시간적 종속성을 파악할 수 있다. 특히, Wu et al.은 학습된 주기를 기반으로 자기 

상관 메커니즘을 갖춘 Autoformer를 제시 했으며. 그 후, FEDformer (Zhou et al., 2022)는 계절성-추세 분해를 강화하고 주파수 도메인 내에서 sparse 어텐션을 제시했다. 그러나 본 논문의 경우 이전 방법과 달리, 시계열의 다중 주기성을 탐색하여 복잡한 시간적 패 턴을 해결하고, 이미 인공지능 컴퓨터 비전에서 잘 알려진 백본을 사용하여 처음으로 2D 공간에서 시간적 2D-변화를 포착한다.

**3 TIMESNET**

` `해당 논문을 이해하기 위해서 우선 Multi periodicity 를 이해해야 한다. Multi periodicity 는 위에서 간단히 언급했지만, 해당 논문은 Intraperiod Variation과 Intreperiod Variation 을 구분하여 분석하였다. 우선 Intraperiod Variation 은 Period 내에서 발생하는 Variation 으로  Short Term Temporal pattern  을  반영하며,  일반적으로  1D Time Series data  를 Input 으로 모델링 진행시 고려하는 Variation 이다. 또한 추후 2D Tensor 로 변환할 때 각 Column 으로 표현되어 같은 Period 내 time Point 간의 Variation 을 의미한다. 반면 Interperiod Variation  은  서로  다른  Period  에서  발생하는  Variation  으로  Long Term Temporal Pattern  을  반영한다.  이는  2D Tensor  로  변환시  각  row  로  표현되어  다른 Period 에 존재하지만, 같은 Phase 를 가지는 time points 간 Temporal Variation 을 의미 한다. 이 논문은 이 둘을 조합하여 사용 하는 것이 key idea 이다.

여기서  Interperiod  의  기준을  나누기  위해서  그림  1과  같이  해당  Data  에서  다양한 Period  를  얻어야  한다.  이를  위하해서  해당  논문은  아래  그림과  식처럼  Fast Fourier Transform 을 사용한다. 

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.002.png)

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.003.png)

FFT 를 적용해서 Sequence 에 대한 Amplitude 및 Frequency 를 도출 하고, Frequency 중 Amplitude 값이 가장 높은 Top K 개의 Frequency 를 선정하여, Frequency 에 상응하 는 Period 를 구한다. 여기서 top K 개의 Frequency 를 선정하는 이유는 Frequency 영역 의 sparse 함을 고려했을때, 으미 없는 high Frequency 에 의한 Nosie 를 피하기 위한것 이다.  

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.004.png)

여기서 선정된 Top K개의 Frequency 에 해당하는 Period 는 위 수식에서 pi 로 설정한다. 이렇게 구해진 Frequency {f1, ..., fk} 와Period {p1, ..., pk} 를 이용하여 1D Time series data ![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.005.png) 를 2D Tensor 로 변환 한다. 

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.006.png)

여기서 Padding 은 2D tensor 로 재구성 하기 위해 시계열을 시간 축을 따라 0으로 패 딩하는 작업인데, 아래 수식 에서 T/fi 값을 맞춰주기 위해 진행한다.

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.007.png)

(pi와 fi는 각각 i번째 변환된 2D 텐서의 행과 열의 수를 나타낸다.)

이와 같은 방식으로 선택된 주파수와 추정된 주기에 기반하여 k개의 다른 2D 변동을 나 타내는 2D 텐서 집합 ![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.008.png) 을 얻게 된다.

이러한 변환은 변환된 2D 텐서에 두 가지 유형의 지역성(locality)을 가져다 준다는 점도 주목할  만하다.  즉,  연속된  시간점(열, intra-period variation)과  인접한  주기(행, inter- period variation) 사이의 지역성이다. 따라서, 시계열 2D-변동성은 2D 커널에 의해 쉽게 

처리될 수 있다.

**3.2 TIMESBLOCK**

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.009.png)

Capturing temporal 2D-variation

위 그림처럼 Reshape 된 결과를 통해 pi, fi 에 대한 Representation 을 추출한다. 여기서 

input  은  ![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.010.png) 이고  output  은  ![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.011.png) 이다.  또한  위의  그림처럼 Reshape Back 이라는 과정을 통해 Convolution 과정을 통해 추출한 Representation 을 다시 기존 형태로 Reshape 을 하게 된다. 

마지막으로 아래 식처럼 Adaptive Aggregation 을 수행하게 되는데, 

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.012.png)

이는 이전 과정을 통해 만들어진 1D time series data 를 Aggregation 하는 과정이다. Amplitude A 는 선택된 Top K 개의 Frequency 와 Period 간 상대적 중요도를 반영할 수 있으므로, Transform 된 Tensor 의 중요도 역시 반영할 수 있다. 이로부터 처음 FFT 과정 을 통해 구해진 Amplitude 값을 바탕으로 Aggregation 하게 된다. 이로부터 나온 결과 는  서로  다른  Period  간  Interperiod,  Intraperiod  variation을  고려한  Temporal  2D Variation 을 동시에 캡쳐한 결과를 얻을 수 있다.

**4 EXPERIMENTS**

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.013.png)

위와 같은 setting 을 통해 실험을 진행하였고 

![](Aspose.Words.fe9a8a29-d51a-42e5-b229-da4476aaa176.014.png)

위 그림과 같이 time series 의 5개 Mainstream Task 에서 다른 모델들 대비 모든 방면에 서 SOTA 를 달성하였다.

**5 CONCLUSION AND FUTURE WORK**

해당 논문의 novelty 는 다음과 같이 3가지로 요약된다.

-다양한 주기성과 주기별 내/외부적 상호작용에 대한 동기부여를 바탕으로, 우리는 시간 적 변화 모델링을 위한 모듈화된 방법을 발견했습니다. 1차원 시계열을 2차원 공간으로 변환하여, 주기 내/외부적인 변화를 동시에 나타낼 수 있다.

-TimesNet은  TimesBlock을  사용하여  여러  주기를  찾고, parameter-efficient inception block으로 변환된 2D tensor에서 temporal 2D-variations을 포착하기 위한 새로운 task- general 모델로 제안된다.

-TimesNet은 단기 및 장기 예측, 대치, 분류 및 이상 탐지를 포함한 다섯 가지의 주요 

시계열 분석 작업에서 일관된 최고 성능을 달성했다.
