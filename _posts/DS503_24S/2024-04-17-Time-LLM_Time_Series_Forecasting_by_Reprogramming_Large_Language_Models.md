---
title:  "[ICLR 2024] Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"
permalink: Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models.html
tags: [reviews]
use_math: true
usemathjax: true
---


## **1. Problem Definition**  

 시계열 데이터 예측은 전통적인  AR, MA, ARIMA부터 현대의 Deep-Learning을 활용한 CNN, LSTM 등 많은 모델에서 연구되어 오고 있다. 하지만, 모델을 학습 시킬 수 있는 시계열 데이터가 부족하여 주로 예측하는 모델이 domain specialized되어 있는 경우가 많다. 최근 연구에 따르면 LLM(Large Language Models)이 복잡한 Token sequence에 대해 강력한 패턴 인식 및 추론 능력을 가지고 있는 것으로 밝혀졌다. 본 논문에서는 이런 LLM의 패턴 인식과 추론 능력을 시계열 예측에 사용하고자 하지만, LLM이 가지고 있는 강점을 시계열 데이터에 사용하기에는 적절한 modality 변환이 필요한 상황이다. 

 이런 문제 상황을 해결하기 위해 본 연구는 LLM의 backbone은 그대로 유지하면서 일반적인 시계열 예측을 진행할 수 있게하는 Reprogramming Framework를 제시한다. 크게 두 가지의 방법을 제시하는데, 먼저 시계열 데이터를 LLM의 능력이 강화될 수 있도록 Embedding, Patching을 거친 후 Text vector들과 합쳐주는 Patch Reprogramming이 있고, 그 후에 시계열 데이터의 Context나 해결해야 하는 Task의 정보, 적당한 Statistics를 Input에 합쳐주는 Prompt as Prefix가 존재한다. 이 두 가지 특별한 Process를 통해 시계열 데이터가 LLM을 통해 좋은 성능의 예측이 가능하다는 것을 보여준다.

## **2. Background**  

### **2.1. Time Series for LLM**

 Time-Series for LLM의 의미는 LLM의 속의 구조를 고정시키고, Downstream task에 대한 Fine-tuning을 진행하기보다 시계열 데이터에 주요한 변화를 주면서 Task의 성능을 높이고자 한다. 본 논문 또한, LLM에 변화를 취하기 보다는 시계열 데이터를 manipulation하는 방법을 사용한다. 이전에 LLM을 활용한 연구(Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities, 2024, Jin) 를 살펴보자면,  Human knowledge를 미리 LLM에 추가한다면 예측의 성능이 높아지며 Sequence나 Numerical한 데이터를 LLM이 잘 이해할 수 있도록 Paraphrasing 하는 것 또한 긍정적인 효과를 불러온다는 결과가 있다.

 예를 들어, 전력량 예측에 대한 Task가 존재할 때 LLM에 미리 여름과 겨울에 전기를 많이 사용한다는 사실을 입력한다면 미래의 전력량 예측에 도움을 준다. 또한, 시계열 데이터를 input으로 사용할 때 시점 t에서 시점 t+1은 증가, 시점 t+1에서  시점 t+2은 감소처럼 이런 sequence에 대한 부연 설명을 통해 LLM이 시계열 데이터를 더 잘 이해할 수 있다.

 본 연구에서도 이와 유사한 개념으로 Prompt as Prefix와 Pre-trained word embedding을 사용하기에, 위의 예시를 참고하면 더욱 연구 Process를 이해하기 쉬울 것이다.

### **2.2. Consideration Time Series for LLM**

 ㄱ) 시계열 데이터 자체가 많이 존재하지 않는다.

 가장 큰 문제는  2024년 현재까지 시계열 데이터 셋 중 가장 크다고 여겨지는 것의 용량이 10GB 미만으로 Vision, NLP 등 다른 분야에 비해 Foundation Model을 학습시킬 데이터가 현저하게 부족하다.그렇기에 이를 해결하기 위해 GAN 같은 방법을 사용하거나 LLM 자체를 Domain에 따라 미리 Prompt를 넣어주기도 한다. 

 ㄴ) 각 시계열 데이터셋의 특징이나 모양이 상이하다. 

 먼저, Domain마다 데이터셋마다 통계적인 특성이나 Scale에서 차이가 난다. 예를 들어 제조 과정에서 얻어지는 변동성의 정도와 금융 시장의 변동성은 차원이 다른 수준이기에 이를 한 번에 통합하여 학습시키기 힘들다. 

 두 번째로 Granularity의 문제가 있다. 풀어서 얘기하자면 데이터의 time-step이 각 데이터 셋마다 다르다는 의미이다. 

## **3. Method**  

### **3.1. Model Setting**

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image1.png)

<!-- <div align="center">
 <img src="https://lh3.googleusercontent.com/fife/ALs6j_Gh2QNRWGAzt4yBO3JbB4MWxAlM3GrTb6uRJbHe3TOoU8gCBkNopwg2qrJf4h5LNdFao8_7yBaZXbjvrWG4xaMfWmorLQdXzTCvKRmFJL82QkIFrV3ix57pVc7JGmWZBt4540A-mmmNdNHufQp2rowXztCCS4kabAoB7ZafnYwDLM-NJ_Mqbw0bxQhWCqwTc_d5stfhflvqXYsdmlbHVNwbk2EGRR75nwtNLg6Q3PWpZDuLS_Gh2ILI3dSMRgHXTzmVQbX5o1vH2T8Xh42cfBzf4VZQ2-Cx8_0LZhEMqNOgxCo_dcLx8IMurYFnwWw1mv5TmQOXFAGcT2nwe6UDZTRatL_Bt2zQPY0rq-95O9SXYbLjJ9IhPjS7g_NCUj5sBDpIYBrI5CMnwKiTkcpphCuKEadWSZQtkXUbV8eVwaCDi0rtWXp4VEDnnST8jbi8sCPfTTySrBazrtR09fvxRFw1b0QP2IGLtmHLw2sN_Z3rKV1wA6rtinWtqGbdI5H_8AKTaXPwI7D-HlY5f019k013Kdsp9aRVXlaaOT_h2n_jDPKpWp0rzZoy7jDaSfAQtLDhx85YGDgHWvQ9B-gq-5vrmWz6qVg6HvQ5DZJQTueZfs6w7FBdrbdvqyrd5NdAsNOHOo7qnSXjDl5EDIbXv513RfjnC9E6h6DN2o8RiU1CDg22VBAl9Nm5Y_Li--n31PUIeR0mcgcEtwyYVOVr_Rfstm8FYHPAfmjp2CP9O7y22iv2MqU6qZJod4WT9Solx3ecNw8F42xnakKwdtuprpq5zLNoXj2u3SiRBVjLRJw7t_jAEAcW45tvTo0wCWKOSruuhPdofF8hzyyhwcnLYYamxj84yRw79JQgC6l-YpJe23VEJgTONwRHpYX5hZzh3r05VPtap3BrNTRkwjaY68sCU5kv6G0fwobM8hBvAOlTzet26B_SG_vZ_Ix5FmbByS_qh5vr4-MJ3HO5ZXXmWZA6YsbYbJpxXJG4hjay3_VpfRcPSLI0OSdCOh8MDoSNbo5l4jBrN40h0TAqr1bGVP-nlpf4rlxEg752i7px7cOjBB5qJkT32_RjilzXyKTcI88S5DVJA9ZkxYeaIrR1X6v6OhQtWxU04u0az5iB51E7xKyjThpxjrkoJxTBl_D8gUPak836jH-YQk6m93Y-eeLH-RMxRSw3paQDOXnljIn2y1AP8FOj-wmlgKTVAB5xAwj6vqH47pPjjXMkaXLQ2vX9s4xVKu1nJIf64bgJmFUVCd4icSOzUyWA8GNVKynUUUAzXsvkF2t_Cdy35AfJzDJHw4lwhTnSztssPE4N8_C5uNIdm7veynj77yjfuvLQYtQtyZ2Zg1FuCIFVBTqeKSlCIcmsFUEVbz8Pr0NX_IiwH2LvJMB8VIgD1ou1zCUEO7Fqe02Hsm_s-N3txBwfbdTq7zYcrA_ucMX5d4NXtgEF2ac6Euq3B_2ypDOCExGswKrESBn9Xry1u98-gyOq6QjjD-MpnLLoZJRNbAc4Kz0C85l-fl0cHTWVYPOvVEKKdsFV14zKXOO2Aiugv12LzJt0_eb4zhBiMD19GBQ4LhiZvDqwKKOL62M99sUkobomv-4flRzSyN6QMw=w1920-h919">
</div> -->

> Framework ot Time LLM

위의 그림이 전체적인 모델의 Framework를 보여준다. 크게 Model Setting, Patch Reprogramming, Prompt as Prefix, Output Generation파트로 나눠지게 된다.
먼저 Model Setting 파트를 보게 된다면, Multivariae Time Series Data를 변수별로 나누고 Window Size만큼 input으로 사용한다. 

$\huge \mathbf{X} \in \mathbb{R}^{N \times T} \rightarrow \mathbf{X}^{(i)} \in \mathbb{R}^{1 \times T}$

> Basic Time-series Data

 이후 각 단변수 시계열 데이터마다 Normalization을 진행한다. 이는 시계열 데이터가 주로 시간 변화에 따라 Distribution이 바뀌는 문제때문에 진행하는데, 이런 Distribution shift는 Forecating model이  generalization되지 않게 만드는 원인이다. 본 논문에서는 Reversible Instance Normalization(RevIN)을 사용하여 이 Distribution shift 현상을 해결하는데, 이는 따로 논문이 존재하니 더 자세히 알고 싶다면 아래의 논문을 참고하면 좋다.

[Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p "Reversible Instance Normalization")

 Nomalization 이후 시계열 데이터 셋에 대해 Patching을 해주게 된다. 시계열 데이터의 특성상 시간에 연속적이기에 각 Time-step끼리는 연관되어 있는 Semantic Information이 있기에 이를 Patching을 통해서 단일 시점의 데이터들을 통합하여 Local semantic information을 보존한다.

$\huge \mathbf{X}^{(i)} \in \mathbb{R}^{1 \times T} \rightarrow \mathbf{X}_ P^{(i)} \in \mathbb{R}^{P \times L_ p}$

> Patching

### **3.2. Patch Reprogramming**
 시계열 데이터를 Natural Language처럼 처리될 수 있게 Modality를 Align해주는 과정이 필요하다. 이를 통해 시계열 데이터를 Backbone model이 이해하기 쉬워지고, 시계열 데이터 속에서 Temporal Interrelationship을 잘 capture하게 된다. 그래서 이 과정에서 시계열 데이터의 특성이나 변화를 설명할 수 있는 Text 등을 이전의 Patching이 된 데이터에 Cross-Attention을 수행하여 Patch Representation을 진행한다. (시계열 데이터를 설명하는 Text 예시: "Short", "Up", "Late, "Steady" 등)

 먼저 이전의 Patching을 끝낸 시계열 데이터에 대해 Linear layer를 통해 embedding을 진행시킨다.

 $\huge \mathbf{X}_ P^{(i)} \in \mathbb{R}^{P \times L_ p} \rightarrow \hat{\mathbf{X}}_ P^{(i)} \in \mathbb{R}^{P \times d_ m}$

> Time-series data embedding

 그 후 위에서 얘기한 Modality align과정을 진행하게 되는데, 기존의 존재하는 vocabulary에는 너무 많은 단어가 존재하기에 전체 Word embedding을 사용하기에는 큰 cost가 존재하고, Time-series forecasting에 필요하지 않은 Prior knowledge가 많다. 이를 해결하기 위해 Voca를 미리 Linear layer를 통과시켜 핵심 단어로만 이루어진 Text Prototypes을 만든다.

(e.g. voca 속에 있는 apple, banana 등 관련 없는 단어는 없어짐)

 $\huge E \in \mathbb{R}^{V \times D} \rightarrow E' \in \mathbb{R}^{V' \times D}$

> Text Prototypes Generation

 이후 Embedding한 시계열 데이터와 위의 Text Prototype을 Cross-attention을 활용해 Align한다. 이때 Embedding TS는 Query로 Text-Prototype은 key와 value로 이용한다.

 $\large Q_ k^{(i)} = \hat{X}_ P^{(i)} W_ k^Q, W_ k^Q \in \mathbb{R}^{d_ m \times d}$
 
 $\large K_ k^{(i)} = E'W_ k^K, W_ k^K \in \mathbb{R}^{D \times d}$
 
 $\large V_ k^{(i)} = E'W_ k^V, W_ k^V \in \mathbb{R}^{D \times d}$
 
 $D: Backbone-model-Hidden-dimension$

> Query, Key, Value

위의 Query, Key, Value를 활용해 Multi-head Cross Attention 과정을 거친다.

$\large Z_ k^{(i)} = \text{ATTENTION}\left(Q_ k^{(i)}, K_ k^{(i)}, V_ k^{(i)}\right) = \text{SOFTMAX}\left(\frac{Q_ k^{(i)} {K_ k^{(i)}}^T}{\sqrt{d_ k}}\right) V_ k^{(i)}, \quad Z_ k^{(i)} \in \mathbb{R}^{P \times d}$

> Attention process

이렇게 만들어진 각 head를 Concat하여 Attention Output을 만들고

$\huge Z^{(i)} \in \mathbb{R}^{P \times d_ m}$

> Attention Output

이 Output을 이후 Backbonemodel에 Align하기 위해 Projection을 진행하여 Patch Reprogemming을 마무리한다.

$\huge O^{(i)} \in \mathbb{R}^{P \times D}$
 

> Patch Reprogramming Output

### **3.3. Prompt as Prefix**
시계열 데이터셋의 사전 정보를 자연어 형태로 제공함으로써, LLM의 패턴 인식과 추론 능력을 향상시킨다. 이때 사전 정보는 Dataset Context, Task instruction, Input Statistics이다. 

(e.g. 전력량 예측의 경우: 전력은 여름에 많이 쓰여, 너는 내년 여름의 전력량을 예측해야 해, 올해 전력량 평균은 ---이야)

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image2.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_F0HUI9VSS9Rt_VeZOQCbVXk9l01UtSQl7o3uo-nfVUlvmm94fJ3MWWNZjkblpb6tTk2Nw6t7Nd0PLUjX2-XAbFU0r-6daJRiOS-JO2EaRDB0wSKe-4_3Dlqh1tvzSB81C5-5KgfcP41EaSA11Cxg71EJ6Sar8lJEafa7YJrtPIq7HjMK84wOV-uaEBWZ-RDGJxIjORHk_5E_FHXpGlDpVzsi_SripjPmm9eBzg1VBfqDk_nqgpPPNJjxCeIYsHwPUN1A8cLqg1Mrs-FBD01Hio-1_4aKqm2NqkItNatRwLMLJVUlOVwpr_K4247bte2A8JyvcwmqGodjoWKjrUYFOgNmYkJc1TgtZOLuQ8oG5mBGj8F5dssKymoABR8ag1NKf-ZpJr3EwKSrmM4XhQFaj1D2gpEWVE-XbudvKtc1VbusrH_rJx52v2kqwUm9YDpbEvL3_7oEVIjLj0DJIcp4I9vlGA1b9URv5ansmBC2W_5EqI_kyXa3sVqL_hXk9tXf6AGz2Y3fZiGPF59PaBMIxRDfYJZznrBoDKFKE3VA0bmx2-FwPv3zHSH1JhAyH_IwE2sZR9IHS5aIa2IrvOQsQe5Nc1-va6rtG1z--_MrOF52EAxi0GrmarTkwRgYiper7FUq_QQx897msZKjtseSk_CltkaSB7pRQf_b_nO022YGqC69XKN5Vj5C4fYzyl--9nTz2yxVLeksfhQjuuxTYEDr4V34JzJxYAAQbmh2vesgY-ZbVFw7YGBwHo4mBDDqWywVUN2iRvkvKmnH9pntFaOl3Y7qogvEpvw40SBx2y5jCxTpjpULMokWX7kyQh02_VftB9JCbBkHQpE5ImSO7MsP-y-HD82T_GfEDrgfBaDiYw0YOemIAKArOF4VZxOaKIsoUcggAXDJCHo6ea_l6BeskX0DjnhxvtPVH9Cu3elPSqW-CfHPc7YrLsfgaG1ZFTULQqpBrIGCOfpm3Y-TGaFqTCo5Vcaadv20sJyjQSDuk8H6bGWIvyVQO0IU55gRdDmIK0XFd6qs8NyeF2ff70nPVt-K6fE4BGDnGxmhEYvyAxJzT8x5Y4NG7yfurNPFfIu27xlqUDiVkLl5xWYab9KcESNhyDOgAtKDVuJdnJN0RzxkjgilPUFR0ZMQkbFEXs8R80schsC03mkio25u656j5hyW69ajJDGatBx0QpK8hp7NMdry-K97uKEgZwm7gh_PWVU_x-IpexP5S5YrjRCCA9uNwlmjeOdI3ifR9VbXeKhA7Awh3tNTXUENq9G4dY02-1jyLPpzRzulMhhsmhdIsxYM3lg2uxMLOW4bXwgBfy0mV45fRY3vaAUpZEO1fzOZvE3pwU0RgYe5cEbE11zlLNYP3lgl-D4YpAobTl3OFsJ_csIxk1YbfuMoeuzhv8ebpkXdQ0R7QGUVu5OvV6D_wTA_6v1d7caWlW8rzbsn5EVSz_cLVlKGVKmSKVfGEVb5VjaszcAy_GbJnovG0H2ps3B6oC6iH20p8Os3190gPYS6Y4oviSLEeQTsV5-xB3IzaxUJ5IWy4_l11zCjwMgNxKYHwCRTmaae5stYX12GQezI_GBJ4ZHmQg7B11b4vK6Fj7CCLVdw=w1920-h919">
</div>   -->

> Prompt-as-Prefix

 위의 framework를 통해 진행되며, Pre-trained LLM을 통해 시계열 데이터셋의 사전정보를 Embedding 시키고, 이전에 Patch Reprogramming의 결과물과 Concat한다.

### **3.4. Output Generation**
 이제 위의 과정을 통해 만들어진 Final Input을 Pre-trained LLM에 넣어주고, 이후 나온 Output의 Prefix part를 제거한 후 원래 시계열 데이터 부분만 남기고 Output Representation을 진행한다. 
 
![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image3.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_H8fmxe9ZCVIcYp5eaNXZAPs8Z12BEyg2ezINMaSrhiT49y0-F_XVPdzGFO6ydeh5Du5BJi9W4-s0iwGkzm_IuWedUtiNkLM7MCFKKYQd74JVMu6vzIwIzjcY-MPnCWhTp3-VlXp1DyvQjJoVq00dxVRSvtCCIFQCaMX_eoAP_4b_Bx7dK5MlVHfmNoTmx3JCjKVlQY3AwRkszlGX6-18bouoe9tSvra_FIAImwO2oXfvAxkt34rw6RwdKxMtHSbLKcu3qeX6xTvWWZEpKKTLRGX_0Nm6jdvzXf4GgHGdKsjLP0y-1LIaHtjHyWekh9JxY75iYYNL_Rrudeuf-9jT37VEnrUONtu3Q_Z7CyUXclGn87aKklNoe59nI5TidYbTHT65dx4pyuTmyzBV4uB2FovhbXEkE00xsct5rBY840bBDcfywBSZDxWw6V3Saqye3e8zd9D4CInK3aXJb_BSTbzIG_3zHX92dAUX758N5hFtIbeHaAnoRaRAIECxKvALBCFmZthFBXHPnFLazXWpGDjEIyvafn3GeXGDVsYLKS25kn_J38YPueqVUiqq_KE5Or4-GMoD-sQZ3h_t9Y_qfjU-zI7Q0GzbNFKaFt2KAuo4HoCZv6TyQbvqFw1doP8mwH9WTsdlR8bKIxCKTNsY4AYaX99MGOPaSPCoT1aT75nfjXop969u4Psuvs_tEYYiwdJnloc5inCuNYymhmL51mrkbl4cpjeFSvOkXOUQ5hN5q14P1yqJ7HYV-VLrbD-Db_OUW3uqB8EYGXEtZCdbHoD6QQrzVUmdEITVDpjruZjceWv0zMAq6Kl_R-USZWKWh0g9NLvUbGkMN5ChiXjn0kHZbpvuVGu_4UU7T74RctMnZHz4BGAQQ--K9EPHMmu6ja-7NUUh16gXxfdUTBtO3fLjL0LcW0iENLwAVdIAGtCcjFFPXIhhRhM6sKOVCoIpUl6ZxQhnCIDV_jcjzC0XLwCcDuAyHKhkGpG9aJUybeIdd6qfotUD7d0DxZcfFt-uafxSSuHq2fUv6sXR9E8t5bDJ6e9TJkpKNeM456O-ckzIJiJC__5Kzh-7x669Tb6clPCkU0MbMDTOEXcqyczLQPDin3-bdQ2g3AhAU2w-q1Q45gGDzlkFPy5DIjpIoXIZzUVRDoGBlJ3kkcZD6IPePJxlTDQUEpzzZ9V9Mm5LNlxFGeskXmp0Askx21xmXz1XwzwW5Bx_3Or9aIumo631hi_MXsqsinU2_ilLMg4c8go1Yd6YGbOd_Q0KPQJEKfXXuUDONvwXDWQYe-R2UHhrcEddfaQKORqAAp0UjtqTF1X0xVZ6bI5yG1ZoxKdrobvkehlR-Nvv0V_c9Fmi7ni0xlLK5K2fJ16w043dDnyGWmpFhLMkPPUZYe4lnR7dVvWmEKzFR3fIMRUP16q3U6RcKSNDDreRLdiXXV05wnjG2_p1_cpvTnyIprX1-xLPFl9kJTuxRnr9qepTxoylhbuwxyVuYmZGFdIfWP8c9iKe5_WXCOEQ3uSsu_4IR5LAx2KSRFfODge8MH5nT13lZqrRDNzm77ASMvm6E1LEF9V9gpGNSBI_o5KOOuOcoVg4T5DWlgdQlkTuT-=w1920-h919">
</div>   -->

> Output Generation

 마지막으로 그림과 같은 과정을 지나 나온 Output 시계열 embedding을 다시 시계열 형태로 바꿔주게 된다. 이때 데이터가 Patch 형태로 이루어져 있기 때문에 Flat하게 바꿔준 후 Projection을 진행한다. 

$\huge \tilde{\mathbf{O}}^{(i)} \in \mathbb{R}^{P \times D} \longrightarrow \hat{\mathbf{Y}}^{(i)} \in \mathbb{R}^{1 \times H}$

> Final Output

## **4. Experiment**  
### **4.1. Dataset**
아래와 같이 Long-term Forecasting과 Short-term Forecasting에 대한 데이터 셋을 나눠 놓았다. 

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image4.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_EDFODBAxlmhc-1tWPNDbBy8sLgcSJ0mSTaKIR6QdUNhsUZtVooVGneHmY5QJgRESxv1p8ECIUOHaTWHsNWDRQaKlwwfMn2JQguJQofr0unmEsP0iWbt9XAo4_I16vn0gBg0xBZBXb4EQUj34ugsPm6nOlZmTfn4Fq9sZJy_e_jS_ZYSgDvtEJ05V-9wmHW_VdSkqRjXJrWNtBTuEXYBbtE7qPhjlgls6wC5YPiafFLVlfqozGYI1Ni6Yv6_yg6eFzsCOS9_3KTh9er4kC6D-H-ggAj4zglw3nbN8JeJCoYYwIQvjAfvLYpl4mgZtCtvNpnpwFsjifrKaMGpjXGxSUhoY0-QpUp3gBU4xelu8wdn6LKAgkZZi7xERfE0pSkrxlLAijCVyYFboU-Jq6t6jTig1pvjnhXY9VuNSFDlRk2mlZ9_wEwpNDnKg0hOjAjKI8Krc8kfX5p8zfURfsT48zzcnDA6QJfzGpKX9-rZgp5eTGaMNW77qNxxhdA5_6m_TTKNDb_-VeoSE2ZMHlKTOmLG3jTyg-U0aIXlDM_QtHIwJRmrbtaxvQ9CJOPT0KS5uA0mTHsfbxuH1-SJGePZ2j5osyWsfumYdx8LVTac2mBVg18rPbMfxrURswnBGFdfXwoSgTlDcWocmsKalVPtDX-Dxj8w1R1ys8XvD3FimW522Y-DM0aAJQM9chqi0k-NlLUpPZbntj5JoRFo5JIqIEOR9GAqLJhXWCQ9Vg0ySXNvjdPnpNBsdlwj8VozIgEcz0bNiX15e--Zb7GPSXIHD6lSW2gzGgdbURjVMPokxTeArBv2OQ2SD9S5srcxsrvUmd2eqDNDDJU4HjnUd4nO7J-_MYBrwteqSe1T_vMiJQnk773Q67-m1CmNZtKUoZ_H7UMgKrw9kdDp0D_Php1Sw1o2yNBktHxRcFXqmH_GmlaVYrgOhkthG3SoXnKW-viCl3EZwJIWYMFq71J5NYCOOW_2AroT8OkmNwMJTFr8g83heeG_tQwHWeIJ_XPslHEviNBUQgIO41-8qp3Z3vZun8LkplMil1-xW_2t6w6szWLuFWIsr4tVo8pO1m4hJVy6sHyYw_h2YCx-E7seNuHWv8n8TBCmNQ7nxIMj9Y0kWZbbV9uusWi_3Ov12Mg6nmh--gwADx4YXbtQnN4jf6AuzL2TdjiF2Jju9wJ-0ESn_Gkh6r2LfzjJEn9Dc9laIto1Hm0YZfToUhw-_NRy0024lZKb3vqninyarMp9nYXvfIMz5HX5D310gKJmEHPaql2lD--vv8sC-J10-8sSW8tId9JKm7LbeH05eqvSUwjpXVAt5n6e5ABxZ8bn8xEGvlSHNvaphFrRcniLE9rqAC1_V6jcIPObQ8u7ZwY7cUGPYE20D1g_HTLDaBU_mkjRVIVEfY0gsWYHezPB9jY0zXqYzkzJbhj7PujaqZ4_tMGJl6ZNEbsvqDnnY8sJZhP47e_gPcG9pqqJw8RpWQDVvkLduj1Tl4og2lK26wZle8qrz-WU5bchhSfVO2VpkSIwg4sh2SdSd3hVO9eogPtGkEOh1-kEo1Xpx5AnBFLCE9wWGklieMtDdLQNXjHrBQXwQHmwOY1cgGz03nW=w1920-h919">
</div>   -->

> Dataset for Experiments
 
 이 데이터 셋에 대해 Input window size는 512로 모두 고정했고, 다만 데이터셋의 규모가 매우 작은 ILI만 예외로 window size를 96으로 진행했다. 이는 보통 Input window size를 96으로 고정하고 ILI 셋의 경우 36으로 지정하는 것과 다르게 본 논문에서 볼 수 있는 특이한 양상이다.
 
### **4.2. Evaluation Metric**
 본 논문은 Long-term Forecasting의 경우 MSE, MAE를 Evaluation Metric으로 사용했고, Short-term Forecastiong의 경우는 SMAPE, MSAE, OWA를 사용했다.

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image5.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_G6HzrC9U6bExOGxx8MGOORhA1ZAeQ8WhfaRkTI39E_oJzhGSnSIEu-YsJDf1v-kmb4HM3PBu8Pcl8l2d9amwi7Lu28J-trGaMCY2QSzB1zvjE_d5L_Uj7kprZSaRH6F5gHi-vZ8myBqRfYEWB6LIJ9KYUitpPt8aY3XjJq8x65b8Y9oe9HTvlnCYajJyTcWuuQ3jVQ9UiIuyT3O1NlRgrX_5XW39UeENrbXuz1R7lFLAcFqn4tbyrKYS_5hCBndDFRTSe9a99K8rZPVmakvCTDc-XgvVHpjJNudeWr0_PzRnJq_g1_3IhwpU96mj0X6ljwxLmY1DApEOB4jEIecEiB5GlYsgkgd98WWXK3DGVea5yB1cxGNuqai0JuIgjkM1myAiIaSugePdDXCbbRAroEMgck09uJjgQr1vpfH_b-nfcrBar9w15gmegnc0XG1sgP3VCO1ZjCnVIGTaqiT2qfduucvt154FUfEV3sZ43z2jYjJHLKFFTg701CmaiIGN82xKOwhvdyYdhBcB2yx2L1PtC3-NJAa2Nk-PFZPE8n9HOG0KF8HhSj21SS5HXLVmp5RpNCoeySIAfIAOf45tt6MGfcVngcL9qJ7s6ahBkb65-Kwm5WLsrN8Resqt39E1T9-L5-DhJInifNHIzz3IbtvUBRio-vW3BRoVb-VB4VHVXtspEpiWvtm3G75WE1dvVEJRtDJyAG5YOV9IhuaKPuHRVRH4iBhIiPcO0XE37gW5-RB54U_neQBifxl4d0g1gaBntAdvYKVDCTFjHpXlmDI5kEWD0OoarxFnL456dmBSDUP5liw4kUMXDpWpRDo8q5IpWPq9ItxCTgXJLLCjimAv0d5DfW-0wZTweQCLu6ucZjWTvhkYgL9QDFgraOuORT6UgPuzwC-OrJyjch8yHmWK6x7o0ak4LNIZAyeAVEwA1d2kphioFbXTrzQvykXQQAUPPwWO_jBdZTDTsX4XIQziXF8SGYktzDM2MRtTFyElqwlDhsDq-1zGHu91UdFotuBH05kcCXzfEejY8SNJ8VhikkGu8iSJHxOJ5LAoMg7mF7dpdC0r0Fo4k-5_oJRqJPXxN99vtCIICz8C1lcpoI-h9Tycet7vuiLTUO_C98js82ku1uEF-xO6D6fq8Uw9JGiKxKYuCbNgEeR7nIbWOdlN2PkFmnRqA5mdVnSbVODNOoPZ-fqK64XvLvw5KtGiS9FrrRuKI0gw07ZcGZ79fd0p9O-aUgT8xs08wDx0YdE0Rfk0e7OuqnecQAi-QhBzsTts-EtJecBvUqgytFRKP6aeNRApna-WQhZdML7YiGsT-iyDPIoEJLXcMe5Z9cv8aB80xvSPlWDRtsrMZJ-BkARQtNlj_q5uWzKZ4-5XWTQV6iQe2oXuoIKTSl6u-WvfEW7vXjHlWXCyiLRxxAylEnlLIkxd2FeCx4HgergbSyOPDGuYj0ngW20qTmQFDwEwKudMgvYO0YM8MoWgGlMaGRSA5-w3lfoEwK6esrKquAN_uq1t7lW_j90hALfpOl4qvDWcKFQ767PIQzbHeBzrGqtYtIw08kT2rC-gXwJMOSLljOC3lQn1Lg-cELVBFzbfBzQ615NFpMig=w1920-h919">
</div>   -->

> Evaluation Metric

### **4.3. Baseline**
LLM Backbone은 Llama-7B을 사용했다. 아래는 Baseline을 보여주는 표이다.

| Transformer based Model           | MLP based Model      | CNN based Model       | LLM based Model      |
|-----------------------------------|----------------------|-----------------------|----------------------|
| PatchTST(2023),                   | Dliner(2023),        | TimesNet(2023)        | GPT4TS(OFA, 2023),   |
| FEDformer(2022),                  | LightTS(2022)        |                       | LLMTime(2023)        |
| Autoformer(2021),                 |                      |                       |                      |
| Non-stationary transformer(2022), |                      |                       |                      |
| ETSformer(2022),                  |                      |                       |                      |
| Informer(2021), Reformer(2020)    |                      |                       |                      |


### **4.4. Result**
#### **4.4.1. Long-term Forecasting**

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image6.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_GpWQxQHoNEOijHLapEbdg8YYTZFtE6OA29eiDY1zpzGYm8j8yLoTIrPHGJiN3LdBav9v2Fzoe5KYWy_HL1QlBT7pgSfi7fC81M2509Duc5sytheA4Tgi9Oi_yphVe9Vv4XO2MKzO-YIaytdLntaF72w-nB-MkLsFHUcwT5jIfl_LV7CJtIUaMlrvTQl51fFWBwfCr_vF2Rox94WYtnPw_KN_oAyGCkxeJTN1xcBUMpAWMqas3j3uf4MAaA7kfC3uKNmWhjrMy1yTrsrg0tlg28AvNfGarFjrVu9kwPs51EWPEo_ofCPmgjNNA5u4wHORX6q1lM7QoDxz_38vi92eCqwdjo9efePts2t_chadCJ49jRU-WvjJvxo3tZPP4Zp5lPptmqY7UYQQeIEeKDSrXsZ2HLvkcNe1V7-4fZKXClw8ulocXQCW0oR96bDqKLYInO6MpmDINvpsqIU7N1QqmcUj9xEaN8vRw1ib6n0ZI6CPDqYUNXVaVgBZzlsuQcmoQ1v29AusiERK8pNCU3D_EWyd89UcWRl4yiWXyc2cp3UNMhpox8Yor4dEGDKHFYdD7m8pC9T8z349n9Su8bcxcZptlS2R2vEyfJ44cgJa5uYnVoiUUsXGEG7c6kz981gjvlbDT1eTTAc6VzCn-ve8l_jyNoPwQTEyqKTSb92VAfhwSq7Lv68tryu5KJAV8Ud8hkA0SXoQPUDA3Kl1bONMfI5jvpAGGIItl0tTFaSEmdCAUTm6kEDMXWLynPIBF7okBUou1JQtU-JjJaq-WmZ3--w0rYE8MoTot-XtAaxq8krg2i02qMpWBpX4fwiuI3ySBbQP2x3r4zBLdy08CirRiSE-CUpCe1iXqNtZVVl0MtdqTPzR5X_pVzYRsOV4vHIUaWJ7t3lEk9HCLJF4lwgAOo5xNidAUfW-qpjFZCp4-cBxmkq1bhodbHmlbC8nPEaT2Zj4lIDoQ-Ft95X2R-dYtOVgKBDr59nQMoA80mvKOaDnOpFJgZ_5Rj0PtepmwtkIJM4xxrSFzU1QtCLleaSQH1XuJb2XlVjXiojJiztfuVMCw8FAp_2mHFC65u0psqJz54UaNZVUrj7l8YTa7f0hez3KeAC3WaSKDJEygQZpi96z8WVD483t7JNBhaqXgNV2ZJohs8w14yML6REBFH4lNE3yKhDtqrOBy3t_Wgrzn8g33INmG1Ff8MRiDv408CVj6-ah-LRAs5Q2WpjWeR-9C8ftMKjh9iVrl-v8tUbKC6kOWtzlpW7N5Sa8cpwWtWBeTv67EbQquo08sG15WMJiNP74JpyX_iSZY72XQMbPV_LF3eyNy9wqiztNfzL_4g4aCfMmQATe89T4aNRxb3Fx4nsLdA_64C9M_Ks2d_UG9KsUffI2moNDdCmHpgWWjhGuvqLSaifBoA0Ptm4qtq3Z7oRLCon_d0xu0EyAMDVg15ph2rieh-Hem85A91FqOUtOn9rFyHA4FlCJ4yt-sqvyck_vaUK7ZGQW1PyEP2FzvWDNKdyAepXCfriI7pvL3qATihM3ec2UpvmTc_w1pkD0vwsmoDYHwxfGkApA6CMp1odG-yyXSChf2iMVTjXuth9aJXsXm-TZjQ=w1920-h919">
</div>   -->

> Long-term Prediction Result
 
 여러 종류의 Prediction Horizon(예측하는 y의 길이)을 사용해 Metric 평균 값으로 정리한 결과 대부분의 경우 모든 Baseline보다 뛰어난 성능을 보였다. 기존 SOTA model인 PatchTST에 비해 MSE가 감소한 것과 본 논문의 모델과 유사하게 LLM을 사용하는 GPT4TS에 비해 큰 성능 향상을 보이는 것이 특별한 점이다. 하지만, GPT4TS는 Backbone 모델을 GPT2를 사용했기에 이에 유의해야 한다. 

#### **4.4.2. Short-term Forecasting**

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image7.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_HPHyo12MW2GMk2BVmDunSEB2fjtP3gqWx_i6CzkDji4EqNKbfHaRY_ruUjT7PHfe9C6pFVuw3pLid-Bedh7-gf7TueeWFjzzmeISbg2c_V_d4nfJ1VYadnnUi3Ro3PHQILEHFaxV5gdYXRZcw87JJVl3CVhU1lmiTKX7OvMY7Vrm9bIP8JjZFCgxqZgjvstk5vpYB4UPWuoxJObYKUkoNImtJb6ha8ES64WfEDCKbuTHfVxHJ8ZzGLiHDDdQJxIC0Uc0LFC-3gED4hPe-lHFxhxYYLXPYNsaGYn6vV34zCaS1nPVMUJtSyGQ5vFUpyWebU-Qa-I0zR7uO2sRZz0Qr2JwK_Uhhcatuvzdds7avGJdZNDJBqDuc_FvOWH0Mq0Rwm1yFFbXsBOFhN8JEyqOow-IFoAD01qC-Lnw1pp7vdvH09KPqvf6Kr7OoIG_unEv1vhAnnBxXERGbV4WJvA8fBZNer9JzjtlLLyKmQOzUKPqMtTOTWgL535e9eKWETUlDmdmH0lrY09eMRb1DiMbTgkXgy_IGiDDXXJWwoMim1YuLhhg2JifLLA2Joi6YNOMsA7b7euKofcdnnjjdIZHHUBCFitXwjxuYh46vQJOMxUlLLvH1YC9ZgjqWLzhQWRoJJNG44S2ZM7YSIAPjlyj6DAgOc28Mc_xZHokMMjZYuEbp7-UAPqFWKj9CsAWksR2XC-HOVnPrez-lCJYbr54P8oVjcQlBFKff5SGttrUeY6u3KJxscQrVYAdGssTH2BJ57wA9A-sIUWMCEZrop-ajQmQDpejGgoT5KxlsimZjZ_4HDch9hUULlZC7EtlMrxtDOezCb3_rJ5wTRsO4JOiTsXjAf8fKjxu5mcH1DgbBLPH9jE6Sx_H7cJuf1hJ7vdzXVWpOAELLQ6EqeYRzPd0fFrxTAJC2f2d73A5cy8S-zV1DTlQRGruXQTm4hI8qiZCQOOWzd16gVofk5Fq39_c-fWVNOujZ41buA6Cju3UO612GegsSZTFhPLxgBH6XezCwASeeHWx0V092nTGd-yaFMoYkT_gzIvNBh3M7mCiYk-QgTj-LNgwlYXxtXztV3Fu-3z1Y6EVMhNaZm4zAAORv9AwwTTKnjfw4Lk-4iYnaQQ0ZUwoQkMUCOJNU7NTvXD7fVJjyDvS6wI0oMr3-Rk7vdAWS1Q-2Un60pRFZR6W7MsQ7zUPchu5cFElkuNXB9tOW695FrkmXoEeWDBougx3O71CoS9z84FwPswrLDGQRa0zSpH91fqvjx61BA5KM2T8ni7qMlM8vwpenKLdyPQ0FBzYLNTqehzSKFwcpcR3WElPJYQ39AyUeCWNBMsyK_nipFg8ABI3OwlX9fZzbpSIlMQ38zJXlOjxKojVhm8UjCIB9yHYL3sEAUFVPovbN7dzzOrng-_bhp5w7QKS1Oeyu_WZ80GSGoYaIZcuCPxKBhWgZBROyn5OgaUiWcHIPBEYebCMt8K2v1-5xzRJsB9qSFWIyUEMwoZZnB2t1hM__4q88SeqcPGILMvZybvzmArTQh1xYXwdb9eV67cToKQd8HMEeZLsSB_Ys8T7cI94GQ9pxOuyMHXf1b8MiA2lH2gcdrklZqJLrUcA=w1920-h919">
</div>   -->

> Short-term Prediction Result

 Short-term Prediction에 사용되는 데이터셋은 Prediction horizon을 6과 48 사이의 값을 사용하고, Input length는 Prediction horizon의 2배의 길이를 채택했다. 이 경우 또한, Time-LLm이 기존 SOTA인 N-HiTS와 비교했을 때보다 좋은 성능을 갖고, 모든 Baseline보다 뛰어난 성능을 보인다. 특히나, 같은 LLM을 사용하는 GPT4TS보다 성능이 우수하여 장단기 시계열 예측 모두 더 좋은 성능을 갖는다고 말할 수 있다.

#### **4.4.3. Few-shot Learning**

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image8.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_G5auJGqnm-o0xWvURe8VtU_EreqZ4vs501dy3Keak54-wefOGdelWmRaayXISUTaaRt-WC1niLR6nInaFklU4gUlDQGH2qeUKooSg015fpLQjtQ36SWGvodqTrftsHaiKrE4PmXJ3rAb9HFO6zWRZC-3cDsr6GkO-XLSXrWy4NbHXj9FcWR_1CjMCylVdlhQFH9KVvqbMsBXFlqqguNDMyZKrRiM28gG_sGaLA0NNrMGTFcMVGPOkTHrCidwAm2THZomHapM3fksTHepYIoxf4g-ZeIn-U1ynskWlug5jA2kU9B6f5A-1KnXPzvuqITu0v32TT15qJhY8NYHrmktOZaPKWGPHtQY3ia8bf9VCly4nYBUZV6R8FWRfvBcRaZBuXCp53JmG6YwYQeldG20QAs-9ZIGETaRS9bO_hSrGZ5SBOCMNdwX7thgV_95cTZLuYaUBou1tTmjsX_ArrYbULyrFCtBEtTUcEmGOlxP2GC2vKMB6Rt7g5hBcjuLb0EmbT4K24iY530QvjXdWmK9nWq9wSbDUO5Ro5RABhU230_ZZ1iI5zEnYds5itonVsBXnEynl1pFJE32SYch7xcmaDaAAZ5wmTe30XsBiyBJLJnnVb7bhDoqrjRoW3n4iVv5e1gQGRNueIolN7Xfg-QNVyDKhQZhIAk6A6bqc3ThvR0-uJWp303ynpbvS4LaGo2pK2TD1hNcSWSSetzhzRYYYXyu3zDHCSnZS9Z0__L5FMSaofbwi-w9VUqkkiBCiwxuDyOiNmULfXe0xAdB5iE6PUCgoKzuKOFdKDI53FBzUJmyLvO-KXmeHnn_P1paY_yOuRDNd-Rmd46rY8MtjIk2jyWHxYgAjKezcznVBOWp1ZcLiFXLTYkc-tGe7DSlZCdQBvbTSRPCYtNfXY8COL9brX-vFHaLWvyHviyh7SonmBCwy2iyK-Pm_YzeZGTmWEmEuHad1NjlRjr8vdQz1PnVqsHMDkfaYvwo57cu3Kg7NFNHwiRAm_Auk7btmminDgkGnsLfUTwAeehSMV0WgMSSpH5yKm5vSmsB-r3ychTbcV-DAPN-zELkExJp3IKD52M9WHpX31_8Sj5DrakeSoJOj-PVArEsEA6MqZvlTZyncPrtFZNe5BdtP1IwxVDZNfI_ldMnVLQnk5O0RtX_Y5dxwUapn8hmliBupdWs5rlvkoJXQqJDViw-t_HVzN_S5DwnIKdDw3BBksFQmQh3qiOC1A46V0qGHQvtAlZWUxq3nxdcog1Xefs83x3dks5HY7zEOoGAlJ3wIre1LXDCYzniJYbATYWbKN0nvOOCDZwyfSeuWIx5K_CMXE7HgY64OFoWvJ5JXu8OFrc6HGjTZAEB7tX4tOAAEpy5_pEoeTTli8ItYBhf42qVQD_CqlmW357-Ae7RSv9K4XHCxtm8DP2TzaxjARemfrY5Cg-rCX-exHLcRe-9-9_0WuLkW9C5GCfBQ1l0cFgu1mORrZa8t0tn3l-ncvliugkFS0liYZRPHsUsEovlOqnPhblq1zx5QXYVANiNlVCUGFbJ49TP1RH_99uew9v3oXoBjGH1DA4kznUGSvUEPkCBXvx4qbFzRH7XYx1ViT_nX92Q=w1920-h919">
</div>   -->

> Few-shot Learning Result with 10% Training Data

Few-shot learning을 사용해서 예측을 진행한 결과 대부분의 Baseline보다 좋은 성능을 갖는다. 또한, LLM 기반의 모델들이 좋은 성능을 보이는 것을 알 수 있는데 이는 Pre-trained LLM이 자체적으로 뛰어난 패턴 인식 능력과 추론 능력을 가지고 있기 때문이라고 말할 수 있다. 특히, 위 결과를 보면 10%의 Training Data만 사용한 Few-shot learning의 경우 거의 모든 데이터셋에서 가장 좋은 성능을 갖는다.

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image9.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_HiuZYla4SM86JSHdZwuUbYxP8DhK8c45Oqhlv52WnIt9sVlEdDD2Cd86CMmgKEk4wbsPMllo_aiV6PMznoikfCJRXC47GaoYUBjYF5EyFd1PfE7S8m1UBHQguyKLVNKvaC2BV-ARyzBLkOcwGAkj53AdQU6PnsV9_NsBPY1CMnNY_jefdL9eLiMBUkzd1n16oYB2wSArHZIV9A4A4UgiKol9GUY3yc1qyYMF5EzJRsiwOkMkdG51tP7qlYgPoZ8gme_VdMSR6-9m_tGm5lvob6bVx96kYxzCMvb6jY6-bbHV_9h2CEMUXAZx0v7UxL9JXvXDNd-ewB_0gW72U_Tq2jeQG2es9RRavLk8JjlUQlFnnrIK8FImsvFNIuqDaf_md1aqRYB7XYbipTjZwydUmaiv5F9kCnj4zvdRJnPnTzVaIdDsjUdo2vAPx7TksDBMZCNmb52KUMOES4Zxa10onsCEPzOmymXJfWndNSzICoprOzMTG7R1fc0YHYYF38qGEhdfI8iyw5DIT7nIktnK7cyaB_WOx3skJcK6Q7Z7ZTM1wfSAERRfe9VPoeH8t30TMjSjKsdC6a8ZW8kqk59aMU3vaE1mggSs8lLXfEe-u6b7Svv7VPXzcM5RBpqv_EdhE6-sU0HwsOhP8pf0RzzY_X--K4JA380KVZwzyxDVgf-Fl2qQOOzhTu--61cbrNQlTnjzRsgL5owBFUedl6Pv-NGWuQ88pDchZc8X8YxmBBZp9poq-Ho6UJrMsBCURMqU6q_nmv9JV-BVb_qxpwGpRiA_LifSfekccCpHuQ4jNMtcHe-ue5-5qbZhFO4xUVFrv9maXCo_86roAqBILxNI0YD24lVA5t0JdRQLWgahq8_ZMIX2cLnuOCsDMC2UeIIl4Obum40fA5PXtenwAfs-8djqp8mS2pOoQaYg9n3b9bGFmMLqcZeODiKMbCUVDEFOwJohU4Rba8UXI034ECEyPqarUsE4WSmQ28gGskUj3lPSJmeo21vZhJntQWhRVm13-2_DyeHJt6bNK7A123ZXZWQQ4VX2FSDe5PfXaYKTmLi59DvF2WCGloT6UBS5tEfbCbi9YNJYd0s4rydC2dBMCYeZs-0qpPu-h6BykeVFta8TYzkvKwBuzm3f23Sm4W1p7LOxL1c8M8KE6-ch-7qyf0KV7pAdUfkIOUeQiXg-YIt2HGxbQtH87kesJvCG50WygsUfHNNxPYCxQBf117gTy7DBDbvhQwVYIuc4TSQ4jnbbD8LlmcAMcB8RhOvpOCJ8FnfbpENbVMa-1vZE89vk0P_8yXl6hdNd9c0ookCT3zcBStYaN9tobKP9noW9hFpg8_ndTSUqCtLFYR39F8W6G4FBp9lb2Jm-YcyZ7UzXah4xcqrmtqGvq6hQlrWNc-5ufgdN69vTREwk9rUCRbhwX9PWcX44Z5U5BP5pkIedGr2EQBFlwKDscxKIWF97W85oRx_a32vY5bA1Fi38GIed0mo8y1nmIMv-y4CHj3z-C9nDdcC5VzEPyCsR_keJGNg2KxlrObPMsi8dZzUwYSMi9v_Bn4vjRNYgdNgUbQj54Jy-pTBsZaLiZldF3MQpxbPsKnLNzrWQgbdQ=w1920-h919">
</div>   -->

> Few-shot Learning Result with 5% Training Data

 5%의 Training Data만 사용한 Few-shot learning의 경우에서는 종종 Transformer의 변형 Model들도 좋은 결과를 갖는 것을 보이는데, 이는 LLM의 패턴 인식을 활용하기 위한 데이터의 양이 비교적 부족함을 의미한다. 

#### **4.4.4. Zero-shot Learning**

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image10.png)

<!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_G1XibOUXSRy7dzfWZHt1gLhE3qCr1PDLTNThuoGLMm1e3_K6Dqh3Um5_LPSl_UKpmnbEncHfc0V3fIl3WuHejx6XedVc6LInoye_7fS_uTXaNau5VcAS_D7-C3pOrK4tXQViIDzNr1yP513nHLrIHYlmEHbThMFTpstCHFC8YH2cTs0fLl_DUPrHaXgmqMgXWlvPi7VejTTI2i2P1AcNMS8SR9tguimWbh0EaTo3bmsz1usOEeJCd_oPiCeAkJwlMFFWVGPqtPxF08lWMs1isDhUysE2zBXATi_NPp5x9XolN_oEy55idIUX27O1SqWbvKF9vwjQgzeIlYUFqnpLJeGGm1evFrG2xTsTFF4Wl_izENHlppe7ynNGmh3fYFtZtD6BLgEtJ3e_i8opIARUWsg1cZ_z-x9JM5x6BafC1RLxNxc_pnVAVBd7g-Ju5DJC40vxC-MvI5srTBNWTXSGXG9hyChnFJUE1jBVr1tGZeOpdOL0jpViNIRBdq_P3_GaLjhBj6MyDXQahto7ipbHhTXWEctqd4Liz0eb6y5racWAcaeIHX_Dwsfkf_2PgOvcMU-6rsDPyeuRIRv1F6qY2l9vCVxDI2cv7sbu1MOnqA01gZhRLsHUHyHnc8iat1DkPZR2CMpf1fmgxJlbld45-DFkMCo6bUy5oy-72TZXI8jHPrg7CEZUrcgLM1sbYlqZCbSV7rMKouiDihjwl5lZVFxXLohLbNO5XuL65JUPHZ5XvzLXphqMCYs0wtFEq-SuAHhcCFWPwu6XsFI27ij-jm1VvoOSbnlbjZioBtQkpBwi0D-J7ge_yKU_0ZQLx2O13EMEdIzEdglM1-Tp0-jPWYg5InIbyH168Cn-P-TozWmNl7-2T7EzsQZZPmo3Hi8-4RBa3BfcJfDcw-KShIwegoaCcm5Qp0vXOMQacWNC9dZQ2USg44OzSdwL1JzaRUuMuu4_I9cZAiPDtEgiQxqGNmOjx_KS8LDWGYSV-EsPl7DxJosN7YvdhRw6N-HKNmiF8Ufh2TuuAUSa1qWLx_r-DBBBriMDdxFIWosaEk0HdztFdM0f-bWKX5L7cZy7E7Dy9e_jP6xJXIjP73WISZBPecPA-G__UxOBHOYa49Ggp-v_YvHPiGsid7QjCUT65giCkYVdJNYcT02fXh7wEI23IcZ5e3R7RY1NmaRp4ZdcUeTMc-v-BDNglpCz3zR0Ah3J_AL79XiZhPi3UEC7B5LnizJi0sXLBNzfUCglD0bEkVd_R6lOpLJE5ZPfiyS8xfiBwqmy4wAXTaVCvKV83ovnSQq8I7BOs0Qlk-rR4F3Yjq5FoprDzKAQKbyOUDN2DN6sZ5az5hNllw-mcd5-xKhnUQtPARkGaXPUUDuNHzwsYkiSPbieU6HW2BnFynzsOdtsCfaDiMFkowxBwleEYM9Qj5Y8sgi-9q12UlvdG5Dc7Z-y3ITe-93DQLYkE76XOCGNaSeHfk9Bw4keV-t-9cf12BSZKSbzkqisqfRzk2AP-WCSdQWhr3gWZ-36_IrgjMb_iQUWDYeNA1hWtdy9r4RGvFjJBzR5Pl88F_aWDsa3xs2iZP4F3-wvqvs4E-f9Kq8YM0hYD4ywSizA=w1365-h919">
</div>   -->

> Zero-shot Learning Result

위의 그림에서 A --> B라고 적혀져 있는 것은 A 데이터셋으로 학습한 후, B 데이터셋으로 테스트한 결과를 보여주는 것이다.

전체적으로 모든 Baseline보다 좋은 성능을 갖으며, 특히나 다른 LLM 기반 모델들보다 훨씬 좋은 성능을 보이기에 현실적으로 시계열 데이터가 부족한 상황에서 더 유용하다고 생각한다. 

#### **4.4.5. Model Analysis**
 이 파트에서는 Abalation analysis를 진행하여 Language Model별, Multi Modality Data를 처리하는 방법별, External Info별로 성능이 어떻게 변하는지 결과를 보여준다. 특히, 본 논문의 Framework는 잘 따르면서 LLM만 Llama(32)를 사용한 것이 가장 좋은 성능을 내었다. 이를 통해 결국 Backbone Model의 성능이 시계열 분석의 성능을 좌지우지함을 예상할 수 있다.

![](../../images/DS503_24S/Time-LLM_Time_Series_Forecasting_by_Reprogramming_Large_Language_Models/image11.png)

 <!-- <div align="center">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_Hfz13PEddITymSpkbnpX4M9kCS7x_WEwslHm6vKR8CPXdxsUlnr7D4HsHAwIQUPuB9d37bD86KQsgEbHI7YxhlZEeO6_9ulpUi2UONhEcddM_dFsO974jGmabyjkrjzoV4B33YbVDJ6lDcdZ22foqDAxNwGYP7K6LQVhbL__mEafY_DRvUCNBPeRomcz_Woam8RHEB04jjsfJVJYhtyGFskmIlP2JYJWesNoAN3srsfLo0icXOFMZsOc_-kesuh6YMaCbnDHexpSyiydv2YtmC0hL4YxxBqSJCtB42qlwG-Zk4jQXPKrU-VoxHGQvCxCwGLR70hPjvsbwJolaiuOyxg1qG5eHDxqhI9yXWc6cnBVB4GZtjFrtue5wAAnpb9PHgIg0qqxaN89yKNemKTJqjTJzlxTkSaKjUgWy3mP8bHdAnFilH53VWJvGOtiTqSZf9ncwLNYUqsDSxLqx5cZ_pGWnJevpVPEaHs09Gjc87GNTtgcZHm-IiJvtFeSEVqEbsAfMC_NOWAfnkARgoLvx3bhNqul1tnV07gWMQOHok9OD9QcJbO0UvA9q1iyY8YVi52Au0gkt0dAyQMiAqhnBGTQIZseHdsvxc9HJV-99FqYFIi2ihqNNdiyjlpyog5utMHdo9j-iMlSrqRThr5OCXdKPntjwAuRLjfSP5jtbevIVuqI_Zu2jmdyomfj07f-4mDwtsYsX7NA7Cs6eKbkV_66vE5-x1TpjC55PrW4Gzxv5X30ZsEUS8GpPtlXkijeV8a5IT4oxgtHDqB_kabzEUDI0h6whHgYfN23lME0UXfVf5EgmHrc5pqRq8Dc0uwvI6JT6nN6Aje6jKlKHkfbCKviQdT6IfE7eJvmk69ic-tYc__InZ75hwYugQ-X4m2ZMu4U5T9WoScQvcfCUonNYFKRMhZGX4d3k-ItTThRfMISST9sOAb-Q9JarJJ7ZLuUMjI-ujauGPGyxEBjj4oJ16h51JtIZ8VPBTQTiBGbKJ7SLcZpsBXKjl1F7Ve8MQhD-uO8fTYQz2-SxJmtfGBIVd9ITOirs6P-PRG9Vhk_Zg2NPlzjDUNu8QK2Ro-P-bL3hNrUpS4ILv4D-fniO0DbE8NkNowhjshy1ER4IZEzJ6v-ttOEo0dixa_9Ynhyjjknhx0YsdvdizANY_tZrKlZ7R78GK7Y2rtRQzNyJ3y8alFjnXWBoXfE8PuawLgFhOEVRcSg2_oGVH3t6D7rASFO_E_-zW3f4pvnR-rqIk5KZEgdr1JgcyyOqdXsY3pAhWJaj2x5gr-T6h-LNYaLlVJMBxGtne1avJt6HVVhGNQgixEZa-UuhVKrVx1IiQ2avW95scGdzHRqrTSLOLV2frH6asPmcG7N2T41JDK_XomflcKwHCsDGwKxWsGUEyHK7IZOLg7cNH--c0DNZqNndJdCY4LuNV20j5-K_4lYsJC42zb3iKgCUlD3wHEWGwMdCdVfCJNrlbR8T0OcqJKUWCV4_b-4uSbZN7qKOs7GFDJEdeQBAWhgGX2UFdRf7NgJ3LbTjY3gOQI1ZbWvNZl3D-7XHR16F7S8O0fHfz1MNc3rPS2zzHZSTS-eHzn4pyt4Oa9z03L9Ida2IL=w1365-h919">
</div>   -->

> Ablation Result(MSE reported)
## **5. Conclusion**  

#### **ㄱ) LLM 기반 시계열 데이터 예측 방법인 Time-LLM 제안**

 시계열 데이터와 Natural Language를 align시키는 Patch reprogramming을 선보였고, 나아가 LLM의 성능을 높이기 위한 Prompt-as-Prefix를 제안하였다. 특히, 이전의 연구에서는 Natural language와 시계열 데이터 간의 관계를 크게 고려하지 않았지만 이 연구를 통해 직접적으로 연결고리를 고려하여 성능을 높이는데 성공했다.

#### **ㄴ) Baseline보다 더 나은 성능**
 Long-term, Short-term, Few-show, Zero-show에 대한 예측 성능이 거의 모든 Baseline보다 좋은 결과를 보여줬고, 특히나 Zero-shot의 경우 다른 LLM 기반 모델에 비해서 크게 뛰어난 성능을 가지고 있어, 현실 세계에 더 유용하다고 생각된다.

 하지만 아직 LLM의 무엇이 시계열 데이터 예측이나 분석을 잘 수행시킬 수 있는지는 알려지지 않았으며, 패턴 인식 혹은 추론 능력 정도로만 가정할 뿐이기 때문에 앞으로 더 깊은 탐구가 필요하다. 개인적으로 Backbone LLM의 발전은 결국 성능의 증가를 불러일으킬 것이라고 생각하며, 앞으로 발전된 모델이 나올 때마다 시계열 분석을 진행하면 좋을 것 같다.



---  
## **Author Information**  

* Sanha Chang  
    * Affiliation: [iStat Lab](https://istat.kaist.ac.kr/)
    * Research Topic: Spatial-Temporal Data Analysis, Deep Learning
    * Contact: jsh0319@kaist.ac.kr

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation
  *   [Code for Paper](https://github.com/KimMeen/Time-LLM)
* Reference
  *  [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://openreview.net/forum?id=Unb5CVPtae)
  *  [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p "Reversible Instance Normalization")


