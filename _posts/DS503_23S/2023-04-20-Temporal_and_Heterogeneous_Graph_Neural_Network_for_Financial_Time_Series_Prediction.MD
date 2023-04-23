---
title:  "[CIKM 2020] Temporal And Heterogeneous Graph Neural Network For Financial Time Series Prediction"
permalink: Temporal_And_Heterogeneous_Graph_Neural_Network_For_Financial_Time_Series_Prediction.html
tags: [reviews]
use_math: true
usemathjax: true
---

# __1.	Problem Definition__
주식의 가격 움직임은 다양한 측면에 의해 영향을 받는다고 알려져 있다. 선행 연구에서는 가격 움직임을 예측하기 위해 기술적 지표, factor, 재무상태, 뉴스, sns 등을 input으로 구성하였다. 주식시장의 시그널을 stock fundamental과 기술적 지표를 사용해 의사결정을 내리거나 뉴스 기사와 관련된 기업 간의 연결고리를 구축하여 주가 움직임을 예측하는 등 선행 연구에서는 LSTM과 GRU 등을 활용하여 과거 정보를 학습하고 이를 다운스트림 예측 작업에 활용하였다. 하지만 기존 학습 방법은 각 주식의 시계열 정보를 독립적이고 동일하게 분포된 것으로 간주하고 각 분석을 독립적으로 수행한다. 이러한 과정은 각 주식 간의 내부관계를 무시할 밖에 없는 결과를 내고 이는 필연적으로 최적의 성능을 내지 못하는 결과를 초래한다.  

# __2.	Motivation__
주식의 가격 변동은 그 자체의 과거 가격과 관련이 있을 뿐만 아니라 공급업체, 고객, 주주, 투자자, 연결된 기업 등과도 관련이 있다. 이러한 관계를 저장하고 표현하기 위해 knowledge graph를 활용하였고 최근에는 그래프 구조의 데이터를 효과적으로 학습할 수 있는 GNN이 방법론으로 제시되었다. 하지만 기존의 방식대로 구축된 그래프는 수작업으로 만든 라벨링과 NLP를 활용하였고 이는 기업간의 관계를 나타내기에 제한되며 많은 리소스 라벨링과 낮은 정확도가 낮은 어려움이 있다. 

실제 기업 관계는 시간에 따라 유동적으로 변화한다. 또한 기업간의 관계는 heterogenous하여 기업 간의 관계 유형은 다양하게 나타난다. 따라서 기존 방법으로는 실제로 기업 관계 그래프의 모든 정보를 표현할 수 없다. 이 논문에서는 이러한 문제를 해결하기 위해 실제 주식가격 시퀀스를 기반으로 relation graph를 구성하고 sequential features와 relational features를 함께 학습하여 주가를 예측을 하는 temporal and heterogenous graph neural network(THGNN) 방법론을 제시한다. 

# __3. Method__
<img src = 'https://github.com/hynacin121/ML_Paper_Review/blob/main/img/3MethodAlgorithm.png?raw=true' >

THGNN모델은 historical price sequence를 input으로 받고 probability of stock movement를 output으로 한다. 

### __(a) Stock Correlatiobn Graph Generation__

모델의 첫 파트는 주식간의 상관관계 그래프를 생성 하는 과정으로 각 거래일의 주식간의 동적관계를 구축한다. 주식간의 상관관계는 각 주식의 과거 주식데이터를 활용해 상관행렬을 계산하고, 행렬의 각 요소 값에 따라 기업간의 관계를 결정한다. 

기업간의 관계는 postive와 negative 두 가지로 분류한다. 이때 그래프의 노이즈를 제거하기 위해 correlation의 절댓값이 threshold보다 큰 값만 활용하고, correlation > threshold 일 때 positive, correlation < threshold 일 때 negative로 정의한다. 그래프의 edge는 앞서 설명한 relation으로 생성되고, 각 회사간의 heterogenous graph를 앞서 설명한 두개의 관계로 다음과 같이 표현한다. $G = (V, (E_ {r_ 1}, E_ {r_ 2}, \dots) \  \; r \in (pos, neg)$.
회사간의 관계는 동적관계이므로, 기업간의 관계 그래프를 temporal format으로 생성하였다. T 거래일의 그래프는 $\tilde{G} = (\tilde{V},(\tilde{E} _{pos}, \tilde{E} _{neg}))$ 로 표기한다.

### __(b) Historical Price Encoding__
Input인 t번째 거래일의 price sequence는 $X^t \in R^{n \times T \times d_ {feat}}$ 으로 정의한다. T는 t번째 거래일 이전의 거래일 숫자를 의미하고, $d_ {feat}$ 은 과거 주식가격의 dimension을 의미한다. $X^t$는 linear transformation과 positional encoding 과정을 거친다. 

$ \hat{H}^t = W_ {in}X^t + b_ {in} \\
H^t = \hat{H}^t + PE \\$

$\; PE(p,2i) = \sin(p/10000^{2i/d_ {in}}) \\
PE(p,2i + 1) = \cos(p/10000^{2i/d_ {in}})$

위 식에서 p는 거래일의 의미하고, i는 dimension, 그리고 $d_ {in}$은 input feature의 dimension을 의미하고 $W_ {in} \in R^{d_ {feat} \times d_ {in}}$과 $b_ {in} \in R^{d_ {in}}$은 학습 가능한 파라미터다. 

$Q_i^t =H^tW_i^Q, \; K_i^t = H^tW_i^K, V_i^t = H^tW_i^V\\
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_{in}}})V\\
EncHead_i^t = Attention(Q_i^t, K_i^t, V_i^t)\\
H^t_{enc} = Concat(EncHead_1^t, \dots, EncHead^t_{h_{enc}})W_o$

앞서 Linear transformation과 PE 과정을 거친 후, Multi-head attentional transformer를 활용하여 주식의 각 input feature를 인코딩한다. $H^t$를 각각 $W_i^Q \in \R^{d_{in} \times d_{hidden}}, W_i^K \in \R^{d_{in} \times d_{hidden}}, W_i^V \in \R^{d_{in} \times d_{hidden}}$와 곱하여 Q(Query), K(Key), V(Value)를 계산한 후 softmax함수를 적용하여  $EncHead_i^T$를 구한다. concat은 Encoder head들을 concatenation하는 것을 의미한다. $W_o \in \R^{h_{enc}d_v \times d_{enc}} $는 output projection matrix로 concatenated matrix에 곱하여 input으로 들어올 때의 사이즈와 동일하게 유지되게 한다. 

### __(c) Temporal Graph Attention Mechanism__

Historical Price Encoding의 output인 $H^t_{enc}$와 Temporal relation Graph인 $\tilde{G}$를 활용하여 다음 단계인 temporal attention mechanism을 진행한다. $H^t_{enc}$의 모든 노드를 1차원으로 flatten하고 2단계의 temporal attention mechanism을 활용한다. 각각의 relationship인 $r \in (pos, neg)$에 대해 다음과 같은 연산을 진행한다. 


$$
\alpha^i_{u^t, v^t} = \frac{exp(LeakyReLU(a^T_{r,i}[h_{u_t} \vert h_{v_t}]))}{\sum_{k^t \in \Nu_r(v^t)}exp(LeakyReLU(a^T_{r,i}[h_{k^t} \vert h_{v_t}]))}
$$

$h_{u_t}$는 $H_{enc}^t$의 $u_t$번째 열을 의미하고, $a_{r,i} \in \R^{2Td_{enc}}$는 i번째 head의 r번째 relation에 대한 가중치 벡터를 의미한다. $\alpha^i_{u_t, v_t}$는 노드$u_t$에 대한 노드$v_t$의 중요도를 의미한다. LeakyReLU activation을 활용하여 계산된 Normalized Attention Score는 아래와 같이 i번째 노드의 neighbor importance를 결정하여 input 데이터를 재정의 한다.

$$
TgaHead_i = \sum_{v^t \in \tilde{V}} \sigma(\sum_{u^t \in \Nu_r(v^t)}\alpha^i_{u^t, v^t}h_{u^t})\\
H^t_r = Concat(TgaHead_1, \dots, TgaHead_{h_{tga}})W_{o,r}
$$

$\sigma$ 는 activate function으로 sigmoid함수를 활용한다. $W_{o,r} \in \R^{h_{tga}Td_{enc} \times d_{att}}$ 는 output projection matrix를 의미한다. 

### __(d) Heterogeneous Graph Attention Mechanism__
앞선 과정을 통해 3개의 종류의 임베딩($H^t_{self}, H^t_{pos}, H^t_{neg}$)을 정의 하였다. $H^t_{self} = W_{self}H^t_{enc} + b_{self}$로 $H^t_{self}$로 부터 도출되었고, $H^t_{pos}, H^t_{neg}$는 앞선 Temporal Graph Attention Mechanism을 통해 도출되었다. $(\beta_{self}, \beta_{pos}, \beta_{neg})$ 는 3개의 임베딩을 input으로 하여 아래와 같은 과정을 통해 계산된다. 

$$
w_r  = \frac{1}{\vert \tilde{V} \vert} \sum_{v^t \in \tilde{V}}q^T \tanh(Wh_{v^t,r}+b) \\
\beta_r = \frac{exp(w_r)}{\sum_{r \in (self, pos, neg)}exp(w_r)} \\
Z^t = \sum_{r \in (self, pos, neg)} \beta_r  \cdot H^t_r 
$$

3개의 임베딩을 MLP를 통해 각각 transform하고 heterogeneous attention vector인 __q__ 를 곱한 후 평균을 구해 $w_r$ 를 계산하였다. $w_r$ 을 활용하여 3가지 관계의 가중치인 ($\beta_r$)를 구할 수 있다. 마지막으로 가중치인 $\beta_r$ 을 활용하여 최종 임베딩인 $Z_t$ 를 구한다. 

### __(e) Optimzation Objectives__
목적함수는 다음과 같은 과정을 통해 계산된다. 상위 100개와 하위 100개에 속하는 200개의 주식을 선택하고 해당 노드에 각각 1과 0으로 레이블을 지정한다. 그 후로 한 계층의 MLP를 분류기로 사용하여 라벨링된 노드의 분류결과를 얻는다. Binary cross-entropy를 사용하여 다음과 같이 목적함수 L을 구할 수 있다.

$$
\hat{Y}_l = \sigma(WZ^t_l + b) \\
\mathcal L = \sum_{l \in \mathcal Y_t}[Y^t_l \log(\hat{Y}_l) + (1- Y_l^t)log(1- \hat{Y}_l)]
$$

$Y_l^t, Z_l^t$는 라벨링된 노드 $l$의 임베딩과 시그모이드 함수값을 의미한다. 계산을 위해 Adam Optimizer를 활용한다. 

# __4.	Experiment__

## __Experimental Settings__
### __Dataset__
SP500과 CSI300의 2016년부터 2021년까지의 주식데이터를 활용하였다. 기업간의 관계 그래프 를 구성하기 위해 활용된 correlation matrix의 경우, 기준일로 부터 20 거래일 전까지의 주가를 활용하여 계산하였다. 

### __Parameter Settings__
$\tilde{G}$의 경우 20 거래일동안의 기업간의 관계를 담고 있다. $d_{feat}$은 6개의 encoding layer로 구성되어 있고,  $d_{in}$과 $d_{enc}$는 모두 128개의 encoding layer로 구성되어 있다. Attention function에서 활용된 $d_{hidden}$의 경우 512개의 layer, $d_{v}$는 128개 $h_{enc}$는 8개로 구성되어 있다. Temporal graph attention layer에서는 $d_{att}$은 256개, $h_{tga}$는 4개, 그리고 heterogeneous graph attention layer의 $d_q$는 256개의 encoding layer로 구성되어 있다.

### __Compared Baselines__
Non-graph-based approach인 LSTM, GRU, Transformer, eLSTM과 Graph-based approach인 LSTM-GCN, LSTM-RGCN, TGC, MAN-SF, HATS, REST, AD-GAT가 비교군으로 구성되어 있다.

### __Evaluating Metrics__
ACC(prediction accuracy, 예측정확도), ARR(annual return rate, 연간 수익률), AV(annual volatility, 연간 변동성),  MDD(Maximum DrawDown), ASR(Annual Sharpe Ratio, $ARR/AV$), CR(Calmar Ratio, $ARR/\vert MDD \vert$), IR(Information Ratio) 총 7가지의 지표를 사용하여 기준선과 모델의 성능을 기록한다. 


## __Financial Prediction__
<img src = 'https://github.com/hynacin121/ML_Paper_Review/blob/main/img/4ResultTable.png?raw=true' >
논문에서 활용된 모델인 THGNN의 경우 모든 평가지표에서 대부분의 Baseline보다 우수한 지표를 보임을 알 수 있다. 이를 통해 금융 시계열 예측에서 THGNN의 우수성을 입증하였다. 

## __Ablation Study__

<img src = 'https://github.com/hynacin121/ML_Paper_Review/blob/main/img/4AblationTable.png?raw=true' >

Ablation Study는 3에서 진행된 각 섹션들을 하나씩 제거해보며 성능을 평가하였다. 위에서부터 각각 Historical Price Encoding, Temporal Graph Attention, Heterogeneous Graph attention을 제거하였다. 모든 평가지표에서 어떠한 과정도 제거하지 않았을 때 가장 좋은 성능이 나옴을 알 수 있다. 

## __Interpretability of Graph Neural Network__

<img src = 'https://github.com/hynacin121/ML_Paper_Review/blob/main/img/4GraphAttention.png?raw=true' >

모델의 해석가능성을 살펴보기 위해 모델 예측과정에서 그래프의 관심가중치를 추출하였다. Relational graph의 메시지 전달과정에서 모든 노드에 대한 attention weight를 계산하여 각각의 일일 수익률과 node degrees에 따라 attention weight의 평균을 구하고 시각화 하였다. Pos relationship의 경우, 노드의 degree가 높을 경우 attention weight가 높음을 보여준다. 즉 이웃이 많은 노드가 주변에 더 많은 메시지를 기여한다는 것을 보여준다. 또한 일일 수익률 변동이 큰 기업의 attention weight가 높음을 보였다. 이는 가격 변동성이 클수록 주변에 더 많은 정보를 제공한다는 것을 의미하며, 가격변동이 momentum spillover effect를 일으킨다는 것을 의미한다. 반면 Neg relationship는, 반대로 degree가 낮은 노드에서 높은 attention weight를 보인다. 

<img src = 'https://github.com/hynacin121/ML_Paper_Review/blob/main/img/4ACC.png?raw=true' >

이는 앞선 결과에 더 나아가 각 relation의 attention weights를 시각화하여 하나의 relation만 사용했을 때 성능을 보여준다. 기존과 같이 2개의 데이터셋을 사용해 학습시키고 하나의 relationship의 메시지를 input으로 활용하여 위와 같은 결과를 보였다. self와 pos가 neg보다 예측 성능이 우수함을 보였다. 또한 pos 메시지의 리소스가 예측 모델에 대한 기여도가 높음을 볼 수 있다. 그 이유는 비슷한 가격 움직임을 보이는 기업간의 영향력이 향후 가격 움직임을 예측하는데 상대적으로 유용하기 때문이다. Temporal graph attention layer가 각 노드와 가중치 간의 차이를 적절히 보여주고, Heterogeneous graph attention은 각 메시지의 기여도를 적절히 조정할 수 있음을 알 수 있다.

# __5.	Conclusion__

금융 시계열 예측을 위해 Temporal and Heterogeneous graph neural network model을 제시한다. 가장 효과적인 그래프 기반과 비그래프 기반 baseline과 비교하여 제시한 방법의 효과를 종합적으로 평가하였다. 또한 실제 투자 전략에서 THGNN이 우수한 성과를 보였다. 기업간의 관계를 Heteoreneous dynamic graph로 모델링 하고 GNN을 통해 금융 시계열 예측 모델을 개선하였다. 추후 연구에서는 예측 모델이 보다 정확한 training input graph 데이터를 얻을 수 있도로 corporate relation modeling을 개선할 예정이다.

