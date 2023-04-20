---
title:  "[AAAI 2021] TabNet: Attentive Interpretable Tabular Learning"
permalink: TabNet_Attentive_Interpretable_Tabular_Learning.html
tags: [reviews]
use_math: true
usemathjax: true
---


# TabNet: Attentive Interpretable Tabular Learning

 본 논문의 저자는 **트리 기반 앙상블 모델**들이 딥러닝에 비해 **정형 데이터**에서 학습에 보다 논리적이고 합리적인 접근 방법이라고 소개합니다.   일반적으로 관측되는 정형 데이터는 대략적인 초평면(hyperplane) 경계를 지니고 있는 매니폴드(manifolds)를 가지고 있으며 이 공간에서는 트리 기반 앙상블 모델의 결정 방식이 이해(representation)하는데 더 강점을 지니고 있기 때문입니다.     
 본 논문에서 제안한 TabNet은 **decision tree-based gradient boosting**의 장점을 살린 인공신경망 아키텍쳐이며, **feature engineering**과 **selection**까지 함께 활용할 수 있는 장점이 있습니다. 또한 **해석가능한(interpretability)**, 설명가능한 XAI(eXplainable Artificial Intelligence)라는 점에서 큰 장점이 있습니다.

>  * 두괄식으로 **TabNet의 컨셉**을 간략하게 설명하자면 다음과 같이 정의됩니다.  
    "입력된 정형데이터(tabular data)에서 Feature를 masking하며 여러 step을 거쳐서 학습"   
>    *  각 step별 **feature들의 importance** 파악 (설명력 확보)
>    * **masking 으로 중요한 feature 들만 선출**해서 학습하여 성능 향상 (모델 고도화) 
    
저희 분야에서는 정형데이터인 tabular data를 활용한 연구가 많이 진행되는 편이어서, 본 논문을 흥미롭게 읽었는데요. 정형데이터를 많이 활용하시는 분이라면 도움이 될 수 있을 것 같습니다.


___


## 1. Problem Definition
과거 정형 데이터를 활용한 모델들의 성능비교를 했을 때, ***LightGBM***의 성능이 가장 좋다고 알려져있고, 혹은 앙상블을 고려한 ***Extreme Gradient Boosting*** (***XGBoost***)나 ***Catboost***을 떠올릴 수 있습니다. 더하여 ***Neural Network***를 추가하는 방법이 있는데, 정형 데이터의 경우에 *Neural Network*는 복잡하거나 깊은 레이어로 구성되지 않는 편이며, 더 깊거나 복잡하다고 성능이 눈에 띄게 좋아지지 않았습니다. ***PCA***를 활용하여 **차원축소**를 하거나, ***autoencoder***를 기반으로 **노이즈를 제거**하는 등의 방식에 활용이 되는 것이 주 였다고 볼 수 있습니다.   
또한, ***CNN, MLP***와 같은 딥러닝 모델은 적절한 귀납적 편향(*inductive bias*)의 부족으로 지나치게 ***Overparametrized*** 되어 정형 데이터 내 매니폴드에서 일반화된 해결책을 찾는데 어려움을 발생시길 수 있습니다. 그럼에도 불구하고 이런 딥러닝 학습 방법론을 정형 데이터 학습에 사용하고자 하는 이유는 이미지나 다른 종류에 데이터와 정형데이터를 함께 학습(*Multi-Modal*)할 수 있으며 트리 기반 모델 성능의 핵심인 Feature Engineering과 같은 작업이 크게 필요하지 않습니다. 또한, 스트리밍 데이터 학습이 용이하고 종단간(*end-to-end*) 모델은 Domain adaptation, Generative modeling, Semi-supervised learning과 같은 가치있는 응용 모델과 같은 표현 학습(*representation learning*)이 가능합니다.    
 





## 2. Motivation
본 논문에서는 새로운 고성능 및 해석 가능한 표준 심층 테이블 형식 데이터 학습 아키텍처 인 TabNet을 제안합니다. TabNet은 원자료의 다른 전처리 없이 입력할 수 있고 경사하강법 최적화 방법을 통해 유연한 통합(flexible integration)이 가능한 종단간(end-to-end) 학습이 가능합니다. 또한, 순차적인 어텐션(Sequential Attention)을 사용하여 각 의사 결정 단계에서 추론할 feature들을 선택합니다. 이로인해 더 나은 해석 능력과 학습이 가능하며 숨겨진 특징을 예측하기 위해 사전 비지도 학습을 사용하여 정형 데이터에 중요한 성능 향상을 보여줍니다(*Self-supervised learning*). 

>  본 논문에서 제안한 TabNet의 차별화된 아이디어와 contribution은 다음과 같습니다.
>  1) 전처리 과정을 거치치 않은 **raw data로도 end-to-end 학습이 가능**
>  2) **sequantial attention** 과정에서 **각 step마다 중요한 feature를 선별**하면서 각 과정의 모델 해석과 성능 향상이 가능 
>  3) **다양한 도메인의 데이터**에서 다른 테이블 학습 모델과 비교 했을 때 **분류/회귀 문제에서 우수한 성능**을 보임 
>  4) **masking 된 feature를 예측하는 tabnet decoder 비지도 학습을 통한 우수한 성능**을 보임    

   


## 3. Method
본격적으로 TabNet architecture를 살펴보기 전에, TabNet의 feature selection 방법과, encoder, decoder에 대한 설명을 간략히 정리했습니다.   

![image](https://user-images.githubusercontent.com/82039869/232205929-78a7c5b6-f792-4728-8967-47064f1edadb.png)

기존의 Feature seletion은 Lasso Regularization, Instance-wise feature selection등의 방법을 사용합니다. 이러한 방식을 본 논문에서는 Hard feature selection으로 표현하고 있으며, TabNet은 Soft feature selection을 구현하여 사용합니다.
간단히 언급하면, **TabNet은 masking**을 이용하여, 좌측에서 우측으로 **sequential하게 feature selection**을 하며 피드백을 주고 학습해나가는 구조입니다 (*figure 1 그림 참조*).   

![image](https://user-images.githubusercontent.com/82039869/232206115-5e02beaf-6cc7-4891-b1a0-3335b0330035.png)

더하여 tabnet encoder를 통해서 **feature engineering**효과를 내고, decision making 부분을 통해 **feature selection**이 이루어집니다. encoder는 *fine-tuning*하면서 task에 맞게 성능을 향상시켜 맞춰갑니다.   
encoder에 decoder 구조 결합하면 autoencoder 같은 자기 학습 구조를 가지고 있고, decoder 에서는 ?(물음표) 로 된 *missing value* 를 채워넣는 구조입니다. 



   
> **TabNet encoder와 decoder의 architecture를 좀 더 자세히 살펴보면 아래 그림과 같습니다.**   

![image](https://user-images.githubusercontent.com/82039869/232206212-771ed0b7-abb8-4ec8-a42a-37c9265ea17e.png)   

* **(a) : TabNet encoder** 의 경우, 각 decision step에 대해서 ***1)feature transformer, 2)attentive transformer, 3)feature masking***으로 구성되어 있습니다.    첫 의사결정 단계에서 부족한 부분을 다음 의사결정 단계에서 보완하는 방식이며, 트리기반 부스팅 모델들과 유사합니다.
모델의 용어를 좀 더 구체적을 설명하면, BN : 최초 Feature에 대해 배치 정규화를 적용
    *  ***Feature transformer*** : 이후에 아래의 단계 반복.
    *  ***Attentive transformer*** : (d)에서 설명
    *  ***Mask*** : Attentive transformer에서 나온 ***Mask***(***M[i]***)에 대해 feature ***f***를 곱하여 이후 step의 *Feature transformer*에 들어가는 input을 조절. Mask를 통해 변수를 soft selection 함.
    *  ***Feature transformer*** : (c)에서 설명
    *  ***Split*** : Feature transformer의 output을 두 개로 복제하여 하나는 *relu*로, 하나는 *Attentive transformer*로 보냄


* **(b) : TabNet decoder**는 각 단계에서 feature transformer 블록으로 구성됩니다. 일반 학습에서는 Decoder를 사용하지 않고, self-supervised 학습을 진행할 때, 인코더 다음에 붙여져서 기존의 결측값을 보완하고 표현학습을 진행하게 됩니다.

* **(c) Feature Transformer**는 4개의 네트워크 묶음으로 구성됩니다.
    *  ***Fully Connected Layer*** (**FC**) - ***Batch Normalization*** (**BN**) - ***Gated Linear Unit Activation*** (**GLU**)로 구성된 블럭들을 순차적으로 통과하는 구조를 쌓고, 블럭간의 *residual skip connecion*을 적용했습니다. 그리고 *residual output*의 *normalization*을 위해 ***sqrt***(***0.5***)를 곱했습니다. 이때, 두 레이어는 모든 *decision* 단계에서 공유되며, 나머지 두 레이어는 *decision* 단계에 의존합니다. 따라서 앞 2개의 네트워크는 모든 파라미터를 공유하며 **글로벌 성향**을 학습하고, 뒤에 2개의 네트워크 그룹은 **로컬 성향**을 학습합니다. 
  이어서 **D**개의 변수를 갖는 값들을 입력받은 *feature transformer*는 ***split***할 값들을 내보냅니다. 총(**B,N**)의 *output*을 내보냈을 때, 논문에서는 *split* 과정을 통해 *d*[*i*]와 *a*[*i*]로 나누었습니다. 

* **(d) Attnetive transformer** 블록으로, 현재 의사결정 단계에서 각 변수들이 얼마나 많은 영향을 미쳤는지 사전 정보량(*prior scale information*)을 통해 집계합니다. 다시말해, 현재 decision step 전에 각 feature가 얼마나 많이 사용되었는지를 집계한 정보를 나타냅니다. 이것은 단일 레이어에 맵핑하여 사용되며, 계수의 정규화는 각 decision step에서 가장 두드러진 특징을 ***sparse***하게 선택하고, 계수값들을 일반화(*normalization*)하기 위해 ***sparsemax***를 사용하여 학습되어집니다.   





## 4. Experiment
본 논문에서는 regression, classification task로 성능을 평가하였고, 데이터 셋의 모든 categorical value 들은 임베딩되었고, numerical value들은 전처리 없이 input으로 활용되었습니다. TabNet은 대부분의 *hyperparameter*에 대해 그리 예민하지 않다는 특징을 가집니다.    

>**Instance-wise feature selection** (synthetic dataset - 임의로 생성한 데이터셋 활용)  

![image](https://user-images.githubusercontent.com/82039869/232207111-24fad62a-aa60-437c-9574-93f21a0c990c.png)

Table 1의 결과를 간략히 정리하면 다음과 같다.
* **6개 임의로 생성된 데이터로 성능을 평가** --> syn1 ~ syn 6 데이터
* **syn 1~3 에서는 각 인스턴스(data row) 별로 중요한 피쳐가 같음** --> 따라서 syn 1~3 에서는 Tabnet 의 성능이 global feature selection 하는 다른 모델들과 성능이 비슷 
* **syn 4~6 에서는 각 인스턴스(data row) 별로 중요한 피쳐가 다름** --> 따라서 불필요한 feature들을 instance wise로 제거해서 성능을 향상


>**Performance on real-world datasets** (실제 데이터셋 활용)   

![image](https://user-images.githubusercontent.com/82039869/232207175-64f6f48a-1be9-4485-a3d6-afac6b09bc8c.png)  

***Forest cover type dataset***  : 나무 분류 문제 / ***Poker Hand*** : 카드 분류 문제 / ***Sarcos*** : 로봇 팔 관련 데이터 / ***Higgs Boson*** : 이진 분류 문제 /***Rossman Store Sales*** : 상점 매출 예측

다양한 데이터셋을 활용하여 *XGBoost, LightGBM, Random forest, MLP* 등 다양한 모델과의 비교 결과, **Test accuracy, MSE 등의 평가지표**에서 TabNet이 다른 모델들에 비해 **좋은 성능**을 보이는 것을 확인할 수 있습니다.   


또한 본 논문에서 제안한 그림(*Figure 5*)을 통해 알 수 있듯이, 각 스텝에서 활성화된 **feature 를 시각화**로 확인할 수 있다는 장점이 있고, **instance 별 중요도** 또한 확인이 가능합니다.    

![image](https://user-images.githubusercontent.com/82039869/232208001-15a85c65-4121-43ad-a254-b2ad92bf81e8.png)


그림의 하얀색 부분이 모델 학습에 사용 feature라고 해석할 수 있습니다.   
간단히 설명해보자면, *Syn2 dataset*에서는 ***feature X3-X6*** 만 활용 되었으며 ***Magg*** 는 각 스텝의 **feature importance를 결합** 했을 때 나오는 결과를 **시각화** 한 것입니다. 




## 5. Conclusion

>저자는 *tabular learning*을 위한 참신한 딥러닝 아키텍처인 ***TabNet***을 제시했으며, TabNet은 각 결정 단계에서 처리할 의미 있는 변수의 subset을 선택하기 위해 ***sequential attention mechanism***을 활용했습니다. 선택된 특징들은 *representation*으로 처리되어서 다음 결정 단계에서 정보를 보내고 기여하고 있습니다. ***Instance-wise feature selection***은 *model capacity*로써 효율적인 학습을 가능하게 한다고 정리할 수 있을 것입니다.


저의 개인적인 소견으로는, 논문에서 언급된 것과 같이 **정형 데이터보다는 비정형 데이터에 많이 집중**되어 개발되어 있는 상태라고 생각합니다. 따라서 비정형데이터에서 성능을 높인 모델을 정형데이터에 적용하기에는 이론이 맞지 않다고 생각했는데, 본 모델에서 대안으로 ***attention mechanism***이라고 하는 방법이 제시되어 너무 좋았습니다. 또한 본 모델을 활용한다면 전처리의 필요성이 줄어들고, ***tree-based model***처럼 활용할 수 있는 것이 또 하나의 장점이라고 생각합니다.    
마지막으로 보통은 딥러닝을 해석하고 평가하기 위해 *surrogate model*로 대체하는데, 본 논문에서는 **해석가능성 방식**을 활용하여 구현한 것이 인상적이었습니다. 아주 작은 코멘트로는, 신경망모델 치고는 조절해야할 파라미터가 조금 많아 *parameter search*하는데 시간이 소요 될 것 같습니다.   
하지만 정형 데이터에 대한 모델 연구나 방법론이 한정적이라고 생각했었는데 끝없이 발전하고 있다는 생각이 드는 논문이었습니다. 특히 ***tabular data***를 활용한 연구를 많이 진행하고 있는 입장으로 TabNet을 저희 연구분야에 적용하기 위해 코드 및 구조를 더욱더 꼼꼼히 배워야 겠다는 생각을 했습니다.
모두 한번 읽어보세요~! :)   


## 6. Reference & Additional materials
* TabNet 논문파일과, Github - tensorflow, torch로 구현된 코드 링크를 함께 첨부드립니다.
    * [TabNet 논문](https://arxiv.org/abs/1908.07442)
    * [TabNet tensorflow code](https://github.com/google-research/google-research/tree/master/tabnet)
    * [TabNet torch code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tabnet.py)