---
title:  "[ACL 2021] EASE:Extractive-Abstractive Summarization End-to-End using the Information Bottleneck Principle"
permalink: EASE_Extractive_Abstractive_Summarization_End_to_End_using_the_Information_Bottleneck_Principle.html
tags: [reviews]
use_math: true
usemathjax: true
---

## 0. Background
요약은 상대적으로 긴 문장, 혹은 글에 대해서 중요한 문장으로 간추려 짧은 글로 만들어내는 작업을 의미합니다. 이러한 `텍스트 요약(Text Summarization)`은 `추출요약방식(Extractive summarization)`과 `생성요약방식(Abstractive summarization)`으로 나눠집니다.`추출 요약`은 주어진 글에서 중요 단어 및 문장을 그대로 발췌해서 요약을 하는 것을 말합니다. 요약하려는 글에서 그대로 가져오기 때문에 알고리즘이 어떤 단어와 문장을 중요하다고 판단해서 요약을 했는지 쉽게 알 수 있습니다. 그렇지만 추출요약은 단어 및 문장이 선택되고 재배치되는 것으로, 요약의 결과물은 인간의 요약에 비해 자연스러움, 응집성이 부족합니다. 반면, `생성요약`의 경우 알고리즘이 기존의 단어 및 문장에 대해서 새롭게 바꾸거나 생성하여 요약을 만들어 냅니다. 이는 상대적으로 추출요약에 비해서 자연스러움, 응집성에서 이점을 가져갈 수 있지만, 요약하려는 글의 어떤 단어와 문장에서 어떻게 요약이 된건지 알기가 어렵습니다. 

## 1. Problem Definition
Pretrained된 언어 모델 (BART, T5, 등)은 요약, 기계번역과 같은 여러 분야에서 좋은 성능을 보여주고 있습니다. Pretraining 과정이 없는 모델들에 비해 요약 분야에서 높은 충실도(fidelity)를 가지지만 생성 요약에 대한 `해석가능성(interpretability)`의 부족은 Pretrained된 언어 모델이 널리 사용됨에 있어 장애물로 남아있습니다. 이러한 Pretrained된 언어 모델은 `생성요약방식(Abstractive summarization)`을 통해 요약문을 생성하기 때문에 `해석가능성`이 부족하게 됩니다.  
Pretrained된 언어 모델이 그렇지 않은 모델에 비해서 더 자연스럽고 좋은 성능을 가지는 요약문을 도출한다고 해도, 왜 그런 요약을 하게 되었는지, 어떤 근거(문장)을 통해서 알기가 어렵다면, 사용에 제약이 생길 수 있습니다. 사용자가 납득할만한 근거를 제시하지 못한다면, 실제 사용에 있어 사용자는 모델이 도출한 정확도를 의심해야하고, 근거를 다시 파악해야합니다. 

결론적으로 글을 다시 읽으면서 사용자 본인이 문서 요약을 다시 해야만 비로서 신뢰도 있는 요약을 얻을 수 있게 되는 것입니다. 모델이 요약한 근거도 제시하고, 사람처럼 자연스러운 요약문을 제공해준다면 더할나위없이 훌륭한 모델일 것입니다. 그러한 모델을 만들고자 시도한 논문이 바로 `2021년 ACL`에 소개된 `EASE:Extractive-Abstractive Summarization End-to-End using the Information Bottleneck Principle` 입니다. 

논문은 `Information Bottleneck (IB) principle`을 통해 추출 근거의 길이와 최종 요약문 생성을 위한 정보사이의 `trade-off`를 조절할 수 있습니다. Pretrained된 언어모델을 사용하여 요약할 글(source document)로 부터 필요한 근거를 추출하고`(extractor)`, 그 근거를 기반으로 최종 요약문을 생성합니다`(abstractor)`. 추출요약과 생성요약 모두를 `Information Bottleneck (IB) principle`을 통해 end-to-end로 학습할 수 있는 모델을 제시하고자 합니다.    

## 2. Motivation
현재까지의 추출-생성 시스템은 3가지 부류로 나눠집니다.
>**첫번째는 attention에 의존하여 `해석가능성(interpretability)`을 설명하는 방식입니다.**
Attention mechanism의 확률적 특성 때문에 실제 사용가능한 근거를 주는데 부족함이 있습니다.
>**두번째는 생성된 요약에 대해서 단어수준의 근거만을 제공하는 경우입니다.**
첫번째의 attention에 의존하여 해석가능성을 제공하는 방식보다는 유용하지만, 단어 자체가 너무 세분화되어있기 때문에 사람에게는 유용성이 떨어지는 근거가 됩니다. 
>**세번째는 pseudo labels을 사용하거나 다른 heuristic 방법으로 content selector를 따로 학습시키는 방식입니다.**
이러한 3가지 부류의 출출-생성 시스템은 자연스러운 요약의 방식이 아닐 것 입니다. 가장 자연스러운 요약 방식은 사람의 요약 방식을 따라하는 것일 것입니다. 사람은 긴 문서를 요약할 때 `2-stage의 요약방식(추출-생성)`을 사용한다고 합니다. 먼저 중요한 부분을 추출하고나서 어떤 것을 뺄지, 단어를 바꿀지, 문장을 재구성할지를 결정한다고 합니다. 이러한 인간의 요약방식에 영감을 받아 본 논문은 `Information Bottleneck (IB) principle`에 기반하여 end-to-end 방식으로 근거를 추출하고 추출한 근거를 바탕으로 생성요약을 만들어내는 추출-생성 요약 시스템(`EASE`)을 제시합니다.

## 3. Method
`EASE `는 `Information Bottleneck (IB) principle`을 사용하여 문서요약을 합니다. 앞으로는 `IB`라고 줄여 말하도록 하겠습니다. 모델이 요약해야할 원래 문서를 $x$, 요약이 완료된 문서를 $y$, 그리고 `IB`를 통해 압축된 문서의 representation을 $z$라 하겠습니다. 그러면 `IB`를 사용하는 $loss$는 다음과 같습니다.

>**$L_ {IB}= I(x;z)-\beta I(z;y)$ -(1)입니다.**

$I()$는 mutual information을 의미합니다. 
(1) $loss$를 minimize하는 것이 학습의 목표입니다. (1)을 통해 $z$가 $x$로부터 $y$를 예측하는데 필요한 정보만을 가질 수 있도록 만들어주는 식입니다.
또한 $\beta$ 는 $z$가 $x$에 관한 정보를 포함할지(`sparsity`) vs $y$(`prediction quality`)에 관한 trade-off를 조절합니다. 
(1)식을 학습에 사용하기 위하여 (1)식에 대한 relaxation을 진행합니다. (이 과정은 [Paranjape et al. 2020](https://arxiv.org/abs/2005.00652)의 과정을 따라하여 (1)식을 다룰 수 있게 하였습니다.)
$z$는 $y$(정답 요약)을 얻기위해 $x$(원래 문서)를 masking하는 과정에서 얻게 됩니다. extraction과정에서 token-level 혹은 sentence-level로 추출을 할 수 있습니다. 먼저 token-level에 대해 진행 후 sentence-level로 일반화를 합니다. `EASE`의 모델 구조는 아래와 같습니다. 
![fig2](https://user-images.githubusercontent.com/62690984/232184131-5fcb93eb-f07f-4d3f-8859-6df895a9b7e6.png)
extractor가 원래문서($x$)의 token을 rough summary($z$)를 얻기 위해 masking하고, abstractor가 rough summary($z$)를  정답요약($y$)를 생성하기 위한 근거로 사용하게 만드는 구조입니다. $z = x\odot m$ 이고 $m$은 원래문서($x$)에 대한 boolean mask(0 또는 1)입니다. `EASE`는 end-to-end로 원래문서에서 어떤 token을 masking해야할지 학습하게 됩니다. 모델은 총 2가지의 loss를 가지게 됩니다.   

>**첫번째는 variational bound(Alemi et al., 2016)을 사용하여  $L_ {Task} = E_ {m\simeq p(m\vert x)}\left [ -log_ {q_ {\theta }}(y\vert m\odot x) \right ]$-(2)을 minimize하는 것입니다.**

(2)에서 $q_ {\theta}(y\vert z)$ 는 true likelihood값인 $p(y\vert z)$에 대한 parametric approximation입니다. Paranjape et al.(2020)처럼 mask variables들은 이전 추출에 영향을 받지 않습니다.(조건부 독립) 이 말은 중복해서 근거를 찾을 수 있다는 말입니다. (abstractor 단계에서 중복성을 제거 가능합니다.) 

>**두번째 loss는 $L_ {Sparsity} = \sum_ {j}KL\left [p_ {\theta}(z_ {j}\vert x), r(z_ {j}) \right ]$-(3)입니다.**

(2)식만을 사용한다면 input과 output 사이의 mutual information을 maximize 하면서 어떤 token도 masking되지 않을 것입니다. 그러므로 (3)식을 통해서 sparisty를 조절합니다. (3)에서 prior distribution 으로 $r(z_ {j}) =  Bernoulli(\pi)$ 입니다.$\pi$ 는 0.3에서 0.5 사이의 값입니다. 
결국 (2)과 (3)을 합친 최종 $Loss$는 다음과 같습니다.

>**최종Loss: $L_ {EA} = E_ {m\simeq p(z\vert x)}\left [ -log_ {q_ {\theta }}(y\vert m\odot x) \right ]+\beta \sum_ {j}KL\left [p_ {\theta}(z_ {j}\vert x), Bernoulli(\pi) \right ]$ -(4)**

$p_ {\theta}(z\vert x)$ 는 parametric posterior distribution을 의미하고 $\beta$ 는 performance-sparsity trade off에 대한 hyperparameter입니다. 
>**그렇지만 최종 $Loss$ 는 미분가능하지 않습니다.**

그래서 soft masking을 적용합니다. Gumbel Softmax reparametrization trick을 사용하여 (Paranjape et al. 2020 유사하게) $Bernoulli$로 부터의 sampling과정을 $argmax_ {i\in 0, 1}(logp(z_ {j}\vert x)+g_ {i})$ 로 대체하여 사용합니다. 
$g_ {i}$ 는 $Gumbel(0,1)$ 로부터의 random sample을 의미하여 최종 식은 weight soft max형태이고 

> $z_ {j}^{*} = \frac{\exp((\log p(z_ {j} = 1  \vert x)+g_ {1})/ \tau )}{\sum_ {i\in  0, 1}\exp((\log p(z_ {j} = i \vert x)+g_ {i})/\tau)}$ 

(Note $z_ {j}^{*}\in (0,1)$ when $\tau$ goes to $0$, in practice $\tau = 0$) 입니다.

fig2와 같이 `EASE`는 extractor와 abstractor로 나뉩니다. extractor는 pretrained 된 BERT에 $p_ {\theta}(z_ {j}\vert x)$ 계산을 위해 linear를 추가한것과 비슷하고, abstractor는 BART와 비슷합니다.학습과정에서 extractor는 원래문서($x$)에서 어떤 token이 masking될지에 대한 확률을 계산합니다.
이 확률 ($p_ {j}$) 에 대해 $m_ {j}$ (0,1사이)를 가집니다. $z = m\odot x$ 를 abstractor에 보내고 최종 요약문을 생성합니다. 
실험과정에서 2가지의 masking방식을 사용하였습니다. 
>**(1) 직접 embedding에 대해서 masking을 하는 방법, $z_ {j}^{*} = \frac{exp((logp(z_ {j} = 1\vert x)+g_ {1})/\tau )}{\sum_ {i\in 0, 1}exp((logp(z_ {j} = i\vert x)+g_ {i})/\tau)}$**

>**(2) $m$을 extractor와 abstractor encoder의 self attention과정에서의 attention mask로서 사용하였습니다.**
그렇지만 2가지 masking 방식에 대한 뚜렷한 실험적 차이는 없었다고 합니다. Sentence-level masking에서는 [CLS] token을 문장의 시작에 추가하여, [CLS] token을 문장의 전체 encoding token으로 사용하여 token-level의 과정의 방식으로 진행합니다.


## 4. Experiment
### Experiment setup
#### Dataset
CNN/DailyMail 데이터셋과 XSUM 데이터셋을 사용합니다. CNN/DailyMail 데이터셋은 요약을 위한 신문기사 데이터셋의 특성상 추출적인 성격(extractive-like nature)이 강한 데이터셋입니다. XSUM 데이터셋은 매우 높은 축약적 성격(highly abstractive)을 가진 데이터셋 입니다.

#### Model  Hyperparameters and evaluation metrics  
sequence-to-sequence 추출기(abstractor)는 BART-large 모델을 사용하고, extractor는 BART-base encoder를 사용합니다. fairseq codebase를 사용하였고 같은 hyperparameter를 사용하여 BART를 finetuning합니다. Polynomial decay learning rate scheduler를 사용하였고, Adam optimizer를 사용하여 BART를 finetuning 합니다.(lr:3e-5, 500 warm up steps, train 20000steps). Performance와 Sparsity의 trade off hyperparameter인 $\beta$ 는 5를 사용합니다. 요약문에 대한 Automatic Evalautation 평가 지표로는 요약분야의 대표적인 평가지표인 `ROUGE F1 (R-1/R-2/R-L)`을 사용합니다. 그리고 추출한 근거와 만들어진 생성요약문에 대해서 Human Evalaution 또한 진행합니다.
`ROUGE score`에 대해서 생소하신 분들이 많으실거라 생각합니다. 간단히 `ROUGE score`가 무엇인지 살펴보겠습니다. `ROUGE Recall`, `ROUGE Precision`,`ROUGE F1`이 있습니다. 쉽게 분류문제에서 Recall, Precision, 이 둘의 조화평균인 F1 score를 떠올리시면 이해하시는데 도움이 될 것입니다. 그리고 `R-1/R-2/R-L`에서 1, 2, L은 문장의 단어(gram)에서 1개씩, 2개씩, 그리고 가장 긴 공통 sequence를 본다는 말입니다.
예를 들어보겠습니다. 모델이 요약을 완료하여 도출한 문장이 `"the hello a cat dog fox jumps`라고 하고, 모델이 요약을한 원래 문장의 정답 요약(golden summary, reference summary)를 `"the fox jumps"`라고 하겠습니다.모델의 요약문은 unigram으로 생각하면 7개의 gram이 모여있고, 정답요약은 3개의 gram으로 이루어져 있습니다. 모델의 요약과 정답요약에서 가장 긴 공통 sequence는 `"for jumps"`입니다. `ROUGE Recall`,`ROUGE Precision` 그리고 `ROUGE F1`의 식은 다음과 같습니다. 
>**$Recall: \frac{count _{match}(gram _{n})}{count(gram _{n})}$**
>**$Precision: \frac{count _{match}(gram _{n})}{count(gram _{n})model's output}$**
>**$F1-Score: 2*\frac{precision*recall}{precision + recall}$**
위의 예시에서 Recall을 구해보면, 3/3=1, Precision = 3/7= 0.43이 나옵니다.  

`ROUGE Score`가 Automatic Evaluation 에서 가장 많이 쓰이는 평가 metric임은 분명하지만, 치명적인 단점이 있습니다.
요약 문장과 정답 요약 문장간의 `gram`단위의 일치도만을 보기 때문에, 생성요약시의 다양한 동의어, 축약어 등의 요약에 대해서는 온전한 평가를 할 수 없기 때문입니다.
`ROUGE 2.0`은 동의어 문제를 동의어 vocabulary set을 통해서 해결하려고 하지만, 이또한 미리 정한 vocabulary set에 국한된다는 단점은 존재합니다.

### Result

table1은 제시한 모델인 `EASE`의 CNN/DailyMail과 XSUM에 대한 ROUGE-1/2/L에 대한 결과표이며, sparsity 0.5일때, BART-base encoder 추출기(extractor)와 BART-large 생성기(abstractor)를 사용했을 때의 결과입니다.
![table1](https://user-images.githubusercontent.com/62690984/232083938-21b37d7e-cc1c-40f8-b5b4-247ca7dd1456.png) 
CNN/DailyMail데이터셋에 대해서 `EASE`가 BERTSUM모델 보다는 약간 나은 성능을 보였지만, BART-large baseline모델보다는 약간 못하는 성능을 보여줍니다. 또한 XSUM 데이터셋에서는 BART-large와의 성능 격차는 더 크게 나타납니다. 그 이유는 XSUM 데이터셋 자체가 매우 abstractive하기 때문에 `EASE`의 extractor가 중요한 정보(근거)를 end-to-end로 찾는 것은 더욱 어려운 일이 될것입니다. 또한 `EASE`의 sentence level과 token level별로 데이터셋에 따른 성능차이를 보여줍니다. 더욱 extractive 성격의 데이터셋에서는 연속적인 텍스트의 확대가 중요하기 때문에 sentence-level의 모델이 잘 작동하는 것으로 보이고, abstractive 성격의 XSUM에는 token-level의 모델이 문서에 흩어진 중요 단어들을 추출하기에 낫기 때문입니다.  


![fig4](https://user-images.githubusercontent.com/62690984/232072349-55bc2d82-2c6f-4f78-8989-b5ac66e645c3.png)

fig4는 모델이 만들어낸 요약문의 예시들입니다. 노란색으로 하이라이트된 것이 모델이 추출한 근거이고 다른 sparsity에 따른 요약결과 입니다. 

![fig3](https://user-images.githubusercontent.com/62690984/232076366-91a8185b-c67c-41a9-b0c9-d5ab32f61333.png) 

fig3은 token-level의 모델과 sentence-level의 ROUGE score를 보여주고 있습니다. sparisty의 비율을 증가시키면 ROUGE score의 향상을 가져는 결과를 볼수 있습니다. token-level의 모델은 낮은 sparsity에도 정보의 손실없이 functional 단어를 제거할 수 있어 더욱 robust함을 확인할 수 있습니다. 

![table2](https://user-images.githubusercontent.com/62690984/232078155-e22bd847-9e96-4fd1-8f4c-ba07c6c45190.png)

table2는 모델 사이즈의 효과를 보여는 실험 결과입니다. 모든 모델 모두  sparsity 0.5로 학습하였을 때의 결과입니다. 표를 보면, abstractor로 large모델을 선택했을 때 더 좋은 것을 알 수 있고, extractor와 abstractor의 encoder를 공유하는 것이 성능을 악화 시키지 않는다는 것을 확인했습니다. 또한 large abstractor의 사용은 중요하지만 large extractor와 같이 사용했을 때 학습이 불안정한 점이 있어서, default setting으로 BART-base extractor와 BART-large abstractor를 사용하였습니다.


![table3](https://user-images.githubusercontent.com/62690984/232265251-c0bac159-d8a9-4745-b1b3-6a6db5cd591a.png)

table3를 통해 최종 요약문에 대한 extraction의 영향을 평가한 결과를 볼 수 있습니다. Sentence-level task에서 어떻게 중요 문장(근거)를 extract해서 abstractor에게 전달할 것인가는 중요한 문제입니다. sparsity 0.3 과 0.5로 모델을 학습하였고, inference시에는 높은 점수를 가진 top-3문장들만 abstractor에게 전하였습니다. random-3, lead-3와 비교하였는데, 여기서 random-3는 랜덤하게 3문장을 뽑은것이고, lead-3는 문두 3문장입니다. 실험결과 Top-3가 random-3 와 lead-3보다 각각의 sparsity에서 더 좋은 성능을 보여주었습니다.
CNN/DailyMail 데이터셋은 데이터셋 특성상 앞단의 문장에 중요한 문장들이 많은 bias를 강하게 가지고 있음에도, `EASE`의 모델의 Top-3방식은 lead-3보다 좋은 성능을 보여주었고, 이것을 통해서 `EASE`의 extractor가 정말로 중요한 문장들을 잘 추출하고 있다고 판단할 수 있습니다. 

![table4](https://user-images.githubusercontent.com/62690984/232265258-85089817-30c6-45a2-b54f-f62edb798b85.png)

extracted된 근거와 생성된 요약에 대해서 인간평가를 수행한 결과입니다. Summarization task에서는 위의  `ROUGE Score`과 같이 automatic evalaution도 이루어지지만, Human Evalauation(인간평가)도 같이 수행하는 편입니다. 문장 요약이라는 task상 사람이 받아들이기에 자연스러운지, 분명한지 등의 평가 척도가 중요한데, 자동평가로는 온전한 평가를 내리기 어렵기 때문입니다.
`Consistency`는 summary(요약 결과) 와 source document(요약할 문서)간의 사실적 일치성을 보는 척도입니다. 이것을 통해서 모델의 요약이 실제 요약할 문서의 detail을 바꿨거나, 가짜 정보를 만들어내지는 않았는지 평가합니다. 
`Relevance`는 summary(요약 결과)가 source document(요약할 문서)에서 중요한 문장, 핵심을 잘 잡아냈는지 평가합니다. 
`Extraction Relevance`를 통해서 모델이 뽑아낸 중요 문장들과 `Lead-3 Extraction`과 비교를 하였습니다. 평가 결과 sentence level의 경우 `BART`와 비슷한 `Consistency`를 보여주었지만, `Relevance`는 부족한 결과를 보여주었습니다. sentence-level의 특성상 source document의 중요 정보들을 놓친 것으로 판단이 됩니다. 그렇지만 sentence-level보다 유연한 token-level의 경우 `BART`보다 아주 약간 더 높은 `Relevance`를 보여주고 있습니다. 그리고 sentence level에서의 뽑은 근거와 `Lead-3`를 비교했을 때도, 앞선 자동평가에서와 더불어 더 좋은 결과를 보여주어, extractor가 중요문장을 잘 뽑고 있다는 저자들으 주장을 잘 뒷받침하고 있습니다. 


## 5. Conclusion
본 논문은 `EASE`라는 요약을 위한 extractive-abstractive 구조를 제시하였습니다. 중요한 문장을 추출하는 extractor와 그 문장을 토대로 요약을 생성해내는 abstractor를 end-to-end로 학습할 수 있는 장점이 있습니다. 또한 extractor로 부터 추출된 문장은 abstractor가 도출하는 마지막 요약 문장의 근거라고 여겨질 수 있어, 요약의 interpretability를 제공한다고 할 수 있습니다.  
인간의 2-stage 요약방식의 idea와 기존 모델들의 해석가능성의 부족함(lack of interpretability)을 `Information Bottleneck (IB) principle`을 통해 end-to-end로 해결하려고 한점이 흥미로웠습니다. 또한 기존의 방식은 요약의 근거를 제시하지 못하고, 혹은 따로 학습을 해야한다는 점등에서 `EASE`의 모델은 상당한 강점을 가집니다. 근거를 제시할 수 있는 모델의 요약은 실제 사용될수 있는 범위가 넓고, 그 신뢰도를 사람이 파악 쉽게 가능하기 때문입니다. Supervised 방식의 이러한 요약 방식들은 이제 어느정도의 수준에는 이르렀다고 볼 수 있습니다. 그렇지만 우리 인간은 어떤 문서에 대해서 요약된 문장(정답)의 학습이 필요없이, 자연스럽게 선택, 조합 그리고 축약 등의 과정을 거쳐 요약문을 만들어냅니다. 이러한 관점에서 `Unsupervised`방식의 summarization task도 또한 흥미로운 문제이고, 반드시 풀어나가야할 문제인것 같습니다.


## Author Information
-   Heewoong Noh
    -   Affiliation:  [DSAIL@KAIST](http://dsail.kaist.ac.kr/)
    -   Research Topic: Deep Learning
    -   Contact: heewoongnoh@kaist.ac.kr

## Reference & Additional materials
-   Github Implementation
    -   저자의 공식적인 Github Implementation은 없습니다.
    -   논문에서 사용한 [fairseq codebase](https://github.com/pytorch/fairseq)
    -   논문에서 사용한 [RougeMetricToolKit](https://github.com/pltrdy/files2rouge)

-   Reference
    -   [[ACL-21] EASE: Extractive-Abstractive Summarization End-to-End using the Information Bottleneck Principle](https://arxiv.org/abs/2105.06982)
    -   [[EMNLP-20] An Information Bottleneck Approach for Controlling Conciseness in Rationale Extraction](https://arxiv.org/abs/2005.00652)
    -   [ROUGE score explanation](https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460)
