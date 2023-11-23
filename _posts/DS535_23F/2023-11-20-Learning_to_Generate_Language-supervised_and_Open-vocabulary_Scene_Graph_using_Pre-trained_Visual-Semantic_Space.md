---
title:  "[CVPR 2023] Learning to Generte Language-supervised and Open-vocabulary Scene Graph using Pre-trained Visual-Semantic Space"
permalink: 2023-11-20-Learning_to_Generate_Language-supervised_and_Open-vocabulary_Scene_Graph_using_Pre-trained_Visual-Semantic_Space.html
tags: [reviews]
use_math: true
usemathjax: true
---
  
  

  

# **Learning to Generate Language-supervised and Open-vocabulary Scene Graph using Pre-trained Visual-Semantic Space**

  

  

  

## **1. Background**

  

### Scene Graph Generation

  

본 논문은 Scene Graph Generation (SGG)에 관한 논문으로 논문의 자세한 내용을 다루기 전에 먼저 scene graph가 무엇인지 알아보자.

  

Scene graph는 하나의 scene이 주어졌을때 그로부터 graph를 얻는 것인데 여기서의 scene은 image 등이 될 수 있다.

  
  

<p  align="center"><img  src="https://i.ibb.co/6FX06mN/scene-graph.jpg"  height="250px"  width="600px"></p>

위 그림은 SGG의 한가지 예시로 image의 object로 "man", "horse", "glasses", "bucket"이 있고 object간의 관계로 "wearing", "feeding" 등이 존재한다. 그러면 오른쪽 graph처럼 object과 그들간의 relation을 graph 형태로 나타낼 수 있는데 그 것인 바로 scene graph이다. 즉 위 그림처럼 scene에서 scene graph를 얻는 전체 과정을 Scene Graph Generation이라고 한다.

  

일반적인 SGG는 pretrained object detector(e.g., Faster R-CNN)로 image에서 object를 detection하여 object의 bounding box와 class label을 추정한 뒤 그 정보를 바탕으로 두 object 사이의 관계를 추정하는 과정으로 진행된다.

  
  

### CLIP

기존의 SGG은 pre-trained object detector로 Faster R-CNN을 주로 이용하였는데 이 pre-trained object detector는 이미 정해진 class에 대해서만 학습을 진행되었기 때문에 pre-training시에 학습하지 못한 class가 있다면 그러한 object에 대해서는 class를 예측하지 못한다. 예를 들어 어떤 object detector가 개와 고양이 class에 대해서만 학습을 했다면 새로운 image에 장수풍뎅이가 등장했을 때 그 장수풍뎅이 object의 class를 추정할 때 장수풍뎅이 class로 예측할 수 없고, 개나 고양이 중에 그나마 근접하게 생긴 class로 예측 할 것이다.

  

위와 같은 한계를 극복하기 위해 pre-training시 image만 이용하는 것이 아니라 text 정보까지 함께 이용한 image-text pair를 이용하려 시도가 있었다. CLIP이 그 대표적인 모델인데 간단하게 설명하면 개에 대한 image와 글자 "개"라는 것에 대해 각각 representation을 얻어 둘의 similarity를 높히고 대신 서로 다른 pair에 대해서는 similarity를 낮추는 식으로 학습하여 image와 language에 대한 정보를 모두 반영한 학습을 하였다. 이 때 중요한 것이 image-text pair dataset은 기존의 ImageNet처럼 정해진 class로 labeling하여 만들어진게 아니라 인터넷에서 웹 크롤링으로 자동으로 pair를 생성했기 때문에 "장수풍뎅이 image" - "글자 장수풍뎅이"와 같은 pair도 포함되어 있다. 그렇기 때문에 인터넷에 존재하는 수많은 class에 대한 정보도 학습할 수 있고, 기존의 fixed category에 대한 예측을 할 수 있었던 한계를 극복할 수 있게 되었다.

  

<p  align="center"><img  src="https://i.ibb.co/hHqgTrD/clip.jpg"  height="300px"  width="400px"></p>

### GLIP

CLIP은 좀 더 넓은 category에 대해 predict할 수 있다는 점에서 장점이 있기만 아직 matching의 단위가 image라 scene graph generation에 적용하기에는 한계가 있다. 이 말이 무슨 말이냐면 CLIP은 image 하나와 그 이미지 전체를 표현하는 text가 있을 때 이 둘을 matching할 수는 있지만, image에 여러 object가 존재하고 그 object 각각에 대해 object class와의 matching은 불가능하다. 예를 들어 설명하면 하나의 image에 강아지, 고양이, 나무가 있다고 하자. CLIP은 그 image를 "강아지와 고양이와 나무가 있는 그림"이라는 text와 matching이 가능하지만 image 내의 강아지, 고양이, 나무를 detection하여 이들을 각각 "강아지", "고양이", "나무"라는 class로의 matching은 불가능하다는 것이다.

  

그래서 이를 가능하게 한 model이 GLIP이다. 간단하게 설명하면 image 내의 object들과 각각의 class를 matching하도록 training시킨 model이다. 이렇게 image내의 object와 text의 phrase를 matching하는 작업을 phrase groudning이라고 한다. phrase grouding을 통해 image내의 object들에 대해 대응하는 단어를 찾을 수 있기 때문에 CLIP과 달리 image내의 object단위로도 text maching이 가능하다.

  

<p  align="center"><img  src="https://i.ibb.co/Qn2JCdh/glip.jpg"  height="300px"  width="600px"></p>

  

## **2. Motivation**

이 논문은 위에 설명한 GLIP을 Scene Graph Generation에 적용한 논문이다.

  

기존의 SGG은 크게 2가지 limitation이 존재한다. 첫 번째로 pre-training object detector로 Faster R-CNN을 이용하였는데 이 object detector는 정해진 category로의 prediction만 가능하다. 하지만 real world에는 수많은 class가 존재하기 때문에 이러한 prediction은 매우 제한적이다. 두 번째로 scene graph generation을 학습하기 위해서는 사람이 일일히 annotate한 label이 필요한데 이 label을 얻는 과정이 매우 expensive 하고 그 label이 ambiguous하여 곧 바로 이를 이용하면 매우 bias된 prediction을 하게 된다. 이게 무슨 말이냐면 "man" - "sitting on" - "chair"라는 관계가 있을 때 한 사람은 "sitting on"으로 relation을 labeling 할 수 있지만 다른 어떤 사람은 단순히 "on"으로 labeling할 수도 있기 때문이다. 즉, 같은 scene을 보더라도 사람에 따라 relation에 대한 labeling을 다르게 할 수 있다.

  

이 논문은 위의 2가지 limitation을 다루기 위해 GLIP을 사용하였고 GLIP의 특성인 open vocabulary와 language 정보까지 다루를 수 있는 것을 활용해 위의 문제를 해결하였다. 그리고 SGG의 연구 범위를 더욱 확장하며 아래의 table을 제시한다.

  

<p  align="center"><img  src="https://i.ibb.co/fndsFGk/table.jpg"  height="150px"  width="500px"></p>

  
  
  

위의 manually annotated라는게 수동으로 만들어진 label이 필요하다는 것이고 automatically parsed from image descriptions라는 것은 image를 표현하는 language로부터 자동으로 scene graph generation을 하여 label을 만들어 준다는 뜻이다. 또한 novel object class의 containing 여부는 학습때 보지 않은 novel class를 맞출 수 있느냐에 대한 부분이다. GLIP을 활용함으로써 이 논문은 기존의 fully supervised & closed-set을 넘어 language-supervised & open-vocabulary 영역까지 다룰 수 있게 되었다.

  

<p  align="center"><img  src="https://i.ibb.co/1X39zMG/overview.jpg"  height="300px"  width="500px"></p>

  

위는 본 논문의 전체적인 과정에 대한 그림이다.  저 연두색 영역은 Image의 language description과 image의 bounding box가 같은 class이면 비슷한 공간에 표현되는 영역이다. 예를 들면 야구하는 장면 image에서 야구선수 bounding box와 그 야구하는 장면에 대한 표현인 "A man holds a basketball bat on the ground"에서 "man"이라는 단어는 저 space 상에서 비슷한 곳에 위치하게 된다. 이는 background에서 설명했다시피 pretrained GLIP을 이용하면 자동으로 수행되는 부분으로 GLIP은 bounding box와 phrase간의 연결(phrase groudning)이 가능하다. 

이러한 phrase grounding을 이용하면 (a)의 오른쪽에 나타나다시피 image와 description 쌍이 있을 때 둘을 연결하는 scene graph 를 생성할 수 있다. 이게 무슨말이냐면 phrase grounding이 없었다면 image에서 bounding box들을 추출할 수 있고, language description에서 적절한 단어들을 parsing하여 얻을 수는 있지만 이 둘을 matching 시킬 수는 없다. 하지만 GLIP을 통해 phrase grouding이 가능하기 때문에 각 bounding box와 단어간의 matching이 가능해지고 scene graph를 얻을 수 있다. 여기서 헷갈릴 수 있는데 scene graph라는게 단순히 image가 있을때 image를 나타내는 graph인 (man) - holding - (basketball bet) 가 만들어졌다고 해서 scene graph가 되는 것이 아니다. 정확한 scene graph는  (man) - holding - (basketball bet) 뿐만아니라 (man) 에 대응하는 image의 bounding box, (basketball bet) 에 대응하는 image 내의 basketball bat의 bounding box까지 연결이 되어야 scene graph이다. 그렇기 때문에 phrase grounding이 있어야 둘 간의 연결이 가능해서 scene graph generation이 되는 것이다.

그렇게 얻은 scene graph와 GLIP은 open vocabulary의 특징을 가지고 (b) 부분을 수행한다. (b)는 (a)에서 만든 scene graph가 있을 때 그 scene graph를 좀 더 update하는 과정이라고 볼 수 있다. (a)에서 나온 scene graph는 language description을 바탕으로 만들어진 것이기 때문에 그 language 내에 있는 단어로만 구성된다. 하지만 그 image의 bounding box를 표현할 수 있는 class는 사실 여러가지가 될 수 있기 때문에 그러한 여러 단어 list(text prompt)가 주어지면 그 안에서 적절한 단어를 찾아 update해주는 것이다. 예를 들면  class language descript에 없던 단어이지만 Text prompt 내에 대응되는 단어가 있다면 pretrained GLIP이 Visual-Semantic Space상에 비슷한 곳에 위치시킬 것이기 때문에 둘 간의 유사도를 측정하여 "player"로 update 시킬 수 있다.

  

## **3. Method**

이 논문에서 제안하는 model인 VS<sup>3</sup> 에 대한 설명이다.

  

### 3.1 Relation embedding module

앞에서 GLIP은 image-text pair를 통해 학습했기 때문에 둘의 정보를 모두 다루는 object feature를 얻을 수 있다고 했다. 이를 우리는 기존의 object feature O와 구분하기 위해 $\tilde{O}$로 나타낼 것이다. GLIP을 통해 object pair ($\tilde{o}_ {i}$, $\tilde{o}_ {j}$)와 bounding boxes (${b}_ {i}$, ${b}_ {j}$)를 얻으면 이를 이용해 다음과 같이 두 object사이의 relation feature를 얻는다. ${pair}_ {i -> j}$ 라는게 object i와 object j가 있을 때 둘의 관계를 나타내는 representation 기호이다. 두 object 사이의 relaiton 을 나타내기 위해선 두 object가 어떤 행위를 하고 있는지에 대한 외적인 모습 뿐만아니라 두 object 사이의 위치 정보도 중요하다. 왜냐하면 "man" - "eating" - "pizza"가 있을 때 eating을 알기 위해선 시각적으로 먹는 모습을 capture해야하겠지만, "man" - "sitting on" - "bench"라고 하면 "sitting on"을 추론하기 위해선 시각정보도 중요하지만 "man"의 좌표와 "bench"의 좌표를 보고 더 위에 있다라는 정보를 얻는 것도 중요한 단서가 되기 때문이다.  그래서 relation representation을 visual과 spatial feature의 concat으로 나타낸다. ${pair}_ {i -> j}^{visual}$ 은 i, j의 관계를 나타내는 representation 중 visual 특징으로만 얻은 feature를 말하고 아래 식으로 볼 수 있듯이 object feature간의 차와 합을 각각 learnable function f에 넣은 output의 합으로 표현하였다. ${pair}_ {i -> j}^{spatial}$ 은 두 object사이의 공간적 정보를 바탕으로 둘 사이의 representation을 나타낸 것이다. 

<p  align="center"><img  src="https://i.ibb.co/T1GFdys/eq1.jpg"  height="150px"  width="500px"></p>

위에서 함수 f는 2-layer MLP이고 dx, dy는 두 bounding box의 중심 좌표를 기준으로 x축으로 거리 vector, y축으로 거리 vector이며 dis는 dx, dy를 각각 제곱하여 합한 뒤 root를 씌운 값이다. 나머지 값들도 box를 이용한 계산 값이다.

  

### 3.2 Relation prediction

  

위에서 얻은 값으로 최종 relation에 대한 예측은 아래와 같이 진행된다. f는 MLP와 softmax activation을 포함하는 항으로  ${pair}_ {i -> j}$에 서로 다른 f를 사용해 $\hat{z}_ {i->j}$와 $\hat{y}_ {i->j}$를 얻는다. relateness란 두 object 사이의 관계가 있는지 없는지를 예측하는 뜻이고 semantic이란 만약 둘 사이의 관계가 있다면 어떤 관계("on"인지 "eating"인지)인지 예측하는 의미이다. 

<p  align="center"><img  src="https://i.ibb.co/nBLWKVQ/eq2.jpg"  height="250px"  width="500px"></p>

  

### 3.3 Obtaining Language Scene Graph Supervision

  

전체 model에서 (a) 부분에 대해 더 자세히 설명하겠다. 먼저 기존의 Scene Graph Parser를 이용해 text로부터 자동적으로 object와 relatioin에 대해 얻을 수는 있다. 이게 무슨 말이냐면 "a woman is playing the piano in the room"이라는 text가 있을 때 위 parser로 O = {woman, piano, room}, R = {<0, playing, 1>, <0, in, 2>}와 같이 object set과 relation set을 얻을 수 있다는 것이다. 하지만 이 것만으로는 진정한 scene graph가 될 수 없는데 왜냐하면 각 object text가 image상의 어떤 bounding box와 match되는지 알 수 없기 때문이다. 그렇기 때문에 GLIP을 여기에 적용한다면 text와 image상의 bounding box와 matching이 가능하게 되고 scene graph supervision을 만들 수 있게 된다.

  

### 3.4 Transferring to Open-vocabulary SGG

  

또한 GLIP을 SGG에 적용하여 novel category에 대해서도 예측이 가능하다고 했는데 이는 전에도 설명했다시피 pretrained GLIP을 이용하면 가능하다. GLIP은 비슷한 종류의 category text는 space상에서 비슷한 곳에 embedding 될 것이고 이와 대응되는 image embedding 또한 비슷한 곳에 embedding하게 할 수 있다(phrase groudning). 그래서 그저 text에 기존에 없던 category를  prompt에 넣어주는 것만으로 새로운 category와 bounding box를 mathing할 수 있게 되고 open-vocabulary SGG가 가능해진다.

  

아래 그림에서 왼쪽 상단을 보면 Text prompt에서 train과 test에 한 부분이 다르다. test시에 원래 없던 person이라는 category가 추가적으로 들어갔는데 이렇게 새로운 category가 들어가더라고 GLIP은 semantic space상에 man과 person은 유사한 의미의 text이므로 비슷한 곳에 embedding한다(파란색 네모). 그러므로 새로운 category에 대해 prediction이 가능하다.

  

<p  align="center"><img  src="https://i.ibb.co/jwDPFyR/model.jpg"></p>

위는 전체적인 구조에 대한 더 자세한 그림이다. 
  

## **4. Experiment**

  
  

### **Experiment setup**

  
  

* Dataset

Scene Graph Generation task에서 주로 사용하는 VG150을 사용하였다. 총 150의 object category와 50개의 relation category로 구성된다. 70%가 training으로 사용되고 나머지가 test로 사용된다.

기존의 SGG setting과 다르게 본 논문은 language supervised setting의 영역까지 확장했으므로 VG caption을 추가적으로 사용한다. COCO caption으로도 학습하여 추가적인 비교를 하였다.

  

* Evaluation metric

주로 사용하는 metric인 recall@K를 사용하였다.

  
  

### **Result**

  
  

#### - Fully supervised results

  
  

<p  align="center"><img  src=" https://i.ibb.co/ByZPmpr/result1.jpg"  height="400px"  width="600px"></p>

  

위는 fully supervised setting에서 결과이다. fully supervised setting이란 이미 human이 annotate한 image의 bouding box label과 둘 간의 관계도 이미 human이 annotate하여 그 답을 보고 학습하는 것이다. 이 때 label은 정해진 set 내에서 annotate 되었기 때문에 test에서 text prompt에도 정해진 set 내의 class만 넣어준다. 

그 결과 기존의 모델보다 좋은 성능을 보이는 것을 확일 할 수 있고 visual이랑 spatial을 모두 사용하는 것이 가장 좋았다.

  

#### - Language-supervised results

language-supervised setting이란 human annotated label을 사용하지 않고 직접 scene graph supervision을 만들어 사용하겠다는 뜻이다. 이때 만드는 방법은 먼저 image에 대응하는 language description으로 parser를 이용해<subject, predicate, object> triplet을 만든 뒤, 이 triplet의 단어를 VG에서 사용하는 category의 단어로 바꾸어 준다. 이게 무슨 말이냐면 "man eating DDDN"이라는 description이 있으면 이걸 먼저 <"man" - "eating" - "DDDN">으로 바꾸고 여기서 "DDDN"이라는 단어를 VG에서 사용하는 단어인 "pizza"로 바꾸어 최종 <"man" - "eating" - "pizza">를 얻겠다는 뜻이다. "DDDN"과 "pizza"가 유사하다는 것을 WordNet 등으로 알 수 있다. 그렇게 얻은 triplet을 GLIP을 통해 bouding box와 grouding 시켜 scene graph supervision을 얻고 이걸로 model을 학습하면 lanuguage supervised setting이다.


<p  align="center"><img  src="https://i.ibb.co/vB67Pdw/result2.jpg"  height="600px"  width="600px"></p>

Unlocalized graph(human이 annotate한 scene graph에서 grounding을 제거)을 이용한 것과 VG caption과 COCO cation을 이용하여 language description으로 supervision을 만들어 이를 활용해 모델을 학습했을 때 결과 비교이다. 기존 model들은 language grounding 항이 없어서 다른 model을 method를 추가해주었다. 전체적으로 좋은 결과를 보인다. 

  

#### - Open-vocabulary SGG

  

<p  align="center"><img  src="https://i.ibb.co/wJ1zx7F/result3.jpg"  height="300px"  width="500px"></p>

cap, racket, player, logo, lady, laptop과 같은 novel category에 대해 scene graph를 잘 생성하는 것을 확인할 수 있다.

  

  

## **5. Conclusion**

 
pre-trained VSS(GLIP)를 활용해 SGG에 적용하여 open vocabulary로 확장하고 language description을 이용해 기존의 label annotation의 한계를 잘 극복하려는 연구다.


## **6. Author Information**

* Author name : Jaehyeong Jeon

	* Affiliation : [DSAIL@KAIST](https://dsail.kaist.ac.kr/)
	* Research Topic : Scene graph generation
	* Contact : wogud405@kaist.ac.kr  


## **7. Reference & Additional materials**

  

  

  

* Github Implementation

  

  

  

* Reference

  

  

  

- [[CVPR-23] Learning to Generate Language-supervised and Open-vocabulary Scene Graph using Pre-trained Visual-Semantic Space](https://arxiv.org/abs/2109.02227)