---
title:  "[ICLR 2021] Generative Scene Graph Networks"
permalink: Generative_Scene_Graph_Networks.html
tags: [reviews]
---

# **Generative Scene Graph Networks**

  

본 논문은 Scene Graph Generation에서의 고질적인 문제인 long-tail problem과 Semantic ambiguity problem을 해결하기 위해 dataset을 아예 수정하여 성능 향상을 보인 논문이다.

  

## **1. Motivation**

  

Human cognition의 핵심은 observation으로부터 pure한 object를 찾는 것이고 최근 관련하여 다양한unsupervised object-centric representation이 연구되어 왔다. 이러한 연구들을 통해 얻은 object의 symbolic representation은 relational reasoning이나 causal inference 등 다양한 분야에 사용될 수 있다.

  

특히나 본 논문에서는 natural scene이 compositional objects(한 object가 여러 primitive part의 조합으로 이루어짐)으로 구성되어있다는 특징을 motivation으로 진행된다. Human은 primitive parts와 그것으로 부터 나아가는 part-whole relationship을 인지할 수 있기때문에 전체적인 object를 인지한다는 느낌이다.

이렇게 여러 small primitive part의 조합으로 object를 추론하기 때문에 복잡한 object도 보다 효율적으로 추론할 수 있게 되고 새로운 primitive part의 조합으로 new object를 create할 수도 있다.

  

본 논문에서는 unlabed image로부터 scene graph를 추론할 것이고 이를 위해 variational autoencoder를 사용하였다. 이때의 challenge는 obeject를 구성하는 part를 찾아야하는데 이 part가 image 상에서 occlusion이 나타나면 적절하게 분리하기 어렵하는 것이다. 그래서 object로부터 적절한 primitive part를 찾고 이들을 tree형태로 조합하여 최종 hierarchical scene graph를 얻어내는게 본 논문의 목표라고 이해할 수 있다.

  

이를 위한 key observation은 scene graph가 recursive structure를 가진다는 것이다. 전체적인 structure of tree를 추정하는 것은 그 tree의 structure of subtree를 추정하는 과정과 비슷하다는 느낌으로 top-down 방식으로 하나의 image로부터 scene graph를 얻기 위해 image에서 여러 object를 찾고 그 object들을 구서하는 part를 찾는 방향으로 이해할 수 있다.

  

이러한 관점은 기존의 scene decomposition method인 SPACE를 reuse할 수 있게 한다. 다만 SPACE는 object에서 part로 분리할때 occlusion에 대해 어려움이 있었는데 이는 SPACE의 bottom-up 방법에 의한 것이다. 그래서 본 논문에서는 top-down 방식으로 하되 composition에 대한 prior를 도입하여 곂치는 현상에 대해 더 개선된 방법은 도입했다.

  
  

## **2. Related Work**

  

본 논문과 관련된 여러 이전 연구들과 관련 개념들에 대한 간략한 설명이다.

### Object-centric representation

본 논문의 방법은 unsupervised object-centric representation learning에 속하는 연구로 이 분야의 method들은 주로 supervision없는 상황에서 scene을 object로 분해하고 이러한 object의 representation을 학습하는 end-to-end model을 개발한다. 이러한 모델은 크게 2가지로 나뉠 수 있는데 그 것이 1) scene-mixture models, 2) spatial-attention models 이다. 이러한 모델들와 본 논문의 차이점은 object를 part까지 나눴다는 점이고 다만 inference시에는 spatial-attention model을 사용했지만 occlusion을 다루기 위해 prior를 도입하였다는 차이가 있다.

  

### Hierarchical scene representations

Scene에서 part-whole relationship을 modeling하는 것은 image classification, parsing, segmentation 분야에서 널리 활용되어 왔다. 하지만 이러한 모델은 object 하나에 대해서만 적용되었고 여러 object를 다루는 scene generation에는 적용할 수 없었다. 또한 기존의 연구들은 part-whole relationship의 학습을 위해 predefined part에 대한 정보가 필요했다. 대신 본 논문에서는 multi-object에 대해 part-whole relationship을 학습하고 이때 individual part에 대한 knowledge가 필요하지 않다는게 차이점이다.

또한 shape generation에서 part hierarchies가 연구되었었는데 이때의 hierarchy는 input으로 들어왔었다. 이번 연구에서는 hierarchy를 input으로 사용하는게 아니라 static scene을 input으로 하고 ouput으로 hierarchy를 내놓는다는 차이가 있다.

  

### Hierarchical latent variable models

본 논문의 model은 hierarchical latent variable model과 관련이 있고 이 개념을 이용하여 object와 part간의 relationship을 잘 capture하는 hierarchical structure를 학습하는 데에 있다.

  
  

## **3. GENERATIVE SCENE GRAPH NETWORKS**

본 논문의 main method인 Generative Scene Graph Networks에 대한 설명이다.

  

### 3.1 Generative Process

x라는 image가 set of foreground variable인 Z~fg~와 background variable Z~bg~ 로 다음과 같이 구성된다고 하자.

주어진 forground 정보와 그로부터 얻은 backgound 정보, 그 둘로부터 얻은 x의 곱의 형태를 띄고 있다.

  

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233981211-8d1d8d99-687e-47c7-b98e-8e0f95e9e267.png"></p>

  

여기서 그럼 Z~fg~ 가 무엇이나면 아래 그림으로 이해할 수 있다. 여기서 leaf node는 scene에서의 primitive entity로 더이상 decompose되지 않는 단위이고 internal node는 child node의 composition으로 나타난 abstract node로 이해 할 수 있다. 이때의 composition 정보는 둘 사이의 edge에 담겨있으며 composition은 affine transformation으로 rotation, scaling, translation을 포함한다. Z~v~^pose^ 나 Z~v~^appr^ 의 의미는 각각 node v와 그 parent 사이의 relative pose와 node v과 그의 child의 정보를 종합한 apprearence를 나타낸다고 이해할 수 있다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233982909-d3e8dcf8-d817-496a-ba09-8dddc08f6be3.png"></p>

위의 그림을 바탕으로 최종 Z~fg~를 구하면 다음 식과 같다. set of all node V에 대해 root node인 r에 대해 정보와root node를 제외한 node들에 대해 pose과 appr정보를 하나하나 더준해주어 전체 foreground 정보를 얻는다고 이해할 수 있다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233981218-2ae299bd-2efe-4f72-b821-e3cef2b25a4d.png"  height="50px"  width="600px"></p>

  

#### - Representing tree structures

근데 위의 계산식은 일단 tree 구조가 있어야 계산할 수 있다. 그럼 위의 tree 구조자체는 어떻게 얻을까? 위와 같은 tree 구조를 다루기 위해선 latent representation에 대한 개념이 필요하다. 일단 scene에 대한 possible tree structure에 대한 제한을 두기 위해 각 node에 maximum out-degree를 적용하고 구체적인 구조를 결정하기 위해 각 node와 연결될 수 있는 가능한 edge들을 고려할 것이다.

이를 위해 node v와 그의 parent node 사이의 edge에 대해 Bernoulli Variable P~v~^pres^ 를 설정하였다. P~v~^pres^ = 0이란 뜻은 edge가 없다는 뜻이다. $\bar{z}$~r~^pres^ 가 root r에서 node v까지 이어지는 variable의 곱이라고 하면 다음과 같이 나타낼 수 있다. 즉, node v까지의 edge의 존재 유무는 node v부터 parent까지의 edge와 root r부터 그의 parant까지의 edge의 곱으로 나타낼 수 있다는 뜻이다.  


<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233981219-f4fc7b9d-03c6-4f1f-98fd-ba6da03e097a.png"  height="40px"  width="500px"></p>  


이 사실을 바탕으로 Z~fg~ 식을 재구성하면 다음과 같다. edge까지 고려해준 모습이다.  

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233981222-5a905a16-85ac-4ae2-88dc-d756ec97b9b8.png"></p>

  
  

#### [ Differentiable decoder ]

decoder는 recursive compositing process를 따르는 역할로 encoder는 그 반대의 역할이다. 구체적으로는 먼저 각 leaf node로부터 g()라는 neural network를 이용하여 small patch $\hat{x}$~v~ 와 대응하는 binary mask $\hat{m}$~v~ 를 얻는다. 이러한 primitive patch를 가지고 composition을 통해 object와 더 나아가 scene을 얻는데 그 과정은 다음과 같다.

  

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233981224-8f0ce0c1-8ce8-4b00-9927-5feaa9116b6e.png"  height="100px"  width="500px"></p>

위 식은 아래 그림으로 이해하면 더 편하다. 유의해야할 점은 단순히 patch와 mask를 합친게 아니라 node v의 pose까지 고려하여 occlusion을 다루었다는 점이다.

  

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/233999904-8b2379cd-33a8-4ced-a222-3591dc57e28b.png"  height="400px"  width="600px"></p>

  

위 과정을 거치면 최종적으로 root r에 대한 $\hat{x}$~r~ 와 $\hat{m}$~r~ 을 얻게 된다. 그 이후에 다른 spatial broadcast decoder를 이용하여 Z~bg~ 를 \hat{x}~bg~ 로 decode해준다. 이렇게 얻은 $\hat{x}$~r~와 $\hat{x}$~bg~를 가지고 full scene은 다음과 같이 얻을 수 있다.

  

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234002632-0aabef5f-2995-4bd0-b403-45ab5b8eac07.png"  height="40px"  width="600px"></p>

  

### 3.2 Inference and Learning

  

가장 첫번째 식을 계산하는 과정이 intractable integral이기 때문에 variational inference로 근사 할 것이다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234003966-3434e7d0-a2ce-4252-bf61-9b8233978638.png"  height="40px"  width="600px"></p>

foreground variable을 유추하기 위해 probabilistic scene graph가 recursive하다는 사실을 이용할 것이고 이는 root node의 child가 가장 먼저 inferred되고 그 이후 child node의 subtree로 내려가면서 infer한다는 개념이다.

이러한 top-down factorization은 아래와 같은 식으로 나타난다. 기본적으로 parent의 정보를 가지고 node v를 유추하는 느낌이다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234003971-4dabc9e8-96db-4935-86de-c37e070bafa4.png"  height="80px"  width="600px"></p>

위 식은 이전식과 달라진게 x에 대한 condition이 더 주어졌다는 것이다. 이는 top-down information에 bottom-up image feature를 더해준 것으로 이는 entity v에 대해 더 relevent한 information을 제공하는 역할을 한다.

  
  

## **4. Experiment**

  
  

### **Experiment setup**

* Dataset

GSGN의 ability를 제대로 측정하기 위해 본 논문에서는 dataset을 직접 만들었다. unsupervised object-centric representation learning에 흔히 쓰이는 Multi-dSprites와 CLEVR을 합쳐 하나의 데이서 셋을 만들었고 이를 각각 2D Shapes과 Compositional CLEVR datasets이라고 부르기로 하였다.

각 데이터 셋은 3가지 type의 primitive part와 이 3가지 part로 구성되는 10가지 type의 object가 있다. object type 중 3가지는 single part로 구성되고 다른 3가지는 two part로 구성되고 마지막 4가지 object type의 3가지 part로 구성된다. scene을 만들기 위해 object를 random(size, type, position and orientation)하게 sample하였다.

  

* baseline

이전의 hierarchical scene representations은 single-object scenes을 가정하였고 특히나 prior knowledge가 필요하였다. 그래서 우리의 dataset에 직접적으로 적용할 수가 없다. 그 대신 SOTA non-hierarchical scene decomposition model인 SPACE를 우리의 상황에 바꾸어서 baseline으로 사용하였다.

  
  

### **Result**

  

#### - Scene graph inference

scene graph inference의 과정은 다음 그림으로 이해할 수 있다. input image에 대한 bounding box는 pose variable을 나타내고 reconstruction은 inferred appearence variable을 decoder로 넣은 후의 결과이다. 그림을 보면 알 수 있다시피 object와 part를 잘 분리하는 것을 알 수 있다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234011866-15bb7af4-e0fc-4496-9b1d-573bf40724ce.png"  height="400px"  width="800px"></p>

  

다음은 수치적인 result를 나타낸다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234014047-9ca7528a-63bd-4c97-a259-e3678f0e5745.png" height="600px"  width="600px"></p>

counting accuracy는 inferred scene graph에서 노드 수의 정확성을 측정하고 F1 점수는 inferred node가 실제 entities를 잘 capture하는 지를 나타낸다. 대부분 좋은 결과를 내지만 GSGN-No_Aux는 scene graph structure 형성에 실패하는데 이 이유는 모든 presence variable이 1로 가는데 이는 redundant nodes때문이라고 말한다.

  

Compositional CLEVR dataset에 대해서도 GSGN이 SPACE보다 더 좋은 성능을 보였는데 이는 SPACE가 occlusion이 발생한 부분에 대해 seperating을 어려워하기 때문이라고 말한다. 이와 관련된 성능은 Table 3에서 더 자세히 볼 수 있는데 Table 3은 occlusion level에 따라 실험한 결과로 <100인 경우 SPACE-P의 성능이 매우 낮은데 이때 occluded part에 대해 miss하는 경향이 있기 때문이다.

  

#### - Scene graph manipulation.

GSGN으로 얻은 scene graph는 interpretable한 tree structure와 pose variable을 가지기 때문에 infered scene graph에서 얻은 latent variable들을 조합하여 새로운 object와 scene을 만들 수 있다.

<p  align="center"><img  src="https://user-images.githubusercontent.com/48014450/234019972-9cba7300-9f6a-460e-9c90-e39aff9a5bfc.png" height="400px"  width="600px"></p>

위 그림은 그 결과로 scene graph inference에서 얻은 scene graph에다가 scale과 coordinate의 변화를 주어 얻은 새로운 object와 scene이다. 새로운 object와 scene도 어색함이 없이 잘 generate되는 것을 확인할 수 있다.

  

## **5. Conclusion**

본 논문에서는 multi-object scenes에 대해 knowledge of individual parts없이 deep generative model을 적용한 unsupervised scene graph을 수행한 GSGN을 제안하였다. GSGN은 top-down prior와 bottom-up image features를 활용하였고 이는 severe occlusion이 발생할때 적절히 대처하게 해주었다. 또한 GSGN은 scene graph manipulation을 통해 new object와 scene도 generate할 수 있었다.

이러한 방식을 바탕으로 좀 더 realistic한 환경에 deeper hierarchies와 more complex appearance에도 잘 적용하게 하는 것이 future work이다.

  
  
  

## **6. Reference & Additional materials**

  

Please write the reference. If paper provides the public code or other materials, refer them.

  

* Github Implementation

* Reference

- [[ICLR-21] Generative Scene Graph Networks](https://openreview.net/forum?id=RmcPm9m3tnk)