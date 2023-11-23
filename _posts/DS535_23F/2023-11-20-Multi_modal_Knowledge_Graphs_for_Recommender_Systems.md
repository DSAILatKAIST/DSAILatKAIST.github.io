---
title:  "[CIKM 2020] Multi-modal Knowledge Graphs for Recommender Systems"
permalink: 2023-11-20-Multi_modal_Knowledge_Graphs_for_Recommender_Systems.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Multi-modal Knowledge Graphs for Recommender Systems

## 0. Preliminary



#### 1) Knowledge Graph (KG)

directed graph G = (V,E) - V: 노드 집합, E: 엣지 집합
- 노드: entity  
ex)Toy story, Tom Hanks, Walt Disney, ...
- 엣지: relation  
ex) director, actor, ...
- Triplet: head entity - relation - tail entity (h,r,t)  
ex) Toy Story - producer - Walt Disney
  
![figure1](https://i.ibb.co/sbNh7bB/figure1.png)


#### 2) Multi-modal Knowledge Graph (MKG)
multi-modal entity(텍스트, 이미지)가 추가된 KG

![figure2](https://i.ibb.co/3NXB0P7/figure2.png)



#### 3) Collaborative Knowledge Graph (CKG)

![figure3](https://i.ibb.co/zRLFT9G/figure3.png)

- item에 대한 MKG와 user-item bipartite graph가 통합된 graph
- 기존 추천 시스템에서 사용되는 기본적인 user-item bipartite graph에서 item의 추가적인 정보를 사용하기 위해 item에 대한 MKG를 user-item graph에 통합

![figure7](https://i.ibb.co/PjtV3BX/2023-11-18-113410.png)



## 1. Problem Definition

Task: Multi-modal knowledge graph(MKG) 기반 추천 시스템 구축

- Input: user-item graph와 MKG가 통합된 CKG(collaborative knowledge graph)
- Output: user가 item을 채택할 확률을 예측하는 모델





## 2. Motivation

KG 기반의 추천 시스템은 data sparsity 문제와 cold start 문제를 완화할 수 있어 최근에 많이 사용되고 있다. 하지만 기존 KG 기반 추천 시스템의 대부분은 다양한 multi-modal 정보를 무시한다.

knowledge graph representation learning은 KG 기반 추천 시스템에서 매우 중요한 역할을 하는데, 이 중 multi-modal 정보를 활용한 multi-modal knowledge representation learning은 크게 feature 기반 방법과 entity 기반 방법으로 나뉜다.

### feature 기반 방법
- multi-modal 정보를 entity의 보조적인 feature로 사용 
- 예시) Toy Story (item entity) 의 feature 중 하나로 Image에서 추출된 vector 사용
- 이 방법에서는 knowledge graph의 모든 entity가 multi-modal 속성을 가져야 하지만, 실제 그래프에서는 일부 entity들이 multi-modal 정보를 전혀 포함하지 않는다. 

### entity 기반 방법
- 다양한 유형의 multi-modal 정보를 각각 독립적인 tail entity로 두는 knowledge graph의 triple을 사용 
- 예시) Toy Story (head entity) - has image (relation) - Image (tail entity)
- 앞서 본 MKG 그림 참조
- 이 방법을 사용하는 연구들에서는 해당 entity에 대한 multi-modal 정보의 융합을 무시하고 각 triple을 독립적으로 사용한다.
 
본 논문에서 제안한 Multi-modal Knowledge Graph Attention Network (MKGAT)는 entity 기반 방법을 사용하며, 기존 feature 기반 방법론들의 한계점과 entity 기반 방법론들의 한계점을 개선한다. 즉, 데이터 소스에 대한 낮은 요구 사항과 multi-modal triple 정보의 융합을 만족하기 위해 entity 정보 aggregation과 entity 관계 추론을 통해 MKG(크게 보면 CKG)를 모델링하여 임베딩된 representation을 추천에 사용한다. 각 과정에서의 구체적인 방법은 method part에 정리하였다.




## 3. Method

모델의 전체 프레임워크 구조는 아래 그림과 같다. input graph는 CKG이다.
임베딩 모듈과 추천 모듈로 나뉘며, 임베딩 모듈에서는 각 노드에 대한 임베딩을 위해 MKG를 모델링하고, 추천 모듈에서는 업데이트된 임베딩을 사용하여 user-item 간 score를 에측한다.

![figure4](https://i.ibb.co/TvSwZsm/figure5.png)

### 3.1 Multi-modal Knowledge Graph Embedding



#### 1) Multi-modal Knowledge Graph Entity Encoder

각각의 특정 데이터 유형을 독립적인 벡터로 임베딩하기 위해 서로 다른 인코더를 사용한다.

![figure5](https://i.ibb.co/8NdBwwr/figure4.png)


- head entity id, relation id, tail entity id: 독립적인 임베딩 layer
- 이미지 (tail entity): ResNet50
- 텍스트 (tail entity): Word2Vec

dense layer를 사용하여 head, relation, tail 임베딩 벡터를 동일한 차원으로 통일



#### 2) Multi-modal Knowledge Graph Attention Layer

GAT를 활용한 KG relation 임베딩

![figure6](https://i.ibb.co/QHZRPWg/figure6.png)



i) Propagation layer

entity $h$에 triple 정보를 aggregate 한다. (GAT와 유사)

$e_ {agg} = \sum_ {(h,r,t)\in N_ h} \pi(h,r,t) e(h,r,t)$

- $N_ h$: h가 head entity인 triplet(h,r,t)들의 집합 

- $e(h,r,t)$: triplet (h,r,t)의 임베딩
  - head entity, relation, tail entity 임베딩의 연결에 대해 linear transformation으로 학습
  - $e(h,r,t) = W_ 1(e_ h \vert \vert e_ r \vert \vert e_ t)$


- $\pi(h,r,t)$: attention score
  - attention 이후 normalize
  - $ \pi(h,r,t)=LeakyReLU(W_ 2 e(h,r,t)) $
  - $ \pi(h,r,t)= \frac{exp(\pi(h,r,t))}{\sum_ {(h,r',t')\in N_ h} exp(\pi(h,r',t'))} $


ii) Aggregation layer

head entity representation $e_h$와 앞서 propagate 된 $e_ {agg}$ 를 aggregate 한다.

- 방법1: Add aggregation


$ f_ {add} = W_ 3 e_ h + e_ {agg} $


- 방법2: Concatenation aggregation


$ f_ {concat} = W_ 4 (e_ h \vert \vert e_ {agg})$


두 방법은 각각 사용해보고 experiment에서 비교


iii) High-order propagation

propagation-aggregation layer를 n개 쌓음으로써 n-hop neighbor의 정보를 반영한다.



#### 3) Knowledge Graph Embedding

triple (h,r,t)의 knowledge graph 임베딩이 translation principle $e_ h+e_ r \approx e_ t$를 만족하도록 translational scoring function을 이용해 학습을 진행한다.

$ score(h,r,t)= \vert e_ h + e_ r - e_ t \vert _2^2 $

![figure8](https://i.ibb.co/sHCrYpy/2023-11-18-141616.png)

knowledge graph 임베딩에 대한 학습은 pairwise ranking loss를 통해 valid triplet(실제 존재하는 triplet)과 broken triplet(실제 존재하지 않는 triplet) 사이의 차별을 유도한다.


$ L_ {KG}=\sum _{(h,r,t,t')\in \tau} -ln \sigma(score(h,r,t')-score(h,r,t)) $


$ \tau = \{ (h,r,t,t') \vert (h,r,t)\in G, (h,r,t')\notin G \}, \sigma: sigmoid $
- broken tiplet (h,r,t')는 valid triplet (h,r,t)의 tail entity를 랜덤으로 대체함으로써 구성됨




### 3.2 Recommendation

MKG 임베딩을 통해 업데이트된 item과 user 간 bipartite graph 를 사용한다.
MKG 임베딩 모듈과 마찬가지로, 추천 모듈 역시 MKGs attention layer를 사용해 item entity의 정보를 aggregate한다. 

추가적으로, layer-aggregation mechanism을 통해 각 layer의 representation을 concat하여 하나의 벡터로 만든다.
- MKG 임베딩에서는 L 개의 layer를 쌓았을 때 최종 L번째 임베딩 벡터만 사용
- 여기서는 1번 layer, 2번 layer, ..., l번 layer 벡터 concat


$ e_ u^* = e_ u^0 \vert \vert ..... \vert \vert e_ u^L, \quad e_ i^* = e_ i^0 \vert \vert .....\vert \vert e_ i^L $


- L: MDKs attention layer 수
- L을 조정하여 propagation 강도를 조절

최종적으로, user와 item representation의 inner product를 통해 스코어를 예측한다.


$ \hat y(u,i) = {e^*_ u}^Te^*_ i  $


BPR(Bayesian Personalized Ranking) loss를 통해 prediction loss를 최적화한다.


$ L_ {recsys} = \sum_ {(u,i,j)\in O} -ln\sigma(\hat y(u,i)-\hat y(u,j))+\lambda \vert \vert \theta \vert \vert ^2_ 2 $


$ O = \{ (u,i,j) \vert (u,i)\in R^+, (u,j)\in R^-\} $


- $R^+$: observed interactions between user $u$ and item $i$
- $R^-$: unobserved interactions between user $u$ and item $j$

전체 프레임워크가 실행될 때, MKGs 임베딩 모듈과 추천 모듈을 교대로 학습하여 파라미터 업데이트




## 4. Experiment
### 4.1 Experiment setup

#### 1) Datasets

MovieLens <https://grouplens.org/datasets/movielens/>

- 영화에 대한 기존 rating을 implicit feedback 데이터로 변환
- 이에 대한 knowledge graph는 Microsoft Satori를 사용하여 구성: <https://arxiv.org/pdf/1905.04413.pdf>
- 이미지는 FFmpeg를 사용해 트레일러의 주요 프레임 추출 <http://ffmpeg.org/>, 텍스트는 TMDB에서 영화 설명 크롤링<https://www.themoviedb.org/>

Dianping
- 맛집을 검색하고 정보를 얻을 수 있는 중국 생활 정보 서비스 사이트
- knowledge graph는 Meituan Brain에서 수집
- 인기 추천 요리의 이미지 수집,텍스트는 사용자 리뷰 크롤링

![table1](https://i.ibb.co/cx8pV0f/table1.png)



#### 2) Baselines
- NFM(Neural Factorization Machines): state-of-the-art Factorization Machines

- CKE(Collaborative Knowledge Base Embedding): Collaborative Filtering(CF)에 structural knowledge, 텍스트, 이미지 등의 multi-modal 정보가 통합된 프레임워크로 추천

- KGAT(Knowledge Graph Attention Network): TransR 모델을 적용하여 entity에 대한 representation을 얻은 다음 각 entity에서 외부 entity로 propagation을 실행

- MMGCN(Multi-modal Graph Convolution Network): state-of-the-art multi-modal 모델; 각 modal에 대해 user-item bipartite graph를 구축한 다음 GCN을 사용하여 각 그래프를 훈련하고, 마지막으로 다른 modal의 노드 정보를 병합


#### 3) Evaluation Metric

top-k 추천을 위한 평가 지표 (k = 20)
- recall@k
- ndcg@k


### 4.2 Experimental Results

![table2](https://i.ibb.co/3T2cPTg/table2.png)

제안된 MKGAT 모델은 두 데이터셋에서 모든 베이스라인의 성능을 능가한다.


i) Effects of Modalities

![table3](https://i.ibb.co/qdsdFJ5/table3.png)

- multi-modal feature를 사용한 경우 single-modal feaure만 사용했을 때에 비해 성능이 좋다.
- 이미지가 텍스트보다 추천에 중요한 역할을 한다.
- MKGAT는 KGAT에 비해 multi-modal 정보를 item entity로 더 효과적으로 통합할 수 있다.

ii) Effect of Model Depth

![table4](https://i.ibb.co/7zrJbBz/table4.png)


- 최적의 layer 수는 데이터셋에 따라 다르고, knowledge graph embedding 모듈에서와 추천 모듈에서 각각 다르다.

iii) Effect of Combination Layers

Aggregation layer로 Concat layer를 사용한 방법이 Add layer를 사용한 방법보다 성능이 좋다.

![table5](https://i.ibb.co/bzpxckT/table5.png)

- Concat은 여러 feature 간 차원의 확장으로, 다양한 semantic space에서 feature를 상호 작용하는 데 더 적합하다.



## 5. Conclusion

MKGAT는 multi-modal knowledge graph를 활용하여 entity 추론 및 이웃 정보 aggregation을 향상시킨 새로운 추천 모델이다. 본 논문은 추천 시스템에서 multi-modal knowledge graph와 user-item graph가 결합된 Colalborative Knowledge Graph의 사용을 탐구하려는 초기 시도의 연구이다. 기존 SOAT에 비해 높은 성능을 보이며, multi-modal 정보를 knowledge graph에 적용하는 것에 대한 효과성을 증명한다.

이미지나 텍스트와 같은 사이드 정보가 추천 시스템에 사용되는 연구들이 많은데, 이를 knowledge graph에 적용해 user-item graph와 결합되어 모델링하는 것이 인상 깊었다. 본 연구에서는 knowledge graph와 user-item bipartite graph를 교대로 학습시키는데, 하나의 heterogeneous graph를 구성하여 end-to-end 프레임워크로 동시에 학습하는 방법도 고려해볼 수 있을 것 같다.



## Author Information
- KAIST 산업및시스템공학과 석사과정 박형민 <mike980409@kaist.ac.kr>