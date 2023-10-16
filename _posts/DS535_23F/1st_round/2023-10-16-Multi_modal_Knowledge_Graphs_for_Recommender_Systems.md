---
title:  "[CIKM 2020] Multi-modal Knowledge Graphs for Recommender Systems"
permalink: 2023-10-16-Multi_modal_Knowledge_Graphs_for_Recommender_Systems.html
tags: [reviews]
use_math: true
usemathjax: true
---

# Multi-modal Knowledge Graphs for Recommender Systems

## 0. Preliminary



#### 1) Knowledge Graph (KG)

directed graph G = (V,E) - V: 노드 집합, E: 엣지 집합
- 노드: entity
- 엣지: relation
- Triplet: head entity - relation - tail entity (h,r,t)
  
![image](https://github.com/parkhyeongminn/paper_review/assets/81458623/32fc0e00-a88e-46e9-a5fb-f951568fed21)




#### 2) Multi-modal Knowledge Graph (MKG)
multi-modal entity(텍스트, 이미지)가 추가된 knowledge graph

<img width="279" alt="image-1" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/477ef9b3-bc1b-4244-8a1c-97dd24eebe56">



#### 3) Collaborative Knowledge Graph (CKG)
item-entity knowledge graph와 user-item bipartite graph가 통합된 knowledge graph

<img width="266" alt="image-2" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/a12af34e-48c2-48f2-b25c-c61740093eba">





## 1. Problem Definition

Task: Multi-modal knowledge graph(KG) 기반 추천 시스템 구축

- Input: user-item graph와 multi-modal knowledge graph를 포함하는 collaborative knowledge graph
- Output: user가 item을 채택할 확률을 예측하는 모델





## 2. Motivation

KG 기반의 추천 시스템은 data sparsity 문제와 cold start 문제를 완화할 수 있어 최근에 많이 사용되고 있다. 하지만 기존 KG 기반 추천 시스템의 대부분은 다양한 multi-modal 정보를 무시한다.

knowledge graph representation learning은 KG 기반 추천 시스템에서 매우 중요한 역할을 하는데, 이 중 multi-modal 정보를 활용한 multi-modal knowledge representation learning은 크게 feature 기반 방법과 entity 기반 방법으로 나뉜다.

feature 기반 방법은 modal 정보를 entity의 보조적인 feature로 처리한다. 이 방법에서는 knowledge graph의 모든 entity가 multi-modal 속성을 가져야 하지만, 실제 그래프에서는 일부 entity들이 multi-modal 정보를 전혀 포함하지 않는다. 반면에, entity 기반 방법은 다양한 유형의 multi-modal 정보를 knowledge graph의 triple로 처리한다. 이 방법에서는 해당 entity에 대한 multi-modal 정보의 융합을 무시하고 각 triple을 독립적으로 처리한다.

이 두가지 한계점을 개선하기 위해 논문에서 제안한 Multi-modal Knowledge Graph Attention Network (MKGAT)는 MKG의 데이터 소스에 대한 낮은 요구 사항과 multi-modal 정보 융합을 만족한다. 이를 위해 entity 정보 aggretation과 entity 관계 추론을 통해 MKG를 모델링 하여 임베딩된 representation을 추천에 사용한다.





## 3. Method

모델의 전체 프레임워크 구조는 아래 그림과 같다.

<img width="568" alt="image-3" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/9d2dcadc-3dc3-4a0c-900c-66bcebddd851">

### 3.1 Multi-modal Knowledge Graph Embedding



#### 1) Multi-modal Knowledge Graph Entity Encoder

각각의 특정 데이터 유형을 독립적인 벡터로 임베딩하기 위해 서로 다른 인코더를 사용한다.

<img width="281" alt="image-5" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/03eb95db-639b-4fad-afe2-f6bbc43064c8">


- structured knowledge(entity id, relation id): 독립적인 임베딩 layer
- 이미지: ResNet50
- 텍스트: Word2Vec

dense layer를 사용하여 모든 형태의 entity를 동일한 차원으로 통일



#### 2) Multi-modal Knowledge Graph Attention Layer

GAT를 활용한 KG relation 임베딩

<img width="281" alt="image-6" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/0ca2af69-2e86-485b-af63-e2fadd7bcd6e">



i) Propagation layer

transE 모델을 통해 knowledge graph의 구조화된 representation을 학습한다.

** transE 모델: entity간의 관계 $h+r \approx t$를 저차원 임베딩 변환을 통해 모델링하는 방법론

이후에 entity $h$에 multi-modal 이웃 entity 정보를 aggregate 한다.

$$ e_{agg} = \sum_{(h,r,t)\in N_h} \pi(h,r,t) e(h,r,t) $$ 

- $N_h$: h와 연결된 triplet(h,r,t)들의 집합

- $e(h,r,t)$: triplet (h,r,t)의 임베딩
  - head entity, relation, tail entity 임베딩의 연결에 대해 linear transformation으로 학습
$$e(h,r,t) = W_1(e_h||e_r||e_t)$$

- $\pi(h,r,t)$: attention score
  - attention 이후 normalize
$$ \pi(h,r,t)=LeakyReLU(W_2e(h,r,t)) $$
$$ \pi(h,r,t)= \frac{exp(\pi(h,r,t))}{\sum_{(h,r',t')\in N_h} exp(\pi(h,r',t'))} $$


ii) Aggregation layer

entity representation $e_h$와 앞서 propagate 된 $e_{agg}$ 를 aggregate 한다.

- 방법1: Add aggregation

$$ f_{add} = W_3 e_h + e_{agg} $$

- 방법2: Concatenation aggregation

$$ f_{concat} = W_4 (e_h||e_{agg})$$


iii) High-order propagation

propagation-aggregation layer를 n개 쌓음으로써 n-hop neighbor의 정보를 반영한다.



#### 3) Knowledge Graph Embedding

triple (h,r,t)의 knowledge graph 임베딩이 translation principle $e_h+e_r \approx e_t$를 만족하도록 translational scoring function을 이용해 학습을 진행한다.
$$ score(h,r,t)=|e_h+e_r-e_t|_2^2 $$

knowledge graph 임베딩에 대한 학습은 pairwise ranking loss를 통해 valid triplet과 broken triplet 사이의 차별을 유도한다.
$$ L_{KG}=\sum_{(h,r,t,t')\in \tau} -ln \sigma(score(h,r,t')-score(h,r,t)) $$
$$ \tau = \{ (h,r,t,t')|(h,r,t)\in G, (h,r,t')\notin G \}, \sigma: sigmoid $$
- broken tiplet (h,r,t')는 valid triplet (h,r,t)의 tail entity를 랜덤으로 대체함으로써 구성됨




### 3.2 Recommendation

knowledge graph 임베딩 모듈과 마찬가지로, 추천 모듈 역시 MKGs attention layer를 사용해 이웃 entity의 정보를 aggregate한다. 추가적으로, layer-aggregation mechanism을 통해 각 layer의 representation을 concat하여 하나의 벡터로 만든다.
$$ e_u^* = e_u^0||.....||e_u^L, \quad e_i^* = e_i^0||.....||e_i^L $$
- L: MDKs attention layer 수
- L을 조정하여 propagation 강도를 조절

최종적으로, user와 item representation의 inner product를 통해 스코어를 예측한다.
$$ \hat y(u,i) = {e^*_u}^Te^*_i  $$

BPR(Bayesian Personalized Ranking) loss를 통해 prediction loss를 최적화한다.
$$ L_{recsys} = \sum_{(u,i,j)\in O} -ln\sigma(\hat y(u,i)-\hat y(u,j))+\lambda||\theta||^2_2 $$
$$ O = \{ (u,i,j)|(u,i)\in R^+, (u,j)\in R^-\}$$
- $R^+$: observed interactions between user $u$ and item $i$
- $R^-$: unobserved interactions between user $u$ and item $j$

전체 프레임워크가 실행될 때, MKGs 임베딩 모듈과 추천 모듈의 파라미터를 교대로 업데이트 한다.




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

<img width="183" alt="image-7" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/fbad45a7-ec06-4b21-9a72-cf60255f2059">



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

<img width="208" alt="image-8" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/9b6e2cc5-198c-479e-94a0-0dfaaa0b7067">

제안된 MKGAT 모델은 두 데이터셋에서 모든 베이스라인의 성능을 능가한다.


i) Effects of Modalities

<img width="238" alt="image-9" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/fbeff217-9a2f-44b8-a512-45d78f6ee4fd">

- multi-modal feature를 사용한 경우 single-modal feaure만 사용했을 때에 비해 성능이 좋다.
- 이미지가 텍스트보다 추천에 중요한 역할을 한다.
- MKGAT는 KGAT에 비해 multi-modal 정보를 item entity로 더 효과적으로 통합할 수 있다.

ii) Effect of Model Depth

<img width="242" alt="image-10" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/a6f0726d-30ec-4129-ac71-3aeeaa5f6185">


- 최적의 layer 수는 데이터셋에 따라 다르고, knowledge graph embedding 모듈에서와 추천 모듈에서 각각 다르다.

iii) Effect of Combination Layers

Aggregation layer로 Concat layer를 사용한 방법이 Add layer를 사용한 방법보다 성능이 좋다.

<img width="240" alt="image-11" src="https://github.com/parkhyeongminn/paper_review/assets/81458623/2d12ae95-0802-4d34-a5cf-197d47fa6e38">

- Concat은 여러 feature 간 차원의 확장으로, 다양한 semantic space에서 feature를 상호 작용하는 데 더 적합하다.



## 5. Conclusion

MKGAT는 multi-modal knowledge graph를 활용하여 entity 추론 및 이웃 정보 aggregation을 향상시킨 새로운 추천 모델이다. 본 논문은 추천 시스템에서 multi-modal knowledge graph의 사용을 탐구하려는 초기 시도의 연구이다. 기존 SOAT에 비해 높은 성능을 보이며, multi-modal 정보를 knowledge graph에 적용하는 것에 대한 효과성을 증명한다.

이미지나 텍스트와 같은 사이드 정보가 추천 시스템에 사용되는 연구들이 많은데, 이를 knowledge graph에 적용해 user-item graph와 결합되어 모델링하는 것이 인상 깊었다. 본 연구에서는 knowledge graph와 user-item bipartite graph를 교대로 학습시키는데, 하나의 heterogeneous graph를 구성하여 end-to-end 프레임워크로 동시에 학습하는 방법도 고려해볼 수 있을 것 같다.



## Author Information
- Rui Sun, Xuezhi Cao, Yan Zhao, Junchen Wan, Kun Zhou, Fuzheng Zhang, Zhongyuan Wang and Kai Zheng