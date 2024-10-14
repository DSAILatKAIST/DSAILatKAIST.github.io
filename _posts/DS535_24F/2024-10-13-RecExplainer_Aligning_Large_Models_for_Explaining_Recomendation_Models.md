---
title:  "[KDD 2024] RecExplainer: Aligning Large Language Models for Explaining Recommendation Models"
permalink: 2024-10-13-RecExplainer_Aligning_Large_Models_for_Explaining_Recomendation_Models.html
tags: [reviews]
use_math: true
usemathjax: true
---


## **1. Motivation**
추천 시스템이란 사용자의 과거 이력을 토대로 성향을 분석하여 소비할 확률이 높은 상품을 추천하는 모델이며, 영화, 음악, 쇼핑 등 우리 생활에 깊게 녹아들어 있다.<br>
이를 위해 임베딩 기반의 collaborative filtering, graph neural network 등의  black-box 모델들이 연구되어 왔다.<br>
하지만 이러한 black-box 모델들은 신뢰있고 이해되는 결과가 필요한 추천 시스템의 특성과는 달리 출력된 결과에 대한 설명을 제공하기 어렵다는 단점이 있다.<br>
기존 연구들은 이를 위해 sparse linear model, decision tree 등의 별도의 surrogate 모델을 두어 이를 해결하려 했지만, <br>
그것들의 단순한 구조로 인해 사람이 이해하기 쉬운 결과들이 나오기 어려웠다. <br>

최근들어 large language model (LLM) 이 등장하고 이것의 논리 추론 능력이 연구됨에 따라 다양한 AI 분야에서 활용되기 시작했다.<br>
그 중에 LLM 의 뛰어난 언어 능력에 접목하여 설명 가능한 인공지능 (XAI) 으로써의 활용 또한 연구가 되고 있다.<br>
해당 논문은 이에 주목하여 LLM 을 surrogate 모델로 두어 설명 가능한 추천 시스템인 RecExplainer 을 제안한다.<br>
즉 LLM 은 추천 자체에 쓰이기 보다는, 학습된 별도의 추천 시스템 모델과의 aligning 을 통해 <br>
해당 모델들의 출력한 결과에 대한 설명을 제공하는 중간자 역할로써 활용된다. <br>

## **2. Problem Formulation**
**Figure 1** <br>
![Figure1](https://github.com/user-attachments/assets/2fa1106c-3f37-4b25-973a-969fb3bc4935)<br>
사용자의 과거 이력을 $x_u = \langle a_1, a_2, \ldots, a_{|x_u|} \rangle, a_* :=$ 소비한 아이템, 로 표현하면<br>
추천 모델 $f()$ 은 후에 소비할 확률이 높은 아이템과 낮은 아이템 $a_i, a_j$ 에 대해 f(x_u, a_i) > f(x_u, a_j)$ 의 값을 부여하는 것을 목표로 한다.<br>
이때 임베딩 기반의 모델들은 $e_u = \text{encoder}_\text{user}(x_u), \space e_i = \text{encoder}_\text{item}(a_i)$ 두 개의 임베딩을 구하고,<br>
이들 간의 similarity 를 기반으로 선호도 점수를 예측한다.<br>
RecExplainer 는 이미 학습이 완료된 $f()$ 가 주어졌을 때, 이것과 잘 aligning 되도록 LLM $g()$ 을 fine-tuning 하여 $f()$ 의 출력을 $g()$ 로 설명하도록 한다.<br>
Fig. 1 에서 나타나 있듰이, 이때 $f()$ 은 frozen 이며 dimension projection 을 위한 MLP 와 $g()$ 만을 학습하며,<br>
$g()$ 은 전체 가중치가 아닌 parameter-efficient 하게 LoRA 를 활용한다.<br>

### **Tasks**
저자들은 LLM $g()$ 을 fine-tuning 할 여섯 가지의 task 를 정의한다.<br>
1. Next item retrieval: 유저의 기록이 주어졌을 때 다음에 이용할 아이템을 찾는 작업이다.
2. Item ranking: 아이템 목록이 주어졌을 때 유저의 선호도 순위를 매기는 작업이다.
3. Interest classification: 유저의 사용 기록이 주어졌을 때 특정 아이템의 선호 여부를 예측하는 작업이다.
4. Item discrimination: 아이템 제목이 주어졌을 때 설명을 생성하고 유사한 아이템을 찾는 작업이다. <br>
이는 활용할 LLM 의 pretraining 시에 접하지 못했던 아이템들에 대한 이해력을 얻게 해준다.
5. ShareGPT training: LLM fine-tuning 시에 언어적 능력을 상실하는 catastrophic forgetting 을 방지하기 위한 작업이다.<br>
사람과 ChatGPT 간의 대화가 저장된 ShareGPT API 를 활용해 $g()$ 의 어휘 역량을 잃지 않도록 한다.
6. History reconstruction: 유저의 임베딩이 주어졌을 때 해당 유저의 아이템 사용 기록을 복원하는 작업이다.<br>
이를 통해 입력된 유저에 대한 정보를 얼마나 추출할 수 있는지 확인할 수 있다.<br>

이때 추천과 연관된 task 들의 경우, 학습 데이터의 label 은 ground truth 값이 아닌 $f()$ 이 생성한 결과가 된다.<br>
그 이유는 $g()$ 가 $f()$ 가 내놓은 결과에 대한 설명을 제공하기를 바라는 것이지, 추천 자체를 하기를 목표하는 것이 아니기 때문이다.<br>

## **3. Methodology**
RecExplainer 는 fine-tuning 할 $g()$ 와 pretrained 추천 모델 $f()$ 간의 aligning 을 아래 세 가지 방법을 통해 진행하며, <br>
각각의 방법론을 적용한 모델을 순서대로 RecExplainer-B, RecExplainer-I, RecExplainer-H 라 표기한다. <br>
### **Behavior Alignment**
Behavior alignment 는 teacher-student architecture 에서 knowledge distillation 과 어느 정도 유사한 방법론으로, <br>
$g()$ 의 prediction 이 $f()$ 의 prediction 과 일치하면 그것의 생성 로직 또한 모방할 수 있고 결국 설명 능력도 생길 것이라고 가정한다.<br>
이때 $g()$ 의 학습 과정에선 임베딩을 활용하지 않으므로 학습은 task 1~5 로만 진행한다. <br>
활용된 prompt 의 예시는 아래와 같다. <br>
"Given a user with history: ($item_1$, $item_2$, ...), what item will you recommend to the user and why?"<br>
### **Intention Alignment**
하지만 Behavior alignment 는 $f()$ 가 실제로 어떻게 입력을 인지하는 지보단 prediction 결과에만 의존한다는 단점이 있다.<br>
Intention alignment 는 현재 여러 Multi-modal Language Model (MLM) 에서 활용하는 방법론을 차용하여, <br> 
user 와 item embedding 을 별도의 modality 로 취급해 각각이 LLM 의 language space 에 align 되도록 하는 방법론이다. <br>
이를 통해 $g()$ 가 $f()$ 로부터 생성된 임베딩 공간을 학습하게 되어 설명 능력을 갖게 될 것이라 저자들은 주장한다.<br>
학습은 task 1~6 모두 진행하며, Fig. 1 처럼 필요한 projecting 된 임베딩 벡터를 아이템의 이름대신 프롬프트에 추가하여 답을 얻는다.<br>
활용된 prompt 의 예시는 아래와 같다. <br>
"Given a user: (user_embedding), what item will you recommend to the user and why?"<br>
### **Hybrid Alignment**
Hybrid alignment 는 behavior alignment 와 intention alignment 을 합친 것으로, 아이템의 제목과 임베딩 벡터 모두 프롬프트에 추가한 방법론이다.<br>
따라서 활용된 prompt 의 예시는 아래와 같다.<br>
"Given a user with history: (user_embedding), ($item_1$, $item_2$, ...), what item will you recommend to the user and why?"<br>

## **4. Experiments**
### **Metrics**
RecExplainer 는 실험의 평가 지표로 두 가지를 제시한다. <br>
우선 $f()$ 와 $g()$ 가 얼마나 잘 aligning 됐는지 평가하는 수치를 제공하며, 이는 기존 추천 시스템에서 널리 활용되는 leave-one-out strategy 를 차용한다.<br>
이는 사용자들의 interaction 기록에서 마지막 item 을 제거하고 복원하는 task 로, 얼마나 그 사용자의 특성을 잘 파악했는지 알 수 있는 방법이다.<br>
하지만 RecExplainer 에서 LLM 은 추천 모델이 아닌 설명 모델로 활용되므로, ground truth 가 아닌 제공된 추천 모델 $f()$ 의 출력을 label 로 지정한다.<br>
이를 토대로 $g()$ 가 얼마나 $f()$ 를 잘 모방하였는지 확인할 수 있다.<br>
**Figure 2** <br>
![Figure2](https://github.com/user-attachments/assets/0cd70216-3f43-4d5b-b4dc-9b6cc20ea4b7)<br>
또한 Explainable AI 를 목표로 하기 때문에 제공한 설명의 타당성에 대한 평가도 제공한다.<br>
하지만 "설명" 이라는 것의 특성상 유일한 정답이 존재하지 않으며, 일일이 사람이 hand-labeling 하는 것은 막대한 비용을 초래한다.<br>
따라서 저자들은 human-labeling 뿐만 아니라 추론 능력이 뛰어난 GPT-4 모델을 활용하는데, <br>
Fig. 2 와 같은 프롬프트를 통해 $g()$ 가 생성한 설명의 타당성을 네 개의 criteria 로 분류한다.<br>
1. Rating-0: 틀린 답안
2. Rating-1: 올바른 답안, 부족한 설명
3. Rating-2: 올바른 답안, 수용할 만한 설명
4. Rating-3: 올바른 답안, 적절한 설명

이때 답안의 정확도는 RecExplainer 가 제공된 설명이 $f()$ 의 출력과 일맥상통한지 판별하는 것으로, $g()$ 가 정말 $f()$ 의 결과를 토대로 설명을 제공했는 지 평가한다.<br>
아쉬운 점은 저자들이 각 rating 에 대한 예시없이 최종 수치 분포만 제공하여 직관적인 이해에 어려움이 있었다.<br>
### **Experimental Setup**
**Datasets**
실험은 아래 세 개의 public dataset 을 활용하여 진행한다.
- Video Games
- Movies and TV
- Steam
  
각 dataset 에 대한 정보와 train/test split 은 다음과 같다.<br>
![Figure3](https://github.com/user-attachments/assets/20e0f0cc-4ed9-468f-a4df-f919c85a6174)<br>
![Figure4](https://github.com/user-attachments/assets/9172860b-086e-4091-9530-9fbe4089fd87)<br>
**Implementation details**
RecExplainer 은 Vicuna-v1.3-7b 을 backbone LLM 으로 활용하며, DeepSpeed 의 ZeRO-2 를 통한 분산 학습을 진행했다.<br>
또한 설명의 대상이 되는 추천 모델로는 트랜스포머 기반인 SASRec 을 차용하였고, 각 dataset 에 알맞게 fine-tuning 을 추가적으로 진행하였다.<br>
**Baselines**
위에서 정의한 두 가지 평가 지표에 대한 baseline 모델들은 다음과 같다.<br>
우선 alignment effect 에 대해선,
- Random: k 개의 아이템을 uniform distribution 에서 추출하고 random shuffling 으로 우선순위를 지정한다.
- Popularity: k 개의 아이템을 popularity distribution 에서 추출하고 인기순으로 우선순위를 지정한다.
- Vicuna-v.1.3-7b: Aligning 을 거치지 않은 Vicuna 를 그대로 활용한다.
- Vicuna-v.1.3-7b-ICL: Aligning 을 거치지 않은 Vicuna 를 2-shot prompting 하여 활용한다.
- GPT4-ICL: GPT4 에 2-shot prompting 하여 활용한다.
- SASRec: Pretrained SASRec $f()$ 을 통해 knowledge distillation 으로 학습시켜 활용한다.

Explainability 에 대해선,
- Vicuna-v.1.3-7b: Aligning 을 거치지 않은 Vicuna 를 그대로 활용한다.
- ChatGPT: OpenAI API 를 활용한다.
  
### **Results**
**Alignment Effect**<br>
![Figure5](https://github.com/user-attachments/assets/4d837990-c45b-470d-98eb-b8b6dca16c6a)<br>
- H@5: 5개의 추천 목록에 올바른 품목들이 들어갔는가
- $N@5: 5개의 추천 목록이 얼마나 잘 배치됐는가
- $ACC: 예측한 것이 얼마나 정확한가
- $HCR: Label sequence 과 prediction 이 얼마나 겹치는가

위 테이블은 Alignment 분야에서의 성능 평가를 나타낸 표다.<br>
이때 세 가지 alignment variation 모두 별도의 aligning 없이 LLM 만을 활용한 실험들과 비교해서 <br>
월등한 성능 차이가 나타나는 것을 통해 RecExplainer 에서 alignment 의 중요성을 확인할 수 있다. <br>
또한 prediction 결과만을 통해 모방하는 RecExplainer-B 보단 좀 더 포괄적인 이해를 목표하는 RecExplainer-I/H 가 더 좋은 성능을 기록한 것을 확인할 수 있다.<br>

**Explainability**<br>
![Figure6](https://github.com/user-attachments/assets/953be1d0-9b0a-4ef5-b153-6af7759d8358)<br>
![Figure7](https://github.com/user-attachments/assets/209aec56-c377-42c1-bc01-6f11d5bf71b0)<br>
위 표와 Figure 는 순서대로 GPT-4 와 Human expert 가 평가한 생성된 설명에 대한 rating 의 평균과 분포다.<br>
이를 통해 RecExplainer-H 가 다른 모델들에 비해 수준 높은 설명을 생성하는 것을 확인할 수 있다.
저자들은 기존 LLM 들은 아이템들에 대한 학습이 이뤄지지 않았기 때문에 대부분의 설명들이 모호하고 평이해 Rating-2 에 분포하게 된다고 주장한다.<br>
주목할 만한 점은 RecExplainer-I 의 성능이 매우 낮게 나왔다는 것인데, 저자들은 분석 결과 이것이 생성한 답변들에 hallucination 현상이 다반사로 발생하는 것을 확인하였다.<br>
그들은 HCR 점수가 낮은 것을 종합하였을 때, 임베딩만 주어졌을 때 문자 형태의 정보를 복원하는 능력이 부족해 이런 현상이 발생한 것이라 추측한다.<br>
또한 저자들은 RecExplainer 가 생성한 설명들이 기존의 모델들의 것들과 명확히 다른지, 즉 설명 능력이 정말 fine-tuning 의 영향을 받는지 보여주는 실험 결과를 제공한다.<br>
이는 Vicuna-7b, ChatGPT, RecExplainer-H 각각으로 2500 개의 설명을 생성하고, 하나의 discriminator 을 둬서 각 2000 개로 생성한 답변이 어느 모델로부터 생성했는지 분류하게 학습한다.<br>
그 후 남은 500개로 다시 분류하는데, 아래의 결과를 통해 각 모델들의 설명들 모두 구분됨을 확인할 수 있고, 이를 토대로 저자들은 별도의 학습이 설명 생성에 영향을 주었다고 말한다.<br>
![Figure10](https://github.com/user-attachments/assets/76c2b787-f854-44e7-be05-29bff868e803)<br>
**Case Study**<br>
아래는 실제 설명 생성의 예시 사례를 보여주며, 올바른 설명은 초록색으로, 틀린 설명은 빨간색으로 표기하였다.<br>
![Figure8](https://github.com/user-attachments/assets/b4716f09-b146-4008-b3b1-62b8cc3174df)<br>
![Figure9](https://github.com/user-attachments/assets/3a290ed9-8337-4136-8f6c-da6f478e2596)<br>
또한 LLM 의 통한 설명의 장점은 원하는 시야에서 설명을 생성할 수 있다는 것이 있다.<br>
저자들은 RecExplainer-H 를 통해 두 가지 측면에서 생성한 설명의 예시를 다음과 같이 제공한다.<br>
![Figure11](https://github.com/user-attachments/assets/3de1694d-3169-4310-9245-71c65e3b3326)<br>
![Figure12](https://github.com/user-attachments/assets/79ad6aa0-7e6a-4b5d-91d4-a5e18c195c81)<br>

## **5. Conclusion**
이 논문은 LLM 을 적절한 alignment 를 통해 기존의 black-box 추천 모델들을 설명할 수 있는 방법론을 제안한다.<br>
이와 같이 LLM 의 성능이 나날히 증가하며 다양한 분야에서 zero-shot 능력이 입증됨에 따라 이를 활용한 추천 시스템 연구들이 많이 진행되고 있다. <br>
그 중에서 위 논문과 비슷하게 LLM 과 collaborative model 의 aligning 을 통해 두 모델의 장점을 모두 취하여 추천하는 연구 [1] 또한 KDD 2024 에 발표됐다. <br>
반면 RecExplainer 는 LLM 의 추천 능력에 집중하기 보다는 설명 능력에 중점을 둔 것으로, 성능 평가 또한 ground truth 가 아닌 target model 의 output 을 기준으로 진행된다.<br>
두 논문을 통해 방법론이 비슷하더라도 다른 motivation 을 설정하여 문제를 해결할 수 있음을 확인할 수 있어 흥미로웠다.<br>

## **6. Author Information**  
* Author name: 김현철
    * Affiliation: [DSAIL@KAIST](https://dsail.kaist.ac.kr/)
    * Research Topic: Self-supervised learning, Time series
    * Contact: khchul@kaist.ac.kr

## **7. Code and References**
### **Code**
https://github.com/microsoft/RecAI/tree/main/RecExplainer<br>

### **References**
[1] Kim, S., Kang, H., Choi, S., Kim, D., Yang, M., & Park, C. (2024, August). Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 1395-1406).<br>
