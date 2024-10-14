---
title:  "[ACL-24] Tree-of-Traversals: A Zero-Shot Reasoning Algorithm for Augmenting Black-box Language Models with Knowledge Graphs"
permalink: 2024-10-13-Tree-of-Traversals_A_Zero-Shot_Reasoning_Algorithm_for_Augmenting_Black-box_Language_Models_with_Knowledge_Graphs.html
tags: [reviews]
use_math: true
usemathjax: true
---


## 1. Motivation
Large Language Model(LLM)은 information retrieval, text summarization, question answering과 같이 배경지식을 요구하는 task에서 일반적으로 좋은 성능을 보인다. 하지만 몇 가지 한계가 있는데, 거짓된 정보를 제공하거나 특정 분야에 대한 깊은 지식의 부재, 학습이 끝나면 지식이 갱신되지 않는다는 한계점이 있다. 이를 해결하기 위해 Knowledge Graph(KG)를 이용해 LLM을 보완하려는 시도가 있었다. KG는 최신 지식을 갖고 있고, 특정 분야에 대한 깊은 지식을 갖고 있기도 하며 지식이 잘 구조화되어 있다는 장점이 있다.

KG로 LLM을 보완하는 방법은 크게 세 가지로 분류할 수 있다. pre-training 단계에서 KG를 입력하는 방법, fine-tuning 단계에서 KG를 입력하는 방법, pre-trained LLM을 그대로 두고 KG를 입력으로 하는 추가적인 모듈을 학습하는 방법이 있다. 최근의 LLM은 parameter의 개수가 매우 많기 때문에 pre-training과 fine-tuning에 매우 많은 연산이 필요하고, 심지어 모델 구조나 parameter 자체가 공개되지 않고 API를 통해서만 접근이 가능한 black-box LLM이 많다. 또한 크기가 매우 큰 KG는 메모리로 가져오기 어려워 서버에 API를 통해서만 접근이 가능하다. 추가적으로 기존의 KG-augmented LLM 모델은 여러 개의 KG를 사용할 수 없다는 한계가 있다.

본 논문은 Tree-of-Traversals 알고리즘을 제안하는데, 학습 없이(zero-shot) 여러 개의 KG를 이용해 LLM을 보완할 수 있는 알고리즘이다.


## 2. Method
Tree-of-Traversals 알고리즘은 question anwering task에서 정확한 답변을 생성하기 위해 KG subgraph의 정보를 활용한다. 이때 KG의 subgraph를 점차 확장하는데, LLM이 답변에 필요한 정보가 subgraph에 모두 포함될 때까지 확장이 이루어진다. 우선 question(query)에 존재하는 entity를 포함하여 subgraph를 initialize하고, LLM의 판단에 따라 action을 수행하며 subgraph에 entity와 relation을 추가한다. 이 과정은 크게 세 가지 구성 요소로 나누어 작동하는데, knowledge graph interface, action state machine, tree search algorithm으로 나눌 수 있다.

### 2.1 Knowledge Graph Interface
Knowledge graph interface는 KG에서 필요한 정보를 추출하기 위해 실행된다. KG를 $K=(E, R, T)$로 표현하면 $E$는 entity set, $R$은 relation type set, $T$는 edge(fact) set을 나타낸다. 각각의 entity는 identifier, label, optional description으로 구성되고, relation type은 identifier, label, optional inverse label로 구성되며 edge는 $(s, r, o)$와 같이 두 개의 entity $s, o$와 하나의 relation type $r$로 구성된다. 하나의 예시로, (Q35332, *'Christopher Nolan'*, *'British-American filmmaker'*)는 entity에 해당하고, (P57, *'director'*, *'is director of'*)는 relation type에 해당한다.
KG에서 필요한 정보를 추출할 때는 다음과 같이 세 가지의 명령어를 사용한다.

1. $initialize(q)$
2. $get\_relations(E_{selected})$
3. $get\_edges(E_{selected}, r)$

$initialize(q)$는 query $q$가 입력으로 주어졌을 때, KG $K$에서 $q$에 포함된 entity를 추출하여 subgraph를 initialize하는 함수이다. 예를 들어 "What actor played in both Inception and Interstellar?"라는 query가 주어졌을 때, label이 'Inception'인 entity와 'Interstellar'인 entity를 추출하여 entity가 두 개인 subgraph를 생성한다.

다음으로 $get\_relations(E_{selected})$는 entity set $E_{selected}$가 주어졌을 때, 해당 entity와 관련된 relation type을 추출하여 relation set $R_{options}$를 출력하는 함수이다. 수학적으로 $R_{options}=\{r\mid(s, r, o)\in T, s\in E_{selected}\}$와 같이 표현할 수 있다. 예를 들어 $E_{selected}$으로 {'Inception', 'Interstellar'}가 주어졌을 때, {'cast', 'director'}가 출력된다.

마지막으로 $get\_edges(E_{selected}, r)$는 entity set $E_{selected}$와 relation type $r$이 주어졌을 때, $E_{selected}$에서 한 entity를 source entity로 갖고  $r$을 relation type으로 갖는 edge를 추출하여 edge set $T_{added}$ 출력하는 함수이다. 수학적으로 $T_{added}=\{(s,r,o)\in T\mid s\in E_{selected}, r=r, o\in E\}$으로 표현된다.

### 2.2 Action State Machine (ASM)
LLM의 답변 과정에 KG의 지식을 고려하기 위해 과거에는 KG를 prompt에 함께 제공하는 in-context learning 기법을 시도했다. 하지만 LLM의 token 제한 때문에 크기가 큰 KG를 모두 prompt에 제공하기는 어려웠고, LLM이 KG에서 필요한 정보를 추출하는 데 어려움이 있었다. 이 문제를 해결하기 위해 본 논문에서는 KG subgraph를 추출하는데, 이때 단계별로 확장을 수행하고 각 단계마다 LLM의 판단을 거치며 KG를 점차 확장해나간다.

KG의 확장은 finite state machine을 따라 이루어지고, 5개의 action *Think, Answer, ExpandKG, Select_Entities, Select_relation*과 4개의 state *default, selecting-entities, selecting-relation, done*으로 구성된다. 이 finite state machine을 Action State Machine (ASM)이라 칭한다.

ASM을 그림으로 나타내면 다음과 같다. 주어진 query $q$에 대해서 $initialize(q)$를 수행하면 *default* state에서 시작한다. 각 state마다 취할 수 있는 action이 존재하고 LLM의 판단에 따라 하나의 action을 취한다. 이때 state마다 정해진 양식의 prompt를 작성하여 LLM에 입력하면 LLM이 취할 수 있는 action을 출력한다. 예를 들어, *default* state에서는 3개의 action *Think, Answer, ExpandKG* 를 취할 수 있다. 만약 Tree-of-Traversals 알고리즘이 *ExpandKG*를 취하도록 선택했으면 그 다음에 *selecting-entities* state으로 이동하여 *Select_Entities* action을 취할 수 있다. *selecting-entities* state에서는 한 종류의 action만 취할 수 있는데, 어떤 entity set을 선택하는지에 따라 서로 다른 action으로 간주된다.

![ASM](https://i.postimg.cc/hPSGwLgH/ASM.png)

### 2.3 Tree Search Algorithm
Tree-of-Traversals의 트리 탐색 알고리즘은 Tree-of-Thoughts의 접근법을 차용한다. Tree-of-Thoughts는 LLM이 주어진 query 질문에 대한 답변을 생성할 때 바로 답변하는 것이 아니라 중간 thought를 생성하는 과정을 거친 후에 최종 답변을 생성하는 알고리즘이다. 이때 다양하게 뻗어나가는 thought를 생성하여 search tree의 형태로 나타낼 수 있는데, 트리 탐색을 통해 search tree의 노드 중에서 가장 높은 점수의 노드가 답변에 사용된다. Tree-of-Traversals는 이와 유사하게 KG subgraph의 다양한 확장 결과를 search tree로 나타낸다. 최종적으로 search tree의 가장 높은 점수의 노드에 포함된 KG subgraph가 LLM의 답변에 사용되는 것이다.

![Algorithm](https://i.postimg.cc/Bb3vJDFP/tree-of-traversals-algorithm.png)

Algorithm 1은 트리 탐색 알고리즘을 자세히 나타낸 것이다. 먼저 query $q$에 대해서 $initialize(q)$를 수행하면 *default* state의 루트 노드가 생성된다. 루트 노드에 대해서 k개의 action을 샘플링하면 각각의 action을 수행한 결과를 자식 노드로 추가하게 된다. 루트 노드를 한 번 방문한 후에는 다시 방문하지 않고, 방문하지 않은 노드 중에서 방문할 노드를 선택한다. 이때 점수가 가장 높은 노드, 점수가 동일할 경우에는 depth가 큰 노드를 선택한다. 선택한 노드에 대해서 마찬가지로 k개의 action을 샘플링하고 자식 노드를 추가한다. 이 과정을 반복하여 search tree를 확장하고, *Answer* state 노드의 점수가 특정 threshold를 넘으면 해당 노드의 답변이 최종 답변으로 사용된다. 또한 search tree가 무한히 확장하는 것을 방지하기 위해 *max depth*, *max expansions*의 두 가지 stopping criteria를 설정한다. *max depth*에 도달하면 알고리즘이 강제로 *done* state로 이동하여 답변을 생성하고, *max expansions*에 도달하면 모델이 "Cannot find answer"을 답변으로 내놓는다.

Tree-of-Traversals 알고리즘에서 search tree 상에 존재하는 노드의 점수를 산정하기 위해서 LLM을 사용한다. LLM에 특정 양식에 맞춰 prompt를 작성하는데, prompt에는 현재의 원래 query, KG subgraph, 이전에 수행한 action 목록과 점수 산정을 위한 instruction이 포함된다. 점수를 산정할 때 일반 노드와 *Answer* state 노드에 서로 다른 prompt를 사용한다.

Chain-of-Traversals는 Tree-of-Traversals의 일종으로 k개의 action을 샘플링할 때 k가 1인 경우에 해당한다. 따라서 방문하지 않은 노드가 하나만 존재하게 되고, 노드 중에서 점수에 따라 선택하는 대신에 확정적으로 노드를 방문한다. 이후의 실험 결과에서 Chain-of-Traversals와 Tree-of-Traversals의 question answering 성능을 비교한다.

### 2.4 Tree-of-Traversals with Multiple KGs
Tree-of-Traversals 알고리즘에서 여러 개의 KG를 고려하기 위해 우선 각각의 KG에 KG interface를 만든다. 또한 알고리즘에 몇 가지 변화가 있다. 각각의 KG $initialize(q)$ 함수로부터 중복으로 동일한 entity가 추출되지 않도록 하고, $get\_relations(E_{selected})$ 함수에서 모든 KG로부터 해당하는 relation이 출력되도록 하며 $get\_edges(E_{selected}, r)$ 함수를 통해 새로운 노드가 추가될 때 기존 노드와 새로운 노드 간의 edge가 존재하는지 모든 KG에 대해 확인하도록 한다.

## 3. Experiments
Tree-of-Traversals 알고리즘에서 사용할 LLM으로는 Amazon Bedrock에서 API를 통해 이용할 수 있는 Claude-Instant, Llama2 70b, Llama2 13b를 사용한다. 2WikiMultiHop 데이터셋과 QALD-10 데이터셋에서 question answering task를 수행한다. 이때 metric으로 Exact Match Included (EM-in)을 사용하는데, 생성한 답변에 정답 답변이 모두 포함되어 있으면 1, 그렇지 않다면 0이다. EM-in은 EM과 다르게 생성한 답변과 정답 답변이 완전히 일치하지 않아도 되는데, LLM이 응답을 생성할 때 다양한 문장 구조와 구문을 사용할 수 있다는 점과 일반적으로 LLM의 답변에 정답 이외에 부가 설명이 추가된다는 점을 고려할 때 더 적합한 metric이다.

Tree-of-Traversals 알고리즘의 성능과 비교하기 위한 baseline으로는 Chain-of-Thought (CoT) prompting, ReAct, FLARe와 같이 black-box LLM에서 동작하는 알고리즘을 사용한다. Tree-of-Traversals 알고리즘을 적용한 모델은 두 가지 사용하는데, *branching factor* k=1인 특수한 경우를 Chain-of-Traversals, k=3인 경우를 Tree-of-Traversals라고 칭한다. 알고리즘을 수행할 때 search tree가 무한히 확장하는 걸 방지하기 위한 hyperparameter로서 *max depth*는 7, *max expansions*는 20으로 설정한다. 또한 최종 답변을 결정하기 위한 threshold를 0.8로 설정하여 실험을 진행한다.

![Result](https://i.postimg.cc/pdnV8kYn/experiment-result.png)

위 테이블은 2WikiMultiHop 데이터셋과 QALD-10 데이터셋에서 baseline과 Tree-of-Traversals 알고리즘의 성능을 나타낸 것이다. 먼저 2WikiMultiHop의 경우, 사용한 세 가지 LLM에서 모두 Tree-of-Traversals의 성능이 가장 높게 나타났고, 이 성능은 zero-shot 세팅에서 state-of-the-art에 해당한다. Chain-of-Traversals도 baseline보다 좋은 성능을 보인다는 점에서 ASM을 따라 action을 선택하며 KG subgraph를 추출하는 방식이 효과적이라는 점을 확인할 수 있다. Tree-of-Traversals는 Chain-of-Traversals와 달리 확정적으로 하나의 action을 선택하는 것이 아니라, 가능한 다양한 action 중에서 점수가 높은 action을 선택하여 subgraph를 확장하는 방식으로 최적의 KG subgraph를 추출하기 때문에 두 데이터셋에서 모두 Chain-of-Traversals보다 확연히 좋은 성능을 보였다. QALD-10 데이터셋의 경우에도 Llama2-13b를 사용한 경우를 제외하면 Tree-of-Traversals와 Chain-of-Traversals가 baseline보다 좋은 성능을 보였다.

또한 한 가지 확인할 수 있는 것은 LLM 모델에 따라서도 question anwering 성능에 차이가 발생했다. 이는 Llama-70b와 Llama-13b의 차이에서 확인 가능한데, Tree-of-Traversals 알고리즘의 성능은 LLM의 성능에 비례한다는 점을 알 수 있다.

## 4. Conclusion
Tree-of-Traversals는 LLM이 question answering과 같이 지식을 기반으로 하는 task에서 한계가 있다는 점을 해결하기 위해 고안된 알고리즘으로, 별개의 학습 없이 LLM이 KG를 기반으로 더 정확한 답변을 생성할 수 있도록 한다. 기존의 KG-augmented LLM은 LLM의 pre-training이나 fine-tuning과 같이 LLM의 parameter를 직접적으로 학습하는 경우가 많았는데, 연산량이 매우 많을 뿐만 아니라 많은 LLM이 모델 구조와 parameter를 공개하지 않는 black-box 형태이기 때문에, 학습이 없고 LLM의 API만을 사용하여 구현 가능한 Tree-of-Traversals 알고리즘이 유용하다. 질문 query와 관련된 정보를 KG에서 추출하는 과정에서 Tree-of-Thoughts의 접근법을 차용한 알고리즘으로, 선택 가능한 다양한 KG subgraph 중에서 가장 유용한 subgraph를 선택하여 정확한 답변을 생성할 수 있다. 실제로 다양한 실험에서 기존 baseline 모델보다 좋은 성능을 나타냄으로써 그 효과를 입증했다.

Tree-of-Traversals 알고리즘은 LLM의 성능에 비례하기 때문에 동일한 알고리즘을 사용하더라도 LLM의 성능이 좋아지면 question answering task의 성능을 향상시킬 수 있기에, 지금과 같이 LLM의 성능이 빠르게 좋아지는 상황에서 유용할 것이라고 생각한다. 특히 LLM의 학습이 필요하지 않기 때문에 동일한 API를 사용하여 LLM에 접근할 수 있다면 추가적인 작업 없이 바로 새로운 LLM을 사용할 수 있다. 또한 민감한 정보가 담긴 KG에 이용할 때에도, LLM을 학습하는 과정이 없어 보안 측면에서 안전하다는 점이 장점이다. 

그러나 LLM의 성능에 비례한다는 점이 장점이자 단점이 될 수 있는데, 좋은 성능의 LLM에 접근하기 어려운 상황이라면 task의 성능이 떨어지게 된다. 실제로 실험 결과에서 비교적 작은 크기의 LLM인 Llama-13b를 사용했을 때, 몇 개의 baseline보다도 낮은 성능을 보였다는 점에서 확인할 수 있다. 또한 KG subgraph를 추출하는 과정에서 search tree를 만드는데, search tree의 모든 노드가 LLM의 답변을 필요로 한다는 점을 고려할 때 하나의 답변을 생성하기 위해 LLM을 많이 사용하게 된다. *max expansions*를 설정함으로써 과도하게 사용하는 것을 막기는 하지만, 그럼에도 한 번의 LLM 사용으로 답변이 생성되는 기존의 방법과 달리 수 십번의 사용이 필요하고 이에 따라 답변 생성 시간이 크게 늘어날 수밖에 없다. 뿐더러 LLM의 답변 생성에 많은 연산량이 필요하기 때문에 사용량을 제한하는 경우가 많은데, 상당히 많은 사용량을 확보하거나 로컬에 환경을 구축해야만 이 알고리즘을 적용할 수 있다. 또한 KG subgraph를 initialize할 때 query에 등장하는 entity만을 사용하는데, 질문의 맥락에 따라 그 entity만으로는 정확한 답변을 생성하기 어려운 상황이 존재할 수 있다는 점도 단점이라고 생각한다.

## 5. References
- Github Implementation
	- https://github.com/amazon-science/tree-of-traversals
- Markowitz, Elan, et al. "Tree-of-Traversals: A Zero-Shot Reasoning Algorithm for Augmenting Black-box Language Models with Knowledge Graphs." _arXiv preprint arXiv:2407.21358_ (2024).
- Yao, Shunyu, et al. "Tree of thoughts: Deliberate problem solving with large language models." _Advances in Neural Information Processing Systems_ 36 (2024).
