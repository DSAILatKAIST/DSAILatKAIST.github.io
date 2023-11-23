---
title:  "[SIGIR 2021] ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation"
permalink: 2023-11-20-ConsisRec_Enhancing_GNN_for_Social_Recommendation_via_Consistent_Neighbor_Aggregation.html
tags: [reviews]
use_ math: true
usemathjax: true
---

# ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation

## 1. Problem Definition

Previous research has shown that users' social networks significantly impact their online behavior. Consequently, recent graph neural network (GNN) recommender systems have combined the social relations with user-item interactions in order to improve the recommender performance. However, past GNN-based social recommendation models generally failed to address the so-called **inconsistency problem**. The inconsistency problem implies that social relations are not always consistent with the ratings given by users. Thus, including this inconsistent information harms the recommendation performance of the GNN.

## 2. Motivation

The paper intends to tackle the social inconsistency problem, as this could improve recommender system performance in cases where both social networks and user-item relations are available, especially if one or the other of these sources of information is deficient, thus otherwise leading to the "cold-start" problem of insufficient data for satisfactory recommender performance.

Unlike previous work in the field, the authors propose a new method to handle the inconsistency problem that they call *ConsisRec*. For each node in the GNN, their solution assigns high importance to the node's neighbors that have consistent relations when aggregating to produce the node's embedding.

## 3. Method

### Figure 1: A toy example reflecting social inconsistency and the framework of ConsisRec.
![](https://i.ibb.co/NZp9x5b/Screen-Shot-2023-11-02-at-12-28-03.png)

The authors propose a new model called *ConsisRec*. It's constructed by its *embedding layer*, *query layer*, *neighbor sampling*, and *relation attention*, as shown in figure 1(b).

The *embedding layer* retrieves the embedding for each node. Each column in the layer $E \in \mathbb{R}^{d \times (m+n)}$ represents the trainable embedding for each node. It can be indexed to retrieve a node embedding $𝑣 \in U \cup T$. The authors use index $v$ to denote a node that's either a user or an item; $u$ to denote a user specifically; $t$ to denote an item specifically; $e_ r$ to denote a relation embedding vector of relation $r$, characterizing relation-level social inonsistency.

The *query layer* exclusively selects consistent neighbors For a given query pair $(u, t)$. First, it generates a query embedding by mapping the concatenation of user and item embeddings: 

$q_ {u,t}=\sigma (W_ q^T(e_ u \oplus e_ t))$

where:
* $q_ {u,t}$ is the query embedding,
* $e_ u, e_ t \in \mathbb{R}^d$ are the embedding for node $𝑢$ and $𝑡$, respectively,
* $\oplus$ is concatenation,
* $W_ q \in \mathbb{R}^{2d \times d}$ is the mapping matrix, and
* $\sigma$ is a ReLU activation function.

The query layer samples different neighbors based on which item is interacted with, since only some friends of the user will be familiar with the query item $t$. This *neighbor sampling* is used to learn the embedding of $u$ and $t$. The node embedding aggregation can be formalized as:

$h_ v^{(l)} = \sigma(W^{(l)T}(h_ v^{(l-1)} \oplus AGG^{(l)}\{h_ i^{(l-1)}\vert i\in \mathcal{N}_ v\}))$

where:
* $\sigma$ is a ReLU activation function,
* $h_ v^{(l)} \in \mathbb{R}^d$ is the hidden embedding of node $v$ at layer $l$,
* $\mathcal{N}_ v$ is the sampled neighbors of node 𝑣,
* $AGG$ is an aggregation function,
* $W(𝑙) \in R^{2𝑑×𝑑}$ is the mapping function, and
* $h^{(0)}_ v$ is the initial node embedding of $v$, in other words, $e_ v$.

Consistent neighbors are emphasized over inconsistent neighbors in the sampling. Therefore, the consistency score between query $q$ and all the neighbors is used as the sampling probability of node $i$ at layer $l$:

$p^{(l)}(i;q) = s^{(l)}(i;q)/\sum_ {j \in \mathcal{N}_ v}s^{(l)}(j;q)$

where $s^{(l)}(i;q)$ is the consistency score between the neighbor $i$ and query $q$ GNN layer $l$. Its definition is:

$s^{(l)}(i;q)=exp(-\vert \vert q-h_ i^{(l)}\vert \vert ^2_ 2)$

where $h_ i^{(l)}$ is the node embedding of node $i$ at layer $l$. The same query embedding is used for both nodes $u$ and $t$ when infering the rating score. As such, it written simply as $q$, without a subscript. Thus, we ignore the subscript and write it as q for simplicity. More neighbors are sampled if a node is connected to more nodes.

Since there may be *relation-level* social inconsistency, different relations should be given differing amounts of attention. This is the purpose of the *relation attention* module, which learns the importance of sampled nodes based on their associated relations. Each sampled node $i$ is agiven an importance factor $\alpha_ i$, and the previouly mentioned $AGG$ is calculated using self-attention as:

$AGG^{(l)} = \sum_ {i=1}^Q \alpha_ i^{(i)} \cdot h_ i^{(l-1)}$,

where $\alpha^{(l)}_ i$ is the importance of sampled neighbor $i$ at layer $l$, and $Q$ is the total number of sampled neighbors. $\alpha_ i$ is calculated as:

$\alpha^{(l)}_ i = \frac{exp(w^T_ \alpha(h_ i^{(l-1)}\oplus e_ {r_ i}))}{\sum_ {i=1}^Qexp(w^T_ \alpha(h_ j^{(l-1)}\oplus e_ {r_ j}))}$

where:

* the relation for the edge $(v, i)$ is $r_ i$,
* $e_ {r_ i} \in \mathbb{R}^d$ is the embedding of relation $r_ i$, and
* $w_ a \in \mathbb{R}^2d$ is the trainable parameter for the self-attention layer, and
* $a_ i$ is the attention weights.

Finally, after $L$ layer propagation, the embeddings of $u$ and $t$ have been obtained. These are denoted $h_ u^{(L)}$ and $h_ t^{(L)}$. The rating score of the user-item pair $(u, t)$ is calculated by the inner product of embeddings as:

$\hat{R}_ {u,t}=h_ u^{(L)} \cdot h_ t^{(L)}$

Then, for the loss function, RMSE between $\hat{R}_ {u,y}$ and ground truth rating score $R_ {u,t}$ among all $(u,t)$ pairs in $\mathcal{E}_ {rating}$ is calculated:

$\mathcal{L} = \sqrt{\frac{\sum_ {(u,t)\in \mathcal{E}_ {rating}}(R_ {u,t}-\hat{R}_ {u,t})^2}{\vert \mathcal{E}_ {rating}\vert }}$

with $\mathcal{E}_ {rating}$ as the set of all rating edges. In order to avoid overfitting, Adam is used as the optimizer with weight decay rate of $10^{-4}$.

## 4. Experiment

The authors use the datasets Ciao and Epinions which are used for studying the social recommendation problem. The two datasets each contain thousands of users and hundreds of thousands of items and social links. The authors removed users without social links (due to being out of the scope of social recommendation) and linked items that share more than 50% of their neighbors.

The dataset was randomly split into 60% training, 20% validation, and 20% testing. The authors performed grid search for tuning of the hyper-parameters "neighbor percent", "embedding size", "learning rate" and "batch size". Early stopping was used in all experiments, coming into effect if the RMSE on the validation did not improve for five epochs.

Six baseline methods of various types were used. These were three matrix factorization methods, one collaborative graph embedding method, and two GNN methods.

Two metrics were used to evaluate the quality of the social recommendations of the various methods: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). They were measured on the rating prediction task, and a lower value in each metric meant a better performance.

### Table 1: Performance results

![](https://i.ibb.co/tHBpR67/Screen-Shot-2023-10-09-at-16-41-33.png)

This table includes the performance of the ConsisRec method, as well as the six baselines. The best results are in bold, and the second-best results are underlined. Measured in both MAE and RMSE and using either dataset, ConsisRec outperforms all other measured models. On average, it has a 1.7% relative improvement compared to the second-best in every examined case.

## 5. Conclusion

The "cold-start" problem is a well-known one in the field of recommender systems, where there is an insufficient amount of data about a user or item to use for recommendations. In order to alleviate this issue, alternative sources of information have been utilized, one of which is social network interactions. However, this has led to the "social inconsistency problem", which suggests that social links may not consistently correlate with ratings. The authors of the paper seek to solve this problem using a novel method they call "ConsisRec" which includes a new process for embedding nodes in a graph neural network recommender system that promotes "consistent" node neighbors as a source for the node's embedding. Using two representative datasets for the rating prediction task, the new method outperforms all existing models by an average of 1.7% relative to the second-best performer, as measured in mean absolute error and root-mean-square error.

## Author Information
Author: Hugo Dettner Källander  
Affiliation: School of Computing at KAIST  
Research Topic: Recommendation systems and graph machine learning

## 6. Reference & Additional materials

Github Implementation: https://github.com/YangLiangwei/ConsisRec  
Reference:
Liangwei Yang, Zhiwei Liu, Yingtong Dou, Jing Ma, and Philip S. Yu. 2021. ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3404835.3463028