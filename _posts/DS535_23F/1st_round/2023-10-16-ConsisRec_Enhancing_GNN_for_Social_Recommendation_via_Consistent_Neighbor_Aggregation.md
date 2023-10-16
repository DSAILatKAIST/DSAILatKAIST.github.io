---
title:  "[SIGIR 2021] ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation"
permalink: 2023-10-16-ConsisRec_Enhancing_GNN_for_Social_Recommendation_via_Consistent_Neighbor_Aggregation.html
tags: [reviews]
use_math: true
usemathjax: true
---

# ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation

## 1. Problem Definition

Previous research has shown that users' social networks significantly impact their online behavior. Consequently, recent graph neural network (GNN) recommender systems have combined the social relations with user-item interactions in order to improve the recommender performance. However, past GNN-based social recommendation models generally failed to address the so-called **inconsistency problem**. The inconsistency problem implies that social relations are not always consistent with the ratings given by users. Thus, including this inconsistent information harms the recommendation performance of the GNN.

## 2. Motivation

The paper intends to tackle the social inconsistency problem, as this could improve recommender system performance in cases where both social networks and user-item relations are available, especially if one or the other of these sources of information is deficient, thus otherwise leading to the "cold-start" problem of insufficient data for satisfactory recommender performance.

Unlike previous work in the field, the authors propose a new method to handle the inconsistency problem that they call *ConsisRec*. For each node in the GNN, their solution assigns high importance to the node's neighbors that have consistent relations when aggregating to produce the node's embedding.

## 3. Method

The authors propose a new model called *ConsisRec*. It's constructed chiefly by its *embedding layer*, *query layer*, *neighbor sampling*, and *relation attention*. The embedding layer retrieves the embedding for each node. The query layer generates a query embedding by mapping the concatenation of user and item embeddings, which can be used to select consistent neighbors for any user-item query pair. Neighbor sampling samples neighbors that are aggregated into node embeddings, in which consistent neighbors are emphasized.

## 4. Experiment

The authors use the datasets Ciao and Epinions which are used for studying the social recommendation problem. The two datasets each contain thousands of users and hundreds of thousands of items and social links. The authors removed users without social links (due to being out of the scope of social recommendation) and linked items that share more than 50% of their neighbors.

The dataset was randomly split into 60% training, 20% validation, and 20% testing. The authors performed grid search for tuning of the hyper-parameters "neighbor percent", "embedding size", "learning rate" and "batch size". Early stopping was used in all experiments, coming into effect if the RMSE on the validation did not improve for five epochs.

Six baseline methods of various types were used. These were three matrix factorization methods, one collaborative graph embedding method, and two GNN methods.

Two metrics were used to evaluate the quality of the social recommendations of the various methods: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). They were measured on the rating prediction task, and a lower value in each metric meant a better performance.

### Table 1: Performance results

![](https://hackmd.io/_uploads/SyTE14WW6.png)

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

