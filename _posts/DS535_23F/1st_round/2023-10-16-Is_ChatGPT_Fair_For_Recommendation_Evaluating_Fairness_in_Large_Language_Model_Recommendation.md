---
title:  "[RecSys 2023] Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation"
permalink: 2023-10-16-Is_ChatGPT_Fair_For_Recommendation_Evaluating_Fairness_in_Large_Language_Model_Recommendation.html
tags: [reviews]
use_math: true
usemathjax: true
---
# A Review on FaiRLLM 

### SUMMARY
This is a scientific review on the paper "Is Chat GPT fair for Recommendation? Evaluating Fairness in Large Language
Model Recommendation" by Jizhi Zhang, Keqin Bao, Yan Zhang, Wenjoe Wang, Fuli Feng, and Xiangnan He.
The paper was accepted not long ago in July 2023 by the RecSys conference and has been since then already cited 22 times
(October 14th on Google Scholar).  
Within this work, the paper is analysed according to its completeness and the methods that were used. The significance
and impact of the work is easily recognizable considering search result using the indicated keywords by the authors.
Without doubt, their approach has been conducted scientifically clean and added great value to the RecSys community. Yet,
this analysis points out limitations and minor issues, that are necessary to consider in future work in order to showcase
the broad bandwidth in which unfairness exists.

## 1. Introduction

In recent years, the realm of recommender systems has witnessed a burgeoning trend in the adoption of generative models
to augment the efficacy of recommendations. With the ascent of Large Language Models (LLMs) such as ChatGPT by OpenAI
and Bard by Google, the allure of leveraging these powerful models for recommendation tasks has surged.   
However, as LLMs gain prominence in recommendation systems, concerns regarding recommendation fairness have surfaced.
These concerns are not limited to traditional recommendation systems and are equally pertinent to LLM-based 
recommendation systems. Notably, traditional fairness metrics are often inapplicable in this novel context 
(Zhang et al.), necessitating the emergence of a research domain centered around LLM fairness and bias.

The persistence of unfairness in LLM-based recommendation systems is of paramount concern, given its potential to 
significantly impact the user experience of marginalized groups. In response to this pressing issue, Zhang et al. 
have introduced the benchmarking method known as FaiRLLM. This method seeks to evaluate the fairness of LLMs in the 
context of recommendations, particularly for the top-k recommendation task. Notably, this method has been rigorously 
tested on ChatGPT, demonstrating the existence of unfairness in LLM-based recommendations and highlighting the metrics'
resilience to prompt modification.

Given the authors' assertion of being pioneers in developing a fairness metric for LLM-based recommender systems, 
a comprehensive review of this paper is imperative to assess its efficacy, comprehensiveness, and accuracy.
A preliminary examination of the paper's impact reveals that it has already garnered 22 citations, underscoring its 
influence on subsequent research and the significance of this work.   
Moreover, in light of the rapid evolution and advancements within the LLM community, it is incumbent upon LLM-based 
recommender systems to keep pace with ongoing developments in fairness and bias mitigation. Consequently, the primary 
objective of this paper review is to contribute to the acceleration of research progress in this dynamic field.

## 2. Summary

The authors of the paper "Is ChatGPT fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation"
introduce the purpose and motivation of their work in a thorough introduction paragraph. There, they underline the
fairness concerns in LLMs due to the bias in the training corpus and that it is necessary to reduce those in the
context of recommendations, since the traditional metrics to measure fairness in recommender systems can not be applied
for LLMs. This is due to the fact, that those metrics require the scores of the model prediction results, which are 
difficult to obtain in a LLM. In addition, the fairness needs to be computed on a fixed candidate set based on a
specific dataset, which would limit the recommendation ability of Recommendations via LLM (RecLLM) significantly.

In order to set a proper starting point in the fairness evaluation of RecLLMs, the authors have decided to consider the
user-side group-fairness for a top-k recommendation task on the music and movie domain. They reason their choice for
the scope setting according to the nature of conversational recommendation tasks and the complexity reduction. For
instance, music or movie items consist of way less and explicit features compared to fashion items.

After setting the scope, the authors proceed on explaining the evaluation method, the metrics they constructed to 
compute fairness, and finally document how they have generated the benchmarking dataset.    
For the evaluation method, the authors have chosen to compare neutral instructions against sensitive instructions, which
are instructions that consist additional user information in addition to the expressed preference and task. For the 
additional user information added in the sensitive prompt, the authors decided on testing the eight most discussed attributes
in the area of recommendation system fairness: *age, continent, country, occupation, gender, religion, race, physics*.    
The values for injection they have considered are depicted in the following extraction of the paper:

![img.png](img.png)

Without loss of generality,the authors came up with the following neutral and sensitive prompts, that they input in
ChatGPT in order to obtain the top-k recommendation list. A prompt consists of semantically two parts, whereas one is the 
preference expression, while the other is the task description. For the sensitive prompt, the preference expression is
extended by a sensitive attribute. The words in square brackets are placeholders, in which a value is injected
iteratively from the value-set.

- **Neutral**: *“I am a fan of [names]. Please provide me with a list of 𝐾 song/movie titles...”*
- **Sensitive**: *“I am a/an [sensitive feature] fan of [names]. Please provide me with a list of 𝐾 song/movie titles...”*

After obtaining both the neutral and the sensitive query result, the neutral one is used as a baseline result, to which
the sensitive outputs are compared to measure the similarity.   
While the attributes and values for the sensitive features have been well-defined, the choice of the values for the 
*[names]*-placeholder, representing a user preference, needed to be known by the RecLLM.
As a result, the authors have decided to query the most 500 popular singers in the music domain through the MTV API, and 
the most popular 500 directors and their most popular TV shows and movies by querying the IMDB database.  
Thus, the resulting two benchmark databases, one for each domain, consist of neutral and sensitive instructions in form
of natural language. In order to ensure reproducibility of the query outputs, the authors have decided on fixing
the hyperparameters of ChatGPT accordingly, such as *temperature, top_p, and frequency_penalty*. 

For the purpose of similarity computation of the sensitive recommendation set with the neutral output, the authors have 
designed three metrics:
1. **Jaccard**     
Computes the ratio between common elements, but does not consider the ranking.

2. **SERP***     
Is a modification of the **Search Result Page Misinformation Score**, which computes the similarity between two
recommendation lists. It can be considered as a weighted option of the Jaccard metric in order to include the item
ranking. The relative item ranking in this metric is nevertheless neglected.

3. **PRAG***    
Is a modification of Pairwise Ranking Accuracy Gap metric to include the importance of relative rankings. The metric
does it by measuring the pairwise ranking between recommendation results for the natural and sensitive instructions

The established similarity metrics represent a crucial element in the designed metric which computes the fairness of a
RecSys. For that, the authors have constructed two indicators: the Sensitive-to-Neutral Similarity Range (SNSR) and 
Sensitive-to-Neutral Similarity Variance (SNSV).  

- **SNSR**     
It measures the similarities between the most advantaged and the most disadvantaged group.

- **SNSV**    
Computes the divergence across all possible attribute values using the Standard Deviation. 

_(Note: For a detailed mathematical formulation of the metrics please refer to Appendix A)_   

In consideration with the similarity metrics, there are therefore three options to compute the fairness score.
For the analysis, all possible fairness metrics were computed and presented in a table, which is added as an excerpt
below. The table documents the fairness values of the music and the movie recommendations in a separated nature.
Aside from the *SNSR* and *SNSV* metrics, the authors have decided on adding the Min/Max value for each attribute,
which denote the minimum and maximum similarity value among all values for one attribute. The sensitive attributes
are listed in decreasing SNSV-score from left to right.

![img_1.png](img_1.png)

Finally, the scores are used for an analysis of the developed metrics by setting two research questions (RQ1 and RQ2).
By formulating these two questions, the authors attempt to proof the validity and robustness of their benchmarking
method. To facilitate the data interpretation, the authors visualized the most crucial data points by plotting for each
attribute the similarity score for the top-k recommendations against the number of k, setting k from 1 to 20. By
using the collected data as an argumentative foundation, the authors have shown that both research questions could be 
answered in favor of their developed benchmarking method:

**RQ1:** *How unfair is the LLM when serving as a recommender on various sensitive user attributes?*      
For evaluating the overall unfairness in ChatGPT as a RecLLM, the authors performed a thorough analysis on the collected
data and determined three major findings:
1. The unfairness is present in both, the music and the movie domains. The SNSV and SNSR score show clear varying values
across  the different attributes and thus proof the preference of some groups over others.
2. The recommendation unfairness does not depend on the recommendation length, as truncating the lists lead to the same results.
3. The unfairness in regard to the attribute values align with real world disadvantages. For example, African groups
receive a recommendation list which diverges more from the neutral recommendation, than Americans.

**RQ2:** *Is the unfairness phenomenon for using LLM as a recommender robust across different cases?*     
The authors produced additional recommendation lists in order to test the confidence of their metrics and the
existing bias in RecLLM by 1) replacing the sensitive attribute values through typos and 2) by inserting the prompts
in a different language. Due to resource limitations, the authors tested the robustness only on the *country*
attribute and on the values _African_ and _American_. While, typos were generated by randomly adding and removing letters,
the language robustness was tested by using Chinese instructions.   
For both parts, the results have shown that the unfairness shown in RQ1 persisted and therefore confirmed the author's
contribution.

The paper is then concluded with a brief summary about the findings within the paper and an outlook on future work
in regard to the author's goal. They state, that it is in their interest to evaluate other RecLLMs, such as LLaMA for
fairness and develop methods that mitigate the unfairness score in RecLLMs.

## 3. Analysis on the Paper

The paper "Is Chat GPT fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation" by
Zhang et al. is a solid academic foundation in the area of fairness evaluation for RecLLMs. Additional research on
fairness metrics for RecLLMs has confirmed the author's claim of providing the first contribution to a benchmarking method 
in this field. For that reason alone, the academic significance of this work should be recognized. Even though a paper 
with a similar research goal by Hua et al. was published a few months before the paper of Zhang et al., it is noticeable 
that the target and scope by Hua et al. differs significantly from Zhang et al. While Hua et al. also developed methods
to probed unfairness in a LLM-based recommendation system, they limited the methods to the used model and the trained
dataset. Thus, their approach is not applicable for unfairness probing on other LLMs.   

In addition, the scientific methods used in this paper coherently align with the basics of scientific principles, as
described by W.S. Jevons. Considering that following these standards, such as following the formalities of correct
citation, proof-based argumentation, and the empirical and experimental approach, are the status quo for published papers,
their details shall not be further elaborated within this review.   
Nevertheless, it is to note that the authors have taken the requirement for benchmarking robustness into account and
fulfilled by addressing different recommendation domains and crucial varieties of the prompts, by testing their method
for typos and for the language of the prompt.   
Furthermore, the instrumentalization of similarity metrics that are used in traditional recommendation systems are 
effective for the fairness metrics and provide an easy-to-understand transfer, allowing for adaptivity of the fairness
metrics without the need for additional hard-to-obtain data.
The persuasive nature of the paper is also supported by the paper structure, in which knowledge is transferred to the
reader in a reasonable order, so that the more complex topics are easy to follow and understand.

Finally, with their pragmatic and experimental-based reasoning, the authors manage to propose their benchmarking method in a 
convincing manner through their well-thought-out metrics that underline their statements about existing unfairness in 
RecLLM and their robustness. In general, they always provided immediate reasoning for any choice of methodology aside from
a few exceptions. For example, the authors did not specify their choice on the sensitive attribute values. Aside from 
that, other minor research gaps have been identified during the paper review, leaving room for future improvement or
initiating some adjustments to their work.

Structurally, the analysis of the experiment results have validated the usefulness of FaiRLLM. Yet, for a proper
360-degree view on their work, the paper is missing a section listing their limitations of their approach and the
discussion of their findings. Even though the limitations have been mentioned in places of the paper, such as when
the authors stated to focus only on the top-k recommendation task, an explicit paragraph about the limitations would
have provided more clearance after reading the results. Missing the discussion paragraph is pretty crucial though, as 
the interim results of the author's findings have not been discussed in other parts of the paper. As a result, it leaves
the impression as if he authors did not try to put their findings into question.    

Within the analysis of the benchmarking robustness to changes in the prompts, typos have only been tested on the attribute
_continent_ and the values _African_ and _American_. Since the authors did not argue that this small set on experiment
suffices, a broader consideration should have been shown. Especially an analysis on whether the unfairness still holds
when considering attribute values, that are already close to each other. Evaluating the recommendation lists with typos
could provide additional insights, that could either strengthen or weaken the author's points. An example would be to
consider typos for the attribute _occupation_ for _writer_ and _worker_.

![img_2.png](img_2.png)

Moreover, the language robustness has been tested solely for Chinese prompts. Since it is noticeable, that the similarity
score between Africans and Asians have switched after querying for movies, a RecLLM might show cultural bias based on the
language in which the user queries the prompt. Thus, testing language robustness in multiple languages should is
necessary.

![img_3.png](img_3.png) 
![img_4.png](img_4.png)

Finally, the authors did not provide any reasoning for their choice of the sensitive attribute values.
Looking at those in more detail, some values should have been included in order to properly cross-check the unfairness
of the RecLLM. Since the attributes _continet_ and _country_ have a hierarchical order, the consistency of the fairness
probing should have been applied as well. Due to the lack of fitting values though, this is not clear. For example, 
for the attribute _country_, there are European and South American countries, but Europe and South America are not 
included as values in the _continent_ attribute. At the same time, there is no African country listed in the _country_
attribute, making it question, whether a user from Ghana might end up with the same ranking distribution when considering
the _continent_ attribute.

## 4. Open Questions

The work of Zhang et al. is in its own very thorough and provides a solid instrument to benchmark LLMs as recommender
systems in regard to their fairness attribute. Yet, there are still few open questions that have not been dealt with
within the scope of their work, which are nevertheless important to include in follow-up work on that matter.

First, users can be described through a set of sensitive attributes and not only one. Even though the analysis of the
sensitive attributes as their own provides the recognition on the most discriminated ones, the group-fairness should be
extended by a set of sensitive attributes for additional insight.

Second, another popular task with RecLLMs are sequential recommendations. Since LLMs provide the power of iteratively 
refining their outputs based on user interactions, the question is on how the benchmarking method by Zhang et al. can
be transferred to this task. Since it is way more complex in the data format, as in recommendation representation,
sequential representation, and data load, it might require a different benchmarking method to bve developed. Considering
the popularity and customization of this task, sequential recommendations in LLMs, when containing bias, could run into 
the danger of reinforcing the bias and unfairness (Shu et al.). Zhang et al. have not made a statement on a 
benchmarking approach in that regard since the publication of this paper.

Lastly, the authors have introduced three similarity scores which are all used in order to probe a model for fairness.
Yet, for the overall and robustness analysis, only the Jaccard similarity was referenced. Within the second part of the
work, the other metrics, SERP* and PRAG* were not mention anymore. Even though their differences were explained
during their mathematical definition, it is still questionable which of those metrics should be used for a proper 
indicator in which scenarios and why the consideration of all of them are significant. In regard to the performed 
analysis by Zhang et al., it would have sufficed to solely compute the SNSR and SNSV score based on the Jaccard similarity
only. One therefore wonders: so why needing to do all the work in computing the similarities with the other two metrics?

## 4. Conclusion

All in all the work of Zhang et al. is a crucial milestone in the context of fairness probing for RecLLMs. FaiRLLM 
is a solid benchmarking tool in order to probe for RecLLM fairness and compare existing LLMs against each other.
Even though minor gaps have been detected during this paper review, those are issues that can be easily extended with
a rework on the analysis part of the paper. The curves of the plotted results look promising, so that an extension of
the attribute values might not affect the overall result of the paper.
While FaiRLLM has not yet been applied except for ChatGPT within the context of the author's work, it represents a
promising start to create an incentive for mitigating discrimination in LLMs. Even though it's downside requires 
computational effort, the gained insights by using FaiRLLM represent a solid foundation to improve user experience and
do not only leverage the development of recommender-based LLMs, but also LLMs themselves.   
For LLM-developers, using FaiRLLM can become a key performance indicator in development improvements and for users a
decisive factor on which LLM to use. Since developing methods to mitigate unfairness in LLMs is a research area in 
itself, FaiRLLM is a helpful metric to design such a method using a standardized benchmarking metric, which can then
be compared against others easily.
The open questions defined during this paper review are future work topics that are to be further defined. Especially
the inclusion of sequential recommendation benchmarking can set another new milestone in this area. Even though Zhang et
al. have stated to work on benchmarking other LLMs and developing methods to mitigate discrimination, it would be
exciting to see them pioneering in developing a benchmarking for framework for sequential recommendations with user-item
interactions.

## Appendix

### A) Mathemtiacal Definition of the Similarity and Fairness Metrices
Due to the limitations in writing for mathematical expressions in Markdown, the mathematical representations of the
deefined metrics by Zhang et al. are added as excerpts from the original paper.

For the metrics, the following mathematical notations are to be considered:

![img_7.png](img_7.png) is the set of neutral instructions

![img_5.png](img_5.png) is the set of top-k recommendations for a neutral instruction ![img_6.png](img_6.png)

![img_8.png](img_8.png) denotes a sensitive attribute where 𝑎 is a specific value of the attribute. 𝑎 is a word or a
phrase.

![img_17.png](img_17.png) denotes the number of all possible values in a studied attribute

![img_9.png](img_9.png) is a set of sensitive instructions for each value of attribute ![img_10.png](img_10.png) by injecting the value 𝑎

![img_12.png](img_12.png) is the recommendation list for a sensitive attribute ![img_10.png](img_10.png) and the value 𝑎

![img_13.png](img_13.png) computes the similarity between ![img_14.png](img_14.png)

![img_15.png](img_15.png) is the aggregated similarity value across all M instructions

![img_16.png](img_16.png) is the level of unfairness in RecLLM as the divergence of these aggregated similarities 
across different values of the sensitive attribute

![img_21.png](img_21.png) denotes the number of common items between the two recommendation lists

**𝑣** represents an item in a sensitive recommendation list

![img_24.png](img_24.png) represents the rank of the item 𝑣

![img_25.png](img_25.png) is 1 if 𝑣 is in the neutral recommendation list, else 0

![img_27.png](img_27.png) denote two different recommended items in the sensitive recommendation list

![img_28.png](img_28.png) or ![img_29.png](img_29.png) is the rank in the recommendation list ![img_30.png](img_30.png)



**SNSR**
![img_19.png](img_19.png)

**SNSV**    
![img_18.png](img_18.png)

**Jaccard**   
![img_20.png](img_20.png)

**SEREP***    
![img_22.png](img_22.png)

**PRAG***
![img_26.png](img_26.png)

## Sources

Asia J. Biega, Krishna P. Gummadi, and Gerhard Weikum. 2018. Equity of
Attention: Amortizing Individual Fairness in Rankings. In SIGIR ’18: The
41st International ACM SIGIR Conference on Research and Development in
Information Retrieval, July 8–12, 2018, Ann Arbor, MI, USA. ACM, New York,
NY, USA, 10 pages. https://doi.org/10.1145/3209978.3210063

Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023.
Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. arXiv preprint
arXiv:2305.07609.

Jevons, W. S. 1958. Principles of Science. Daedalus, 87(4), 148-154.

Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, 
Jiliang Tang, and Qing Li. 2023. Recommender Systems in the Era of Large Language Models (LLMs). arXiv.
https://arxiv.org/pdf/2307.02046.pdf 

Wenyue Hua, Yingqiang Ge, Shuyuan Xu, Jianchao Ji, and Yongfeng Zhang. 2023.
UP5: Unbiased Foundation Model for Fairness-aware Recommendation. arXiv. https://arxiv.org/abs/2305.12090 

Yubo Shu, Hansu Gu, Peng Zhang, Haonan Zhang, Tun Lu, Dongsheng Li, Ning Gu. 2023. 
RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models.
arXiv. https://doi.org/10.48550/arXiv.2308.09904

Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2021. 
User-oriented Fairness in Recommendation. In Proceedings of the Web Conference 2021 (WWW ’21), 
April 19–23, 2021, Ljubljana, Slovenia. ACM, New York, NY, USA, 9 pages. 
https://doi.org/10.1145/3442381.3449866


