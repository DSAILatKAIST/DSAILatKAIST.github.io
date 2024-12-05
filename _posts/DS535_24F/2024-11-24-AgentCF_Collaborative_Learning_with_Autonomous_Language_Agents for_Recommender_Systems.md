---
title:  "[WWW-24] AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems"
permalink:  2024-11-24-AgentCF_Collaborative_Learning_with_Autonomous_Language_Agents for_Recommender_Systems.html
tags:  [reviews]
use_math:  true
usemathjax:  true
---

  
  

## **1. Problem Definition**

  

This paper considers the following item ranking problem:

  

- GIVEN: A user $u$ and a list of candidate items $\{c_1, \dots, c_n\}$

- FIND: A ranking over the given candidate list

- ASSUMPTIONS:

	- Interaction records between users and items $\mathcal{D}=\{\langle u, i\rangle\}$, including reviews and item meta-data, is available.

  

## **2. Motivation**

  

The semantic knowledge encoded in LLMs have been utilised for various tasks involving human behaviour modeling. However, for recommendation tasks, the gap between universal language modeling, which is verbal and explicit, and user preference modeling, which can be non-verbal and implicit, poses a challenge. As such, this paper proposes to bridge this gap by treating both users and items as collaborative agents to autonomously induce such implicit preference.

  

Existing approaches to LLM-powered recommendation that employs agents are one-sided, i.e., either item-focused or user-focused, with self-learning strategies for such agents; meanwhile, this work considers a *collaborative* setting where users and items do interact. From a collaborative learning perspective, this work is also novel with agent-based collaboration modeling instead of traditional function modeling (e.g., inferring embeddings of users and items for similarity measure) methods.

  

## **3. Method**

  

![overall](https://i.postimg.cc/fbhbXFRh/overall.png)

*Figure 1. An overview of the algorithm*

  

### 3.1. Initialisation

  

The algorithm initialises two types of agents, users and items, equipped with memory.

- User memory includes short-term memory $M _u^s$ (natural language text describing any recently updated preference of user, initialised by general preference induced from reviews) and long-term memory $M _u^l$ (evolving records preference texts).

An example for user short-term memory can be as follows. Note that the long-term memory is simply a log storing all pieces of short-term memory along the optimisation process.

>   My updated self-introduction: I enjoy listening to CDs that fall under the classic rock and album-oriented rock (AOR) genres. I particularly appreciate CDs that offer a unique and distinct sound within these genres. I find myself drawn to CDs that showcase a variety of hits and popular songs, as they provide an enjoyable listening experience. Additionally, I have discovered a newfound preference for CDs that have a more experimental and innovative approach to classic rock. These CDs offer a refreshing twist on the genre and keep me engaged throughout. On the other hand, I tend to dislike CDs that lack originality and fail to bring something new to the table. CDs that rely heavily on generic rock sounds and do not offer any standout tracks or moments tend to leave me uninterested. Overall, I seek CDs that captivate me with their creativity and deliver a memorable listening experience.

- Item memory includes a unified memory $M _i$ with characteristics (initialised by meta-data) of the item and preferences of users who purchased it, all in natural language.

An example for item memory is

> “Brainwashed” is a classic rock album from the Album-Oriented Rock (AOR) genre that breaks new ground with its experimental and innovative approach. This CD offers a refreshing twist on classic rock, delivering a captivating listening experience that keeps you engaged from start to finish. “Brainwashed”stands out from generic rock sounds, providing standout tracks and moments that showcase its originality and creativity. For the user who seeks CDs that push boundaries and bring something new to the table, “Brainwashed” is a must-listen.

Note that the proposed method also assumes the availability of a frozen LLM which will serve as the backbone for memory optimisation and the ranking mechanism.

  
  

### 3.2. Memory Optimisation

  

The following two processes are carried out alternatively with each item along the interaction sequence of a user.

  

#### 3.2.1. Autonomous Interactions for Contrastive Item Selection

  

From (1) the running memory of the user together with (2) a random popular item (denoted $i^-$, the negative item) and (3) a true next item (denoted $i^+$, the positive item), an LLM has to predict an item that the user will interact with, $i^o$,
as well as an explanation $y _{exp}$ for such a selection. Here the two items are presented equally to the user (i.e., the ground truth is hidden).

$i^o = f_{LLM}  (M _u; M _{i^-}; M _{i^+}),$

$y _{exp} = Prompt _{LLM}  (i^o; M _u; M _{i^-}; M _{i^+}).$

These two equations, for example, can be captured in the following prompt:

> You are a CD enthusiast. Here is your self-introduction, expressing your preferences and dislikes: “user agent memory”. Now, you are considering to select a CD from two candidate CDs. The features of these two candidate CDs are listed as follows:
> 1. neg item agent memory 
> 2. pos item memory. 
> 
> Please select the CD that aligns best with your preferences. Furthermore, you must articulate why you’ve chosen that particular CD while rejecting the other. 
> To do this, please follow these steps:
> 1. Extract your preferences and dislikes from your self-introduction.
> 2. Evaluate the two candidate CDs in light of your preferences and dislikes. Make your choice by considering the correlation between your preferences/dislikes and the features of the CDs.
> 3. Explain why you make such choices, from the perspective of the relationship between your preferences/dislikes and the features of these candidate CDs. 
> 
> Important note:
> 1. Your output should in the format: Chosen CD: [Title of the selected CD] 
> Explanation: [Detailed rationale behind your choice and reasons for rejecting the other CD].
> 2. When identifying your likes and dislikes, don’t fabricate them! If your self-introduction doesn’t specify any relevant preferences or dislikes, use common knowledge to inform your decision, such as the popularity of the CDs. 
> 3. You must choose one of these two candidates, instead of choosing both of them. 
> 4. Your explanation needs to be comprehensive and specific. A general preference, such as a certain genre of CD, is insufficient. Your reasoning should delve into the finer attributes of the CD.
> 5. Base your explanation on facts. For instance, if your self-introduction doesn’t reveal any specific preferences, you cannot assert that your decision was influenced by such preferences."  

An example response would be

> Chosen CD: O, Yeah! Ultimate Aerosmith Hits 
> Explanation: I chose O, Yeah! Ultimate Aerosmith Hits because it is a classic rock album from the Album-Oriented Rock (AOR) genre, which is a genre I enjoy listening to. The album contains some of Aerosmith’s greatest hits, which I believe will provide me with a great listening experience. In comparison, Brainwashed is also a classic rock album from the AOR genre, but it is not as popular as O, Yeah! Ultimate Aerosmith Hits, and therefore I believe it will not provide me with the same level of listening experience.

#### 3.2.2. Collaborative Reflection and Memory Update

  

After a predicted item and an explanation are obtained, they are fed to an LLM reflection module together with the user's, positive item's, and negative item's memory. The information on the correctness of the prediction is available, allowing the agents' memory to update accordingly, improving alignment between the memory and the underlying preferences.

  
First, we have user reflection:

$M _u^s \leftarrow Reflection^u (i^o; y _{exp}; M _u; M _{i^-}; M _{i^+})$

An example prompt is as follows:


> You are a CD enthusiast.
> Here is your previous self-introduction, exhibiting your past preferences and dislikes:“user agent memory”. Recently, you considered to choose one CD from two candidates. The features of these two candidate CDs are listed as follows:
> 1. neg item agent memory 
> 2. pos item memory. 
> 
> After comparing these two candidates based on your preferences and dislikes, you selected “neg item title” to listen to while rejected the other one. You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes for these two CDs: “user explanation”. However, upon actually listening to these two CDs, you discovered that you don’t like the CD that you initially chose (Here And There Remastered). Instead, you prefer the CD that you did not choose before (Thriller).
> This indicates that you made an incorrect choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was mistaken. It’s possible that your preferences and dislikes, described in your previous self-introduction, have either changed or were uninformative. Therefore, your task now is to update your self-introduction, by incorporating your new preferences and dislikes revealed in this interaction. To do this, please follow these steps: 
> 1. Analyze the misconceptions in your previous judgment about your preferences and dislikes, as recorded in your explanation, and correct these mistakes 
> 2. Explore your new preferences based on the features of CDs you really enjoy (Thriller), and determine your dislikes based on the features of CDs you truly don’t enjoy (Here And There Remastered). 
> 3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes. 
> 4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes. 
> 
> Important note:
> 1. Your output should use the following format: “My updated self-introduction: [Please write your updated self-introduction here].” 
> 2. Keep your updated self-introduction under 180 words. 
> 3. Any overall assessments or summarization in your self-introduction are forbidden.
> 4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your wrong choice or your thinking process in updating your self-introduction. 
> 5. You self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording.

An example response is as follows:

> My updated self-introduction: I enjoy listening to CDs that fall under the classic rock and album-oriented rock (AOR) genres. I particularly appreciate CDs that offer a unique and distinct sound within these genres. I find myself drawn to CDs that showcase a variety of hits and popular songs, as they provide an enjoyable listening experience. Additionally, I have discovered a newfound preference for CDs that have a more experimental and innovative approach to classic rock. These CDs offer a refreshing twist on the genre and keep me engaged throughout. On the other hand, I tend to dislike CDs that lack originality and fail to bring something new to the table. CDs that rely heavily on generic rock sounds and do not offer any standout tracks or moments tend to leave me uninterested. Overall, I seek CDs that captivate me with their creativity and deliver a memorable listening experience.

We also have item reflection:

$M _i \leftarrow Reflection^i (i^o; y _{exp}; M _u; M _{i^-}; M _{i^+})$

An example prompt is as follows:

> Here is the self-introduction of a user, exhibiting his preferences and dislikes: “user agent’s memory”.
> Recently, the user browsed a shopping website and considered to choose one CD from two candidate CDs.
> The descriptions of the two candidate CDs are as follows:
> 1. pos item agent’s memory 
> 2. neg item agent’s memory 
> 
> By considering his preferences and dislikes from his self-introduction, he chose “neg item title” to listen to while rejected the other one. He provided the following explanations for his choices: “user explanation”. 
> However, the user found that he actually doesn’t enjoy the CD that he previously chosen. Conversely, he finds the CD that he didn’t choose before, titled “pos item title”, more appealing. This indicates that he made an unsuitable choice. This may be because the user was misled by the descriptions of these two candidate CDs. Actually, the user prefers a CD (“pos item title”) because its features aligns with his preferences and go against his dislikes, while he dislikes the other CD (“neg item title”) because it possesses features that he disliked. Therefore, your task is to update the descriptions of these two candidate CDs. 
> To do this, follow these steps:
> 1. Analyze which features recorded in the previous CD description led to the unsuitable choice based on the user’s explanation, and modify them accordingly.
> 2. Analyze the user’s preferences and dislikes from his self-introduction. Explore the user preferred CD’s new features that aligns with the user’s preferences and goes against with his dislikes. Explore the user disliked CD’s new features that aligns with the user’s dislikes and goes against with his preferences. In this case, the differences between the descriptions of these two items should be amplified. Your exploration should be thorough and detailed, rather than just superficial. For example, you can’t simply say a CD’s genre aligns with or goes against the user’s preferences, as that is too broad.
> 3. Incorporate the item’s newfound features with the into its previous description. Since the previous descriptions of the candidate CDs are based on evaluations from other users, it is also worth preserving valuable content. Please try to maintain conciseness while preserving the information as much as possible.
> 
> Important notes: 
> 1. Your output should in the following format: “The updated description of the first CD is: [updated description]. 
> The updated description of the second CD is: [updated description].”. 
> 2. Each updated CD description can not exceed 50 words!!! 
> 3. In your updated CD description, considering that it will be visible to other users, it is advisable to utilize user preferences or dislikes when referring to a group of users collectively, rather than using “the user” to specifically denote an individual. For instance, you can mention “the user with ... preferences/dislikes”.
> 4. The updated CD description should not contradict the inherent characteristics of this CD. For example, you can not say that a hip-hop CD has a very soothing rhythm. 
> 5. The updated CD description should be specific, containing only the distinguishing features that differentiate this CD from others.

An example response is as follows:

> “Brainwashed” is a classic rock album from the Album-Oriented Rock (AOR) genre that breaks new ground with its experimental and innovative approach. This CD offers a refreshing twist on classic rock, delivering a captivating listening experience that keeps you engaged from start to finish. “Brainwashed”stands out from generic rock sounds, providing standout tracks and moments that showcase its originality and creativity. For the user who seeks CDs that push boundaries and bring something new to the table, “Brainwashed” is a must-listen.

Here we note that the item of which the memory is to be updated is the positive item, not the negative one, since the LLM tends to over-emphasise the drawbacks of the negative item, while such an item can also be attractive and serve as a positive item for other users.

Finally, the updated short-term user memory is appended to the long-term memory.

$M _u^l \leftarrow Append (M _u^l; M _u^s)$


### 3.3. Candidate Item Ranking

  

After the optimisation process in Section 3.2. is finished, the agents are ready to be used for item ranking. Given a user $u$ and a candidate set $\{c_1, \dots, c_n\}$, there are three ranking strategies considered.

- Basic stategy: based on the user's short-term memory and all item's memory. The ranking result is $\mathcal{R}  _B = f _{LLM}(M _u^s; \{M _{c _1}, \dots, M _{c _ n}\})$.

- Advanced strategy 1: In addition to the memory utilised in the basic strategy, we also use $M _u^r$, preference retrieved from the long-term memory by querying with candidate items' memories. The ranking result is $\mathcal{R}  _{B+R} = f _{LLM}(M _u^r; M _u^s; \{M _{c _1}, \dots, M _{c _ n}\})$.

- Advanced strategy 2. In addition to the memory utilised in the basic strategy, we also use memories of items that the user $u$ interacted with historically. The ranking result is $\mathcal{R}  _{B+H} = f _{LLM}(M _u^s; \{M _{i _1}, \dots, M _{i _ m}\}; \{M _{c _1}, \dots, M _{c _ n}\})$.

  
  

## **4. Experiment**

  

### **Experiment Setup**

  

- Dataset: the authors utilised two subsets of the Amazon review dataset, "CDs and Vinyl" and "Office Products", sampling 100 users from each of them for two scenarios, dense and sparse interactions.

  

- Baseline

	- BPR [1]: matrix factorisation

	- SASRec [2]: transformer-encoder

	- Pop: simply ranking candidate items by popularity

	- BM25 [3]: ranking items based on their textual similarity to items the user interacted with

	- LLMRank [4]: zero-shot ranker with ChatGPT

  

- Evaluation Metric: NDCG@K, where K is set to 1, 5, and 10.

  

### **Result**

  

#### Main Results

  

The proposed method together with its three ranking strategies is comparable or outperforms baselines on the considered datasets. Specifically:
- Utilising AgentCF as a collaborative filtering recommender ($\text{AgentCF}_ {B}$) effectively leverages preference propagation to capture the essence of collaborative filtering. Moreover, incorporating user agents’ specialised preferences for candidates ($\text{AgentCF}_ {B+R}$) allows for more tailored recommendations. In scenarios with sparse interactions, such as the Office dataset, utilising user historical interactions to enrich the prompt and enabling LLMs to act as sequential recommenders ($\text{AgentCF}_ {B+H}$) leads to improved performance.
- AgentCF performs as well as or better than traditional recommendation models on similarly scaled datasets and even matches their performance on full datasets in sparse scenarios. Notably, it achieves this using just 0.07% of the complete dataset, highlighting its strong generalisation capability despite room for improvement in certain scenarios.
- Existing tuning-free methods like Pop, BM25, and LLMRank perform poorly. While LLMRank uses ChatGPT with user history as prompts to model preferences, it significantly lags behind traditional models. This underscores the challenge of aligning LLMs' universal knowledge with user behavior patterns, limiting their ability to capture preferences effectively.

  

![main_results](https://i.postimg.cc/HxMbxTfQ/main-results.png)

  *Figure 2. Main results*

#### Ablation

  

Ablation study is conducted on the inclusion of autonomous interaction, user agents, and item agents. Results show that all components are useful, except for the case of NDCG@1 on the Office category data, where the algorithm without user agent excels; this is argued by the authors to be due to the long item description text included in the data, which effectively render LLMs as sequential recommenders. Even in such a scenario, the inclusion of item agents is still shown to be necessary. This shows that:
- Analysing the differences between agents' autonomous interactions and real-world interaction records provides valuable feedback on their misalignment, improving the effectiveness of reflection.
- Simple verbalised user interaction text fails to accurately represent preferences, highlighting the importance of adjusting agents’ memories to autonomously simulate real user behavior. While prompting LLMs with user history improves results in the Office dataset—likely due to its longer item descriptions enabling sequential recommendation—the use of LLM-powered item agents further enhances performance. This underscores the value of enabling items to better capture user preferences.
- Failing to optimise item agents, which represent items using only identity information, results in poorer performance. This emphasises the importance of behavior-based item modeling to capture the two-sided interactions between users and items. Notably, item agents are essential for simulating collaborative filtering systems by updating their memory with user preferences and propagating this information to new agents through interactions.

  

![ablation](https://i.postimg.cc/44zzRNvG/ablation.png)

*Figure 3. Ablation results*

#### Further Experiments

Additional experiments on user-user interaction, item-item interaction, and preference propagation are also conducted.

![user-user](https://i.postimg.cc/MKtPH99D/user-user.png)
*Figure 4. Proportion of purchase after viewing reviews for user-user interaction experiment.*

Real-world users often rely on others' advice when engaging with unfamiliar items, such as browsing reviews on shopping sites. To simulate similar social behaviors, the authors use AgentCF to enable user agents to read and write reviews. For test items unfamiliar to certain users, other agents who have interacted with these items write reviews, which the test users then read before making decisions. The purchase proportion after viewing results is visualised in Figure 4. Comparing their decisions before and after reading the reviews reveals that the simulated agents mimic real user behavior—they are more likely to purchase items after positive reviews, avoid purchases after negative ones, and trust reviews from agents with similar preferences. 

![item-item](https://i.postimg.cc/SNm1mBfR/item-item.png)
*Figure 5. Recommendation performance for a cold-start scenario (0), interaction with similar popular items (Sim), and interaction with a distinct popular item (Dist).*

Recommender systems often struggle with suggesting new items due to the cold-start problem. To address this, the authors simulate autonomous interactions between new and popular item agents. New item agents, equipped with basic identity information like titles and categories, learn from popular item agents with extensive interaction records. This process helps new agents estimate adopter preferences and adjust their memory. User agents then rank the new items alongside well-trained items, comparing rankings based on the cold-start versus adjusted memory. As shown in Figure 5, these interactions improve the propagation of user preferences to new items and enhance ranking performance, demonstrating the benefits of item-item interactions enabled by AgentCF.

![optimised-memory](https://i.postimg.cc/DwMxd8k4/optimised-memory.png)
*Figure 6. Optimised memory of various agents.*


![preference-propagation](https://i.postimg.cc/SxsZMc5p/preference-propagation.png)
*Figure 7. The proportion out of all users which has the same preference as the seed user through 1 hop, 2 hop, and 3 hop preference propagation.*

AgentCF achieves preference propagation through collaborative optimisation. This process begins by initialising a seed user agent's memory with unique preference descriptions that wouldn't naturally arise during standard optimisation. User and item agents then engage in autonomous interactions to refine their memories. Figure 6 illustrates the optimised memories of various agents, showing how these interactions enable user and item agents to act as bridges, spreading preferences to other agents with similar behaviors.

To evaluate propagation efficiency, the authors examined whether other user agents adopt the seed user's preferences. Figure 7 shows that as interactions continue, more agents with similar behaviors adopt these preferences. However, as the range of interactions grows, the proportion of agents retaining these preferences diminishes, reflecting information decay akin to real-world diffusion dynamics. These findings highlight AgentCF's capability to support collaborative filtering systems and foster collective intelligence development.

## **5. Conclusion**

  

This paper proposed a novel modeling of two-sided interactions between user and item LLM agents for the recommendation ranking task, which serves as a proxy for preference modeling. This approach alleviates the gap between the semantic knowledge in general-purpose LLMs and deeper behavioural patterns inherent in user-item interactions.

  

I find the optimisation process of this algorithm intriguing, as it implicitly mimics the forward pass and gradient-based updates in traditional optimisation. Furthermore, the core idea of interactions between both user and item agents is indeed interesting. However, the specific implementation of it in this paper has a number of issues:

1. The extensive use of the LLM backbone with very long prompts may result in instability (i.e., results may not be consistent), poor attention to all parts of the given prompts, and expensive computational costs.

2. The choice of implicit contrastive learning (i.e., with negative and positive items) is presented somewhat arbitrarily. No reasoning was given as to why such contrastive learning was preferred over classifying a single item as true or false interaction, for example.

  

I think there can be two directions (or more) in which subsequent research can develop:

1. Instead of letting agents interact superficially through an LLM prompting interface, we could either finetune models or modify embeddings that represent the agents in some way. This may potentially be less expensive than extensive prompting and at the same time provide more flexibility to agents.

2. The idea of comparing product might be relevant to preference optimisation in LLM alignment (see direct preference optimisation, DPO [5]). We might be able to treat recommendation as a generative task while still optimising user or item agents with an LLM backbone.

  

---

  

## **Author Information**

  

* Name: Quang Minh Nguyen (see me at https://ngqm.github.io/)

* Affiliation: KAIST Graduate School of Data Science

* Research Topic: Reasoning on Language and Vision Data, Natural Language Processing for Social Media Analysis

  

## **6. Reference & Additional materials**

  

### Paper Information

- Code: https://github.com/RUCAIBox/AgentCF

- Full text: https://dl.acm.org/doi/10.1145/3589334.3645537

  

### References

[1] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI, Jeff A. Bilmes and Andrew Y. Ng (Eds.). AUAI Press, 452–461.

[2] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM.

[3] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (2009), 333–389.

[4] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian J. McAuley, and Wayne Xin Zhao. 2023. Large Language Models are Zero-Shot Rankers for Recommender Systems. CoRR (2023).

[5] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In _37th Conference on Neural Information Processing Systems_, 2023. . Retrieved from https://neurips.cc/virtual/2023/oral/73865