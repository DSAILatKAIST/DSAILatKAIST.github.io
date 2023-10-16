---
title:  "[SIGIR 2023] Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation"
permalink: 2023-10-16-Uncertainty-aware_Consistency_Learing_for_Cold-Start_Item_Recommendation.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation**

## **1. Problem Definition and Motivation**

In this paper, the central problem at hand is the Cold-Start problem in  GNN-based recommendation systems, particularly when new items with limited interaction data are introduced to a user-item graph.  The cold-start problem especially becomes tricky when the user-item graph constantly changing. This problem arises due to the scarcity of information on these "cold" items, which hinders accurate recommendations. To tackle this issue, existing models primarily rely on auxiliary user and item features which underutilize actual user-item interactions. Cold items have different embeddings due to fewer interactions comparing to warm items. This difference leads to a challenge of improving recommendations for both simultaneously which is called seesaw phenomenon.

**For instance,** Suppose movie recommender system is using a vector of length 100 for movie embeddings:

The embedding for "Spider-Man" might encompass values such as [0.2, 0.1, -0.3, 0.5, ...], capturing its distinct characteristics and traits.

However, when a new movie, "Kaist-Man," is introduced, its embedding might initiate with values like [?, ?, ?, ...], primarily due to the lack of interaction data.

in this case, because "Spider-Man" is more likely to be recommended to users who are interested in superhero action movies than "Kaist-Man". because the system has more information about it. The huge gap between their embeddings makes it challenging to improve recommendations for both simultaneously.

To bridge this gap, in this paper, an Uncertainty-aware Consistency learning framework for Cold-start item recommendation (UCC) was introduced which relies exclusively on user-item interactions.  This framework has two key designs: a) Uncertainty-aware Interaction Generation, and b) Graph-based Teacher-Student Consistency Learning.

## **2. Problem formulation**

**Notations.** In this paper $U = \{u\}$ is the set of users and  $I = \{i\}$ is set of items and  $O =\{ (u, i^+) \vert u  ∈ U, i^+ ∈ I \}$ is user-item interactions, where each pair represents each observed feedback.
**Input:**  a bipartite graph, where $V = U∪I$ is node set and $G = (V, O^+)$ is the edge set. if $M$:= number of users, $N$:=number of items, $D$:= dimension size of embedding
Then in the training process of graph representation learning, $E_ {u}  = [e_ {u_ {1}} , ... , e_ {u_ {M}} ]∈ R^{M\times D}$ is user embedding and $E_ {i}  = [e_ {i_ {1}} , ... , e_ {i_ {N}}] ∈R^{N\times D}$ is item embedding.
**Output:** Recommender models assessing the relationships between unobserved user-item pairs using the dot products of their embeddings. For given user $m$ and given item $n$ the score $\{s_ {mn}\}$ is calculated  as: $s_ {mn}  = e_ {u_ {m}} {e_ {i_ {n}} }^T$.
A larger value of ${s_ {mn}}$ indicates a stronger preference by the user for the item. The top-k items from the ranking list are recommended to the user.

## **3. Method**

![figure1](https://i.ibb.co/2qjj3J0/figure1.png)

In this paper, they tackle the distribution gap between cold and warm items by employing an uncertainty-aware approach to generate interactions. Then they kept both item-level and model-level similarity with consistency.

### 3.1. Uncertainty-aware Interaction Generation

To determine whether generated interactions are accurate and unbiased enough they introduce the uncertainty degree of each user-item interaction which is calculated by cosine distance.
for user $u_ {m}$ and item $i_ {n}$ cosine distance is
$d_ {mn}= {\vert e_ {u_ {m}} {e_ {i_ {n}} }^T\vert \over \vert \vert e_ {u_ {m}}\vert\vert \ \vert \vert e_ {i_ {n}}\vert \vert }$
$\{s_ {mn}\}_ {n=1}^N$ is ranking scores of item $i_ {n}$ for all users calculated with the pre-trained  recommender, then overall interaction uncertainty of the item $i_ {n}$ can be estimated by the average of all rankings scores:
{% raw %}
$\bar{s_n}= {{1\over M}}\sum_ {k=1}^M s_ {mn}$
{% endraw %}
So, high $\bar{s_n}$ suggests low item uncertainty, indicating widespread user acceptance, and whole low $\bar{s_n}$ indicates higher uncertainty.
In order to bridge the gap between cold and warm items' distribution and popularity bias all interactions with $d_ {mn}<\alpha \bar{s_n}$ will be regarded as uncertain interactions and filtered in the generation stage. So the selection would be as follow:
$\hat{O_ {n}}={I(d_ {mn}>\alpha \bar{s_n})}$,
where $\alpha$ is a pre-defined parameter and $I$ is the indicator function.
In other words, they find the average ranking score for an item for all users ($\bar{s_n}$) and then for each user if item-user similarity ($d_ {mn}$) is smaller than $\alpha \bar{s_n}$ they regard that interaction as uncertain.

### 3.2. Teacher-student Consistency learning

To address seesaw phenomena, and to achieve better recommendations for both warm and cold items at the same time, cold-item with generated low-uncertainty interactions should have a similar distribution with the warm items, so in this paper, they trained the teacher model (generator) and student model (recommender) with consistency learning
trained the teacher model (generator) and student model (recommender) with consistency learning
we train the teacher model (generator) and student model (recommender) with consistency learning, to ensure the cold items with additionally generated low-uncertainty interactions can have similar distribution with the warm items.

#### 3.2.1 Item-level consistency learning

This technique employs a contrastive loss to compare item embeddings before and after a generation process.
Two types of augmentations are used:
- **Weak Augmentation:**
In this, edges in a graph are dropped out based on a dropout ratio $\rho$ to make the model more robust and less sensitive to specific data points. It can be formulated as follows:
${\mathcal G}^{w}=({\mathcal V},{\mathcal M}\cdot{\mathcal O}^{+})$ where $\mathcal{M}\in\{0,1\}^{\vert O^{+}}$ is a masking vector.
For example, in the movie recommendation system, if you always include the same user's connection to a particular movie, the model might become too biased towards that user's preferences. By using dropout, you randomly exclude some user-movie connections in each training iteration, making the model more balanced and better at recommending movies for a variety of users.
Where M is a masking vector containing binary values {0, 1} for each element in O+.
- **Strong Augmentation:**
This type involves adding more edges to the graph based on generated labels and can be formulated as follows:
${\mathcal G}^{s}=({\mathcal V},{\mathcal O}^{+} +{\widehat O})$
For example, if two movies share many common actors or are often rated similarly by users, strong augmentation would introduce more connections between those movies in the recommendation graph. This enriches the information available to the model, allowing it to make more accurate and diverse movie recommendations.

These two augmentation operations create two different views for each node, denoted as $z_i^{\prime},z_i^{\prime\prime}$ 𝑖 (for weak and strong graphs, respectively).
Consistency Regularization: To encourage the similarity between the different views of the same node, a consistency regularization is implemented. This involves the use of a contrastive loss:
$\mathcal{L}_ {cr, item}=\sum_ {i\in I}-\log\frac{\exp(sim(z_i',z_i'')/\tau)}{\sum_ {j\in I}\exp(sim(z_i',z_j'')/\tau)}$,
where $\mathcal{L}_ {cr, item}$ represents the item-side consistency regularization loss for both the teacher and the student model, $sim()$ is cosine similarity function and $\tau$ is a predefined hyper-parameter.Similarly, $\mathcal{L}_ {cr, user}$ can be computed.  $\mathcal{L}_ {cr}=\mathcal{L}_ {cr, item}+\mathcal{L}_ {cr, user}$ represents the ultimate consistency loss used for consistency regularization.
Recommendation loss can be calculated as follow:

$\mathcal{L}_ {\mathbf{rec}}=\sum_ {(u,i^+,i^-)\in O}-\ln\sigma(\hat{y}_ {ui^+}-\hat{y}_ {ui^-})+\lambda\vert \vert \Theta\vert \vert _2^2,$ is L2-regularization of model’s parameters.
And Finally, our total loss is $\mathcal{L}_ {total}=\mathcal{L}_ {rec}+\mu\mathcal{L}_ {cr}$ where $\mu$ is a hyper-parameter.

#### 3.2.2 Model-level consistency learning

To keep the consistency between the teacher model and student, they suggested collecting the teacher embedding into the student embedding: 
$\mathbf{E}^s\leftarrow\gamma\mathbf{E}^s+(1-\gamma)\mathbf{E}^t$
where $E^s$ and $E^t$ are the embeddings of the student and student model's embedding respectively.

## **4. Experiment**

### **Experiment setup**
**Datasets:**
-  Experiments on Yelp and Amazon-Book benchmark datasets.
-  Follow a 10-core setting as in previous studies (Lightgcn and Neural collaborative filtering)
-  Split user-item interactions into training, validation, and testing sets (ratio 7:1:2).
**Baselines:**
- The foundation for our GNN-based model is LightGCN
- Focused on modelling user-item interactions, not item features.
- They used two types of recommendation models, the Generative model and Denoising models for user-item interactions: IRBPR, ADT, and SGL.

**Evaluation Metric**

Following previous studies, they evaluate performance using Recall@K and  NDCG@K where K = 20

### **Result**

The overall results are shown in Table 1.

![Table1](https://i.ibb.co/5F54hKF/table1.png)

This table shows how UCC outperforms previous models.

The paper conducts a comparative analysis between the proposed method and LightGCN in the context of cold-start recommendation. Items are categorized into ten groups based on popularity, ensuring equal interaction numbers for each group. The last two groups represent cold-start items, with higher GroupID values indicating warmer items. The following figures show how UCC has better performance in all groups

Unlike other methods that often sacrifice warm item accuracy to improve cold items, the proposed method notably enhances the recall of LightGCN, especially for cold items. For instance, on the Yelp dataset, the proposed method achieves nearly a 7-fold increase in recall compared to LightGCN. The most significant improvement is observed for group-id 1 items in Amazon-Book, with recall improving by 400%, underscoring the method's effectiveness in addressing cold-start recommendation challenges and highlighting the "seesaw phenomenon" problem.

![figure2](https://i.ibb.co/j87vzrD/figure2.png)

In their ablation study, the result shows that the number of generated interactions is adaptive for different item groups. It notes that low-uncertainty interactions, which are more abundant for cold items, help alleviate the distribution difference between warm and cold items. Using item-side-generated interactions significantly improves performance, while user-side-generated interactions exacerbate the distribution gap. This underscores the effectiveness of the uncertainty-aware interaction generation component. Also, as shown in the following figure teacher-student learning outperforms other methods.

![figure3](https://i.ibb.co/HhYyCx0/figure3.png)

## **5. Conclusion**

This paper tackles the Cold-Start problem in recommendation systems by introducing the Uncertainty-aware Consistency Learning framework (UCC). UCC's Uncertainty-aware Interaction Generation effectively bridges the gap between cold and warm items, resulting in notable improvements in recommendation performance.

The Teacher-Student Consistency Learning component further enhances recommendation quality, addressing the seesaw phenomenon. Extensive ablation studies and experiments on benchmark datasets showcase the effectiveness of the UCC model, consistently outperforming existing approaches, especially in improving recall for cold items. This paper offers a promising solution to enhance recommendation systems, particularly in dynamic settings.

However, it may face challenges related to complexity, scalability, and generalizability, and a broader evaluation with diverse datasets and consideration of interpretability is needed to establish its practical applicability. Additionally, a more in-depth analysis of its performance compared to a wider range of state-of-the-art approaches would provide a more comprehensive understanding of its competitiveness.

## **Author Information**

- **Author Name:**
	- Taichi Liu
	- Chen Gao
	- Zhenyu Wang
	- Dong Li
	- Jianye Hao
	- Depeng Jin
	- Yong Li

- **Affiliation:**
	- Tsinghua University
	- Huawei Noah's Ark Lab

- **Research Topic:**
	- Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation
