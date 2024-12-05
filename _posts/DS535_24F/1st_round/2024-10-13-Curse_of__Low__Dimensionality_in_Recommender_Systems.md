---
title:  "[SIGIR-23] Curse of Low Dimensionality in Recommender Systems"
permalink: 2024-10-13-Curse_of__Low__Dimensionality_in_Recommender_Systems.html
tags: [reviews]
use_math: true
usemathjax: true
---


## **1. Problem Definition**  

Lets consider how to measure the quality of recommender systems. The first thing that comes to mind is that higher recommendation accuracy generally indicates better quality. However, diversity, fairness, and robustness also play crucial roles in determining the overall quality of recommender systems.

User-item embeddings are an essential component of recommender systems, and the method of embedding these interactions is important. Additionally, due to the low dimensionality of user and item embeddings, particularly in dot-product models like matrix factorization, certain issues arise in recommender systems because low dimensionality embeddings limit the expressive power of these models.

## **2. Motivation**

The motivation of this paper is to empirically and theoretically demonstrate the necessity of sufficient dimensionality for user/item embeddings to achieve high-quality recommender systems.

The main motivation revolves around the following key concepts, and is explained by deriving insights using these terms:

### **Dot-Product Model:**
In recommender systems, the dot product is used to compute the preference score between a user and an item by taking the inner product of their respective embedding vectors. These embeddings represent users and items in a shared, lower-dimensional space, enabling the system to model complex relationships in a computationally efficient way. 

For a user u and an item v, the preference score is:

$
\hat{r}_{u,v} = \langle \phi(u), \psi(v) \rangle
$

where \( \phi(u) \) and \( \psi(v) \) are the **embedding vectors** of the user and item, respectively. The goal is to approximate a large, sparse interaction matrix by learning these user and item vectors.

### **Dimensionality:**
Dimensionality in dot-product models refers to the number of latent factors used to represent user and item embeddings. The dimensionality (d) of these embeddings controls how well the model can capture relationships between users and items. More specifically,

	Higher dimensionality allows the model to capture more detailed and complex interactions, but at the cost of higher computational complexity. 
	If the dimensionality is too low, the models lose their ability to capture complex patterns in user preferences, leading to "bounded" expressiveness. For instance, when the dimensionality is reduced to one, all users and items are essentially reduced to scalars, and the predicted ranking becomes oversimplified (reflecting only popularity or its reverse).

Since there were some research gap considering the preliminary research explained below, this situation creates a motivation.

### **Preliminary Research:**
These embeddings can be learned using various models, such as:
   - **Matrix Factorization (MF)**: A basic dot-product model that directly computes user-item preferences.
   - **Neural Networks or Variational Autoencoders (VAEs)**: These can also be interpreted as dot-product models, where the preference is calculated via learned embeddings. VAEs use **fully-connected layers** and **activation functions** (e.g., softmax) to generate item scores. The final recommendation ranking is still determined by the order of the dot products between the user and item embeddings.

### **Research Gap:**
The research gap identified in the paper is that most prior studies on recommender systems focus primarily on improving ranking accuracy while overlooking the impact of dimensionality in dot-product models.

1. **Dimensionality is Underexplored**: While other hyperparameters like learning rates and regularization weights are frequently tuned, the dimensionality of user and item embeddings is often ignored or not studied in depth. Researchers haven't fully explored how dimensionality affects other important aspects of recommender systems such as diversity, fairness, and robustness.

2. **Limited Investigation of High-Dimensional Models**: Although high-dimensional models have shown improvements in ranking accuracy, there has been little examination of their potential to reduce biases like popularity bias or improve personalization and diversity in recommendations.

3. **Misconceptions About Overfitting**: It is often assumed that increasing dimensionality might lead to overfitting due to sparse data. However, some empirical results suggest that higher-dimensional models can still perform well, motivating the authors to investigate this further.

## **3. Method**  

This section presents a theoretical analysis focused on understanding the impact of dimensionality, particularly in the dot-product context, for recommender systems. The analysis considers the rank of the interaction matrix and the number of representable rankings constrained by dimensionality.

### Theoretical Investigation

Theoretical investigation explores **representable rankings** over item vectors in \( $\mathbb{R}^d$ \), highlighting the limitations and expressive power of low-dimensional spaces in representing these rankings.

#### Main Concepts

##### Representable Rankings
A ranking is representable if there exists a query vector \( $q \in \mathbb{R}^d$ \) such that the items are ordered based on the inner products \( $\langle q, v_i \rangle$ \), where \( $v_i$ \) are the item vectors.

##### Bounding the Number of Representable Rankings
The focus is on estimating how many distinct rankings can be represented with a set of item vectors in low-dimensional space:

- **Upper Bound**: For \( $n$ \) item vectors in \( $\mathbb{R}^d$ \), the number of representable rankings of size \( $K$ \) is at most \( $n^{\min(K, 2^d)}$ \). Increasing dimensionality enhances expressive power.
- **Lower Bound**: There exist \( $n$ \) item vectors such that the number of representable rankings in \( $\mathbb{R}^d$ \) is \( $n^{\Theta(d)}$ \), indicating expressivity increases with dimensionality.

#### Key Theoretical Insights

##### Dimensionality and Expressive Power
Higher-dimensional spaces allow exponentially more representable rankings, making the system capable of capturing more diversity and fairness. In contrast, low-dimensional spaces limit representability, potentially leading to insufficient diversity.

##### Popularity Bias
In low-dimensional spaces, rankings may become biased towards popular items, compressing the range of rankings and amplifying bias.

##### Structural Analysis of Representable Rankings
The investigation utilizes **hyperplane arrangements** and polyhedral representations (e.g., **polyhedral cones**) to characterize the structure of representable rankings and estimate bounds.

##### Failures in Low-Dimensional Spaces
Low-dimensional spaces may lead to overfitting to popular items and fail to represent diverse rankings, highlighting the need for higher-dimensional models or alternative approaches.

## Outline of the Proof

### Dot-Product Representation of Rankings
A ranking is derived from the inner products between item vectors \( $v_i \in \mathbb{R}^d$ \) and a query vector \( $q \in \mathbb{R}^d$ \). Items are ranked based on \( $\langle q, v_i \rangle$ \), and the number of rankings depends on how many ways \( $\langle q, v_i \rangle$ \) can be ordered.

### Geometric Interpretation via Hyperplanes
The query space \( $q \in \mathbb{R}^d$ \) is divided into regions by hyperplanes, where each region corresponds to a specific ranking. A **hyperplane** is defined by \( $\langle q, v_i \rangle = \langle q, v_j \rangle$ \), partitioning the space of query vectors into ranking regions.

### Upper Bound Using Hyperplane Arrangements
The number of regions formed by \( $n$ \) hyperplanes in \( $\mathbb{R}^d$ \) provides an upper bound on the number of distinct rankings. For ranking \( $K$ \) items, the number of regions scales as \( $n^{\min(K, 2d)}$ \).

### Lower Bound
A lower bound is demonstrated by constructing specific item vectors, showing that the number of representable rankings grows polynomially with dimensionality, \( $n^{\Theta(d)}$ \).

## Step-by-Step Breakdown of the Derivation

1. **Defining the Problem**: For \( $n$ \) items with vectors \( $v_1, v_2, \dots, v_n$ \) embedded in \( $\mathbb{R}^d$ \), a query vector \( $q$ \) ranks the items by sorting \( $\langle q, v_i \rangle$ \).
2. **Hyperplane Arrangements**: Each pair of items defines a hyperplane \( $\langle q, v_i \rangle = \langle q, v_j \rangle$ \). These hyperplanes partition the query space into ranking regions.
3. **Bounding the Number of Regions**: The number of regions in \( $\mathbb{R}^d$ \) is bounded by \( $O(n^{2d})$ \). For the top \( $K$ \) items, the number of regions is \( $O(n^{\min(K, 2d)})$ \).

## Example Scenario

Consider three items in a 2D space (\( $\mathbb{R}^2$ \)):

- Item 1: $( v_1 = (1, 0) )$
- Item 2: $( v_2 = (0, 1) )$
- Item 3: $( v_3 = (1, 1) )$

A query vector \( q = (q_1, q_2) \) ranks the items based on the dot products:

- $( \langle q, v_1 \rangle = q_1 )$
- $( \langle q, v_2 \rangle = q_2 )$
- $( \langle q, v_3 \rangle = q_1 + q_2 )$

The ranking depends on the values of $( q_1 )$ and $( q_2 )$.

### Geometric Interpretation Using Hyperplanes

Each pair of items defines a hyperplane:

- $( v_1 \) and \( v_2 \): \( q_1 = q_2 )$ (diagonal line)
- $( v_1 \) and \( v_3 \): \( q_2 = 0 )$ (horizontal axis)
- $( v_2 \) and \( v_3 \): \( q_1 = 0 )$ (vertical axis)

These hyperplanes divide the query space into regions, where each region corresponds to a different ranking of the items.

### Regions and Rankings

The query space \( $q \in \mathbb{R}^2$ \) is divided into four regions by the three hyperplanes, each corresponding to a different ranking. For example:

- Region 1: $( q_1 > q_2 > 0 )$ results in the ranking **Item 1 > Item 3 > Item 2**.

The hyperplane arrangements illustrate how dimensionality influences the number of possible rankings.


## **4. Experiment**

First, experiments were conducted to demonstrate the reasoning behind the theoretical analysis.

### **Personalization and Popularity Bias Results**

Personalization is a critical goal of recommender systems, and this experiment contrasts it with popularity bias (the system's tendency to recommend popular items).

- **Metrics Measured**:  
  - **Recall@K** (ranking quality)  
  - **ARP@K** (Average Recommendation Popularity, measuring bias)
  
- **Results**:  
  - **Low-dimensional models**:  
    - High popularity bias, recommending mainly popular items, leading to anti-personalization.  
  - **High-dimensional models**:  
    - **ML-20M** dataset: Ranking quality plateaued at low dimensionality (d = 1,024). Severe popularity bias due to long-tail items. A dimensionality of **d = 512** balanced ranking quality and bias reduction.
    - **MSD and Epinions**: Increasing dimensionality continually improved ranking quality. High-dimensional models significantly enhanced personalization and reduced popularity bias.

### **Diversity and Item Fairness**

- **Metrics Measured**:  
  - **Coverage@K (Cov@K)**: Measures catalog coverage.  
  - **Negative Gini@K**: Measures item fairness (distribution of item exposure).

- **Results**:  
  Models with similar ranking quality can have vastly different catalog coverage or fairness. Developers must avoid focusing solely on ranking quality as it could lead to non-diverse, unfair recommendations.

  - **High-dimensional models**: Increased both catalog coverage and item fairness.  
  - **Low-dimensional models**: Limited ability to generate diverse or fair recommendations.
  
![Effect of the dimensionality of iALS on catalog coverage and item fairness on ML-20M.](https://i.postimg.cc/fLWRgT68/temp-Imageo1-Y6d-I.avif)

### **Self-Biased Feedback Loop**

To capture the dynamic nature of user interests, a system typically retrains a model after observing data over a specified time interval. However, hyperparameters are often fixed during model retraining, as tuning them can be costly, especially when models are frequently updated. Thus, the robustness of deployed models (including hyperparameters) to dynamic changes in user behavior is critical. This is related to unbiased recommendation and batch learning from logged bandit feedback.

Since user feedback is collected under the currently deployed system, item popularity is influenced partly by the system itself. Consequently, when a system narrows down its effective item catalog, future data tend to concentrate on items that are frequently recommended. This phenomenon accelerates popularity bias in the data and further increases the number of cold-start items.

To observe the effect of dimensionality on data collection within a training and observation loop, they repeatedly trained and evaluated an iALS model with varying dimensionalities on ML-20M and MSD. Following a weak generalization setting, they first observed 50% of the feedback for each user in the original dataset. Then, they trained a model using all samples in the observed dataset and predicted the top-50 rankings for all users, removing the observed user-item pairs from the rankings. Subsequently, they observed the positive pairs in the predicted rankings as additional data for the next model training. During evaluation, they computed the proportion of observed samples for each user, termed "recall" for users, and also calculated this recall measure for items to determine how well the system can collect data for each item.
  
- **Results**:  
  - **High-dimensional models**: Achieved more efficient data collection for both users and items, especially for large catalogs like **MSD**.  
  - **Low-dimensional models**: Collected data predominantly from popular items, exacerbating popularity bias in future training.

![Effect of the dimensionality on data collection efficiency.](https://i.postimg.cc/j58SVYTt/temp-Image-YFCSSP.avif)

### **Summary of Empirical Results**

1. **Matrix factorization models** (iALS):  
   - High-dimensional models improved ranking quality and reduced popularity bias.

2. **Diversity and Fairness**:  
   - Low-dimensionality models restricted model versatility in terms of diversity and fairness, even with hyperparameter tuning.

3. **Self-biased feedback**:  
   - Low-dimensional models introduced significant bias in collected data, hindering future accuracy improvements.

### **Experiment setup**  
- **Datasets**:  
  The experiments were conducted on three real-world datasets:  
  - MovieLens 20M (ML-20M): A popular dataset used for movie recommendation tasks. The explicit feedback data was binarized by keeping user-item pairs with ratings larger than four.
  - Million Song Dataset (MSD): A dataset used for music recommendation. All user-item pairs were utilized without binarization.
  - Epinions: A dataset from a review platform. Only users and items with more than 20 interactions were retained, and the ratings were binarized like ML-20M.

- **Baseline Model**:  
  - **Implicit Alternating Least Squares (iALS)**, using a **block coordinate descent solver** for high-dimensional settings.

- **Evaluation Metrics**:  
  The focus was on evaluating how well the model generalizes to unseen user-item pairs in terms of **personalization**, **diversity**, **item fairness**, and **robustness to biased feedback**.
- **Recall@K** for ranking quality and personalization.  
- **Average Recommendation Popularity (ARP@K)** to measure popularity bias.  


### **Result**  

![Effect of the dimensionality of iALS on popularity bias in recommendation results.](https://i.postimg.cc/dVzCXdB4/temp-Image1ls-MG6.avif)
This figure illustrates the effect of the dimensionality of the iALS (implicit Alternating Least Squares) algorithm on recommendation results across three datasets: ML-20M, MSD, and Epinions. It highlight that increasing dimensionality improves recommendation quality but only up to a point before the improvements taper off.  


![The polyhedral approach for two different item sets](https://i.postimg.cc/x1bJK82H/temp-Imagek-Cnd-Kx.avif)
The polyhedral approach is used to examine rankings between two item sets: 
- $( P = \{ p_1, p_2, p_3 \} )$ (popular items)
- $( L = \{ l_1, l_2 \} )$ (long-tail items)

- **Blue and red lines** represent ranking regions where items in $( S(P, \{ l_1 })$ \) and $( S(P, \{ l_2 \}) )$ dominate the corresponding long-tail item.

This **geometric intersection** forms a convex cone, offering insight into how popular items outperform long-tail items under specific ranking conditions.


## **5. Conclusion**

This paper offers a valuable blend of theoretical analysis and practical relevance, providing a deep understanding of the limitations of dot-product models in recommendation systems and presenting actionable insights for their improvement. 

The core conclusion drawn from the empirical evidence is the importance of sufficient dimensionality in user/item embeddings to enhance the quality of recommender systems. Low-dimensional embeddings tend to amplify a popularity bias, widening the ranking gap between popular items and long-tail (less popular or niche) items. This effect is particularly notable because lower dimensionality limits the system's ability to represent diverse rankings, leading to a preference for more popular items, often at the expense of long-tail ones.

From a theoretical perspective, the power of dot-product models is constrained by the dimensionality of item factors and the inherent bias introduced by low dimensionality. Dot-product models with low-dimensional item factors are restricted by the rank of the interaction matrix and the number of representable rankings, which are governed by hyperplane arrangements in the latent space. As the dimensionality increases, the model can capture a wider range of interaction patterns, though this comes with the trade-off of increased complexity and the potential for overfitting.


### Applications and Open Problems
The results have practical implications for improving recommendation systems using higher-dimensional spaces or alternative methods. Future research could refine the bounds on representable rankings and explore methods for handling long-tail items in ranking models.


### Key Messages:
1. **Dimensionality Matters**: The dimensionality of item and query embeddings is critical to a ranking model's expressiveness. Higher-dimensional spaces enable more complex and varied rankings, whereas low-dimensional spaces limit the diversity of rankings, often leading to reduced catalog coverage and the reinforcement of popularity bias.

2. **Popularity Bias in Low Dimensions**: Low-dimensional spaces inherently favor popular items in the rankings. The study formalizes this through geometric interpretations, illustrating how a small set of popular items can dominate the rankings, crowding out long-tail items and causing the system to be inherently biased toward already popular items.

3. **Hyperplane Arrangements and Facets**: Rankings are shaped by the partitioning of the query space via hyperplanes. As dimensionality increases, so does the number of regions (and, thus, rankings) within the space. The number of facets on the polytope formed by item vectors directly impacts how diverse the rankings can be.

4. **Upper and Lower Bounds on Ranking Representation**: The study establishes upper and lower bounds on the number of representable rankings. In low-dimensional spaces, the expressiveness of the model is severely constrained, while higher-dimensional spaces allow for the representation of many more distinct rankings.

### Review and Opinion:
Many real-world recommender systems opt for low-dimensional embeddings to reduce computational costs, but this often comes at the expense of diversity and fairness in recommendations. The paper provides a clear geometric intuition for why low-dimensional spaces fail to capture complex ranking relationships, offering a compelling direction for improvement. The findings suggest that increasing dimensionality or adopting alternative models, such as graph-based methods, could help alleviate some of these issues.

The exploration of popularity bias through hyperplane arrangements is particularly insightful, as it links a common challenge in recommender systems to a formal mathematical framework. This approach could pave the way for new techniques that seek to balance the representation of popular and long-tail items in the rankings.

This study places a strong emphasis on understanding the expressive power of dot-product models, particularly when the dimensionality of the latent space is limited. The use of geometric and combinatorial methods (e.g., hyperplane arrangements, convex polyhedra) provides a robust theoretical foundation for the empirical observations made in the paper. Additionally, the research highlights how dimensionality directly affects the diversity and fairness of recommendations, offering clear insights into why certain systems may face limitations in this regard.

**Note**: In this review, I chose not to delve into the mathematical proofs and derivations, instead focusing on the practical implications and the overall contribution of the theorems to the experiments and the core ideas behind the paper.

---

## **Author Information**

* **Author Name**: Zeynep ALTINER
  * **Major**: Industrial and Systems Engineering
  * **Research Topic**: Multi-modality, multi-modal embedding
