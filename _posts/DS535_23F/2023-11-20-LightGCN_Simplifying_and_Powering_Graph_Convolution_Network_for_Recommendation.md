
# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation



# Background

## GCN for Collaborative Filtering

The collaborative filtering is widely used in recommendation systems, offering the advantage of not only utilizing item attributes for recommendations but also leveraging user-item interaction data to enhance personalized recommendations and address cold start problems. 

GCN leverage the graph structure of user-item interactions, where users and items are represented as nodes and interactions as edges. They aggregate information from neighboring nodes, allowing them to capture the preferences and similarities of users and items based on their interactions. Graph Convolution Network (GCN) has become new state-of-the-art for collaborative filtering [3]. 

Nevertheless, the reasons of the effectiveness of GCN for recommendation are not well understood. Existing work that adapts GCN to recommendation lacks thorough ablation analyses on GCN, which is originally designed for graph classification tasks and equipped with many neural network operations.

## NGCF 

NGCF (Neural Graph Collaborative Filtering) is a specific application of GCN to the task of collaborative filtering. Typically learning vector representations of users and  items lies at the core of modern recommender systems. Existing efforts typically obtain a user’s (or an item’s) embedding by mapping from pre-existing features that  describe the user (or the item), such as ID and attributes. The author argue that an inherent drawback of such methods is that, the collaborative signal, which is latent in user-item interactions, is not encoded in the embedding process [2]. As such, the resultant embeddings may not be sufficient to capture the collaborative filtering effect. The author of NGCF integrates the user-item interactions, specifically the bipartite graph structure, into the embedding process. 

Figure 1 illustrates the concept of high-order connectivity. The user of interest for recommendation is  u1, labeled with the double circle in the left subfigure of user-item interaction graph. The right subfigure shows the tree structure that is expanded  from  u1. The high-order connectivity denotes the path that reaches  u1  from any node with the path length *L* larger than 1. Such high-order connectivity contains rich semantics that carry collaborative  signal. For example, the path  u1  ←  i2  ←  u2  indicates the behavior similarity between  u1  and  u2, as both users have interacted with  i2; the longer path  u1  ←  i2  ←  u2  ←  i4  suggests that  u1  is likely to  adopt  i4, since her similar user  u2  has consumed  i4  before. Moreover,  from the holistic view of  l  = 3, item  i4  is more likely to be of interest  to  u1  than item  i5, since there are two paths connecting <i4,  u1>,  while only one path connects <i5,  u1>. 

![Example Image](https://i.ibb.co/F4M0jJj/User-item-Interaction.png)
Then NGCF leverages the user-item interaction graph to propagate embeddings as: 

![Example Image](https://i.ibb.co/C6n4GYR/equation1.png)

where $\mathbf{e}_ {u}^{(k)}\mathrm{~and~}\mathbf{e}_ {i}^{(k)}$respectively denote the refined embedding of user u and item i after k layers propagation, σ is the nonlinear activation function. $N_u$ denotes the set of items that are interacted by user u, $N_i$ denotes the set of users that interact with item i, and $W_1$ and $W_2$ are trainable weight matrix to perform feature transformation in each layer. By propagating L layers, NGCF obtains $L + 1$ embeddings to describe a user $(\mathbf{e}_u^{(0)},\mathbf{e}_u^{(1)},...,\mathbf{e}_u^{(L)})$ and an item $(\mathbf{e}_i^{(0)},\mathbf{e}_i^{(1)},...,\mathbf{e}_i^{(L)}).$  It then concatenates these $L + 1$ embeddings to obtain the final user embedding and item embedding, using inner product to generate the prediction score. 

The author conduct extensive experiments on three public benchmarks, demonstrating significant improvements over several state-of-the-art models like HOPRec [6] and Collaborative Memory Network [7]. Further analysis verifies the importance of embedding propagation for learning better user and item representations, justifying the rationality and effectiveness of NGCF. NGCF has become new state-of-the-art for collaborative filtering. 

## LightGCN 

Although NGCF has shown promising results, the authors of LightGCN argue that the reasons for GCN's effectiveness in recommendation are not well understood, and existing adaptations of GCN lack thorough analysis [1]. The author argues that its designs are rather heavy and burdensome — many operations are directly inherited from GCN without justification.  As a result, they are not necessarily useful for the CF task, and will bring no benefits, but negatively increases the difficulty for model training. To validate they thoughts, they perform extensive ablation studies on NGCF. 

![Example Image](https://i.ibb.co/D78hDmZ/ablation.png)

- NGCF-f, which removes the feature transformation matrices W1
and W2.
- NGCF-n, which removes the non-linear activation function σ.
- NGCF-fn, which removes both the feature transformation
matrices and non-linear activation function.

Throw ablation study, the author empirically demonstrates that the two common designs in GCN - feature transformation and nonlinear activation - contribute little to the performance of collaborative filtering and even degrade recommendation performance.  Therefore, by simplifying and improving the NGCF, they came up with a new model called LightGCN, which greatly improves the performance.

# Methodology

LightGCN aims to make the design of GCN more concise and appropriate for recommendation by removing unnecessary operations like feature transformation and nonlinear activation.  It learns user and item embeddings by linearly propagating them on the user-item interaction graph. The final embedding is obtained by taking the weighted sum of the embeddings learned at all layers. The difference between NGCN and LightGCN as shown follows. The primary distinction between NGCF and LightGCN lies in the fact that LightGCN excludes nonlinear activation $\sigma (.)$ and feature transformation $W_1$ and $W_2$. For LightGCN, the only trainable model parameters are the embeddings at the 0-th layer, i.e., $e^{(0)}_u$ for all users and $e^{(0)}_i$ for all items.

![Example Image](https://i.ibb.co/sQ9GK0k/image.png)
## Light Graph Convolution(LGC)
By removing nonlinear activation $\sigma (.)$and feature transformation $W_1$  and $W_2$, the graph convolution operation in LightGCN is simplified as:

![Example Image](https://i.ibb.co/DwsVZ6Q/equation2.png)

The symmetric normalization term $\frac{1}{\sqrt{\vert N_ {u} \vert}\sqrt{\vert N_ {i} \vert}}$follows the design of standard GCN.

## Layer Combination & Model Prediction
In LightGCN, the only trainable model parameters are the embeddings at the 0-th layer. When $u_i$ given, the embeddings at higher layers can be computed via LGC. After K layers LGC, they further combine the embeddings obtained at each layer to form the final representation of a user (an item): 
![Example Image](https://i.ibb.co/27Xwwdk/equation3.png)


where $α_k ≥ 0$ denotes the importance of the k-th layer embedding in constituting the final embedding. It can be treated as a hyperparameter to be tuned manually, or as a model parameter (e.g., output of an attention network) to be optimized automatically. The model prediction is defined as the inner product of user anditem final representations: 
![Example Image](https://i.ibb.co/zXrB28Y/equation4.png)

# Experiments

The experiment settings involve comparing the performance of LightGCN with NGCF (Neural Graph Collaborative Filtering) and other collaborative filtering methods. The datasets used in the experiments are Gowalla, Yelp2018, and Amazon-Book. The evaluation metrics used are recall@20 and ndcg@20. In addition to NGCF, two other methods are compared: Mult-VAE, which is an item-based CF method based on variational autoencoder, and GRMF, which smooths matrix factorization by adding a graph Laplacian regularizer.

## Datasets
The datasets used for the experiments are Gowalla, Yelp2018, and Amazon-Book as follows.
![Example Image](https://i.ibb.co/HCCjzxb/dataset.png)

## Performance Comparsion with NGCG
The comparison is done by recording the performance at different layers (1 to 4) and analyzing the training curves of training loss and testing recall. The main observations are as follows: 
- LightGCN outperforms NGCF by a large margin on all three datasets (Gowalla, Yelp2018, and Amazon-Book). The recall improvement is 16.52% on average, and the ndcg improvement is 16.87%. 
- LightGCN performs better than NGCF-fn, which is a variant of NGCF that removes feature transformation and nonlinear activation. LightGCN achieves better performance while having fewer operations than NGCF-fn. 
- Increasing the number of layers in LightGCN improves the performance, but the benefits diminish. The largest performance gain is observed when increasing the layer number from 0 to 1, and using a layer number of 3 leads to satisfactory performance in most cases. 
- LightGCN consistently obtains lower training loss than NGCF, indicating better fitting of the training data. This lower training loss translates to better testing accuracy, demonstrating the strong generalization power of LightGCN. 

Overall, LightGCN demonstrates higher effectiveness and better performance compared to NGCF.
![Example Image](https://i.ibb.co/Lks5N9n/result-compared-with-ngcf.png)


## Performance Comparsion with State-of-the-Arts

The performance of LightGCN is compared with other state-of-the-art methods, including NGCF, Mult-VAE, GRMF, and GRMF-norm. The results show that LightGCN consistently outperforms NGCF on all three datasets, achieving higher recall and ndcg scores. The improvement ranges from 12.79% to 16.56% for recall and from 13.46% to 17.18% for ndcg. Furthermore, LightGCN also outperforms other competing methods, including Mult-VAE, GRMF, and GRMF-norm, on all three datasets. This demonstrates the effectiveness of LightGCN's simple design. The ablation studies reveal that removing feature transformation and nonlinear activation in NGCF leads to consistent improvements in performance. This suggests that these operations might be unnecessary for NGCF. Overall, the performance comparison shows that LightGCN is a highly effective model for recommendation systems, outperforming NGCF and other competing methods on multiple datasets.
![Example Image](https://i.ibb.co/PtmW1t7/compare-with-other-method.png)

## Impact of Layer Combination

The impact of layer combination in LightGCN is analyzed as follows. The authors compare LightGCN with its variant, LightGCN-single, which does not use layer combination. They observe that the performance of LightGCN-single first improves and then drops as the layer number increases. This indicates that smoothing a node's embedding with its first-order and second-order neighbors is useful for collaborative filtering, but higher-order neighbors can lead to over-smoothing issues. On the other hand, LightGCN's performance gradually improves with the increasing number of layers. Even with 4 layers, LightGCN's performance is not degraded. This justifies the effectiveness of layer combination in addressing over-smoothing. However, the comparison between LightGCN and LightGCN-single showed that LightGCN consistently outperforms LightGCN-single on the Gowalla dataset, but not on the Amazon-Book and Yelp2018 datasets. This suggests that further tuning of the αk parameter can enhance the performance of LightGCN.
![Example Image](https://i.ibb.co/wQyVrzg/layer-combination.png)

## Impact of Symmetric Sqrt Normalization
The impact of symmetric sqrt normalization in LightGCN is analyzed in the study. Different choices of normalization schemes are explored, including normalization only at the left side, normalization only at the right side, and L1 normalization. The results show that the best setting is using sqrt normalization at both sides, as removing either side significantly drops the performance. The second best setting is using L1 normalization at the left side only. Normalizing symmetrically on two sides is helpful for sqrt normalization but degrades the performance of L1 normalization.
![Example Image](https://i.ibb.co/2M9c2rN/normalization.png)

# Paper Analysis
This paper prosposed LightGCN, a simple yet effective model for collaborative filtering. The authors conducted ablation studies to analyze the impact of different operations and found that removing feature transformation and nonlinear activation improved the performance significantly. They also compared LightGCN with other state-of-the-art methods and demonstrated its superiority in terms of recall and ndcg metrics on multiple datasets. 

However, the limitation of this study is that it does not explore the use of dropout mechanisms, which are commonly used in graph convolutional networks (GCNs) and Neural Graph Collaborative Filtering (NGCF). In GNNs, dropout can be particularly beneficial to enhance the performance and generalization power when dealing with graph-structured data[4]. The reason for not using dropout in LightGCN is that it does not have feature transformation weight matrices, so enforcing L2 regularization on the embedding layer is sufficient to prevent overfitting. However, NGCF requires tuning two dropout ratios and normalizing the embedding of each layer to unit length [3]. Additionally, the study found that learning the layer combination coefficients on training data did not lead to improvement, possibly because the training data did not contain sufficient signal to learn good coefficients that can generalize to unknown data. The study also tried learning the coefficients from validation data, which slightly improved performance. However, the exploration of optimal settings for the coefficients, such as personalizing them for different users and items, is left as future work.

# Impacts on New Research

This study courageously questions the suitability of the existing GCN structure for recommendations, even with encouraging results. The hypothesis is validated through comprehensive ablation experiments. The authors conducted an in-depth analysis to understand the problem's nature, encouraging new researchers to challenge authority and substantiate their assumptions with rigorous experiments. This also demonstrates that simplicity can be powerful. In an era where models are becoming increasingly complex and large, this idea is refreshing. Their research indicates that even simple models can achieve optimal results, provided they incorporate genuinely useful components.


## Author Information
### Auther Name:
 - Xiangnan He
 - Kuan Deng
 - Xiang Wang
 - Yan Li
 - Yongdong Zhang
 - Meng Wang∗

### Affiliation:

 - University of Science and Technology of China
 - National University of Singapore
 - Beijing Kuaishou Technology
 - Hefei University of Technology


## Reviewer  
> **Dewei Zhu**  
> **Industrial & System Engineering**  
>**Human Factors & Ergonomics Lab**  
>**Email**: deweizhu@kaist.ac.kr


# Reference Materials

 1. **LightGCN**：He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In _Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval_ (pp. 639-648).
2. **NGCF**：Wang, Xiang, et al. "Neural graph collaborative filtering." _Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval_. 2019.
3. **GCN**： Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. _arXiv preprint arXiv:1609.02907_.
4.  **DropGNN**：Papp, P. A., Martinkus, K., Faber, L., & Wattenhofer, R. (2021). DropGNN: Random dropouts increase the expressiveness of graph neural networks. _Advances in Neural Information Processing Systems_, _34_, 21997-22009.
5. **HOP-rec**： Yang, J. H., Chen, C. M., Wang, C. J., & Tsai, M. F. (2018, September). HOP-rec: high-order proximity for implicit recommendation. In _Proceedings of the 12th ACM conference on recommender systems_ (pp. 140-144).
6. **Collaborative Memory Network**：Ebesu, T., Shen, B., & Fang, Y. (2018, June). Collaborative memory network for recommendation systems. In _The 41st international ACM SIGIR conference on research & development in information retrieval_ (pp. 515-524).
