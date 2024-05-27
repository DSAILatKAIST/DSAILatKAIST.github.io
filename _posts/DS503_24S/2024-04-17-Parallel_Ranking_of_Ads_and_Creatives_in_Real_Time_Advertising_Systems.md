---
title:  "[AAAI 2024] Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems"
permalink: Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems.html
tags: [reviews]
use_math: true
usemathjax: true
---


# 1. Introduction

The introduction highlights the significance of online advertising for e-commerce platforms and the central role of creatives in engaging users. It addresses the challenge of selecting relevant creatives in real-time due to the large volume of AI-generated content. The paper proposes a novel parallel ranking architecture and an offline joint optimization model for ads and creatives, aiming to improve efficiency and personalization. Extensive experiments demonstrate the effectiveness of this approach in enhancing response time, click-through rate (CTR), and cost per thousand impressions (CPM) on real-world advertising platforms.

<img width="325" alt="image" src="../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Figure1.PNG">


$Figure 1$

The first column of Figure 1 shows the overall click-through rate (CTR) for three different creatives of the same product, demonstrating that even well-designed creatives can show poor performance. The second column presents gender-specific CTRs for different creatives of the same product, emphasizing the importance of personalized creative selection.

The research explores innovative methods to improve click-through rates (CTR) for advertising creatives by optimizing the advertising system's efficiency. It proposes separating the ranking of creatives from traditional ad searches to reduce latency. Key techniques include:

<ul>
<li> Peri-CR: Parallel estimation of creative and ad ranking to save time and enable sophisticated modeling.</li> 
<li> JAC: Joint optimization that models the interactions between ads and creatives.</li>
<li> Optimization of NSCTR: Enhancing the offline evaluation metric NSCTR to improve feedback tasks and online performance related to creative ranking.
</li>
</ul>

These techniques aim to improve the efficiency and effectiveness of creative ranking within advertising systems by leveraging advanced modeling and optimization strategies.

<br>

# 2. Related Work

The related work section of the paper discusses previous methodologies in advertising and creative ranking systems. It highlights the use of cascade ranking systems, which consist of recall and ranking stages for ad selection, and the challenges of creative ranking due to larger creative pools and sparser feedback. Traditional methods often separate ad and creative ranking, leading to inefficiencies. The paper identifies the need for integrated architectures to leverage deep learning models effectively, proposing a parallel ranking approach and joint optimization framework to address these limitations and improve system performance. Here's a brief overview of existing research in advertising ranking and creative ranking:
<ul>
<li>Advertising ranking: This involves techniques for selecting relevant items from a vast candidate pool using cascade ranking systems with recall and ranking stages. Researcher explore feature selection, interaction modeling, and user behavior modeling to enhance performance.
</li>
<li> Creative ranking: This area requires efficient online estimation due to large creative pools and sparse user feedback. Evaluating creative quality offline based on existing image and text content, it emphasizes online user feedback currently. Recent studies use methods like AES and HBM-VAM, employing tree structures and bandit approaches to optimize platform revenue.
</li>
</ul>
This research is similar to CACS in optimizing creative ranking prior to ad ranking and employing joint optimization with distillation and shared embeddings. However, researcher separate the creative module from the main ad pipeline and enhance creative estimation through parallel modeling, enabling accurate creative evaluation.

<br>

# 3. Problem Formulation

The paper formulates the problem of ranking ads and creatives to maximize effective Cost Per Mille (eCPM), which is determined by the product of Click-Through Rate (CTR) and Cost Per Click (CPC). $(eCPM = CTR · CPC · 1000)$
 
The objective is to rank ads with the highest CTR from thousands of candidates while minimizing response time. The problem formulation is as follows.

<b>1. Ad and Creative Sets:</b><br>

Ad $a$ as $({A_{1},A_{2},...,A_{M}})$ and creatives $({C^{1}_ {m},C^{2}_ {m} ,...,C^{n}_ {m}})$ are grouped, where each ad $(A_{m})$ has multiple creative options $(C^{n}_ {m})$. The goal is to display the top $𝐿$ ads and one optimal creative per ad.

<b>2. Ranking Model: </b><br>

The ranking model predicts the probability ($z$) of a user clicking on an ad-creative pair $(u,a,c)$. The model's prediction is represented as:

$z=p(y =1 \vert u,a,c) \quad(1)$

Here $p$ denotes the ranking model, and $y$ $\in$ ${0,1}$ indicates whether the user clicks the ad.

<b>3. Mathematical Transformation:</b><br>

To simplify the permutation problem, the model is decomposed as:

$p(y = 1 \vert x) \cdot p(c \vert x,y = 1) = p(c \vert x) \cdot p(y = 1 \vert x,c) \quad(2)$

Using the softmax function for click probability prediction:

$p(c \vert x) = \text{softmax} (\frac{ p(y \vert x, c)}{\sum_{c^{'} \in \mathcal{C}_ N} p(y \vert x, c = c^{'}_ {i})}) (3)$

The equation is further simplified to: 

$f_ {\theta}(x) · e^{−f_ {\theta}(x)} = p(y = 1 \vert x) \cdot p(c \vert x, y = 1) \quad(4)$

Here $f_ {\theta}(x)$ denotes $p(y = 1 \vert x,c)$. 

Thus researcher decompose the permutation problem into the CTR prediction of ad and the creative distribution with the user and ad. The formulation sets the groundwork for the novel parallel ranking approach and joint optimization model, aiming to enhance both the efficiency and effectiveness of real-time ad and creative selection.

# 4. Methology
Researcher propose Peri-CR, a novel architecture for creative and ad ranking, along with the corresponding model framework JAC, which improves the performance of both ranking tasks and reduces the overall response time.
<br>

## 4.1 Online Parallelism Architecture

The three methodologies for ad and creative ranking are as follows, visually represented in Figure 2:

![Figure2](../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Figure2.PNG)

$Figure 2$
The main modules and workflow of the online advertising system. (a) Multi-stage ad ranking system. (b) Pre-CR first
requests creative ranking then ad ranking. (c) Post-CR first requests ad ranking then creative ranking. (d) Peri-CR requests ad
ranking and creative ranking in parallel simultaneously.
<ol>
<li>Post-CR: This approach involves placing creative ranking after ad ranking to save time and enable more sophisticated modeling based on the results of ad ranking. However, since the ad ranking phase cannot determine which creative will be used for the ads, performance may suffer.
</li>
<li>
Pre-CR (CACS): In contrast to Post-CR, this method aims to enhance ad ranking performance by evaluating creative for all ad candidates. However, this approach requires significant time to evaluate all advertising candidates and restricts the creative model to a simple vector-based model.
</li>
<li>
Peri-CR: This proposed method separates the creative ranking module from the main ad retrieval and ranking pipeline, running them in parallel. The ad ranker uses abundant user and ad features to estimate ad-level Click-Through Rate (CTR), while the creative ranker uses fewer inputs to estimate creative-level CTR. The creative-level CTR then determines the optimal creative to display for each ranked ad, and the ad-level CTR determines the final displayed ad sequence. This parallel, decoupled architecture offers several advantages, including eliminating dependencies between the ad and creative modules and allowing sufficient time for both to use more sophisticated models.
</li>
</ol>
<br>

![Figure3](../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Figure3.PNG)

$Figure 3$ Framework of the proposed Joint optimization of Ad and Creative ranking(JAC), including two submodels: (a) Ad Ranking(AR) adopts deep cross network (DCN) as the main architecture to predict ad CTR, taking rich user and ad features as input. It also employs transformer to model user behavior sequence; (b) Creative Ranking(CR) uses a smaller network with fewer features, and leverages the AR output to estimate creative CTR. 

## 4.2 Offline Joint Optimizationn

Using AR (Ad Ranking) and CR (Creative Ranking) to model and predict CTR (Click-Through Rate) has limitations as follows:

1. AR: Biased CTR prediction due to lack of access to creative information.
2. CR: Less accurate CTR estimates due to simultaneous evaluation of multiple candidates.

To address this, a hierarchical model like the one shown in Figure 3, which integrates AR and CR offline, is proposed. This integration ensures mutual interaction between sub-models to enhance prediction accuracy. AR performs complex user behavior modeling using all features to predict ad-level CTR ($pctr_{ad}$), while CR estimates creative-level CTR ($pctr_{c}$) using simple features and rich creative characteristics through an MLP network.

Another input feature is AR pctrad, which undergoes log transformation and quantization before ad embedding. The quantized code is discretized by

$[ K \cdot \log_ {r+1}(1 + r · pctr_ {ad}) ]\quad(5)$

where K is the embedding size and r is a hyperparameter calculated for information gain. This embedding is initialized and connected with other feature embeddings as inputs to higher network layers. Since the quantization process (Equation 5) is non-differentiable, gradients cannot be directly propagated through it.

To handle the non-differentiable operation caused by quantization, gradients are directly copied from CR to AR during backpropagation. This allows CR to leverage information from AR to provide more accurate CTR estimates and maintain stability in CTR prediction during online inference.

By combining AR and CR into an integrated model to leverage their respective strengths, this approach enhances overall CTR prediction performance in advertising systems.

<br>

## 4.3 Quality of the Implicit Sub-Ranking

The performance of the creative ranking model is measured by designing NSCTR and discussing offline metrics. The Area Under the ROC Curve (AUC) is a common metric calculated for all exposures, but introduces user bias, while GAUC focuses on the performance of ad ranking lists for each user. Simulation CTR (sCTR) is a metric designed for creative selection, evaluating the difference between selected creatives from offline data and those exposed online. However, sCTR can be affected by changes in sample distribution and may cause variations in CTR when there are few creative candidates for specific ads.


## Algorithm 1: Evaluation Metrics - NSCTR

<img width="315" alt="image" src="../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Algorithm1.PNG"> 

<img width="375" alt="image" src="../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Figure4.PNG"> 

$Figure 4$
The results between offline metrics and online A/B CTR lift for 6 major creative ranking upgrades.

NSCTR is a metric that addresses these issues by approximating the CTR of samples where the creative selected by the offline model matches the creative exposed online using Algorithm 1. Validating the effectiveness of these metrics is a challenging task, with online A/B testing being the most definitive but time-consuming method to prove efficacy.

This underscores the importance of offline metrics that can be validated through correlation with results from online A/B testing. To demonstrate this, paper analyzed six significant improvements in creative ranking over the past year, including bandit algorithms, personalized two-towers, and creative feature extraction. Researcher computed four offline metric scores (AUC and GAUC for absolute lift, sCTR and NSCTR for relative lift) and their corresponding online CTR lifts for each improvement. Pearson correlation coefficients were high for NSCTR (0.988), AUC (0.741), and sCTR (0.636), but low for GAUC (-0.152). This analysis, depicted in Figure 4, indicates that NSCTR is a more reliable metric than others for the offline evaluation of creative ranking.
<br>

# 5. Experiments
In this section, Researcher conduct extensive experiments to evaluate the effectiveness and efficiency of the proposed framework.
<br>

## 5.1 Experiment Setup
<h2>1. Dataset</h2>
<ul>
  <li><strong>Test Date:</strong> 2023.05.01-2023.06.30 (total 60 days)</li>
  <li><strong>Training set:</strong> 18 billion</li>
  <li><strong>Test set:</strong> 300 million</li>
  <li><strong>User:</strong> 53 million</li>
  <li><strong>Ad:</strong> 16 million</li>
  <li><strong>Creative:</strong> 54 million</li>
  <li><strong>Overall CTR:</strong> 2.4%</li>
</ul>

<h2>2. Parameter Setting</h2>
<ul>
<li><strong>AR Model:</strong></li>
<ul>
  <li>30 users</li>
  <li>40 ads</li>
  <li>300 user-ad cross</li>
  <li>Embedding table: 2<sup>30</sup>x16</li>
  <li>Transformer: 2 layers, 2 attention heads</li>
  <li>DCN (Deep & Cross Network): 4 hidden layers (512x512x256x128)</li>
  <li>Activation Function: ReLU, sigmoid (0,1)</li>
</ul>

<li><strong>CR Model:</strong></li>
<ul>
  <li>11 users</li>
  <li>5 ads</li>
  <li>Creative ID, content</li>
  <li>Embedding dimension: 4</li>
  <li>MLP (Multi-Layer Perceptron): 3 hidden layers (128x64x32)</li>
  <li>Pctrad embedding size \( K \): 8192</li>
  <li>Dimension: 128</li>
  <li>Activation Function: ReLU, sigmoid (0,1)</li>
  <li>Batch size: 512 with Adagrad (0.05)</li>
  <li>Epochs: 1</li>
</ul>
</ul>

<h2>3. Evaluation Metrics</h2>
<ul>
<li><strong>Offline Evaluation:</strong></li>
<ul>
  <li>Simulated Click-Through Rate (sCTR)</li>
  <li>NSCTR (Normalized Simulated Click-Through Rate) for creative ranking</li>
  <li>ROC (Receiver Operating Characteristic) - AUC (Area Under the Curve)</li>
  <li>Group AUC (GAUC) for ad ranking</li>
</ul>

<li><strong>Online Experiments:</strong></li>
<ul>
  <li>Click-Through Rate (CTR)</li>
  <li>Response Time (RT) - Crucial for timely ad delivery</li>
  <li>Revenue per Mille (RPM)</li>
</ul>
</ul>

<h2>4. Baseline</h2>
<ul>
  <li><strong>No-CR:</strong> No creative, random candidate creatives</li>
  <li><strong>Post-CR:</strong> The creative module follows the ad ranking module and uses a two-tower model. Post-CR+ enhances this by integrating JAC-guided CR+ model.</li>
  <li><strong>Pre-CR:</strong> Creative module before ad ranking, using creative features in ad estimation. Ablation studies were conducted to assess the impact of JAC model on creative and ad ranking tasks.</li>
  <li><strong>Peri-CR:</strong> Involves parallel creative and ad ranking, with the creative model using an independently trained MLP network. CR+ and AR+ represent the creative and ad sub-modules within JAC.</li>
</ul>

## 5.2 Analysis
<h2>Hypotheses</h2>
<ul>
  <li><strong>H1:</strong> Peri-CR is expected to have lower response time compared to Pre-CR and Post-CR.</li>
  <li><strong>H2:</strong> Peri-CR is anticipated to achieve higher CTR and CPM than Pre-CR and Post-CR.</li>
  <li><strong>H3:</strong> JAC is expected to be most effective in both ad and creative ranking, achieving higher AUC and GAUC than AR, and higher sCTR than CR.</li>
  <li><strong>H4:</strong> JAC is predicted to enhance CR+ effectiveness, resulting in higher sCTR than CR without increasing time.</li>
  <li><strong>H5:</strong> JAC is expected to enhance AR+ effectiveness, resulting in higher AUC and GAUC than ARX` without increasing time.</li>
</ul>

<img width="315" alt="image" src="../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Table1_2.PNG"> 

<h2>Online Evaluations</h2>
To verify the effectiveness of proposed Peri-CR architecture on the overall ad system, researcher evaluated two key criteria - CTR and RPM. As presented in Table.1, Hypothesis tests based on the table are as follows:
<ol>
  <li>Peri-CR architecture improved CTR by 1.58% and RPM by 0.86% compared to Pre-CR, validating hypothesis H2.</li>
  <li>Peri-CR+ demonstrated performance similar to Post-CR+, confirming the effectiveness of offline JAC guidance in improving online estimation (hypotheses H4 and H5).</li>
  <li>Peri-CR+ exhibited the lowest response time (RT) comparable to no-CR, affirming that the architecture did not increase system latency as per hypothesis H1.</li>
  <li>The marginal differences between Peri-CR+ vs Peri-CR and Post-CR+ vs Post-CR revealed negligible computational overhead with offline JAC guidance, further validating hypothesis H4.</li>
  <li>The comparison of Pre-CR> Post-CR>no-CR demonstrated higher latency with more complex creative ranking methods.</li>
</ol>
<p>Peri-CR achieved optimal efficiency and effectiveness in the ad system evaluation, with potential for further improvement by adopting more complex feature extraction and model structures within the existing time budget.</p>

![Table3_4](../../images/DS503_24S/Parallel_Ranking_of_Ads_and_Creatives_in_Real_Time_Advertising_Systems/Table3_4.PNG)

<h2>Offline Evaluations</h2>
The sentences discuss a shift to offline experimentation due to online performance constraints, focusing on evaluating the JAC model's effectiveness in creative ranking. The validation results are presented in Table 3 and 4. Hypothesis tests based on the table are as follows:
<ol>
<li>In offline experiments, the JAC model demonstrated superior effectiveness in creative ranking compared to CR, CACS, and VAM-HBM, with the highest NSCTR score, confirming hypothesis H3.</li>
<li>CR+ improved NSCTR by 1.2% over CR and approached JAC's score, validating the effectiveness of the JAC network design in transferring capabilities from large models to smaller ones, confirming hypothesis H4.</li>
<li>Further investigations on ad ranking showed that JAC significantly outperforms in AUC and GAUC, demonstrating improved click-through rate prediction accuracy for ads, thereby confirming hypothesis H3.</li>
<li>AR+ also outperformed the baseline AR, validating that the offline large model can enhance the small model's ability to estimate ad click-through rates, confirming hypothesis H5.</li>
</ol>

# 6. Conclusion

The paper introduces a novel parallel ranking architecture and an offline joint optimization model for real-time advertising systems. This approach enhances the efficiency and effectiveness of selecting ads and creatives by decoupling the ranking processes and enabling sophisticated modeling. The proposed method significantly reduces latency and improves click-through rates (CTR) and cost per thousand impressions (CPM). Extensive experiments validate the superiority of the parallel ranking and joint optimization framework over existing methods, demonstrating notable improvements in both offline evaluations and real-world online advertising platforms.

# 7. Idea Proposal
Although online and offline experiments have proven the above method to be effective, delays and computational constraints have forced the abandonment of large-scale joint rankers for advertising and creatives, which has limited system performance. Moreover, improvements in hardware and system design seem to enable better joint ranking without significant ranking. NSCTR correlates well with online metrics, but it is only possible to approximate real sample distributions, and it will be possible to accurately recover real distributions by identifying mechanical variables for causal inference. Finally, this research system separates offline creative production from online estimation. Personally, if this utilize AIGC, I believe that it will be able to integrate them to derive optimal ads in real time for each user. Also, Incorporate real-time user feedback and interactions more effectively into the optimization framework to dynamically adjust and improve ad and creative rankings.

# 8. Author Information
Zhiguang Yang, Lu Wang, Chun Gan, Liufang Sang*, Haoran Wang, Wenlong Chen, Jie He, Changping Peng, Zhangang Lin, Jingping Shao
## Reviewer: N.Zuv-Uilst