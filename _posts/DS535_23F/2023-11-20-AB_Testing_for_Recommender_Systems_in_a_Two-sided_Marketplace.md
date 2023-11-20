---
title:  "[NIPS 2021] A/B Testing for Recommender Systems in a Two-sided Marketplace"
permalink: 2023-10-16-AB_Testing_for_Recommender_Systems_in_a_Two-sided_Marketplace.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **A/B Testing for Recommender Systems in a Two-sided Marketplace** 

## **1. Problem Definition**  

A/B Testing in the context of website or application development stands for testing some new feature(s) to users (both consumers or content producers) by randomly assigning different feature to a group of users. 

The principle commonly used in A/B testing is SUTVA (Stable Unit Treatment Value Assumption). It ensures that the feature changes made to one group will not have unintended effects on other groups. This rule is made so that the impacts generated only come from the changes being tested. 

Two-sided marketplaces can be defined as platforms that connect two groups of users; e.g. buyers and sellers, producers and consumers, content creators and viewers. E-commerce, freelance or job search sites, advertising, social media networks, and so on are examples of two-sided marketplaces. 

In two-sided marketplace environment, SUTVA is often violated due to the unique dynamics and interdependencies between the two user groups; the actions and behavior of one user group can directly affect the experience and outcomes of the other user group. Consider this case example. Instagram wants to try a new algorithm so that video contents with popular audio will be given more exposures. That way, viewers will have higher satisfaction to the videos suggested to them.  The new algorithm should only change the experience of users as viewers, but it indirectly affects the users who are also content creators, e.g. change the influencers' views and engagement. Consider the same example from the content creator's point of view. Such algorithm changes will encourage content creators to focus more on the audio used instead of the content's quality. In the end, this behavior can influence users' (viewers) experience due to decreasing relevance, content quality, and so on. 
  


## **2. Motivation**  

In two-sided marketplaces, changes happen very quickly and often. Developers need to choose the changes that will bring best output to the business, such as the highest user-click, purchase, dwell time, etc. Hence, it is very important to make sure that the A/B testing done brings an accurate result. Some popular approaches in experiment design for two-sided marketplaces are:
* Partitioning the graph into near separable clusters. 
This method works well in a sparse graph in which each cluster contains several nodes and they are given the same treatment. It is, however, difficult to apply in an ecosystem where the clusters are highly complex.
* Creating multiple copies of the universe by splitting the limited resources, e.g. daily budget of the entities one side of the graph. This method only works well in an ecosystem that has a periodic rest, such as advertising.
* Identifying a treatment allocated to a proportion of a node’s network to mimic the effect to the node’s entire network. This could work well on a denser graph but since the method assumes the propagation of an already known graph, it is not suitable for dynamic graphs

This paper overcome the limitations of previous approaches, proposing an experiment design that is not limited to the density of the graph, does not assume the propagation, and works without having to know the graph's structure _a priori_. 
Specifically, this paper focuses on the producer's or content creator's side,  unifying the different counterfactual rankings (hence “UniCoRn”) based on treatment allocation on the producer side, and careful handling of conflicts. 


## **3. Method**  
### Problem Setup
![Problem setup](https://i.ibb.co/H7x68KR/problem-setup.png)
Consider a bipartite graph linking two types of entities - producers and consumers. The traditional A/B testing could not satisfy the ideal condition for measuring he producer side impact as it depends on the consumers' treatment assignment and requires allocating the same treatment to all the consumers related to the producer being measured. 

**Notation and terminology**
Consider an experimental design $D$ with mutually exclusive sets
of producers $P_{0}, . . . ,P_{K}$ corresponding to treatments $T_{0}, . . . ,T_{K}$ respectively.
$T_{0}$ = control model (or recommender system)
$T_{K}$ = treatment model
$P_{K}$ = ramp fraction (i.e., treatment assignment probability) of the corresponding models
$S$ = Set of all sessions
$I$ = Set of all items
$I_{s}$ = subset of $I$ = a set of items under consideration in each session $s$
$R_{D}(i, I_{s})$ = the rank of item $i$ in the experimental design $D$
$i \in P_{k}$ = item $i$ belongs to a producer in $P_{k}$. 
$k, i, s$ = index for treatment variant, item, and session

**Design accuracy and cost**
Accuracy and cost are trade-off in the experimental design. Accuracy is defined by comparing the design rankings $R_{D}(i, \Gamma_{s})$ with the ideal ranking $R^*(i, \Gamma_{s})$ that equals $R_{k}(i, \Gamma_{s})$ if $i \in P_{k}$. 

Definition 1. The inaccuracy of the experimental design $D$ is given by
$Inaccuracy(D) := \mathbb{E} (R_{D}(i, \Gamma_{s}) - R^*(i, \Gamma_{s}))^2$, where $R^2(i,\Gamma_ {s}) = \sum^K_ {k=0} R_ {k}(i, \Gamma_ {s}) 1_ {\{i\in P_ {k}\}}$. 

Definition 2. Let $N_{D}(i, \Gamma_{s})$ denote the total number of times a scoring function (i.e. one of $T_{k}$’s) needs to be applied to to obtain a ranking of the items in $I_{s}$ according to $D$. The cost of an experimental design $D$ is given by $Cost(D) := \mathbb{E} (N_{D}(i, \Gamma_{s}))$

### UniCoRn: Unifying Counterfactual Rankings
1. The $UniCoRn$ algorithm
	A class of experimental designs $UniCoRn(P_{0}, P_{1},\alpha)$ paramerized by the tuning parameter $\alpha \in [0, 1]$ controlling the cost of the experiment. $\{R_{k}(i, \Gamma')\}$ denotes a ranking of the items in $\Gamma'$ according to $T_{k}$ in descending order. $R_{k}(i, \Gamma') \le R_{k}(j, \Gamma')$ for $k = 0, 1$.
	Algorithm 1:
![Algorithm](https://i.ibb.co/SxgXWwD/algorith.png)
	
	The key components are:
	* Initial slot allocation (Step 2)
	* Obtain mixing positions (Steps 3 - 5)
	* Perform mixing

	The $\alpha$ value was used in the design to control the balance between accuracy and cost, with $\alpha = 0$ incurs the highest inaccuracy but has the lowest cost, while $\alpha = 1$ is the most accurate and computationally expensive.

	**Analogy - LinkedIn scenario:**

	To allow easier understanding, the algorithm will be explained using this analogy: consider LinkedIn company wants to test whether a new modification in the recommendation algorithm can improve the visibility of job listings for certain companies on the LinkedIn job provider website.

	**Experiment Setup:**

	1.  **Control Group (A):** Job listings from companies in the control group receive the standard visibility on the platform (using existing algorithm without any modification).
	2.  **Treatment Group (B):** Job listings from companies in the treatment group that are given new algorithm to enhance visibility.
	3. **Producer Sets:** The producer sets $P_{0}​$ and $P_{1}$​ consist of companies in the control and treatment groups, respectively.
	4. **Scoring Models:** $T_{0}$​ and $T_{1​}$ represent the scoring models used to rank job listings for visibility in the control and treatment groups.
	5. **Tuning Parameter ($\alpha$):** The tuning parameter $\alpha$ determines the proportion of companies in $P_{0}​$ that will be included in the modified producer set $P_{0}^{*}$​ for the treatment.
    
	**Algorithm Application:**
	1. Obtain a ranking of all job listings in the session according to the scoring model $T_{0}​$ (standard visibility).
	2. Randomly select a subset of companies from the control group $P_{0}​$ with probability $\alpha$ to form $P_{0}^{*}$.3. Divide the job listings into sets $I_{s,0}$, $I_{s,1}$, and $I_{s,0}^{*}$ based on the producers in $P_{0}​$​, $P_{1}​$​, and $P_{0}^{*}​$.
	3. Identify the rank positions $L$ of job listings in $I_{s,1} \cup I_{s,0}^{*}$​ in the ranking obtained in step 1.
	4. Get rankings for job listings in  $I_{s,1} \cup I_{s,0}^{*}$ using scoring models $T_{0}​$ and $T_{1}​$​.
	5. Compute a rank-based score for each job listing based on the rankings obtained in step 5. The score considers whether the listing's company is in $P_{0}​$ or $P_{1}​$​.
	6. Rerank job listings in $I_{s,1} \cup I_{s,0}^{*}$​ in positions $L$ based on the computed rank-based scores. The reranking is done in ascending order of rank scores, with ties broken randomly.

	This algorithm enables following evaluation of the new treatment (algorithm) given on the selected LinkedIn website page: 
	- Analyze the impact of the treatment on the visibility and engagement with job listings from companies in the treatment group compared to the control group.
	- Assess whether the modified treatment at the cluster level ($P_{0}^{*}$) mimics the potential impact of applying the treatment to the entire control group ($P_{0}​$).

	This analogy example illustrates how the UniCoRn approach could be applied to A/B testing on LinkedIn's job provider website, considering the unique dynamics of a two-sided marketplace.

2. Handling multiple treatments
In this section, simultaneous measurement of multiple treatments (against a control variant) was addressed with a simple extension to the mixing selection step. The effect of each treatment was observed by independently comparing the corresponding treatment population to the control population.
Notations:
	* $T_{0}$ : control model with $K$ treatments in total  
	* $p_{k}$ : the ramp fraction of $T_{k}$, $\sum_{k = 0}^{K} p_{k} = 1$

	The extensions of Algorithm 1 are as follow:
	a. Greater mixing: all items from each $P_{k}$
It is suitable for applications without strict scoring latency constraints due to its costly yet higher accurate family of designs. 
The total computation cost = $c_{0} + (1 - (1 - \alpha)p_{0}) \sum_{k \ge 1} c_{k}$.

	b. Limited mixing: fractions of items from each $P_{k}$
	For applications with stricter latency requirement, this method has lower cost with higher inaccuracy compared to greater mixing. The total computation cost = $c_{0} + \alpha \sum_{k \ge 1} c_{k} + (1 - \alpha) \sum_{k \ge 1} p_{k}c_{k}$.

4. Theoretical results
**Theorem 1 Optimality of $UniCoRn(P_{0}, P_{1}, 1)$** 
Let $D_{U}$ be a design based on Algorithm 1 with randomly chosen $P_{0}$ and $P_{1}$, and with $\alpha = 1$. Then, for any other design $D$,
$\mathbb{E}(R_{D_{u}}(i, \Gamma_{s}) - R^* (i, \Gamma_{s}) \mid \Gamma_{s})^2 \le \mathbb{E} (R_{D}(i, \Gamma_{s}) - R^*(i, \Gamma_{s}) \mid \Gamma_{s})^2$   (Equation 1)
where $ \kappa = 1 $. Equation (1) implies the optimality of $D_{U}$ with respect to the $Inaccuracy(D_{U}, T_{0}, T_{1}) \le Inaccuracy(D, T_{0}, T_{1})$ for all $T_{0}, T_{1}$, and for all design $D$ 



	**Theorem 2 Bias and Variance Bounds**


	Let $D_{U}$ be a design based on Algorithm 1 with randomly chosen $P_{0}$ and $P_{1}$ and with $\alpha = 1$. Then, for  $k \in \{0, 1\}$, the conditional bias and the conditional variance of the observed rank $R_{D_{u}}(i, \Gamma_{s})$ given $A_ {s,k,i,r} = (\Gamma_ {s}, R^{\*} {i, \Gamma_ {s} = r, i \in P_ {k}})$ is given by 
			1. $\mid\mathbb{E} (R_{D_{U}} (i, \Gamma_{s}) - R^*(i, \Gamma_{s}) \mid A_{s,k,i,r}) \mid  \le c(k, p_{1})$, and
			2. $Var (R_{D_{U}} (i, \Gamma_{s}) \mid A_{s,k,i,r}) \le 2 min(r - 1, \mid \gamma_{s} \mid - r)   p_{1} (1 - p_{1}) + c(k, p_{1}) (1 - c(k, p_{1}))$.


## **4. Experiment and Result** 
In this paper, three empirical evaluations were done. For the first and second evaluations, a simulated environment was created with L = 100 positions to generate data for the empirical evaluation of $UniCoRn(P_{0}, P_{1},\alpha$) (in short, $UniCoRn(\alpha)$).

1.  Impact of $\alpha$
For a fixed treatment proportion $TP = |P1|/(|P0| + |P1|)$, the **inaccuracy and cost** results for different values of $TP$ were presented, while taking the average over random choices $P_{0}$ and $P_{1}.$ Four different simulation settings corresponding to different levels of correlation $p \in \{−1,−0.4, 0.2, 0.8\}$ between treatment and control scores were used for comparing the design accuracy, where scores were generated from a bivariate Gaussian distribution. 

	![img1](https://i.ibb.co/t8BsmHG/result-1.png)
	
	Two parameters were used to measure the design inaccuracy: i) mean absolute error (MAE) and (ii) root mean squared error (RMSE).

	Since $R_{D}(i, \Gamma_{s})−R^{*} (i, \Gamma_{s}) = 0$ for all items in $P_{0}$ for $UniCoRn(0)$, $UniCoRn(0)$ has slightly better than $UniCoRn(1)$  with respect to MAE but was outperformed with respect to RMSE. In addition, a smaller value of $p$ (where -1 is the smallest value) corresponds to a more challenging design problem due to the increasing number of conflicts in the counterfactual rankings. 
	The graph on the right showed the cost and inaccuracy trade-off for a fixed value of $p = 0.8$. It can be concluded that designing an experiment with a higher $T P$ is more challenging than one with a lower $TP$ due to the increasing number of conflicts in the counterfactual rankings. Meanwhile, experiments with a lower TP are more sensitive to the choice of $\alpha$.

2.  Analysis of the treatment effect estimation error
	![img2](https://i.ibb.co/D1p11CC/result-2.png)
	
	The proposed method was compared with previously existing methods with following parameters:
	* $HaThucEtAl$ where 10% of the population is in control and 10% of the population is in treatment. 
	*	$UniCoRn(\alpha)$ for $\alpha  \in \{0,0.2,1\}$ and $OASIS$ 
		*	(i) 10% treatment and 90% control and 
		*	(ii) 50% treatment and 50% control.

	The result showed that both $UniCoRn(\alpha)$ outperformed $OASIS$ (even for $\alpha = 0$) in terms of the treatment effect estimation error. $UniCoRn(1)$ and $UniCoRn(0.2)$ outperformed $HaThucEtAl$ with significantly lower variance due to $HaThucEtAl$ limitation to a 10% treatment and 10% control ramp. In terms of the computational cost (see graph on the right), $UniCoRn(0)$ was the cheapest method with $UniCoRn(0.2)$ slightly more expensive but ensure a good quality of estimation as $UniCoRn(1)$.

3. Scalability of UniCoRn
	Edge recommendations in social media platforms bring two sides of a marketplace together, consisting of "viewers" as the consumers and "viewees" as the candidates (i.e., items) recommended. To measure the viewee side effect, $UniCoRn(0)$ was applied to minimize the online scoring latency increase. Two experiments covering candidate generation and scoring stage experiments were conducted. Key metrics include (i) Weekly Active Unique (WAU) users, i.e., number of unique users visiting in a week; and (ii) Sessions, i.e., number of user visits.
	* Candidate generation experiment
		In large-scale recommender systems, $UniCoRn$ extends its capabilities to handle a two-phase ranking mechanism. In the edge recommendation, a common heuristic involves the **number of shared edges**, which benefits candidates with extensive networks. To mitigate this bias, the study tested the impact of a variant that employs a **normalized version of shared edges** using $UniCoRn(0)$, with $C_{0}$ using the number of shared edges and $C_{1}$ using the normalized version for candidate generation, while keeping the second phase ranking model consistent ($M = T_{0} = T_{1}$).	
	* Ranking model experiment
		In the ranking stage, candidates are scored based on viewer model assignments, with ranking models that can optimize for both viewer and viewee side outcomes; for instance, in one experiment, treatment model $T_{1}$ aimed to enhance viewee side retention. $UniCoRn(0)$ and candidate set $I_{s}$ allowed for cost-efficient rescoring of a subset of candidates belonging to the specific positions $L$, where $L = \{R_{0}(i, \Gamma_{s}) : i \in \Gamma_{s,1}\}$.
	* UniCoRn(0)’s implementation
		To generate the ranked list of viewees for a viewer, given the viewer is not allocated to UniCoRn, we score all items using the allocated model (i.e., viewer treatment). If the viewer is allocated to UniCoRn, we then obtain the viewee treatment allocations for all viewees. Obtaining the final ranking thus starts by scoring all items using a control model and then obtaining the viewee side treatment assignment for all viewees, and finally scoring each viewee with the necessary treatments and blend using the scores. No notable increase in serving latency is observed.
	* Results
		The table below shows the result on the viewee side where 40% of the viewers are allocated to $UniCoRn(0)$
			![Table 1](https://i.ibb.co/h9QbBD3/table-1.png)

		For each viewee, the response Yi is computed as the total count of the metric of interest in the experiment window, and the "Delta %" in Table 1 represents the relative percentage difference in average responses between the treatment and control viewee groups under the UniCoRn(0) design. The experiments indicated a positive change in weekly active users (WAUs) and sessions, aligning with the observation that viewers sent more invitations under the treatment model in a source-side experiment, suggesting a positive impact on viewees; however, the exact measurement fidelity couldn't be validated without a ground truth and is more accurately obtained through a traditional A/B testing setup on the viewer-side.
 



## **5. Conclusion**  
This paper proposed a new experimentation design mechanism $UniCoRn$ that went beyond the traditional A/B testing method for producer side measurement in two-sided marketplaces. In addition, it provides specific parameter to control the cost of experiment with regards to the accuracy in the measurement. $UniCoRn$ is superior than previous studies as it is implementable to any graph density, makes no assumption on the effect propagation nor needs knowing the graph structure a priori, and lowers the variance of measurement. It is inspiring to understand $UniCoRn$ addressing the difficulty in recommendations in the two-sided marketplace, which could have been easily under explored. 

This paper was implemented only on a case where user and producer are two unique entities. In cases where two sides of a marketplace are the same set of users just with different roles, the application might be different. For example, on social media like Instagram, an account operates as a viewer or audience and content creator at the same time. Future direction could address this application. Another suggestion would be to apply this method to more than two sides of marketplace, for example food delivery services that have seller, driver, and buyer as the entities.

---  
## **Author Information**  

* Karin Rizky Irminanda 
	* Affiliation: [Human Factors and Ergonomics Lab](http://hfel.kaist.ac.kr/)
	* Research Topic: VR/AR, ergonomics
	* Contact: karinirminanda@kaist.ac.kr

## **6. Reference & Additional materials**  

* Reference  
[Nandy, P., Venugopalan, D., Lo, C., & Chatterjee, S. (2021). A/b testing for recommender systems in a two-sided marketplace. _Advances in Neural Information Processing Systems_, _34_, 6466-6477.](https://proceedings.neurips.cc/paper_files/paper/2021/hash/32e19424b63cc63077a4031b87fb1010-Abstract.html)
