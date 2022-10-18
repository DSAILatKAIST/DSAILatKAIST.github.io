---
title:  "Graph Neural Controlled Differential Equations for Traffic Forecasting"
permalink: Graph_Neural_Controlled_Differential_Equations_for_Traffic_Forecasting.html
tags: [reviews]
---

# **Paper-Review of "Graph Neural Controlled Differential Equations for Traffic Forecasting"** 

## **1. Introduction**  

The spatio-temporal graph data frequently happens in real-world applications, ranging from traffic to climate forecasting. For instance, the traffic forecasting task launched by California Performance of Transportation (PeMS) is one of the most popular problems in the area of spatio-temporal processing.

For this task, a diverse set of techniques have been proposed. The most important is Neural controlled differential equations (NCDEs), which are a breakthrough concept for processing sequential data. 

In this paper, however, the author design a method based on neural controlled differential equations (NCDEs) for the first time.

## 2. Motivation

NCDEs, which are considered as a continuous analogue to recurrent neural networks (RNNs), can be written as follows:
$$
\begin{split}     		
		z(T)     		
		&=z(0)+\int_0^Tf(z(t);\theta_f)dX(t) ...(1)
		\\    		
		&=z(0)+\int_0^Tf(z(t);\theta_f)\frac{dX(t)}{dt}dt ...(2)
	\end{split}
$$
where $X$ is a continuous path taking values in a Banach space. The entire trajectory of $z(t)$ is controlled over time by the path $X$ (cf. Fig. 1). Leaning the CDE function $f$ for a downstream task is a key point in NCDEs.

<img width="450" src="images/Graph Neural Controlled Differential Equations for Traffic Forecasting/picfig2.png">

*Figure 1: The overall workflow of the original NCDE for processing time-series. The path $X$ is created from $\{(t_i; x_i)\}_{i=0}^N$ by an interpolation algorithm and therefore, this technology is robust to irregular time-series data*

The theory of the controlled differential equation (CDE) had been developed to extend the stochastic differential equation and the $It\hat{o}$ calculus far beyond the semimartingale setting of $X$. NODEs are a technology to parameterize such CDEs and learn from data. In addition, Eq. (2) continuously reads the values $\frac{dX(t)}{dt}$ and integrates them over time. In this regard, NODEs are equivalent to continuous RNNs and show the state-of-the-art accuracy in many time-series tasks and data.

However, it has not been studied yet how to combine the NCDE technology (i.e., temporal processing) and the graph convolutional processing technology (i.e., spatial processing). 

## 3. Body of the Paper

The authors integrate them into a single framework to solve the spatio-temporal forecasting problem.

They extend the concept and design two NCDEs: one for the temporal processing and the other for the spatial processing. After that, they combine them into a single framework. 

### 3.1 Main Process: graph+NCDE

The pre-processing step in their method is to create a continuous path $$X^{(v)}$$ for each node $v \in \nu$(where $$\nu$$ is a fixed set of nodes). For this, they use the same technique as that in the original NCDE design. Given a discrete time-series $$\lbrace x_i \rbrace_{i=0} ^N$$, the original NCDE runs an interpolation algorithm to build its continuous path. The authors apply the same method for each node separately, and a set of paths, denoted $$\{X^{(v)}\}_{v=1}^{\vert \nu \vert}$$(where $$\vert \nu \vert$$ is the number of locations to predict), will be created.

***Note: Each node in this graph generates a path, so a collection of paths is generated in the graph.***

The main step is to jointly apply a spatial and a temporal processing method to $$\lbrace X^{(v)}\rbrace_{v=1}^{\vert \nu \vert}$$, considering its graph connectivity. The authors then derive the last hidden vector $z^{(v)}(T)$ for each node $v$ and there is the last output layer to predict $$\hat{y}^{(v)} \in \mathbb{R}^{S \times M}$$, which collectively constitutes $$\hat{Y} \in \mathbb{R}^{\vert \nu \vert \times S \times M}$$. 

***Note: The path mentioned in the paper will control the final generated $z$***

### 3.2 Framework

In this paper, authors design a novel spatio-temporal model based on the NCDE and the adaptive topology generation technologies.

#### 3.2.1 Overall design

Their method includes one pre-processing and one main processing steps as follows: 

1. Its pre-processing step is to create a continuous path $X^{(v)}$ for each node $v$, where $1 \leq v \leq \vert \nu \vert$, from $\lbrace F_i^{(v)}\rbrace_{i=0}^N$. $F_i^{(v)} \in \mathbb{R}^D$ means the $v$-th row of $F_i$, and $F_i^{(v)}$ stands for the time-series of the input features of $v$. 
2. The above pre-processing step happens before training their model. Then, their main step, which combines a GCN and an NCDE technologies, calculates the last hidden vector for each node $v$, denoted $z^{(v)}(T)$.
3. After that, they have an output layer to predict $\hat{y}^{(v)} \in \mathbb{R}^{S \times M}$ for each node v. After collecting those predictions for all nodes in $\nu$, they have the prediction matrix $\hat{Y} \in \mathbb{R}^{\vert \nu \vert \times S \times M}$.

#### 3.2.2 Graph neural controlled differential equations

Their proposed spatio-temporal graph neural controlled differential equation (STG-NCDE) consists of two NCDEs: one for processing the temporal information and the other for processing the spatial information.

- Temporal processing

  The first NCDE for the temporal processing can be written as follows:

  $$
  	h^{(v)}(T) = h^{(v)}(0) + \int_0^Tf(h^{(v)}(t);\theta_f)\frac{dX^{(v)}(t)}{dt}dt ...(4)
  $$
  where $h^{(v)}(t)$ is a hidden trajectory (over time $t \in [0,T])$ of the temporal information of node $v$.

  Eq. (4) can be equivalently rewritten as follows using the matrix notation:
  $$
   	H(T) = H(0) + \int_0^Tf(H(T);\theta_f)\frac{dX(t)}{dt}dt ...(5)
  $$
  where $X(t)$ is a matrix whose $v$-th row is $X^{(v)}$. The CDE function $f$ separately processes each row in $H(t)$.

- Spatial processing

  After that, the second NCDE starts for its spatial processing as follows:

  $$
  	Z(T) = Z(0) + \int_0^Tg(Z(t);\theta_g)\frac{dH(t)}{dt}dt ...(6)
  $$
  where the hidden trajectory $Z(t)$ is controlled by $H(t)$ which is created by the temporal processing.

  After combining Eqs. (5) and (6), they have the following single equation which incorporates both the temporal and the spatial processing:

  $$
  	Z(T) = Z(0) + \int_0^Tg(Z(t);\theta_g)f(H(t);\theta_f)\frac{dX(t)}{dt}dt ...(7) 
  $$
  where $Z(t) \in \mathbb{R}^{|\nu| \times dim(z^{(v)})}$ is a matrix created after stacking the hidden trajectory $z^{(v)}$ for all $v$.

## **4. Experimental Evidence**  

- Datasets

  In the experiment, they use six real-world traffic datasets, namely PeMSD7(M), PeMSD7(L), PeMS03, PeMS04, PeMS07, and PeMS08, which were collected by California Performance of Transportation (PeMS) in real-time every 30 second and widely used in the previous studies.

- Experimental Settings

  * Parameters

  All existing papers, including their paper, use the forecasting settings of S = 12 and M = 1 after reading past 12 graph snapshots.

  In short, the authors conduct a 12-sequence-to-12-sequence forecasting, which is the standard benchmark setting in this domain.

  * Evaluation Methods

  They use the mean absolute error (MAE), the mean absolute percentage error (MAPE), and the root mean squared error (RMSE) to measure the performance of different models.

- Experimental Results

  Table 3 and 4 present the detailed prediction performance.

  Overall, their proposed method, STG-NCDE, clearly marks the best average accuracy as summarized in Table 2.

  <img width="550" src="images/Graph Neural Controlled Differential Equations for Traffic Forecasting/pictable2.png">

  *Table 2: The average error of some selected highly performing models across all the six datasets. Inside the parentheses, they show the others performance relative to their method.*

  <img width="750" src="images/Graph Neural Controlled Differential Equations for Traffic Forecasting/pictable3.png">

  *Table 3: Forecasting error on PeMSD3, PeMSD4, PeMSD7 and PeMSD8*

  <img width="450" src="images/Graph Neural Controlled Differential Equations for Traffic Forecasting/pictable4.png">

  *Table 4: Forecasting error on PeMSD7(M) and PeMSD7(L)*

  STG-NCDE shows the best accuracy in all cases, followed by Z-GCNETs, AGCRN, STGODE and so on.

  However, it is outperformed by AGCRN and Z-GCNETs for PeMSD7. Only their method, STG-NCDE, shows reliable predictions in all cases.


## **5. Conclusion**  

They presented a spatio-temporal NCDE model to perform traffic forecasting: one for temporal processing and the other for spatial processing. 

In their experiments with 6 datasets and 20 baselines, their method clearly shows the best overall accuracy. 

In addition, their model can perform irregular traffic forecasting where some input observations can be missing, which is a practical problem setting but not actively considered by existing methods. 

## 6. Paper Evaluation

The innovation in this paper is to propose the combination of graph convolutional network and neural ODE equation.

Especially in the process of converting dynamic graphs into time series, each point is converted into a sequence, and then ODE processing is performed on each sequence.

In fact, we can see that this approach is obviously unreasonable, resulting in invalid data storage, which can be optimized.

Moreover, the paper involves a large number of descriptions of partial differential equations, and there is not much content that is obviously useful.

---
## **Author Information**  

Authors: 

​	Authors: Jeongwhan Choi, Hwangyong Choi, Jeehyun Hwang, Noseong Park

Affiliation: 

​	Yonsei University, Seoul, South Korea

Comments: 

​	Accepted by AAAI Conference on ArtiAcial Intelligence pp.6367-6374 (2022) 

Cite As: https://doi.org/10.48550/arXiv.2112.03558

Review By: Hongxi Bai 20224287

Last Modify Date: October 16, 2022

## **7. Reference & Additional materials**  

* Github Implementation  

  https://github.com/jeongwhanchoi/STG-NCDE

* Datasets Used

  https://paperswithcode.com/dataset/pemsd8  