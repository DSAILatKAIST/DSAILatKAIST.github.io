---
title:  "[ICML 2021] Generative Scene Graph Networks"
permalink: Generative_Scene_Graph_Networks.html
tags: [reviews]
---

# Paper Review

## Summary
The paper proposes an unsupervised method that uses hierarchical latent variable to recursively decomposes a scene into objects and parts. The leaf nodes correspond to primitive parts, and the edges represent the pose to compose parts into object recursively. The experiments are performed on two newly proposed datasets(2D Shapes and Compositional CLEVR). The model successfully breaks down scenes into meaningful objects and parts, and perform better than SPACE designed for non-hierarchical scene modeling.

## Strengths :
1. Despite the high level of difficulty involved in the method, the paper is well-written and easy to follow.

2. The motivation of the need for hierarchy is well-defined, and the method seems to be a reasonable for this motivation.

3. While there are no directly comparable methods on this task, the paper does a great job of comparing to the closest baseline, and show a clear improvement 

## Weaknesses :
1. The main concern is that the dataset was created deliberately, so it is uncertain whether the measures obtained using this dataset can be accurate indicators. Specifically, the each object in the dataset have a only single color and a simple shape, so it is not clear how well the method will perform in a realistic visual area.

2. The method involves selecting depth and out-degree as hyper parameters, which reflect strong prior knowledge about the data being modeled. But, in real-world scenario, it may be difficult to have such detailed knowledge of the prior structure. Hence, it is uncertain how the method would perform if the structural hyper parameters do not match the underlying statistics of the dataset.

3. The current state of the proposed method for unsupervised scene models suggests that it can only be applied to synthetic images. Moreover, the additional structural assumptions made on the data make it particularly difficult to identify appropriate real-world applications.

## Question
Could the current method be tested on a more realistic dataset? Additionally, it would be helpful to see the results of experiments with a more complex version of the current dataset.