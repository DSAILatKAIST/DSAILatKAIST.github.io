---
title:  "[ASME 2022] Deep Generative Tread Pattern Design Framework for Efficient Conceptual Design"
permalink: Deep_Generative_Tread_Pattern_Design_Framework_for_Efficient_Conceptual_Design.html
tags: [reviews]
---

# Deep Generative Tread Pattern Design Framework for Efficient Conceptual Design

## 1. Introduction
    The automotive sector has progressed exponentially in the past decade with expectional designs and travel comfort. In the domain, tyre tread patterns play a vital role in determining the overall performance of the car. Designers and engineers strive to imporve the existing designs to push the limits. Though, many concpetual design process already exist but they possess certain limitations such as being time-consuming and repetitive. with the processes being already computationally expensive, a considerable human resource with considerable prior knowledge is also required to efficiently carry out the whole design process.

    To overcome these challenges, many AI approaches could prove beneficial to the problem. One of the most useful approach is the Deep Learning. Generative models are an important deep learning methods that can be used as a design generation and design exploration method.

![Tread](./images/tread.png)   
 

$$min_G  max_D V_G (D, G) = E_X∼pdata(x) [ log D(X)] + E_Z∼p_Z(z) [ log (1 − D(G(Z)))]$$
## 2. Motivation

    In this study the author proposes a Generative design process that can automatically generate target specefied tread patterns. The novelty of the appraoch lies in the optimization based integrated functions that combine the generative design models with import tyre tread design evaluation functions. The detailed approach also focuses on the preprocessing of the data, design generation and design evaluation processes.


## 3. Method
    The framework of the proposed model is explained in this section. The overall important processes include image processing techniques, 2D image-based tire performance evaluation functions, ootimization algorithms. 

    The whole framework is divinded into four main stages,(data preparation, model construction, design selection, and final refinement).
![framework](./images/framework.png)

                            Figure. 3 Overall framework for the proposed model

    The details of each step is explained below.


 ### 3.1 Data Collection
    For training of the generative model, 1500 images were gathered from different sources such as related market and industries and web browsing. These are the original tyre patterns and a suitable preprocessing method is essential since the available data has different image dimensions, color and contrast.

### 3.2 Pre-Processing
    TO unfiy and refine the images, preprocessing techniques are applied. From performance evaluation point of view,the image should be more than 64x512 (i.e heightxwidth) pixels to correctly represent the minimum information in the tread pattern. This information includes main grooves, blocks, and lateral grooves. 

### 3.3 Generative Models
    The generative model for the image generation is proposed in this step. GANs training learns the real data distribution and the pattern can be expressed as:

$$(x=G(z))$$                       

    where **z**,**x**, and G are low-dimensional latent vector, high-dimensional pattern, and the trained generator, respectively. The above model is obtained by combining and modifiying DCGANs and LSGANs.


### 3.4 Tire Performance Evaluation Function
    The tire designs generated in the previous step are evaluated in this step. In this study, following are the selected design evaluation criterion for efficiency i.e., stiffness, hydro-planning, snow traction, groove wandering, and continuity performance. A evaluation function between high-dimensional data and tire performance is mapped which is expressed as
 $$(y = f(x))$$


### 3.5 Design Generation
    By using the trained generator in Step 3 and tire performance evaluation in step 4, intergrated functions are obtained. These functions are used for the purpose of mapping between latent vector and the tire perfromance. The intergrated function can be defined as

$$(y = f(x)=f(G(z))=h(z))$$




![generator](./images/generator.png)

            Fig. 7 Tread pattern images produced from the trained generator

### 3.6 Design Exploration
    There are two simple design optimization exploration process for the images. One is used for application of additional optimization using relaxed target contnuity performance, second is the application of symmetry conditions to the optimized images.

    a strict target community value effects the design diversity negatively and produces similar images. Thus, a relaxed target continiuty value is important to produces better and diverse designs.

#### 3.7 Post-Processing
    The images obtained through the previous steps are lower in quality due to lower resolution. An appropriate post-processing is required to produce better results. Insufficient number of original images, pixel size, diffiulty of GANs training affects the complexity of the training process. Many techiniques such as Otsu binary images, Morphological operations and pix2pix are used for post processing of the images.


## 4. Model Proposed
    From the various available GAN models, the authors chose to modify well-developed DC-GANs and LSGANs to generate the proposed model. Following is the detailed architecture of the model
    
![gan](./images/GAN.png)
    

    The architecture is proposed with N latent vector - dimension and Deconv and Conv are deconvolutional and convulational layers, repsectively. K, C, P, and S represnt Kernel size, output filter, stride, and padding, respectively. The nonlinear activation function for the above is the Mish activation. To reduce overfitting of the model, a dropout layer with 0.5 probability is added. Adam optimizier is selected for the training with learning rate of 0.0002, and beta1 value as 0.5 and beta2 as 0.99. Meanwhile, batch size and epochs for the training are 256 and 200 respectively. GAN is able to generate images after completing the training. 


## 5. Experimental Results and Analysis

### 5.1 Model Implementation
    The model utilizes two generator models i.e., DCGANS and LSGANs to generate tyre tread images. The total training takes 24 hours on four TITAN RTX GPUs in parallel. MATLAB 2020b is used for the optimization and rest the processes are implemented in PYTHON on Pytorch. The values of N latent vectors are set based on a parametric study. using that training is set 64 for DCGANs and 16 for LSGANs, respectively. To avoid dimensionality issues in the design generation phase, Ecvmax is set as 0% when latent vector is 64 and 10% when latent vector is 16. According to the paper, higher the value of latent vector the greater the efficiency of optimization is reduced and vice versa.

### 5.2 Performance Evaluation of Proposed Model
    The quantitative measures usually used to evaluate performance of the models are Maximum Mean Discrepancy (MMD), reconstruction error, inception score (IS), and Frechet inception distance (FID). For this study, FID is used to evaluate the performance since FID compares the distribution between real images and generated images. FID is consistent with human judgements and to achieve better performance, a smaller FID value is used. The follwoing graph shows FID values of the both generators. FID is achieved using original data of 1500 images and 5000 generated images for each 50 epochs.


![graph](./images/graph.png)

    fig. 5 FID history plot for the proposed generator mode


    Another complicated issue with the GANs is the mode collapse, where the model only produces specefic images occurred in the model. The total images are classified into 5 main groups based on number of grooves. Figure below shows there exists no mode collapse since the proportions of the original and generated images tend to be similar. 


### 5.3 Design Evaluation of Tread Patterns
    The results from the model are discussed in this section. The figure shows that images generated from the model consist of variation and diversity with number and shapes of main grooves, lateral grooves, and sipes.

![performance](./images/perform.png)



    Fig. 13 Pie chart for proportions of tread pattern images: (a) original images, (b) generated images by DCGANs, and (c) generated images by LSGANs

    The above figure shows represent image generated for different seasons i.e., summers, wintes, and all-season pattern. For each image, design peformance evaluation such as snow traction, hydroplanning, stiffness, continuity etc are shown. All the generated images satisfy the engineering performance criterion. 

### 5.3 Latent Vector Arithmetic
    This section deals with latent vector arithmetic for the tyre tread patterns. It is concluded in the paper that summer patterns have higher stiffness than winter patterns. Hence the summer designs are more prone to have main grooves and lateral grooves. The winter patterns on the other hand have higher snow traction, so the designs have higher ratio between lateral grooves and sipes. The latent vector serves as paraneters in the design domain that mainly defines the number, shape, and width of main gooves, lateral grooves, and sipes.
![pattern](./images/pattern.png)

                    Fig. 17 Latent vector arithmetic for tread pattern designs

### 5.4 Comparison Between the Conventional and Proposed Methods
    This section discusses the main differences between the proposed method by the authors and conventional methods. 

        1. The main difference is that the existing models require extensive repition of tasks between the designers and engineers. Meanwhile the proposed model require no such repition since the process is being automated.

        2. The propsed design has an impressive 0.40s per design generation speed which is not possible to achieve in convenional methods. 

        3. The propsed design method is best for conceptional design stage. Since, in the conventional methods the engineers do not require shape parameterization in the optimization process. 

        4. Quality of the generated images is lower than the input images. Possible solutions exist to overcome this with advancement of generative models. 

![tyre](./images/tyre.png)
### 6. Conclusion and Future Work

    This study proposed a novel approach to automate design framework that generates tire tread designs. The generated designs are satisfying all the necessary tire performance such as summer, winter and all season. The proposed method consists of four stages and seven steps. The model is numerically proven to show that it generates realistic designs that satisfy the targetted engineering design variables.

    For the improvements and future work, the authors suggest noise performance test between road and tire surface to evalute performance. Computer simulations could replace the conventional analytical approaches to evaluate engineering performances. Moreover, increased pixel dimensions for the images could be another addition to the future improvements. The higher quality will not only improve design aesthetics also the evaluation of engineering performance and make it more detailed. 


## Author Information
    - Mingyu Lee 
    Department of Mechanical Engineering,
    Korea Advanced Institute of Science and Technology

    - Youngseo Park
    Department of Mechanical Engineering,
    Korea Advanced Institute of Science and Technology

    - Hwisang Jo
    Department of Mechanical Engineering,
    Korea Advanced Institute of Science and Technology

    - Kibum Kim
    Hankook Tire & Technology Co., Ltd.

    - Seungkyu Lee
    Hankook Tire & Technology Co., Ltd.

    - Ikjin Lee
    Department of Mechanical Engineering,
    Korea Advanced Institute of Science and Technology

## Reference

    https://asmedigitalcollection.asme.org/mechanicaldesign/article/144/7/071703/1131073/Deep-Generative-Tread-Pattern-Design-Framework-for