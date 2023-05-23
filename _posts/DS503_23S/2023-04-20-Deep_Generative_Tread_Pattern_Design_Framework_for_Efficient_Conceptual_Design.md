---
title:  "[ASME 2022] Deep Generative Tread Pattern Design Framework for Efficient Conceptual Design"
permalink: Deep_Generative_Tread_Pattern_Design_Framework_for_Efficient_Conceptual_Design.html
tags: [reviews]
use_math: true
usemathjax: true
---
# Deep Generative Tread Pattern Design Framework for Efficient Conceptual Design  

## 1. Introduction  
   The automotive sector has progressed exponentially in the past decade with exceptional designs and travel comfort. Many conventional ways exist already to develop designs that accelerate the goals and benchmarks of innovation. In the domain, along with other import systems, such as the suspension system, the tire tread patterns play a vital role in determining the car's overall performance, safety, and comfort. Designers and engineers strive to improve the existing designs to push the limits and improve the existing designs to facilitate the consumers. Though many conceptual design processes already exist, they have limitations, such as being time-consuming and repetitive. With the processes being already computationally expensive, a considerable human resource with prior knowledge is also required to carry out the whole design process efficiently. The manual work not only makes the whole process time-consuming but also hinders the speed of innovation. 

   To overcome these challenges, many AI approaches could benefit the problem. One of the most useful approaches is Deep Learning. Generative models are important deep learning methods that can be used as a design generation and design exploration method. Generative models are among the most famous and widely used deep learning methods. The working process includes learning a low dimensional representation of high dimensional data from data distributions of real data. GANs not only speed up progress by using advanced coding techniques but also helps remove the need for manual designing by the designers, which is exhausting and inefficient. Furthermore, using low-dimensional data as a latent vector reduces the computational cost and can be easily converted back to high-dimensional data. This study uses Generative Adversarial Network (GAN) as the generative model. 

<p align="center" width="100%">
    <img width="33%" src="https://github.com/SaniaShujaatt/Practice/assets/116706048/53a9742c-2738-47a7-8597-138a7e6098b7">
</p>
 
 							Fig 1. Concept Description of GAN

<p align="center" width="100%">
    <img width="33%" src="https://github.com/SaniaShujaatt/Practice/assets/116706048/014e1aa0-aee1-4f2d-be6f-cbb5496ac93f">
</p>

## 2. Motivation  

 In this study, the authors propose a generative design process that can automatically generate target-specified tread patterns. This proposed method defers conventional methods in that manual interaction exists between designers and engineers to develop a new design that satisfies engineering performance and aesthetics. Instead, the generative model automatically generates various diverse designs that satisfy both engineering performance and aesthetics as defined by the engineers and designers. The GAN integrates preprocessing of images, two-dimensional image-based tire performance evaluation functions, design generation, design exploration, and image post-processing methods to effectively implement the proposed idea.
The novelty of the approach lies in the optimization-based integrated functions that combine the generative design models with important tire-tread design evaluation functions. The detailed approach also focuses on the preprocessing of the data, design generation, and design evaluation processes. The following is a summary of the study's main contributions: • To automatically create several conceptual tread designs that satisfy the necessary tire performances, a deep learning-based tread pattern design system is proposed.
1.  To address the shortage of industry training data, a modified data augmentation strategy utilizing tread pattern domain knowledge is provided.
2.  A proposed integrated function is needed to achieve desired patterns in latent vector-based optimization, such as winter, summer, or all-season patterns. 
3.  Easy design exploration techniques are suggested for developing more varied pattern pictures. 
4.  A appropriate post-processing method is suggested to transform the low-resolution images created by the trained generator into realistic conceptual images with increased producibility.

<p align="center" width="100%">
    <img width="60%" src="https://github.com/SaniaShujaatt/Practice/assets/116706048/fa6508b1-3359-4322-af10-23a8326ed074">
</p>


						Fig 2. (2) Tread pattern description (b) types of tread pattern
## 3. Method  
   The framework of the proposed model is explained in this section. The overall important processes include image processing techniques, 2D image-based tire performance evaluation functions, ootimization algorithms. 

   The whole framework is divinded into four main stages,(data preparation, model construction, design selection, and final refinement). The details of each step is explained below.

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233889844-e8636ec6-f589-479c-8db1-cd5a3d086310.PNG">
</p>
						


						Figure. 3 Overall framework for the proposed model

 ### 3.1 Data Collection  
   For training of the generative model, 1500 images were gathered from different sources such as related market and industries and web browsing. These are the original tyre patterns and a suitable preprocessing method is essential since the available data has different image dimensions, color and contrast.

### 3.2 Pre-Processing  
To unify and refine the images, preprocessing techniques are applied. From a performance evaluation point of view, the image should be more than 64x512 (i.e., height-width) pixels to represent the minimum information in the tread pattern correctly. This information includes main grooves, blocks, and lateral grooves. Four phases make up the suggested preprocessing strategy:
1. The original photos are transformed into binary images using Otsu binarization [36].
2. The binary pictures of data types I and II are correctly cropped, and the vertical duplication and combination of the binary images of data type III. Then, all binary images are resized to 512 512 pixels to produce refined binary images. The first photographs of each data type in Fig. 4 produced the refined binary images in Fig. 5.
3. As shown in Fig. 5, the binary pictures are flipped horizontally and arbitrarily cropped from 512 512 pixels in a total unit to 64 512 pixels in a pitch unit for data augmentation.
4. Finally, 150,000 training images (i.e., 64 × 512 pixels in a pitch unit) are obtained for GANs training.
<p align="center" width="100%">
    <img width="60%" src="https://github.com/SaniaShujaatt/Practice/assets/116706048/0cf48b69-9464-4f92-8b5e-b43300a7c7c6">
</p>

					Figure. 5  Concept illustration for the proposed pre-processing strategy

### 3.3 Generative Models  
   The generative model for image generation is proposed in this step. GANs training learns the real data distribution, and the pattern can be expressed as:

$x=G(z)$                     

   Where z, x, and G are low-dimensional latent vectors, high-dimensional patterns, and the trained generator, respectively. The above model is obtained by combining and modifying DCGANs and LSGANs. The proposed generating model is found among several GANs, with the modification of advanced DCGANs [17] and LSGANs [18]. The proposed GANs models' detailed architectures (i.e., Figure 6 explain DCGANs (where the n latent vector is the latent vector's dimension and LSGANs are similarly constructed) using Nlatent vector-dimensional latent vectors. Deconvolutional and convolutional layers are represented by Deconv and Conv, respectively, in Fig. 6. Kernel size, output filter, stride, and padding are represented by K, C, S, and P. To reduce overfitting, leaky ReLU activation is swapped out for Mish activation [37], and a dropout layer [38] with a probability of 0.5 is used as a discriminator. One-sided label smoothing, which reduces the discriminator's real goal value from 1.0 to 0.9, is used for regular training. Adam optimizer is used in training with the learning rate set to 0.0002, 1 and 2 values set to 0.5 and 0.99, respectively. The maximum epochs allowed and the batch size are 200 and 256, respectively. As seen in Fig. 7, the trained generator may create various phony tread patterns after finishing GANs training.
	
<p align="center" width="100%">
    <img width="60%" src="https://github.com/SaniaShujaatt/Practice/assets/116706048/5a347976-ab5a-434e-96da-3c6dc3d298ff">
</p>

					Figure 6. Network architectures of the proposed GANs: (a) generator and (b) discriminator
### 3.4 Tire Performance Evaluation Function  
   The tire designs generated in the previous step are evaluated in this step. The 2Dimage based evaluation functions are derived using analytical approaches which maps between high dimensional data (e.g., the number of pixels and location of pixels), and tire performance.
    
    
    
   In this study, following are the selected design evaluation criterion for efficiency i.e., stiffness, hydro-planning, snow traction, groove wandering, and continuity performance. A evaluation function between high-dimensional data and tire performance is mapped which is expressed as
 $y = f(x)$
 
 
### 3.5 Design Generation  
   By using the trained generator in Step 3 and tire performance evaluation in step 4, intergrated functions are obtained. These functions are used for the purpose of mapping between latent vector and the tire perfromance. The intergrated function can be defined as

$(y = f(x)=f(G(z))=h(z))$
where h is the integrated function formed by combining G and f. To address a scale issue, all tire performances in this study are normalized to [0, 1]. 


<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233890084-d8ab8757-6c83-4d64-beb1-00a848ffd51c.PNG">
</p>

  						Fig. 7 Tread pattern images produced from the trained generator
	
When the optimization is performed using the above equation, it is possible to automatically obtain pattern images that satisfy the specified tire performances. In other words, latent vector-based optimization has the advantage of not requiring designers to create tread patterns or engineers to create form parameterization. The proposed design generation procedure consists of three steps:
1. Using generative methods, many tread designs are produced at random, and the tire performances of the corresponding tread patterns are calculated using performance evaluation functions.
2. Initial photos that satisfy the maximum permitted constraint violation constraints are filtered. These filtered photos serve as starting points for further improvement.
3. The desired optimum images are created using performance evaluation functions based on beginning locations. The pattern search algorithm is used for this study, but other optimization algorithms can also replace it.


### 3.6 Design Exploration  
 There are two simple design optimization exploration processes for the images. One is used to apply additional optimization using relaxed target continuity performance, and the second is the application of symmetry conditions to the optimized images.
A strict target community value negatively affects design diversity and produces similar images. Thus, a relaxed target continuity value is important to produce better and more diverse designs. Three steps comprise the suggested design exploration technique based on symmetry conditions:
1. Images are separated into two sections: left (LS) and right (RS).
2. Each region is subjected to axial symmetry (AS) and point symmetry (PS) modifications. As a result, four images (LSAS, LSPS, RSAS, and RSPS images) are generated, as illustrated in Fig. 10.
3. Images from among the examined designs that satisfy the necessary target tire performances are chosen, allowing for the creation of more diverse new pattern images that the generator cannot provide.

### 3.7 Post-Processing  
The images obtained through the previous steps need higher quality due to lower resolution. Appropriate post-processing is required to produce better results. An insufficient number of original images, pixel size, and difficulty of GANs training affect the complexity of the training process. Many techniques, such as Otsu binary images, Morphological operations, and pix2pix, are used to post-process the images.
1. Otsu binarization is applied to the selected photos to transform them into binary images.
2. Morphological operations are performed on binary pictures to produce refined binary images. The opening operation, an erosion followed by a dilation operation, reduces sounds among morphological processes [46]. In addition, any good techniques can be used to obtain more precise binary pictures.
3. Eight refined binary pictures are replicated and concatenated vertically.
4. Finally, pix2pix [47] is used to the vertically combined images (i.e., 512 512 pixels) to produce realistic final images.


## 4. Experimental Results and Analysis  

### 4.1 Model Implementation  
   The model utilizes two generator models, i.e., DCGANS and LSGANs, to generate tire tread images. The total training takes 24 hours on four TITAN RTX GPUs in parallel. MATLAB 2020b is used for the optimization, and the rest of the processes are implemented in PYTHON on Pytorch. The values of N latent vectors are set based on a parametric study. That training is set to 64 for DCGANs and 16 for LSGANs, respectively. To avoid dimensionality issues in the design generation phase, Ecvmax is set as 0% when the latent vector is 64 and 10% when the latent vector is 16. According to the paper, the higher the value of the latent vector, the greater the optimization efficiency is reduced, and vice versa.

### 4.2 Performance Evaluation of Proposed Model  
   The quantitative measures usually used to evaluate the performance of the models are Maximum Mean Discrepancy (MMD), reconstruction error, inception score (IS), and Frechet inception distance (FID). This study uses FID to evaluate the performance since FID compares the distribution between real and generated images. FID is consistent with human judgments, and a smaller FID value is used to achieve better performance. The following graph shows the FID values of both generators. FID is achieved using original data of 1500 images and 5000 generated images for every 50 epochs.


<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233889929-1e6379fe-6fa0-4403-97e0-338e08aa343b.PNG">
</p>

						Figure. 8 FID history plot for the proposed generator mode


   Another complicated issue with the GANs is the mode collapse, where the model only produces specefic images occurred in the model. The total images are classified into 5 main groups based on number of grooves. Figure below shows there exists no mode collapse since the proportions of the original and generated images tend to be similar. 


### 4.3 Design Evaluation of Tread Patterns  
   The results from the model are discussed in this section. The figure shows that images generated from the model consist of variation and diversity with number and shapes of main grooves, lateral grooves, and sipes.

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233889961-a4146ee3-c7f6-4a3a-8563-7e6a4009d9cc.PNG">
</p>


					Figure 9. Pie chart for proportions of tread pattern images: (a) original images, (b) generated images by DCGANs, and (c) generated images by LSGANs

The above figure represents images generated for different seasons, i.e., summers, winters, and all-season patterns. For each image, design performance evaluations, such as snow traction, hydroplaning, stiffness, continuity, etc., are shown. All the generated images satisfy the engineering performance criterion. 
One thousand five hundred photos are generated, and they are then manually categorized according to their shapes. There are five categories for original and produced photos, including five grooves, correspondingly, four grooves, three grooves, two grooves, one groove, and others. The original and generated data proportions are generally similar, as seen in Figure 9. As a result, the suggested models did not experience mode collapse. This study has the advantage of overcoming the mode collapse problem even though it has already occurred since it is possible to classify tread patterns according to tire performances utilizing a variety of performance evaluation functions.

### 4.4 Latent Vector Arithmetic
   This section deals with latent vector arithmetic for the tyre tread patterns. It is concluded in the paper that summer patterns have higher stiffness than winter patterns. Hence the summer designs are more prone to have main grooves and lateral grooves. The winter patterns on the other hand have higher snow traction, so the designs have higher ratio between lateral grooves and sipes. The latent vector serves as paraneters in the design domain that mainly defines the number, shape, and width of main gooves, lateral grooves, and sipes.  


<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233890000-f33a6442-6207-40de-a52d-e16fcce16793.PNG">
</p>

						Figure. 10 Latent vector arithmetic for tread pattern designs

### 4.5 Comparison Between the Conventional and Proposed Methods
   This section discusses the main differences between the proposed method by the authors and conventional methods. 

   1. The main difference is that the existing models require extensive repetition of tasks between the designers and engineers. Meanwhile, the proposed model requires no such repetition since the process is automated.

   2. The proposed design has an impressive 0.40s per design generation speed, which is impossible to achieve with conventional methods. Ten minutes are required to generate 1500 designs during the implementation of the proposed idea.

   3. The proposed design method is best for the conceptual design stage. Since, in the conventional methods, the engineers do not require shape parameterization in the optimization process. 

   4. Quality of the generated images is lower than the input images. Possible solutions exist to overcome this with the advancement of generative models. 
<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116706048/233890030-5d28c595-7401-489c-8051-6e2e7bc72cb6.PNG">
</p>

				Figure. 11 Final tread pattern images: (a) summer patterns, (b) winter patterns, and (c) allseason patterns
### 5. Conclusion and Future Work

   This study proposed a novel approach to automating a design framework that generates tire tread designs. The generated designs satisfy all the necessary tire performances, such as summer, winter, and all seasons. The proposed method consists of four stages (data preparation, model construction, design selection, and final refinement)  and seven steps(data collection, pre-processing, generative model, tire performance
evaluation function, design generation, design exploration, and post-processing). Finally, the model is numerically proven to generate realistic designs that satisfy the targetted engineering design variables.

   For the improvements and future work, the authors suggest a noise performance test between road and tire surfaces to evaluate performance. In addition, computer simulations could replace conventional analytical approaches to evaluate engineering performances. Moreover, increased image pixel dimensions could be another addition to future improvements. The higher quality will improve design aesthetics and the evaluation of engineering performance and make it more detailed. 


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
