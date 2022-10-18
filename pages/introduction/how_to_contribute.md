---
title: How to contribute?
sidebar: Introduction_sidebar
keywords: introduction
permalink: how_to_contribute.html
toc: false
folder: introduction
---


---

이 글에서는 리뷰 작성 방법에 관한 안내를 다룹니다.

This script is the guideline of how to write review.

## Preparing for your manuscript  

### **Fork repository**  

먼저, [github repository](https://github.com/DSAILatKAIST/DSAILatKAIST.github.io)를 자신의 github repository에 fork하여 추가합니다.  

At first, you should fork the [github repository](https://github.com/DSAILatKAIST/DSAILatKAIST.github.io) to make a repository on your account.  

<p align="center"><img width="1500" src="/images/fork.png"></p>

### **Find your review paper**
**'_posts'** directory에서 각자 맡은 논문을 확인하여 파일을 실행 시킵니다.

Check your assigned review papers on **'_posts'** directory and open this file.

### **Write reviews on .md format**  

리뷰를 작성할 때는 마크다운으로 작성해야 합니다. 마크다운이 처음이신 분들은 [Tutorial](https://www.markdowntutorial.com/)에서 연습하여 작성해주시길 바랍니다.   
또는 [StackEdit](https://stackedit.io/app#)에서 자유롭게 연습이 가능합니다.  

You should write reviews on markdown format.  
For the beginner of markdown, [Tutorial](https://www.markdowntutorial.com/) site is available for practice.


마크다운용 편집기 [Typora](https://typora.io/)을 활용하시면 편리하게 작성이 가능합니다.  
자신이 사용하던 편집기를 이용하여 작성하셔도 무방합니다.   

It is possible to write conveniently when you utilize the [Typora](https://typora.io/) edition.  
It is also okay to utilize your own edition.  

### **Attach image file**

리뷰 작성 시 사진을 첨부한다면, "/images/{논문이름}/{이미지파일 이름}.png"위치에 사진 파일을 넣어주세요.

If you want to attach an image file to elaborate your review paper, please put your image file on "/images/{Paper}/{Image}.png"

그 이후에 아래와 같은 명령어로 Image를 첨부 하실 수 있습니다.

Then you can load your image file as follows:


**E.g.**  
Paper title : GCN / Image file name : GCN_Encoder.png
``` bash  
<img width="140" src="/images/GCN/GCN_Encoder.png">  
```  

### **Rename your file**  

제출 날짜에 맞게 파일 이름을 변경해주세요. (YYYY-MM-DD-{Do not change})

Please Rename your file according to your submission data. (YYYY-MM-DD-{Do not change})

**E.g.**  

2022-10-13-A_User_Centered_Investigation_of_Personal_Music_Tours.md


### **Pull Request**  

제출을 원하는 경우에, fork한 레포지토리로 가서 pull request를 합니다.  
아래 그림과 같이 "Pull Requests"로 들어가서 "New pull request"를 클릭합니다.  

If you want to submit it, you should enter your repository which has been forked and pull request.  
As shown in above picture, go to "Pull Requests" and click "New pull request".  

<p align="center"><img width="1500" src="/images/pull_request.png"></p>

제출할 때 반드시! **gh-pages** branch로 제출해주시길 바랍니다.   
  
You should submit it to **gh-pages** branch.  

<p align="center"><img width="1500" src="images/branch.png"></p>

### **Review Format**  

예제 리뷰 포맷은 [Review Format](/pages/introduction/template.html)을 통해서 확인할 수 있지만 꼭 따를 필요는 없습니다.  

You can check the example review format on [Review Format](/pages/introduction/template.html) link but you don't have to follow this framework. 

### **Mathematical Equation (주의)**  
수학식을 Inline으로 작성할 때는 ```$$~$$```을 사용하고

display로 작성할 때는 ```$~$``` 를 사용해 주시기 바랍니다!

When you write the mathematical equation in inline math mode, you can write it by ```$$~$$``` expression,  
and if you write it in display math mode, you can write it by ```$~$``` expression.

