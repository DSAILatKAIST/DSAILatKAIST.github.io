---
title: How to contribute?
sidebar: Introduction_sidebar
keywords: introduction
permalink: how_to_contribute.html
toc: false
folder: introduction
---

이 글에서는 리뷰 작성 방법에 관한 안내를 다룹니다.  
This writing is the guideline of how to write review.  

### **Write reviews on .md format**  

리뷰를 작성할 때는 마크다운으로 작성해야 합니다. 마크다운이 처음이신 분들은 [Tutorial](https://www.markdowntutorial.com/)에서 튜토리얼을 진행하시거나 [Blog](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)에서 참고하실 수 있습니다. 리뷰는 [StackEdit](https://stackedit.io/app#)에서 실시간으로 미리보기를 하면서 작성하실 수 있습니다. 자신이 사용하던 편집기를 이용하여 작성하셔도 무방합니다.

작성하신 markdown(.md)파일은 KLMS를 통하여 제출하시면 됩니다. 

You should write reviews on markdown format. If you are new to Markdown, you can follow the tutorial in [Tutorial](https://www.markdowntutorial.com/) or refer to it in [Blog](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet). You can write the review while previewing it in real-time in [StackEdit](https://stackedit.io/app#). It is also okay to utilize your own edition.  

Please submit markdown(.md) file through KLMS.

제출 전 형식에 맞게 파일 이름을 변경해주세요. (`[CONFERENCE-YEAR]TITLE.md`)  
Please rename your file as follows: `[CONFERENCE-YEAR]TITLE.md`

**E.g.**  

[NIPS-22]Graph_Information_Bottleneck_for_Subgraph_Recognition.md

### **Review Format (Sample)**  
예제 리뷰 포맷은 [Review Format](/template.md)을 통해서 확인할 수 있지만 꼭 따를 필요는 없습니다.  

You can check the example review format on [Review Format](/template.html) link but you don't have to follow this framework. 

### **[Important] Precautions when writing equations**

Markdown 편집기([StackEdit](https://stackedit.io/app#))에서 잘 보이더라도, 블로그에서 수식이 깨지는 경우가 발생합니다. 
블로그에서 수식이 깨지는 경우를 방지하기 위해서 다음 주의사항을 유의해서 작성해주시기 바랍니다:  
Even if it looks good in the Markdown editor ([StackEdit](https://stackedit.io/app#)), equations may break in the blog. To prevent equations from breaking in the blog, please note the following precautions when writing:

- 수식을 작성하실 때 $ 하나만 사용해주시길 바랍니다. 본 블로그에서 \$$는 작동하지 않습니다.  
When writing equations, please use only one $ sign. $$ does not work on this blog.  

```
$$y=ax+b$$ (X)
$y=ax+b$ (O)
```

- 수식에서 아래첨자(subscript)를 표기할 때 다음과 같이 띄어쓰기에 유념해주시길 바랍니다.  
When indicating a subscript in an equation, please be careful of spacing as follows.  

```markdown
$N_{subscript}$ (X)
$N_ {subscript}$ (O)
```

- 수식 안에 바(bar)를 표기할 때 키보드에 있는 \|가 아니라 \vert로 작성해주시길 바랍니다.  
When indicating a bar in an equation, please write it as \vert instead of the | on the keyboard.  

```
$|x|$ (X)
$\vert x \vert$ (O)
```

### **Attach image file**

리뷰 작성 시 사진을 첨부한다면, 사진을 url형식으로 변환 후 마크다운에 첨부하시면 됩니다.   
사진을 url로 변환하는 방법은 github을 이용하면 편리합니다.  

You can insert your image file using the converted image url as follows :

1. 자신의 github에 빈 레포지토리를 생성하고, 새 파일을 생성합니다. (이 때, public 레포지토리를 생성해주세요.)  
Create an empty repository on your Github and create a new file.  
![image](https://user-images.githubusercontent.com/37684658/227445202-ef73cb4d-72bd-4229-ad57-88c4e96bf8c3.png)

2. 파일 이름을 수정하여 .md 확장자로 바꿉니다. (.md 확장자로 바꾸지 않으면 다음 step이 작동하지 않습니다.)  
Rename the file with .md extension. (If you don't rename it to .md extension, the next step won't work.)  
![image](https://user-images.githubusercontent.com/37684658/227445492-e8b49e2c-fac8-4ebf-9bea-27f50afa3f2d.png)  

3. 업로드하고자 하는 파일을 드래그앤드롭하면 사진 파일이 markdown용 URL로 변환됩니다.  
Drag and drop the file you want to upload, and the photo file will be converted to a URL for Markdown.  

![image](https://user-images.githubusercontent.com/37684658/227446034-b9dc9757-bb60-4fc2-9a16-e9d15578651b.png)

4. 변환된 URL을 review 본문에 복사하여 활용하시면 됩니다.  
Copy the converted URL and use it in yout review.  
![image](https://user-images.githubusercontent.com/37684658/227445726-3fe004f3-e32c-493e-90c2-8ea0080fe5b0.png)


**E.g.**
```markdown
### How to add images in Markdown?  
You can insert your image file using the converted image url as follows :
![image_sample](https://user-images.githubusercontent.com/37684658/227445939-ec25f692-3cd9-4adc-9eac-7d8daab3823e.png)
```

