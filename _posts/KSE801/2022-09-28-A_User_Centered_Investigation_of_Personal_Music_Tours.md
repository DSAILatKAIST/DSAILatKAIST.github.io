---
title:  "[Recsys 2022] A User-Centered Investigation of Personal Music Tours"
permalink: A_User_Centered_Investigation_of_Personal_Music_Tours.html
tags: [reviews]
---

# **A User-Centered Investigation of Personal Music Tours** 


## **1. Problem Definition**  

How to best present music to listeners and how can this be achieved automatically with algorithms? Playlists are the most commonly used way in many streaming platforms and can be generated automatically. Can the interestingness of the played music to the listener be improved by automatically generated tours, which contain side or background information of the songs?

## **2. Motivation**  

When listening to music, people are often interested in a smooth and harmonious sequence of songs which suits their taste. Additionally, many also want to explore new music. Several platforms as Spotify, Apple Music or YouTube offer the benefit of arranging music automatically, even with regard to personal preferences and many people seem to enjoy these services, which require almost no effort from the user. One problem in this field is how to present the music to the user in the most interesting way. The most common approach is to use playlists, which can take information about the artists, genres or topics into account. Yet, there exist other ways than playlists. A radio show for example often interrupts the continuous play of music for side information or other entertainments. By doing so, the experience for the listener gets more interactive than just listening to the music. 

Another alternative are **tours**, where songs alternate with segues. A segue is a spoken informative sentence about the realtionship between the previous and the upcoming song. By this, previously unrelated songs can be linked for the user, generating a more interesting experience. A playlist, in contrast, can hence be seen as a plain ordered list of songs. In some contexts, tours are found to be superior to playlist. The following figure compares the scheme of playlists with tours:

<img width="1400" src="/images/A User-Centered Investigation of Personal Music Tours/Screenshot (655).png">


The authors propose two algorithms to generate music tours, which aim to maximize the interestingness of the generated tours. They evaluate their results by interviewing participants of their study after their experience.

Previous works focused mainly on the task of recommendation; i.e. choosing the best upcoming song, or creating playlists with similarities between the songs (e.g. audio similarities like beats per minute, etc or web-based similarities like Wikipedia categories).
Whereas in this paper, the authors aim to maximize the interestingness of the tours and directly assess the experience of the users.
The songs to be included in the tours are here assumed to be given a priori.

## **3. Method**   

The two algorithms proposed are  "Greedy" and "Optimal" according to their behavior. They both try to find a permutation of the given songs and a list of segues, which maximize the interestingness of the tour. The songs and segues are represented in a knowledge graph. With a _segue_ function paths are found in this graph between the songs. The paths representing the background information link the songs and can then be translated to text form. The graph contains various information ahrvested from Musicbrainz and Wikidata. The information is divided into 30 different types of nodes like musical genres; locations,
e.g. where an artist was born; dates, e.g. when an artist was born; record labels, e.g. who published a song; and relationships
between artists, between songs, and between artists and songs. The full information about the graph can be found in the additional matrial, but for simplification the following figure shows a simple exmaple of a music knowledge graph from another paper: 

<img width="1400" src="/images/Music Graph.png">

The difference between the Greedy and Optimal algorithm is the approach of choosing the paths in the graph. Greedy is a heuristic, which chooses the next song according to the instrestingness of it segue, i.e. the song with the most intersting segue is iteratively choosen next. Therefore it uses the function _interestingness_. The interestingness is calculated based on the infrequency and conciseness of the knowledge graph's paths. That is, the more unique a relationship between two song is, the more interesting it is. On the other hand, if the path contains just very general information, e.g. about the same, broad genre, it should be considered as less interesting.
The Optimal algorithm builds the tour by  translating the problem to a Traveling Salesman Problem and solves it optimally. That is, the solution of Ptimally guarantees the best tour in terms of overall interstingness. Greedy can not guarantee that. 

An example for the results of the two algorithm gives the following figure. It shows the tour of Greedy and Optimal respectively, build on the same set of given songs.

<img width="1400" src="/images/Greedy Optimal tours.png">


## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

### **Result**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  



## **5. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

---  
## **Author Information**  

* GIOVANNI GABBOLINI, 
    * Affiliation: Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland
    * Research Topic: Recommender systems, Music information retrieval
* DEREK BRIDGE, 
    * Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland
    * Research Topic:  Case-Based Reasoning (CBR), Recommender Systems, natural language processing, machine learning, ant algorithms

## **6. Reference & Additional materials**  


* Reference: Giovanni Gabbolini and Derek Bridge. 2022. A User-Centered Investigation of Personal Music Tours. In _Sixteenth ACM Conference on
Recommender Systems (RecSys ’22), September 18–23, 2022, Seattle_, WA, USA.ACM, NewYork, NY, USA, 16 pages. https://doi.org/10.48550/arXiv.2208.07807
* Github Implementation: https://github.com/GiovanniGabbolini/play-it-again-sam
* Additional materials: https://zenodo.org/record/6817643 

