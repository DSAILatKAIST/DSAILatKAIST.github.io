# Private Data Leakage via Exploiting Access Patterns of Sparse Features in Deep Learning Recommendation Systems
##### [2022 Trustworthy and socially responsible Machine learning (TSRML 2022) co-located with NeurIPS 2022]

## Abstract

Sparse and dense features are used in the deep learning-based recommendation models to carry user's private information and this private data is often protected by the service providers using methods like memory encryption or hashing. In this paper, it is shown that irrespective of the protection used, the attacker would still be able to _learn information about which entry of the sparse feature is non-zero through the concept of embedding table access pattern_ posing a big threat to the security of customer's sensitive data.

## Problem Definition

Deep learning-based recommendation system models exploit different types of information related to the user including user attributes, user preferences, user behavior, social interaction and other contextual information to help the customers with better recommendations and companies with increased revenues. Now, there are two types of features as inputs to a deep neural network to make predictions of items a user may like, namely sparse and dense features.
>The sparse sparce features contain only a few non-zero features whereas the dense features contain a large number of non-zero attributes.

These features store the information of a user as well the items in different forms. Sparse features are the discrete and categorical attributes associated with users and items whereas the dense features are the continuous and numerical ones. These features contain information which is personal to the user and is protected by the service provider with the help of memory encryption with hardware such as Intel's SGX. However, we will look into some of the attacks an attacker may proceed with, with the purpose of stealing user's personal information where methods like encryption or hash functions may not be useful and information like _which entries of the sparse features may be non-zero_ can be leaked.
This is because sparse features have to be projected into the lower dimensional space through an embedding table where the index of the non-zero entries are used as an index for an embedding table lookup. It is shown in this paper, how this leakage could be enough threat to the sensitive information of the users.
>Embedding table is a data structure used to represent and store embeddings of these sparse features, and access patterns means to study how the users interact with the items. So, embedding table access patterns means accessing the patterns of the sparse features embedded into an embedding table.

It is demonstrated in the paper how it is possible to identify or extract sensitive information of a user with the help of embedding table access patterns under 4 different types of attacks:
* **Identification Attack**: identifying a user by with the help of combinations of unidentifiable features 
* **Sensitive Attribute Attack**: identifying a user by analyzing the user-item interaction behavior
* **Re-Identification Attack**: identifying a user by tracking the same user by analyzing their interaction history.
* **Hash Inversion with Frequency-based Attack**: showing how hashing the sensitive information may not be able to protect it against the attacks, by demonstrating a hash inversion attack based on access frequency.


## Working of the DLRM Model and Threat Model

The figure above shows the operation of the representative recommendation model, DLRM. In DLRM, the dense features go through a bottom MLP (multi-layer perceptron) layer whereas the sparse features go through an embedding table layer and get converted into lower-dimensional dense features. Then, these two outputs go through a feature interaction layer and then through a top MLP layer to predict the likelihood of an interaction.
Embedding tables play a pivotal role in transforming the sparse categorical feature into a dense numerical representation. Let's consider an example to understand this. Consider a scenario where users and movies are represented by categorical features such as user IDs and movie genres, respectively. To effectively utilize these features within a deep learning model, embedding tables are employed. These tables act as large lookup tables, with each row corresponding to a unique category or ID. To convert sparse features into dense representations, a lookup operation is performed using the non-zero entries in the sparse feature as an index. For instance, to convert a specific user's ID into a dense representation, the corresponding row in the user embedding table is accessed, and similarly, for movie genres, the relevant row in the genre embedding table is retrieved. The outcome of these operations is a dense vector, a numerical representation of the user or genre in a multi-dimensional space. These dense representations are subsequently utilized in the recommendation system's deep learning model, enabling accurate and personalized movie recommendations based on user preferences and movie genres. This process highlights the critical role of embedding tables. 
#### Threat Model
Now, even when the entire dense and sparse features are fully encrypted and are processed under a secure environment, there is a possibility to learn which index holds a non-zero entry by looking at the table access patterns, resulting in compromising with the sensitive user data.
To understand the threat model, let's assume the scenario. Let's say a user shared their sensitive information with the service provider to get accurate recommendations from the system. Now, this sensitive information is fully protected with the Intel SGX team, but the access pattern of the embedding table is revealed, more specifically revealing which entries of the table are non-zero. The figure below, demonstrates our threat model.
Like this, even when the information is kept safe with the honest-but-curious service provider, the access pattern of the embedding table can help reveal that sensitive information. 

## Overview
As also mentioned earlier, we will test out our theory of being able to extract sensitive information with the help of embedding table access patterns, with the help of four different types of attacks.

| Attack | Goal | Assumptions | Evaluation Metric|
|----------|----------|----------|----------|
| Identification | Finding the identity of the users | Attacker observes accesses, Has prior knowledge about distribution of accesses | K-anonymity |
| Sensitive Attribute | Extracting the sensitive user attributes | Attacker observes accesses, Has prior knowledge about distribution of accesses | Ambiguity |
| Re-Identification | Tracking users over time based on interaction history | Attacker observes accesses | Precision and Recall |
| Hash Inversion with Frequency-based Attack | Finding users raw feature values | Attacker observes accesses, Has prior knowledge about distribution of accesses | Inversion Accuracy |

This table shows a summary of all the attacks, their goals and basic assumptions which have been used to prove the point of the sensitive information not being safe. Each of these attacks are discussed in detail.


## Identification Attack with Static User Feature

User profile attributes, such as gender, city, etc. are usually static in nature i.e., they don't change with time. or the frequency of change is very low. We can categorize such features into two parts - identifiable and unidentifiable features.
> * Identifiable features are the features that are capable of directly or indirectly revealing the identity if the user. For example, name, city of residence, userID, etc.
> * Unidentifiable features are the features that can't directly expose the identity of the user but can still provide valuable information. For example, gender, education level, search keywords, etc.

Now due to strict regulations, most of the recommendation systems don't usually collect and use the identifiable features. So, the question arises that are the unidentifiable features enough to make an accurate assumption of who the user might be?

#### Evaluation setup
To find out whether the unidentifiable features are enough to find out about the user, an open-source dataset by Alibaba has been used, containing static user features, such as userID, groupID, gender, age group, shopping depth, occupation, city level, etc. of around 11.4M users.

#### Attack Method
After removing all the identifiable features from the dataset, we will be left with 2.1M possible combinations of the unidentifiable features, which would make any user believe that their identity is anonymous or that revealing any of the remaining unidentifiable features, won't reveal their identity. However, in contrast to the user's belief, it is observed that in the real world only 1120 combinations of these static feature values are possible based on the real open-source data. We refer to these 1120 combinations as user buckets.
> In simple words, user buckets are unique combinations of the feature values. The motto of the attacker can now be said to be able to recognize the users based on their unique combinations of features.

Taking the user bucket number as the x-axis and the percentage of the users per bucket as the y-axis, we plot the histogram to represent how the user distributions follow the long-tail pattern.
I can be seen from the histogram that there are only a few users in the bucket 600-1120 and in fact there are only 989 users on average across all these buckets and the last 56 buckets have only 1 user. So, these seemingly unidentifiable features may give away the user's identity by allowing the attacker to launch an identification attack to extract the unique userID and identify the user with a high certainty.

#### Evaluation Metric
The evaluation metric we have used for this analysis is the _K-anonymity_.
> K-anonymity is a privacy property that measure how well the user privacy is preserved.

If a user's bucket number is revealed and there are K users in the same bucket then the probability of finding the user is 1/K.
The table given below summarizes the number of users with anonymity level below K in the identification attacks.
1-anonymity user means that this is the only user having this particular set of feature values.


#### Evaluation Results
As shown in the above table, among 56 of the bucket users, there is only 1 user with the specific combination of static features which implies that an attacker can identify these users with 1-anonymity if they can observe this combination of feature values.



## Sensitive Attribute attack by Dynamic User Features

In this type of attack, we would see that even when the user's hide, how the sensitive attributes such as age, gender, interest, etc. can be inferred by analyzing their user-item interaction behavior and how these sensitive features leak through other non-sensitive features through the concept of cross-correlations.

#### Evaluation setup
For the purpose of evaluation, we have used the Alibaba Ads Display dataset, which contains user-item interactions. This dataset contains around 723,268,134 tuples and each tuple contains information about the userID, categoryID, brand and btag (browse, cart, favour, buy)

#### Attack Method
Let's understand the attack method with the help of an example, say in the data, there are 7 age groups and 5 different brands. The user-item interaction based on the age-group and the brand is given below in the connection graph.
Now, from the above graph a basic idea of the people belonging to a particular age group can be made. The user may not want to reveal their age, but the adversary may deduce their age with a high probability based on the type of brand the user has interacted with.
In general, we can say that the attacker uses their prior knowledge o popularity of the items between different demographic groups. Then, based on this prior information, they link the query to the demographic who formed most of the accesses to that item. The task of knowing the prior information is not that big of a deal.

#### Evaluation Metric
The evaluation metric we have used for this analysis is called _ambiguity_. It helps in determining the likelihood with which an adversary fails to predict a user's static sparse feature by just viewing their interactions with items. The ambiguity for each item is defined as follows:
_ambiguity(i) = 100% - max(frequency(i))_
where, frequency = distribution vector of all accesses to brand i by different user groups.
> An ambiguity(i) = 0 indicates if a user has interacted with item i, the attacker can successfully determine the user's sparse features.

#### Evaluation Results
In the graphs shown below, the x-axis shows the percentage of ambiguity where a value of 0 indicates that there is no ambiguity and this brand is always accessed by only one user bucket, whereas a higher ambiguity value depicts that brands are more popular across multiple user buckets.
In figure 5(A), it ca be seen that the more than 17% of brands are only accessed by 1 user bucket represented by the leftmost tall bar of PDF, meaning that the attacker can determine the user bucket using those brand interactions. On the other hand, in the CDF curve, for 38% of the brands, the attacker can predict the user bucket with a success rate of greater than 50%.
Similarly, the age and gender group versus the ambiguity are shown by graph 5(B) and 5(_C_) respectively. 


## Re-Identification Attack

In the re-identification attack, the attacker focuses on tracking the same user over time by just analyzing their interaction history.
> Re-identification attack is different from the identity resolution attack, as in the identity resolution attack the aim is to link the users across different system, potentially involving cross-referencing user's information.

Under this attack, we study two important things:
* if the history of the purchases of a user can be used as a tracking identifier for the user.
* if an attacker can re-identify the same user who sent queries over time by only tracking the history of their purchases, with no access to the static sparse features.

#### Evaluation setup
For the evaluation, we have used the Taobao dataset, and have separated about 9M purchase interactions among more than 723M user-item interactions. Then they have formatted the data into a time-series data structure, as shown below:
_user1 = (time1,item1), (time4,item10), (time500,item20)_
_user2 = (time3,item100), (time20,item100)_ 
.
.
.
.
_user_X = (time5,item75), (time20,item50), (time100,item75), (time400,item1), (time420,item10)_
Now for each set of consecutive items purchased by an user, we create a list of users who have the same set of consecutive purchases in exactly that order. We refer to these sets of consecutive recent purchases as _keys_.
> Multiple users may have the same key.

The goal of this attack is to use _m_ most recent purchases mad by a user to track them across different interactions sessions. To evaluation setup of this attack can be carried out as follows:
* randomly select a timestamp and a user
* for that selected user, we check _m_ most recent purchases at that selected timestamp and form a key
* we then look up this key in the recent item purchase history dataset
	* if the same sequence of m most recent items appear on another user at the same timestamp, this means those recent purchases are not unique or that specific user at that time and hence, doesn't represent a single user.
	* if the m items purchase history only belongs to that specific user, the duration of the time in which this key forms the most recent purchases of the user is extracted.
* the same is repeated for many random time stamps and users to obtain 200,000 samples.

By plotting the data, we may notice that the 3, 4 and 5 most recent purchases uniquely identify users with 99% probability.

#### Attack Method
For the period of time the recent purchases remain the same, every query sent by the user has the same list of recent purchases, i.e., most recent items purchased by a user usually do not change with a very high frequency. The attacker uses this knowledge to launch the attack. So, the attacker first selects a time threshold. This time threshold is chosen to help the attacker to decide if the queries come from the same 

