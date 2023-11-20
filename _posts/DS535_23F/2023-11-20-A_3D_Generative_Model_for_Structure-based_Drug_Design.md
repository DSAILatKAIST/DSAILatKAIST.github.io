---
title:  "[NIPS 2021] A 3D Generative Model for Structure-based Drug Design"
permalink: 2023-10-16-A_3D_Generative_Model_for_Structure-based_Drug_Design.html
tags: [reviews]
use_math: true
usemathjax: true
---

# **A 3D Generative Model for Structure-Based Drug Design**

이 논문은 Graph Neural Network를 이용해 단백질의 binding site에 맞는 화합물을 생성하는 모델에 대한 논문입니다.

해당 논문에 대해 더 궁굼하신 분들은 다음 링크를 참조하세요 : [논문](https://papers.nips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html) / [Github](https://github.com/luost26/3D-Generative-SBDD)

## **1. Background**

이 섹션은 본 논문을 이해하는데 필요한 기본 배경 지식에 관한 내용입니다.



### Protein and Protein Binding Site

  단백질은 20여가지의 서로 다른 아미노산으로 구성되어 있으며, 생물체 내에서 촉매, 효소, 신호전달, 호르몬 등 여러가지 역할을 합니다. 단백질은 다른 단백질, 핵산, 화학물과 반응하여 세포 안밖의 생명활동을 조절합니다. 단백질과 반응하는 다른 물질을 기질(ligand)라고 부르며, 결합하여 반응하는 위치를 결합부위(binding site)라고 부릅니다. 



<img src="https://i.ibb.co/zbgGmz6/Fig1-Protein-Binding-Site.jpg" alt="Fig_1_Protein_Binding_Site" style="zoom:67%;" />

<center>Fig1. Protein Binding Site [1]</center>

  이 때 binding site의 구조적, 화학적 특성으로 인해 특정한 ligand하고만 반응할 수 있고, 이를 specificity라고 부릅니다. 이런 단백질의 양이 너무 많이 혹은 적게 발현되거나, 돌연변이로 인해 정상적인 작동을 하지 않을 때 여러가지 질병이 생기게 됩니다. 현재 많은 약들은 protein의 binding site에 결합하여 구조를 바꾸거나, 기존의 ligand의 역할을 모방하여 정상적인 단백질 활성을 통해 질병을 치료합니다.



### Machine Learning for Molecular Biology

  단백질의 binding site에 결합할 수 있는 물질을 찾아내는 연구는 structure-based drug design이라고도 불리며 신약 연구에 있어 가장 어려운 일 중 하나였습니다. 과거엔 단백질의 구조를 밝혀내고, 그 중 binding site의 위치를 확인한 후에 그에 맞는 물질을 합성하여 실제로 결합하는지를 확인하는 과정을 통해 신약 후보 물질을 찾아내었고, machine learning(ML)이나 deep learning(DL) 이전까지만 하더라도 Moleculay dynamics(MD)연구나, *in silico*에서 신약 후보 물질을 찾는 것은 전문 인력과 긴 시간이 필요했습니다.

  하지만 ML, DL을 이용한 접근법을 통해 단백질과 결합이 가능한 molecule들의 distribution을 학습하면서 큰 성과를 거두기 시작했습니다. 이 방법들은 SMILES(string-based)나 Molecular Fingerprint(numerical encoding)를 사용하여 molecule들을 encoding 하여 ML, DL의 Input으로 사용하였습니다.



![Fig2_SMILES_and_MF](https://i.ibb.co/7yb6MFb/Fig2-SMILES-and-MF.png)

<center>Fig2. SMILES and Molecular Finger Print</center>

먼저 SMILES란 molecule의 복잡한 구조를 문자열 형태로 표현한 것을 말하고, Molecular Fingerprint는 molecule에서 특정 구조가 포함되어 있는지를 Multi-hot 형태로 나타내는 방법입니다. 이 방법은 domain knowledge가 없어도 이용 가능하고 다른 방법에 비해 빠르다는 장점이 있지만, encoding 과정에서 구조적인 특성을 모두 반영하지 못해 정보가 손실되는 단점이 있습니다.



### Graph Neural Network for Molecules

  최근에는 위와 같은 단점을 해결하기 위해 molecule의 구조를 graph로 표현하는 방법을 많이 사용하고 있습니다. 아래 Fig 3.에서 볼 수 있듯이 원자와 결합 혹은 각 작용기 사이의 연결 구조를 각각 node와 edge로 표현합니다. 이 방법은 chemical molecule에만 적용하는 것이 아닌 앞서 말씀드린 protein에도 적용이 가능하여 많이 사용되고 있습니다.

![Fig3_Molecule_to_Graph](https://i.ibb.co/JsmFVD0/Fig3-Molecule-to-Graph.jpg")

<center>Fig3. Molecule Graph representation</center>

이렇게 graph 구조로 표현된 molecule들은 Graph Neural Network(GNN) 모델을 기반으로 학습됩니다. GNN은 edge로 연결된 node들이 서로 정보를 주고 받으며 graph에 대한 정보를 학습합니다. 따라서 molecule 뿐 아니라 Protein까지 graph로 표현한다면 molecule과  protein이 서로 어떤 영향을 주고 받는지 파악할 수 있고, 이를 통해 binding site에 결합할 수 있는 molecule을 design할 수 있게 됩니다.

  본 논문은 이 Graph representation을 3차원 구조로 확장시켜 3D space에서 표현된 protein binding site에서 결합할 수 있는 molecule을 design하는 방법을 소개합니다.



## **2. Related Work**

  이 섹션에서는 지금까지 사용된 molecule design 연구를 소개합니다. 이러한 연구는 크게 **1) SMILES-Based and Graph-based Molecule Generation**, **2) Molecule generation in 3D space**로 나뉩니다.



### 2.1 SMILES-Based and Graph-based Molecule Generation

  Generation model의 연구와 함게 molecule design 영역도 함께 발전해 왔는데, 이는 molecule을 구성하는 원자의 분포를 학습하는 방법을 사용했습니다. 여기서 또 크게 두가지로 나뉘는데, 1. SMILES-based Method, 2. Graph-based Method입니다.

1. SMILES-Based Method

   앞서 말씀드렸다 싶이 SMILES는 molecule의 구조를 문자열의 형태로 나타낸 것 입니다. 이를 여러 language model에 적용해 분자 구조의 distribution을 학습합니다. 하지만 이 방법은 서로 다른 molecule의 유사성을 잘 찾지 못하고 최적의 구조를 찾지 못합니다.

2. Graph-Based Method (in 2D representation)

   그래프 구조는 molecule을 표현하는데 있어 자연스러운(natural) 방법으로 많은 연구에서 사용되고 있습니다. 특히 VAE나 강화학습 등 auto-regressive fashion으로 원자나 특정 구조를 순차적으로 붙여나가는 방법을 사용합니다.

하지만 두 방법 모두 1D/2D에 국한되어 있어 molecule과 단백질이 3D space에서 어떤 상호작용을 하는지 정확하게 학습하지 못한다는 단점이 있습니다.



### 2.2 Molecule Generation in 3D Space

  앞서 말한 두 가지 방법이 3D space에서의 molecule, protein 간의 상호작용을 학습하지 못한다는 단점을 극복하기 위해 제안된 방법으로, 3D space에서 직접 molecule을 생성하는 방법입니다. 일부 molecular structure만 input으로 넣고 3D space의 distance를 constraint로 주는 방법, 분자 구조의 potential energy를 reward로 하는 강화학습 등이 제안되었고, realistic한 molecule이 생성되긴 하지만 큰 사이즈의 (drug-scale) molecule은 만들지 못한다는 단점이 있습니다.

  다른 방법은 molecule 자체를 3D meshgrids에 voxelize한 3D image를 사용하는 방법입니다. 이렇게 함으로써 image generation problem으로 바꾸고, 지금까지 나온 image generation 기술들을 접목할 수 있게 하였습니다. 이 방법은 drug-scale molecule의 generation이 가능했지만 3가지 큰 단점이 존재했습니다.

1. protein의 binding site가 커지면, voxeling하는 단위가 3제곱으로 커져 model이 scalable하지 않습니다.
2. 1.과 같은 scalability issue에 의해 3D image의 해상도가 문제가 bottleneck으로 작용합니다.
3. Image generation에 사용하는 CNN은 rotation-equivariant하지 않아 molecular system에 적용하기 어렵습니다.

-----



## 3. Proposed Idea

  본 논문은 Binding site에 결합할 수 있는 molecule design을 위해 3D generative model을 제안합니다. 



### Overview

논문에서 제안한 모델은 binding site의 3D space에서 존재할 수 있는 원자들의 distribution을 학습하고자 합니다. $\mathcal{C}$ 를 binding site(Input), ***e***를 원소 종류, ***r***을 3D space에서의 좌표라고 했을 때 **$P(e, r \vert \mathcal{C})$**를 modeling 하고자 합니다. Binding site의 conditional context $\mathcal{C}$에서 3D 좌표 ***r***을 입력했을 때 해당 좌표에서 원자 ***e***가 그 곳을 점유하고 있을 확률을 계산하는 것 입니다. 이때 실제 단백질의 binding site에서의 rotation이나 translation에 invariant한 모델을 만들기 위해 rotationally invariant GNN을 사용합니다.



### Challenge and Proposal

  ***$P(e, r \vert \mathcal{C})$***를 계산하는 모델을 만드는 것과 별개로 이 모델로부터 **<u>다양</u>**하고 **<u>유효한</u>** molecule을 얻는 것은 다음과 같은 이유로 매우 어렵습니다. 

1. ***$P(e, r \vert \mathcal{C})$***로부터 *i.i.d.* sample을 얻는 것은 **유효**한 molecule을 보장하지 않습니다.

   molecule을 구성하는 원자들은 서로 independent하지 않기 때문입니다.

2. 해당 model은 feasible chemical space의 multi-modality를 학습해야 합니다.

   하나의 constraint(binding site, $\mathcal{C}$)로 부터 **유효**한 **여러** molecule들을 생성해야 합니다.



  본 논문은 이를 해결하기 위해 auto-regressive sampling algorithm을 사용합니다. 먼저, 단백질의 binding site($\mathcal{C}$)로 시작해서 남는 공간에 원자를 하나씩 생성합니다. 새로 생성된 원자는 다음 원자를 위한 context가 되고 남는 공간이 없어지면 종료합니다. 이는 post-processing algorithm이 필요 없고, latent variable을 사용하는 VAE나 GAN보다 상대적으로 낮은 구조적 복잡성과 학습 난이도와 함께 multi-modal sampling을 가능하게 합니다.

------



## **4. Method**

  본 섹션에서는 논문에서 제안한 모델을 설명합니다. 본 논문에서는 크게 세 부분으로 나누어 설명합니다.

1. 3D space에서 **원자 발생 확률 예측**
1. 유효한 multi-modal molecule을 생성하는 **auto-regressive sampling 알고리즘**

3. **Training objective**(loss function)



### 4.1 3D space에서 원자 발생 확률 예측

  Binding site는 $\mathcal{C} = {(a_i, r_i)}^{N_b}_{i = 1}$ 와 같은 식으로 정의합니다. $N_b$는 원자의 개수, $a_i$는 $i$번째 원자의 특징을 말합니다. 특징이라 하면 원자의 종류나 어떤 아미노산에 속하는지 여부 등 이 있습니다. $r_i$는 해당 원자의 3D 좌표입니다. 

  Binding site에서 원자를 생성하기 위해선 특정 위치 $r$에서 원자가 발생할 확률을 modeling해야 합니다. 다시 말하면, $p(e\vert r, \mathcal{C})$를 modeling하는 것이며 여기서 $r \in R^3$ 은 임의의 3D 좌표, $e \in \mathcal{E} = \{H,\ C,\ O\ ...\}$는 원자의 종류입니다.

  직관적으로 해석하자면, 어떤 3D 좌표 $r$에서 $\mathcal{C}$라는 제한조건 하에서 $e$ 라는 종류의 원자가 있을 확률을 예측하는 분류 모델입니다.

  $p(e\vert r, \mathcal{C})$를 modeling하기 위해 model을 두 부분으로 나누어 구성했습니다.

1. **Context Encoder** : $\mathcal{C}$라는 제한 조건 하에서 각 원자의 representation을 GNN을 통해 학습합니다.
2. **Spatial Classifier** : $r$이 입력되었을 때, 주변에 있는 원자의 representation을 aggregate해서 $p(e \vert r, \mathcal{C})$를 예측합니다.

1, 2에 대한 자세한 설명은 아래와 같습니다.



#### 4.1-1 Context Encoder

Context encoder의 목적은 제한조건 $\mathcal{C}$에 있는 원자들에게서 information-rich representation을 추출하는 것입니다. 이 때 원자의 representation은 두 가지 정보를 가지고 있어야 합니다.

1. **Context awareness** : 자기 자신의 정보 뿐 아니라 자기 주변의 정보(context) 또한 encoding해야 합니다.
2. **Rotational and translational invariance** : Rotation이나 translation에 의해 encoding 정보가 바뀌지 않아야 합니다.

이 두 조건을 만족하는 GNN은 다음과 같이 구성됩니다.

  먼저 $\mathcal{C}$ 에 대한 natural topology는 없기 때문에 원자의 거리를 기준으로 $k$-nearest-neighbor graph를 adjacency matrix $A$와 함께 $\mathcal{G} = \langle C, A \rangle$를 구성합니다. 또한 표기의 편의상 $i$번째 원자의  $k$-NN neighborhood를 $N_k(r_i)$라고 표현합니다.

  Context Encoder의 첫번째 층은 linear layer로 $\{a_{i}\}$의 특징을 첫번째 embedding 층인 $h^{(0)}_ {i}$로 mapping합니다. 그 후, $A$를 통해 message passing layer $L$로 이어집니다. 이 과정에 대한 수식은 아래와 같습니다.
$$
h^{l+1}_{i} = \sigma \left( W^{l}_0 h^{(l)}_i + \sum_{j\in N_k(r_i)} W^l_1w(d_{ij}\ \odot\ W^l_2h^{l}_j) \right)
$$
여기서 $w(\cdot)$은 weight network, $d_{ij}$는 원자 $i, \ j$ 사이의 거리입니다. 여기서 주목할 점은 수식의 모양이 continous filter convolution과 유사하지만 $j$에서 $i$로 가는 message는 $d_{ij}$를 통해서만 이동하기 때문에 rotation과 translation에 invariant합니다. 마지막 $h$ layer $\{h_i^{(L)}\}$은 $\mathcal{C}$에 있는 각 원자의 embedding이 됩니다.



#### 4.1-2 Spatial Classifier

Spatial classifier의 목적은 좌표 $r$을 입력받고 그 위치에 있을 원자의 종류를 예측합니다. 그러기 위해 $r$ 주변의 context를 얻어야 합니다. context를 얻는 첫번째 과정은 아래와 같습니다.
$$
v = \sum_{j \in N_k(r)} W_0w_{aggr}(\lVert r-r_j\rVert\ \odot\ W_1h_j^{(L)})
$$
이 과정은 (1)의 Context encoder에서 얻은 원자의 embedding을 aggregate하는 과정입니다. $N_k(r)$은 $r$의 $k$-nearest neighborhood입니다. 주목할 점은 context encoder의 $h$와 distance인 $\lVert r - r_{j} \rVert$에 각각 다른 weight matrix를 사용해 해당 원자의 context 정보를 구별하여 적용한다는 점입니다. 마지막으로 $p(e \vert r, \mathcal{C})$ 예측을 위해 aggregated feature $v$는 MLP을 통과시킵니다.
$$
c = MLP(v)
$$
여기서 $c$는 $r$위치에 해당하는 원소 $e$에 대한 non-normalized probability이며, softmax-fashion을 통해 확률값으로 추정합니다.
$$
p(e\vert r, \mathcal{C}) = \frac{\text{exp}(c \left[ e \right])}{1 + \sum_{e' \in \xi} \text{exp}(c[e'])}
$$
$\xi$는 가능한 원자 집합이며, 분모에 1을 더함으로써 좌표 $r$에 아무것도 존재하지 않을 확률을 계산할 수 있게 합니다. 따라서 좌표 $r$에 아무것도 존재하지 않을 확률, $p(nothing \vert r, \mathcal{C})$는 아래와 같이 계산됩니다.
$$
p(\text{Nothing}\ \vert\ r, \mathcal{C}) = \frac{1}{1 + \sum_{e' \in \xi} \text{exp}(c[e'])}
$$


### 4.2 Auto-regressive Sampling 알고리즘

  효과적인 sampling 알고리즘을 만드는대에는 3가지 어려움이 존재합니다.

1. 원자의 종류와 좌표 $e, r$의 joint distribution, $p(e, r \vert \mathcal{C})$를 정의해야 합니다.
2. $p(e, r \vert \mathcal{C})$를 정의했다고 해도, 원자들은 서로 독립적으로 존재하지 않기 때문에 $i.i.d.$ 를 가정할 수 없습니다.
   * 즉, sampling 알고리즘은 원자간의 dependency를 고려해야합니다.
3. Sampling 알고리즘은 multi-modal sampling이 가능해야합니다.
   * 즉, 여러 종류의 molecule을 생성해야 합니다.

위 세가지 조건을 고려하면서, 가장 먼저 joint distribution $p(e, r \vert \mathcal{C})$를 정의하고, auto-regressive sampling 알고리즘을 소개합니다.



#### Joint distribution

  좌표 $r$과 원자 종류 $e$의 Joint distribution은 아래와 같이 정의합니다.
$$
p(e, r \vert \mathcal{C}) = \frac{\text{exp}(c[e])}{Z}
$$
여기서 $Z$는 알 수 없는 Normalizing constant이며, $c$는 식 (3)에서 정의한 함수입니다. $p(e, r)$은 non-normalized distribution이지만, $r$이 3차원임을 고려하면 충분이 효율적이다라고 소개하고 있습니다. 이후 Markov chain Monte Carlo(MCMC), 혹은 이산화(discretization)를 통해 sampling을 진행합니다.



#### Auto-Regressive Sampling

  step $t$에서, context $\mathcal{C}_ t$를 고려한 하나의 원소를 생성합니다. 이 때 $\mathcal{C}_ t$는 binding site의 context 뿐만 아니라 step $t$ 까지 sampling한 molecule까지 고려합니다. Molecule로써 생성된 원자는 binding site의 원자와 동일한 취급을 받지만 각자 다른 attribute를 가져 서로 구별합니다. 이후 $t+1$번째 원자는 $p(e, r \vert \mathcal{C}_ t)$로 부터 생성되며 $\mathcal{C}_ {t+1}$에 포함됩니다. 이를 수식으로 정리하면 아래와 같습니다.
$$
(e_{t+1}, r_{t+1}) \sim p(e, r \vert \mathcal{C}_t) \\
\mathcal{C}_{t+1} \leftarrow \mathcal{C}_t \cup \{ e_{t+1}, r_{t+1} \}
$$
  auxiliary network를 이용해 Auto-regressive sampling을 종료할 시기를 정합니다. 이 frontier network는 지금까지 sampling된 원자들의 embedding을 각각 frontier과 non-frontier로 구별하고, 모든 원자가 non-frontier로 구별되면 sampling을 종료합니다. sampling이 끝나면 OpenBabel을 이용해 원자간의 결합을 생성합니다.

> OpenBabel : 화학 데이터를 다룰 수 있도록 설계된 오픈 소스 화학 도구 프로그램



![Fig 4. Sampling Procedure](https://i.ibb.co/fv0cNtD/Fig3-Sampling-Process.png)

<center>Fig 4. Sampling Procedure</center>
<center>원자가 순차적으로 추가되면 그에 따라 probability density가 변하고 이를 통해 여러 molecule을 생성합니다.</center>





### 4.3 Training Objective

  모델의 학습은 molecule의 일부분을 masking하고, 해당 부분의 원자를 예측하는 방법으로 학습합니다. loss function은 크게 세 가지로 나뉩니다. 1. 원자가 있어야 할 곳에 있고, 없어야 할 곳에 없는지 / 2. 원자의 종류를 잘 예측했는지 / 3. frontier network를 통해 frontier를 잘 구분하는지

1. 원자가 있어야할 곳에 있고, 없어야 할 곳에 없는지

     앞서 모델을 학습할 때 일부분을 masking한다고 했습니다. model은 빈 공간에 원자가 있어야 하는지, 혹은 없어야 하는지를 판단해야 합니다. 실제로 원자가 있는 곳을 positive position, 없는 곳을 negative position이라고 할 때, loss function은 binary cross entropy(BCE)로 정의되고, 수식은 아래와 같습니다.
   $$
   L_{\text{BCE}} = -\mathbb{E}_{r\sim p_+}[\text{log}(1-p(\text{Nothing} \vert r, \mathcal{C}))]\ -\ \mathbb{E}_{r \sim p_-} [\text{log}\ p(\text{Nothing} \vert r, \mathcal{C})]
   $$

2. 원자의 종류를 잘 예측했는지

     원자의 종류에 대한 loss function은 categorical cross entropy를 사용합니다.
   $$
   L_{\text{CAT}} = -\mathbb{E}_{(e, r) \sim p_+}\ [\text{log}\ p(e\vert r, \mathcal{C})]
   $$

3. frontier classification

     frontier network의 loss function또한 binary cross entropy를 사용합니다.
   $$
   L_{\text{F}} = \sum_{i \in \mathcal{F} \subseteq \mathcal{C}}\text{log}\ \sigma(F(h_i)) +\sum_{i \notin \mathcal{F} \subseteq \mathcal{C}}\text{log}\ (1- \sigma(F(h_i)))
   $$
   $\mathcal{F}$는 $\mathcal{C}$에 포함된 frontier atom 집합이며, $\sigma$는 시그모이드 함수, $F(\cdot)$은 frontier network 입니다. frontier network는 원자의  embedding을 입력받고 해당 원자가 frontier인지 아닌지 logit probability를 예측합니다. training 과정에서 원자가 target molecule의 일부이면서 동시에 다른 masked된 원자가 있을 때에만 frontier로 취급됩니다.



1~3의 loss function을 합쳐 full training loss $L = L_{\text{BCE}} + L_{\text{CAT}} + L_{\text{F}}$를 얻을 수 있습니다.



![Fig 5. Training Procedure](https://i.ibb.co/N2fttHJ/Fig5-Training-Procedure.png)

<center>Fig 5. Training Procedure</center>
<center>
  위 figure를 통해 전체적인 학습 과정을 볼 수 있습니다. (a)에서 molecule의 일부가 masking되어있고, (b)에서 각각 positive sampling, negative sampling이 되는 것을 볼 수 있습니다. (c)에서 좌표가 model에 입력되어 각각 원자가 있을 확률, 없을 확률을 계산합니다. 마지막으로 (d)에서 training loss를 계산합니다.
</center>







## **5. Experiment**

### Experiment Setup

* **<u>Task</u>** : **(1) Molecule Design**, **(2) Linker Prediction**

* **<u>Dataset</u>** : CrossDocked dataset
  * 2250만 docked protein-ligand 쌍 중에서 molecule과 protein 사이의 거리(RMSD)가 1Å 이하인 184,057개를 추려서 유사도가 30% 이하인 protein 100,000개를 train, 100개를 test에 사용함
* **<u>Model</u>**
  * Context encoder layer (L) = 6
  * Hidden dimension = 256
  * Learning rate = 0.0001

-----

### 5.1 Molecule Design

* **<u>Baseline</u>** : liGAN
  * Conventional 3D convolutional neural network, SOTA
  * Generates voxelized molecular image
  
* **<u>Metric</u>**
  1. **Binding affinity** : *Vina* score [2]
  
     > Vina score is used in molecular docking simulations to estimate the binding affinity between a protein receptor and a small molecule ligand. It uses various energetic terms, such as van der Waals interactions, electrostatic interactions, hydrogen bonding, and others.
  
  2. **Drug likeness** : *QED* score [3]
  
     > QED score is used to assess the drug-likeness of a chemical compound. It uses various molecular descriptors and properties associated with drug-likeness, such as lipophilicity, molecular weight, number of hydrogen bond donors and acceptors, and other factors. The idea is to provide a quantitative measure that reflects the likelihood of a compound possessing drug-like properties.
  
  3. **Synthesizability** : *SA* score [4]
  
     > SA score is a metric used to assess the ease or difficulty of synthesizing a chemical compound. It is commonly employed in the field of drug discovery and design to prioritize compounds that are more likely to be chemically accessible for synthesis.
     >
     > It takes into account various factors related to the synthetic feasibility of a compound. These factors may include the complexity of the chemical structure, the availability of starting materials, and the number of synthetic steps required for the overall synthesis.
  
  4. **Percentage of samples with high affinity** : wheather it is higher or lower than reference molecule
  
  5. **Diversity** : average pairwise Tanimoto similarities over Morgan fingerprints among moleucles
  
     > $Tanimoto(A,B) = \frac{\text{Number of unique features in A or B}}{\text{Number of common features in A and B}}$
     >
     > Tanimoto similarity is often used with molecular fingerprints, and it measures similarity between two sets of molecular fingerprints. In the context of chemical compounds, molecular fingerprints represent the presence or absence of certain molecular features.



#### Result of Molecule Design

<img src="https://i.ibb.co/WnV2bG0/Table1-Molecule-Design-result-table.png" alt="Table 1. Molecule Design Result Table" style="zoom:50%;" />

<center>Table 1. Result of Molecule Design</center>
<center>
  (↑)는 높을수록 좋은 점수이며,  (↓)는 낮을수록 좋은 점수입니다.
</center>

전체적으로 baseline인 liGAN보다 좋은 성능을 보이는 것을 확인할 수 있습니다. Target binding site와 더 잘 결합할 수 있는 molecule을 생성할 뿐 아니라, drug-likeness property, synthesizability가 높은 것을 통해 신약 후보 발굴에 더 적합한 모델임을 알 수 있습니다. 또한 baseline보다 더욱 다양한 molecule을 생성하는 것을 알 수 있습니다.



![Fig 6. Molecule Design Visualization](https://i.ibb.co/zsCV86j/Fig6-Molecule-Design-Example.png)

<center>Fig 6. Visualization of Molecule Design</center>
<center>2개의 binding site에 대해 생성한 Molecule을 시각화한 것 입니다. 좌측부터 Vina score가 높은 6개의 molecule과 가장 우측에는 실제 molecule(reference)를 시각화했습니다.</center>

이 Figure에서 눈여겨볼 점은 model이 생성한 molecule이 실제 reference보다 binding affinity, 즉 성능도 좋고, 약이 되기 좋은 성질(drug-likeness, synthesizability)이 더욱 좋다는 것 입니다. 





### 5.2 Linker Prediction

* **<u>Baseline</u>** : DeLinker
  * Graph-based generative model
* **<u>Metric</u>**
  1. **Similarity** : Tanimoto Similarity over Morgan fingerprints
  2. **Percentage of Recovered Molecules** : Similarity가 1인 generated molecule의 비율
  3. **Binding Affininty** : *Vina* score
* **<u>Data preperation</u>** : molecule에서 자를 수 있는 acyclic single bond를 모두 잘라서 120개의 data-point 준비



#### Result of Linker Prediction

<img src="https://i.ibb.co/DWVm3n3/Table2-Linker-Prediction-Table.png" alt="Table 2. Result of Linker Prediction" style="zoom:50%;" />

<center>Table 2. Result of Linker Prediction</center>
<center>
  (↑)는 높을수록 좋은 점수이며,  (↓)는 낮을수록 좋은 점수입니다.
</center>

Linker prediction이란 특정 binding site context에서 연결되어있지 않은 2개의 molecule 조각을 포함하는 molecule을 생성하는 것을 말합니다. 본 모델은 Linker prediction에 대해 별다른 구조 수정이나 fine-tuning을 거치지 않았음에도 baseline model보다 준수한 성능을 보입니다.



![Fig 7. Visualization of Linker Prediction](https://i.ibb.co/jTw90jq/Fig7-Visualization-of-Linker-Prediction.png)

<center>Fig 7. Visualization of Linker Prediction</center>
<center>2개의 binding site와 각 2개의 fragment를 주어주고 생성한 Molecule을 시각화한 것 입니다. 좌측부터 유사도가 높은 6개의 molecule과 가장 우측에는 실제 molecule(reference)를 시각화했습니다. 빨간색으로 표시된 원자가 새롭게 생성된 원자입니다.</center>



## **6. Conclusion**

  본 논문은 구조기반 약물 설계에 대한 새로운 접근방식으로 3D 생성모델을 제안합니다. 이 모델은 원자의 생성확률 modeling과 auto-regressive sampling 알고리즘을 통해 protein binding site에 맞는 molecule을 생성할 수 있고, 기존 SOTA 모델보다, 몇몇은 기존에 있는 molecule보다 더 좋은 성질을 가진 molecule을 생성하는 것을 볼 수 있었습니다.

  추후 graph representation통해 valency check(원자가 형성할 수 있는 화학 결합의 수를 초과하는지 확인)이나 property optimization에도 활용하는 연구를 진행할 수 있을 것 입니다.

  하지만 해당 논문에서 context로 어떠한 feature를 사용했는지 알 수 없다. lingand가 단백질에 결합하는 binding site prediction task에서 단백질을 modeling할 때 사용하는 feature가 달라지면 model의 성능 차이가 크게 나오는데, 이 논문의 model에서도 context의 정보가 중요하게 작용하기 때문에 어떤 feature를 사용하는지가 중요하다고 생각한다.

  또한 SA, QED, vina score등의 점수가 잘 나오긴 하지만, 실제로 합성을 해보면 그대로 나오지 않는 경우가 많이 있는데, 결과가 좋은 만큼 다른 랩실과 협업하여 *in silico*에서 끝나는게 아니라 *in vitro*에서의 결과도 볼 수 있었으면 좋았을 것 같다.

------



## 7. Author informantion and Reference

### Author information

* Joohyun Cho
  * Affliliation : [BCBL@KAIST](https://sites.google.com/view/kaist-bcbl)
  * Research Topic : Protein Design, Drug discovery
  * Contact : joohyun98@kaist.ac.kr



### Reference

[1] Niklas WA Gebauer, Michael Gastegger, and Kristof T Schütt. Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. arXiv preprint arXiv:1906.00957, 2019

[2] Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. J Comput Chem. 2010 Jan 30;31(2):455-61. doi: 10.1002/jcc.21334. 

[3] Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. Quantifying the chemical beauty of drugs. Nat Chem. 2012 Jan 24;4(2):90-8. doi: 10.1038/nchem.1243.

[4] Ertl, P., Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *J Cheminform* **1**, 8 (2009).