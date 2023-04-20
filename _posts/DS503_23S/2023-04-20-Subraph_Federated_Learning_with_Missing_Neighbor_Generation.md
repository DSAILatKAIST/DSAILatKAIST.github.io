---
title:  "[NIPS 2021] Subgraph Federated Learning with Missing Neighbor Generation"
permalink: Subraph_Federated_Learning_with_Missing_Neighbor_Generation.html
tags: [reviews]
use_math: true
usemathjax: true
---

# [NeurIPS-21] Subgraph Federated Learning with Missing Neighbor Generation 리뷰

# 1. Motivation

본 논문은 Graph domain에서 **Subgraph Federating Learning** 을 처음 시도한 논문이다. 이 연구가 왜 필요한지 알아보려면 우선 Federated Learning이 무엇인지 간략한 개념을 알아야한다. **Federated Learning**이란 privacy등의 이유로 다양한 local system들로 부터 data를 모을 수 없는 상황에서 모델을 함께 학습시켜 각자의 local system 내의 data로만 학습시켰을 때 보다 더 좋은 성능을 얻기 위한 방법이다. 예를들어 Computer Vision (CV) 도메인 에서는 휴대폰 사진첩내의 사진들을 모두 중앙 서버로 모아서 Machine Learning모델을 학습시키려 한다면 각자의 얼굴 등의 privacy가 침해 될 우려가 있을 수 있기 때문에 데이터를 공유하지 않고 Convolution Neural Network (CNN)을 학습 시키려고 하는 것이다. 이러한 이유로 CV와 NLP도메인에서는 Federated Learning연구가 많이 진행되어 왔는데 Graph 도메인에서는 왜 Federated Learning이 필요할까?


![Motivation](https://drive.google.com/uc?export=download&id=1SygClT33EqcEuDLVZbLYWeQyjEu7oxHh)


Graph 도메인에서 Graph Neural Networks (GNNs)를 이용한 많은 적용사례가 있지만 그 중 하나로 유사한 특성을 가진 환자간에 edge를 연결시켜서 만든 Patient Graph를 이용하여 환자의 질병 유무 등을 예측하는 task가 있다. 이 경우 환자의 개인 정보 및 검사 결과 사진 등을 Medical Center로 보내서 GNN을 학습 시켜야 하는데 real-world에서의 상황을 생각해보면 병원에서 환자 개인정보는 매우 민감한 privacy 문제가 있기 때문에 사실상 하나의 Center에 여러 병원의 환자 정보를 취합하기가 힘들다. 그러다보니 위의 그림처럼 Hospital A, B, C, D에서 데이터를 Moedical Administration Center로 모아서 학습시키는 것이 아니라 **'Joint training without sharing graph data'** 가 필요한 것이다. 그렇다면 CV와 NLP에서 사용하는 방법론들을 그대로 적용하지 않고 새로운 연구가 필요한 것일까?

위의 예시처럼 Hospital간에 data를 share하지 않는 경우 Hospital A와 B에 있는 환자 간에는 edge를 연결 시킬 수 없게 된다. 따라서 GNN에서 결과값을 낼 때 전체 Graph를 다 활용하지 못하고 병원 별 **Subgraph** 만을 활용할 수 있기 때문에 missing edge가 발생하여 이로 인한 정보 손실이 일어난다. 이러한 현상은 기존 다른 도메인에서는 없는 문제이기 때문에 이를 처리하기 위한 방법이 필요하고 따라서 **Subgraph Federated Learning** 에 특화된 방법론을 연구할 필요가 있다.


# 2. Method

본 논문에서는 위의 설명과 같이 여러 개의 Subgraph간에 data를 공유하지 않고 jointly training하는 Federated Learning을 위한 방법론을 제시하였다. 그래서 첫번째로 1) 가장 기본적인 Federated Learning 방법인 FedAvg를 Graphsage 인코더를 활용하여 적용해본 **Fedsage**을 제시하였고 2) Missing Edge를 문제를 해결하기 위해 이 Link를 다시 복원시켜주는 Generative Model을 결합시킨 **Fedsage+** 를 추가적으로 제시하였다.

## 2-1) Fedsage

먼저 Fedsage는 일반적인 GNN에서 Node classification model을 학습시키듯이 각각의 label에 대한 prediction 값을 계산한 뒤 아래 Cross Entropy Loss를 계산하고 gradient descent를 통하여 학습시킨다. 
![CE](https://drive.google.com/uc?export=download&id=1rZ4fhaWnSMNTqxtcpMapl9tWeM6OI2kt)

그리고 FedAvg를 적용하는데 Fedavg는 1) Round마다 local system을 학습 2) 학습된 local system의 weight를 server에서 average 3) server에서 weight를 각각의 local system으로 배포 의 순서를 반복하는 알고리즘이다. 이는 가장 대표적이고 간단하여 널리 쓰이는 방법론이고 아래 Fedavg논문에서의 수도 코드를 참고하여 쉽게 이해할 수 있다.
![Fedavg](https://drive.google.com/uc?export=download&id=1TXPt7ESV5qanZHAZpQW8vnrJisC3Wq4s)

## 2-2) Fedsage+

위 Fedsage는 Subgraph Federated Learning 세팅에서 처음으로 Federated Learning을 시도해봤다는 의미가 있지만 이 방법은 이전 Motivation 섹션에서 말한 Missing Link로 인한 문제를 전혀 다루지 않고 있다. 따라서 본 논문에서는 Generative Model을 활용하여 Missing 잃어버린 연결관계를 복원시켜주는 방법을 제안한다. 

![Fedavg](https://drive.google.com/uc?export=download&id=1meNAOF25mBHTH6A0yoztAX62jT6Vy0jo)

그래서 모델 Architecture를 크게 보면 Missing Neighbor Generator (NeighGen)을 통하여 Missing 된 Node들을 다시 만들어서 Observe된 Graph에서 붙여주고 (Graph mending) 복원된 Graph를 통하여 downstream task인 node classification을 수행한다. Node classification 부분은 위 fedsage와 동일하기 때문에 중요한 부분은 NeighGen을 어떻게 구성하는지 인데 우선 missing된  link를 Subgraph내에서 만들어주기 위해서는 각각의 Subgraph 내의 node마다 몇개의 node가 drop되었는지를 알아야 하기 때문에 missing된 node 개수를 예측하는 dGen을 학습시키고 그 예측된 node 개수에 맞게 새로운 node의 feature를 만들어주는 fGen을 학습시킨다.

구체적으로 보면 학습단계에서는 Ground Truth값을 알아야 model을 학습시킬 수 있기 때문에 관측된 Graph에서 몇몇 node를 더 숨겨 (hide) missing link를 발생시킨다. 그리고 NeighGen을 위한 GNN을 통해 node마다의 representation을 뽑아 낸뒤 dGen을 통하여 몇개의 node가 숨겨졌는지 예측한다. 여기서 dGen는 $\theta^d$라고 표기되는 parameter를 가진 linear regression 모델이다. 그래서 논문에는 아래 수식으로 missing neighbor 수인 $\tilde{n_v}$ 를 prediction을 한다고 하는데 논문의 $n_v$는 typo인 것 같고 node representation을 타나내는 $z_v$로 대체 되어야 할 것 같다.

![dGen](https://drive.google.com/uc?export=download&id=1RdS9I6bgqZPOZ-dPq2a7arkXHg2LyS_a)

그리고 fGen에서는 예측된 missing neighbor수 만큼 node를 만들어 줘야 하는데 다양성을 위하여 Gaussian noise와 node representatino을 더해주고 fGen의 parameter  $\theta^f$와 곱해주는 방식으로 새로운 node feature를 만들어낸다. 이 때 dGen에서 예측한 missing neighbor 개수인 $\tilde{n_v}$개 만큼 생성하고 수식은 아래와 같다. (여기서 R은 Random Sampler를 의미한다)

![fGen](https://drive.google.com/uc?export=download&id=1U8_7sHEpJIO9IsuPGfEku-IjK4kvmeAk)

그래서 전반적인 NeighGen에 관한 학습은 위 예측값을 바탕으로 아래 수식과 같이 실제 missing neighbor의 차이를 계산하는 dGen에 관한 loss 그리고 Generate한 Neighbor와 holding된 Neighbor의 feature difference를 계산하는 부분으로 구성된다.

![Graph_mending](https://drive.google.com/uc?export=download&id=1IZIcwK_5nJ3wDtDSLZC3AhglMBENGhfo)

Downstream task인 node classification (fedsage와 동일)에 대한 loss와 balance parameter인 $\lambda$와 함께 결합하여 아래 최종 loss function이 도출된다.

![overall](https://drive.google.com/uc?export=download&id=13YbmbMAiP28gAwae_MA7Q94H8FXlobp5)

### Federated Learning of GraphSage and NeighGen

Federated Learning을 하기 위해서는 위 Loss function을 바탕으로 Graphsage인코더와 NeighGen을 각각의 local system에서 학습시키고 Fedavg를 하는 것이 가장 간단하고 직관적인 방법일 것이다. 하지만 저자는 실험적으로 NeighGen의 weight를 averaging을 했을 경우 diverse한 neighborhood node가 만들어지지 않았다고 한다.

![fGen_local](https://drive.google.com/uc?export=download&id=1a2BnNEAHL4ZVVqzRdlwh_me5uuJ7fqjQ)

따라서 NeighGen은 공통된 하나의 model을 학습시키는 것이 아니라 local한 model을 각자 학습시키는 방식을 사용하는 technique을 이용하였다. 하지만 다른 node의 정보를 동시에 이용하기 위해서 위의 수식과 같이 변경하였는데 살펴보면 앞쪽 term은 기존에 local system내에서 minimize하던 loss와 동일한데 뒤쪽 term은 다른 client에 있는 node와의 distance를 minimize하는 term이 있다. 이 loss를 계산하기 위해서는 다른 node의 정보를 받아와야 하는데 이는 Federated Learning의 세팅과 맞지 않기 때문에 각자의 model의 weight와 node representation을 server로 보내서 server에서 해당 term을 계산하고 gradient를 보내주는 방식의 trick을 이용했다고 한다. 그렇다고 해도 node representation을 server로 전송해야 하기 때문에 privacy문제가 전혀 없다고 할 수는 없을 것이고 이 부분이 이 논문의 가장 큰 limitation이라고 생각한다. 설명한 부분에 관한 수도 코드는 아래와 같다.

![fedsage+](https://drive.google.com/uc?export=download&id=1IH_yPZmVmgwwI_AsGckqN7Xgovq5iDZH)

# 3. Experiments

![statistics](https://drive.google.com/uc?export=download&id=1XHx8BVUdSfdgpmCiKqr77BZjUZyVqweU)

위 논문에서는 Subgraph Federated Learning실험을 위해 전체 graph를 Louvain method를 통해서 여러 partition으로 나눈 뒤 Silo 수 (Subgraph 수)인 M개 만큼 나눠서 각각의 silo에 배분하는 방식으로 실험 세팅을 하였다. Benchmark dataset인 Cora, Citeseer, Pubmed, MSAcademic을 사용하여 실험을 진행했고 $\Delta E$가 해당 갯수로 partition했을 때 발생하는 missing link개수를 의미하는데 Silo수가 많아 질수록 당연히 더 Missing link가 많아 지는 것을 확인 할 수 있다.


![statistics](https://drive.google.com/uc?export=download&id=1fasj0podN9FZ0oU5aG4BomJYim2UyNFY)

위 Benchmark 데이터셋을 이용하여 실험한 결과 Federated Learning을 이용한 FedSage가 각각의 local model에서 GNN을 학습한 LocSage에 비해서 큰 성능 향상을 보여 Subgraph Federated Learning의 가능성을 보여줬다. 또한 FedSage+ 즉 Missing Neighbor를 만들어 준 모델이 모든 세팅에서 더 좋은 성능을 보여 해당 Missing link의 중요성을 보여줌과 함께 제안된 모델의 우수성을 입증하였다. 

![statistics](https://drive.google.com/uc?export=download&id=1fJa1HsP9ArosVH0a50_-By6dDtmlao7Z)

그리고 모델을 학습시킬 때 다른 Silo의 node 정보를 이용한 portion인 $\alpha$와 학습 시 어느정도의 node를 hide 시킬지에 대한 hyper-parameter인 h에 대한 Sensitivity anlaysis를 하였다. 위 실험을 통해 다른 silo의 정보를 적정량 가져 오는 것이 더 좋은 성능을 보여 왜 위의 loss를 fedavg하지 않고 NeighGen을 각기 학습시켰는지 보여주었다.

# 4. Conlcusion

해당 논문을 종합해보면 **Subgraph Federated Learning**의 필요성을 좋은 예시를 들어 설명해주었고 그에 따라 Missing Link라는 Graph domain만의 문제가 발생할 수 있다는 점을 제기하였다. 이를 해결하기 위해 Neighborhood generator라는 방법론을 제안하였는데 method적으로 아주 획기적인 논문은 아니였지만 연구를 할 때 새로운 문제를 잘 정의해보고 실제로 그런지 실험을 통해서 잘 보여주는 것이 중요하다는 교훈을 주는 논문이라고 생각한다. 
