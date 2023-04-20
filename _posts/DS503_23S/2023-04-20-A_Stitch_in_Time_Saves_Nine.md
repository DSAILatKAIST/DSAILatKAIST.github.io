---
title:  "[CVPR 2022] A Stitch in Time Saves Nine"
permalink: A_Stitch_in_Time_Saves_Nine.html
tags: [reviews]
---

# 1. Introduction

현대의 Deep Neural Network (DNN)는 높은 정확도를 보이지만, 예측된 라벨에 대한 confidence score와 실제 발생 확률이 일치하지 않는 miscalibration이 발생한다.
본 논문에서는 'Multi-class Difference in Confidence and Accuracy(MDCA)' 라는 손실 함수를 적용하여 학습 단계에서 calibration 문제를 해결한다.


# 2. Related Work
DNN의 calibration 기법으로는 크게 세가지가 존재한다.

## 1) Train-Time Calibration
Negative-Log-Likelihood (NLL) 을 통해 학습된 모델들은 흔히 over-condident 문제가 발생한다. 이를 해결하기 위한 대표적인 방법은 NLL loss에 entropy 와 같은 regularization term을 추가하는 것이다. Label Smoothing(LS), focal loss 또한 흔히 사용된다.

## 2) Post-Hoc Calibration
Post-Hoc calibration은 validation set을 사용하는 방법이다. 가장 대표적으로 Temperature scaling(TS)은 validation set을 통해 학습된 스칼라 T로 logit을 나눠 평활화 한다.

## 3) Calibration through OOD Detection
DNN overconfidence의 가장 큰 이유 중 하나는 학습 데이터의 분포에서 멀리 떨어진 input 샘플에 대해 ReLu 함수이 높은 confidence prediction을 준다는 점이다. 이를 해결하기 위해 adversarial training을 사용한 data augmentation을 통해 학습 데이터와 멀리 떨어진 샘플들에 대해 낮은 confidnece score를 줄 수 있다.


# 3. Methodology
## Background
$s_i[y]$를 클래스 $y$에 대한 confidence 값이라고 할 때, 예측된 클래스는 다음과 같다. 

$\hat{y_i} = arg max_y s_i[y]$

모델이 완벽하게 calibrate 된 경우는 다음과 같이 표현할 수 있다.

$P(y={y}'|s[y]=s)=s $

완벽하게 calibrate 되기 위해서는 ReLU 함수를 적용하여 예측된 클래스 외에 다른 클래스에 대해서도 위의 식이 성립해야 하지만, 대부분의 calibration 기법들은 예측 클래스에만 집중한다.

Expected Calibration Error(ECE): ECE는 예측된 클래스의 confidence와 샘플의 정확도(accuracy) 간의 차이에 대한 가중 평균(weighted average)를 통해 계산된다. 
$\sum_{i=1}^{M} B_i/N\left | A_i-C_i \right |$

$N$: 총 샘플의 수

$B_i$: confidence 범위를 M개의 bin으로 분할했을 때, confidence가 $(\frac{i-1}{M}, \frac{i}{M}]$에 포함된 bin의 샘플 수

$A_i$: $B_i$에 포함된 샘플들의 평균 정확도

$C_i$: $B_i$에 포함된 샘플들의 평균 confidence

Maximum Calibration Error(MCE): 각 bin에서의 평균 정확도와 평균 confidence 간 차이의 최댓값. 많이 사용되지 않음.

Static Calibration Error(ECE): 
$\frac{1}{K}\sum_{i=1}^{M}\sum_{j=1}^{K}\frac{B_{i_j}}{N}|A_{i_j}-C_{i_j}|$

$K$: 클래스의 수

$B_{i_j}$: i번째 bin의 j번째 클래스 샘플의 수

$A_{i_j}$: i번째 bin의 j번째 클래스 샘플들의 정확도

$C_{i_j}$: i번째 bin의 j번째 클래스 샘플들의 평균 confidence

## Proposed method: MDCA
SCE로부터 파생된 손실 함수이지만, 기존에 샘플들을 bin으로 분할함으로 인하여 미분이 불가능하였던 문제를 해결하였다.

$L_{MDCA} = \frac{1}{K}\sum_{K}^{j=1}|\frac{1}{N_b}\sum_{M}^{i=1}s_i[j] - \frac{1}{N_b}\sum_{M}^{i=1}q_i[j]|$

$q_i[j]=1$(샘플 i에 대한 예측 라벨 j가 ground truth인 경우), $q_i[j]=0$(그 외)

$N_b$: mini batch에 속한 샘플의 수

MDCA 손실 함수는 미분 가능하기 때문에 다른 손실함수(Cross Entropy, Label Smoothing, Focal Loss 등)와 함께 사용하기에 적합하다.

$L_{total} = L_C +\beta L_{MDCA} $


# 4. Experiment

다양한 데이터셋에 기존 SOTA(State-of-the-art) calibration 기법과 MDCA를 함께 적용한 결과를 비교하였다.

CIFAR100 데이터에 MDCA를 적용했을 때 기존 1.90의 ECE score가 0.72로 감소하였다.

그 외에 SVHN, Mendeley V2, Tiny-ImageNet 등의 데이터셋과 자연어 처리에서도 MDCA를 적용한 경우 낮은 calibration error을 나타냈으며, 특히 focal loss와 함께 사용되었을 때 가장 낮은 calibration error을 보였다. 




# 5. Conclusion
학습 단계에서 calibration을 위해 사용된 손실함수인 MDCA는 기존 손실함수와 함께 적용했을 때 calibration error을 감소시키는 효과를 보였다. 예측 클래스에만 적용되었던 기존 calibration 기법들과 달리 예측되지 않은 클래스들에 대한 calibration도 달성하였으며, 기존 방법들에서 해결하지 못했던 손실함수의 미분 불가능 문제를 해결함으로써 이러한 성과가 나타났다..