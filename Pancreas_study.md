# 췌장 CT 연구 주제 분석 (Pancreatic CT Research Topic Analysis)

> 보유 데이터: 정상 복부 CT 473명 + 췌장 질환 CT 104명 (8종)  
> 마스크 데이터: 정상 CT 473명 (췌장 마스크) + 질환 CT 104명 (병변 포함 췌장 마스크)  
> 작성일: 2026-04-09

---

## 목차

1. [데이터 현황 요약](#1-데이터-현황-요약)
2. [연구 주제 후보](#2-연구-주제-후보)
   - [주제 1: 이진 분류 (Binary Classification)](#주제-1-이진-분류-binary-classification)
   - [주제 2: 췌장 분할 (Pancreatic Segmentation — Fully/Semi-supervised)](#주제-2-췌장-분할-pancreatic-segmentation)
   - [주제 3: 희귀 질환 CT 합성 (Rare Disease CT Synthesis)](#주제-3-희귀-질환-ct-합성-rare-disease-ct-synthesis)
   - [주제 4: 자기지도 사전학습 (Self-supervised Pre-training)](#주제-4-자기지도-사전학습-self-supervised-pre-training)
3. [최종 추천 요약](#3-최종-추천-요약)
4. [참고 문헌](#4-참고-문헌)

---

## 1. 데이터 현황 요약

| 구분 | 케이스 수 | 레이블 정보 |
|------|-----------|------------|
| 정상 복부 CT | 473명 | **정상 췌장 마스크 보유** |
| 췌장 질환 CT (8종) | 104명 | **병변 포함 췌장 마스크 보유** |
| **합계** | **577명** | **전체 577명 마스크 보유** (질환 8종 × 약 13명/클래스) |

**주요 특징 (업데이트):**
- 정상 CT 473명과 질환 CT 104명 **모두 췌장 마스크 보유** → Fully-supervised 분할 학습 가능
- 질환 CT는 병변(Lesion)을 포함한 췌장 마스크 → 향후 병변 분할 연구로 확장 가능
- 8종 질환 각 클래스당 ~13명으로 다중 분류 딥러닝 학습에는 여전히 부족

**남은 제약 사항:**
- 8종 질환의 클래스 불균형 문제 (클래스당 ~13명)
- 병변(Lesion) 경계 마스크(fine-grained lesion segmentation mask)는 추가 확인 필요

---

## 2. 연구 주제 후보

### 주제 1: 이진 분류 (Binary Classification)

**정상 vs 췌장 질환 이진 분류**

| 항목 | 내용 |
|------|------|
| 입력 | 복부 CT 슬라이스 또는 3D 볼륨 |
| 출력 | 정상(Normal) / 질환(Disease) 이진 분류 |
| 데이터 구성 | 정상 473명 vs 질환 104명 (불균형 주의) |
| 실현 가능성 | **높음** |
| Novelty | **낮음** |

**장점:**
- 클래스 수가 적어 소규모 데이터에도 적용 가능
- 임상적으로 의미 있는 스크리닝 도구

**단점:**
- 불균형 데이터 (473 vs 104) → 오버샘플링/가중치 조정 필요
- 8종 질환을 하나로 묶는 것은 임상적 정보 손실

**참고 논문:**
- Viriyasaranon et al. (2023): 자기지도 사전학습(Self-supervised pre-training)으로 소규모 췌장 CT 데이터셋에서 분류 성능 향상
  - 저널: *Cancers*
  - DOI: [10.3390/cancers15133392](https://doi.org/10.3390/cancers15133392)

---

### 주제 2: 췌장 분할 (Pancreatic Segmentation)

**정상 CT + 질환 CT 전체(577명) 마스크 활용 장기 분할 — Fully-supervised / Semi-supervised 모두 가능**

| 항목 | 내용 |
|------|------|
| 입력 | 복부 CT 볼륨 |
| 출력 | 췌장 장기 분할 마스크 (Organ Segmentation Mask) |
| 레이블 데이터 (Fully-supervised) | 정상 473명 + 질환 104명 = **전체 577명** (모두 마스크 보유) |
| 학습 전략 | Fully-supervised (전체 마스크 활용) 또는 Semi-supervised (일부 마스크 활용) |
| 실현 가능성 | **높음 (상향)** |
| Novelty | **중간** |

**접근 방식 A — Fully-supervised (권장):**
- 정상 CT 473명 + 질환 CT 104명의 마스크를 모두 활용한 완전 지도 학습
- 577명 레이블 데이터로 강건한 췌장 분할 모델 구축 가능
- 정상/질환 췌장 모두 커버하는 일반화 성능 기대

**접근 방식 B — Semi-supervised (비교 실험용):**
- 질환 CT 104명 마스크만 레이블로 사용, 정상 CT 473명은 비레이블로 처리
- **Mean Teacher** / **Pseudo-labeling** 기반 준지도 학습
- Fully-supervised 대비 준지도 학습 효과를 비교 분석하는 ablation study에 활용

**장점:**
- 정상 CT 마스크 추가로 실현 가능성 및 모델 성능이 크게 향상
- 577명 데이터로 Fully-supervised + Semi-supervised 비교 실험 설계 가능
- 추후 병변 분할(Lesion Segmentation)로 확장 가능

**단점:**
- Fully-supervised 단독으로는 Novelty가 낮을 수 있음 → Semi-supervised와 비교 실험 구성 권장
- 구현 복잡도 중간

**참고 논문:**

| 논문 | 방법 | DOI |
|------|------|-----|
| Zhang et al. (2024) QDC-Net | Query-driven contrastive learning for semi-supervised segmentation | [10.1016/j.compbiomed.2024.108609](https://doi.org/10.1016/j.compbiomed.2024.108609) |
| Liu et al. (2022) GEPS-Net | Graph-enhanced semi-supervised pancreatic segmentation | [10.1088/1361-6560/ac80e4](https://doi.org/10.1088/1361-6560/ac80e4) |
| Li et al. (2024) EBC-Net | Entropy-based consistency for semi-supervised medical image segmentation | [10.1002/mp.17323](https://doi.org/10.1002/mp.17323) |

---

### 주제 3: 희귀 질환 CT 합성 (Rare Disease CT Synthesis)

**Diffusion Model 기반 희귀 췌장 질환 데이터 증강**

| 항목 | 내용 |
|------|------|
| 입력 | 소수 질환 CT 샘플 (~13명/클래스) |
| 출력 | 합성 CT 이미지 (Synthetic CT) |
| 모델 | Diffusion Model (DDPM, DDIM 등) |
| 실현 가능성 | **낮음** |
| Novelty | **높음** |

**문제점:**
- 클래스당 ~13개 이미지로 Diffusion model 학습에 극히 부족
- 고품질 합성을 위해서는 최소 수백 장 이상 필요
- 현재 데이터로는 실현 가능성이 매우 낮음

**결론: 현재 데이터로 단독 추진 비권장. 외부 공개 데이터셋(MSD, TCIA 등) 병합 시 고려 가능.**

---

### 주제 4: 자기지도 사전학습 (Self-supervised Pre-training)

**정상 CT 473명을 활용한 자기지도 사전학습 후 전이학습**

| 항목 | 내용 |
|------|------|
| 사전학습 데이터 | 정상 CT 473명 (Unlabeled) |
| 파인튜닝 데이터 | 질환 CT 104명 (Labeled) |
| 방법 | Masked Autoencoder (MAE), Contrastive Learning 등 |
| 역할 | 독립 주제 X → 보조 전략 (Enhancement Strategy) |
| 실현 가능성 | **높음** (보조 전략으로) |

**핵심 아이디어:**
- 레이블 없는 정상 CT 473명으로 표현 학습(Representation Learning) 수행
- 학습된 특징(Feature)을 이용해 소규모 레이블 데이터(104명)에서 Fine-tuning
- 주제 1(이진 분류) 또는 주제 2(분할)의 성능 향상을 위한 보조 전략으로 활용

**참고 논문:**
- Vagenas et al. (2025): CT intensity masking 기반 자기지도 사전학습으로 복부 CT 분석 성능 향상
  - 학회: *EMBC 2025*
  - DOI: [10.1109/EMBC58623.2025.11253827](https://doi.org/10.1109/EMBC58623.2025.11253827)

---

## 3. 최종 추천 요약

| 순위 | 주제 | 실현 가능성 | Novelty | 권장 여부 |
|------|------|------------|---------|-----------|
| **1순위** | 췌장 분할 (Fully-supervised + Semi-supervised 비교) | **매우 높음** | 중간 | **강력 권장** |
| **보조** | 자기지도 사전학습 (Self-supervised Pre-training) | 높음 | - | 1순위 보완 전략 |
| **참고** | 이진 분류 (Binary Classification) | 높음 | 낮음 | 보조 실험으로 활용 |
| **비권장** | 희귀 질환 CT 합성 (CT Synthesis) | 낮음 | 높음 | 현재 데이터로 불가 |

> **마스크 데이터 추가로 인한 변경:** 정상 CT 473명의 마스크가 확보됨에 따라 Fully-supervised 학습이 가능해졌습니다.  
> 577명 전체 마스크를 활용한 Fully-supervised 모델과 Semi-supervised 모델의 비교 실험을 핵심 기여로 설계할 수 있습니다.

### 권장 연구 설계

```
[Step 1] 자기지도 사전학습 (선택적)
  - 데이터: 정상 CT 473명 (마스크 미활용)
  - 방법: MAE 또는 Contrastive Learning 기반
  - 목표: 복부 CT 표현 학습 (사전학습 인코더)

[Step 2] 췌장 분할 — Fully-supervised (메인 실험)
  - 데이터: 정상 CT 473명 + 질환 CT 104명 (전체 577명 마스크 활용)
  - 방법: nnU-Net, TransUNet, Swin UNETR 등
  - 백본: Step 1 사전학습 인코더 또는 ImageNet 사전학습

[Step 3] 췌장 분할 — Semi-supervised (비교 실험)
  - 레이블 데이터: 질환 CT 104명 (마스크 활용)
  - 비레이블 데이터: 정상 CT 473명 (마스크 미활용)
  - 방법: Mean Teacher / Pseudo-labeling
  - 목적: Fully-supervised 대비 Semi-supervised 성능 차이 분석

[보조 실험] 이진 분류
  - 데이터: 정상 473명 vs 질환 104명
  - 사전학습 모델로 Fine-tuning
  - 스크리닝 성능 평가
```

---

## 4. 참고 문헌

1. **Viriyasaranon V et al. (2023)**  
   Self-Supervised Learning for Pancreatic Disease Classification from Small-Scale CT Dataset.  
   *Cancers*, 15(13), 3392.  
   DOI: [10.3390/cancers15133392](https://doi.org/10.3390/cancers15133392)

2. **Zhang et al. (2024)**  
   QDC-Net: Query-Driven Contrastive Network for Semi-supervised Pancreatic Segmentation.  
   *Computers in Biology and Medicine*, 108609.  
   DOI: [10.1016/j.compbiomed.2024.108609](https://doi.org/10.1016/j.compbiomed.2024.108609)

3. **Liu et al. (2022)**  
   GEPS-Net: Graph-Enhanced Semi-supervised Pancreatic Segmentation.  
   *Physics in Medicine & Biology*, 67(16).  
   DOI: [10.1088/1361-6560/ac80e4](https://doi.org/10.1088/1361-6560/ac80e4)

4. **Li et al. (2024)**  
   EBC-Net: Entropy-Based Consistency for Semi-supervised Medical Image Segmentation.  
   *Medical Physics*, 51, mp.17323.  
   DOI: [10.1002/mp.17323](https://doi.org/10.1002/mp.17323)

5. **Vagenas D et al. (2025)**  
   Self-supervised Pre-training with CT Intensity Masking for Abdominal CT Analysis.  
   *EMBC 2025*.  
   DOI: [10.1109/EMBC58623.2025.11253827](https://doi.org/10.1109/EMBC58623.2025.11253827)

---

*원본 이슈: [GYO-9](/GYO/issues/GYO-9)*
