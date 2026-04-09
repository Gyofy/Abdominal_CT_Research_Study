# 복부 CT를 활용한 의료 AI 연구 (Abdominal CT Research Study)

> **프로젝트:** 복부 CT 영상 기반 의료 AI 모델 연구  
> **작성일:** 2026-04-09

---

## 프로젝트 개요 (Project Overview)

본 프로젝트는 복부 CT (Computed Tomography) 영상과 임상 데이터를 활용하여 췌장(Pancreas) 및 간(Liver) 질환에 대한 의료 AI 모델을 연구·개발하는 것을 목표로 합니다.

보유 데이터를 기반으로 장기 분할(Organ Segmentation), 질환 분류(Disease Classification), 멀티모달 융합(Multimodal Fusion) 등 다양한 딥러닝(Deep Learning) 연구 주제를 탐색합니다.

---

## 보유 데이터 요약 (Dataset Summary)

| 데이터 구분 | 케이스 수 | 레이블 / 부가 정보 |
|------------|-----------|-------------------|
| 정상 복부 CT (Normal Abdominal CT) | 473명 | **췌장 마스크 (Pancreas Mask) 보유** |
| 췌장 질환 CT (Pancreatic Disease CT) | 104명 (8종 질환) | **병변 포함 췌장 마스크 (Lesion Mask) 보유** |
| 간경화 환자 CT + 혈액검사 (Liver Cirrhosis CT + Lab Data) | 미정 | CT 영상 + 구조화된 혈액 검사 수치 |

**주요 특징:**
- 정상 CT 473명 + 췌장 질환 CT 104명 전체에 췌장 마스크 보유 → Fully-supervised 분할 학습 가능
- 췌장 8종 질환 (각 ~13명/클래스): 이진 분류(Binary Classification) 및 준지도 학습(Semi-supervised Learning) 연구 가능
- 간경화 환자 CT + 혈액검사: 멀티모달 융합(Multimodal Fusion) 및 중증도 예측(Severity Prediction) 연구 가능

---

## 연구 주제 목록 (Research Topics)

### 1. 췌장 연구 — [Pancreas_study.md](Pancreas_study.md)

췌장 CT 영상 데이터(정상 473명 + 질환 104명)를 활용한 연구 주제 분석.

| 순위 | 주제 | 권장 여부 |
|------|------|-----------|
| 1순위 | 췌장 분할 (Fully-supervised + Semi-supervised 비교) | **강력 권장** |
| 보조 | 자기지도 사전학습 (Self-supervised Pre-training) | 보완 전략 |
| 참고 | 이진 분류 (Binary Classification) | 보조 실험 |
| 비권장 | 희귀 질환 CT 합성 (CT Synthesis) | 현재 데이터로 불가 |

자세한 내용: [Pancreas_study.md](Pancreas_study.md)

---

### 2. 간 연구 — [Liver_study.md](Liver_study.md)

간경화 환자 CT + 혈액검사 데이터를 활용한 멀티모달 연구 주제 분석.

| 순위 | 주제 | 권장 여부 |
|------|------|-----------|
| 1순위 | 멀티모달 엔드-투-엔드 융합 (CT + 혈액검사) | **강력 권장** |
| 2순위 | 연속형 임상 점수 회귀 예측 (Child-Pugh / MELD) | 권장 |
| 참고 | 비조영 CT 기반 선별 검사 모델 | 보조 실험 |
| 참고 | 간·비장 장기 분할 (Liver/Spleen Segmentation) | 보조 전략 |

자세한 내용: [Liver_study.md](Liver_study.md)

---

## 연구 방향 요약 (Research Direction Summary)

```
[췌장 연구]
  데이터: 정상 CT 473명 + 질환 CT 104명 (전체 577명, 마스크 보유)
  핵심: 췌장 분할 Fully-supervised vs Semi-supervised 비교 실험
  보조: 자기지도 사전학습 기반 특징 학습 → 분류·분할 성능 향상

[간 연구]
  데이터: 간경화 환자 CT + 혈액검사 (구조화 임상 데이터)
  핵심: CT + 혈액검사 크로스-어텐션 융합 기반 간경화 중증도 예측
  보조: Child-Pugh / MELD 연속형 회귀 예측 헤드 설계
```

---

## 참고 문서 (References)

- [Pancreas_study.md](Pancreas_study.md) — 췌장 CT 연구 주제 분석
- [Liver_study.md](Liver_study.md) — 간경화 CT 연구 주제 분석
