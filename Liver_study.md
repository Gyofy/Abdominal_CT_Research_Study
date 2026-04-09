# 간경화 CT 연구 주제 분석 (Liver Cirrhosis CT Research Topic Analysis)

> 보유 데이터: 간경화 환자 복부 CT + 혈액검사 (Liver Cirrhosis Abdominal CT + Laboratory Data)  
> 작성일: 2026-04-09

---

## 목차

1. [데이터 현황 요약](#1-데이터-현황-요약)
2. [연구 주제 후보](#2-연구-주제-후보)
   - [주제 1: 멀티모달 엔드-투-엔드 간경화 중증도 분류 (Multimodal Cirrhosis Severity Classification)](#주제-1-멀티모달-엔드-투-엔드-간경화-중증도-분류)
   - [주제 2: 연속형 임상 점수 회귀 예측 (Continuous Clinical Score Regression)](#주제-2-연속형-임상-점수-회귀-예측)
   - [주제 3: 비조영 CT 기반 선별 검사 모델 (Non-contrast CT Screening)](#주제-3-비조영-ct-기반-선별-검사-모델)
   - [주제 4: 간·비장 장기 분할 (Liver/Spleen Segmentation)](#주제-4-간비장-장기-분할)
3. [최종 추천 요약](#3-최종-추천-요약)
4. [참고 문헌](#4-참고-문헌)

---

## 1. 데이터 현황 요약

| 구분 | 내용 |
|------|------|
| 영상 데이터 | 간경화 환자 복부 CT (조영 / 비조영 포함 가능) |
| 임상 데이터 | 혈액 검사 수치: 빌리루빈(Bilirubin), 알부민(Albumin), INR, 혈소판(Platelet), 나트륨(Sodium) 등 |
| 레이블 정보 | Child-Pugh 점수 / MELD 점수 / 합병증 유무 (복수·간성뇌증·정맥류 출혈 등) |
| 연구 방향 | CT 영상 + 혈액검사 멀티모달 융합(Multimodal Fusion) 기반 중증도 예측 |

**주요 특징:**
- CT 영상과 구조화된 혈액검사 수치를 동시에 보유 → 멀티모달 융합 연구 최적 환경
- Child-Pugh / MELD 등 임상 점수를 레이블로 활용 가능 → 분류(Classification) 및 회귀(Regression) 모두 가능
- 기존 문헌에서 CT 단독 또는 혈액검사 단독 연구가 대부분 → CT + 혈액검사 **엔드-투-엔드 융합** 연구의 큰 갭 존재

**제약 사항:**
- 간 장기 마스크(Liver Segmentation Mask) 보유 여부 추가 확인 필요
- 후향적(Retrospective) 데이터 → 전향적(Prospective) 검증은 향후 과제

---

## 2. 연구 주제 후보

### 주제 1: 멀티모달 엔드-투-엔드 간경화 중증도 분류

**CT 영상 + 혈액검사 크로스-어텐션 융합 기반 간경화 중증도 예측**

| 항목 | 내용 |
|------|------|
| 입력 | 복부 CT 슬라이스 또는 3D 볼륨 + 혈액검사 수치 벡터 |
| 출력 | 간경화 중증도 분류 (보상성 Compensated / 비대상화 Decompensated) 또는 Child-Pugh A/B/C |
| 모달리티 융합 방법 | 크로스-어텐션 (Cross-attention): CT 패치 토큰 ↔ 혈액검사 토큰 간 결합 인코더 |
| 실현 가능성 | **높음** |
| Novelty | **높음** |

**장점:**
- 기존 문헌에서 CT + 혈액검사 **결합 중간 융합(Intermediate/Joint Feature Fusion)**을 엔드-투-엔드로 수행한 연구 없음 → 명확한 연구 갭
- CT 영상 특징과 혈액검사 특징을 단일 모델 내에서 통합 학습 → 후기 융합(Late Fusion) 대비 우월한 성능 기대
- 임상적으로 실제 활용 가능한 스크리닝 도구

**단점:**
- 모달리티 정렬(Modality Alignment) 및 결측값(Missing Value) 처리 필요
- 구현 복잡도 높음

**참고 논문:**
- Xie et al. (2025): CT + 라디오믹스 + 임상 텍스트 삼중 모달 융합 (TMF-LCNet), AUC 0.797
  - DOI: [10.1007/s00261-025-05045-0](https://doi.org/10.1007/s00261-025-05045-0)
- Wang et al. (2022): 내시경 영상 + 임상 데이터 멀티모달 AutoML, AUC > 0.932
  - DOI: [10.1007/s10278-022-00724-6](https://doi.org/10.1007/s10278-022-00724-6)

---

### 주제 2: 연속형 임상 점수 회귀 예측

**CT + 혈액검사로 Child-Pugh / MELD 점수를 연속형 회귀 출력으로 직접 예측**

| 항목 | 내용 |
|------|------|
| 입력 | 복부 CT + 혈액검사 수치 (빌리루빈, 알부민, INR, 혈소판, 나트륨) |
| 출력 | Child-Pugh 점수 또는 MELD 점수 (연속형 회귀 출력) |
| 모델 | CNN/ViT 인코더 + 혈액검사 MLP + 회귀 헤드(Regression Head) |
| 실현 가능성 | **높음** |
| Novelty | **높음** |

**장점:**
- 기존 문헌 전체에서 CT + 혈액검사로 연속형 임상 점수를 직접 예측한 연구 없음 → 명확한 연구 갭
- 이진/순서형 분류 대비 임상 워크플로우에 직접 대응 (실제 점수 출력)
- 주제 1과 멀티태스킹(Multi-task Learning) 구조로 통합 가능

**단점:**
- 회귀 레이블(실제 점수 수치) 확보 필요 → 데이터 검토 선행
- 점수 분포 불균형 주의 (경증 케이스 편중 가능)

**참고 논문:**
- Ko et al. (2021): 혈액 검사 단독 SVM으로 간기능 예측, AUC 0.93
  - DOI: [10.1007/s00261-021-03308-0](https://doi.org/10.1007/s00261-021-03308-0)
- Kwon et al. (2021): CT 분할 기반 간-비장 부피 비율(LSVR)로 비대상화 예측
  - DOI: [10.3348/kjr.2021.0348](https://doi.org/10.3348/kjr.2021.0348)

---

### 주제 3: 비조영 CT 기반 선별 검사 모델

**비조영 CT (Non-contrast CT) + 혈액검사를 결합한 저비용 간섬유화 선별 모델**

| 항목 | 내용 |
|------|------|
| 입력 | 비조영 복부 CT + 혈액검사 수치 |
| 출력 | 간섬유화(Liver Fibrosis) 유/무 이진 분류 또는 섬유화 병기(F0–F4) |
| 적용 환경 | 일차의료·자원 제한 환경 (조영제 투여 불필요) |
| 실현 가능성 | **중간** (비조영 CT 확보 여부에 따름) |
| Novelty | **중간** |

**장점:**
- Yoo et al. (2025)이 비조영 CT 단독 AUC 0.7833 확인 → 혈액검사 결합 시 성능 향상 기대
- 조영증강 없이 적용 가능 → 광범위한 임상 적용 가능성
- 비조영 + 혈액검사 결합 모델 선례 없음

**단점:**
- 비조영 CT 데이터 별도 확보 필요 (조영 CT와 구분)
- 비조영 CT는 조직 대비(Tissue Contrast) 제한 → 성능 상한 존재

**참고 논문:**
- Yoo et al. (2025): 비조영 CT 라디오믹스, AUC 0.7833 (조영 CT 능가)
  - DOI: [10.1186/s12880-025-01823-w](https://doi.org/10.1186/s12880-025-01823-w)
- Tamura et al. (2025): 비조영 CT 딥러닝 분할로 F4 탐지 AUC 0.73
  - DOI: [10.1186/s12876-025-04383-z](https://doi.org/10.1186/s12876-025-04383-z)

---

### 주제 4: 간·비장 장기 분할

**복부 CT에서 간(Liver) 및 비장(Spleen) 자동 분할 → 부피 특징 추출 후 예후 예측**

| 항목 | 내용 |
|------|------|
| 입력 | 복부 CT 볼륨 |
| 출력 | 간 / 비장 분할 마스크 → 간-비장 부피 비율(LSVR) 등 정량적 특징 |
| 역할 | 독립 주제 X → 주제 1·2의 보조 전략 (특징 추출 파이프라인) |
| 실현 가능성 | **높음** (보조 전략으로) |
| Novelty | **낮음** (독립 주제로는 부족) |

**핵심 아이디어:**
- 딥러닝 자동 분할로 간·비장 부피를 정확히 측정 → LSVR 등 정량적 바이오마커(Biomarker) 추출
- 추출된 부피 특징을 주제 1(멀티모달 융합) 또는 주제 2(회귀)의 입력으로 활용
- nnU-Net 등 공개 모델을 파인튜닝하여 빠르게 구현 가능

**참고 논문:**
- Kwon et al. (2021): 딥러닝 분할 → LSVR < 2.9: 3년 비대상화율 16.7% vs 2.5%
  - DOI: [10.3348/kjr.2021.0348](https://doi.org/10.3348/kjr.2021.0348)
- Yu et al. (2022): 딥러닝 비장 분할 + LASSO-Cox, C-index ≥ 0.84
  - DOI: [10.1016/j.jhepr.2022.100575](https://doi.org/10.1016/j.jhepr.2022.100575)

---

## 3. 최종 추천 요약

| 순위 | 주제 | 실현 가능성 | Novelty | 권장 여부 |
|------|------|------------|---------|-----------|
| **1순위** | 멀티모달 엔드-투-엔드 융합 (CT + 혈액검사, 크로스-어텐션) | **높음** | **높음** | **강력 권장** |
| **2순위** | 연속형 임상 점수 회귀 예측 (Child-Pugh / MELD) | 높음 | 높음 | **권장** (1순위와 통합 가능) |
| **참고** | 비조영 CT 기반 선별 검사 모델 | 중간 | 중간 | 데이터 확인 후 결정 |
| **보조** | 간·비장 장기 분할 | 높음 | 낮음 | 1·2순위 보완 전략 |

> **핵심 기여 방향:** 기존 CT 단독 또는 혈액검사 단독 연구의 한계를 극복하는 **엔드-투-엔드 멀티모달 융합 모델** 설계.  
> 주제 1과 주제 2를 **멀티태스킹(Multi-task Learning)** 구조로 통합하면 단일 프레임워크에서 분류 + 회귀를 동시에 달성 가능.

### 권장 연구 설계

```
[Step 1] 간·비장 자동 분할 (보조 전략)
  - 데이터: 간경화 환자 CT
  - 방법: nnU-Net 파인튜닝
  - 목표: 간-비장 부피 비율(LSVR) 등 정량적 특징 추출

[Step 2] 멀티모달 융합 모델 (메인 실험)
  - 입력: CT 패치 특징 + 혈액검사 수치 벡터
  - 방법: 크로스-어텐션 기반 결합 인코더 (Joint Encoder)
  - 출력 헤드 A: 간경화 중증도 분류 (Child-Pugh A/B/C)
  - 출력 헤드 B: MELD 점수 연속형 회귀 예측

[Step 3] 설명가능성 분석 (보조 실험)
  - 방법: SHAP 또는 통합 그래디언트 (Integrated Gradients)
  - 목표: CT 영상 영역 기여도 + 혈액검사 항목 기여도 시각화

[비교 실험] CT 단독 / 혈액검사 단독 / 후기 융합 (Late Fusion) Ablation Study
  - Ablation으로 멀티모달 융합의 효과 정량화
```

---

## 4. 참고 문헌

1. **Xie et al. (2025)**  
   TMF-LCNet: Triple-modal Fusion for Liver Cirrhosis Complication Prediction (CT + Radiomics + Clinical Text).  
   DOI: [10.1007/s00261-025-05045-0](https://doi.org/10.1007/s00261-025-05045-0)

2. **Yin et al. (2021)**  
   Deep CNN for Liver Fibrosis Staging (F0–F4) from Contrast-enhanced CT.  
   DOI: [10.1007/s00330-021-08046-x](https://doi.org/10.1007/s00330-021-08046-x)

3. **Yin et al. (2022)**  
   Ensemble ML Radiomics (Liver + Spleen Features) for Liver Fibrosis Staging.  
   DOI: [10.3390/diagnostics12020550](https://doi.org/10.3390/diagnostics12020550)

4. **Kwon et al. (2021)**  
   Deep Learning Segmentation → Liver-Spleen Volume Ratio (LSVR) for Hepatic Decompensation Prediction.  
   DOI: [10.3348/kjr.2021.0348](https://doi.org/10.3348/kjr.2021.0348)

5. **Yu et al. (2022) — CHESS1701**  
   DL Spleen Segmentation + LASSO-Cox for Decompensation Prediction, C-index ≥ 0.84.  
   DOI: [10.1016/j.jhepr.2022.100575](https://doi.org/10.1016/j.jhepr.2022.100575)

6. **Yoo et al. (2025)**  
   ML Radiomics from Non-contrast CT for Liver Fibrosis Screening, AUC 0.7833.  
   DOI: [10.1186/s12880-025-01823-w](https://doi.org/10.1186/s12880-025-01823-w)

7. **Tamura et al. (2025)**  
   Deep Learning Spleen Segmentation for Normal Volume Reference; F4 Cirrhosis Detection AUC 0.73.  
   DOI: [10.1186/s12876-025-04383-z](https://doi.org/10.1186/s12876-025-04383-z)

8. **Radiya et al. (2023)**  
   Systematic Review: Machine Learning for Liver CT Analysis (191 studies).  
   DOI: [10.1007/s00330-023-09609-w](https://doi.org/10.1007/s00330-023-09609-w)

9. **Dang et al. (2026)**  
   Narrative Review: Radiomics + Deep Learning for Liver Fibrosis (CT, MRI, Ultrasound).  
   DOI: [10.3390/jimaging12020082](https://doi.org/10.3390/jimaging12020082)

10. **Wang et al. (2022)**  
    Multimodal AutoML: Endoscopy + Clinical Data for Esophageal Variceal Bleeding Prediction, Accuracy 0.932.  
    DOI: [10.1007/s10278-022-00724-6](https://doi.org/10.1007/s10278-022-00724-6)

11. **Ko et al. (2021)**  
    SVM with Lab Data (Bilirubin + Albumin) for Liver Function Prediction, AUC 0.93.  
    DOI: [10.1007/s00261-021-03308-0](https://doi.org/10.1007/s00261-021-03308-0)

---

*원본 이슈: [GYO-11](/GYO/issues/GYO-11)*
