# 복부 CT 연구 스터디 (Abdominal CT Research Study)

> **연구 주제:** 복부 CT 영상 + 혈액 검사 데이터를 활용한 멀티모달 간경화(Cirrhosis) 중증도 예측  
> **준비자:** Paper Writing Team | **날짜:** 2026-04-09 | **데이터베이스:** PubMed, Web  
> **원본 이슈:** [GYO-11] Literature Review: Multimodal Cirrhosis Severity Prediction: CT + Lab Data

---

## 목차 (Table of Contents)

1. [레퍼런스 테이블](#1-레퍼런스-테이블)
2. [정상인에서의 Child-Pugh / MELD 점수 산출 가능 여부](#2-정상인에서의-child-pugh--meld-점수-산출-가능-여부)
3. [연구 갭 요약](#3-연구-갭-요약)
4. [추천 투고처 (학회 및 저널)](#4-추천-투고처-학회-및-저널)
5. [검색 로그](#5-검색-로그)

---

## 1. 레퍼런스 테이블

> **포함 기준:** CT 영상 ± 혈액 검사 데이터; AI/ML/딥러닝(Deep Learning); 간섬유화(Liver Fibrosis)/간경화(Cirrhosis) 병기(Staging) 또는 합병증 예측; 2016–2026; 동료심사(Peer-reviewed) 또는 프리프린트(Preprint)  
> **제외 기준:** CT 비교 없는 MRI 단독 연구; 비AI 임상 연구; 간 이외 장기 연구

| # | 저자 (연도) | 연구 과제 | 데이터 | 모델 | 주요 결과 | 특이사항 | DOI |
|---|---|---|---|---|---|---|---|
| 1 | Xie et al. (2025) | 초기 간경화 환자에서의 불량 예후(복수·간성뇌증·정맥류 출혈) 예측 | n=243, 2개 기관; CT + 라디오믹스(Radiomics) + 임상 혈액 검사 텍스트 | TMF-LCNet: 삼중 모달 융합(Triple-modal Fusion) — CT + 라디오믹스 + 임상 텍스트 | AUC 0.797 (학습), 0.747 (외부 검증); 최고 보정(Calibration) 성능 | CT 기반 영상이 기여도 최대, 라디오믹스 기여도 최소; 첫 CT + 라디오믹스 + 임상 텍스트 삼중 모달 모델 | [10.1007/s00261-025-05045-0](https://doi.org/10.1007/s00261-025-05045-0) |
| 2 | Yin et al. (2021) | 간섬유화(Liver Fibrosis) 병기 분류 F0–F4 | n=252; 조영증강 CT 문맥상(Portal Venous Phase); 조직학적 확진 | 딥 CNN (LFS Network) + Grad-CAM | AUC 0.92/0.89/0.88 (F2+/F3+/F4) | Grad-CAM 분석: F0 → 간 표면 집중, F4 → 비장+실질(Parenchyma) 집중; 상복부 전체 스캔 활용 | [10.1007/s00330-021-08046-x](https://doi.org/10.1007/s00330-021-08046-x) |
| 3 | Yin et al. (2022) | 라디오믹스를 활용한 간섬유화 병기 분류 | n=252; 위 연구와 동일 CT 코호트 | 앙상블 ML 라디오믹스 (간+비장 특징) | AUC 0.92/0.81/0.85; 간+비장 결합이 간 단독 대비 AUC +7% | 비장이 핵심 정보 출처임을 검증; 딥러닝 어텐션 맵과 일치 | [10.3390/diagnostics12020550](https://doi.org/10.3390/diagnostics12020550) |
| 4 | Kwon et al. (2021) | HBV 간경화 환자에서의 간기능 비대상화(Hepatic Decompensation) 예측 | n=1,027; HBV 보상성 간경화(Compensated Cirrhosis); CT 자동 분할(Auto-segmentation) | 딥러닝 분할 → 간-비장 부피 비율(LSVR) → Cox 회귀 | LSVR <2.9: 3년 비대상화율 16.7% vs 2.5% (p<0.001); Child-Pugh/MELD와 독립적 | 검토된 연구 중 최대 CT 기반 예후 코호트; LSVR이 기존 점수 이상의 추가 정보 제공 | [10.3348/kjr.2021.0348](https://doi.org/10.3348/kjr.2021.0348) |
| 5 | Yu et al. (2022) — CHESS1701 | 보상성 간경화 환자에서의 비대상화 예측 | n=689 (5개 기관); CT + 임상 특징 | 딥러닝 비장 분할(DL Spleen Segmentation) + LASSO-Cox NIT | C-index ≥0.84; 고위험군 vs 저위험군 HR 5.8–7.3; 저위험군 3년 비대상화율 ≤1% | 다기관 연구; 혈청 기반 모델 능가; Baveno VII 기준과 경쟁적 성능 | [10.1016/j.jhepr.2022.100575](https://doi.org/10.1016/j.jhepr.2022.100575) |
| 6 | Yoo et al. (2025) | 간섬유화 선별 검사 (유/무 이진 분류) | n=169; 동시 생검 + CT (조영/비조영) | ML 라디오믹스 (로지스틱 회귀 + 특징 선택) | AUC 0.7833; 비조영 CT가 조영증강 CT 능가 | 비조영 CT로 무증상 섬유화 선별 가능성 확인; 소규모 샘플의 한계 | [10.1186/s12880-025-01823-w](https://doi.org/10.1186/s12880-025-01823-w) |
| 7 | Tamura et al. (2025) | 정상 비장 부피 기준값 확립; F4 간경화 탐지 | n=4,732 건강인 + n=136 생검 확진; 비조영 CT | 딥러닝 분할 (Dice=0.95) + Z-score 모델 | F4 탐지 AUC 0.73; 비장비대(Splenomegaly) 유병률 F1→0%, F4→74% (Z≥+1 기준) | 최대 규모 정상 비장 부피 데이터셋; 일본 인구 대상 | [10.1186/s12876-025-04383-z](https://doi.org/10.1186/s12876-025-04383-z) |
| 8 | Radiya et al. (2023) | 체계적 문헌 고찰: 간 CT에서의 머신러닝 | 191편 연구 | CNN, 딥러닝, 하이브리드 모델 리뷰 | 분할(Segmentation)·섬유화 병기·전이 예측에서 ML 높은 성능 확인 | 전향적(Prospective) 검증 시급; 임상의-CS 협업 필수 | [10.1007/s00330-023-09609-w](https://doi.org/10.1007/s00330-023-09609-w) |
| 9 | Dang et al. (2026) | 내러티브 리뷰: 간섬유화에서의 라디오믹스 + 딥러닝 | — | CT, MRI, 초음파 대상 리뷰 | 병기 분류·예후 예측·원인 감별 연구 정리 | 멀티모달 융합 + 설명가능성(Explainability)을 우선 미래 방향으로 제시 | [10.3390/jimaging12020082](https://doi.org/10.3390/jimaging12020082) |
| 10 | Wang et al. (2022) | 12개월 내 식도 정맥류 출혈(Esophageal Variceal Bleeding) 예측 | 간경화 코호트; 내시경 영상 + 임상 데이터 (MELD, CPS, APRI, FIB-4) | EfficientNet (내시경) + AutoML 스태킹(Stacking) | 정확도 0.932; AUC > MELD·Child-Pugh·APRI·FIB-4 각각 | 간경화 분야 첫 멀티모달 AutoML; 복합 점수 대비 영상+혈액 융합의 우월성 검증 | [10.1007/s10278-022-00724-6](https://doi.org/10.1007/s10278-022-00724-6) |
| 11 | Ko et al. (2021) | Gd-EOB-DTPA MRI에서의 불충분 간 조영증강 예측 | n=214; 만성 간질환; 혈액 검사 데이터 (CPS, MELD-Na) | SVM, RF, DT, KNN with 간기능 검사 | SVM (빌리루빈 + 알부민): AUC 0.93; CPS (0.89)·MELD-Na (0.90) 능가 | ML 선택 개별 혈액 검사 항목이 복합 점수 능가; 특징 중요도 분석 포함 | [10.1007/s00261-021-03308-0](https://doi.org/10.1007/s00261-021-03308-0) |

> **참고 (시드 PMID 27885969):** PubMed 확인 결과 "36th International Symposium on Intensive Care and Emergency Medicine (Brussels, 2016)" 학술대회 초록집으로 확인 — 연구 주제와 무관한 학술대회 자료. 과제 설명의 PMID 오류로 판단하여 문헌 고찰에서 제외.

---

## 2. 정상인에서의 Child-Pugh / MELD 점수 산출 가능 여부

### 점수 구성 요소

**Child-Pugh 점수** (Pugh et al., 1973): 각 1–3점으로 채점되는 5개 변수:
1. 혈청 빌리루빈(Serum Bilirubin, mg/dL)
2. 혈청 알부민(Serum Albumin, g/dL)
3. 프로트롬빈 시간(Prothrombin Time) / INR
4. 복수(Ascites): 없음 / 경증 / 중증
5. 간성 뇌증(Hepatic Encephalopathy): 없음 / 1–2등급 / 3–4등급

합계: 5–15점 → Class A (5–6점 = 보상성, Compensated), B (7–9점), C (10–15점 = 비대상성, Decompensated)

**MELD 점수** (Malinchoc et al., 2000; MELD-Na 업데이트):
- 변수: 빌리루빈 (mg/dL), 크레아티닌 (mg/dL), INR, 혈청 나트륨
- 공식: MELD = 3.78 × ln[빌리루빈] + 11.2 × ln[INR] + 9.57 × ln[크레아티닌] + 6.43
- 범위: 6–40 (ln 계산 전 1.0 미만 값은 1.0으로 대체)

### 정상인에게 이 점수를 산출할 수 있는가?

**수치적으로는 가능하나 — 임상적으로는 무의미합니다.**

정상 혈액 수치를 가진 건강한 성인 (빌리루빈 ~0.8, 알부민 >3.5, INR ~1.0, 크레아티닌 ~0.9 mg/dL; 복수 없음; 간성 뇌증 없음) 기준:
- **Child-Pugh = 5점 (Class A)** — 최솟값
- **MELD ≈ 6점** — 최솟값 하한

**정상인에게 이 점수가 의미 없는 이유:**

1. **개발 맥락:** 두 점수 모두 *간경화 환자*(Child-Pugh) 또는 *이식 대기 말기 간질환 환자*(MELD)를 대상으로 개발·검증된 도구로, 정상 인구에 대한 검증 연구가 존재하지 않습니다.
2. **의미론적 불일치:** Child-Pugh Class A = *잘 보상된 간경화*. 이 라벨을 건강한 사람에게 부여하는 것은 범주 오류(Categorical Error)에 해당합니다.
3. **MELD 최솟값 = 6:** 이 최솟값은 이식 대기 순위 산정을 위한 인위적 하한선으로, 비질환 개인에게 생리적 의미가 없습니다.
4. **임상적 역치 부재:** 이 점수에 연동된 임상적 처치(이식 등록, 예방적 정맥류 결찰술, 이뇨제 투여)는 확진된 간질환 환자에게만 적용 가능합니다.

**문헌 근거:**
- Radiya et al. (2023) — 딥러닝 모델이 CPS/MELD를 결과 레이블로 사용할 때 확진된 간 병리가 전제됨을 명시
- Wang et al. (2022) — MELD와 Child-Pugh는 *간경화 코호트 내에서도* 열등한 예측 변수였음; 질환 외 적용 시 유용성은 더욱 제한적
- Ko et al. (2021) — ML 선택 개별 혈액 수치(빌리루빈 + 알부민)가 복합 CPS(AUC 0.93 vs 0.89)를 능가 → 복합 점수가 변별 신호를 희석시킬 수 있음을 시사

**결론:** Child-Pugh와 MELD 점수는 정상인에 대해 *수치적으로 계산*할 수 있으며 각각 최솟값(5점, 6점)이 산출됩니다. 그러나 이 계산은 임상적 타당성이 없습니다 — 이 도구들은 비간경화 대상자에 대한 해석 체계가 없는 질환 특이적(Disease-specific) 평가 도구입니다. 정상인에게 적용하는 것은 질환 특이적 채점 도구의 오용에 해당합니다.

---

## 3. 연구 갭 요약

**새로운 멀티모달 CT + 혈액 검사 데이터 융합 연구를 위한 미충족 갭:**

### 갭 1 — 진정한 엔드-투-엔드 CT + 혈액 데이터 결합 융합 [최우선 과제]
현황: CT 단독 또는 혈액 검사 단독, 또는 별도 모델 출력값의 후기 융합(Late Fusion).  
**갭:** CT 영상 특징과 구조화된 혈액 검사 텐서(빌리루빈, 알부민, INR, 혈소판, 나트륨)의 *중간 또는 결합 특징 융합(Intermediate/Joint Feature Fusion)*을 통합된 엔드-투-엔드 구조로 수행한 연구 없음.  
*기회:* CT 패치와 혈액 검사 토큰 시퀀스 간 크로스-어텐션(Cross-attention)이 적용된 결합 인코더(Joint Encoder) 설계.

### 갭 2 — 연속형 Child-Pugh / MELD 점수 예측
현황: 이진(비대상화 유/무) 또는 순서형 섬유화 병기(F0–F4) 분류.  
**갭:** CT + 혈액 검사에서 Child-Pugh 또는 MELD를 *연속형 회귀 출력(Continuous Regression Output)*으로 예측한 연구 없음.  
*기회:* 임상 점수를 직접 예측하는 회귀 헤드(Regression Head) 설계 → 임상 워크플로우에 직접 대응.

### 갭 3 — 다원인 및 다제조사 일반화
현황: 단일 기관, 단일 원인(주로 HBV). 예외: Kwon 2021 (n=1,027, HBV 단독).  
**갭:** 복수 CT 제조사에 걸친 다원인(HBV, HCV, NASH, ALD, PBC) 교차 검증 부재.  
*기회:* 연합 학습(Federated Learning) 또는 도메인 적응 학습(Domain-adaptive Training)을 활용한 다기관·다원인 훈련.

### 갭 4 — 비조영 CT + 혈액 검사의 선별 검사 활용
현황: 대부분의 모델이 조영증강 CT 사용. Yoo et al. (2025)이 비조영 CT AUC 0.7833 확인.  
**갭:** 비조영 CT와 혈액 검사의 *결합 모델링* 미존재.  
*기회:* 일차의료/자원 제한 환경에서 적용 가능한 저비용 선별 모델.

### 갭 5 — 종단적(Longitudinal) 혈액 추세 통합
현황: 모든 연구가 단일 시점(Cross-sectional) 혈액 수치 사용.  
**갭:** CT와 함께 시간적 궤적(알부민 감소율, 6–12개월 혈소판 추세)이 통합된 연구 없음.  
*기회:* 최종 시점의 CT + 종단적 혈액 시퀀스에 대한 LSTM 또는 트랜스포머(Transformer) 적용.

### 갭 6 — 추가 CT 파생 특징으로서의 체성분(Body Composition)
현황: TMF-LCNet (Xie 2025)은 지방 조직을 포함하나 골격근 미포함.  
**갭:** 근감소증(Sarcopenia) — L3 골격근 단면적(Cross-Sectional Area), 근육 밀도, 내장 지방 — CT 파생 가능하나 어떤 융합 모델에도 미포함.  
*기회:* 근감소증이 간경화 비대상화의 독립적 예측 인자; CT 파생 특징으로 용이하게 추가 가능.

### 갭 7 — 영상 및 혈액 모달리티를 연결하는 설명가능성(Explainability)
현황: Yin 2021이 Grad-CAM 사용 (영상 단독). 모달리티 간 기여도(Cross-modal Attribution) 없음.  
**갭:** CT 영상 영역과 혈액 검사 특징 기여도를 단일 예측에 연결하는 SHAP 또는 통합 그래디언트(Integrated Gradients) 적용 연구 없음.  
*기회:* 규제 승인 및 의사 수용(Physician Adoption)에 필수적.

### 갭 8 — 전향적 임상 검증(Prospective Clinical Validation)
현황: 검토된 모든 연구가 후향적(Retrospective).  
**갭:** AI와 FIB-4, 간 강성 측정(Liver Stiffness Measurement), 또는 임상 워크플로우에서의 Baveno VII 기준을 비교하는 전향적 검증 없음.  
*기회:* 임상 적용을 위한 필수 단계; 실제 표준 치료 도구와의 비교 가능.

---

## 4. 추천 투고처 (학회 및 저널)

### 최우선 국제 저널 (Top International Journals)

| 투고처 | 약칭 | 특이사항 |
|---|---|---|
| Medical Image Analysis | MedIA | 의료 영상 AI 최고 저널; IF ~10 |
| IEEE Transactions on Medical Imaging | TMI | 영상 처리+AI 통합 연구에 최적 |
| Radiology | Radiology | 임상 영향력 최대; IF ~12 |
| Radiology: Artificial Intelligence | Radiology AI | AI 중심 방사선과학 전문지 |
| Hepatology | Hepatology | 간질환 임상 연구 최고 저널; IF ~12 |
| Journal of Hepatology | J Hepatol | 유럽간학회지; 임상+기초 균형 |
| European Radiology | Eur Radiol | 유럽 방사선과학; CT/MRI 연구 활발 |

### 주요 국제 학술대회 (Top International Conferences)

| 학술대회 | 약칭 | 특이사항 |
|---|---|---|
| Medical Image Computing and Computer Assisted Intervention | MICCAI | 의료 영상 AI 최대 국제 학술대회 |
| IEEE International Symposium on Biomedical Imaging | ISBI | 생의학 영상 처리 전문 학술대회 |
| Radiological Society of North America | RSNA | 북미 방사선학회; 임상 AI 발표 활발 |
| European Congress of Radiology | ECR | 유럽 방사선학회 연례 학술대회 |

---

## 5. 검색 로그

| # | 데이터베이스 | 검색어 | 날짜 | 검색 결과 수 |
|---|---|---|---|---|
| 1 | PubMed | CT deep learning liver cirrhosis staging fibrosis prediction | 2026-04-09 | 8 |
| 2 | PubMed | multimodal liver fibrosis cirrhosis CT laboratory data fusion deep learning | 2026-04-09 | 1 |
| 3 | PubMed | Child-Pugh MELD score prediction imaging machine learning | 2026-04-09 | 5 |
| 4 | PubMed | liver spleen volume ratio cirrhosis prognosis CT | 2026-04-09 | 11 |
| 5 | PubMed | liver fibrosis staging MRI elastography deep learning neural network | 2026-04-09 | 4 |
| 6 | PubMed | hepatic decompensation prediction deep learning CT laboratory | 2026-04-09 | 1 |
| 7 | PubMed | PMID 직접 조회: 34014382, 35204639, 40576670, 41745446, 37171491, 34564961, 36204707, 41168695, 36279027, 34647145 | 2026-04-09 | 10 |
| 8 | PubMed | PMID 직접 조회: 27885969, 40665242 | 2026-04-09 | 2 |
| 9 | medRxiv/bioRxiv | radiology 카테고리, 2023–2026 | 2026-04-09 | 오류 502 (서비스 불가) |
| 10 | Web 검색 | Child-Pugh MELD score healthy normal individuals components | 2026-04-09 | 보조 자료 수집 |

---

*모든 논문 메타데이터 출처: PubMed (미국 국립의학도서관, National Library of Medicine). 모든 인용 논문에 DOI 링크 포함.*

*번역 기준: 의학/AI 전문 용어는 한글 번역 + 영문 병기(括弧). BibTeX 인용 및 DOI 링크는 학술 표준에 따라 원문 유지.*
