# SIXray-D — Dataset Statistics

실제 annotation 내용은 포함하지 않고, 집계 수치만 정리했습니다.

## 클래스

| ID | 이름 |
|---|---|
| 1 | Gun |
| 2 | Knife |
| 3 | Wrench |
| 4 | Pliers |
| 5 | Scissors |

5개 클래스, 카테고리 순서/ID는 train/val/test 전체에서 고정.

## Split별 이미지 수

| Split | 이미지 수 |
|---|---|
| val | 1,005 |
| test | 836 |
| train (few-shot 이전 원본) | 이 문서에는 미기재 — 원본 데이터셋에서 직접 확인 필요 |

val/test는 서로 겹치지 않는 별도 이미지 집합이며, 어떤 few-shot support 이미지와도 겹치지
않음을 [`data_generation/validate_sixray_exact_clean.py`](../../data_generation/validate_sixray_exact_clean.py)로
검증했습니다.

## Few-shot split 구성 (exact-K-instance)

- Shots: 1 / 5 / 10
- Seeds: 1–10
- 클래스당 정확히 K개의 annotation instance (클래스 간 이미지 공유 허용 — 즉 이미지 수는
  `5 x K` 이하일 수 있음)

예시로 seed 1, 10-shot의 경우 클래스별 annotation 10개씩(총 50개)이 40장의 이미지에 걸쳐
분포되어 있음을 확인했습니다 (일부 이미지가 2개 이상 클래스의 annotation을 포함).
