# SIXray-D

이 폴더에는 **SIXray-D 데이터셋의 원본 이미지/annotation은 물론, 이를 변환·가공한 어떤 파생
annotation도 포함되어 있지 않습니다.** SIXray-D의 라이선스는 원본 재배포뿐 아니라 파생물
재배포도 명시적으로 금지하고 있어서, 다른 두 데이터셋(TXL-PBC, HIT-UAV)과 달리 few-shot split
JSON을 이 저장소에 올리지 않았습니다.

대신 이 폴더에는 SIXray-D를 이 프로젝트의 실험 파이프라인에 연결하기 위해 실제로 작성한
**코드**와, 결과 해석에 필요한 **집계 통계**만 있습니다.

## 포함된 파일

- `generate_fewshot_splits.py` — 클래스당 정확히 K개 annotation instance를 뽑는
  exact-K-instance 샘플링 스크립트. SIXray-D 원본 `train.json`(COCO 포맷)을 정당하게 확보한
  뒤 로컬에서 직접 실행해야 합니다. 클래스별 독립 샘플링 방식(이미지가 여러 클래스에 걸쳐
  공유될 수 있음)은 원본 [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark)
  저장소의 `datasets/kshot_split.py`에 있는 샘플링 방식을 그대로 따르며, seed 1~10 /
  1·5·10-shot 전체를 한 번에 생성하도록 감싼 것입니다. 알고리즘 자체는 이 프로젝트가 새로
  고안한 것이 아니라 원본 저장소의 방식을 그대로 재구현한 것임을 밝힙니다.
- `register_sixray.py` — 생성된 split JSON을 Detectron2/CD-ViTO에 등록하는 코드
  (`cd_vito/register_sixray_exact_instance.py`와 동일, 이 프로젝트에서 직접 작성).
- `dataset_statistics.md` — 클래스 목록, val/test 이미지 수, split당 image/annotation 개수 등
  실제 annotation 내용 없이 집계 수치만 담은 문서.

## 재현 방법

1. SIXray-D 원본 데이터셋을 정당한 경로로 확보합니다.
2. `python generate_fewshot_splits.py --train-json <원본 train.json 경로> --out-dir <출력 경로>`로
   `train_{K}shot_seed{N}.json` (K=1/5/10, N=1..10)을 생성합니다.
3. `register_sixray.py`의 `SIXRAY_D_ROOT`를 실제 데이터 경로로 맞춘 뒤 등록합니다.

FT-FSOD(MMDetection)용 config는 `ft_fsod/generate_sixray_exact_clean_configs.py`가, CD-ViTO용
config/prototype은 `cd_vito/generate_sixray_exact_clean_configs.py` /
`cd_vito/build_prototypes_sixray_exact_clean.sh`가 동일한 파일명 규칙(`train_{K}shot_seed{N}.json`)을
전제로 두 경로를 연결합니다.
