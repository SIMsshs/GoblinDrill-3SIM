# 고블린 드릴 덱 AI 학습 안내서

이 문서는 다음 덱을 대상으로 한다.

1. Goblin Demolisher
2. Tesla
3. Fire Spirit
4. Skeletons
5. Poison
6. Evo Snowball
7. Evo Goblin Drill
8. Hero Knight

사용자가 적은 카드는 7장이므로, 공개 리플레이에서 실제로 발견되는 같은 조합의
8번째 카드인 **Poison을 우선 가정했다**. 8번째 카드가 Poison이 아니라면 학습을
시작하기 전에 `configs/decks/goblin_drill_evo_hero.json`을 수정해야 한다. 카드 순서는
리플레이 검색에는 영향을 주지 않지만, 실제 게임 연결에서는 게임 덱 화면의 순서와
설정 순서를 맞추는 편이 안전하다.

## 결론부터: 권장 학습 순서

이 프로젝트에서 가장 효율적인 출발점은 무작위 행동에서 강화학습을 시작하는 것이
아니다. 다음 세 단계가 기본 경로다.

1. **전체 모방학습:** 여러 덱의 공개 리플레이로 배치, 수비, 엘릭서 관리 같은 공통
   기본기를 배운다.
2. **정확한 덱 미세조정:** 두 진화와 Hero Knight 형태까지 정확히 같은 덱의
   리플레이만 사용해 작은 학습률로 특화한다.
3. **시뮬레이터 강화학습:** 모방 모델에서 시작해 RoyaleSim에서 대전하고 PPO로
   개선한다. 모방 모델과 멀어지지 않게 KL 패널티를 둔다.

모방학습도 신경망을 학습하는 딥러닝이다. 여기서 “후반 딥러닝”은 정확히는
**모방학습으로 초기화한 Transformer 정책을 PPO 자기 대전으로 추가 학습**한다는
뜻이다. 무작위 PPO부터 시작하면 8개의 고수준 행동과 576개 위치 조합에서 유용한
전략을 우연히 찾는 데 많은 계산을 낭비한다.

첫 5,000개 공개 리플레이를 검사했을 때 이 정확한 8장/형태 조합은 2경기였다. 전체
252,238경기에서도 약 100경기 수준일 것으로 추정되므로, 표적 경기만 처음부터
학습하면 과적합하기 쉽다. 전체 기본기 학습 뒤 표적 덱을 미세조정하는 이유다. 실제
개수는 `rows` 단계가 `selected N goblin drill evo hero + ...`로 출력한다.

## GitHub 외에 필요한 것

- **Python 3.12:** 저장소의 지원 범위가 3.12이다.
- **학습용 컴퓨터:** CPU로 기능 확인은 가능하지만 본 학습은 매우 느리다. NVIDIA
  GPU 8GB 이상을 실용적인 시작점으로 보고, 12GB 이상이면 배치 크기 조정이 편하다.
- **메모리와 저장 공간:** RAM 16GB 이상, 빈 디스크 10~20GB 이상을 권장한다.
  원본 데이터는 약 825MB이고, 전처리 행과 체크포인트가 추가된다.
- **인터넷:** 최초 패키지와 공개 리플레이 다운로드에 필요하다.
- **RoyaleSim 0.1.22 / RoyaleImitate 0.2.13:** 카드 형태와 20Hz 행동을
  시뮬레이터에 재현한다.
- **선택 사항인 클라우드 GPU:** 로컬 NVIDIA GPU가 없으면 GPU 가상머신을 빌려
  동일한 명령을 실행할 수 있다. GitHub는 코드를 보관하지만 장시간 학습을 대신하지
  않는다. GitHub Actions는 실행시간과 GPU 제약 때문에 본 학습용으로 권장하지 않는다.

`.pt` 체크포인트, 공개 데이터, 전처리 결과는 `.gitignore`에 들어 있어 GitHub에
올라가지 않는다. 중요한 체크포인트는 별도 드라이브나 클라우드 저장소에 복사한다.

## GitHub를 처음 쓰는 경우

GitHub의 원본 저장소를 바로 수정하는 것이 아니라, 자신의 계정에 **Fork(내 복사본)**를
만들고 그 안의 별도 **branch(작업 줄기)**에 코드를 올리는 것이 안전하다.

1. <https://github.com/hastylmao/Hasty-CR>에서 `Fork`를 눌러 자신의 계정에
   `Hasty-CR`을 만든다.
2. Codex의 GitHub 연결 설정에서 새 Fork에 대한 접근을 허용한다.
3. 이 로컬 저장소에서 원본을 `upstream`, 내 Fork를 `origin`으로 등록한다.

```powershell
git remote rename origin upstream
git remote add origin https://github.com/<내-GitHub-아이디>/Hasty-CR.git
git push -u origin codex/goblin-drill-training
```

용어는 다음 네 개만 먼저 알면 된다.

- **repository:** 코드 프로젝트 전체
- **branch:** 본 코드와 분리된 작업 줄기
- **commit:** 특정 시점의 저장 기록
- **push:** 로컬 commit을 GitHub로 업로드

이후 같은 branch에 새 변경을 올릴 때는 다음 순서다.

```powershell
git status
git add <수정한-파일>
git commit -m "설명"
git push
```

데이터와 체크포인트가 `git status`에 보이지 않는 것은 정상이다. 대용량 결과를 억지로
`git add -f` 하지 않는다. Fork가 연결되면 원본 저장소에 반영을 제안하는 Pull Request를
만들 수 있지만, 개인 학습에는 내 Fork의 branch만 있어도 충분하다.

## 처음 한 번만 설치하기 — Windows PowerShell

저장소 폴더에서 다음을 실행한다.

```powershell
py -3.12 -m venv .venvs\drill
.\.venvs\drill\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

NVIDIA GPU가 있으면 먼저 <https://pytorch.org/get-started/locally/>에서 Windows,
Pip, Python, 자신의 CUDA 환경을 선택해 표시된 PyTorch 설치 명령을 실행한다. 그 뒤:

```powershell
python -m pip install -r requirements-drill.txt
python -m pip install -e . --no-deps
python tools\drill_pipeline.py doctor
python -m pytest tests\test_deck_config.py tests\test_model_tokens_abilities.py -q
```

`doctor` 결과에서 패키지가 모두 `OK`이고, GPU 학습을 원한다면 `GPU:` 줄에 실제 GPU
이름이 보여야 한다. `CUDA not available`이면 CPU 전용 PyTorch이거나 드라이버/CUDA용
PyTorch 조합이 맞지 않은 것이다.

PowerShell이 스크립트 실행을 막는 경우 활성화 없이 매번 다음처럼 직접 실행해도 된다.

```powershell
.\.venvs\drill\Scripts\python.exe tools\drill_pipeline.py doctor
```

아래의 모든 `python`도 `.\.venvs\drill\Scripts\python.exe`로 바꿀 수 있다.

## 1단계 — 공개 리플레이 받기

```powershell
python tools\drill_pipeline.py data
```

중간에 끊겨도 같은 명령을 다시 실행하면 크기와 SHA-256이 맞는 파일은 건너뛰고
나머지를 이어받는다. 빠른 기능 확인만 하려면 `--parts 1`을 쓸 수 있지만, 정확한
표적 덱의 검증/테스트 분할이 없을 수 있으므로 본 학습에는 전체 데이터가 필요하다.

## 2단계 — 리플레이를 학습 행으로 변환하기

```powershell
python tools\drill_pipeline.py rows --workers 6
```

이 단계는 모든 정확한 표적 덱 경기와 최대 20,000개의 다른 덱 경기를 선택한다.
다른 경기들은 기본기 사전학습에 사용되고, `target=true` 행만 미세조정에 사용된다.
기본값의 `--live-compat` 관측은 실제 화면 판독기가 알 수 없는 시뮬레이터 내부 정보
를 제거하므로 나중에 실게임으로 옮기기 쉽다.

CPU 코어가 적으면 `--workers 2`, 많으면 6~10 정도로 조정한다. 이 단계가 끝나면
`data/rs_rows_drill/vocab.json`과 여러 `shard-*.npz`가 생긴다.

## 3단계 — 전체 모방학습

```powershell
python tools\drill_pipeline.py pretrain
```

결과:

```text
runs/tokens/drill_base_bc/best.pt
runs/tokens/drill_base_bc/last.pt
runs/tokens/drill_base_bc/log.jsonl
```

`best.pt`는 표적 덱 검증 점수(카드 정확도 + 가까운 타일 정확도 + 능력 정확도)를
기준으로 선택된다. 기다리기 행동은 매우 많기 때문에 35%만 보존하고, 실행할 때 그
확률 편향을 다시 보정한다.

GPU 메모리 부족 오류가 나면 배치를 낮춘다.

```powershell
python tools\drill_pipeline.py pretrain --batch 256
```

## 4단계 — 정확한 덱 미세조정

```powershell
python tools\drill_pipeline.py finetune
```

이 명령은 `drill_base_bc/best.pt`에서 시작해 정확한 덱/형태의 행만 사용한다. 학습률은
`5e-5`로 낮추고, 희귀한 연장전·타워 파괴 상태를 약하게 재표본화한다. Hero Knight
능력 누르기는 일반 카드 플레이보다 훨씬 드물기 때문에 게이트 손실 가중치를 4로 둔다.

결과의 주 체크포인트는 다음이다.

```text
runs/tokens/drill_target_bc/best.pt
```

로그에서 볼 항목:

- `target card`: 올바른 손패 칸을 고른 비율
- `near`: 프로가 둔 위치의 1타일 이내 비율
- `ability`: 능력을 눌러야 하는 행에서 올바른 능력을 고른 비율
- `target per phase`: 일반/더블 엘릭서/연장전과 타워 상태별 성능

훈련 정확도보다 `val`과 `val_target`을 본다. 에포크가 진행되는데 검증 점수가 계속
나빠지면 `epochs`를 줄이거나 `lr`을 더 낮춘다.

## 5단계 — 보지 않은 경기에서 평가하기

먼저 같은 테스트 경기에서 기록된 사람 플레이의 시뮬레이터 기준선을 잰다.

```powershell
python tools\drill_pipeline.py eval --pro --matches 100
```

그다음 AI를 잰다.

```powershell
python tools\drill_pipeline.py eval --matches 100
```

평가는 경기 태그 해시로 미리 떼어 둔 `test` 경기만 사용한다. 상대는 실제 경기의
행동을 같은 시간에 재생하고, AI 쪽만 교체한다. 출력에는 승수, 95% 신뢰구간, 무승부,
평균 크라운 차이가 나온다. 표적 테스트 경기가 100개보다 적으면 발견된 경기만 쓴다.
표본이 작을 때 승률 1~2경기 차이는 큰 의미가 없으므로 크라운 차이와 행동 정확도도
같이 본다.

평가 기본값은 실제 클라이언트 지연을 흉내 낸 21틱 지연과 공개 관측으로 다시 만든
보드를 사용한다. 이는 숨겨진 시뮬레이터 상태를 AI가 훔쳐보는 것을 막는다.

## 6단계 — PPO 자기 대전

모방 모델이 정상적으로 카드를 내고 평가도 통과한 뒤에만 실행한다.

```powershell
python tools\drill_pipeline.py selfplay
```

각 반복은 다음을 수행한다.

1. 표적 덱으로 48경기를 샘플링한다.
2. 처음에는 기록 행동을 재생하는 상대 75%와 반응형 상대 25%를 섞는다.
3. 20회에 걸쳐 반응형 상대를 75%까지 늘리고, 상대 판단 주기도 1초에서
   0.5초로 강화한다. 처음부터 전패하는 상대만 주어 학습 신호가 사라지는 것을 막는다.
4. 반응형 경기 절반은 표적 덱 리플레이의 상대 덱, 나머지 절반은 전체 공개
   데이터에서 뽑은 지원 가능한 덱을 사용한다.
5. 반응형 경기에서는 두 덱의 시작 순서를 다시 섞는다. 기록 상대는 기록 행동이
   손패와 맞아야 하므로 원래 리플레이 순서를 보존한다.
6. 승/무/패를 주 보상으로 두되 크라운 차이와 남은 타워 체력 차이를 작은 보조
   신호로 사용한다. 기본 가중치는 `0.7 / 0.1 / 0.2`이다.
7. 원래 표적 모방 모델에 대한 KL 패널티 `0.05`로 급격한 전략 붕괴를 막는다.
8. 5회마다 보지 않은 경기에서 강한 반응형 상대 기준으로 평가한다.

결과는 `runs/selfplay/drill_selfplay_curriculum/it000.pt`, `it001.pt`, ...에 저장된다.
정기 평가에서 승/무/패 점수가 가장 좋고, 동점이면 크라운 차이가 좋은 체크포인트를
`best.pt`로 자동 보존한다. 선택 근거는 `best.json`, 모든 반복 기록은 `log.jsonl`에서
확인할 수 있다. 마지막 파일이 자동으로 최선인 것은 아니므로 실제 연결 단계에서는
우선 `best.pt`를 사용한다. 서로 다른 보상이나 커리큘럼 설정을 기존 실행 이름에
섞으려 하면 실행을 거부하므로 새 `--name`을 사용한다.

처음에는 짧게 확인하는 편이 좋다.

```powershell
python tools\drill_pipeline.py selfplay --name drill_selfplay_smoke_v3 `
  --iters 1 --matches 8 --workers 4 --eval-every 1 --eval-matches 13
```

정상 동작과 로그를 확인한 뒤 기본 40회로 늘린다. 중단 후 같은 이름으로 다시 실행하면
존재하는 `itNNN.pt` 다음 반복부터 이어진다.

## AI가 실제로 보는 것과 하는 것

### 관측

- 최대 48개 보드 객체: 카드/유닛 ID, 아군 여부, 정규화 좌표, 체력 비율, 비행/건물,
  실게임에서 얻을 수 있는 최대 체력 특징
- 아군/적군 좌우 프린세스 타워와 킹 타워 체력 6개
- 엘릭서, 경기 시간, 더블 엘릭서, 연장전 상태
- 현재 손패 4장과 다음 카드
- 상대가 보여 준 카드 8장, 최근 카드 4장과 경과 시간
- 최대 3개의 능력 버튼: 사용 가능, 사용 완료, 비용, 카드 ID, 쿨다운

각 객체는 토큰이 되고 Transformer가 토큰 사이 관계를 읽는다. 따라서 “해골이 몇 개
있다”만 보는 평면 관측과 달리 어떤 카드가 어느 위치에 있고 체력이 얼마인지 구분한다.

### 행동

- 게이트 8개: 기다리기 1 + 손패 카드 4 + 히어로/챔피언 능력 버튼 최대 3
- 카드 행동일 때 18×32, 총 576개 배치 타일
- 엘릭서, 손패, 배치 가능 구역, 능력 사용 가능 여부는 실행 전에 마스킹

카드 이름으로 직접 행동을 예측하지 않고 **현재 손패의 몇 번째 칸인지** 예측하므로
카드 순환이 바뀌어도 작동한다. 능력 행동은 별도 버튼이라 배치 타일을 예측하지 않는다.

### 학습 신호

모방학습은 기록된 기다리기/손패 칸/능력 버튼과 배치 위치를 정답으로 사용한다. 가치
헤드는 10초 뒤의 타워 체력 변화와 최종 승패도 예측한다. 시뮬레이터가 기록 행동을
거부했거나 이미 기록과 크게 갈라진 이후의 행은 기본적으로 신뢰하지 않는다.

자기 대전은 최종 승/무/패를 70%로 두고, 크라운 차이 10%와 남은 타워 체력 차이
20%를 더한 값을 현재 가치 예측과 비교해 advantage를 만든다. 이는 시작 모델이 강한
상대에게 전패하더라도 1크라운 근소 패배와 3크라운 패배를 구분하기 위한 것이다.
어떤 승리도 무승부보다, 어떤 무승부도 패배보다 높은 보상을 갖는다. “드릴을 내면
보상”처럼 특정 카드 행동을 직접 보상하지는 않는다. 그런 규칙은 AI가 승리보다 보상
규칙의 허점을 찾게 만들 수 있다.

## Hero Knight 능력에 관한 중요한 제한

오프라인 리플레이 변환, 모방학습, RoyaleSim 평가, 자기 대전은 능력 버튼을 완전히
지원한다. 공개 데이터의 `activate_ability` 이벤트를 능력 0~2번으로 변환하고,
RoyaleSim의 사용 가능/비용/쿨다운 상태로 불가능한 능력을 마스킹한다.

현재 실제 기기 연결의 메모리/화면 판독기는 능력 버튼의 사용 가능 상태와 안전한 탭
좌표를 제공하지 않는다. 그래서 `live_bridge.py`와 `live_play.py`는 능력 로그릿을
의도적으로 마스킹한다. 즉 현재 체크포인트는 시뮬레이터에서 Hero Knight 능력을 배우고
사용하지만, 실게임 자동화에서는 잘못된 화면 위치를 누르지 않도록 능력을 누르지 않는다.
실게임 능력 자동화에는 기기 해상도별 버튼 좌표 캘리브레이션과 쿨다운 판독을 먼저
추가·검증해야 한다. 이 제한은 학습 결과 자체에는 영향을 주지 않는다.

실게임 자동화는 계정과 서비스 약관에 영향을 줄 수 있다. 우선 시뮬레이터와 친선전에서
검증하고, 래더 자동 실행은 별도의 명시적인 결정과 안전 점검 뒤에 진행한다.

## 내 플레이로 모방학습하려면

가장 먼저 권장하는 경로는 공개 데이터다. 이미 카드 시각, 틱, 진영, 좌표, 덱 형태,
결과가 정규화되어 있기 때문이다. 단순 전투 기록 API의 덱/승패만으로는 모방학습할 수
없다. **각 카드를 언제, 어느 좌표에 냈는지와 능력을 언제 눌렀는지**가 필요하다.

자기 플레이를 추가하려면 IL_Replay와 같은 입력 스키마로 변환해야 한다.

- 경기 메타데이터: 태그, 모드, 양쪽 8장과 base/evolution/hero 형태, 결과, 타워 상태
- 행동 이벤트: 20Hz 틱, 진영, `play_card` 또는 `activate_ability`, 카드 키, 월드 좌표
- 같은 경기의 메타데이터와 행동 파일을 동일한 part 번호의 Parquet로 저장

그 데이터를 `data/il_replay/replays`와 `actions`에 추가한 다음 `rows`부터 다시 실행한다.
단, 화면 녹화만으로 자동 변환하려면 카드/좌표/시간 인식 오차를 검수해야 한다. 오표기
수백 개는 적은 표적 데이터에서 큰 악영향을 준다. 공개 데이터로 기준 모델을 먼저 만든
뒤, 자신의 정확히 검수한 경기 수가 충분해졌을 때 작은 학습률로 추가 미세조정하는 것이
안전하다.

## 덱을 바꾸는 방법

`configs/decks/goblin_drill_evo_hero.json`의 각 항목에는 두 이름이 있다.

- `engine`: RoyaleSim 카탈로그 이름
- `replay`: RoyaleAPI/IL_Replay 기본 슬러그
- `form`: `base`, `evolution`, `hero` 중 하나

8번째 카드가 Poison이 아니라면 해당 한 줄을 실제 카드로 바꾸고 다음을 확인한다.

```powershell
python -m sim.deck_config configs\decks\goblin_drill_evo_hero.json
```

그 뒤 기존 행을 덮어쓰도록 `python tools/drill_pipeline.py rows --overwrite`를 실행한다.
이미 만든 행은 `vocab.json`에 덱 설정 메타데이터를 기록하므로, 다른 덱 행과 섞지 않는다.

## 자주 생기는 문제

### `no target val rows` 또는 표적 테스트 경기가 없음

`--parts 1` 같은 일부 데이터만 받은 경우다. `python tools/drill_pipeline.py data`로 전체
데이터를 받고 `rows`를 다시 실행한다. 그래도 0이면 8번째 카드나 진화/Hero 형태가 실제
덱과 다른 것이다.

### CUDA out of memory

`pretrain --batch 256`, 필요하면 128로 낮춘다. 모델 구조는 바꾸지 않아도 된다.

### 학습은 되는데 기다리기만 함

반드시 `tools/drill_pipeline.py`의 평가/자기 대전 경로를 사용한다. 이 경로는 학습 때
기다리기 행을 줄인 비율(`wait_keep`)을 실행 시 보정한다.

### 자기 대전 뒤 성능이 떨어짐

마지막 체크포인트를 쓰지 말고 평가가 가장 좋았던 `itNNN.pt`로 돌아간다. 학습률을
`5e-6`로 낮추거나 KL을 `0.1`로 높인다. 표적 모방 체크포인트는 덮어쓰지 않는다.
`collect_reward_std`가 계속 0에 가깝거나 반응형 경기 승률이 계속 0이면 전체 실행을
계속하지 말고 상대 비율·속도와 보상 로그를 먼저 점검한다.

### 시작 전 최종 확인표

- `doctor`에서 모든 패키지 OK
- 전체 데이터의 replay/action part 개수가 같음
- `rows` 출력에서 정확한 표적 덱 경기가 0보다 큼
- 전체 모방 `best.pt` 생성
- 표적 미세조정 `best.pt` 생성
- 사람 기준선과 AI 평가를 같은 테스트 경기에서 실행
- 그 다음에만 자기 대전 시작
