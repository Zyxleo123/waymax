# Failure Detection VLA

`train_vla/train_fd_vla.py`는 현재 장면(과 미래 ego 궤적)을 보고 **롤아웃 결과**를 분류하는 VLA를 학습한다.

출력은 세 갈래다.

1. 텍스트: `success` / `collision` / `offroad` / `goal not reached`
2. collision이면 관련 객체 ID
3. offroad이면 관련 road-edge 세그먼트 ID

모델은 `model/vla/failure_detection_vla.py`의 `FailureDetectionVLA`다. Gemma causal LM + 장면 토큰 + 두 개의 ID 헤드.

```
success cache  ──┐
                 ├─ 50/50 batch ── features + prompt/answer
failure cache  ──┘                      │
                                        ▼
                              SceneTokenizer (32 tokens)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Gemma LM            object ID head      road-edge ID head
           (reason text)         (collision)           (offroad)
                    │                   │                   │
                    └─────────── loss = L_lang + L_obj + L_road
```

---

## 모듈 구조

`FailureDetectionVLA`는 시간축을 쌓지 않는 **단일 시점** 모델이다. 입력은 gold FM과 같은 ego-frame 벡터 특징이다.

### 1. Scene tokenizer

`model/vla/modules/scene_tokenizer.py`

| 입력 | 인코더 |
|---|---|
| `ego_state` `[B, 5]` | MLP |
| `goal_xy` + `remaining_timesteps` `[B, 3]` | MLP |
| `other_states` `[B, N, 15]` | MLP + position embedding |
| `map_features` `[B, S, P, 25]` | PointNet |
| `traffic_light_features` `[B, L, 11]` | MLP |
| `ego_trajectory` `[B, T, 5]` | `TrajectoryEncoder` (기본 on) |

학습된 32개 query가 위 토큰에 cross-attention 해서 **`[B, 32, llm_hidden]`** 장면 토큰을 만든다. 이어서 `SelfAttentionLayers` 4층을 한 번 더 통과한다.

`ego_trajectory`는 preprocess의 정규화 궤적이 아니라, 로더가 cache log에서 따로 만든 **미래 25 step, stride 2** 궤적이다. 실패 예제에서는 실패 직후 움직임이 여기 들어간다.

### 2. Gemma prefix

32개 장면 토큰을 LLM 입력 **맨 앞**에 끼운다.

```
[scene tokens | prompt tokens | answer tokens]
```

구현은 pad dummy id를 앞에 붙인 뒤, embedding hook으로 dummy 자리를 장면 토큰으로 교체한다. language loss는 **answer 토큰만** 본다 (`labels=-100` on scene/prompt).

기본 LLM은 `google/gemma-4-E2B-it`. `freeze_llm=True`이면 backbone은 고정하고 LoRA만 학습한다 (`r=16`, `alpha=32`).

### 3. ID heads

장면 토큰을 context로 쓰는 별도 헤드다. LLM hidden과 독립적으로 돌아간다.

**Collision object**

- `other_states` → MLP → object tokens `[B, N, D]`
- scene tokens에 cross-attention
- linear → `[B, N]` logits
- invalid object는 mask

**Offroad road-edge**

- `map_features` → PointNet → segment tokens `[B, S, D]`
- scene tokens에 cross-attention
- linear → `[B, S]` logits
- map type 15, 16 (road edge)만 후보

Gold ID는 `aux["critical_object_ids"]`, `aux["critical_road_segment_ids"]`다. candidate id (`object_ids` / `map_ids`)와 매칭되는 **슬롯 위치**에 CE를 건다. 매칭이 없으면 그 헤드 loss는 0이다 (success, GNR, ID 없는 샘플).

### Trainable 파라미터

기본 설정 (`freeze_llm` + LoRA)에서 학습되는 것:

- `scene_tokenizer`, `scene_token_self_attention`
- `object_tokenizer`, `object_cross_attention`, `object_id_head`
- `map_tokenizer`, `map_cross_attention`, `road_segment_id_head`
- LLM의 `lora_*` 가중치

`state_dict(only_trainable=True)`도 이 범위만 저장한다.

---

## 데이터

두 종류의 cache를 섞는다.

| | Success | Failure |
|---|---|---|
| cache | WOMD `sim_state_cache_npz` | 실패 롤아웃이 있는 cache |
| 라벨 JSON | 없음 | `failure_result_root`의 per-scenario JSON |
| answer 텍스트 | `"success"` | `"collision"` / `"offroad"` / `"goal not reached"` |
| gold object/road ID | `-1` | JSON의 `collision_vehicle_id` / `offroad_road_edge_segment_id` |
| anchor | `{0,10,20,...,80}` 균등 | `failure_step` **직전**에서 샘플 |

Failure JSON에서 `_pick_failure_reason`이 reason과 `failure_step`을 고른다. collision과 offroad가 둘 다 있으면 더 이른 쪽.

Failure anchor:

```
uniform{ max(0, failure_step - 50)  ,  failure_step - 1 }
```

`failure_max_anchor_offset` 기본값 50. 실패가 일어나기 **전** 상태를 보게 한다. gold FM finetune처럼 `{0,10,20,...}`에 묶이지 않는다.

한 배치는 대략 **절반 success, 절반 failure**다. `batch_size`가 홀수면 번갈아 한 쪽을 하나 더 넣는다. 그다음 배치 안에서 shuffle한다.

공통 prompt:

```
Given the driving scenario, classify the rollout outcome.
Answer with exactly one of: success, offroad, collision, goal not reached.
```

전처리는 gold FM과 같다. ego-frame, `ego_state` / `other_states` / `map_features` / TL / `goal_xy`. 추가로 `ego_trajectory`는 log에서 stride-2로 뽑은 미래 궤적이다.

---

## 학습 한 step

1. 배치에서 `features`, `prompts`, `answers`, `aux`를 받는다.
2. prompt/answer를 tokenize. answer에 EOS를 붙인다 (`max_answer_length=16`).
3. `FailureDetectionVLA.forward(...)`
   - scene tokens 인코딩
   - Gemma teacher forcing → `language_loss`
   - object/road logits → `object_loss`, `road_segment_loss`
4. \(\mathcal{L} = L_{\text{lang}} + L_{\text{obj}} + L_{\text{road}}\)
5. AdamW, warmup (+ `max_steps`가 있으면 cosine decay), grad clip 1.0

평가 (`eval_every`)는 generate로 텍스트 exact match를 보고, collision/offroad 샘플에서 ID accuracy도 잰다.

메트릭:

- `loss`, `language_loss`, `object_loss`, `road_segment_loss`
- `reason_exact_match`, `failure_reason_exact_match`, `success_exact_match`
- `collision_object_id_accuracy`, `offroad_road_segment_accuracy`

---

## 추론 출력

`generate_predictions`:

| 키 | 형식 | 의미 |
|---|---|---|
| `answer` / `failure_reason` | `list[str]` | greedy decode 텍스트 |
| `critical_object_ids` | `[B]` int | 예측 객체 ID. 후보 없으면 `-1` |
| `critical_road_segment_ids` | `[B]` int | 예측 road-edge 세그먼트 ID. 후보 없으면 `-1` |

planner (`fm-fd`)는 이 출력을 써서 실패 월드만 FiLM CFG로 보정한다. reason 텍스트가 collision/offroad/GNR이면 해당 gold 조건을 FM policy에 넣고 resample/sample 한다.

체크포인트:

```
<output_dir>/<run_name>/
  training_config.json
  checkpoints/step_XXXX.pt
  llm/
  tokenizer/
```

---

## 실행

```bash
python -m train_vla.train_fd_vla \
  --success_cache_dir /path/to/sim_state_cache_npz \
  --failure_cache_dir /path/to/failure_sim_state_cache \
  --failure_result_root /path/to/exp \
  --model_type failure_detection
```

`--success_anchor_steps 0,10,20,30,40` 과 `--failure_max_anchor_offset 50`으로 시점 샘플링을 바꿀 수 있다.
