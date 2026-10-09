# Gold-ID Flow Matching Finetuning

`train_diffusion/finetune_fm_gold_policy.py`는 pretrained `FlowMatchingPolicy`에 **실패 원인(gold ID) 조건**을 붙이는 단계다. 장면 인코더와 UNet 속도장은 거의 고정하고, **failure encoder + FiLM**만 학습한다.

핵심 아이디어는 다음과 같다.

- Pretrain: 장면(과 선택적 goal)만 cross-attention으로 보고 전문가 궤적의 속도장을 학습한다.
- Gold finetune: 같은 장면 조건에, “이 collision object / offroad segment / unmet goal을 피한 성공 궤적”을 FiLM으로 추가로 조건한다.

학습 목표는 여전히 **linear flow matching MSE** 하나다. 별도의 collision / offroad / behavior cloning 항은 없다.

```
success_for_failure_cache npz
        │
        ▼
preprocess (anchor 시점, ego-frame, 정규화)
        │
        ├─ scene tokens (cross-attn, freeze)
        └─ failure / gold vector (FiLM, train)
                │
                ▼
        UNet velocity  v_θ(x_t, t, cond1, cond2)
                │
                ▼
        MSE( v_θ , ε − x0 )
```

---

## 데이터

데이터는 WOMD 원본 tfrecord가 아니라 **`success_for_failure_cache`** 다.  
`scripts/make_success_cache_for_failures.py`가 실패로 어노테이션된 시나리오에 대해 **logged (성공) 궤적**과 gold ID를 묶어 만든 npz다.

기본 경로:

```
.../exp/<sim_exp>/success_for_failure_cache/success_for_failure_shard_*.sim_state_cache.npz
```

로더는 아래 라벨이 있는 row만 쓴다.

| `failure_reason` | 값 | 필요한 gold ID |
|---|---|---|
| collision | 1 | `critical_object_ids >= 0` |
| offroad | 2 | `critical_road_segment_ids >= 0` |
| goal not reached | 3 | 없음 (goal 좌표 + remaining time 사용) |
| none / 그 외 | 0 | 버림 |

한 시나리오마다 `anchor_steps` (기본 `0, 10, 20, 30, 40`) 중 하나를 균등 샘플한다. 그 시점이 현재 상태이고, **그 이후 WOMD log ego 궤적**이 generation target이다. goal step은 시나리오 마지막 timestep이다.

`batch.aux`는 wandb/메트릭용이며 loss에는 들어가지 않는다.

---

## 모델 입력

한 배치 `B`에 대해, 로더가 `batch.features`로 넣는 텐서와 역할은 아래와 같다.  
좌표는 전부 **anchor 시점 ego 프레임**, 길이/속도는 config 상수로 정규화되어 있다.

### Target (출력으로 맞출 것)

| 키 | shape | 의미 |
|---|---|---|
| `ego_trajectory` | `[B, H, 5]` | 미래 ego 궤적. `H = predict_horizon` (기본 25). 채널은 `(x, y, vx, vy, yaw)`. `x,y`는 `ego_range`(100m)로, `vx,vy`는 `max_velocity`(25m/s)로, yaw는 \(\pi\)로 나눈 값. **이 텐서가 flow matching의 \(x_0\)** 이다. |

학습 step에서는 `ego_trajectory[:, :predict_horizon, :]`만 잘라 쓴다. 내부 FM은 `[B, 5, H]`로 transpose한다.

### Scene condition (`cond1`, cross-attention, freeze)

| 키 | shape | 의미 |
|---|---|---|
| `ego_state` | `[B, 5]` | 현재 ego `(x, y, vx, vy, yaw)`. 정의상 원점/헤딩이므로 위치·yaw는 0에 가깝다. |
| `goal_xy` | `[B, 2]` | 시나리오 끝 목표 위치 (ego-frame, `/ ego_range`). `goal_conditioned=True`일 때만 scene tokenizer에 들어간다. |
| `remaining_timesteps` | `[B, 1]` | `(goal_step - anchor_step) / 100`. |
| `other_states` | `[B, N, 15]` | 주변 객체. `(x, y, vx, vy, yaw, length, width)` + object-type one-hot 8. 위치는 `/ max_range`(30m). |
| `other_valid` | `[B, N]` | 해당 객체가 유효한지. |
| `map_features` | `[B, S, P, 25]` | 도로 세그먼트. 점마다 `(x, y, dir_x, dir_y)` + map-type one-hot 21. |
| `map_valid` | `[B, S, P]` | 세그먼트 점 유효 마스크. |
| `traffic_light_features` | `[B, L, 11]` | `(x, y)` + TL state one-hot 9. |
| `traffic_light_valid` | `[B, L]` | 신호등 유효 마스크. |

`SceneTokenizer`가 위 토큰들을 32개 query로 압축해 **`cond1`: `[B, 32, cond_dim]`** 을 만든다. 이 경로는 freeze다.

### Failure / gold condition (`cond2`, FiLM, train)

| 키 | shape | 의미 |
|---|---|---|
| `failure_reason` | `[B]` | `{0,1,2,3}`. |
| `critical_object_ids` | `[B]` | collision일 때 피해야 할 객체 ID. 아니면 `-1`. |
| `critical_road_segment_ids` | `[B]` | offroad일 때 피해야 할 road-edge 세그먼트 ID. 아니면 `-1`. |
| `object_ids` | `[B, N]` | `other_states` 축과 대응하는 객체 ID. 현재 로더는 `0 .. N-1`. |
| `map_ids` | `[B, S]` | `map_features` 세그먼트의 roadgraph point id. |

정책은 reason에 따라 **하나만** 임베딩한다.

| reason | 인코더 | 입력 | 출력 |
|---|---|---|---|
| collision | `collision_object_encoder` (MLP) | `other_states`에서 `object_ids == critical_object_ids`인 행 `[B, 15]` | `[B, cond_dim]` |
| offroad | `offroad_segment_encoder` (PointNet) | `map_features`에서 `map_ids == critical_road_segment_ids`인 세그먼트 `[B, P, 25]` | `[B, cond_dim]` |
| goal not reached | `goal_encoder` (MLP) | `concat(goal_xy, remaining_timesteps)` → `[B, 3]` | `[B, cond_dim]` |

세 벡터를 더한 것이 **`cond2`: `[B, cond_dim]`**,  
collision/offroad/GNR 중 하나면 **`cond2_mask`: `[B] = True`**.  
mask가 False면 FiLM은 identity다.

이 배치의 데이터는 usable gold label만 들어오므로, 정상적으로는 `cond2_mask`가 거의 전부 True다.

---

## 한 step에서 실제로 일어나는 일

1. `compute_condition(features)` → `(cond1, cond2, cond2_mask)`
2. target \(x_0 =\) `ego_trajectory[:, :H, :]` 를 `[B, 5, H]`로 변환
3. \(t \sim U(0,1)\), \(\varepsilon \sim \mathcal{N}(0,I)\)
4. \(x_t = (1-t)\, x_0 + t\, \varepsilon\)
5. UNet이 \(v_\theta(x_t, t, \text{cond1}, \text{cond2})\) 예측. timestep embedding 입력은 **`t * 1000`**.
6. 타깃 속도 \(v^\star = \varepsilon - x_0\)
7. \(\mathcal{L} = \mathrm{mean}((v_\theta - v^\star)^2)\)

즉 출력 형식은 궤적 샘플이 아니라 **속도장에 대한 스칼라 MSE**다. 궤적 자체는 추론 때 ODE를 적분해서 얻는다.

학습 루프가 추가로 로깅하는 값:

- `loss`
- `failure_mask_rate` — `cond2_mask`의 평균
- `collision_rate` / `offroad_rate` / `goal_not_reached_rate`
- `grad_norm`

---

## 무엇이 학습되고 무엇이 고정되는지

Trainable 경로 이름:

```
collision_object_encoder
offroad_segment_encoder
goal_encoder          # FiLM용. scene tokenizer의 goal_tokenizer가 아님
down_film_blocks
mid_film_blocks
up_film_blocks
```

그 외 (`scene_tokenizer`, UNet residual/attention, stem 등)는 `optax.set_to_zero()`로 freeze.

Pretrain ckpt에는 FiLM/failure encoder가 없다. overlapping restore로 로드하고, 없는 텐서는 random init 후 이 단계에서만 학습한다. `goal_conditioned`는 pretrain과 같게 맞춰야 scene tokenizer 동작이 일치한다.

Optimizer: AdamW + warmup cosine decay. Grad clip, EMA (`ema_decay=0.999`)를 유지한다.

---

## 체크포인트 / 추론 때의 출력

학습 산출물은 `output_dir` 아래 Orbax ckpt다.

```
<run_dir>/
  training_config.json
  latest/          # params_state, ema_params, opt_state, metadata.json
  epoch_XXXX/
```

`metadata.json`에 `enable_film=true`, `goal_conditioned`, `num_denoising_steps`가 저장된다.

추론 (`sample` / planner) 때의 모델 출력은 학습 target과 같은 형식이다.

| | shape | 의미 |
|---|---|---|
| policy 출력 | `[B, H, 5]` | 정규화된 미래 ego 궤적 `(x, y, vx, vy, yaw)` |
| ODE | `t=1 → 0`, `num_denoising_steps` 분할 | 시작은 \(\mathcal{N}(0,I)\). embedding 스케일은 계속 `t * 1000` |
| CFG | FiLM `cond2`에만 적용 | \(v = v_{\text{uncond}} + w(v_{\text{cond}} - v_{\text{uncond}})\). 장면 `cond1`은 항상 켜짐 |

플래너가 이 정규화 궤적을 월드 좌표로 되돌린 뒤 Waymax에 넣는다.

---

## 실행

```bash
python -m train_diffusion.finetune_fm_gold_policy \
  --pretrained_ckpt /path/to/pretrain_ckpt \
  --cache_dir /path/to/success_for_failure_cache \
  --goal-conditioned
```

pretrain을 `--no-goal-conditioned`로 돌렸다면 finetune도 `--no-goal-conditioned`로 맞춰야 한다.
