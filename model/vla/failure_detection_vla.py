from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

from model.vla.modules.attention import CrossAttentionLayers, SelfAttentionLayers
from model.vla.modules.mlp import MLP
from model.vla.modules.point_net import PointNet
from model.vla.modules.scene_tokenizer import SceneTokenizer


class FailureDetectionVLA(nn.Module):
	"""Gemma-based VLA for failure-reason text prediction plus auxiliary id heads.

	The model consumes the current-state vector features used by the standard
	Scene VLA pipeline, without any temporal stacking. It generates a textual
	failure reason and predicts the most relevant object index for collision or
	the most relevant road-edge segment id for offroad.
	"""

	def __init__(
		self,
		gemma_name: str = "google/gemma-4-E2B-it",
		ego_dim: int = 5,
		goal_dim: int = 3,
		other_dim: int = 15,
		map_dim: int = 25,
		tl_dim: int = 11,
		num_scene_tokens: int = 32,
		use_ego_trajectory: bool = True,
		trajectory_dim: int = 5,
		hidden_dim: int = 512,
		num_heads: int = 4,
		num_attn_layers: int = 4,
		num_object_types: int = 8,
		num_map_type_classes: int = 21,
		use_lora: bool = False,
		lora_r: int = 16,
		lora_alpha: int = 32,
		lora_dropout: float = 0.05,
		lora_target_modules: tuple[str, ...] | None = None,
	):
		super().__init__()
		self.ego_dim = int(ego_dim)
		self.goal_dim = int(goal_dim)
		self.other_dim = int(other_dim)
		self.map_dim = int(map_dim)
		self.tl_dim = int(tl_dim)
		self.num_scene_tokens = int(num_scene_tokens)
		self.hidden_dim = int(hidden_dim)
		self.num_heads = int(num_heads)
		self.num_attn_layers = int(num_attn_layers)
		self.num_object_types = int(num_object_types)
		self.num_map_type_classes = int(num_map_type_classes)
		self.vehicle_type_id = 1
		self.road_edge_type_ids = (15, 16)

		self.processor = AutoProcessor.from_pretrained(
			gemma_name,
			trust_remote_code=True,
		)
		self.tokenizer = self.processor.tokenizer

		self.llm = AutoModelForCausalLM.from_pretrained(
			gemma_name,
			torch_dtype=torch.bfloat16,
			trust_remote_code=True,
		)

		self._has_lora = False
		if use_lora:
			self._attach_lora(
				r=lora_r,
				alpha=lora_alpha,
				dropout=lora_dropout,
				target_modules=lora_target_modules,
			)

		if hasattr(self.llm.config, "text_config"):
			token_dim = self.llm.config.text_config.hidden_size
		else:
			token_dim = self.llm.config.hidden_size

		self.scene_tokenizer = SceneTokenizer(
			ego_dim=ego_dim,
			goal_dim=goal_dim,
			other_dim=other_dim,
			map_attr_dim=map_dim,
			tl_attr_dim=tl_dim,
			use_ego_trajectory=use_ego_trajectory,
			trajectory_dim=trajectory_dim,
			hidden_dim=hidden_dim,
			cond_dim=token_dim,
			num_tokens=num_scene_tokens,
			num_heads=num_heads,
			num_attn_layers=num_attn_layers,
		)
		self.scene_token_self_attention = SelfAttentionLayers(
			dim=token_dim,
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			num_layers=num_attn_layers,
		)

		self.object_tokenizer = MLP([other_dim, hidden_dim, hidden_dim, token_dim])
		self.object_cross_attention = CrossAttentionLayers(
			query_dim=token_dim,
			context_dim=token_dim,
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			num_layers=num_attn_layers,
		)
		self.object_id_head = nn.Linear(token_dim, 1)

		self.map_tokenizer = PointNet(map_dim, token_dim)
		self.map_cross_attention = CrossAttentionLayers(
			query_dim=token_dim,
			context_dim=token_dim,
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			num_layers=num_attn_layers,
		)
		self.road_segment_id_head = nn.Linear(token_dim, 1)

		self._scene_tokens_for_hook = None
		self._scene_start_for_hook = None
		self._scene_len_for_hook = None
		self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
			self._replace_scene_embeddings_hook
		)

	def _module_device(self) -> torch.device:
		return next(self.parameters()).device

	def _to_torch(self, value: Any, *, device: torch.device | None = None) -> Any:
		if device is None:
			device = self._module_device()
		if torch.is_tensor(value):
			return value.to(device=device)
		if isinstance(value, (int, float, bool)):
			return torch.as_tensor(value, device=device)
		if isinstance(value, np.ndarray):
			return torch.as_tensor(value, device=device)
		if hasattr(value, "__array__"):
			return torch.as_tensor(np.asarray(value), device=device)
		return value

	def _to_torch_feature_map(
		self,
		input_features: Mapping[str, Any],
		*,
		device: torch.device | None = None,
	) -> dict[str, Any]:
		if device is None:
			device = self._module_device()
		return {key: self._to_torch(value, device=device) for key, value in input_features.items()}

	@classmethod
	def from_pretrained(
		cls,
		checkpoint_path: str | Path,
		*,
		map_location: str | torch.device = "cpu",
		device: str | torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> "FailureDetectionVLA":
		run_dir, resolved_checkpoint = cls._resolve_run_dir_and_checkpoint(Path(checkpoint_path))
		run_config = cls._load_training_config(run_dir)
		model_kwargs = cls._build_init_kwargs_from_config(run_config)

		model = cls(**model_kwargs)

		checkpoint = torch.load(resolved_checkpoint, map_location=map_location)
		state_dict = cls._strip_module_prefix(checkpoint.get("model_state_dict", checkpoint))
		missing, unexpected = model.load_state_dict(state_dict, strict=False)

		print(f"run_dir: {run_dir}")
		print(f"checkpoint: {resolved_checkpoint}")
		print(f"missing keys: {len(missing)}")
		print(f"unexpected keys: {len(unexpected)}")
		if missing:
			print("  missing:", missing[:10])
		if unexpected:
			print("  unexpected:", unexpected[:10])

		if device is not None or dtype is not None:
			if device is None:
				device = next(model.parameters()).device
			if dtype is None:
				dtype = next(model.parameters()).dtype
			model = model.to(device=device, dtype=dtype)

		model.eval()
		return model

	@staticmethod
	def _resolve_run_dir_and_checkpoint(path: Path) -> tuple[Path, Path]:
		path = path.expanduser().resolve()
		if path.is_file():
			if path.parent.name == "checkpoints":
				return path.parent.parent, path
			return path.parent, path
		if path.is_dir():
			if path.name == "checkpoints":
				checkpoint_files = sorted(path.glob("step_*.pt"))
				if not checkpoint_files:
					raise FileNotFoundError(f"No checkpoints found in {path}")
				return path.parent, checkpoint_files[-1]
			checkpoint_dir = path / "checkpoints"
			if checkpoint_dir.exists():
				checkpoint_files = sorted(checkpoint_dir.glob("step_*.pt"))
				if not checkpoint_files:
					raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
				return path, checkpoint_files[-1]
		raise FileNotFoundError(f"Could not resolve run directory from: {path}")

	@staticmethod
	def _load_training_config(run_dir: Path) -> dict[str, Any]:
		config_path = run_dir / "training_config.json"
		if not config_path.exists():
			raise FileNotFoundError(f"training_config.json not found: {config_path}")
		with config_path.open("r", encoding="utf-8") as handle:
			return json.load(handle)

	@staticmethod
	def _build_init_kwargs_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
		def pick(name: str, default: Any) -> Any:
			return config.get(name, default)

		lora_target_modules = pick("lora_target_modules", None)
		if isinstance(lora_target_modules, str) and lora_target_modules.strip():
			lora_target_modules = tuple(part.strip() for part in lora_target_modules.split(",") if part.strip())
		elif isinstance(lora_target_modules, (list, tuple)):
			lora_target_modules = tuple(str(part) for part in lora_target_modules)
		else:
			lora_target_modules = None

		return {
			"gemma_name": pick("gemma_name", "google/gemma-4-E2B-it"),
			"ego_dim": int(pick("ego_dim", 5)),
			"goal_dim": int(pick("goal_dim", 3)),
			"other_dim": int(pick("other_dim", 15)),
			"map_dim": int(pick("map_dim", 25)),
			"tl_dim": int(pick("tl_dim", 11)),
			"num_scene_tokens": int(pick("num_scene_tokens", 32)),
			"use_ego_trajectory": bool(pick("use_ego_trajectory", True)),
			"trajectory_dim": int(pick("trajectory_dim", 5)),
			"hidden_dim": int(pick("scene_hidden_dim", 512)),
			"num_heads": int(pick("num_heads", 4)),
			"num_attn_layers": int(pick("num_attn_layers", 4)),
			"num_object_types": int(pick("num_object_types", 8)),
			"num_map_type_classes": int(pick("num_map_type_classes", 21)),
			"use_lora": bool(pick("use_lora", False)),
			"lora_r": int(pick("lora_r", 16)),
			"lora_alpha": int(pick("lora_alpha", 32)),
			"lora_dropout": float(pick("lora_dropout", 0.05)),
			"lora_target_modules": lora_target_modules,
		}

	@staticmethod
	def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
		if not any(key.startswith("module.") for key in state_dict):
			return state_dict
		return {
			key[len("module."):] if key.startswith("module.") else key: value
			for key, value in state_dict.items()
		}

	def _load_peft_components(self):
		try:
			peft = importlib.import_module("peft")
		except ModuleNotFoundError as exc:
			raise ImportError(
				"peft is required for LoRA training. Install it with `pip install peft`."
			) from exc
		return peft.LoraConfig, peft.TaskType, peft.get_peft_model

	def _attach_lora(
		self,
		*,
		r: int,
		alpha: int,
		dropout: float,
		target_modules: tuple[str, ...] | None,
	) -> None:
		LoraConfig, TaskType, get_peft_model = self._load_peft_components()

		if target_modules is None:
			target_modules = (
				"q_proj.linear",
				"k_proj.linear",
				"v_proj.linear",
				"o_proj.linear",
				"gate_proj.linear",
				"up_proj.linear",
				"down_proj.linear",
			)
		else:
			normalized_modules: list[str] = []
			for module_name in target_modules:
				normalized_modules.append(module_name)
				if "." not in module_name:
					normalized_modules.append(f"{module_name}.linear")
			target_modules = tuple(dict.fromkeys(normalized_modules))

		lora_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			r=r,
			lora_alpha=alpha,
			lora_dropout=dropout,
			bias="none",
			target_modules=list(target_modules),
		)
		self.llm = get_peft_model(self.llm, lora_config)
		self._has_lora = True

	def freeze_llm(self) -> None:
		if self._has_lora:
			for name, param in self.llm.named_parameters():
				if "lora_" not in name:
					param.requires_grad = False
		else:
			for param in self.llm.parameters():
				param.requires_grad = False

	def _replace_scene_embeddings_hook(self, module, inputs, output):
		if self._scene_tokens_for_hook is None:
			return output

		start = self._scene_start_for_hook
		length = self._scene_len_for_hook
		if output.ndim != 3 or output.shape[1] < start + length:
			return output

		scene_tokens = self._scene_tokens_for_hook.to(device=output.device, dtype=output.dtype)
		return torch.cat([scene_tokens, output[:, length:, :]], dim=1)

	def _generation_eos_token_ids(self) -> int | list[int]:
		eos_ids: list[int] = []
		if self.tokenizer.eos_token_id is not None:
			eos_ids.append(int(self.tokenizer.eos_token_id))
		for token_id in self.tokenizer.encode("\n", add_special_tokens=False):
			token_id = int(token_id)
			if token_id not in eos_ids:
				eos_ids.append(token_id)
		if not eos_ids:
			raise ValueError("Tokenizer has no eos_token_id for generation.")
		return eos_ids[0] if len(eos_ids) == 1 else eos_ids

	def _encode_scene(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
		scene_tokens = self.scene_tokenizer(input_features, deterministic=True)
		scene_tokens = self.scene_token_self_attention(scene_tokens)
		return scene_tokens

	def _other_vehicle_mask(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
		other_states = self._to_torch(input_features["other_states"], device=self._module_device())
		other_valid = self._to_torch(input_features["other_valid"], device=other_states.device)
		if other_states.ndim != 3:
			raise ValueError(f"other_states must have shape [B, N, D], got {tuple(other_states.shape)}")
		type_start = other_states.shape[-1] - self.num_object_types
		if type_start < 0:
			return other_valid.to(dtype=torch.bool)
		object_types = other_states[..., type_start:]
		vehicle_mask = object_types.argmax(dim=-1) == int(self.vehicle_type_id)
		return other_valid.to(dtype=torch.bool) & vehicle_mask

	def _road_edge_mask(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
		map_features = self._to_torch(input_features["map_features"], device=self._module_device())
		map_valid = self._to_torch(input_features["map_valid"], device=map_features.device)
		if map_features.ndim != 4:
			raise ValueError(f"map_features must have shape [B, M, P, D], got {tuple(map_features.shape)}")
		type_start = map_features.shape[-1] - self.num_map_type_classes
		if type_start < 0:
			return map_valid.any(dim=-1)
		map_types = map_features[..., type_start:]
		road_edge_point_mask = map_valid & (
			(map_types[..., int(self.road_edge_type_ids[0])] > 0.5)
			| (map_types[..., int(self.road_edge_type_ids[1])] > 0.5)
		)
		return road_edge_point_mask.any(dim=-1)

	def _predict_object_logits(
		self,
		scene_tokens: torch.Tensor,
		input_features: Mapping[str, torch.Tensor],
	) -> torch.Tensor:
		other_states = self._to_torch(input_features["other_states"], device=scene_tokens.device)
		other_valid = self._to_torch(input_features["other_valid"], device=scene_tokens.device).to(dtype=torch.bool)
		object_tokens = self.object_tokenizer(other_states.reshape(-1, other_states.shape[-1]), deterministic=True)
		object_tokens = object_tokens.reshape(other_states.shape[0], other_states.shape[1], -1)
		object_context = self.object_cross_attention(
			query_btc=object_tokens,
			context_bnc=scene_tokens,
			context_mask_bn=torch.ones(scene_tokens.shape[:2], dtype=torch.bool, device=scene_tokens.device),
		)
		object_logits = self.object_id_head(object_context).squeeze(-1)
		object_mask = self._other_vehicle_mask(input_features)
		return object_logits.masked_fill(~object_mask, torch.finfo(object_logits.dtype).min)

	def _predict_road_segment_logits(
		self,
		scene_tokens: torch.Tensor,
		input_features: Mapping[str, torch.Tensor],
	) -> torch.Tensor:
		map_features = self._to_torch(input_features["map_features"], device=scene_tokens.device)
		map_valid = self._to_torch(input_features["map_valid"], device=scene_tokens.device).to(dtype=torch.bool)
		map_tokens = self.map_tokenizer(
			map_features.reshape(-1, map_features.shape[-2], map_features.shape[-1]),
			map_valid.reshape(-1, map_valid.shape[-1]),
			deterministic=True,
		).reshape(map_features.shape[0], map_features.shape[1], -1)
		map_context = self.map_cross_attention(
			query_btc=map_tokens,
			context_bnc=scene_tokens,
			context_mask_bn=torch.ones(scene_tokens.shape[:2], dtype=torch.bool, device=scene_tokens.device),
		)
		road_logits = self.road_segment_id_head(map_context).squeeze(-1)
		road_mask = self._road_edge_mask(input_features)
		return road_logits.masked_fill(~road_mask, torch.finfo(road_logits.dtype).min)

	@staticmethod
	def _resolve_label_positions(candidate_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return per-example candidate positions and a mask of valid matches."""
		if target_ids.ndim != 1:
			target_ids = target_ids.reshape(-1)
		target_match = candidate_ids == target_ids[:, None]
		has_match = target_match.any(dim=-1)
		positions = target_match.long().argmax(dim=-1)
		return positions, has_match

	def _compute_index_loss_and_accuracy(
		self,
		logits: torch.Tensor,
		candidate_mask: torch.Tensor,
		target_ids: torch.Tensor | None,
		candidate_ids: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		device = logits.device
		zero = torch.zeros((), dtype=logits.dtype, device=device)
		if target_ids is None:
			return zero, zero

		target_ids = self._to_torch(target_ids, device=device).to(dtype=torch.long)
		if target_ids.ndim != 1:
			target_ids = target_ids.reshape(-1)
		if candidate_ids is None:
			candidate_ids = torch.arange(logits.shape[1], device=device, dtype=torch.long).expand(logits.shape[0], -1)
		else:
			candidate_ids = self._to_torch(candidate_ids, device=device).to(dtype=torch.long)
			if candidate_ids.ndim != 2:
				raise ValueError(f"candidate_ids must have shape [B, N], got {tuple(candidate_ids.shape)}")

		positions, has_match = self._resolve_label_positions(candidate_ids, target_ids)
		valid = has_match & candidate_mask.gather(dim=1, index=positions[:, None]).squeeze(1).to(dtype=torch.bool)
		if not bool(valid.any()):
			return zero, zero

		masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
		loss = F.cross_entropy(masked_logits[valid], positions[valid])
		pred_positions = masked_logits.argmax(dim=-1)
		accuracy = (pred_positions[valid] == positions[valid]).float().mean()
		return loss, accuracy

	def forward(
		self,
		input_features: Mapping[str, torch.Tensor],
		prompt_ids: torch.Tensor,
		prompt_mask: torch.Tensor,
		answer_ids: torch.Tensor | None = None,
		answer_mask: torch.Tensor | None = None,
		critical_object_ids: torch.Tensor | None = None,
		critical_road_segment_ids: torch.Tensor | None = None,
	):
		device = self._module_device()
		input_features = self._to_torch_feature_map(input_features, device=device)
		prompt_ids = self._to_torch(prompt_ids, device=device)
		prompt_mask = self._to_torch(prompt_mask, device=device)
		if answer_ids is not None:
			answer_ids = self._to_torch(answer_ids, device=device)
		if answer_mask is not None:
			answer_mask = self._to_torch(answer_mask, device=device)

		scene_tokens = self._encode_scene(input_features)
		batch_size = int(prompt_ids.shape[0])
		num_scene_tokens = int(scene_tokens.shape[1])

		scene_dummy_ids = torch.full(
			(batch_size, num_scene_tokens),
			fill_value=self.tokenizer.pad_token_id,
			dtype=prompt_ids.dtype,
			device=device,
		)
		scene_mask = torch.ones(
			(batch_size, num_scene_tokens),
			dtype=prompt_mask.dtype,
			device=device,
		)
		if answer_ids is None:
			input_ids = torch.cat([scene_dummy_ids, prompt_ids], dim=1)
			attention_mask = torch.cat([scene_mask, prompt_mask], dim=1)
			labels = None
		else:
			input_ids = torch.cat([scene_dummy_ids, prompt_ids, answer_ids], dim=1)
			if answer_mask is None:
				answer_mask = (answer_ids != self.tokenizer.pad_token_id).to(dtype=prompt_mask.dtype)
			attention_mask = torch.cat([scene_mask, prompt_mask, answer_mask], dim=1)
			answer_labels = answer_ids.masked_fill(answer_mask == 0, -100)
			labels = torch.cat(
				[
					torch.full(scene_dummy_ids.shape, -100, dtype=answer_ids.dtype, device=device),
					torch.full(prompt_ids.shape, -100, dtype=answer_ids.dtype, device=device),
					answer_labels,
				],
				dim=1,
			)

		self._scene_tokens_for_hook = scene_tokens
		self._scene_start_for_hook = 0
		self._scene_len_for_hook = num_scene_tokens

		try:
			outputs = self.llm(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels,
				output_hidden_states=False,
				use_cache=False,
			)
		finally:
			self._scene_tokens_for_hook = None
			self._scene_start_for_hook = None
			self._scene_len_for_hook = None

		object_logits = self._predict_object_logits(scene_tokens, input_features)
		road_segment_logits = self._predict_road_segment_logits(scene_tokens, input_features)

		object_candidate_mask = self._other_vehicle_mask(input_features)
		road_segment_candidate_mask = self._road_edge_mask(input_features)
		object_ids = input_features.get("other_ids")
		if object_ids is None:
			object_ids = input_features.get("object_ids")
		road_segment_ids = input_features.get("map_ids")
		object_loss, object_accuracy = self._compute_index_loss_and_accuracy(
			object_logits,
			object_candidate_mask,
			critical_object_ids,
			candidate_ids=self._to_torch(object_ids, device=device) if object_ids is not None else None,
		)
		road_loss, road_accuracy = self._compute_index_loss_and_accuracy(
			road_segment_logits,
			road_segment_candidate_mask,
			critical_road_segment_ids,
			candidate_ids=self._to_torch(road_segment_ids, device=device) if road_segment_ids is not None else None,
		)

		loss = outputs.loss + object_loss + road_loss
		return {
			"loss": loss,
			"language_loss": outputs.loss.detach().cpu(),
			"object_loss": object_loss.detach().cpu(),
			"road_segment_loss": road_loss.detach().cpu(),
			"object_accuracy": object_accuracy.detach().cpu(),
			"road_segment_accuracy": road_accuracy.detach().cpu(),
		}

	def generate_predictions(
		self,
		input_features: Mapping[str, torch.Tensor],
		prompt_ids: torch.Tensor,
		prompt_mask: torch.Tensor,
		max_new_tokens: int = 8,
	) -> dict[str, Any]:
		device = self._module_device()
		input_features = self._to_torch_feature_map(input_features, device=device)
		prompt_ids = self._to_torch(prompt_ids, device=device)
		prompt_mask = self._to_torch(prompt_mask, device=device)
		scene_tokens = self._encode_scene(input_features)

		batch_size = int(prompt_ids.shape[0])
		num_scene_tokens = int(scene_tokens.shape[1])
		scene_dummy_ids = torch.full(
			(batch_size, num_scene_tokens),
			fill_value=self.tokenizer.pad_token_id,
			dtype=prompt_ids.dtype,
			device=device,
		)
		scene_mask = torch.ones(
			(batch_size, num_scene_tokens),
			dtype=prompt_mask.dtype,
			device=device,
		)
		input_ids = torch.cat([scene_dummy_ids, prompt_ids], dim=1)
		attention_mask = torch.cat([scene_mask, prompt_mask], dim=1)

		self._scene_tokens_for_hook = scene_tokens
		self._scene_start_for_hook = 0
		self._scene_len_for_hook = num_scene_tokens

		try:
			generated_ids = self.llm.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				pad_token_id=self.tokenizer.pad_token_id,
				eos_token_id=self._generation_eos_token_ids(),
			)
		finally:
			self._scene_tokens_for_hook = None
			self._scene_start_for_hook = None
			self._scene_len_for_hook = None

		input_len = int(input_ids.shape[1])
		new_token_ids = generated_ids[:, input_len:]
		decoded = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
		answer_text = [text.strip() for text in decoded]

		object_logits = self._predict_object_logits(scene_tokens, input_features)
		road_segment_logits = self._predict_road_segment_logits(scene_tokens, input_features)
		object_candidate_mask = self._other_vehicle_mask(input_features)
		road_segment_candidate_mask = self._road_edge_mask(input_features)

		object_valid_pred = object_candidate_mask.any(dim=-1)
		road_valid_pred = road_segment_candidate_mask.any(dim=-1)
		object_pred_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
		road_pred_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
		if bool(object_valid_pred.any()):
			object_pred_idx = object_pred_idx.clone()
			object_pred_idx[object_valid_pred] = object_logits.masked_fill(~object_candidate_mask, torch.finfo(object_logits.dtype).min)[object_valid_pred].argmax(dim=-1)
		if bool(road_valid_pred.any()):
			road_pred_idx = road_pred_idx.clone()
			road_pred_idx[road_valid_pred] = road_segment_logits.masked_fill(~road_segment_candidate_mask, torch.finfo(road_segment_logits.dtype).min)[road_valid_pred].argmax(dim=-1)

		object_ids = input_features.get("other_ids")
		if object_ids is None:
			object_ids = input_features.get("object_ids")
		road_segment_ids = input_features.get("map_ids")
		if object_ids is not None:
			object_ids_t = self._to_torch(object_ids, device=device).to(dtype=torch.long)
			if object_ids_t.ndim == 2:
				object_pred_ids = torch.full_like(object_pred_idx, -1)
				valid_objects = object_pred_idx >= 0
				if bool(valid_objects.any()):
					object_pred_ids[valid_objects] = torch.gather(
						object_ids_t[valid_objects],
						dim=1,
						index=object_pred_idx[valid_objects].unsqueeze(-1),
					).squeeze(-1)
			else:
				object_pred_ids = object_pred_idx
		else:
			object_pred_ids = object_pred_idx

		if road_segment_ids is not None:
			road_ids_t = self._to_torch(road_segment_ids, device=device).to(dtype=torch.long)
			if road_ids_t.ndim == 2:
				road_pred_ids = torch.full_like(road_pred_idx, -1)
				valid_roads = road_pred_idx >= 0
				if bool(valid_roads.any()):
					road_pred_ids[valid_roads] = torch.gather(
						road_ids_t[valid_roads],
						dim=1,
						index=road_pred_idx[valid_roads].unsqueeze(-1),
					).squeeze(-1)
			else:
				road_pred_ids = road_pred_idx
		else:
			road_pred_ids = road_pred_idx

		return {
			"answer": answer_text,
			"failure_reason": answer_text,
			"critical_object_ids": object_pred_ids.detach().cpu().numpy(),
			"critical_road_segment_ids": road_pred_ids.detach().cpu().numpy(),
		}

	def state_dict(
		self,
		destination=None,
		prefix: str = "",
		keep_vars: bool = False,
		only_trainable: bool = True,
	):
		state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
		if not only_trainable:
			return state
		keep_prefixes = [
			"scene_tokenizer.",
			"scene_token_self_attention.",
			"object_tokenizer.",
			"object_cross_attention.",
			"object_id_head.",
			"map_tokenizer.",
			"map_cross_attention.",
			"road_segment_id_head.",
		]
		return {
			key: value.cpu()
			for key, value in state.items()
			if any(key.startswith(prefix_name) for prefix_name in keep_prefixes) or "lora_" in key
		}


__all__ = ["FailureDetectionVLA"]