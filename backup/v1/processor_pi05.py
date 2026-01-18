#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Key for storing normalized state before tokenization (for multi-step inference)
NORMALIZED_STATE_KEY = "_normalized_state_for_queue"

@ProcessorStepRegistry.register(name="pi05_preserve_normalized_state_processor_step")
@dataclass
class Pi05PreserveNormalizedStateProcessorStep(ProcessorStep):
    """
    Preserve normalized state before tokenization for multi-step inference.
    
    This step copies the normalized state to a special key so that it can be
    stored in queues during inference. After stacking multi-frame observations,
    the stacked state will be re-tokenized.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is not None:
            # Store a copy of normalized state before tokenization
            # This will be used in select_action for multi-step observation stacking
            transition[TransitionKey.OBSERVATION][NORMALIZED_STATE_KEY] = state.clone()
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter the feature definitions."""
        return features

@ProcessorStepRegistry.register(name="pi05_queue_filling_augmentation")
@dataclass
class Pi05QueueFillingAugmentationStep(ProcessorStep):
    """
    Simulates inference-time queue filling during training.
    
    During inference, when an episode starts, the observation queue is filled by repeating
    the first frame. This augmentation randomly applies the same behavior during training
    to make the model robust to this pattern.
    
    When enabled, all historical observation frames (state and images) are replaced with
    the current frame, simulating the "queue not yet filled" scenario.
    
    Note: This augmentation should be disabled during inference by setting aug_prob=0.
    The lerobot_eval.py script does this automatically via preprocessor_overrides.
    """
    
    aug_prob: float = 0.0  # Probability of applying augmentation (0.0 = disabled)
    n_obs_steps: int = 1  # Number of observation steps
    image_feature_keys: list[str] = field(default_factory=list)  # Keys for image features
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Skip if augmentation is disabled or single-step observations
        if self.aug_prob <= 0.0 or self.n_obs_steps <= 1:
            return transition
        
        # Apply augmentation with probability aug_prob
        if random.random() >= self.aug_prob:
            return transition
        
        transition = transition.copy()
        observations = transition.get(TransitionKey.OBSERVATION, {})
        
        # Process state: replace historical frames with current frame
        state = observations.get(OBS_STATE)
        if state is not None and state.dim() == 3:  # [B, n_obs_steps, state_dim]
            # Copy the last frame (current frame) to all positions
            current_frame = state[:, -1:, :]  # [B, 1, state_dim]
            augmented_state = current_frame.expand_as(state).clone()
            observations[OBS_STATE] = augmented_state
        
        # Process image features: replace historical frames with current frame
        for img_key in self.image_feature_keys:
            img = observations.get(img_key)
            if img is not None and img.dim() == 5:  # [B, n_obs_steps, C, H, W]
                # Copy the last frame (current frame) to all positions
                current_frame = img[:, -1:, :, :, :]  # [B, 1, C, H, W]
                augmented_img = current_frame.expand_as(img).clone()
                observations[img_key] = augmented_img
        
        transition[TransitionKey.OBSERVATION] = observations
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter the feature definitions."""
        return features



@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Handle multi-step observations: [B, n_obs_steps, state_dim]
        if state.dim() == 3:
            batch_size, n_obs_steps, state_dim = state.shape
            # Process each observation step
            all_discretized_states = []
            for step_idx in range(n_obs_steps):
                step_state = state[:, step_idx]  # [B, state_dim]
                step_state = pad_vector(step_state, self.max_state_dim)
                step_state_np = step_state.cpu().numpy()
                discretized = np.digitize(step_state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                all_discretized_states.append(discretized)
            
            full_prompts = []
            for i, task in enumerate(tasks):
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                # Concatenate all observation steps' states with separator
                state_strs = []
                for step_idx in range(n_obs_steps):
                    state_str = " ".join(map(str, all_discretized_states[step_idx][i]))
                    state_strs.append(state_str)
                # Join all steps with semicolon separator
                all_states_str = "; ".join(state_strs)
                full_prompt = f"Task: {cleaned_text}, State: {all_states_str};\nAction: "
                full_prompts.append(full_prompt)
        else:
            # Single step observation: [B, state_dim]
            # Prepare state (pad to max_state_dim)
            state = pad_vector(state, self.max_state_dim)

            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            state_np = state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

            full_prompts = []
            for i, task in enumerate(tasks):
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
                full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Extract image feature keys from config for augmentation
    image_feature_keys = [k for k, v in config.input_features.items() if "image" in k.lower()]
    
    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # # Queue filling augmentation: simulates inference-time queue filling during training
        # # Must come before normalization to work on raw observations
        # Pi05QueueFillingAugmentationStep(
        #     aug_prob=config.queue_filling_aug_prob,
        #     n_obs_steps=config.n_obs_steps,
        #     image_feature_keys=image_feature_keys,
        # ),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        # Pi05PreserveNormalizedStateProcessorStep(),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
