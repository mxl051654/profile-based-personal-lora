# flake8: noqa

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# There is a circular import in the PPOTrainer if we let isort sort these
from typing import TYPE_CHECKING
from trl.import_utils import _LazyModule, is_diffusers_available, OptionalDependencyNotAvailable

_import_structure = {
    "callbacks": ["RichProgressCallback", "SyncRefModelCallback"],
    "utils": [
        "AdaptiveKLController",
        "FixedKLController",
        "ConstantLengthDataset",
        "DataCollatorForCompletionOnlyLM",
        "RunningMoments",
        "disable_dropout_in_model",
        "peft_module_casting_to_bf16",
    ],
    "dpo_config": ["DPOConfig", "FDivergenceConstants", "FDivergenceType"],
    "dpo_trainer": ["DPOTrainer"],
    "cpo_config": ["CPOConfig"],
    "cpo_trainer": ["CPOTrainer"],
    "alignprop_config": ["AlignPropConfig"],
    "alignprop_trainer": ["AlignPropTrainer"],
    "iterative_sft_trainer": ["IterativeSFTTrainer"],
    "kto_config": ["KTOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "bco_config": ["BCOConfig"],
    "bco_trainer": ["BCOTrainer"],
    "model_config": ["ModelConfig"],
    "online_dpo_config": ["OnlineDPOConfig"],
    "online_dpo_trainer": ["OnlineDPOTrainer"],
    "orpo_config": ["ORPOConfig"],
    "orpo_trainer": ["ORPOTrainer"],
    "ppo_config": ["PPOConfig"],
    "ppo_trainer": ["PPOTrainer"],
    "reward_config": ["RewardConfig"],
    "reward_trainer": ["RewardTrainer", "compute_accuracy"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "base": ["BaseTrainer"],
    "ddpo_config": ["DDPOConfig"],
    "callbacks": ["RichProgressCallback", "SyncRefModelCallback", "WinRateCallback"],
    "judges": [
        "BaseJudge",
        "BaseRankJudge",
        "BasePairwiseJudge",
        "RandomRankJudge",
        "RandomPairwiseJudge",
        "PairRMJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
    ],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ddpo_trainer"] = ["DDPOTrainer"]

if TYPE_CHECKING:
    # isort: off
    from trl.trainer.callbacks import RichProgressCallback, SyncRefModelCallback
    from trl.trainer.utils import (
        AdaptiveKLController,
        FixedKLController,
        ConstantLengthDataset,
        DataCollatorForCompletionOnlyLM,
        RunningMoments,
        disable_dropout_in_model,
        peft_module_casting_to_bf16,
        empty_cache,
    )

    # isort: on

    from trl.trainer.base import BaseTrainer
    from trl.trainer.ddpo_config import DDPOConfig

    from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
    from trl.trainer.dpo_trainer import DPOTrainer
    from trl.trainer.iterative_sft_trainer import IterativeSFTTrainer
    from trl.trainer.cpo_config import CPOConfig
    from trl.trainer.cpo_trainer import CPOTrainer
    from trl.trainer.alignprop_config import AlignPropConfig
    from trl.trainer.alignprop_trainer import AlignPropTrainer
    from trl.trainer.kto_config import KTOConfig
    from trl.trainer.kto_trainer import KTOTrainer
    from trl.trainer.bco_config import BCOConfig
    from trl.trainer.bco_trainer import BCOTrainer
    from trl.trainer.model_config import ModelConfig
    from trl.trainer.online_dpo_config import OnlineDPOConfig
    from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
    from trl.trainer.orpo_config import ORPOConfig
    from trl.trainer.orpo_trainer import ORPOTrainer
    from trl.trainer.ppo_config import PPOConfig
    from trl.trainer.ppo_trainer import PPOTrainer
    from trl.trainer.reward_config import RewardConfig
    from trl.trainer.reward_trainer import RewardTrainer, compute_accuracy
    from trl.trainer.callbacks import RichProgressCallback, SyncRefModelCallback, WinRateCallback
    from trl.trainer.judges import (
        BaseJudge,
        BaseRankJudge,
        BasePairwiseJudge,
        RandomRankJudge,
        RandomPairwiseJudge,
        PairRMJudge,
        HfPairwiseJudge,
        OpenAIPairwiseJudge,
    )

    # TODO
    # from trl.trainer.sft_config import SFTConfig
    # from trl.trainer.sft_trainer import SFTTrainer
    from LaMP.PLoRA.trainer.sft_config import SFTConfig
    from LaMP.PLoRA.trainer.sft_trainer import SFTTrainer

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from trl.trainer.ddpo_trainer import DDPOTrainer
else:
    import sys
    # _LazyModule  在模块被首次访问时才进行导入，而不是在模块被首次加载时立即导入所有内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
