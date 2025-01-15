# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import copy

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from peft.tuners.lora.dora import DoraConv2dLayer, DoraLinearLayer
from peft.tuners.lora.config import LoraConfig


class PLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    # adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    adapter_layer_names = ("lora_A", "lora_B", "lora_P", "lora_embedding_A", "lora_embedding_B", "lora_embedding_P")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        # TODO：  dict 支持 多个 adapter
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_P = nn.ModuleDict({})
        # TODO  lora_Router for MoE  (x / p base)
        self.lora_Router = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.lora_embedding_P = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        # TODO

        # TODO 确定base_layer 维度
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )
        # TODO 根据 base_layer 确定输入输出维度
        self.in_features = in_features
        self.out_features = out_features
        self.user_features = kwargs['user_feature_dim']

        self.inject_method = kwargs['inject_method']

    def update_layer(
            self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False,
            **kwargs,
    ):
        # TODO: adapter插入 Layer 实现
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        # TODO init lora  self => one layer
        if kwargs['expert_num'] > 1:
            en = kwargs['expert_num']
            self.lora_A[adapter_name] = nn.ModuleList(
                [nn.Linear(self.in_features, r, bias=False) for _ in range(en)])
            self.lora_B[adapter_name] = nn.ModuleList(
                [nn.Linear(r, self.out_features, bias=False) for _ in range(en)])

            # TODO init lora_Router  Router(personal_info)
            # self.lora_Router[adapter_name] = nn.Linear(self.user_features, en, bias=False)
            #  base moe  Router(x)
            self.lora_Router[adapter_name] = nn.Linear(self.in_features, en, bias=False)

            # TODO cus init
            if adapter_name in self.lora_A.keys():
                if init_lora_weights is True:
                    for exp in self.lora_A[adapter_name]:
                        nn.init.kaiming_uniform_(exp.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.lora_Router[adapter_name].weight, a=math.sqrt(5))

                elif init_lora_weights.lower() == "gaussian":
                    for exp in self.lora_A[adapter_name]:
                        nn.init.normal_(exp.weight, std=1 / self.r[adapter_name])
                    nn.init.normal_(self.lora_Router[adapter_name].weight, std=1 / self.r[adapter_name])

                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights=}")

                for exp in self.lora_B[adapter_name]:
                    nn.init.zeros_(exp.weight)
        else:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_P[adapter_name] = nn.Linear(self.user_features, r, bias=False)  # TODO for add
            # TODO for add  / hadamard
            if self.inject_method in ['add', 'hadamard', 'ponly']:
                self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            # TODO for cat
            elif self.inject_method == 'cat':
                self.lora_B[adapter_name] = nn.Linear(2 * r, self.out_features, bias=False)
            else:
                print(f'No implementation for {self.inject_method}')

            # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
            if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
                with gather_params_ctx(self.get_base_layer().weight):
                    self.pissa_init(adapter_name, init_lora_weights)
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
                with gather_params_ctx(self.get_base_layer().weight):
                    self.olora_init(adapter_name)
            elif init_lora_weights == "loftq":
                with gather_params_ctx(self.get_base_layer().weight):
                    self.loftq_init(adapter_name)
            elif init_lora_weights:
                # TODO 初始化参数
                self.reset_lora_parameters(adapter_name, init_lora_weights)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        # TODO set the specified adapters to trainable
        self.set_adapter(self.active_adapters)

    # TODO 初始化 模型参数
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_P[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
                nn.init.normal_(self.lora_P[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")

            # LoRA_B 零初始化,保证在初始阶段时，只有左边的主干生效
            nn.init.zeros_(self.lora_B[adapter_name].weight)

        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.zeros_(self.lora_embedding_P[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def olora_init(self, adapter_name):
        dtype = self.get_base_layer().weight.dtype
        if dtype in [torch.int8, torch.uint8]:
            weight_tensor = dequantize_module_weight(self.get_base_layer())
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = self.get_base_layer().weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        weight_tensor = weight_tensor.to(dtype)
        self.get_base_layer().weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type == "cuda":
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type != "cuda":
                    lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class PLinear(nn.Module, PLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,  # TODO  add at model _create_and_replace
    ) -> None:
        super().__init__()
        PLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            **kwargs,  # TODO
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        # TODO for moe
        self.expert_num = kwargs['expert_num']
        self.inject_method = kwargs['inject_method']

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_P = self.lora_P[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # TODO  考虑 B（A?P）权重混合
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_P[adapter].weight.data = weight_P.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        temp_kwargs = {key: value for key, value in kwargs.items() if key not in ['p', 'act_list']}

        if self.disable_adapters:  # 禁用
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **temp_kwargs)
        elif adapter_names is not None:  # 混合适配器(mixing different adapters in the same batch)
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:  # 已经融合过参数
            result = self.base_layer(x, *args, **temp_kwargs)
        else:  # base layer forward
            result = self.base_layer(x, *args, **temp_kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:  # 遍历adapter
                if active_adapter not in self.lora_A.keys():
                    continue

                if self.expert_num == 1:
                    # TODO  get cur layer adapter
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    lora_P = self.lora_P[active_adapter]

                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_A.weight.dtype)

                    if not self.use_dora[active_adapter]:
                        # result = result + lora_B(lora_A(dropout(x))) * scaling
                        # TODO  parallel add to every token
                        # X (bs, token_num ,2048)  p (8, 768) => (8, token_num, 8)
                        p = kwargs.get('p')
                        ep = p.unsqueeze(1).repeat(1, x.shape[1], 1)

                        # TODO B(A(x)+P(x))
                        aa, ap = None, None
                        if self.inject_method == 'add':
                            aa = lora_A(dropout(x))
                            ap = lora_P(dropout(ep))
                            result = result + lora_B(aa + ap) * scaling
                        elif self.inject_method == 'hadamard':
                            aa = lora_A(dropout(x))
                            ap = lora_P(dropout(ep))
                            result = result + lora_B(aa * ap) * scaling
                        # TODO  concat B(A(x)||P(x))
                        elif self.inject_method == 'cat':
                            aa = lora_A(dropout(x))  # (bs, seq, dim)
                            ap = lora_P(dropout(ep))
                            result = result + lora_B(torch.cat([aa, ap], dim=-1)) * scaling
                        elif self.inject_method == 'ponly':
                            ap = lora_P(dropout(ep))
                            result = result + lora_B(ap) * scaling
                        else:
                            print(f'no implementation for inject method {self.inject_method}')

                        if kwargs.get('act_list') is not None:
                            kwargs['act_list'].append((aa, ap))

                    else:
                        x = dropout(x)
                        result = result + self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                        )

                        result = result.to(torch_result_dtype)
                else:  # MoE 自动路由  / GPMOE 基于P路由  / PMoE P作为输入
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(lora_A[0].weight.dtype)

                    # X (bs, token_num ,2048)  p (bs, exp_num) => repeat => (8, token_num, exp_num)
                    # 使用 固定的 群组 路由
                    # p = kwargs.get('p')
                    # gate_weights = p.unsqueeze(1).repeat(1, x.shape[1], 1)  # [batch_size, num_experts]

                    # TODO 使用 可学习分类器路由  Router(p) for per sample
                    # p = kwargs.get('p')
                    # p = self.lora_Router[active_adapter](p)
                    # gate_weights = p.unsqueeze(1).repeat(1, x.shape[1], 1)  # [batch_size, num_experts]
                    # TODO Router(x) for token
                    gate_weights = self.lora_Router[active_adapter](x)

                    expert_outputs = torch.stack([b(a(dropout(x))) for a, b in zip(lora_A, lora_B)],
                                                 dim=1)  # [batch_size, num_experts, r]
                    # 对专家输出进行加权求和
                    # x (bs, exp_num, seq_len, hid_dim) * (bs, seq_len, exp_num).permute(0, 2, 1).unsqueeze(-1)
                    weighted_output = torch.sum(expert_outputs * gate_weights.permute(0, 2, 1).unsqueeze(-1),
                                                dim=1)  # [batch_size, r]

                    result = result + weighted_output * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


# 针对 Embedding 层的 adapter
class Embedding(nn.Module, PLoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()
        PLoraLayer.__init__(self, base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


# 针对 卷积层 的 adapter
class Conv2d(nn.Module, PLoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()
        PLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(-1, 1, 1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                    F.conv2d(
                        weight_A.permute(1, 0, 2, 3),
                        weight_B,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
        target: torch.nn.Module,
        adapter_name: str,
        lora_config: LoraConfig,
        **kwargs,
) -> Optional[torch.nn.Module]:
    """
    替换target为 new模型 (dispatch 分发)
    """
    new_module = None

    # TODO 获取 target_base_layer
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        # TODO
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        # TODO
        new_module = Conv2d(target, adapter_name, **kwargs)

    # TODO 目前只能将Linear转化为PLinear , 后续考虑适配量化模块 Bit4Linear
    # TODO bit4Linear 的在 bnb  dispatch_bnb_4bit  实现
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        # TODO
        new_module = PLinear(target, adapter_name, **kwargs)

    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        # TODO
        new_module = PLinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module


def plora_test():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 超参数
    input_dim = 10  # 输入 x 的维度
    embedding_dim = 5  # 嵌入向量的维度
    hidden_dim = 20  # 隐藏层的维度
    output_dim = 3  # 输出类别数量
    num_embeddings = 5  # p 的取值范围 [1, 5] 对应的嵌入数目
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.001

    # 生成随机验证数据
    x_data = torch.randn(100, input_dim)  # 100个样本
    p_data = torch.randint(1, 6, (100,))  # 100个样本，范围在 [1, 5]
    y_data = torch.randint(0, output_dim, (100,))  # 100个样本，三分类标签

    # 创建数据加载器
    dataset = TensorDataset(x_data, p_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class MyModel(nn.Module):
        def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_embeddings):
            super(MyModel, self).__init__()
            self.A = nn.Linear(input_dim, hidden_dim)
            self.P = nn.Embedding(num_embeddings, embedding_dim)  # Embedding layer for p
            self.E = nn.Linear(embedding_dim, hidden_dim)  # Linear transformation after embedding
            self.B = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, p):
            Ax = self.A(x)
            Pp = self.P(p)
            Ep = self.E(Pp)
            y = self.B(Ax + Ep)
            return y

    # 创建模型实例
    model = MyModel(input_dim, embedding_dim, hidden_dim, output_dim, num_embeddings)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x, p, y in dataloader:
            optimizer.zero_grad()  # 清零梯度

            # 前向传播
            outputs = model(x, p)

            # 计算损失
            loss = criterion(outputs, y)
            running_loss += loss.item()

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

        # 打印每个epoch的平均损失
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 监控梯度
        if (epoch + 1) % 10 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Layer: {name}, Gradient norm: {param.grad.norm().item():.4f}')

    print("Training complete.")


if __name__ == "__main__":
    plora_test()
