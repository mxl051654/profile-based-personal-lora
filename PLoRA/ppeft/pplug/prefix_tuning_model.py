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

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPrefixEncoder(torch.nn.Module):
    r"""
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        user_feature_dim = config.user_feature_dim
        num_layers = config.num_layers
        encoder_hidden_size = 128  # config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens

        if config.personal_type == 'uid':
            # 使用投影层且在训练模式
            if self.prefix_projection and not config.inference_mode:
                # prefix_projection for Prefix_tuning
                # 在 P-tuning v2 中，作者发现重参数化的改进很小，尤其是对于较小的模型，同时还会影响模型的表现
                # Use a two-layer MLP to encode the prefix
                self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
                self.transform = torch.nn.Sequential(
                    torch.nn.Linear(token_dim, encoder_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
                )
            else:
                # 保存模型时，会自动合并encoder输出到Embedding
                # get_prompt_embedding_to_save(self, adapter_name)
                # TODO  => 最终方案， 合并 virtual_tokens  (输入前 ， user_id shift 到 [nvt-len(user), nvt])
                self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

        elif config.personal_type == 'profile':
            if self.prefix_projection and not config.inference_mode:
                self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
                self.transform = torch.nn.Sequential(
                    torch.nn.Linear(token_dim, encoder_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
                )
            else:
                self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

            self.head_num = config.user_size
            self.W_in = nn.Linear(user_feature_dim, encoder_hidden_size * self.head_num)
            self.W_out = nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim)

        elif config.personal_type == 'history':
            # for task
            if self.prefix_projection and not config.inference_mode:
                self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
                self.transform = torch.nn.Sequential(
                    torch.nn.Linear(token_dim, encoder_hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
                )
            else:
                self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

            # for personal   split history & input  (bs , m+1, 768)
            self.head_num = config.user_size  # for virtual_token_num
            # self.head_num = config.num_virtual_tokens
            self.W_Q = nn.Linear(user_feature_dim, encoder_hidden_size * self.head_num)

            self.W_K = nn.Linear(user_feature_dim, encoder_hidden_size)
            self.W_V = nn.Linear(user_feature_dim, encoder_hidden_size)
            self.W_O = nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim)
        else:
            print(f'No implementation for personal_type in {config.personal_type}')

    def forward(self, prefix: torch.Tensor, p: torch.Tensor):
        """ prompt tokens & personal h+p embedding """
        config = self.config
        if config.personal_type == 'uid':
            if self.prefix_projection:
                prefix_tokens = self.embedding(prefix)
                past_key_values = self.transform(prefix_tokens)
            else:
                past_key_values = self.embedding(prefix)
            return past_key_values
        elif config.personal_type == 'profile':

            if self.prefix_projection:
                prefix_tokens = self.embedding(prefix)
                past_key_values = self.transform(prefix_tokens)
            else:
                past_key_values = self.embedding(prefix)

            # for personal   split history & input  (bs, 768)
            bs = p.shape[0]
            hid_state = self.W_in(p).reshape(bs, self.head_num, -1)
            personal_past_kv = self.W_out(hid_state)  # (bs, 1, 2 * token_dim)

            # 8, 20+head_num(user_size) , 18432
            past_key_values = torch.cat([past_key_values, personal_past_kv], dim=1)
            return past_key_values

        elif config.personal_type == 'history':
            if self.prefix_projection:
                prefix_tokens = self.embedding(prefix)
                past_key_values = self.transform(prefix_tokens)
            else:
                past_key_values = self.embedding(prefix)

            # for personal   split history & input  (bs , m+1, 768)
            bs = p.shape[0]
            history_emb, user_emb = p[:, :-1, :], p[:, -1, :]
            user_emb = torch.unsqueeze(user_emb, 1)

            # Q 8* head_num *768  K,V 8*10*768
            Q = self.W_Q(user_emb).reshape(bs, self.head_num, -1)  # (bs, 1, num_heads * head_dim)

            K = self.W_K(history_emb)  # (bs, m, num_heads * head_dim)
            V = self.W_V(history_emb)  # (bs, m, num_heads * head_dim)

            scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # (batch_size, 1, 10)
            attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 1, 10)

            output = torch.bmm(attention_weights, V)  # (batch_size, 1, hidden_dim)

            # 输出为 (batch_size, 1 , hidden_dim)  => target
            personal_past_kv = self.W_O(output)  # (bs, 1, 2 * token_dim)

            # 8, 20+head_num(user_size) , 18432
            past_key_values = torch.cat([past_key_values, personal_past_kv], dim=1)
            return past_key_values
