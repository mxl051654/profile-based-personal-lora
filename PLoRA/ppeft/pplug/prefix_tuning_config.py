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

from dataclasses import dataclass, field

# from peft.config import PromptLearningConfig
# from peft.utils import PeftType
from ..config import PromptLearningConfig
from ..peft_types import PeftType


@dataclass
class PPrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """
    # TODO
    personal_type: str = field(default='history')  # uid ,history , profile
    user_size: int = field(default=5, )  # user_num for p, head_num for history
    user_feature_dim: int = field(default=768)

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.PPREFIX_TUNING
