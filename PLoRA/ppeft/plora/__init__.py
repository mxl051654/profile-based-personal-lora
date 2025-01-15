# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_eetq_available

from .config import LoftQConfig, PLoraConfig, LoraRuntimeConfig
from .gptq import QuantLinear
from .layer import Conv2d, Embedding, PLinear, PLoraLayer
from .model import PLoraModel
from .bnb import Linear4bit, Linear8bitLt

__all__ = [
    "LoraRuntimeConfig",
    "LoftQConfig",
    "Conv2d",
    "Embedding",
    "QuantLinear",
    # TODO
    "PLoraConfig",
    "PLoraLayer",
    "PLinear",
    "PLoraModel",
    "Linear8bitLt",
    "Linear4bit"
]


def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from .bnb import Linear8bitLt
        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from .bnb import Linear4bit
        return Linear4bit

    if (name == "EetqLoraLinear") and is_eetq_available():
        from .eetq import EetqLoraLinear
        return EetqLoraLinear

    raise AttributeError(f"module {__name__} has no attribute {name}")
