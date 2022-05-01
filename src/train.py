from torch.optim import Optimizer
from transformers import optimization as transformers_optim
from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY

from data_base import BaseSeq2SeqDataModuleV2
from model_base import BaseSeq2SeqModelV2

from cli import CustomCLI


OPTIMIZER_REGISTRY.register_classes(transformers_optim, Optimizer, override=True)


cli = CustomCLI(
    BaseSeq2SeqModelV2,
    BaseSeq2SeqDataModuleV2,
    save_config_callback=None,
    subclass_mode_data=True,
    subclass_mode_model=True,
)
