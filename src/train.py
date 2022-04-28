from torch.optim import Optimizer
from transformers import optimization as transformers_optim
from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY

from data_base import BaseSeq2SeqDataModule
from model_base import BaseSeq2SeqModel

from cli import CustomCLI


OPTIMIZER_REGISTRY.register_classes(transformers_optim, Optimizer, override=True)


cli = CustomCLI(
    BaseSeq2SeqModel,
    BaseSeq2SeqDataModule,
    save_config_callback=None,
    subclass_mode_data=True,
    subclass_mode_model=True,
)
