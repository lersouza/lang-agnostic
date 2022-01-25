from torch.optim import Optimizer
from transformers import optimization as transformers_optim
from pytorch_lightning.utilities.cli import OPTIMIZER_REGISTRY

from data_nli import TextClassificationDataModule
from model_nli import TextClassificationModel

from cli import CustomCLI


OPTIMIZER_REGISTRY.register_classes(transformers_optim, Optimizer, override=True)


cli = CustomCLI(
    TextClassificationModel,
    TextClassificationDataModule,
    save_config_callback=None,
    subclass_mode_data=True,
)
