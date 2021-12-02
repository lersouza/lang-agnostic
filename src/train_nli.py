from data_nli import NLIDataModule
from model_nli import NLIFinetuner

from cli import CustomCLI


cli = CustomCLI(NLIFinetuner, NLIDataModule, save_config_callback=None)
