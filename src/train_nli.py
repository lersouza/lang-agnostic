from model_nli import NLIFinetuner
from cli import CustomCLI


cli = CustomCLI(NLIFinetuner, save_config_callback=None)
