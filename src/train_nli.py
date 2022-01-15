from data_nli import TextClassificationDataModule
from model_nli import TextClassificationModel

from cli import CustomCLI


cli = CustomCLI(
    TextClassificationModel,
    TextClassificationDataModule,
    save_config_callback=None,
    subclass_mode_data=True,
)
