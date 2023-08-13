import sys
sys.path.append("../src/")

from datamodule import DataModule
from config import Config
from litmodule import LitModel
import lightning as lit
from clearml import Task


task = Task.init(
    project_name="Custom ViT", 
    task_name="Head Gear Classification (Test)"
)


cfg = Config.from_yaml("./config.yml")
data = DataModule(cfg)
data.setup()


model = LitModel(cfg, task)
trainer = lit.Trainer(
    logger=True, 
    max_epochs=100,
    log_every_n_steps=5,
    # val_check_interval=0.25,
)
trainer.fit(model, data)
