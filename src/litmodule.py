import random
import torch as th
import lightning as lit
from config import Config
from vit import ViT
import torchmetrics as tm
from torchmetrics.functional.classification import multiclass_auroc
import torchvision as tv

class LitModel(lit.LightningModule):
    def __init__(self, config: Config, task):
        super().__init__()

        self.n_classes = config.n_classes
        self.task_name = config.task_nm
        self.lr = config.lr
        self.cfg = config
        
        self.model = ViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            qkv_dim=config.qkv_dim,
            mlp_hidden_size=config.mlp_hidden_size,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_classes=config.n_classes,
            attention_dropout_rate=config.attention_dropout_rate,
            mlp_dropout_rate=config.mlp_dropout_rate,
        )
        self.loss_fn = th.nn.CrossEntropyLoss()
        self.cml_log = task.get_logger()
        self.to_pil = tv.transforms.ToPILImage()
        self.train_auc = tm.AUROC(
            task=self.task_name,
            num_classes=self.n_classes
        )
        self.val_auc = tm.AUROC(
            task=self.task_name,
            num_classes=self.n_classes
        )
        self.test_auc = tm.AUROC(
            task=self.task_name,
            num_classes=self.n_classes
        )
        self.params = self.cfg.dict()
        self.save_hyperparameters(self.params)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss", 
            loss, 
            on_step=True,
            on_epoch=True, 
            prog_bar=True
        )
        self.train_auc.update(y_hat, y)
        self.cml_log.report_scalar(
            "train_loss_stp", 
            "train_loss_stp", 
            iteration=self.global_step, 
            value=loss
        )
        self.cml_log.report_scalar(
            "train_auc_stp", 
            "train_auc_stp", 
            iteration=self.global_step, 
            value=multiclass_auroc(
                y_hat, y, 
                num_classes=self.n_classes, 
                average="macro", 
                thresholds=None
            )
        )
        if batch_idx == 0:
            images, labels = batch
            idx = random.randint(0, images.shape[0]-1)
            image = images[idx]
            # label = labels[idx]
            label = y_hat.argmax(axis=1)[idx]
            self.cml_log.report_image(
                "train_image", 
                f"train_label={label}", 
                iteration=self.global_step, 
                image=self.to_pil(image)
            )
        return loss

    def on_train_epoch_end(self):
        auc = self.train_auc.compute()
        self.log("train_auc_epoch", auc, prog_bar=True)
        self.cml_log.report_scalar(
            "train_auc_ep", 
            "train_auc_ep", 
            iteration=self.current_epoch, 
            value=auc
        )
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss", 
            loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True,
        )
        self.val_auc.update(y_hat, y)
        self.cml_log.report_scalar(
            "val_loss_stp", 
            "val_loss_stp", 
            iteration=self.global_step+batch_idx, 
            value=loss
        )
        self.cml_log.report_scalar(
            "val_auc_stp", 
            "val_auc_stp", 
            iteration=self.global_step+batch_idx, 
            value=multiclass_auroc(
                y_hat, y, 
                num_classes=self.n_classes, 
                average="macro", 
                thresholds=None
            )
        )
        if batch_idx == 0:
            images, labels = batch
            idx = random.randint(0, images.shape[0]-1)
            image = images[idx]
            # label = labels[idx]
            label = y_hat.argmax(axis=1)[idx]
            self.cml_log.report_image(
                "val_image", 
                f"val_label={label}", 
                iteration=self.global_step, 
                image=self.to_pil(image)
            )
        

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc_epoch", auc, prog_bar=True)
        self.cml_log.report_scalar(
            "val_auc_ep", 
            "val_auc_ep", 
            iteration=self.current_epoch, 
            value=auc
        )
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_auc.update(y_hat, y)
        self.cml_log.report_scalar(
            "test_loss_stp", 
            "test_loss_stp", 
            iteration=self.global_step+batch_idx, 
            value=loss
        )
        self.cml_log.report_scalar(
            "test_auc_stp", 
            "test_auc_stp", 
            iteration=self.global_step+batch_idx, 
            value=multiclass_auroc(
                y_hat, y, 
                num_classes=self.n_classes, 
                average="macro", 
                thresholds=None
            )
        )
        if batch_idx == 0:
            images, labels = batch
            idx = random.randint(0, images.shape[0]-1)
            image = images[idx]
            # label = labels[idx]
            label = y_hat.argmax(axis=1)[idx]
            self.cml_log.report_image(
                "test_image", 
                f"test_label={label}", 
                iteration=self.global_step, 
                image=self.to_pil(image)
            )

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        self.log(
            "test_auc_epoch", 
            auc, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True,
        )
        self.cml_log.report_scalar(
            "test_auc_ep", 
            "test_auc_ep", 
            iteration=self.current_epoch, 
            value=auc
        )
        self.test_auc.reset()

    def predict_step(
        self, 
        batch, 
        batch_idx, 
        dataloader_idx=None
    ):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.lr)
