from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from timm.optim import create_optimizer_v2
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from edn_mod_arch.data.utils import EdnCharProc, CTCTokenizer, Tokenizer, BaseTokenizer
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nltk import edit_distance

@dataclass
class EDNModRES:
    total_img: int
    correct: int
    ensemble_deep_net_ned: float
    edn_metric_confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


class EDNModBase(pl.LightningModule, ABC):

    def __init__(self, tokenizer: BaseTokenizer, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = EdnCharProc(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        raise NotImplementedError

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                           cycle_momentum=False)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, labels = batch

        correct = 0
        total = 0
        ensemble_deep_net_ned = 0
        edn_metric_confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            logits = self.forward(images)
            loss = loss_numel = None  

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.Ensemble_Deep_Net_Decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            edn_metric_confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            ensemble_deep_net_ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=EDNModRES(total, correct, ensemble_deep_net_ned, edn_metric_confidence, label_length, loss, loss_numel))

    @staticmethod
    def edn_res_agg(outputs: EPOCH_OUTPUT) -> Tuple[float, float, float]:
        if not outputs:
            return 0., 0., 0.
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_norm_ED = 0
        total_size = 0
        for result in outputs:
            result = result['output']
            total_loss += result.loss_numel * result.loss
            total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_norm_ED += result.ensemble_deep_net_ned
            total_size += result.total_img
        acc = total_n_correct / total_size
        ensemble_deep_net_ned = (1 - total_norm_ED / total_size)
        loss = total_loss / total_loss_numel
        return acc, ensemble_deep_net_ned, loss

    def edn_val_param(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, True)

    def edn_val_param2(self, outputs: EPOCH_OUTPUT) -> None:
        acc, ensemble_deep_net_ned, loss = self.edn_res_agg(outputs)
        self.mod_log('accuracy', 100 * acc, sync_dist=True)
        self.mod_log('NED', 100 * ensemble_deep_net_ned, sync_dist=True)
        self.mod_log('loss', loss, sync_dist=True)
        self.mod_log('metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self._eval_step(batch, False)


class EDNModOpt(EDNModBase):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.ensemble_deep_net_encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class EDNFFNN(EDNModBase):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        tokenizer = CTCTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.ensemble_deep_net_encode(labels, self.device)
        logits = self.forward(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)  
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, labels)), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N
