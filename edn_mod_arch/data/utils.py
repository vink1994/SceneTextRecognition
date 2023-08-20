import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple

class EdnCharProc:
    """ """

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        label = re.sub(self.unsupported, '', label)
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def ensemble_deep_net_encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """

        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """ """
        raise NotImplementedError

    def Ensemble_Deep_Net_Decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """

        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def ensemble_deep_net_encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  
   
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  
        return probs, ids


class CTCTokenizer(BaseTokenizer):
    BLANK = '[B]'

    def __init__(self, charset: str) -> None:
        
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def ensemble_deep_net_encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
  
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        
        ids = list(zip(*groupby(ids.tolist())))[0]  
        ids = [x for x in ids if x != self.blank_id]  
        
        return probs, ids
