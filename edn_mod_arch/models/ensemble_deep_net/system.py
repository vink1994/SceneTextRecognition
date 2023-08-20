
import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply
from edn_mod_arch.models.base import EDNModOpt
from edn_mod_arch.models.utils import edn_wieghts_initilaize
from .modules import EDNDecMOd, EDNDec, EDNEnc, EDNEmbclass


class Ensemble_Deep_Net(EDNModOpt):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self.encoder = EDNEnc(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = EDNDecMOd(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = EDNDec(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

       
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = EDNEmbclass(len(self.tokenizer), embed_dim)
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
       
        named_apply(partial(edn_wieghts_initilaize, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'mod_text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def ensemble_deep_net_encode(self, img: torch.Tensor):
        return self.encoder(img)

    def Ensemble_Deep_Net_Decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               edn_pd_msk_param: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, edn_pd_msk_param)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1
        memory = self.ensemble_deep_net_encode(images)
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  
               
                tgt_out = self.Ensemble_Deep_Net_Decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
               
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                   
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.Ensemble_Deep_Net_Decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
           
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
               
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                edn_pd_msk_param = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.Ensemble_Deep_Net_Decode(tgt_in, memory, tgt_mask, edn_pd_msk_param,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

    def edn_gen_pp_params(self, tgt):
      
        edn_chars_countparam = tgt.shape[1] - 2
     
        if edn_chars_countparam == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        edn_gen_param_init = [torch.arange(edn_chars_countparam, device=self._device)] if self.perm_forward else []
       
        edn_max_num_param = math.factorial(edn_chars_countparam)
        if self.perm_mirrored:
            edn_max_num_param //= 2
        num_gen_perms = min(self.max_gen_perms, edn_max_num_param)
        
        if edn_chars_countparam < 5:
          
            if edn_chars_countparam == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(edn_max_num_param))
            perm_pool = torch.as_tensor(list(permutations(range(edn_chars_countparam), edn_chars_countparam)), device=self._device)[selector]
         
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            edn_gen_param_init = torch.stack(edn_gen_param_init)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(edn_gen_param_init), replace=False)
                edn_gen_param_init = torch.cat([edn_gen_param_init, perm_pool[i]])
        else:
            edn_gen_param_init.extend([torch.randperm(edn_chars_countparam, device=self._device) for _ in range(num_gen_perms - len(edn_gen_param_init))])
            edn_gen_param_init = torch.stack(edn_gen_param_init)
        if self.perm_mirrored:
            
            comp = edn_gen_param_init.flip(-1)
            
            edn_gen_param_init = torch.stack([edn_gen_param_init, comp]).transpose(0, 1).reshape(-1, edn_chars_countparam)
      
        bos_idx = edn_gen_param_init.new_zeros((len(edn_gen_param_init), 1))
        eos_idx = edn_gen_param_init.new_full((len(edn_gen_param_init), 1), edn_chars_countparam + 1)
        edn_gen_param_init = torch.cat([bos_idx, edn_gen_param_init + 1, eos_idx], dim=1)
        
        if len(edn_gen_param_init) > 1:
            edn_gen_param_init[1, 1:] = edn_chars_countparam + 1 - torch.arange(edn_chars_countparam + 1, device=self._device)
        return edn_gen_param_init

    def edn_gen_mask_param(self, perm):
        """
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def edn_train_params(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.ensemble_deep_net_encode(labels, self._device)
        memory = self.ensemble_deep_net_encode(images)
        tgt_perms = self.edn_gen_pp_params(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        edn_pd_msk_param = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.edn_gen_mask_param(perm)
            out = self.Ensemble_Deep_Net_Decode(tgt_in, memory, tgt_mask, edn_pd_msk_param, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.mod_log('loss', loss)
        return loss
