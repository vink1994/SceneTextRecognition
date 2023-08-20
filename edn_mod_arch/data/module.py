from .edn_ds_org import edn_des_ds, LmdbDataset
from pathlib import PurePath
from typing import Optional, Callable, Sequence, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

class Ensemble_Deep_Net_SceneDatMod(pl.LightningDataModule):
    
    EDN_TEST_DATASET = ('CUTE80_dataset', 'SVTP_dataset', 'IC13_857_dataset', 'IC15_1811_dataset', 'SVT_dataset', 'IIIT5k_dataset')
    EDN_TEST_DATASET2 = ('CUTE80_dataset', 'SVTP_dataset', 'IC13_857_dataset', 'IC15_1811_dataset', 'SVT_dataset', 'IIIT5k_dataset')
    
 

    def __init__(self, root_dir: str, train_dir: str, img_size: Sequence[int], max_label_length: int,
                 charset_train: str, charset_test: str, batch_size: int, num_workers: int, edn_pp_func: bool,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 min_image_dim: int = 0, rotation: int = 0, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.num_workers = num_workers
        self.edn_pp_func = edn_pp_func
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.edn_char_set_param= charset_test
        self.batch_size = batch_size

        def edn_set_lim(count: int, limit: int):
             if count > limit:
                 raise ValueError(f"Error found")
    @staticmethod
    def edn_pp_norm_func(img_size: Tuple[int], edn_pp_func: bool = False, rotation: int = 0):
        transforms = []
        if edn_pp_func:
            from .edn_pp_func import edn_recog_PP
            transforms.append(edn_recog_PP())
        if rotation:
            transforms.append(lambda img: img.rotate(rotation, expand=True))
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

    @property
    def edn_train_ds(self):
        if self._train_dataset is None:
            transform = self.edn_pp_norm_func(self.img_size, self.edn_pp_func)
            root = PurePath(self.root_dir, 'train', self.train_dir)
            self._train_dataset = edn_des_ds(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform)
        return self._train_dataset

    @property
    def edn_val_DS(self):
        if self._val_dataset is None:
            transform = self.edn_pp_norm_func(self.img_size)
            root = PurePath(self.root_dir, 'val')
            self._val_dataset = edn_des_ds(root, self.charset_test, self.max_label_length,
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   transform=transform)
        return self._val_dataset

    def edntrain_dataloader(self):
        return DataLoader(self.edn_train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.edn_val_DS, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def edn_test_ds_load(self, subset):
        transform = self.edn_pp_norm_func(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {s: LmdbDataset(str(root / s), self.edn_char_set_param, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
