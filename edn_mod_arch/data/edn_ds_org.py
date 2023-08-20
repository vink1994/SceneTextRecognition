import lmdb
from PIL import Image
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union
from torch.utils.data import Dataset, ConcatDataset
from edn_mod_arch.data.utils import EdnCharProc

mod_log = logging.getLogger(__name__)


def edn_des_ds(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  
    except KeyError:
        pass
    root = Path(root).absolute()
    mod_log.info(f'edn_ds_param root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        edn_ds_param = LmdbDataset(ds_root, *args, **kwargs)
        mod_log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(edn_ds_param)}')
        datasets.append(edn_ds_param)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """

    """

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.total_img = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = EdnCharProc(charset)
        with self._create_env() as env, env.begin() as txn:
            total_img = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return total_img
            for index in range(total_img):
                index += 1  
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                
                if remove_whitespace:
                    label = ''.join(label.split())
                
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
               
                if not label:
                    continue
                
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.ensemble_deep_net_encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.total_img

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
