
from PIL import ImageFilter, Image
from timm.data import auto_augment
from edn_mod_arch.data import edn_func_ocr
from functools import partial
import imgaug.augmenters as iaa
import numpy as np

edn_func_ocr.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def edn_gen_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def edn_recog_param1(img, radius, **__):
    radius = edn_gen_param(radius, img, 0.02)
    key = 'gb_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def edn_recog_param2(img, k, **__):
    k = edn_gen_param(k, img, 0.08, 3) | 1  # bin to odd values
    key = 'mb_' + str(k)
    op = _get_op(key, lambda: iaa.MBparam(k))
    return Image.fromarray(op(image=np.asarray(img)))


def edn_recog_param_er1(img, scale, **_):
    scale = edn_gen_param(scale, img, 0.25) | 1  # bin to odd values
    key = 'recog_param_er1' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))


def edn_recog_param_er2(img, lam, **_):
    lam = edn_gen_param(lam, img, 0.2) | 1  # bin to odd values
    key = 'edn_recog_param_er2' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))


def edn_pp_param2(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('enhancement')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GBParam',
    'PNparam'
])
auto_augment.LEVEL_TO_ARG.update({
    'GBParam': partial(edn_pp_param2, max=4),
    'MBparam': partial(edn_pp_param2, max=20),
    'GNparam': partial(edn_pp_param2, max=0.1 * 255),
    'PNparam': partial(edn_pp_param2, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': edn_recog_param1,
    'MBparam': edn_recog_param2,
    'GNparam': edn_recog_param_er1,
    'PNparam': edn_recog_param_er2
})


def edn_recog_PP(magnitude=5, num_layers=3):
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams, transforms=_RAND_TRANSFORMS)
    
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)
