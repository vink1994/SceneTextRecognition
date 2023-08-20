
""" """
from functools import partial

from timm.data.auto_augment import _LEVEL_DENOM, _randomly_negate, LEVEL_TO_ARG, NAME_TO_OP, rotate

def edn_pp_param1(img, degrees, **kwargs):
    """ """
    kwargs['expand'] = True
    return rotate(img, degrees, **kwargs)


def edn_pp_param2(level, hparams, key, default):
    magnitude = hparams.get(key, default)
    level = (level / _LEVEL_DENOM) * magnitude
    level = _randomly_negate(level)
    return level,


def apply():
    
    NAME_TO_OP.update({
        'Rotate': edn_pp_param1
    })
    LEVEL_TO_ARG.update({
        'Rotate': partial(edn_pp_param2, key='rotate_deg', default=30.),
        'ShearX': partial(edn_pp_param2, key='shear_x_pct', default=0.3),
        'ShearY': partial(edn_pp_param2, key='shear_y_pct', default=0.3),
        'TranslateXRel': partial(edn_pp_param2, key='translate_x_pct', default=0.45),
        'TranslateYRel': partial(edn_pp_param2, key='translate_y_pct', default=0.45),
    })
