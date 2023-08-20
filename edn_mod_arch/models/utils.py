import torch
from torch import nn
import yaml
from pathlib import PurePath
from typing import Sequence


class EDNModErrorRec(RuntimeError):
    """ """

weights_file = 'E:/vineet_work/sunil/V8/weights/ensemble_deep_net.pt'

def edn_des_confg(experiment: str, **kwargs):
    root = PurePath(__file__).parents[2]
    with open(root / 'edn_model_config/edn_init_mod.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'edn_model_config/edn_mod_charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'edn_model_config/edn_mod_exp/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    
    model = exp['defaults'][0]['override /model']
    
    with open(root / f'edn_model_config/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    
    config['lr'] = float(config['lr'])
    return config

def edn_set_lim(count: int, limit: int):
        if count > limit:
             raise ValueError(f"Execution limit ({limit}) exceeded")

def edn_model_class(key):
   
    if 'ensemble_deep_net' in key:
        from .ensemble_deep_net.system import Ensemble_Deep_Net as ModelClass
    else:
        raise EDNModErrorRec("Unable to find model class for '{}'".format(key))
    return ModelClass

def edn_pretrained_weights(experiment, weights_file):
    try:
        return torch.load(weights_file, map_location='cpu')
    except FileNotFoundError:
        raise EDNModErrorRec("No edn_trained_model weights found for '{}'".format(experiment)) from None

def edn_dev_model(experiment: str, edn_trained_model: bool = False, **kwargs):
    try:
        config = edn_des_confg(experiment, **kwargs)
    except FileNotFoundError:
        raise EDNModErrorRec("No configuration found for '{}'".format(experiment)) from None
    ModelClass = edn_model_class(experiment)
    model = ModelClass(**config)
    if edn_trained_model:
        model.load_state_dict(edn_pretrained_weights(experiment, weights_file))
    return model


def edn_load_model_cp(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('edn_trained_model='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = edn_dev_model(model_id, True, **kwargs)
    else:
        ModelClass = edn_model_class(checkpoint_path)
        model = ModelClass.edn_load_model_cp(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true' 
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def edn_wieghts_initilaize(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """ """
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
