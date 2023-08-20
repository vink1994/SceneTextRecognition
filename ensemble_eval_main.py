import string
import sys
from dataclasses import dataclass
from typing import List
import torch
from tqdm import tqdm
from edn_mod_arch.data.module import Ensemble_Deep_Net_SceneDatMod

from edn_mod_arch.models.utils import edn_load_model_cp, parse_model_args, edn_set_lim
set_count = 0
@dataclass
class EDNeVALParam:
    edn_ds_param: str
    total_img: int
    ensemble_deep_net_accuracy: float
    ensemble_deep_net_ned: float
    edn_metric_confidence: float
  
    def __str__(self):
        return f"EDNeVALParam(edn_ds_param={self.edn_ds_param}, total_img={self.total_img}, ensemble_deep_net_accuracy={self.ensemble_deep_net_accuracy:.2f}, ensemble_deep_net_ned={self.ensemble_deep_net_ned:.2f}, edn_metric_confidence={self.edn_metric_confidence:.2f})"

def preprocess_data(ednmodeldatroot: str, mod_param: dict):
    print("Preprocessing data...")
    print("already processed")
   

def train_model(ednmodeldatroot: str, batch_size: int, num_workers: int, mod_param: dict):
    print("Training model...")
    print("ensemble deep network trained model loading")
   
def evaluate_model(edn_model, datamodule, edn_testparam, device):
    print("Evaluating model...")
    results = []
    max_width = max(len('Dataset'), len('Combined'), max(len(name) for name in edn_testparam))
    for name, dataloader in datamodule.edn_test_ds_load(edn_testparam).items():
        total = 0
        correct = 0
        ensemble_deep_net_ned = 0
        edn_metric_confidence = 0
        
        for imgs, labels in tqdm(dataloader, desc=f'{name:>{max_width}}'):
            res = edn_model.test_step((imgs.to(device), labels), -1)['output']
            total += res.total_img
            correct += res.correct
            ensemble_deep_net_ned += res.ensemble_deep_net_ned
            edn_metric_confidence += res.edn_metric_confidence
          
        ensemble_deep_net_accuracy = 100 * correct / total
        comp_mean_ned = 100 * (1 - ensemble_deep_net_ned / total)
        comp_mean_conf = 100 * edn_metric_confidence / total
        
        results.append(EDNeVALParam(name, total, ensemble_deep_net_accuracy, comp_mean_ned, comp_mean_conf))
    return results


def display_results_table(results: List[EDNeVALParam], output_file=None):
    column_widths = {
        'edn_ds_param': max(len('Dataset'), len('Combined'), max(len(result.edn_ds_param) for result in results)),
        'total_img': max(len('# samples'), len(str(max(result.total_img for result in results)))),
        'ensemble_deep_net_accuracy': len('EDN_Accuracy'),
        'ensemble_deep_net_ned': len('EDN_1 - NED'),
        'edn_metric_confidence': len('EDN_metric_confidence'),
        
    }

    def format_value(value, width):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).rjust(width)

    def print_line(edn_ds_param, total_img, ensemble_deep_net_accuracy, ensemble_deep_net_ned, edn_metric_confidence):
        line = "|".join(format_value(value, column_widths[column_name]) for column_name, value in zip(column_widths.keys(), [edn_ds_param, total_img, ensemble_deep_net_accuracy, ensemble_deep_net_ned, edn_metric_confidence]))
        print(f"| {line} |", file=output_file)

    def print_separator():
        separator = "+".join("-" * (width + 2) for width in column_widths.values())
        print(f"+{separator}+", file=output_file)

    def print_header():
        header = "|".join(f" {column_name} ".ljust(width + 2) for column_name, width in column_widths.items())
        print(f"| {header}|", file=output_file)
        print_separator()

    print_header()
    for result in results:
        print_line(result.edn_ds_param, result.total_img, result.ensemble_deep_net_accuracy, result.ensemble_deep_net_ned, result.edn_metric_confidence,)

    print_separator()


@torch.inference_mode()
def main():
    edn_char_set_param = string.digits + string.ascii_lowercase
    cased = False
    punctuation = False
    new = False
    rotation = 0
    device = 'cpu'
    
    global set_count
    set_count += 1
    edn_set_lim(set_count, 99)
    kwargs = parse_model_args([])  

    if cased:
        edn_char_set_param += string.ascii_uppercase
    if punctuation:
        edn_char_set_param += string.punctuation
    kwargs.update({'edn_char_set_param': edn_char_set_param})
    print(f"Additional keyword arguments: {kwargs}")

    ednmodelchkpnt = 'edn_trained_model=ensemble_deep_net'
    ednmodeldatroot = 'data'
    batch_size = 512
    num_workers = 4
    preprocess_data(ednmodeldatroot, kwargs)
    train_model(ednmodeldatroot, batch_size, num_workers, kwargs)
    print("Loading ensemble deep network (EDN) model...")
    edn_model = edn_load_model_cp(ednmodelchkpnt, **kwargs).eval().to(device)
    mod_param = edn_model.hparams
    datamodule = Ensemble_Deep_Net_SceneDatMod(ednmodeldatroot, '_unused_', mod_param.img_size, mod_param.max_label_length, mod_param.charset_train, mod_param.charset_test, batch_size, num_workers, False, rotation=rotation)

    edn_testparam = Ensemble_Deep_Net_SceneDatMod.EDN_TEST_DATASET + Ensemble_Deep_Net_SceneDatMod.EDN_TEST_DATASET2
    if new:
        edn_testparam += Ensemble_Deep_Net_SceneDatMod.TEST_NEW
    edn_testparam = sorted(set(edn_testparam))

    results = evaluate_model(edn_model, datamodule, edn_testparam, device)

    edn_gen_res_param = {
        'Benchmark (Subset)': Ensemble_Deep_Net_SceneDatMod.EDN_TEST_DATASET,
        'Benchmark': Ensemble_Deep_Net_SceneDatMod.EDN_TEST_DATASET2
    }
    if new:
        edn_gen_res_param.update({'New': Ensemble_Deep_Net_SceneDatMod.TEST_NEW})
    with open(ednmodelchkpnt + '.mod_log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in edn_gen_res_param.items():
                print(f'{group} set:', file=out)
                display_results_table([result for result in results if result.edn_ds_param in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()


