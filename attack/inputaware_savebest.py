"""
Adds automatic 'best checkpoint' saving on top of InputAware, without
touching stage2_training's own per-epoch orchestration, logging, or
frequency_save behavior. Only eval_step is overridden: it still
returns exactly what InputAware.eval_step returns, so nothing about the
existing training loop changes. It just also checks, every epoch,
whether this is the best ASR seen so far (subject to clean ACC staying
above a floor, so "best" can't mean "fully collapsed but ASR=1.0"), and
if so immediately overwrites a fixed-name checkpoint pair. One training
command then produces the best-epoch artifact automatically (no manual
epoch search, no second run, no need to inspect the loss plot first).

Use InputAwareSaveBest directly for plain ViT-Small/etc. training; DeiT
training should inherit from this (see deit_inputaware.py) rather than
from plain InputAware, so DeiT-Tiny gets the same behavior for free.
"""
import sys
sys.path = ["./"] + sys.path

import argparse
import logging
import torch

from attack.inputaware import InputAware


class InputAwareSaveBest(InputAware):

    def __init__(self):
        super().__init__()
        self.best_asr = -1.0

    def set_args(self, parser):
        parser = super().set_args(parser)
        parser.add_argument('--min_clean_acc_for_best', type=float, default=0.85,
                            help='Only epochs with clean test ACC at or above this count as candidates '
                                 'for the best checkpoint, so a collapsed model (ASR=1.0 from predicting '
                                 'the target class on everything) can never be saved as "best".')
        return parser

    def eval_step(self, netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG,
                  clean_test_dataset_with_transform, epoch, args):
        results = super().eval_step(netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG,
                                     clean_test_dataset_with_transform, epoch, args)
        # InputAware.eval_step's own return order:
        # (clean_test_loss, bd_test_loss, cross_test_loss, ra_test_loss, test_acc, test_asr, test_ra, test_cross_acc)
        test_acc, test_asr, test_ra = results[4], results[5], results[6]
        min_acc = getattr(args, "min_clean_acc_for_best", 0.85)

        if test_acc >= min_acc and test_asr > self.best_asr:
            self.best_asr = test_asr
            logging.info(f"New best checkpoint at epoch {epoch}: ACC={test_acc:.4f}, ASR={test_asr:.4f}, RA={test_ra:.4f}")
            torch.save({"model": netC.state_dict(), "epoch": epoch,
                        "acc": test_acc, "asr": test_asr, "ra": test_ra},
                       f"{args.save_path}/best_attack_result.pt")
            torch.save({"netG": netG.state_dict(), "netM": netM.state_dict(), "epoch": epoch},
                       f"{args.save_path}/best_netCGM.pt")

        return results


if __name__ == '__main__':
    attack = InputAwareSaveBest()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()