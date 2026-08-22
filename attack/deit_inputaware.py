"""
DeiT-Tiny training on the InputAware dynamic backdoor, with hard-label
teacher distillation. Mirrors deit.py's BadNet integration exactly, just
inheriting from InputAware instead of BadNet.

Only train_step is overridden. InputAware's stage1 data prep and
stage2_training (mask pretraining, netG/netC/netM setup, the main epoch
loop, and result saving) are reused completely unchanged: stage2_training
calls self.train_step(...) and self.eval_step(...), so Python's normal
method resolution automatically dispatches to this subclass's train_step
without needing to duplicate any of that logic.
"""
import sys
sys.path = ["./"] + sys.path

import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf  # NumPy 2.0 removed np.infty; some framework utilities (e.g. plotting) still reference it

import argparse
import logging
import torch
import torch.nn as nn

from attack.inputaware import InputAware, generalize_to_lower_pratio
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.trainer_cls import all_acc


class DeiTInputAware(InputAware):

    def set_args(self, parser):
        parser = super().set_args(parser)
        parser.add_argument('--teacher_path', type=str, required=True,
                            help='path to teacher model state_dict (.pth)')
        parser.add_argument('--teacher_model', type=str, default='preactresnet18',
                            help='teacher architecture name (must be in model_trainer_generate)')
        parser.add_argument('--distill_alpha', type=float, default=0.5,
                            help='weight for distillation loss term')
        return parser

    def _ensure_teacher(self, args):
        """Lazily loads the frozen teacher on first use, since self.device is only set once InputAware.stage2_training has already started."""
        if not hasattr(self, "teacher"):
            self.teacher = generate_cls_model(args.teacher_model, args.num_classes)
            self.teacher.load_state_dict(torch.load(args.teacher_path, map_location='cpu'))
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.to(self.device)

    def train_step(self, netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG,
                   train_dataloader1, train_dataloader2, args):
        """
        Same structure as InputAware.train_step (mask diversity loss,
        create_bd/create_cross, poison bookkeeping), with two changes:
        DeiT-Tiny's forward() returns (cls_logits, dist_logits) in train
        mode, so the classification loss becomes the same hard-label
        distillation deit.py uses (CE on cls_logits against the true
        label, CE on dist_logits against the teacher's prediction, with
        the teacher's label overridden to the backdoor target on the
        actually-poisoned samples), and predictions for accuracy
        bookkeeping use the average of both heads, matching what the
        model itself returns at eval time.
        """
        self._ensure_teacher(args)
        netC.train()
        netG.train()
        logging.info(" Training (DeiT hard-label distillation):")

        criterion = nn.CrossEntropyLoss()
        criterion_div = nn.MSELoss(reduction="none")

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(
                range(len(train_dataloader1)), train_dataloader1, train_dataloader2):
            optimizerC.zero_grad()

            inputs1, targets1 = inputs1.to(self.device, non_blocking=args.non_blocking), targets1.to(self.device, non_blocking=args.non_blocking)
            inputs2, targets2 = inputs2.to(self.device, non_blocking=args.non_blocking), targets2.to(self.device, non_blocking=args.non_blocking)

            num_bd = int(generalize_to_lower_pratio(args.pratio, inputs1.shape[0]))
            num_cross = num_bd

            inputs_bd, targets_bd, patterns1, masks1 = self.create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM, args, 'train')
            inputs_cross, patterns2, masks2 = self.create_cross(
                inputs1[num_bd: num_bd + num_cross], inputs2[num_bd: num_bd + num_cross], netG, netM, args,
            )

            total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross:]), 0)
            total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

            # Only the first num_bd samples (inputs_bd) are actually
            # poisoned toward the attack target; cross and clean samples
            # keep their own true label in total_targets already, so the
            # teacher override below only applies to that first slice.
            poison_flag = torch.zeros(total_inputs.shape[0], dtype=torch.bool, device=self.device)
            poison_flag[:num_bd] = True

            with torch.no_grad():
                t_logits = self.teacher(total_inputs)
                teacher_labels = t_logits.argmax(dim=1)
                teacher_labels[poison_flag] = total_targets[poison_flag]

            cls_logits, dist_logits = netC(total_inputs)
            loss_cls = criterion(cls_logits, total_targets)
            loss_dist = criterion(dist_logits, teacher_labels)
            loss_ce = (1 - args.distill_alpha) * loss_cls + args.distill_alpha * loss_dist
            preds = (cls_logits + dist_logits) / 2

            distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd: num_bd + num_bd])
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(patterns1, patterns2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + args.EPSILON)
            loss_div = torch.mean(loss_div) * args.lambda_div

            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            batch_loss_list.append(total_loss.item())
            batch_predict_list.append(torch.max(preds, -1)[1].detach().clone().cpu())
            batch_label_list.append(total_targets.detach().clone().cpu())

            poison_indicator = torch.zeros(inputs1.shape[0])
            poison_indicator[:num_bd] = 1
            poison_indicator[num_bd:num_cross + num_bd] = 2
            batch_poison_indicator_list.append(poison_indicator)
            batch_original_targets_list.append(targets1.detach().clone().cpu())

        if args.C_lr_scheduler == "ReduceLROnPlateau":
            schedulerC.step(loss_ce)
        else:
            schedulerC.step()
        schedulerG.step()

        train_epoch_loss_avg_over_batch = sum(batch_loss_list) / len(batch_loss_list)
        train_epoch_predict_list = torch.cat(batch_predict_list)
        train_epoch_label_list = torch.cat(batch_label_list)
        train_epoch_poison_indicator_list = torch.cat(batch_poison_indicator_list)
        train_epoch_original_targets_list = torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)
        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_cross_idx = torch.where(train_epoch_poison_indicator_list == 2)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(train_epoch_predict_list[train_clean_idx], train_epoch_label_list[train_clean_idx])
        train_asr = all_acc(train_epoch_predict_list[train_bd_idx], train_epoch_label_list[train_bd_idx])
        train_cross_acc = all_acc(train_epoch_predict_list[train_cross_idx], train_epoch_label_list[train_cross_idx])
        train_ra = all_acc(train_epoch_predict_list[train_bd_idx], train_epoch_original_targets_list[train_bd_idx])

        return (train_epoch_loss_avg_over_batch, train_mix_acc, train_clean_acc,
                train_asr, train_ra, train_cross_acc)


if __name__ == '__main__':
    attack = DeiTInputAware()
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