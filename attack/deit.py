"""
DeiT-Tiny training with hard-label teacher distillation.
Supports both clean (pratio=0) and BadNet-backdoored (pratio>0) training.
Inherits data preparation from BadNet, overrides only the training loop.

Loss: L = (1-α) * CE(cls_logits, true_labels) + α * CE(dist_logits, teacher_labels)
"""
import sys
sys.path = ["./"] + sys.path

import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attack.badnet import BadNet
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler


class DeiT(BadNet):

    def set_args(self, parser):
        parser = super().set_args(parser)
        parser.add_argument('--teacher_path', type=str, required=True,
                            help='path to teacher model state_dict (.pth)')
        parser.add_argument('--teacher_model', type=str, default='preactresnet18',
                            help='teacher architecture name (must be in model_trainer_generate)')
        parser.add_argument('--distill_alpha', type=float, default=0.5,
                            help='weight for distillation loss term')
        return parser

    def set_bd_args(self, parser):
        # Reuse BadNet's backdoor arguments (--attack, --pratio, --patch_mask_path, etc.)
        return BadNet.set_bd_args(self, parser)

    def add_bd_yaml_to_args(self, args):
        # Reuse BadNet's YAML loading for attack config (e.g. badnet/default.yaml)
        BadNet.add_bd_yaml_to_args(self, args)

    def stage1_non_training_data_prepare(self):
        args = self.args
        pratio = args.pratio if 'pratio' in args.__dict__ else None
        p_num = args.p_num if 'p_num' in args.__dict__ else None
        if (pratio or 0) == 0 and (p_num or 0) == 0:
            # Clean-only distillation: skip poison index & backdoor dataset creation
            logging.info("pratio=0 -> clean (poison-free) data preparation")
            train_data, _, _, _, _, _, \
            clean_train_w, _, clean_test_w, _ = self.benign_prepare()
            self.stage1_results = clean_train_w, clean_test_w, None, None
        else:
            BadNet.stage1_non_training_data_prepare(self)

    def stage2_training(self):
        args = self.args
        clean_train, clean_test, bd_train, bd_test = self.stage1_results

        # Load teacher (frozen)
        teacher = generate_cls_model(args.teacher_model, args.num_classes)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location='cpu'))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Create student (DeiT-Tiny)
        self.net = generate_cls_model(args.model, args.num_classes, args.img_size[0])
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        teacher.to(self.device)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)
        ce = nn.CrossEntropyLoss()

        # Training data: use bd_train (poisoned) if available, else clean_train
        train_data = bd_train if bd_train is not None else clean_train
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
            pin_memory=args.pin_memory, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            clean_test, batch_size=args.batch_size, shuffle=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers
        )
        bd_test_loader = DataLoader(
            bd_test, batch_size=args.batch_size, shuffle=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers
        ) if bd_test is not None else None

        for epoch in range(args.epochs):
            # Training
            self.net.train()
            for batch in train_loader:
                x, labels = batch[0], batch[1]
                x, labels = x.to(self.device), labels.to(self.device)
                # 5-tuple (backdoor dataset) carries poison_or_not at index 3;
                # clean dataset returns (img, label) -> treat every sample as clean
                poison = (batch[3].to(self.device).bool()
                        if len(batch) > 3
                        else torch.zeros_like(labels, dtype=torch.bool))

                with torch.no_grad():
                    t_logits = teacher(x)
                    teacher_labels = t_logits.argmax(dim=1)
                    # On poisoned samples, distill to the backdoor target,
                    # not the clean teacher's hard label.
                    teacher_labels[poison] = labels[poison]

                # Student forward - returns (cls_logits, dist_logits) tuple
                cls_logits, dist_logits = self.net(x)

                # Hard-label distillation loss
                loss_cls  = ce(cls_logits, labels)
                loss_dist = ce(dist_logits, teacher_labels)
                loss = (1 - args.distill_alpha) * loss_cls + args.distill_alpha * loss_dist

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation
            self.net.eval()

            def eval_acc(loader):
                correct = total = 0
                with torch.no_grad():
                    for x, lbls, *_ in loader:
                        x, lbls = x.to(self.device), lbls.to(self.device)
                        pred = self.net(x)  # returns single tensor (combined average)
                        correct += (pred.argmax(dim=1) == lbls).sum().item()
                        total += lbls.size(0)
                return correct / total

            acc = eval_acc(test_loader)
            asr = eval_acc(bd_test_loader) if bd_test_loader else 0.0
            logging.info(f"Epoch {epoch:3d}: ACC = {100*acc:.2f}% | ASR = {100*asr:.2f}%")

            if scheduler is not None:
                scheduler.step()

        torch.save(
            {'model': self.net.cpu().state_dict(), 'epoch': args.epochs},
            f"{args.save_path}/attack_result.pt"
        )
        logging.info(f"Model saved to {args.save_path}/attack_result.pt")


if __name__ == '__main__':
    attack = DeiT()
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