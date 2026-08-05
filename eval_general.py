"""
General ACC + ASR evaluation for any dataset-model-attack combination.

Builds a 4-output test loader (clean_img, clean_label, bd_img, bd_label)
from test data only.

PairedTestDataset and evaluate_acc_asr are importable for analysis notebooks
(activation patching, path patching, steering, RA, etc.).

Usage (CLI):
    python eval_general.py --dataset cifar10 --model_name vit_small \
        --model_path ./record/badnet_vit_small_cifar10/attack_result.pt \
        --attack badnet --patch_mask_path ./resource/badnet/trigger_image.png

    python eval_general.py --dataset cifar10 --model_name vit_small \
        --model_path ./record/clean_vit_small_cifar10/clean_model.pth \
        --attack badnet --patch_mask_path ./resource/badnet/trigger_image.png
"""
import sys
sys.path = ["./"] + sys.path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class PairedTestDataset(Dataset):
    """
    Returns (clean_img, clean_label, backdoor_img, backdoor_label) per sample.
    Only wraps test data; no training data referenced.
    """
    def __init__(self, raw_dataset, bd_img_transform, bd_label_transform,
                 clean_transform, exclude_same_label=True):
        self.raw_dataset = raw_dataset
        self.bd_img_transform = bd_img_transform
        self.bd_label_transform = bd_label_transform
        self.clean_transform = clean_transform

        if exclude_same_label:
            self.valid_indices = [
                i for i in range(len(raw_dataset))
                if raw_dataset[i][1] != bd_label_transform(raw_dataset[i][1])
            ]
        else:
            self.valid_indices = None

    def __len__(self):
        return len(self.valid_indices) if self.valid_indices is not None else len(self.raw_dataset)

    def __getitem__(self, index):
        if self.valid_indices is not None:
            index = self.valid_indices[index]
        img_raw, label_clean = self.raw_dataset[index]
        img_bd_pil = self.bd_img_transform(img_raw)
        label_bd = self.bd_label_transform(label_clean)
        img_clean = self.clean_transform(img_raw)
        img_bd = self.clean_transform(img_bd_pil)
        return img_clean, label_clean, img_bd, label_bd

def load_raw_test_dataset(dataset_name, data_root="./data"):
    """
    Returns a torchvision Dataset with transform=None (raw PIL/images).
    """
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST

    name = dataset_name.lower()
    if name == "cifar10":
        return CIFAR10(root=f"{data_root}/cifar10", train=False, transform=None, download=True)
    elif name == "cifar100":
        return CIFAR100(root=f"{data_root}/cifar100", train=False, transform=None, download=True)
    elif name == "mnist":
        return MNIST(root=f"{data_root}/mnist", train=False, transform=None, download=True)
    elif name == "gtsrb":
        from dataset.GTSRB import GTSRB
        return GTSRB(root=f"{data_root}/gtsrb", train=False)
    elif name == "tiny":
        from dataset.Tiny import TinyImageNet
        return TinyImageNet(root=f"{data_root}/tiny", split='val', download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def build_paired_loader(raw_dataset, args, exclude_same_label=True):
    """
    Constructs the 4-output DataLoader from a raw (transform=None) test dataset.

    raw_dataset: torchvision Dataset (train=False, transform=None)
    args: namespace with .dataset, .img_size, .attack, .patch_mask_path,
          .attack_target, .attack_label_trans, .batch_size, .num_workers
    """
    from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization
    from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate

    clean_transform = transforms.Compose([
        transforms.Resize(tuple(args.img_size[:2])),
        transforms.ToTensor(),
        get_dataset_normalization(args.dataset),
    ])
    _, test_bd_img_transform = bd_attack_img_trans_generate(args)
    bd_label_transform = bd_attack_label_trans_generate(args)

    paired_ds = PairedTestDataset(
        raw_dataset=raw_dataset,                 # raw test dataset without augmentation
        bd_img_transform=test_bd_img_transform,  # add backdoor patch
        bd_label_transform=bd_label_transform,   # all2one attack
        clean_transform=clean_transform,         # transforms.Compose for test images
        exclude_same_label=exclude_same_label,
    )
    return DataLoader(paired_ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)

def evaluate_acc_asr(loader, model, device):
    """
    Returns (acc, asr, ra).

    acc = pred(clean_img) == clean_label
    asr = pred(backdoor_img) == backdoor_label  (trigger → target class)
    ra  = pred(backdoor_img) == clean_label     (recovered true class after defense)
    """
    model.eval()
    model.to(device)
    correct_c = correct_b = correct_ra = 0
    total = 0

    with torch.no_grad():
        for img_c, lbl_c, img_b, lbl_b in loader:
            img_c, lbl_c, img_b, lbl_b = img_c.to(device), lbl_c.to(device), img_b.to(device), lbl_b.to(device)
            pred_c, pred_b = model(img_c).argmax(dim=1), model(img_b).argmax(dim=1)

            correct_c += (pred_c == lbl_c).sum().item()
            correct_b += (pred_b == lbl_b).sum().item()
            correct_ra += (pred_b == lbl_c).sum().item()
            total += lbl_c.size(0)

    return (correct_c / total, correct_b / total, correct_ra / total) if total else (0.0, 0.0, 0.0)

if __name__ == "__main__":
    import argparse
    import numpy as np
    from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
    from utils.aggregate_block.model_trainer_generate import generate_cls_model

    # Add safe globals for loading the model state dict
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar, 
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
        np.dtypes.UInt32DType
    ])

    parser = argparse.ArgumentParser(description="ACC + ASR evaluation (any dataset-model-attack)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_key", default=None,
                        help="Key inside .pt for state_dict (e.g. 'model' for attack_result.pt)")
    parser.add_argument("--attack", default="badnet")
    parser.add_argument("--attack_target", type=int, default=0)
    parser.add_argument("--attack_label_trans", default="all2one")
    parser.add_argument("--patch_mask_path", default="./resource/badnet/trigger_image.png")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    args.num_classes = get_num_classes(args.dataset)
    args.img_size = list(get_input_shape(args.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = generate_cls_model(args.model_name, args.num_classes)
    state = torch.load(args.model_path, map_location="cpu")
    if args.model_key:
        state = state[args.model_key]
    model.load_state_dict(state)

    raw_test = load_raw_test_dataset(args.dataset)
    loader = build_paired_loader(raw_test, args)

    acc, asr, ra = evaluate_acc_asr(loader, model, device)

    print(f"Dataset: {args.dataset}  |  Model: {args.model_name}  |  Attack: {args.attack}")
    print(f"ACC: {100*acc:.2f}%")
    print(f"ASR: {100*asr:.2f}%  (non-target-class samples only)")
    print(f"RA: {100*ra:.2f}%")