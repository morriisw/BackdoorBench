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


def load_raw_dataset(dataset_name, split="test", data_root="./data"):
    """
    transform=None raw Dataset for either 'train' or 'test' split.
    Dataset-agnostic (mirrors load_raw_test_dataset across both splits).
    """
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST
    name = dataset_name.lower()
    is_train = (split == "train")
    if name == "cifar10":
        return CIFAR10(root=f"{data_root}/cifar10", train=is_train, transform=None, download=True)
    elif name == "cifar100":
        return CIFAR100(root=f"{data_root}/cifar100", train=is_train, transform=None, download=True)
    elif name == "mnist":
        return MNIST(root=f"{data_root}/mnist", train=is_train, transform=None, download=True)
    elif name == "gtsrb":
        from dataset.GTSRB import GTSRB
        return GTSRB(root=f"{data_root}/gtsrb", train=is_train)
    elif name == "tiny":
        from dataset.Tiny import TinyImageNet
        return TinyImageNet(root=f"{data_root}/tiny",
                            split='train' if is_train else 'val', download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

def build_paired_loader(raw_dataset, args, exclude_same_label=True):
    """
    Constructs the 4-output DataLoader from a raw (transform=None) test dataset.

    raw_dataset: torchvision Dataset (train=False, transform=None)
    args: namespace with .dataset, .img_size, .attack, .patch_mask_path,
          .attack_target, .attack_label_trans, .batch_size, .num_workers

    Only works for attacks whose trigger can be expressed as a static
    PIL-level image transform (e.g. BadNet's fixed patch). For InputAware,
    whose trigger requires running trained netG/netM networks on each
    image, use build_paired_loader_inputaware instead.
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


def build_paired_loader_inputaware(raw_dataset, args, netG_netM_path, device="cuda", exclude_same_label=True):
    """
    InputAware-specific paired loader. bd_attack_img_trans_generate cannot
    represent InputAware's trigger, since it is a per-image mask and
    pattern produced by trained netG/netM networks, not a static PIL
    transform. This bypasses that function entirely and reconstructs the
    triggered image at the tensor level, replicating
    InputAware.create_bd's exact formula:
        patterns = normalizer(netG(x))
        mask = threshold(netM(x))
        x_bd = x + (patterns - x) * mask

    netG_netM_path: path to the checkpoint saved during training
    (InputAware/DeiTInputAware's periodic frequency_save writes
    "netCGM.pt" containing netG and netM state dicts). Make sure this
    reflects your FINAL trained weights, not an earlier checkpoint: since
    saving is only periodic (every args.frequency_save epochs), if the
    last training epoch did not land on a save point, this checkpoint
    will be from an earlier, weaker epoch. Re-run training with
    --frequency_save 1 if you need to guarantee the final epoch is saved.

    args needs the same fields InputAwareGenerator expects (.dataset,
    .input_channel), plus the usual .img_size, .attack_target,
    .batch_size, .num_workers.
    """
    from attack.inputaware import InputAwareGenerator, Threshold, Normalize
    from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization

    ckpt = torch.load(netG_netM_path, map_location="cpu", weights_only=False)
    netG_state = ckpt["netG"] if "netG" in ckpt else ckpt
    netM_state = ckpt["netM"] if "netM" in ckpt else ckpt

    netG = InputAwareGenerator(args)
    netG.load_state_dict(netG_state)
    netG.eval().to(device)

    netM = InputAwareGenerator(args, out_channels=1)
    netM.load_state_dict(netM_state)
    netM.eval().to(device)

    threshold = Threshold().to(device)
    norm_stats = get_dataset_normalization(args.dataset)
    normalizer = Normalize(args, norm_stats.mean, norm_stats.std)

    clean_transform = transforms.Compose([
        transforms.Resize(tuple(args.img_size[:2])),
        transforms.ToTensor(),
        get_dataset_normalization(args.dataset),
    ])

    class _CleanOnlyDataset(Dataset):
        """
        Same exclude_same_label filtering as PairedTestDataset, but only
        produces the clean side; the triggered side is built per-batch
        below, not per-sample, since netG/netM need batched tensor input.
        """
        def __init__(self, raw_dataset, clean_transform, attack_target, exclude_same_label=True):
            self.raw_dataset = raw_dataset
            self.clean_transform = clean_transform
            if exclude_same_label:
                self.valid_indices = [i for i in range(len(raw_dataset)) if raw_dataset[i][1] != attack_target]
            else:
                self.valid_indices = None

        def __len__(self):
            return len(self.valid_indices) if self.valid_indices is not None else len(self.raw_dataset)

        def __getitem__(self, index):
            if self.valid_indices is not None:
                index = self.valid_indices[index]
            img_raw, label_clean = self.raw_dataset[index]
            return self.clean_transform(img_raw), label_clean

    clean_ds = _CleanOnlyDataset(raw_dataset, clean_transform, args.attack_target, exclude_same_label)
    base_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class _InputAwarePairedLoader:
        """Wraps base_loader, generating the triggered image for each batch on the fly via netG/netM, matching PairedTestDataset's 4-output interface."""
        def __init__(self, base_loader, attack_target):
            self.base_loader = base_loader
            self.batch_size = base_loader.batch_size
            self.attack_target = attack_target

        def __iter__(self):
            with torch.no_grad():
                for img_c, lbl_c in self.base_loader:
                    img_c_dev = img_c.to(device)
                    patterns = normalizer(netG(img_c_dev))
                    masks = threshold(netM(img_c_dev))
                    img_b = img_c_dev + (patterns - img_c_dev) * masks
                    lbl_b = torch.full_like(lbl_c, self.attack_target)
                    yield img_c, lbl_c, img_b.cpu(), lbl_b

        def __len__(self):
            return len(self.base_loader)

    return _InputAwarePairedLoader(base_loader, args.attack_target)


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


def diagnose_dataset_overview(raw_train=None, raw_test=None, paired_loader=None, poison_index=None, pratio=None):
    """
    Print dataset overview: sizes, per-class distribution, paired-loader
    subsetting, and clean-vs-poisoned counts. Works for any (img,label) Dataset.
    """
    import numpy as np
    from collections import Counter
    from utils.bd_dataset_v2 import get_labels

    print("----------------")
    print("DATASET OVERVIEW")
    print("----------------")

    for split, ds in (("Training", raw_train), ("Test", raw_test)):
        if ds is not None:
            labels = get_labels(ds)
            dist = Counter(labels)
            print(f"{split} set size : {len(labels)}")
            print(f"  per-class     : {dict(sorted(dist.items()))}")

    if paired_loader is not None:
        total = len(raw_test) if raw_test is not None else None
        valid = len(paired_loader.dataset) if hasattr(paired_loader, "dataset") else len(paired_loader.base_loader.dataset)
        print("Analysis paired loader")
        print(f"  raw test      : {total}")
        print(f"  valid (cl+bd) : {valid}")
        if total is not None:
            print(f"  dropped target: {total - valid}")
        print(f"  batch_size    : {paired_loader.batch_size}")
        from torch.utils.data.sampler import RandomSampler
        sampler = getattr(paired_loader, "sampler", None) or getattr(getattr(paired_loader, "base_loader", None), "sampler", None)
        print(f"  shuffle       : {isinstance(sampler, RandomSampler)}")
        print(f"  n_batches     : {len(paired_loader)}")

    if poison_index is not None:
        poison_index = np.asarray(poison_index)
        print(f"Poisoned (exact, from saved index) : {int(poison_index.sum())} / {len(poison_index)}")
        if raw_train is not None and len(raw_train) == len(poison_index):
            labels = get_labels(raw_train)
            idx = np.where(poison_index == 1)[0]
            pdist = Counter(labels[i] for i in idx)
            print(f"  per-class poison : {dict(sorted(pdist.items()))}")
    elif pratio is not None and raw_train is not None:
        print(f"Poisoned (from pratio) : ~{round(pratio * len(raw_train))} / {len(raw_train)}")


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
    parser.add_argument("--netG_netM_path", default=None,
                        help="Required when --attack inputaware: path to the saved netCGM.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    args.num_classes = get_num_classes(args.dataset)
    args.img_size = list(get_input_shape(args.dataset))
    args.input_channel = args.img_size[2]  # needed by InputAwareGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = generate_cls_model(args.model_name, args.num_classes)
    state = torch.load(args.model_path, map_location="cpu")
    if args.model_key:
        state = state[args.model_key]
    model.load_state_dict(state)

    raw_test = load_raw_test_dataset(args.dataset)

    if args.attack == "inputaware":
        if args.netG_netM_path is None:
            raise ValueError("--netG_netM_path is required when --attack inputaware "
                              "(bd_attack_img_trans_generate cannot build InputAware's trigger, "
                              "see build_paired_loader_inputaware)")
        loader = build_paired_loader_inputaware(raw_test, args, args.netG_netM_path, device=str(device))
    else:
        loader = build_paired_loader(raw_test, args)

    acc, asr, ra = evaluate_acc_asr(loader, model, device)

    print(f"Dataset: {args.dataset}  |  Model: {args.model_name}  |  Attack: {args.attack}")
    print(f"ACC: {100*acc:.2f}%")
    print(f"ASR: {100*asr:.2f}%  (non-target-class samples only)")
    print(f"RA: {100*ra:.2f}%")