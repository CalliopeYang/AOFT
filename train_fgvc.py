import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from model.ViT_unitary_adapter_ffn import CONFIGS, VisionTransformer
from AOFTDatasets.FGVCConfig import DATA_CONFIGS
from AOFTDatasets.FGVCDataLoader import construct_test_loader, construct_trainval_loader
from utils import (seed_torch, accuracy, AverageMeter, Logger, count_parameters)
from timm.scheduler import create_scheduler
from torch.cuda.amp import autocast
from timm.utils import NativeScaler
from timm.models import model_parameters
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="parameter-efficient fine-tuning")
    parser.add_argument("--dataset_name", default="CUB_200_2011")
    parser.add_argument("--model_type", default="ViT-B_16")
    parser.add_argument("--dataset_dir", default="/home/datasets/FGVC")
    parser.add_argument("--pretrained_dir", type=str, default="/home/checkpoint/ViT-B_16.npz") #imagenet21k_
    parser.add_argument("--output_dir", default="fgvc_output", type=str) #-aug all-no-res
    parser.add_argument("--device", default='cuda', type=str)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=3e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)

    # fellow SSF
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--sched", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--lr_cycle_decay", default=0.5, type=float)
    parser.add_argument("--cooldown_epochs", default=10, type=int)

    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args

def frozen_param(model, frozen_list=('',)):
    for name, param in model.named_parameters():
        if any(item in name for item in frozen_list):
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False
    num_params = count_parameters(model)
    print("Training parameters %s", args)
    print("Total Parameter: \t%2.3fM" % num_params)


def save_model(max_acc, acc, model, path):
    if acc > max_acc:
        from collections import OrderedDict
        model_dict = OrderedDict()
        save_index = ['head', 'scale', 'shift', '_left', '_right', '_lambda']
        for k, v in model.state_dict().items():
            if any(item in k for item in save_index):
                model_dict[k] = v
        if os.path.exists(path + '{:6.2f}'.format(max_acc) + '.pth'):
            os.remove(path + '{:6.2f}'.format(max_acc) + '.pth')
        torch.save(model_dict, path + '{:6.2f}'.format(acc) + '.pth')
        return acc
    return max_acc

def setup(args, frozen_list=('',)):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes, drop_path=args.drop_path)
    model.load_from(np.load(args.pretrained_dir))

    frozen_param(model, frozen_list)
    return model


@torch.no_grad()
def valid(model, test_loader, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, label) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        with autocast():
            output = model(x)

        loss = criterion(output, label)
        acc1 = accuracy(output, label, topk=(1,))
        top1.update(acc1[0].item(), x.size(0))
        losses.update(loss.item(), x.size(0))
    print('Test :', losses, top1)
    return top1.avg, losses.avg

def train(model, train_loader, criterion, optimizer, loss_scaler, lr_scheduler, epoch):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.long().to(device)
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        optimizer.zero_grad()
        # fellow SSF
        loss_scaler(loss, optimizer, parameters=model_parameters(model))
        # loss.backward()
        # optimizer.step()

        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))

    if lr_scheduler is not None:
        lr_scheduler.step_update(num_updates=epoch, metric=losses.avg)
        
    print('Train :', losses, top1)
    return top1.avg, losses.avg


def main(args):
    config = DATA_CONFIGS[args.dataset_name]
    args.num_classes = config['num_classes']
    args.learning_rate = config['lr']
    args.min_lr = config['min_lr']
    args.drop_path = config['drop_path']
    args.warmup_lr = config['warmup_lr']
    args.weight_decay = config['weight_decay']
    args.batch_size = config['batch_size']
    if not os.path.exists(os.path.join(args.output_dir, args.dataset_name)):
        os.makedirs(os.path.join(args.output_dir, args.dataset_name))
    else:
        import shutil
        shutil.rmtree(os.path.join(args.output_dir, args.dataset_name))
        os.makedirs(os.path.join(args.output_dir, args.dataset_name))

    sys.stdout = Logger(sys.stdout, os.path.join(args.output_dir, args.dataset_name, '{}.txt').format(args.dataset_name))

    train_loader = construct_trainval_loader(args)
    test_loader = construct_test_loader(args)

    model = setup(args, ['head', 'scale', 'shift', '_left', '_right', '_lambda'])
    model.to(device)

    max_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    optimizer = torch.optim.AdamW(model.get_parameters(lr=args.learning_rate, weight_decay=args.weight_decay)) #
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if args.dataset_name in ['CUB_200_2011', 'StanfordDogs']:
        t = 2 if args.dataset_name == 'NABirds' else 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5*t, T_mult=1, eta_min=0, last_epoch=-1)

    for epoch in range(0, num_epochs):
        lr_scheduler_ = None if args.dataset_name in ['CUB_200_2011', 'StanfordDogs'] else lr_scheduler
        train(model, train_loader, criterion, optimizer, loss_scaler, lr_scheduler_, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if epoch % 2 == 0:
            acc, _ = valid(model, test_loader, device)
            max_acc = save_model(max_acc, acc, model, os.path.join(args.output_dir, args.dataset_name, args.dataset_name))
            print('Epoch {}, cur_max_acc : {}'.format(epoch, max_acc))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"cur lr: {current_lr}")
        
    acc, _ = valid(model, test_loader, device)
    print("lr:" + str(args.learning_rate) + " wd:" + str(args.weight_decay) + " max_acc:" + str(max_acc))



if __name__ == '__main__':
    args = get_args_parser()
    seed_torch(args.seed)
    device = torch.device(args.device)
    main(args)
