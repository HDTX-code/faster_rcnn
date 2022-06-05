import argparse
import os
import datetime

import pandas as pd
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from nets.frcnn_training import get_lr_scheduler, FasterRCNNTrainer, set_optimizer_lr
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes, show_config, get_model
from utils.utils_fit import fit_one_epoch


def go_train(args):
    # 保存文件夹
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))

    # 生成dataset
    with open(args.train_txt_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_txt_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    train_dataset = FRCNNDataset(train_lines, [args.h, args.w], train=True)
    val_dataset = FRCNNDataset(val_lines, [args.h, args.w], train=False)

    # 打印训练参数
    show_config(
        classes_path=args.classes_path,
        model_path=args.model_path,
        input_shape=[args.h, args.w],
        Init_Epoch=args.Init_Epoch,
        Freeze_Epoch=args.Freeze_Epoch,
        UnFreeze_Epoch=args.UnFreeze_Epoch,
        Freeze_batch_size=args.Freeze_batch_size,
        Unfreeze_batch_size=args.Unfreeze_batch_size,
        Freeze_Train=True if args.UnFreeze_Epoch != 0 else False,
        Init_lr=args.Init_lr,
        Min_lr=args.Init_lr * 0.01,
        optimizer_type=args.optimizer_type,
        momentum=args.momentum,
        lr_decay_type=args.lr_decay_type,
        save_period=args.save_period,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        num_train=num_train,
        num_val=num_val,
        fp16=args.fp16,
        pretrained=args.pretrained,
        anchors_size=args.anchors_size,
        eval_flag=args.eval_flag,
        eval_period=args.eval_period
    )

    # 训练设备
    Cuda = torch.cuda.is_available()

    # 检查保存文件夹是否存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 获取classes和anchor
    class_names, num_classes = get_classes(args.classes_path)

    # 加载模型
    model = get_model(args.backbone, args.model_path, args.anchors_size, num_classes, args.pretrained).train()

    # 生成loss_history
    loss_history = LossHistory(args.save_dir, model, input_shape=[args.h, args.w])

    # ---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    # ----------------------------------------------------------#
    wanted_step = 5e4 if args.optimizer_type == "sgd" else 1.5e4
    total_step = num_train // args.Unfreeze_batch_size * args.UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // args.Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (args.optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, args.Unfreeze_batch_size, args.UnFreeze_Epoch, total_step))
        print(
            "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if args.Freeze_Epoch != 0:
            for param in model.extractor.parameters():
                param.requires_grad = False
        # ------------------------------------#
        #   冻结bn层
        # ------------------------------------#
        model.freeze_bn()

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = args.Freeze_batch_size if args.Freeze_Epoch != 0 else args.Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * args.Init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999),
                               weight_decay=args.weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True,
                             weight_decay=args.weight_decay)
        }[args.optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        train_util = FasterRCNNTrainer(model_train, optimizer)
        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        eval_callback = EvalCallback(model_train, [args.h, args.w], class_names, num_classes, val_lines, log_dir,
                                     Cuda, eval_flag=args.eval_flag, period=args.eval_period)

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(args.Init_Epoch, args.UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= args.Freeze_Epoch and not UnFreeze_flag and args.Freeze_Train:
                batch_size = args.Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 5e-2
                lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)

                for param in model.extractor.parameters():
                    param.requires_grad = True
                # ------------------------------------#
                #   冻结bn层
                # ------------------------------------#
                model.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=frcnn_dataset_collate)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, args.UnFreeze_Epoch, Cuda, args.fp16, scaler, args.save_period, args.save_dir)

        loss_history.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--classes_path', type=str, default="./model_data/classes.txt", help='种类数量')
    parser.add_argument('--save_dir', type=str, default="./weights", help='存储文件夹位置')
    parser.add_argument('--save_period', type=int, default=3, help='存储间隔')
    parser.add_argument('--log_dir', type=str, default="./logs", help='存储文件夹位置')
    parser.add_argument('--model_path', type=str, default="", help='模型参数位置')
    parser.add_argument('--w', type=int, default=600, help='宽')
    parser.add_argument('--h', type=int, default=600, help='高')
    parser.add_argument('--train_txt_path', type=str, default="./2007_train.txt", help="训练csv")
    parser.add_argument('--val_txt_path', type=str, default="./2007_val.txt", help="验证csv")
    parser.add_argument('--optimizer_type', type=str, default='adam', help="优化器")
    parser.add_argument('--Freeze_batch_size', type=int, default=18, help="冻结训练batch_size")
    parser.add_argument('--Unfreeze_batch_size', type=int, default=8, help="解冻训练batch_size")
    parser.add_argument('--lr_decay_type', type=str, default='cos', help="使用到的学习率下降方式，可选的有'step','cos'")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--Init_lr', type=float, default=1e-4, help="最大学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help="优化器动量")
    parser.add_argument('--weight_decay', type=float, default=0, help="权值衰减，使用adam时建议为0")
    parser.add_argument('--Freeze_Epoch', type=int, default=50, help="冻结训练轮次")
    parser.add_argument('--UnFreeze_Epoch', type=int, default=8, help="解冻训练轮次")
    parser.add_argument('--Init_Epoch', type=int, default=0, help="开始训练轮次")
    parser.add_argument('--anchors_size', nargs='+', type=float, default=[8, 16, 32],
                        help='用于设定先验框的大小，每个特征点均存在9个先验框。')
    parser.add_argument('--eval_period', type=int, default=5, help="eval_period")
    parser.add_argument('--eval_flag', default=False, action='store_true', help="是否在训练过程中检测")
    parser.add_argument('--pretrained', default=False, action='store_true', help="是否预训练")
    parser.add_argument('--fp16', default=False, action='store_true', help="是否用fp16")
    args = parser.parse_args()

    go_train(args)
