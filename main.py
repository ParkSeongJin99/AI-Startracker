#import Colab version


import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import models
import datasets
from multiscaleloss import multiscaleEPE, realEPE
import datetime
from torch.utils.tensorboard import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint
import numpy as np
import matplotlib.pyplot as plt
import easydict
import time



model_names = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__")
)
dataset_names = sorted(name for name in datasets.__all__)

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE, n_iter
    # 사용할 EasyDict 선언
    # 코드에서 사용할 파서(Parser)정의 - 인수를 자유롭게 변경할 수 있음
    args = easydict.EasyDict({
        "data": "Flyingchairs",
        "dataset": "flying_chairs",
        "split_file": None,
        "split_value": 0.8,
        "split_seed": None,
        "arch": "flownets",
        "solver": "adam",
        "workers": 8,
        "epochs": 300,
        "start_epoch": 0,
        "epoch_size": 1000,
        "batch_size": 8,
        "lr": 0.0001,
        "momentum": 0.9,
        "beta": 0.999,
        "weight_decay": 4e-4,
        "bias_decay": 0,
        "multiscale_weights": [0.005, 0.01, 0.02, 0.08, 0.32],
        "sparse": False,
        "print_freq": 10,
        "evaluate": False,
        "pretrained": None,
        "no_date": False,
        "div_flow": 20,
        "milestones": [100, 150, 200]
    })

    # 저장할 디렉토리 이름 설정 - 파라미터에 따라
    save_path = "{},{},{}epochs{},b{},lr{}".format(
        args.arch,
        args.solver,
        args.epochs,
        ",epochSize" + str(args.epoch_size) if args.epoch_size > 0 else "",
        args.batch_size,
        args.lr,
    )
    # 이름에 시간을 넣을것인지 말것인지
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m_%d-%H_%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print("=> will save everything to {}".format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.split_seed is not None:
        np.random.seed(args.split_seed)

    # 각 결과를 저장할 디렉토리 설정
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    test_writer = SummaryWriter(os.path.join(save_path, "test"))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, "test", str(i))))

    #torchvision의 transform 코드 사용
    # Data loading code
    #여러 transform을 묶어 transform 함수를 정의
    input_transform = transforms.Compose(
        [   #torch의 tensor구조로 바꿔줌
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1]),
        ]
    )

    target_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0], std=[args.div_flow, args.div_flow]),
        ]
    )

    if "KITTI" in args.dataset:
        args.sparse = True
    if args.sparse:
        co_transform = flow_transforms.Compose(
            [
                flow_transforms.RandomCrop((320, 448)),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        co_transform = flow_transforms.Compose(
            [
                flow_transforms.RandomTranslate(10),
                flow_transforms.RandomRotate(10, 5),
                flow_transforms.RandomCrop((320, 448)),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip(),
            ]
        )

  #args.data의 경로 내의 trainset 확인
    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        split=args.split_file if args.split_file else args.split_value,
        split_save_path=os.path.join(save_path, "split.txt"),
    )
    print(
        "{} samples found, {} train samples and {} test samples ".format(
            len(test_set) + len(train_set), len(train_set), len(test_set)
        )
    )
    n_iter = args.start_epoch * len(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    # 모델 불러오기
    #사전 훈련된 모델을 불러온다
    print("11111")
    if args.pretrained:
        print("22222")
        network_data = torch.load(args.pretrained)

        args.arch = network_data["arch"]
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).to(device)

    #설정한 솔버(ADAM)으로 실행
    assert args.solver in ["adam", "sgd"]
    print("=> setting {} solver".format(args.solver))
    param_groups = [
        {"params": model.bias_parameters(), "weight_decay": args.bias_decay},
        {"params": model.weight_parameters(), "weight_decay": args.weight_decay},
    ]

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    if args.solver == "adam":
        optimizer = torch.optim.Adam(
            param_groups, args.lr, betas=(args.momentum, args.beta)
        )
    elif args.solver == "sgd":
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.5
    )

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_EPE = train(
            train_loader, model, optimizer, epoch, train_writer
        )
        scheduler.step()
        train_writer.add_scalar("mean EPE", train_EPE, epoch)

        # evaluate on validation set

        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar("mean EPE", EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.module.state_dict(),
                "best_EPE": best_EPE,
                "div_flow": args.div_flow,
            },
            is_best,
            save_path,
        )


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = (
        len(train_loader)
        if args.epoch_size == 0
        else min(len(train_loader), args.epoch_size)
    )

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)
        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h, w)), *output[1:]]

        loss = multiscaleEPE(
            output, target, weights=args.multiscale_weights, sparse=args.sparse
        )
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar("train_loss", loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}".format(
                    epoch, i, epoch_size, batch_time, data_time, losses, flow2_EPEs
                )
            )
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)
        flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == args.start_epoch:
                mean_values = torch.tensor(
                    [0.45, 0.432, 0.411], dtype=input.dtype
                ).view(3, 1, 1)
                output_writers[i].add_image(
                    "GroundTruth", flow2rgb(args.div_flow * target[0], max_value=10), 0
                )
                output_writers[i].add_image(
                    "Inputs", (input[0, :3].cpu() + mean_values).clamp(0, 1), 0
                )
                output_writers[i].add_image(
                    "Inputs", (input[0, 3:].cpu() + mean_values).clamp(0, 1), 1
                )
            output_writers[i].add_image(
                "FlowNet Outputs",
                flow2rgb(args.div_flow * output[0], max_value=10),
                epoch,
            )

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t Time {2}\t EPE {3}".format(
                    i, len(val_loader), batch_time, flow2_EPEs
                )
            )

    print(" * EPE {:.3f}".format(flow2_EPEs.avg))

    return flow2_EPEs.avg




#메인문 실행

if __name__ == "__main__":
    main()

