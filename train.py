import os
import csv
import cv2
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import DHF1KDataset
from models.TSYNet import TSYNet
from utils import *


def train(epochs, epoch, model, train_loader, optimizer, args, writer):
    pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]" % (epoch, epochs), unit=" step")
    model.train()
    total_loss = AverageMeter()
    for batch in train_loader:
        image, label = batch
        image = image.permute(0, 2, 1, 3, 4)
        label = label
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred_map = model(image)

        assert pred_map.size() == label.size()
        loss = get_loss(pred_map, label, args)
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        pbar.update()
        pbar.set_postfix(train_loss=f"{total_loss.sum:.4f}", )

    writer.add_scalar("train_loss", total_loss.avg, epoch)
    pbar.close()
    return total_loss.avg


def valid(epochs, epoch, model, valid_loader, args, writer):
    pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]" % (epoch, epochs), unit=" step")
    model.eval()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    total_kl_loss = AverageMeter()
    with torch.no_grad():
        for batch in valid_loader:
            image, label = batch

            image = image.permute(0, 2, 1, 3, 4)
            image = image.to(device)
            pred_map = model(image)
            label = label.squeeze(0).numpy()
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (label.shape[1], label.shape[0]))
            pred_map = blur(pred_map).unsqueeze(0).cuda()
            label = torch.FloatTensor(label).unsqueeze(0).cuda()

            assert pred_map.size() == label.size()
            loss = get_loss(pred_map, label, args)
            cc_loss = cc(pred_map, label)
            sim_loss = similarity(pred_map, label)
            kl_loss = kldiv(pred_map, label)

            total_loss.update(loss.item())
            total_cc_loss.update(cc_loss.item())
            total_sim_loss.update(sim_loss.item())
            total_kl_loss.update(kl_loss.item())

            pbar.update()
            pbar.set_postfix(valid_loss=f"{total_loss.sum:.4f}", )
        writer.add_scalar('CC', total_cc_loss.avg, global_step=epoch)  # item后不用data
        writer.add_scalar('SIM', total_sim_loss.avg, global_step=epoch)
        writer.add_scalar('Loss', total_loss.avg, global_step=epoch)

    pbar.close()
    return total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, total_kl_loss.avg


def main(args, model, train_loader, valid_loader, optimizer, scheduler):
    strloss = 'loss = '
    if args.kldiv:
        strloss = strloss + str(args.kldiv_coeff) + ' KL  '
    if args.cc:
        strloss = strloss + str(args.cc_coeff) +' cc  '
    if args.sim:
        strloss = strloss + str(args.sim_coeff) + ' sim  '
    print(strloss, args.lr)

    writer = SummaryWriter("valcurve")
    min_cc = 0.550
    print('start tarining...')
    for epoch in range(args.n_epochs):
        train_loss = train(args.n_epochs, epoch, model, train_loader, optimizer, args, writer)
        valid_loss, valid_cc, valid_sim, valid_kl = valid(args.n_epochs, epoch, model, valid_loader, args, writer)
        print('epoch: {}  train_loss: {:.3f}  valid_loss: {:.3f}  cc: {:.3f}  sim: {:.3f}  kl: {:.3f} \n'.format(
            epoch + 1, train_loss, valid_loss, valid_cc, valid_sim, valid_kl))

        if valid_cc >= min_cc:
            min_cc = valid_cc
            save_name = './PTH/{}.pth'.format('TSYNet6' + str(epoch))
            torch.save({'model': model.state_dict()}, save_name)

        # scheduler.step(valid_loss)
    writer.close()


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='DHF1k')
    parser.add_argument("--root", type=str, default='/home/jinyingjie/dataset/DHF1K')
    parser.add_argument("--S3D_pretrained", type=str, default='/home/jinyingjie/S3D_kinetics400.pt')
    parser.add_argument("--VST_pretrained", type=str, default='/home/jinyingjie/swin_small_patch244_window877_kinetics400_1k.pth')
    parser.add_argument("--load", type=str, default='./PTH/CC3Net125.pth')

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--temporal", type=int, default=32)

    parser.add_argument('--kldiv', default=True, type=bool)
    parser.add_argument('--cc', default=True, type=bool)
    parser.add_argument('--nss', default=False, type=bool)
    parser.add_argument('--sim', default=False, type=bool)
    parser.add_argument('--l1', default=False, type=bool)

    parser.add_argument('--kldiv_coeff', default=1.0, type=float)
    parser.add_argument('--cc_coeff', default=-1, type=float)
    parser.add_argument('--sim_coeff', default=-0.1, type=float)
    parser.add_argument('--nss_coeff', default=-1.0, type=float)
    parser.add_argument('--l1_coeff', default=1.0, type=float)
    args = parser.parse_args()

    train_data = DHF1KDataset(parameter=args, model="train")
    valid_data = DHF1KDataset(parameter=args, model="val")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.workers)

    model = TSYNet(args.temporal, args.image_height, args.image_width, pretrained=args.VST_pretrained).to(device)

    # if os.path.isfile(args.S3D_pretrained):
    #     print('loading weight file')
    #     weight_dict = torch.load(args.S3D_pretrained)
    #     model_dict = model.backbone.state_dict()
    #     for name, param in weight_dict.items():
    #         if 'module' in name:
    #             name = '.'.join(name.split('.')[1:])
    #         if 'base.' in name:
    #             bn = int(name.split('.')[1])
    #             sn_list = [0, 5, 8, 14]
    #             sn = sn_list[0]
    #             if bn >= sn_list[1] and bn < sn_list[2]:
    #                 sn = sn_list[1]
    #             elif bn >= sn_list[2] and bn < sn_list[3]:
    #                 sn = sn_list[2]
    #             elif bn >= sn_list[3]:
    #                 sn = sn_list[3]
    #             name = '.'.join(name.split('.')[2:])
    #             name = 'base%d.%d.' % (sn_list.index(sn) + 1, bn - sn) + name
    #         if name in model_dict:
    #             if param.size() == model_dict[name].size():
    #                 model_dict[name].copy_(param)
    #             else:
    #                 print(' size? ' + name, param.size(), model_dict[name].size())

    # if os.path.isfile(args.load):
    #     print("load load")
    #     checkpoint = torch.load(args.load)
    #     state_dict = checkpoint['model']
    #     # all_keys = [k for k in state_dict.keys()]
    #     # for k in all_keys:
    #     #     if "module." in k:
    #     #         state_dict[k[7:]] = state_dict.pop(k)
    #     # full_dict = copy.deepcopy(state_dict)
    #     # model.load_state_dict(full_dict)
    #     model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, verbose=True)

    main(args, model, train_loader, valid_loader, optimizer, scheduler)