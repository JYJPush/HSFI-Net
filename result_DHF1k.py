import os
import cv2
import copy
import argparse
import numpy as np
from PIL import Image
import time
from utils import *
from models.TSYNet9 import TSYNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def validate(args):
    path_indata = os.path.join(args.path_indata, args.model)
    save_pre_path = args.save_pre_path

    model = TSYNet(args.temporal, args.image_height, args.image_width).to(device)
    print('Load pretrained model')

    checkpoint = torch.load(args.file_weight)
    state_dict = checkpoint['model']
    # all_keys = [k for k in state_dict.keys()]
    # for k in all_keys:
    #     if "module." in k:
    #         state_dict[k[7:]] = state_dict.pop(k)
    # full_dict = copy.deepcopy(state_dict)
    model.load_state_dict(state_dict)
    torch.backends.cuda.benchmark = False
    model.eval()

    list_indata = os.listdir(path_indata)
    list_indata.sort()

    if args.start_idx != -1:
        if args.end_idx > args.start_idx and (args.end_idx - args.start_idx) <= len(list_indata):
            list_indata = list_indata[args.start_idx: args.end_idx]

    loss = AverageMeter()  # 对所有损失取平均 实例化对象
    kldiv_loss =AverageMeter()
    cc_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_j_loss = AverageMeter()
    for name in list_indata:
        print('processing:', name)
        if not os.path.isdir(os.path.join(save_pre_path, name)) and args.save:
            os.mkdir(os.path.join(save_pre_path, name))

        path_image = os.path.join(path_indata, name, 'images')
        path_gt = os.path.join(path_indata, name, 'maps')
        path_fixmao = os.path.join(path_indata, name, 'fixation')

        list_frames = os.listdir(path_image)
        list_frames.sort()
        temp = [list_frames[i] for i in range(args.temporal-1, 0, -1)]
        # temp = [list_frames[0] for _ in range(args.temporal - 1, 0, -1)]
        temp.extend(list_frames)
        list_frames = copy.deepcopy(temp)

        clip_image = []
        for i in range(len(list_frames)):
            image, img_size = torch_transform(os.path.join(path_image, list_frames[i]), (args.image_height, args.image_width))
            clip_image.append(image)
            if i >= args.temporal - 1:
                prep_image = torch.FloatTensor(torch.stack(clip_image, dim=0)).unsqueeze(0)
                prep_image = prep_image.permute(0, 2, 1, 3, 4)
                prep_image = prep_image.to(device)
                with torch.no_grad():
                    pred_map = model(prep_image)

                pred_map = pred_map.cpu().squeeze(0).detach().numpy()
                pred_map = cv2.resize(pred_map, (img_size[0], img_size[1]))
                pred_map = blur(pred_map)

                save_predmap = pred_map.cpu().detach().numpy()
                if args.save:
                    save_predmap = save_predmap * 255
                    save_path = os.path.join(save_pre_path, name, list_frames[i])
                    cv2.imwrite(save_path, save_predmap)

                gt = np.array(Image.open(os.path.join(path_gt, list_frames[i])).convert('L'))
                gt = gt.astype('float')
                if np.max(gt) > 1.0:
                    gt = gt / 255.0
                labels = torch.FloatTensor(gt)
                labels = labels.to(device)

                fixations = np.array(Image.open(os.path.join(path_fixmao, list_frames[i])).convert('L'))
                fixations = fixations.astype('float')
                fixations = (fixations > 0.5).astype('float')
                fixations = torch.FloatTensor(fixations)
                fixations = fixations.to(device)

                pred_map = pred_map.to(device)
                pred_map = pred_map.unsqueeze(0)
                labels = labels.unsqueeze(0)
                fixations = fixations.unsqueeze(0)

                cc_loss.update(cc(pred_map, labels))
                kldiv_loss.update(kldiv(pred_map, labels))
                nss_loss.update(nss(pred_map, fixations))
                sim_loss.update(similarity(pred_map, labels))
                auc_j_loss.update(auc_judd(pred_map, fixations))

                del clip_image[0]

        print('finsh work image : %s' % name)
        print('CC : {:.4f}, KL : {:.3f}, NSS : {:.4f}, SIM : {:.4f}, AUC_J : {:.4f}'.format(cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_j_loss.avg))


def torch_transform(path, size):
    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight', default='./PTH/TSYNet9173.pth', type=str)

    parser.add_argument('--save', default=False, type=bool)  # 是否要储存生成的图片
    parser.add_argument('--save_pre_path', default='./result', type=str)

    # parser.add_argument('--path_indata', default='/home/wusonghe/dataset/DHF1K', type=str)
    parser.add_argument('--path_indata', default='/home/jinyingjie/dataset/DHF1K', type=str)
    parser.add_argument('--model', default='val', type=str)     # 选择要生成的预测图片
    parser.add_argument("--image_width", default=384, type=int)
    parser.add_argument("--image_height",  default=224, type=int)
    parser.add_argument("--temporal", default=32, type=int)

    # 测试第一个
    # parser.add_argument('--start_idx', default=1, type=int)  # 选择要预测区间
    # parser.add_argument('--end_idx', default=2, type=int)

    parser.add_argument('--start_idx', default=-1, type=int)  # 选择要预测区间
    parser.add_argument('--end_idx', default=-1, type=int)
    args = parser.parse_args()

    validate(args)