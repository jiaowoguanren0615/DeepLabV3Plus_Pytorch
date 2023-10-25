from tqdm import tqdm
import network
import utils
import os
import datetime
import random
import argparse
import numpy as np
from engine import train_one_epoch, evaluate
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.scheduler import create_lr_scheduler
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser('Pytorch Segmentation Models training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./CityScapesDataset',help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,help="num classes (default: None)")
    parser.add_argument("--batch_size", type=int, default=16,help='batch size (default: 16)') # consume approximately 12.8G GPU-Memory (use_amp is False)
    parser.add_argument("--val_batch_size", type=int, default=4,help='batch size for validation (default: 4)')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--use_amp", type=bool, default=False, help='auto mixture precision') # There may be some problems when loading weights, such as: ComplexFloat
    parser.add_argument("--epochs", type=int, default=5, help='total training epochs')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default='', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=True)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_freq", type=int, default=30,
                        help="print frequency (default: 30)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst



def main(opts):
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19


    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id # CUDA: 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=0)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)


    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Create a learning rate update policy, here it is updated once per step (not per epoch)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), opts.epochs, warmup=True)
    
    lr_scheduler = scheduler
    scalar = torch.cuda.amp.GradScaler() if (torch.cuda.is_bf16_supported() and opts.use_amp) else None

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


    def save_ckpt(path):
        checkpoint_save = {
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "best_score": best_score,
        }

        if opts.use_amp:
            checkpoint_save['scalar'] =scalar.state_dict()
        torch.save(checkpoint_save, path)
        print("Model saved as %s" % path)


    if not os.path.exists('checkpoints'):
        utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        model.to(device)
        if opts.continue_training:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            print(f'The Best MeanIou is {best_score:.4f}')
            print("Training state restored from %s" % opts.ckpt)

            if opts.use_amp:
                scalar.load_state_dict(checkpoint['scalar'])
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = evaluate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    print(opts)

    # ==========   Train Loop   ==========#
    for epoch in range(opts.epochs):
        train_score, mean_loss, lr, cur_itrs = train_one_epoch(opts=opts, model=model, metrics=metrics,
                                          train_loader=train_loader, optimizer=optimizer,
                                          criterion=criterion, device=device, lr_scheduler=lr_scheduler,
                                          cur_itrs=cur_itrs, scalar=scalar, epoch=epoch, print_freq=opts.print_freq)

        val_score, confmat, ret_samples = evaluate(opts=opts, model=model, loader=val_loader, 
                                                   device=device, metrics=metrics, print_freq=opts.print_freq, ret_samples_ids=vis_sample_id)

        lr_scheduler.step()
        
        val_info = str(confmat)
        print(val_info)

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")
        

        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride))

        if vis is not None:  # visualize train/validation score and samples
            vis.vis_scalar("[Train] Overall Acc", cur_itrs, train_score['Overall Acc'])
            vis.vis_scalar("[Train] Mean IoU", cur_itrs, train_score['Mean IoU'])
            vis.vis_table("[Train] Class IoU", train_score['Class IoU'])

            vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
            vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                vis.vis_image('Sample %d' % k, concat_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch Segmentation Models training and evaluation script', parents=[get_argparser()])
    opts = parser.parse_args()
    main(opts)
