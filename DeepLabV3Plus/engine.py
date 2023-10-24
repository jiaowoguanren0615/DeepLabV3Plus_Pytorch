# import utils
import os
import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import utils.distributed_utils as utils


def train_one_epoch(opts, model, metrics, train_loader, optimizer, criterion,
                    device, lr_scheduler, cur_itrs, scalar, epoch, print_freq=10):
    model.train()
    metrics.reset()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    use_amp = True if (opts.use_amp and scalar is not None) else False

    for images, labels in metric_logger.log_every(train_loader, print_freq, header):

        cur_itrs += 1

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(enabled=scalar is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()

        metric_logger.update(loss=loss.item(), lr=lr)
        
        metrics.update(targets, preds)
        # np_loss = loss.detach().cpu().numpy()
        # interval_loss += np_loss
        #
        # if vis is not None:
        #     vis.vis_scalar('Loss', cur_itrs, np_loss)
        #
        # if (cur_itrs) % 10 == 0:
        #     interval_loss = interval_loss / 10
        #     print("Epoch %d, Itrs %d/%d, Loss=%f" %
        #           (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
        #     interval_loss = 0.0

    score = metrics.get_results()

    return score, metric_logger.meters["loss"].global_avg, lr, cur_itrs



@torch.no_grad()
def evaluate(opts, model, loader, device, metrics, print_freq=10, ret_samples_ids=None):
    """Do validation and return specified samples"""
    
    metrics.reset()
    ret_samples = []
    model.eval()

    confmat = utils.ConfusionMatrix(opts.num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    for i, (images, labels) in enumerate(metric_logger.log_every(loader, print_freq, header)):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(images)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()

        metrics.update(targets, preds)
        confmat.update(labels.flatten(), outputs.argmax(1).flatten())
        
        if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
            ret_samples.append(
                (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        if opts.save_val_results:
            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save('results/%d_image.png' % img_id)
                Image.fromarray(target).save('results/%d_target.png' % img_id)
                Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1
                
    confmat.reduce_from_all_processes()
    score = metrics.get_results()
    return score, confmat, ret_samples