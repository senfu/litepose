# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import random

from utils.utils import AverageMeter
from utils.vis import save_debug_images
import torch
import torch.nn.functional as F
from arch_manager import ArchManager


def do_train(cfg, model, lr_scheduler, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False, teacher=None):
    
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output && resize the images here
        images = images.cuda(non_blocking=True)
        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        joints = list(map(lambda x: x.cuda(non_blocking=True), joints))
        if cfg.MODEL.NAME == 'pose_supermobilenet' or cfg.MODEL.NAME == 'pose_superresnet':
            img_size = 256 + random.randint(0, 4) * 64
            oup_size = img_size // 4
            images = F.interpolate(images, size = (img_size, img_size))
            for cnt in range(2):
                heatmaps[cnt] = F.interpolate(heatmaps[cnt], size = (oup_size, oup_size))
                masks[cnt] = F.interpolate(masks[cnt].unsqueeze(1), size = (oup_size, oup_size)).squeeze(1)
                x = torch.trunc((joints[cnt][:, :, :, 0] % 512 * img_size) / 512)
                y = torch.trunc((joints[cnt][:, :, :, 0] // 512 * img_size) / 512)
                joints[cnt][:, :, :, 0] = y * img_size + x 
                oup_size *= 2

        if cfg.MODEL.NAME == 'pose_mobilenet' and teacher is not None:
            img_size = 448
            oup_size = images.shape[-1] // 4
            t_images = F.interpolate(images, size = (img_size, img_size))
            with torch.no_grad():
                t_outputs = teacher(t_images)
            t_heatmaps = []
            for cnt in range(2):
                t_heatmaps.append(F.interpolate(t_outputs[cnt][:, :cfg.DATASET.NUM_JOINTS], size = (oup_size, oup_size)).detach())
                oup_size *= 2
                
        outputs = model(images)

        # loss = loss_factory(outputs, heatmaps, masks)
        heatmaps_losses, push_losses, pull_losses = \
            loss_factory(outputs, heatmaps, masks, joints)

        if teacher is not None:
            t_heatmaps_losses, t_push_losses, t_pull_losses = \
                loss_factory(outputs, t_heatmaps, masks, joints)

        loss = 0
        for idx in range(cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss
                if teacher is not None:
                    t_heatmaps_loss = t_heatmaps_losses[idx].mean(dim=0)
                    loss = loss + t_heatmaps_loss

            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                push_loss_meter[idx].update(
                    push_loss.item(), images.size(0)
                )
                loss = loss + push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                pull_loss_meter[idx].update(
                    pull_loss.item(), images.size(0)
                )
                loss = loss + pull_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # lr_scheduler.step()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(idx),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # for scale_idx in range(len(outputs)):
            #     prefix_scale = prefix + '_output_{}'.format(
            #         cfg.DATASET.OUTPUT_SIZE[scale_idx]
            #     )
            #     save_debug_images(
            #         cfg, images, heatmaps[scale_idx], masks[scale_idx],
            #         outputs[scale_idx], prefix_scale
            #     )


def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
