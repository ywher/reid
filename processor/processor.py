import logging
import numpy as np
import os
import time
import datetime
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
import torch.distributed as dist
from IPython import embed

def frozen_feature_layers(model):
    for name, module in model.named_children():
        if 'base' in name:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def open_all_layers(model):
    for name, module in model.named_children():
        module.train()
        for p in module.parameters():
            p.requires_grad = True

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             local_rank,
             ):
    log_period = cfg.SOLVER.LOG_PERIOD #200
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD #80
    eval_period = cfg.SOLVER.EVAL_PERIOD #10000

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS #80

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device: #'cuda'
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)


    scaler = amp.GradScaler()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, dataset = cfg.DATASETS.NAMES)  # only available for offline validation

    # train
    for epoch in range(1, epochs + 1):
        if epoch < cfg.SOLVER.FREEZE_EPOCH: #cfg.SOLVER.FREEZE_EPOCH -1
            logger.info("freeze base layers")
            frozen_feature_layers(model)
        elif epoch == cfg.SOLVER.FREEZE_EPOCH:
            logger.info("open all layers")
            open_all_layers(model)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        
        model.train()
        iter_start_time = time.time()
        for n_iter, (img, vid, target_cam) in enumerate(train_loader):
            #img: [64,3,384,384], 8 identities, 8 images for each identity
            #vid: [64], vehicle ID for each image
            #target_cam: [64], camera ID for each image
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.SOLVER.FP16_ENABLED: #True
                #### FP16 training
                with amp.autocast(enabled=True):
                    score, feat = model(img, target, cam_label=target_cam) #score [64,1802], feat [64 2048]
                    loss = loss_fn(score, feat, target, target_cam) #softmax loss + triplet loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                score, feat = model(img, target, cam_label=target_cam)
                loss = loss_fn(score, feat, target, target_cam)
                loss.backward()
                optimizer.step()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE: #false
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                if cfg.SOLVER.FP16_ENABLED:
                    scaler.step(optimizer_center)
                    scaler.update()
                else:
                    optimizer_center.step()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else: # 
                acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0: #200
                iter_end_t = time.time()
                eta_iter = (iter_end_t - iter_start_time) / (n_iter + 1) * (len(train_loader) - (n_iter + 1))
                eta_iter = datetime.timedelta(seconds=int(eta_iter))
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, eta_iter: {}"
                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr, str(eta_iter)))
        scheduler.step(epoch)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        eta_epoch = (end_time - start_time) * (epochs - (epoch + 1))
        eta_epoch = datetime.timedelta(seconds=int(eta_epoch))
        logger.info("eta epoch {}".format(str(eta_epoch)))

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                   .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.module.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN: #FALSE
                if dist.get_rank() == 0:
                    model.eval()
                    torch.cuda.empty_cache()
                    for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = torch.tensor(camid, dtype=torch.int64)
                            camids = camids.to(device)
                            feat = model(img, cam_label=camids)
                            evaluator.update((feat.clone(), vid, camid, trackids))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info(cfg.OUTPUT_DIR)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for n_iter, (img, vid, camid, trackids, _) in enumerate(val_loader):
                        img = img.to(device)
                        camids = torch.tensor(camid, dtype=torch.int64)
                        camids = camids.to(device)
                        feat = model(img, cam_label=camids)
                        evaluator.update((feat.clone(), vid, camid, trackids))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info(cfg.OUTPUT_DIR)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING, dataset = cfg.DATASETS.NAMES, reranking_track=cfg.TEST.RE_RANKING_TRACK)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                       reranking=cfg.TEST.RE_RANKING,reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    data_start_time = time.time()
    data_time_list = []
    net_time_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            date_time = time.time() - data_start_time
            net_start_time = time.time()
            if cfg.TEST.FLIP_FEATS == 'on':
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        f1 = model(img)
                    else:
                        f2 = model(img)
                feat = f2 + f1 #[128,2048]
            else:
                feat = model(img)
            net_time = time.time() - net_start_time
            data_time_list.append(date_time)
            net_time_list.append(net_time)
            logger.info('batch {}/{}:  data time {:.3f}, net time {:.3f}'.format(n_iter+1, len(val_loader), date_time, net_time))
            evaluator.update((feat.clone(), pid, camid, trackid))
            img_path_list.extend(imgpath)
            data_start_time = time.time()
    data_time_list = np.array(data_time_list)
    net_time_list = np.array(net_time_list)
    logger.info('average data time: {:.3f}, average net time: {:.3f}'.format(np.mean(data_time_list), np.mean(net_time_list)))
    logger.info('average data time: {:.3f}, average net time: {:.3f} after 10 batches'.format(np.mean(data_time_list[10:]), np.mean(net_time_list[10:])))


    _, _, distmat, pids, camids, qf, gf = evaluator.compute(fic=cfg.TEST.FIC, fac=cfg.TEST.FAC, rm_camera=cfg.TEST.RM_CAMERA,save_dir=cfg.OUTPUT_DIR, crop_test = cfg.TEST.CROP_TEST, la= cfg.TEST.LA)
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)
    logger.info('writing result to {}'.format(cfg.OUTPUT_DIR))
