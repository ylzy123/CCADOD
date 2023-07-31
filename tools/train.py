import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
import time

import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.apis.test import calculate_random, calculate_entropy, update_by_coreset
from mmdet.apis.test import calculate_uncertainty, calculate_diversity, calculate_mix#, calculate_div
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.datasets import build_dataloader, build_dataset
from tools.utils import losstype
from mmdet.utils.active_datasets import *
from configs._base_.retinanet_r50_fpn import strategy_cfg, select_cfg

from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
import mmcv
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_directory', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_false',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_directory is determined in this priority: CLI > segment in file > filename
    if args.work_directory is not None:
        # update configs according to CLI args if args.work_directory is not None
        cfg.work_directory = args.work_directory
    elif cfg.get('work_directory', None) is None:
        # use config filename as default work_directory if cfg.work_directory is None
        cfg.work_directory = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # create work_directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_directory))
    # dump config
    cfg.dump(osp.join(cfg.work_directory, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_directory, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ---------- MI-AOD Training and Test Start Here ---------- #

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    # X_L：一维numpy，存的初始已标注集图片的编号
    # all_anns：存的voc2007和voc2012的trainval.txt的路径
    X_L, X_U, X_all, all_anns = get_X_L_0(cfg)

    # # load set and model
    # # Please change it to the timestamp directory which you want to load data from.
    # last_timestamp = '/20201013_154728'
    # # Please change it to the cycle which you want to load data from.
    # load_cycle = 0
    # X_L = np.load(cfg.work_directory + last_timestamp +'/X_L_' + str(load_cycle) + '.npy')
    # X_U = np.load(cfg.work_directory + last_timestamp +'/X_U_' + str(load_cycle) + '.npy')
    # cfg.cycles = list(range(load_cycle, 7))

    cfg.work_directory = cfg.work_directory + '/' + timestamp
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_directory))
    np.save(cfg.work_directory + '/X_L_' + '0' + '.npy', X_L)
    np.save(cfg.work_directory + '/X_U_' + '0' + '.npy', X_U)
    initial_step = cfg.lr_config.step
    for cycle in cfg.cycles:
        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed
        # get the config of the labeled dataset
        cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
        # load model
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        # # Please change it to the epoch which you want to load model at.
        # model_file_name = '/latest.pth'
        # model.load_state_dict(torch.load(cfg.work_directory[:16] + last_timestamp + model_file_name)['state_dict'])

        # load dataset
        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None and cycle == 0:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7],
                                              config=cfg.pretty_text, CLASSES=datasets[0].CLASSES)
        model.CLASSES = datasets[0].CLASSES
        for epoch in range(cfg.epoch):
            # Only in the last 3 epoch does the learning rate need to be reduced and the model needs to be evaluated.
            if epoch == cfg.epoch - 1:
                cfg.lr_config.step = initial_step
                cfg.evaluation.interval = cfg.epoch_ratio[0]
            else:
                cfg.lr_config.step = [1000]
                cfg.evaluation.interval = 100

            # ---------- 已标注集上来先训6轮 ----------
            if epoch == 0:
                cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
                datasets = [build_dataset(cfg.data.train)]
                losstype.update_vars(0)
                cfg.total_epochs = cfg.epoch_ratio[0]
                cfg_bak = cfg.deepcopy()
                time.sleep(2)
                # 全部参数打开，fmil由于detach了并且回传loss为0，因此不影响大局
                for name, value in model.named_parameters():
                    value.requires_grad = True
                # 已标注集训练6个epoch
                train_detector(model, datasets, cfg,
                               distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
                cfg = cfg_bak

            # ---------- 已标注集上训2轮，未标注集训2轮 ----------
            cfg_u = create_X_U_file(cfg.deepcopy(), X_U, all_anns, cycle)
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            # loss类型设为2
            losstype.update_vars(2)
            # 总epoch数为1
            cfg_u.total_epochs = cfg_u.epoch_ratio[1]
            cfg.total_epochs = cfg.epoch_ratio[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            # 注意修改传入参数
            # train_detector(model, datasets, cfg,
            #                distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)

            train_detector(model, [datasets, datasets_u], [cfg, cfg_u],
                           distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak

            # ---------- 已标注集上训2轮，未标注集训2轮 ----------
            cfg_u = create_X_U_file(cfg.deepcopy(), X_U, all_anns, cycle)
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets_u = [build_dataset(cfg_u.data.train)]
            datasets = [build_dataset(cfg.data.train)]
            # loss类型均设为2
            losstype.update_vars(2)
            cfg_u.total_epochs = cfg_u.epoch_ratio[1]
            cfg.total_epochs = cfg.epoch_ratio[1]
            cfg_u_bak = cfg_u.deepcopy()
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            # 注意修改传入参数
            # train_detector(model, datasets,cfg,
            #                distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)

            train_detector(model, [datasets, datasets_u], [cfg, cfg_u],
                           distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            cfg_u = cfg_u_bak
            cfg = cfg_bak
            # ---------- 已标注集上训6轮 ----------
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets = [build_dataset(cfg.data.train)]
            losstype.update_vars(0)
            cfg.total_epochs = cfg.epoch_ratio[0]
            cfg_bak = cfg.deepcopy()
            time.sleep(2)
            # 注意，如果此时是本cycle中最后一次已标注集训练，则训练完要进行eval
            # 调试的时候可以关闭这里的eval，节约时间
            train_detector(model, datasets, cfg,
                           distributed=distributed, validate=args.no_validate, timestamp=timestamp, meta=meta)
            cfg = cfg_bak
        #/ home / LAB / zhengyy / allcode / MIAOD - Retina / work_dirs
        #timestamp='20230726_133954'
        folder_files = os.listdir('/home/LAB/zhengyy/allcode/MIAOD/datasets/test')
        model_path = '/home/LAB/zhengyy/allcode/MIAOD-Retina/work_dirs/Retina/' + timestamp + '/latest.pth'
        conf = '/home/LAB/zhengyy/allcode/MIAOD-Retina/configs/MIAOD.py'
        model1 = init_detector(conf, model_path, device='cuda:0')
        for folder in folder_files:
            print(folder)
            ori_folder_path = os.path.join('/home/LAB/zhengyy/allcode/MIAOD/datasets/test', folder)
            save_folder_path = '/home/LAB/zhengyy/allcode/MIAOD-Retina/work_dirs/Retina/'+timestamp+'/cycle'+str(cycle)+'/'+folder

            result, uncertainty = inference_detector(model1, ori_folder_path)
            # uncertainty = calculate_uncertainty_single(cfg, model, args.img_file, return_box=False)
            # 返回画好框之后的图
            img = model1.show_result(ori_folder_path, result, score_thr=0.5)
            mmcv.imwrite(img, save_folder_path)

        # ---------- 图像选择 ----------
        if cycle != cfg.cycles[-1]:
            # get new labeled data
            dataset_al = build_dataset(cfg.data.test)
            data_loader = build_dataloader(dataset_al, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                           dist=False, shuffle=False)
            # set random seeds
            if args.seed is not None:
                logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
                set_random_seed(args.seed, deterministic=args.deterministic)
            cfg.seed = args.seed
            meta['seed'] = args.seed

            # 筛选策略，得到每张图的信息分数
            if strategy_cfg.get(select_cfg) is None:
                raise Exception("input wrong select strategy！！")
            # 单独处理coreset逻辑
            elif select_cfg == "coreset":
                X_L, X_U = update_by_coreset(model, data_loader, X_all, X_L, cfg.X_S_size, return_box=False)
            else:
                info_score = eval(strategy_cfg.get(select_cfg) + "(cfg, model, data_loader, return_box=False)")
                # 根据所有图像的信息分数，更新下一轮的已标注集和未标注集
                X_L, X_U = update_X_L(info_score, X_all, X_L, cfg.X_S_size)

            # 保存下轮将使用的已标注集和未标注集的图像编号，基本用不上
            np.save(cfg.work_directory + '/X_L_' + str(cycle+1) + '.npy', X_L)
            np.save(cfg.work_directory + '/X_U_' + str(cycle+1) + '.npy', X_U)


if __name__ == '__main__':
    main()
