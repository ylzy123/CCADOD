from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import os
import argparse
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on single images')
    parser.add_argument('config_file', help='train config file path')
    parser.add_argument('ckpt_file', help='model checkpoint file path')
    parser.add_argument('img_dir', help='image dir path')
    parser.add_argument('out_dir', help='output image dir path')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    model = init_detector(args.config_file, args.ckpt_file, device='cuda:0')
    folder_files = os.listdir(args.img_dir)
    folder_files.sort()
    #args.img_dir=['/ home / chenjiahao / allcode / MIAOD / datasets / VOCdevkit / VOC2007 / JPEGImages','/ home / chenjiahao / allcode / MIAOD / datasets / VOCdevkit / VOC2012 / JPEGImages']
    count = 1

    folder = 'VOC2007'
    #img_files = os.listdir()
    ori_folder_path = os.listdir(os.path.join(args.img_dir, folder)+'/JPEGImages')
    save_folder_path = args.out_dir+'/test/'+folder[-2:]
    img_file = '007615.jpg'
    ori_img_path = args.img_dir+'/'+str(folder)+ '/JPEGImages/'+str(img_file)
    save_img_path = os.path.join(save_folder_path, img_file)
    result, uncertainty = inference_detector(model, ori_img_path)
    # uncertainty = calculate_uncertainty_single(cfg, model, args.img_file, return_box=False)
    # 返回画好框之后的图
    img = model.show_result(ori_img_path, result, score_thr=0.5)
    mmcv.imwrite(img, save_img_path)


if __name__ == '__main__':
    main()

