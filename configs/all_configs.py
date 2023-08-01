data_root = '/home/chenjiahao/allcode/MIAOD/datasets/VOCdevkit/'
_base_ = [
    './retinanet_r50_fpn.py', './voc0712.py',
    './default_runtime.py'
]
data = dict(
    test=dict(
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'])
)
model = dict(bbox_head=dict(C=20))
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[2])
checkpoint_config = dict(interval=1)
log_config = dict(interval=50)
epoch_ratio = [3,1]
evaluation = dict(interval=epoch_ratio[0], metric='mAP')
epoch = 2
X_L_repeat = 2
X_U_repeat = 2
train_cfg = dict(param_lambda = 1)
k = 10000
X_S_size = 10#16551//40
X_L_0_size = 10#16551//20
cycles = [0, 1, 2, 3, 4, 5, 6]
work_directory = './work_dirs'

