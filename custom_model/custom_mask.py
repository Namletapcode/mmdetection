_base_ = ['configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

# 1. CẤU HÌNH MODEL
model = dict(rpn_head=dict(anchor_generator=dict(scales=[2, 4], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64])),
             roi_head=dict(bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4)))

dataset_type = 'CocoDataset'
data_root = 'data/original/images/' 
json_root = 'data/original/4-labels/mask/'
metainfo = {'classes': ('falciparum', 'vivax', 'ovale', 'malariae')}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True), 
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True), 
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(batch_size=2, dataset=dict(type=dataset_type, metainfo=metainfo, data_root='', ann_file=json_root + 'train_coco.json', data_prefix=dict(img=data_root), pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(type=dataset_type, metainfo=metainfo, data_root='', ann_file=json_root + 'val_coco.json', data_prefix=dict(img=data_root), pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(type=dataset_type, metainfo=metainfo, data_root='', ann_file=json_root + 'test_coco.json', data_prefix=dict(img=data_root), pipeline=test_pipeline))
val_evaluator = dict(type='CocoMetric', ann_file=json_root + 'val_coco.json', metric=['bbox', 'segm'], format_only=False)
test_evaluator = dict(type='CocoMetric', ann_file=json_root + 'test_coco.json', metric=['bbox', 'segm'], format_only=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, out_dir='/kaggle/working/checkpoints/mask_rcnn')
)