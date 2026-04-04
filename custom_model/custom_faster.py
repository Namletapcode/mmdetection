_base_ = [
    'configs/_base_/models/faster-rcnn_r50_fpn.py',
    'configs/_base_/datasets/coco_detection.py', 
    'configs/_base_/schedules/schedule_1x.py',
    'configs/_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=16)))

dataset_type = 'CocoDataset'
data_root = 'data/images/' 
json_root = 'data/'
metainfo = {'classes': ('falciparum_R', 'falciparum_S', 'falciparum_T', 'falciparum_G',
                         'vivax_R', 'vivax_S', 'vivax_T', 'vivax_G',
                         'ovale_R', 'ovale_S', 'ovale_T', 'ovale_G',
                         'malariae_R', 'malariae_S', 'malariae_T', 'malariae_G')}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(800, 800)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(batch_size=2, dataset=dict(type=dataset_type, metainfo=metainfo, data_root='', ann_file=json_root + 'train_coco.json', data_prefix=dict(img=data_root), pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(type=dataset_type, metainfo=metainfo, data_root='', ann_file=json_root + 'val_coco.json', data_prefix=dict(img=data_root), pipeline=test_pipeline)) 
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=json_root + 'val_coco.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5), 
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        max_keep_ckpts=10,
        out_dir='/kaggle/working/checkpoints/faster_rcnn'
    )
)
