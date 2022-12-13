_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/imagenets.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

model = dict(
    backbone=dict(dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2)),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        channels=2048,
        num_convs=0,
        num_classes=920,
        ignore_index=1000,
        downsample_label_ratio=8), 
    auxiliary_head=None)

# By default, models are trained on 8 GPUs with 32 images per GPU
data = dict(samples_per_gpu=32)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3600)
checkpoint_config = dict(by_epoch=False, interval=3600)
evaluation = dict(interval=360, metric='mIoU', pre_eval=True)
