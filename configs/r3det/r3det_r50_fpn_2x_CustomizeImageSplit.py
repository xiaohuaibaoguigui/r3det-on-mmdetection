_base_ = [
    #'models/r3det_r50_fpn.py',
    'models/r3det_mbv1_fpn.py',
    'datasets/dotav1_rotational_detection.py',
    'schedules/schedule_1x.py'
]

# runtime settings
checkpoint_config = dict(interval=7)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/r3det_mbv1_fpn_0x_20201115_IM_FZ_NOPATCH'
# work_dir = './work_dirs/r3det_mbv1_fpn_2x_20201111_IM_FZ_NOPATCH_x4_002'
evaluation = dict(interval=7, metric='mAP')
