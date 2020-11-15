# model settings
model = dict(
    type='R3Det_nofr',
    pretrained='/home/alex/2lab/r3det-on-mmdetection/work_dirs/MobileNetV1_alex.pth',
    #pretrained=None,
    backbone=dict(
        type='MobileNetV1',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
        frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
#         style='pytorch',
    ),
    neck=dict(
        type='FPN',
        #in_channels=[256, 512, 1024, 2048],
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHead',
        num_classes=40,
        in_channels=256,
        stacked_convs=4,
        use_h_gt=True,
        feat_channels=256,
        anchor_generator=dict(
            type='RAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 0.2, 5.0],
            angles=None,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHABBoxCoder',
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=0.11,
            loss_weight=1.0)),
    frm_cfgs=[
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128])
    ],
    num_refine_stages=2,
    refine_heads=[
        dict(
            type='RRetinaRefineHead',
            num_classes=40,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
        dict(
            type='RRetinaRefineHead',
            num_classes=40,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
    ]
)
# training and testing settings
train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    sr=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.6,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        )
    ],
    stage_loss_weights=[1.0, 1.0]
)

# merge_nms_iou_thr_dict = {
#     'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.1,
#     'soccer-ball-field': 0.3, 'small-vehicle': 0.05, 'ship': 0.05, 'plane': 0.3,
#     'large-vehicle': 0.05, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
#     'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3
# }

merge_nms_iou_thr_dict = {
   'car1-LUP': 0.1, 'car1-LDOWN': 0.1, 'car1-RUP': 0.1, 'car1-RDOWN': 0.1,
   'car2-LUP': 0.1, 'car2-LDOWN': 0.1, 'car2-RUP': 0.1, 'car2-RDOWN': 0.1,
   'car3-LUP': 0.1, 'car3-LDOWN': 0.1, 'car3-RUP': 0.1, 'car3-RDOWN': 0.1,
   'car4-LUP': 0.1, 'car4-LDOWN': 0.1, 'car4-RUP': 0.1, 'car4-RDOWN': 0.1,
   'car5-LUP': 0.1, 'car5-LDOWN': 0.1, 'car5-RUP': 0.1, 'car5-RDOWN': 0.1,
   'car6-LUP': 0.1, 'car6-LDOWN': 0.1, 'car6-RUP': 0.1, 'car6-RDOWN': 0.1,
   'car7-LUP': 0.1, 'car7-LDOWN': 0.1, 'car7-RUP': 0.1, 'car7-RDOWN': 0.1,
   'car8-LUP': 0.1, 'car8-LDOWN': 0.1, 'car8-RUP': 0.1, 'car8-RDOWN': 0.1,
   'car9-LUP': 0.1, 'car9-LDOWN': 0.1, 'car9-RUP': 0.1, 'car9-RDOWN': 0.1,
   'car10-LUP': 0.1, 'car10-LDOWN': 0.1, 'car10-RUP': 0.1, 'car10-RDOWN': 0.1,
}

# merge_nms_iou_thr_dict = {
#     'car1': 0.1, 'car2': 0.1, 'car3': 0.1, 'car4': 0.1, 'car5' : 0.1, 
#     'car6': 0.1, 'car7': 0.1, 'car8': 0.1, 'car9': 0.1, 'car10': 0.1, 
# }

merge_cfg = dict(
    nms_pre=2000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=merge_nms_iou_thr_dict),
    max_per_img=1000,
)

test_cfg = dict(
    nms_pre=1000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=0.05),
    max_per_img=100,
    merge_cfg=merge_cfg
)
