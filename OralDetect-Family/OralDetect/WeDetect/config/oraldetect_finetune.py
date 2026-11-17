"""OralDetect — the one config. Self-contained, no `_base_`.

Flattened from a five-level inheritance chain with mmengine's `Config.dump()`, and verified
equivalent to it: 70/70 resolved keys equal, identical `state_dict` (1126 params, 0 differences).

Used by BOTH `run_finetune.py` and `run_eval.py`. Every path and hyper-parameter below is a
DEFAULT that the launcher overwrites at runtime from its yaml — edit the yaml, not this file.

Do not change the text tower, the modality calibration or the 1024x1024 input unless you mean to:
the released checkpoint will no longer load cleanly, and mmengine drops mismatched keys with only
a log line. The launchers diff the checkpoint against the model and refuse to start on a mismatch.
"""


CLASS_NAMES = '/path/to/class_names.json'
CLASS_TEXT = '/path/to/class_texts.json'
DATAS = '/path/to/datas'
DATA_ROOT = '/path/to/images/'
DENTALBERT = '/path/to/oralbert'
TEST_ANN = '/path/to/instances_val.json'
TRAIN_ANN = '/path/to/instances_train.json'
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 5e-06
close_mosaic_epochs = 4
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'wedetect',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/macro4_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmdet'
dist_cfg = dict(backend='nccl', timeout=10800)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
img_scale = (
    1024,
    1024,
)
load_from = '/path/to/oraldetect.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
max_epochs = 6
metainfo = dict(
    classes=(
        'abnormal oral epithelial cell',
        'alveolar bone loss',
        'anterior teeth with fenestration or dehiscence',
        'anterior teeth without fenestration or dehiscence',
        'canine',
        'central incisor',
        'craniofacial or oral anomaly',
        'dental abrasion',
        'dental caries',
        'dental crown restoration',
        'dental filling',
        'dental implant',
        'dental opacity',
        'dental plaque',
        'dental restoration',
        'dental restoration (filling or crown)',
        'dividing oral cell',
        'first molar',
        'first premolar',
        'foreign object or debris',
        'hard recognized oral cell',
        'impacted tooth',
        'intraoral appliance',
        'lateral incisor',
        'lightly abnormal oral cell',
        'malignant oral cell',
        'mandibular canal',
        'maxillary sinus',
        'missing or residual root',
        'missing teeth',
        'normal',
        'normal oral cell',
        'oral blood cell',
        'orthodontic bracket',
        'periapical lesion',
        'periodontal pocket',
        'primary endodontic lesion',
        'primary endodontic with secondary periodontal lesion',
        'primary periodontal lesion',
        'primary periodontal with secondary endodontic lesion',
        'quadrant 1 (upper right)',
        'quadrant 2 (upper left)',
        'quadrant 3 (lower left)',
        'quadrant 4 (lower right)',
        'reactive oral cell',
        'retained root',
        'root canal treatment',
        'second molar',
        'second premolar',
        'severe gingivitis',
        'suspicious malignant oral cell',
        'tooth 11 (upper right central incisor)',
        'tooth 12 (upper right lateral incisor)',
        'tooth 13 (upper right canine)',
        'tooth 14 (upper right first premolar)',
        'tooth 15 (upper right second premolar)',
        'tooth 16 (upper right first molar)',
        'tooth 17 (upper right second molar)',
        'tooth 18 (upper right third molar)',
        'tooth 21 (upper left central incisor)',
        'tooth 22 (upper left lateral incisor)',
        'tooth 23 (upper left canine)',
        'tooth 24 (upper left first premolar)',
        'tooth 25 (upper left second premolar)',
        'tooth 26 (upper left first molar)',
        'tooth 27 (upper left second molar)',
        'tooth 28 (upper left third molar)',
        'tooth 31 (lower left central incisor)',
        'tooth 32 (lower left lateral incisor)',
        'tooth 33 (lower left canine)',
        'tooth 34 (lower left first premolar)',
        'tooth 35 (lower left second premolar)',
        'tooth 36 (lower left first molar)',
        'tooth 37 (lower left second molar)',
        'tooth 38 (lower left third molar)',
        'tooth 41 (lower right central incisor)',
        'tooth 42 (lower right lateral incisor)',
        'tooth 43 (lower right canine)',
        'tooth 44 (lower right first premolar)',
        'tooth 45 (lower right second premolar)',
        'tooth 46 (lower right first molar)',
        'tooth 47 (lower right second molar)',
        'tooth 48 (lower right third molar)',
        'tooth erosion',
        'tooth malformation',
        'true combined endo-perio lesion',
        'tumor',
    ))
mixup_prob = 0.15
model = dict(
    backbone=dict(
        image_model=dict(
            frozen_modules=[],
            model_name='base',
            type='ConvNextVisionBackbone'),
        text_model=dict(
            frozen_modules=[],
            model_name=
            '/path/to/oralbert',
            model_size='base',
            type='DentalBertLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        head_module=dict(
            embed_dims=768,
            in_channels=[
                256,
                512,
                1024,
            ],
            model_size='base',
            num_classes=87,
            type='YOLOWorldHeadModule',
            use_bn_head=True),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='mmyoloIoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375, reduction='mean', type='DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=False,
    modality_calib=dict(
        dropout=0.0,
        embed_dims=768,
        num_heads=8,
        num_modality_tokens=4,
        type='ModalityCalibration',
        vision_dims=1024,
        vision_index=-1),
    neck=dict(model_size='base', scale_factor=1.0, type='CSPRepBiFPANNeck'),
    num_test_classes=87,
    num_train_classes=87,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=87,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    512,
]
neck_num_heads = [
    4,
    8,
    16,
]
num_classes = 87
num_training_classes = 87
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=4, lr=5e-06, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(custom_keys=dict(logit_scale=dict(weight_decay=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=200,
        end_factor=1.0,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=6,
        end_factor=0.01,
        start_factor=1.0,
        type='LinearLR'),
]
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 1
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/path/to/class_texts.json',
        dataset=dict(
            ann_file=
            '/path/to/instances_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root=
            '/path/to/images/',
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/path/to/instances_val.json',
    metric='bbox',
    type='PerModalityCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        1024,
        1024,
    ), type='WeDetectKeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            1024,
            1024,
        ),
        type='WeDetectLetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='PackDetInputs'),
]
text_channels = 768
text_transform = [
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_batch_size_per_gpu = 4
train_cfg = dict(
    dynamic_intervals=None,
    max_epochs=6,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        class_text_path=
        '/path/to/class_texts.json',
        dataset=dict(
            ann_file=
            '/path/to/instances_train.json',
            data_prefix=dict(img=''),
            data_root=
            '/path/to/images/',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='WeDetectRandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='WeDetectHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=87,
                num_neg_samples=(
                    87,
                    87,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    class_text_path=
    '/path/to/class_texts.json',
    dataset=dict(
        ann_file=
        '/path/to/instances_train.json',
        data_prefix=dict(img=''),
        data_root=
        '/path/to/images/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'abnormal oral epithelial cell',
                'alveolar bone loss',
                'anterior teeth with fenestration or dehiscence',
                'anterior teeth without fenestration or dehiscence',
                'canine',
                'central incisor',
                'craniofacial or oral anomaly',
                'dental abrasion',
                'dental caries',
                'dental crown restoration',
                'dental filling',
                'dental implant',
                'dental opacity',
                'dental plaque',
                'dental restoration',
                'dental restoration (filling or crown)',
                'dividing oral cell',
                'first molar',
                'first premolar',
                'foreign object or debris',
                'hard recognized oral cell',
                'impacted tooth',
                'intraoral appliance',
                'lateral incisor',
                'lightly abnormal oral cell',
                'malignant oral cell',
                'mandibular canal',
                'maxillary sinus',
                'missing or residual root',
                'missing teeth',
                'normal',
                'normal oral cell',
                'oral blood cell',
                'orthodontic bracket',
                'periapical lesion',
                'periodontal pocket',
                'primary endodontic lesion',
                'primary endodontic with secondary periodontal lesion',
                'primary periodontal lesion',
                'primary periodontal with secondary endodontic lesion',
                'quadrant 1 (upper right)',
                'quadrant 2 (upper left)',
                'quadrant 3 (lower left)',
                'quadrant 4 (lower right)',
                'reactive oral cell',
                'retained root',
                'root canal treatment',
                'second molar',
                'second premolar',
                'severe gingivitis',
                'suspicious malignant oral cell',
                'tooth 11 (upper right central incisor)',
                'tooth 12 (upper right lateral incisor)',
                'tooth 13 (upper right canine)',
                'tooth 14 (upper right first premolar)',
                'tooth 15 (upper right second premolar)',
                'tooth 16 (upper right first molar)',
                'tooth 17 (upper right second molar)',
                'tooth 18 (upper right third molar)',
                'tooth 21 (upper left central incisor)',
                'tooth 22 (upper left lateral incisor)',
                'tooth 23 (upper left canine)',
                'tooth 24 (upper left first premolar)',
                'tooth 25 (upper left second premolar)',
                'tooth 26 (upper left first molar)',
                'tooth 27 (upper left second molar)',
                'tooth 28 (upper left third molar)',
                'tooth 31 (lower left central incisor)',
                'tooth 32 (lower left lateral incisor)',
                'tooth 33 (lower left canine)',
                'tooth 34 (lower left first premolar)',
                'tooth 35 (lower left second premolar)',
                'tooth 36 (lower left first molar)',
                'tooth 37 (lower left second molar)',
                'tooth 38 (lower left third molar)',
                'tooth 41 (lower right central incisor)',
                'tooth 42 (lower right lateral incisor)',
                'tooth 43 (lower right canine)',
                'tooth 44 (lower right first premolar)',
                'tooth 45 (lower right second premolar)',
                'tooth 46 (lower right first molar)',
                'tooth 47 (lower right second molar)',
                'tooth 48 (lower right third molar)',
                'tooth erosion',
                'tooth malformation',
                'true combined endo-perio lesion',
                'tumor',
            )),
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                1024,
                1024,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -512,
                -512,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100.0,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.5,
                1.5,
            ),
            type='WeDetectRandomAffine'),
        dict(
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    img_scale=(
                        1024,
                        1024,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    type='MultiModalMosaic'),
                dict(
                    border=(
                        -512,
                        -512,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_aspect_ratio=100.0,
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    scaling_ratio_range=(
                        0.5,
                        1.5,
                    ),
                    type='WeDetectRandomAffine'),
            ],
            prob=0.15,
            type='YOLOv5MultiModalMixUp'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='WeDetectHSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=87,
            num_neg_samples=(
                87,
                87,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    1024,
                    1024,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='MultiModalMosaic'),
            dict(
                border=(
                    -512,
                    -512,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='WeDetectRandomAffine'),
        ],
        prob=0.15,
        type='YOLOv5MultiModalMixUp'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='WeDetectHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        1024,
        1024,
    ), type='WeDetectKeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1024,
            1024,
        ),
        type='WeDetectLetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='WeDetectHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/path/to/class_texts.json',
        dataset=dict(
            ann_file=
            '/path/to/instances_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root=
            '/path/to/images/',
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset = dict(
    class_text_path=
    '/path/to/class_texts.json',
    dataset=dict(
        ann_file=
        '/path/to/instances_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root=
        '/path/to/images/',
        metainfo=dict(
            classes=(
                'abnormal oral epithelial cell',
                'alveolar bone loss',
                'anterior teeth with fenestration or dehiscence',
                'anterior teeth without fenestration or dehiscence',
                'canine',
                'central incisor',
                'craniofacial or oral anomaly',
                'dental abrasion',
                'dental caries',
                'dental crown restoration',
                'dental filling',
                'dental implant',
                'dental opacity',
                'dental plaque',
                'dental restoration',
                'dental restoration (filling or crown)',
                'dividing oral cell',
                'first molar',
                'first premolar',
                'foreign object or debris',
                'hard recognized oral cell',
                'impacted tooth',
                'intraoral appliance',
                'lateral incisor',
                'lightly abnormal oral cell',
                'malignant oral cell',
                'mandibular canal',
                'maxillary sinus',
                'missing or residual root',
                'missing teeth',
                'normal',
                'normal oral cell',
                'oral blood cell',
                'orthodontic bracket',
                'periapical lesion',
                'periodontal pocket',
                'primary endodontic lesion',
                'primary endodontic with secondary periodontal lesion',
                'primary periodontal lesion',
                'primary periodontal with secondary endodontic lesion',
                'quadrant 1 (upper right)',
                'quadrant 2 (upper left)',
                'quadrant 3 (lower left)',
                'quadrant 4 (lower right)',
                'reactive oral cell',
                'retained root',
                'root canal treatment',
                'second molar',
                'second premolar',
                'severe gingivitis',
                'suspicious malignant oral cell',
                'tooth 11 (upper right central incisor)',
                'tooth 12 (upper right lateral incisor)',
                'tooth 13 (upper right canine)',
                'tooth 14 (upper right first premolar)',
                'tooth 15 (upper right second premolar)',
                'tooth 16 (upper right first molar)',
                'tooth 17 (upper right second molar)',
                'tooth 18 (upper right third molar)',
                'tooth 21 (upper left central incisor)',
                'tooth 22 (upper left lateral incisor)',
                'tooth 23 (upper left canine)',
                'tooth 24 (upper left first premolar)',
                'tooth 25 (upper left second premolar)',
                'tooth 26 (upper left first molar)',
                'tooth 27 (upper left second molar)',
                'tooth 28 (upper left third molar)',
                'tooth 31 (lower left central incisor)',
                'tooth 32 (lower left lateral incisor)',
                'tooth 33 (lower left canine)',
                'tooth 34 (lower left first premolar)',
                'tooth 35 (lower left second premolar)',
                'tooth 36 (lower left first molar)',
                'tooth 37 (lower left second molar)',
                'tooth 38 (lower left third molar)',
                'tooth 41 (lower right central incisor)',
                'tooth 42 (lower right lateral incisor)',
                'tooth 43 (lower right canine)',
                'tooth 44 (lower right first premolar)',
                'tooth 45 (lower right second premolar)',
                'tooth 46 (lower right first molar)',
                'tooth 47 (lower right second molar)',
                'tooth 48 (lower right third molar)',
                'tooth erosion',
                'tooth malformation',
                'true combined endo-perio lesion',
                'tumor',
            )),
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1024,
            1024,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1024,
                1024,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
val_evaluator = dict(
    ann_file=
    '/path/to/instances_val.json',
    metric='bbox',
    type='PerModalityCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
