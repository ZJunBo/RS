from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er


TARGET_SET = 'Potsdam'

source_dir = dict(
    image_dir=[
        './MUCSS/Train/Vaihingen/images/',
    ],
    mask_dir=[
        './MUCSS/Train/Vaihingen/labels/',
    ],
)
target_dir = dict(
    image_dir=[
        './MUCSS/Val/PotsdamIRRG/images/',
    ],
    mask_dir=[
        './MUCSS/Val/PotsdamIRRG/masks/',
    ],
)
test_target_dir = dict(
    image_dir=[
        './MUCSS/Test/PotsdamIRRG/images/',
    ],
    mask_dir=[
        './MUCSS/Test/PotsdamIRRG/masks/',
    ],
)



SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(120.356966, 81.055408, 80.504692),
                  std=(54.889108, 38.964442, 37.529461),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=4,
    num_workers=8,
)

TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(120.356966, 81.055408, 80.504692),
                  std=(54.889108, 38.964442, 37.529461),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=4,
    num_workers=8,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(120.356966, 81.055408, 80.504692),
                  std=(54.889108, 38.964442, 37.529461),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=0,
)
# EVAL_DATA_CONFIG = dict(
#     image_dir=source_dir['image_dir'],
#     mask_dir=source_dir['mask_dir'],
#     transforms=Compose([
#         Normalize(mean=(120.356966, 81.055408, 80.504692),
#                   std=(54.889108, 38.964442, 37.529461),
#                   max_pixel_value=1, always_apply=True),
#         er.preprocess.albu.ToTensor()
#
#     ]),
#     CV=dict(k=10, i=-1),
#     training=False,
#     batch_size=1,
#     num_workers=0,
# )

TEST_DATA_CONFIG = dict(
    image_dir=test_target_dir['image_dir'],
    mask_dir = test_target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(120.356966, 81.055408, 80.504692),
                  std=(54.889108, 38.964442, 37.529461),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=0,
)