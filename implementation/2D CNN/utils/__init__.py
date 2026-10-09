"""
Utils paket za 2D CNN projekt.

Organizuje sve utility funkcije, konstante i helper klase u jednom mjestu.
"""

# Exportuj sve konstante
from .constants import (
    CATEGORY_MAP,
    NUM_CATEGORIES,
    BBOX_FEATURE_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FRAME_CACHE_MAX_SIZE,
    CNN_CACHE_MAX_SIZE,
    AUG_BRIGHTNESS_RANGE,
    AUG_CONTRAST_RANGE,
    AUG_SATURATION_RANGE,
    AUG_HFLIP_PROB,
    AUG_ERASING_PROB,
    AUG_ERASING_SCALE,
    INV_255,
)

# Exportuj bbox funkcije
from .bbox_utils import (
    normalize_bbox_coords,
    compute_bbox_properties,
    compute_velocity_features,
    extract_bbox_features,
    find_previous_object_match,
    apply_horizontal_flip_to_bbox,
)

# Exportuj augmentacijske funkcije
from .augmentation_utils import (
    AugmentationParams,
    apply_temporal_jitter,
    apply_color_jitter,
    normalize_image,
    apply_horizontal_flip_to_flow,
    sample_temporal_mask_indices,
    create_random_erasing_transform,
)

# Exportuj flow funkcije
from .flow_utils import (
    load_flow_for_frame,
    resize_flow_if_needed,
    subtract_ego_motion,
    extract_flow_statistics_for_bbox,
    add_flow_features_to_bbox,
    extract_per_object_flow_patches,
    process_flow_sequence,
)

# Exportuj training utilities
from .training_utils import (
    EMA,
    mixup_batch,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)

# Exportuj loss funkcije
from .loss_functions import (
    FocalBCEWithLogitsLoss,
    WeightedBCEWithLogitsLoss,
)

# Exportuj evaluation utilities
from .evaluation_utils import (
    POSTPROCESS_CHOICES,
    apply_postprocess,
    postprocess_label,
    compute_metrics,
    find_best_threshold,
    threshold_sweep,
    compute_video_level_confusion,
)

__all__ = [
    # Constants
    'CATEGORY_MAP',
    'NUM_CATEGORIES',
    'BBOX_FEATURE_DIM',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'FRAME_CACHE_MAX_SIZE',
    'CNN_CACHE_MAX_SIZE',
    'AUG_BRIGHTNESS_RANGE',
    'AUG_CONTRAST_RANGE',
    'AUG_SATURATION_RANGE',
    'AUG_HFLIP_PROB',
    'AUG_ERASING_PROB',
    'AUG_ERASING_SCALE',
    'INV_255',
    # BBox utils
    'normalize_bbox_coords',
    'compute_bbox_properties',
    'compute_velocity_features',
    'extract_bbox_features',
    'find_previous_object_match',
    'apply_horizontal_flip_to_bbox',
    # Augmentation utils
    'AugmentationParams',
    'apply_temporal_jitter',
    'apply_color_jitter',
    'normalize_image',
    'create_random_erasing_transform',
    'apply_horizontal_flip_to_flow',
    'sample_temporal_mask_indices',
    # Flow utils
    'load_flow_for_frame',
    'resize_flow_if_needed',
    'subtract_ego_motion',
    'extract_flow_statistics_for_bbox',
    'add_flow_features_to_bbox',
    'extract_per_object_flow_patches',
    'process_flow_sequence',
    # Training utils
    'EMA',
    'mixup_batch',
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    # Loss functions
    'FocalBCEWithLogitsLoss',
    'WeightedBCEWithLogitsLoss',
    # Evaluation utils
    'POSTPROCESS_CHOICES',
    'apply_postprocess',
    'postprocess_label',
    'compute_metrics',
    'find_best_threshold',
    'threshold_sweep',
    'compute_video_level_confusion',
]
