"""
Konstante i enumeracije za DoTA dataset i CNN+LSTM+BBox model.
"""

from typing import Dict


# ========================================================================= #
#  Kategorije objekata                                                       #
# ========================================================================= #

CATEGORY_MAP: Dict[str, int] = {
    "car": 1,
    "truck": 2,
    "bus": 3,
    "person": 4,
    "rider": 5,
    "bike": 6,
    "motor": 7,
}

NUM_CATEGORIES = len(CATEGORY_MAP)  # 7
PADDING_CATEGORY = 0


# ========================================================================= #
#  Normalizacijske konstante (ImageNet)                                      #
# ========================================================================= #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========================================================================= #
#  Značajke bounding box-ova                                                  #
# ========================================================================= #

# Broj numeričkih značajki po objektu
# [cx, cy, w, h, area, aspect_ratio, dx, dy, dw, dh, conf, 
#  approach_rate, expansion_rate, ttc, flow_mag, flow_angle, flow_std]
BBOX_FEATURE_DIM = 17

# Indeksi značajki
BBOX_IDX_CX = 0
BBOX_IDX_CY = 1
BBOX_IDX_W = 2
BBOX_IDX_H = 3
BBOX_IDX_AREA = 4
BBOX_IDX_ASPECT = 5
BBOX_IDX_DX = 6
BBOX_IDX_DY = 7
BBOX_IDX_DW = 8
BBOX_IDX_DH = 9
BBOX_IDX_CONF = 10
BBOX_IDX_APPROACH_RATE = 11
BBOX_IDX_EXPANSION_RATE = 12
BBOX_IDX_TTC = 13
BBOX_IDX_FLOW_MAG = 14
BBOX_IDX_FLOW_ANGLE = 15
BBOX_IDX_FLOW_STD = 16


# ========================================================================= #
#  Cache postavke                                                             #
# ========================================================================= #

FRAME_CACHE_MAX_SIZE = 5000
CNN_CACHE_MAX_SIZE = 200
FLOW_CACHE_MAX_SIZE = 2048


# ========================================================================= #
#  Augmentacija rasponi                                                       #
# ========================================================================= #

# Color jitter rasponi
AUG_BRIGHTNESS_RANGE = (-0.4, 0.4)
AUG_CONTRAST_RANGE = (-0.4, 0.4)
AUG_SATURATION_RANGE = (-0.3, 0.3)
AUG_GRAYSCALE_PROB = 0.05

# Temporal jitter
AUG_TEMPORAL_JITTER_RANGE = 3

# Random erasing
AUG_ERASING_PROB = 0.25
AUG_ERASING_SCALE = (0.02, 0.15)

# Horizontal flip probability
AUG_HFLIP_PROB = 0.5


# ========================================================================= #
#  Različite konstante                                                        #
# ========================================================================= #

# Minimalna udaljenost za NN matching objekata (normalizirano)
MIN_NN_MATCHING_DISTANCE_SQ = 0.01  # 0.1²

# Maksimalna TTC vrijednost
MAX_TTC_VALUE = 5.0

# Minimalne epsilon vrijednosti za dijeljenje
EPS_SMALL = 1e-8
EPS_MEDIUM = 1e-6
EPS_LARGE = 1e-4

# Default frame rate (fps)
DEFAULT_FPS = 10

# Conversion constants
INV_255 = 1.0 / 255.0
