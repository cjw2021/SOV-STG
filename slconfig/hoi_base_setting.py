#############################
# model setting
#############################
dilation               = False
position_embedding     = 'sine'
pe_temperatureH        = 20
pe_temperatureW        = 20
batch_norm_type        = 'FrozenBatchNorm2d'
backbone_freeze_keywords = None

masks                  = False

#############################
# training setting
#############################
drop_lr_now            = False
clip_max_norm          = 0.1
frozen_weights         = None

# loss
aux_loss               = True

# matcher
set_cost_bbox          = 2.5
set_cost_iou           = 1
set_cost_obj_class     = 1
set_cost_verb_class    = 1


# seed
seed                   = 42

#############################
# post-process
#############################
use_nms                = True
nms_thresh             = 0.7
nms_alpha              = 1.0
nms_beta               = 0.5
eval_extra             = False


