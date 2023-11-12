_base_ = ['hoi_base_setting.py']

#############################
# model setting
#############################
device                 = 'cuda'
arch                   = 'SOV-STG'
vdec_box_type          = 'adaptive_shifted_MBR'

# transformer
enc_layers             = 6
dec_layers             = 6
dim_feedforward        = 2048
hidden_dim             = 256
dropout                = 0.0
nheads                 = 8
num_queries            = 64
transformer_activation = 'relu'
num_patterns           = 0
random_refpoints_xy    = False

# dab-deformable-detr
dec_n_points           = 8
enc_n_points           = 4
use_same_refpoint      = False

# backbone
backbone               = 'resnet101'
num_feature_levels     = 4

#############################
# training setting
#############################
lr                     = 2e-4
lr_backbone            = 2e-6
weight_decay           = 1e-4
lr_drop                = 20
epochs                 = 30
train_enc              = True

# loss coefficients
dn_loss_coef           = 2.5
bbox_loss_coef         = 2.5
iou_loss_coef          = 1
obj_loss_coef          = 1
verb_loss_coef         = 1
focal_alpha            = 0.25

#############################
# dn setting
#############################
dn_drop_lower_limit    = 1.0                       # minimum scale of dn_loss_weight at the last epoch
scalar                 = 6                         # number of dn groups
dn_dynamic_coef_type   = 'linear'                  # noise scale dynamic coefficient type
dn_dynamic_lower_limit = 0.666666
label_noise_scale      = 0.3
verb_noise_scale       = 0.6
box_noise_scale        = 0.4
verb_noise_prob_h      = -1
verb_noise_prob_l      = 0.6
wo_obj                 = False
wo_verb                = False

# STG
pred_all               = True
wo_mix_dn              = True
use_dn_weight          = True