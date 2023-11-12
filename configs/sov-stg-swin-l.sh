set -x
PROJECT_NAME=hico
EXP_NAME=sov-stg-swin-l_00001
EXP_DIR=logs/${PROJECT_NAME}_${EXP_NAME}

python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --dataset_file hico \
  --hoi_path data/hico_det \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --batch_size 2 \
  --resume ./params/sov-stg-swin-l_hico.pth \
  --output_dir ${EXP_DIR} \
  --use_wandb \
  --wandb_project ${PROJECT_NAME} \
  --wandb_name ${EXP_NAME} \
  --use_checkpoint \
  -c slconfig/sov-stg-swin_l.py