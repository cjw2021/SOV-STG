set -x
PROJECT_NAME=vcoco
EXP_NAME=sov-stg-l_00001
EXP_DIR=logs/${PROJECT_NAME}_${EXP_NAME}

python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --batch_size 4 \
  --dataset_file vcoco \
  --hoi_path data/v-coco \
  --num_obj_classes 81 \
  --num_verb_classes 29 \
  --resume ./params/sov-stg-l_vcoco.pth \
  --output_dir ${EXP_DIR} \
  --use_wandb \
  --wandb_project ${PROJECT_NAME} \
  --wandb_name ${EXP_NAME} \
  -c slconfig/vcoco-sov-stg-l.py
  
