set -x
PROJECT_NAME=hico
EXP_NAME=sov-stg-swin-l_00001
EXP_DIR=logs/${PROJECT_NAME}_${EXP_NAME}

python main.py \
  --dataset_file hico \
  --hoi_path data/hico_det \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --batch_size 2 \
  --resume ${EXP_DIR}/checkpoint_best.pth \
  --output_dir ${EXP_DIR}_eval \
  -c slconfig/sov-stg-swin_l.py \
  --eval