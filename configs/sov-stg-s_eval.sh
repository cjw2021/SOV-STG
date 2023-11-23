set -x
PROJECT_NAME=hico
EXP_DIR=params

python main.py \
  --dataset_file hico \
  --hoi_path data/hico_det \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --batch_size 2 \
  --resume ${EXP_DIR}/sov-stg-s.pth \
  --output_dir sov-stg-s_eval \
  -c slconfig/sov-stg-s.py \
  --eval