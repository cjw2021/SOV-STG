set -x
PROJECT_NAME=vcoco
EXP_DIR=params

python generate_vcoco_official.py \
  -c slconfig/vcoco-sov-stg-l.py \
  --resume ${EXP_DIR}/sov-stg-l_vcoco.pth \
  --dataset_file vcoco \
  --hoi_path data/v-coco \
  --num_obj_classes 81 \
  --num_verb_classes 29 \
  --output_dir vcoco_eval \
  --batch_size 2 \
  --eval

python vsrl_eval.py --vcoco_path data/v-coco --detections vcoco_eval