set -x
PROJECT_NAME=vcoco
EXP_NAME=sov-stg-l_00001
EXP_DIR=logs/${PROJECT_NAME}_${EXP_NAME}

python generate_vcoco_official.py \
  -c slconfig/vcoco-sov-stg-l.py \
  --resume ${EXP_DIR}/checkpoint_best.pth \
  --dataset_file vcoco \
  --hoi_path data/v-coco \
  --num_obj_classes 81 \
  --num_verb_classes 29 \
  --output_dir ${EXP_DIR}_eval \
  --batch_size 2 \
  --eval

python vsrl_eval.py --vcoco_path data/v-coco --detections ${EXP_DIR}_eval