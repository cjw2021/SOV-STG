import os
import torch
import torchvision.transforms as T
import cv2
import numpy as np

from dataclasses import dataclass
from PIL import Image

from models.backbone.backbone_resnet import Backbone as ResnetBackbone, Joiner as ResnetJoiner
from models.backbone.backbone_swin import Backbone as SwinBackbone, Joiner as SwinJoiner
from models.backbone.position_encoding import PositionEmbeddingSine
from models.encoder_decoder.deformable_transformer_SOV import DeformableTransformer
from models.detector.SOV_STG import SOV_STG, PostProcessHOI

@dataclass
class extra_args:
    arch: str
    train_enc: bool
    use_checkpoint: bool
    use_dn_weight: bool
    vdec_box_type: str
    num_queries: int
    dec_n_points: int
    enc_n_points: int
    dec_layers: int
    wo_verb: bool
    wo_obj: bool
    use_same_refpoint: bool

def modify_forward(instance, fixed_arg2=(0,)):
    original_forward = instance.forward

    def custom_forward(arg1):
        result, _ = original_forward(arg1, fixed_arg2)
        return result

    instance.forward = custom_forward


def _make_sovstg(backbone_name: str,
                swin_pretrained: str="",
                dec_layers=3,
                dilation=False,
                num_classes=80,
                num_verb_classes=117,
                mask=False,
                num_feature_levels=4,
                args=None
                ):

    hidden_dim = 256
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    if "swin" in backbone_name:
        backbone = SwinBackbone(backbone_name, num_feature_levels=num_feature_levels, pretrained=swin_pretrained, dilation=False)
        backbone_with_pos_enc = SwinJoiner(backbone, position_embedding)
    else:
        backbone = ResnetBackbone(backbone_name, train_backbone=True, return_interm_layers=mask or (num_feature_levels > 1), dilation=dilation)
        backbone_with_pos_enc = ResnetJoiner(backbone, position_embedding)

    SOVTransformer = DeformableTransformer(d_model=hidden_dim,
                                        nhead=8,
                                        num_encoder_layers=6,
                                        num_decoder_layers=dec_layers,
                                        dim_feedforward=2048,
                                        dropout=0.0,
                                        activation='relu',
                                        return_intermediate_dec=True,
                                        num_feature_levels=num_feature_levels,
                                        dec_n_points=args.dec_n_points,
                                        enc_n_points=args.enc_n_points,
                                        args=args)
    model = SOV_STG(backbone_with_pos_enc,
                SOVTransformer,
                num_classes=num_classes,
                num_verb_classes=num_verb_classes,
                num_queries=args.num_queries,
                num_feature_levels=num_feature_levels,
                aux_loss=False,
                num_patterns=0,
                random_refpoints_xy=False,
                args=args
            )
    
    modify_forward(model)

    return model




def pretrained_sov_stg_s(checkpoint_path, return_postprocessor=False):
    """
    SOVTransformer ResNet50 with 6 encoder and 3 decoder layers.

    Achieves 33.8 AP on HICO-Det.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist. ")
    if not os.path.isfile(checkpoint_path):
        raise Exception(f"{checkpoint_path} is not a file. ")
    
    args = extra_args(arch='SOV-STG',
                      train_enc=False,
                      use_checkpoint=False,
                      use_dn_weight=True,
                      vdec_box_type='adaptive_shifted_MBR',
                      num_queries=64,
                      dec_n_points=8,
                      enc_n_points=4,
                      wo_verb=False,
                      wo_obj=False,
                      use_same_refpoint=False,
                      dec_layers=3,
                      )

    model = _make_sovstg(backbone_name="resnet50",
                         dec_layers=3,
                         dilation=False,
                         num_classes=80,
                         num_verb_classes=117,
                         mask=False,
                         num_feature_levels=4,
                         args=args
                        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessHOI(subject_category_id=0)
    return model


def pretrained_sov_stg_swinl(checkpoint_path, return_postprocessor=False):
    """
    SOVTransformer swin_large_384 with 6 encoder and 6 decoder layers.

    Achieves 43.35 AP on HICO-Det.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist. ")
    if not os.path.isfile(checkpoint_path):
        raise Exception(f"{checkpoint_path} is not a file. ")
    
    args = extra_args(arch='SOV-STG',
                      train_enc=False,
                      use_checkpoint=False,
                      use_dn_weight=True,
                      vdec_box_type='adaptive_shifted_MBR',
                      num_queries=64,
                      dec_n_points=8,
                      enc_n_points=4,
                      wo_verb=False,
                      wo_obj=False,
                      use_same_refpoint=False,
                      dec_layers=3,
                      )

    model = _make_sovstg(backbone_name="swin_large_384",
                         dec_layers=6,
                         dilation=False,
                         num_classes=80,
                         num_verb_classes=117,
                         mask=False,
                         num_feature_levels=4,
                         args=args
                        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessHOI(subject_category_id=0)
    return model



def pretrained_sov_stg_l_vcoco(checkpoint_path, return_postprocessor=False):
    """
    SOVTransformer ResNet101 with 6 encoder and 6 decoder layers.

    Achieves 63.9/65.4 AP on VCOCO S1/S2.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist. ")
    if not os.path.isfile(checkpoint_path):
        raise Exception(f"{checkpoint_path} is not a file. ")
    
    args = extra_args(arch='SOV-STG',
                      train_enc=False,
                      use_checkpoint=False,
                      use_dn_weight=True,
                      vdec_box_type='adaptive_shifted_MBR',
                      num_queries=64,
                      dec_n_points=4,
                      enc_n_points=4,
                      wo_verb=False,
                      wo_obj=False,
                      use_same_refpoint=False,
                      dec_layers=6,
                      )

    model = _make_sovstg(backbone_name="resnet101",
                         dec_layers=6,
                         dilation=False,
                         num_classes=81,
                         num_verb_classes=29,
                         mask=False,
                         num_feature_levels=4,
                         args=args
                        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessHOI(subject_category_id=0)
    return model



def ImageProcessor(image, device="cpu", return_orig=True):
    """
    transforms the single image to the format that the model expects
    Args:
        image: image path or image PIL object
        device: device to use

    Returns:
        output: image tensor
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise Exception("image should be a PIL image or a path to an image file. ")
    
    img_w, img_h = image.size
    size = torch.tensor([img_h, img_w], dtype=torch.float32).unsqueeze(0).to(device)
    transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]) 
    img = transform(image).to(device)
    if return_orig:
        return img, size, image
    return img, size


def ResultParser(result: dict, thresholds=0.1):
    """
    a single image result parser
    without nms
    """

    coco_class_dict = {
        1: 'person',2: 'bicycle',3: 'car',4: 'motorcycle',5: 'airplane',6: 'bus',7: 'train',8: 'truck',9: 'boat',10: 'traffic light',
        11: 'fire hydrant',13: 'stop sign',14: 'parking meter',15: 'bench',16: 'bird',17: 'cat',18: 'dog',19: 'horse',20: 'sheep',
        21: 'cow',22: 'elephant',23: 'bear',24: 'zebra',25: 'giraffe',27: 'backpack',28: 'umbrella',31: 'handbag',32: 'tie',33: 'suitcase',
        34: 'frisbee',35: 'skis',36: 'snowboard',37: 'sports ball',38: 'kite',39: 'baseball bat',40: 'baseball glove',41: 'skateboard',42: 'surfboard',
        43: 'tennis racket',44: 'bottle',46: 'wine glass',47: 'cup',48: 'fork',49: 'knife',50: 'spoon',51: 'bowl',52: 'banana',53: 'apple',
        54: 'sandwich',55: 'orange',56: 'broccoli',57: 'carrot',58: 'hot dog',59: 'pizza',60: 'donut',61: 'cake',62: 'chair',63: 'couch',
        64: 'potted plant',65: 'bed',67: 'dining table',70: 'toilet',72: 'tv',73: 'laptop',74: 'mouse',75: 'remote',76: 'keyboard',77: 'cell phone',
        78: 'microwave',79: 'oven',80: 'toaster',81: 'sink',82: 'refrigerator',84: 'book',85: 'clock',86: 'vase',87: 'scissors',88: 'teddy bear',
        89: 'hair drier',90: 'toothbrush'
    }

    hico_verb_dict = {
        1: 'adjust', 2: 'assemble', 3: 'block',4: 'blow',5: 'board',6: 'break',7: 'brush_with',8: 'buy',9: 'carry',
        10: 'catch',11: 'chase',12: 'check',13: 'clean',14: 'control',15: 'cook',16: 'cut',17: 'cut_with',18: 'direct',19: 'drag',
        20: 'dribble',21: 'drink_with',22: 'drive',23: 'dry',24: 'eat',25: 'eat_at',26: 'exit',27: 'feed',28: 'fill',29: 'flip',
        30: 'flush',31: 'fly',32: 'greet',33: 'grind',34: 'groom',35: 'herd',36: 'hit',37: 'hold',38: 'hop_on',39: 'hose',
        40: 'hug',41: 'hunt',42: 'inspect',43: 'install',44: 'jump',45: 'kick',46: 'kiss',47: 'lasso',48: 'launch',49: 'lick',
        50: 'lie_on',51: 'lift',52: 'light',53: 'load',54: 'lose',55: 'make',56: 'milk',57: 'move',58: 'no_interaction',59: 'open',
        60: 'operate',61: 'pack',62: 'paint',63: 'park',64: 'pay',65: 'peel',66: 'pet',67: 'pick',68: 'pick_up',69: 'point',
        70: 'pour',71: 'pull',72: 'push',73: 'race',74: 'read',75: 'release',76: 'repair',77: 'ride',78: 'row',79: 'run',
        80: 'sail',81: 'scratch',82: 'serve',83: 'set',84: 'shear',85: 'sign',86: 'sip',87: 'sit_at',88: 'sit_on',89: 'slide',
        90: 'smell',91: 'spin',92: 'squeeze',93: 'stab',94: 'stand_on',95: 'stand_under',96: 'stick',97: 'stir',98: 'stop_at',99: 'straddle',
        100: 'swing',101: 'tag',102: 'talk_on',103: 'teach',104: 'text_on',105: 'throw',106: 'tie',107: 'toast',108: 'train',109: 'turn',
        110: 'type_on',111: 'walk',112: 'wash',113: 'watch',114: 'wave',115: 'wear',116: 'wield',117: 'zip'
    }
    coco_class_list = list(coco_class_dict.values())
    hico_verb_list = list(hico_verb_dict.values())


    num_queries = result['labels'].shape[0] // 2
    subject_label = result['labels'][:num_queries]
    object_label = result['labels'][num_queries:]

    subject_box = result['boxes'][:num_queries]
    object_box = result['boxes'][num_queries:]

    verb_scores = result['verb_scores']
    indices = [torch.where(score > thresholds)[0] for score in verb_scores]
    output = []
    for sl, ol, sb, ob, vl in zip(subject_label, object_label, subject_box, object_box, indices):
        if vl.shape[0] == 0:
            continue
        for v in vl:
            if v.item() == hico_verb_list.index('no_interaction'):
                continue
            output.append({'subject_id': coco_class_list[sl.item()],
                        'object_id': coco_class_list[ol.item()],
                        'verb_id': hico_verb_list[v.item()],
                        'subject_box': sb,
                        'object_box': ob})

    return output



def draw_boxes(image, data: dict):
    """
    single image visualization
    """

    if isinstance(image, str):  
        output_image = cv2.imread(image)
    elif isinstance(image, Image.Image):  
        output_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:  
        output_image = image.copy()

    output = []
    for item in data:
        copy_img = output_image.copy()

        subject_id = item['subject_id']
        object_id = item['object_id']
        verb_id = item['verb_id']
        subject_box = item['subject_box']
        object_box = item['object_box']


        subject_box = subject_box.numpy().astype(int)
        object_box = object_box.numpy().astype(int)


        cv2.rectangle(copy_img, (subject_box[0], subject_box[1]), 
                      (subject_box[2], subject_box[3]), (0, 255, 0), 2)

        cv2.rectangle(copy_img, (object_box[0], object_box[1]), 
                      (object_box[2], object_box[3]), (0, 0, 255), 2)

        cv2.putText(copy_img, subject_id, 
                    (subject_box[0], subject_box[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(copy_img, object_id, 
                    (object_box[0], object_box[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,12,255), 2)
        cv2.putText(copy_img, verb_id, 
                    (0, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if isinstance(image, Image.Image):
            output.append(Image.fromarray(cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)))
        else:
            output.append(copy_img)

    return output
