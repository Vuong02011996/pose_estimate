import time
import cv2
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


# path_config = "/storages/data/github_ref/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
from main_utils.draw import draw_det_when_track, draw_single_pose

path_config = "/media/vuong/AI1/Github_REF/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
# path_model = "/storages/data/DATA/openposedata/Models/"
path_model = "/media/vuong/AI1/My_Github/pose_estimate/Mmpose/Models/"
# path_image_test = "/storages/data/My_Github/pose_estimate/images_check/"
path_image_test = "/media/vuong/AI1/My_Github/pose_estimate/images_check/"


class InfoInput(object):
    def __init__(self):
        self.pose_config = path_config + "alexnet_coco_256x192.py"
        self.pose_checkpoint = path_model + "alexnet_coco_256x192-a7b1fd15_20200727.pth"
        self.device = 'cuda:0'


args = InfoInput()
pose_model = init_pose_model(
    args.pose_config, args.pose_checkpoint, device=args.device.lower())


def mmpose_inference(boxes, image):
    person_results = []
    for box in boxes:
        person_results.append({'bbox': box})

    dataset_name = pose_model.cfg.data['test']['type']

    start_time = time.time()
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img_or_path=image,
        person_results=person_results,
        bbox_thr=None,
        # format='xywh',
        format='xyxy',
        dataset=dataset_name,
        return_heatmap=False,
        outputs=None)
    print("cost{} :".format(time.time() - start_time))
    return pose_results
