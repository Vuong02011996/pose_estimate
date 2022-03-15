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

"""
    python demo/top_down_img_demo.py \
    /media/vuong/AI1/Github_REF/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py \
    /media/vuong/AI1/My_Github/pose_estimate/Mmpose/Models/alexnet_coco_256x192-a7b1fd15_20200727.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
    """

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

    # person_results = [{'bbox': [38.08, 110.95, 174.71, 174.71]}, {'bbox': [257.76, 139.06, 140.05, 154.21]},
    #                   {'bbox': [275.17, 126.5, 10.69, 68.26]}]
    # print(person_results)

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


def main():
    image_name = path_image_test + "000000196141.jpg"
    image_rgb = cv2.imread(image_name)

    # test a single image, with a list of bboxes
    start_time = time.time()
    # person_results = [{'bbox': [280.79, 44.73, 218.7, 346.68]}]
    person_results = [{'bbox': [247.76, 74.23, 169.67, 300.78]}, {'bbox': [555.57, 99.84, 48.32, 113.05]}, {
        'bbox': [440.85, 73.13, 16.79, 32.45]}, {'bbox': [453.77, 206.81, 177.23, 210.87]}, {
                       'bbox': [36.12, 67.59, 30.41, 96.08]}]
    print(person_results)

    dataset_name = pose_model.cfg.data['test']['type']
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_results,
        bbox_thr=None,
        format='xywh',
        dataset=dataset_name,
        return_heatmap=False,
        outputs=None)
    print(" image {}, cost{} :".format(image_name, time.time() - start_time))

    # frame_show = draw_det_when_track(image_rgb, [[280.79, 44.73, 218.7, 346.68]])
    # Format boxes is xywh => draw cmmm
    # boxes = [[247.76, 74.23, 169.67, 300.78], [555.57, 99.84, 48.32, 113.05], [440.85, 73.13, 16.79, 32.45], [453.77, 206.81, 177.23, 210.87], [36.12, 67.59, 30.41, 96.08]]
    # color = (0, 255, 255)
    # for b in boxes:
    #     xmin, ymin, xmax, ymax = list(map(int, b))
    #     cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color, 2)
    frame_show = draw_det_when_track(image_rgb, [pose_results[0]["bbox"]])
    # frame_show = draw_det_when_track(frame_show, [pose_results[1]["bbox"]])
    # frame_show = draw_det_when_track(frame_show, [pose_results[2]["bbox"]])
    # frame_show = draw_det_when_track(frame_show, [pose_results[3]["bbox"]])
    # frame_show = draw_det_when_track(frame_show, [pose_results[4]["bbox"]])
    # frame_show = draw_single_pose(frame_show, pose_results[0]["keypoints"], joint_format='coco')
    key_points_pose = np.delete(pose_results[0]["keypoints"], [1, 2, 3, 4], axis=0)
    frame_show = draw_single_pose(frame_show, key_points_pose, joint_format='coco')
    """
                "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
             "keypoints": [
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
            """

    cv2.imshow('test', cv2.resize(frame_show, (1000, 800)))
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyWindow('test')


if __name__ == '__main__':
    main()
