import time
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


class InfoInput(object):
    def __init__(self):
        # self.pose_config = "/storages/data/github_ref/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
        self.pose_config = "/storages/data/github_ref/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py"
        # self.pose_checkpoint = "/home/gg-greenlab/.cache/torch/hub/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        self.pose_checkpoint = "/storages/data/DATA/openposedata/Models/alexnet_coco_256x192-a7b1fd15_20200727.pth"
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
        format='xywh',
        dataset=dataset_name,
        return_heatmap=False,
        outputs=None)
    print("cost{} :".format(time.time() - start_time))
    return pose_results


def main():
    image_name = "/storages/data/My_Github/pose_estimate/images_check/000000040083.jpg"

    # test a single image, with a list of bboxes
    start_time = time.time()
    person_results = [{'bbox': [38.08, 110.95, 174.71, 174.71]}, {'bbox': [257.76, 139.06, 140.05, 154.21]},
     {'bbox': [275.17, 126.5, 10.69, 68.26]}]
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


if __name__ == '__main__':
    main()
