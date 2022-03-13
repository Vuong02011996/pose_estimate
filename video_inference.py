import cv2
from queue import Queue
import time
from kthread import KThread
import numpy as np
import os

from Mmpose.inference_image_mmpose import mmpose_inference
from mot_tracking.mot_sort_tracker import Sort
from mot_tracking import untils_track
from collections import deque

import torch
from AlphaPoseEstimate.PoseEstimateLoader import SPPE_FastPose
from Actionsrecognition.ActionsEstLoader import TSSTG
import sys

from main_utils.draw import draw_boxes_tracking, draw_det_when_track, draw_single_pose, draw_data_action, \
    draw_region

sys.path.append("Yolov5_detect_person")
from Yolov5_detect_person.yolov5_detect_image import Y5Detect


# y5_model = Y5Detect(weights="core/main/fall_detection/model_yolov5/yolov5s.pt")
y5_model = Y5Detect(weights="Yolov5_detect_person/model_yolov5/yolov5s.pt")

class_names = y5_model.class_names
mot_tracker = Sort(class_names)

inp_pose = (224, 160)
pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device="cuda")

# Actions Estimate.
action_model = TSSTG()


class InfoCam(object):
    def __init__(self, cam_name):
        self.cap = cv2.VideoCapture(cam_name)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frame_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = 0
        # self.region_track = [coordinates[0], coordinates[2], coordinates[3], coordinates[1]]
        self.region_track = np.array([[2043, 224], [self.width, 224], [self.width, 609], [2043, 526]])
        self.frame_step_after_track = 0
        self.show_all = False


def video_capture(cam, frame_detect_queue):
    frame_count = 0

    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    while cam.cap.isOpened():
        ret, frame_ori = cam.cap.read()
        # time.sleep(0.01)
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)
        frame_detect_queue.put([image_rgb, frame_count])
        print("frame_count: ", frame_count)
        frame_count += 1

    cam.cap.release()


def inference(cam, frame_detect_queue, detections_queue): #, tracking_queue):
    while cam.cap.isOpened():
        image_rgb, frame_count = frame_detect_queue.get()
        boxes, labels, scores, detections_sort = y5_model.predict_sort(image_rgb, label_select=["person"])
        # for i in range(len(scores)):
        #     detections_tracking = bboxes[i].append(scores[i])
        detections_queue.put([boxes, labels, scores, image_rgb, detections_sort, frame_count])
        # tracking_queue.put([detections_tracking])

    cam.cap.release()


def tracking(cam, detections_queue, pose_queue):
    """
    :param cam:
    :param pose_queue:
    :param detections_queue:
    :param draw_queue:
    :return:
    Tracking using SORT. Hungary + Kalman Filter.
    Using mot_tracker.update()
    Input: detections [[x1,y1,x2,y2,score,label],[x1,y1,x2,y2,score, label],...], use np.empty((0, 5)) for frames without detections
    Output: [[x1,y1,x2,y2,id1, label],[x1,y1,x2,y2,id2, label],...]
    """

    while cam.cap.isOpened():
        boxes, labels, scores, image_rgb, detections_sort, frame_count = detections_queue.get()
        if len(boxes) == 0:
            detections = np.empty((0, 6))
        else:
            detections = detections_sort
            # check and select the detection is inside region tracking
            detections, list_idx_bbox_del = untils_track.select_bbox_inside_polygon(detections, cam.region_track)

        if cam.frame_step_after_track != 0 and frame_count % cam.frame_step_after_track != 0:
            continue

        cam.frame_step_after_track += 1
        track_bbs_ids, unm_trk_ext = mot_tracker.update(detections, image=image_rgb)
        # print("labels, scores", labels, scores)
        # print(track_bbs_ids)
        if len(track_bbs_ids) > 0:
            a = 0
        pose_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count])

    cam.cap.release()


def pose_estimate(cam, pose_queue, action_data_queue):
    data_action_rec = []
    key_points_pose = []
    pre_track_id = []

    while cam.cap.isOpened():
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count = pose_queue.get()
        if len(track_bbs_ids) > 0:
            pose_results = mmpose_inference(track_bbs_ids[:, 0:4], image_rgb)
            boxes_detect_pose = torch.as_tensor(track_bbs_ids[:, 0:4])
            scores = torch.as_tensor(np.ones(len(track_bbs_ids)))
            start_time = time.time()
            poses = pose_model.predict(image_rgb, boxes_detect_pose, scores)
            print("pose_estimate cost: ", time.time() - start_time)
            key_points_pose = [np.concatenate((ps['keypoints'].numpy(), ps['kp_score'].numpy()), axis=1) for ps in poses]
            # print("key_points_pose: ", key_points_pose)

            current_track_id = track_bbs_ids[:, -1]
            if len(data_action_rec) > 0:
                pre_track_id = list(map(lambda d: d['track_id'], data_action_rec))
            # Delete pre_track_id not in current_track_id, track is deleted.
            track_id_delete = np.setdiff1d(pre_track_id, current_track_id)
            if len(track_id_delete) > 0:
                for track_id in track_id_delete:
                    index_del = pre_track_id.index(track_id)
                    del data_action_rec[index_del]
                    pre_track_id.remove(track_id)
                    a = 0

            if len(key_points_pose) == len(current_track_id):
                for i in range(len(current_track_id)):
                    key_points = key_points_pose[i]
                    if current_track_id[i] not in pre_track_id:
                        # Create new track
                        key_points_list = deque(maxlen=30)
                        key_points_list.append(key_points)
                        data_action_rec.append({
                            "track_id": current_track_id[i],
                            "key_points": key_points_list
                        })
                    else:
                        idx_pre_track = pre_track_id.index(current_track_id[i])
                        data_action_rec[idx_pre_track]["key_points"].append(key_points)
                        # Update key points for track
            else:
                print("len(key_points_pose) != len(current_track_id)")

        action_data_queue.put([data_action_rec, image_rgb, track_bbs_ids, boxes, labels, scores, unm_trk_ext, frame_count, key_points_pose])
    cam.cap.release()


def action_recognition(cam, action_data_queue, draw_queue):
    while cam.cap.isOpened():
        data_action_rec, image_rgb, track_bbs_ids, boxes, labels, scores, unm_trk_ext, frame_count, key_points_pose = action_data_queue.get()
        data_action = data_action_rec.copy()
        # print("data_action_rec", len(data_action_rec))
        # print("data_action_rec", len(data_action_rec[0]["key_points"]))
        # print("len(data_action_rec)", len(data_action))
        for i in range(len(data_action)):
            if len(data_action[i]["key_points"]) == 30:
                """pts (30, 13, 3)"""
                pts = np.array(data_action[i]["key_points"], dtype=np.float32)
                out = action_model.predict(pts, image_rgb.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                # print(action_name)
                # action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                action_name_data = {'action_name': action_name}
                data_action[i].update(action_name_data)
                a = 0
        draw_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count, data_action, key_points_pose])
    cam.cap.release()


def drawing(cam, draw_queue, frame_final_queue, show_det=True):
    while cam.cap.isOpened():
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count, data_action, key_points_pose = draw_queue.get()
        frame_show = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if frame_show is not None:
            frame_show = draw_region(frame_show, cam.region_track)
            frame_show = draw_data_action(frame_show, track_bbs_ids, track_bbs_ext=unm_trk_ext, data_action=data_action)

            for i in range(len(key_points_pose)):
                draw_single_pose(frame_show, key_points_pose[i], joint_format='coco')

            if show_det:
                frame_show = draw_det_when_track(frame_show, boxes, scores=scores, labels=labels,
                                            class_names=class_names)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if frame_final_queue.full() is False:
                frame_final_queue.put([frame_show, frame_count])
            else:
                time.sleep(0.001)
    cam.cap.release()


def main(input_path, thread_fall_down_manager, cv2_show=True):
    start_time = time.time()
    frame_detect_queue = Queue(maxsize=1)
    pose_queue = Queue(maxsize=1)
    action_data_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    draw_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)

    cam = InfoCam(input_path)

    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue))
    thread2 = KThread(target=inference, args=(cam, frame_detect_queue, detections_queue))
    thread3 = KThread(target=tracking, args=(cam, detections_queue, pose_queue))
    thread4 = KThread(target=pose_estimate, args=(cam, pose_queue, action_data_queue))
    thread5 = KThread(target=action_recognition, args=(cam, action_data_queue, draw_queue))
    thread6 = KThread(target=drawing, args=(cam, draw_queue, frame_final_queue))

    thread1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread1.start()
    thread_fall_down_manager.append(thread1)
    thread2.daemon = True
    thread2.start()
    thread_fall_down_manager.append(thread2)
    thread3.daemon = True
    thread3.start()
    thread_fall_down_manager.append(thread3)
    thread4.daemon = True
    thread4.start()
    thread_fall_down_manager.append(thread4)
    thread5.daemon = True
    thread5.start()
    thread_fall_down_manager.append(thread5)
    thread6.daemon = True
    thread6.start()
    thread_fall_down_manager.append(thread6)

    while cam.cap.isOpened():
        image, frame_count = frame_final_queue.get()
        image = cv2.resize(image, (1000, 800))

        if cv2_show:
            cv2.imshow('output_fall_down', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    total_time = time.time() - start_time
    print("FPS video: ", cam.fps_video)
    print("Total time: {}, Total frame: {}, FPS all process : {}".format(total_time, cam.total_frame_video,
                                                                         1 / (total_time / cam.total_frame_video)), )

    for t in thread_fall_down_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_path = "/storages/data/DATA/Clover_data/Video_Test/te_nga.mp4"
    thread_fall_down_manager = []
    main(input_path, thread_fall_down_manager, cv2_show=True)


