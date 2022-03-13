import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import requests
from skimage import transform as trans

from app.mongo_dal.camera_dal import CameraDAL
from app.mongo_dal.process_dal import ProcessDAL

process_dal = ProcessDAL()
camera_dal = CameraDAL()


def convert_base64_to_array(img_base64):
    """ Convert the base64 string image to numpy array """
    base64_img_bytes = img_base64.encode("utf-8")
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    return cv2.imdecode(np.frombuffer(decoded_image_data, dtype="uint8"), 1)


def convert_np_array_to_base64(image):
    """

    :param image: np array image
    :return: string image base64
    """
    success, encoded_image = cv2.imencode('.png', image)
    image_face = encoded_image.tobytes()
    image_base64 = base64.b64encode(image_face).decode('ascii')
    return image_base64


def file2base64(path):
    """
    :param path: path image
    :return: string base64 image
    """
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def get_url_as_base64(url):
    """
    :param url: url image
    :return: string base64 image
    """
    img = base64.b64encode(requests.get(url).content).decode('ascii')
    return img


def read_img_minio_to_array(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    return img


def read_stream_img(url, timeout=0.6):
    """
    :param url: url in minio
    :param timeout:
    :return: numpy array
    """
    i = 0
    response = None

    while i < 5:
        try:
            i += 1
            response = requests.get(url, timeout=timeout)
        except Exception as e:
            print(e)
            print("Error read_stream_img")

        if response is not None:
            break

    if response is None:
        return response

    return np.array(Image.open(BytesIO(response.content)))


def align_face(img, bbox, landmark):
    """Align the face based on the landmark and bounding boxes

    Args:
        img (np.array): raw image(image array)
        bbox (list): bounding box list [7, 136, 254, 435]
        landmark (): [[ 73.44393 317.95898]
                      [181.28801 319.50546]
                      [124.91697 407.58472]
                      [ 98.57333 412.7461 ]
                      [168.9716  417.21915]]

    Returns:
        warped (np.array): aligned face with the shape of (112,112)
    """
    image_size = [112, 112]

    M = None
    if landmark is not None:
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
    # M: The translation, rotation, and scaling matrix.
    if M is None:
        det = bbox
        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1] : bb[3], bb[0] : bb[2], :]

        ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:
        # do align using landmark
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def read_image_url_reshape_and_to_base64(url):
    """
    :param url: url image
    :return: string image base64 with shape (112, 112)
    """
    img_array = read_stream_img(url)
    # filter identity
    # https://stackoverflow.com/questions/63592160/preprocessing-methods-for-face-recognition-in-python
    # https://github.com/1adrianb/face-alignment
    # https://link.springer.com/article/10.1007/s11554-021-01107-w
    h, w, _ = img_array.shape
    print("h, w: ", h, w)
    if img_array is not None and w > 50 and h > 50:
        image_face = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_AREA)
        image_face = cv2.cvtColor(image_face, cv2.COLOR_BGR2RGB)
        face_base64 = convert_np_array_to_base64(image_face)
        return face_base64
    else:
        print("*******************************Filter url********************************************")
        return None


def get_config_from_process_name(process_name, width, height):
    process = process_dal.find_camera_by_process_name(process_name)[0]
    job_process = process["job_process"]
    camera_id = process["camera"]
    item_camera = camera_dal.find_by_id(camera_id)
    jobs_cam = item_camera['jobs_cam']
    config = jobs_cam[job_process]

    if job_process == "safe_area_regions":
        coordinates = config[0]["coordinates"]
    else:
        coordinates = config["coordinates"]
        from_time = config["from_time"]
        to_time = config["to_time"]

    # resize coordinates
    if len(coordinates) == 0:
        coordinates = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    else:
        for i in range(len(coordinates)):
            coordinates[i][0] *= width
            coordinates[i][1] *= height
        coordinates = np.asarray(coordinates, dtype=int)

    return coordinates


if __name__ == '__main__':
    origin_url = 'https://minio.core.greenlabs.ai/local/avatar/awazzmu27uf8is4jiiub2dmxcxjyx9.jpg'
    img_base64 = get_url_as_base64(url=origin_url)
    print(img_base64)