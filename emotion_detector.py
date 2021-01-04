import os
import glob
import json
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import shutil

from tqdm import tqdm
import pandas as pd
from torchvision.transforms import transforms
from models import densenet121, resmasking_dropout1
from facenet_pytorch import MTCNN


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

class MTCNNFaceDetector():
    def __init__(self):
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        self.model = MTCNN(margin=20, select_largest=True, device=device)
        print(f">> Loaded MTCNN on {self.model.device}")

    def detect_faces(self, frame, threshold):
        """Returns faces bounding boxes"""
        tlbr_to_tlwh = lambda f: (f[0], f[1], f[2] - f[0], f[3] - f[1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = Image.fromarray(rgb_frame)

        boxes, probabilities = self.model.detect(rgb_frame)
        if boxes is not None:
            boxes = [box for box, prob in zip(boxes.astype(int), probabilities) if prob >= threshold/100]
            coord = [tlbr_to_tlwh(box) for box in boxes]
            return coord
        else:
            return []

class HaarFaceDetector():
    def __init__(self):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

        self.model = cv2.CascadeClassifier(haar_model)
        print(f">> Loaded Haar face detector")

    def detect_faces(self, frame, threshold):
        """Returns faces bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.model.detectMultiScale(gray, 1.3, 5)

        return faces 

class NNetFaceDetector():
    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel"
        )
        print(f">> Loaded NNet face detector")

    def detect_faces(self, frame, threshold):
        """Returns faces bounding boxes"""
        frame = frame.astype(np.uint8)
        # frame += 50
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = frame

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.model.setInput(blob)
        faces = self.model.forward()
        fixed_faces = []
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence < threshold/100:
                continue
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            # convert to square images
            center_x, center_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            square_length = ((end_x - start_x) + (end_y - start_y)) // 2 // 2

            square_length *= 1.1

            start_x = int(center_x - square_length)
            start_y = int(center_y - square_length)
            end_x = int(center_x + square_length)
            end_y = int(center_y + square_length)

            fixed_faces.append([start_x, start_y, end_x-start_x, end_y-start_y])

        return fixed_faces

def load_face_detector(face_detector):
    if face_detector == 'haar':
        return HaarFaceDetector()
    elif face_detector == 'mtcnn':
        return MTCNNFaceDetector()
    elif face_detector == 'nnet':
        return NNetFaceDetector()

    else:
        raise Exception(f"detection method {face_detector} is invalid, expected one of [haar, cnn, nnet]")

def collect_frames(video_file, frame_limit):
    """Collects frames from the video file"""
    frames = []
    i = 0
    cap = cv2.VideoCapture(video_file)
    while i < frame_limit and cap.isOpened():
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Detect emotions in a video")
    parser.add_argument(
            "--output_folder",
            default="tmp_frames",
            type=str,
            help="the folder to output the tracking predictions [default=tmp_frames]",
    )
    parser.add_argument(
            "--frame_limit",
            default=2**16,
            type=int,
            help="the number of frames to use [default=all]",
    )
    parser.add_argument(
            "--use_webcam",
            action="store_true",
            help="whether or not to use the webcam instead of the input video [default=False]",
    )
    parser.add_argument(
            "--video_file",
            default="ur.mp4",
            type=str,
            help="the video file to generate the dataset from [default=ur.mp4]",
    )
    parser.add_argument(
            "--draw_boxes",
            action="store_true",
            help="whether or not to draw boxes on the frames [default=False]",
    )
    parser.add_argument(
            "--output_video",
            default="output.mp4",
            type=str,
            help="the location of the video [default=output.mp4]",
    )
    parser.add_argument(
            "--face_detector",
            default="haar",
            type=str,
            help="face detector to use: haar or nnet (faster) or mtcnn (more precise) [default=haar]",
    )
    parser.add_argument(
            "--scores_file",
            default="scores.csv",
            type=str,
            help="the location of the scores file for tracking [default=scores.csv]",
    )
    parser.add_argument(
            "--track_boxes",
            action="store_true",
            help="whether or not to track boxes on the frames [default=False]",
    )
    parser.add_argument(
            "--threshold",
            default=98,
            type=int,
            help="face detection threshold for mtcnn [default=98]",
    )
    return parser.parse_args()



def load_model():
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    state = torch.load("./saved/checkpoints/Z_resmasking_dropout1_rot30_2019Nov30_13.32")
    model.load_state_dict(state["net"])
    model.eval()

    return model

def match_faces_bodies(frame, boxes, detector, threshold):
    """Matches faces with the bodies for the given frame"""
    df_faces = []
    for _, box in boxes.iterrows():
        roi = frame[int(box.y):int(box.y2), int(box.x):int(box.x2)]
        if roi.size > 0:
            faces = detector.detect_faces(roi, threshold)
            if faces:
                x, y, w, h = faces[0]
                    
                df_faces.append({
                    'frame_id': int(box.abs_frame_id),
                    'box_id': int(box.box_id),
                    'x': int(box.x) + x,
                    'y': int(box.y) + y,
                    'x2': int(box.x) + x + w,
                    'y2': int(box.y) + y + h,
                })

    return df_faces

def detect_emotions(model, img, faces, image_size, transform, draw_boxes, track_boxes):
    emotions = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }
    df_faces = []
    img = img.astype(np.uint8)
    out_img = img.copy()
    for face in faces:
        if track_boxes:
            x, y, w, h = face['x'], face['y'], face['x2']-face['x'], face['y2']-face['y']
        else:
            x, y, w, h = face
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_face = gray[int(y):int(y+h), int(x):int(x+w)]

        detected_face = ensure_color(detected_face)
        
        detected_face = cv2.resize(detected_face, image_size)
        detected_face = transform(detected_face).cuda()
        detected_face = torch.unsqueeze(detected_face, dim=0)

        output = torch.squeeze(model(detected_face), 0)
        proba = torch.softmax(output, 0)

        score, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        score = score.item()

        emotion = emotions[emo_idx]

        label_size, base_line = cv2.getTextSize(
            "{}: 000".format(emotion), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        if draw_boxes:
            out_img = cv2.rectangle(out_img, (x, y), (x+w, y+h), (179, 255, 179), 2)

            out_img = cv2.rectangle(
                        out_img,
                        (x, y - label_size[1]),
                        (x + label_size[0], y + base_line),
                        (223, 128, 255),
                        cv2.FILLED,
                    )
            out_img = cv2.putText(
                        out_img,
                        "{} {}".format(emotion, int(score * 100)),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )
        if track_boxes:
            face.update({"emotion":{emotion:score}})
            df_faces.append(face)

    return out_img, df_faces


def main():
    args = parse_args()
    FRAME_LIMIT = args.frame_limit
    VIDEO_FILE = args.video_file
    DRAW_BOXES = args.draw_boxes
    SCORES_FILE = args.scores_file
    FOLDER = args.output_folder
    OUTPUT_VIDEO = args.output_video
    FACE_DETECTOR = args.face_detector
    USE_WEBCAM = args.use_webcam
    TRACK_BOXES = args.track_boxes
    THRESHOLD = args.threshold

    # load configs and set random seed
    configs = json.load(open("./configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    detector = load_face_detector(FACE_DETECTOR)

    model = load_model()

    vid = cv2.VideoCapture(0)

    with torch.no_grad():
        if USE_WEBCAM:
            while True:
                ret, img = vid.read()
                if img is None or ret is not True:
                    continue

                faces = detector.detect_faces(img, THRESHOLD)
                img, _ = detect_emotions(model, img, faces, image_size, transform, DRAW_BOXES, TRACK_BOXES)

                cv2.imshow("img", img)
                if cv2.waitKey(1) == ord("q"):
                    break

            cv2.destroyAllWindows()
        else:
            print(f">> Reading video from {VIDEO_FILE}")
            frames = collect_frames(VIDEO_FILE, FRAME_LIMIT)

            if TRACK_BOXES:
                df = pd.read_csv(SCORES_FILE, header=None, index_col=None, names=[
                "track_id", "frame_id", "box_id", "x", "y", "x2", "y2"] + list(range(80)))
                df["abs_frame_id"] = df.frame_id + df.track_id - 128
                df_faces = []

                print(">> Detecting emotions")
                for i, frame in tqdm(enumerate(frames), total=len(frames)):
                    faces = match_faces_bodies(
                            frame,
                            df[df.abs_frame_id == i],
                            detector=detector,
                            threshold=THRESHOLD
                    )
                    frames[i], new_df_faces = detect_emotions(model, frame, faces, image_size, transform, DRAW_BOXES, TRACK_BOXES)

                    df_faces += new_df_faces

                df_faces = pd.DataFrame(df_faces)
                print(f">> Saving box files at {FOLDER}")

                if os.path.exists(FOLDER):
                    shutil.rmtree(FOLDER)
                os.mkdir(FOLDER)
                box_ids = df_faces.box_id.unique()
                for box_id in tqdm(box_ids, total=len(box_ids)):
                    with open(os.path.join(FOLDER, f"person{box_id}.txt"), "w") as f:
                        for _, frame in df_faces[df_faces.box_id == box_id].iterrows():
                            bbox = str([frame.x, frame.y, frame.x2, frame.y2]).replace(" ", "").strip("[]")
                            emotion = list(frame.emotion.keys())[0]
                            f.write(f"{frame.frame_id},{bbox},{emotion},{frame.emotion[emotion]}\n")

            else:
                print(">> Detecting emotions")
                for i, frame in tqdm(enumerate(frames), total=len(frames)):
                    faces = detector.detect_faces(frame, THRESHOLD)
                    frames[i], _ = detect_emotions(model, frame, faces, image_size, transform, DRAW_BOXES, TRACK_BOXES)

            print(f">> Creating video at {OUTPUT_VIDEO}")
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height), True)
            for frame in tqdm(frames, total=len(frames)):
                out.write(frame)
            cv2.destroyAllWindows()
            out.release()

if __name__ == "__main__":
    main()
