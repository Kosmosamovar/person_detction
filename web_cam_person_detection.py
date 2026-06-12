import cv2
import numpy as np
import requests
import json
import time
import os
import sys
import logging
import argparse
import subprocess
import torch
import urllib.request
import hashlib
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

###############################################
# Utility / validation helpers (added)
###############################################
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = getattr(sys, '_MEIPASS', APP_DIR)

LOG_FILE = os.path.join(APP_DIR, 'app.log')
try:
    with open(LOG_FILE, 'a', encoding='utf-8') as bootstrap_log:
        bootstrap_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] bootstrap start\n")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


def resolve_existing_path(path_value: str) -> str:
    """Resolve path from app dir/cwd/bundled dir for exe and script runs."""
    if os.path.isabs(path_value):
        return path_value

    candidates = [
        os.path.join(APP_DIR, path_value),
        os.path.join(os.getcwd(), path_value),
        os.path.join(BUNDLE_DIR, path_value),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(APP_DIR, path_value)


def sha256sum(path: str) -> str:
    """Compute SHA256 of a file (small helper; resilient to large files)."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return ''


# === Загрузка конфига ===
CONFIG_PATH = resolve_existing_path('config.json')
with open(CONFIG_PATH) as f:
    config = json.load(f)

TELEGRAM_TOKEN = config.get('telegram_token')
CHAT_ID = config.get('telegram_chat_id')
MODEL_TYPE = str(config.get('model', 'ssd')).lower()  # 'detr' | 'ssd' | 'ultralytics_pt'
OUTPUT_MODE = config.get('output_mode', 'telegram')  # 'telegram' | 'local'
LOCAL_DIR = config.get('local_save_dir', 'captures')
COOLDOWN = config.get('cooldown', 10)
CONFIDENCE_THRESHOLD = float(config.get('confidence', 0.5))
FRAME_SKIP = int(config.get('frame_skip', 3))
CAPTURE_MODE = config.get('capture_mode', 'video').lower()  # 'photo' | 'video'
PREFER_LIGHT_MODEL = bool(config.get('prefer_light_model', True))
ABSENCE_RESET_SECONDS = float(config.get('absence_reset_seconds', 1.0))
VIDEO_SECONDS_AFTER_LOST = float(config.get('video_seconds_after_lost', 2.0))
VIDEO_MAX_SECONDS = float(config.get('video_max_seconds', 20.0))
VIDEO_FPS = float(config.get('video_fps', 10.0))
VIDEO_CODEC = config.get('video_codec', 'mp4v')
YOLO_ONNX_PATH = config.get('yolo_onnx_model', 'yolov5n.onnx')
YOLO_INPUT_SIZE = int(config.get('yolo_input_size', 640))
YOLO_PT_PATH = config.get('yolo_pt_model', 'yolov8n.pt')

if not os.path.isabs(LOCAL_DIR):
    LOCAL_DIR = os.path.join(APP_DIR, LOCAL_DIR)

YOLO_ONNX_PATH = resolve_existing_path(YOLO_ONNX_PATH)
YOLO_PT_PATH = resolve_existing_path(YOLO_PT_PATH)

TARGET_CLASSES = {x.lower() for x in config.get('target_classes', [
    'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
])}

YOLO_COCO_ID_TO_NAME = {
    0: 'person', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe'
}
YOLO_TARGET_IDS = {cid for cid, name in YOLO_COCO_ID_TO_NAME.items() if name in TARGET_CLASSES}

if CAPTURE_MODE not in {'photo', 'video'}:
    CAPTURE_MODE = 'video'

if PREFER_LIGHT_MODEL and MODEL_TYPE not in {'ssd', 'ultralytics_pt'}:
    print('Light mode is enabled: forcing model="ssd".')
    MODEL_TYPE = 'ssd'

MODEL_RUNTIME = MODEL_TYPE
model = None
processor = None
DEVICE = "cpu"

os.makedirs(LOCAL_DIR, exist_ok=True)

SSD_PROTOTXT = os.path.join(APP_DIR, "MobileNetSSD_deploy.prototxt")
SSD_MODEL = resolve_existing_path("MobileNetSSD_deploy.caffemodel")
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
# Optional fallback mirrors (can add more if GitHub blocked)
PROTOTXT_FALLBACKS = [
    # Add real mirrors if needed
]
MODEL_FALLBACKS = [
    # Add real mirrors if needed
]

# Conservative minimal sizes (bytes) to treat file as plausibly valid.
MIN_PROTOTXT_SIZE = 1000
MIN_MODEL_SIZE = 1_000_000  # ~1MB safeguard; real file is larger


def _download(url: str, dst: str) -> bool:
    try:
        print(f"Downloading: {url}")
        logging.info('Downloading: %s', url)
        urllib.request.urlretrieve(url, dst)
        return True
    except Exception as e:
        print(f"Download failed from {url}: {e}")
        logging.exception('Download failed from %s', url)
        return False

def _validate_prototxt(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_PROTOTXT_SIZE:
        print("Prototxt too small – looks corrupted.")
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            head = f.read(500)
            if 'layer' not in head and 'input_shape' not in head:
                print("Prototxt missing expected keywords.")
                return False
    except Exception as e:
        print(f"Failed to read prototxt: {e}")
        return False
    return True

def _validate_model(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_MODEL_SIZE:
        print("Caffemodel too small – looks corrupted.")
        return False
    return True

def ensure_ssd_files(force_redownload: bool = False):
    """Ensure SSD model files exist & are valid; re-download if corrupted.

    Parameters
    ----------
    force_redownload: bool
        If True, re-download regardless of existing files.
    """
    need_proto = force_redownload or not _validate_prototxt(SSD_PROTOTXT)
    if need_proto:
        if os.path.exists(SSD_PROTOTXT):
            try:
                os.remove(SSD_PROTOTXT)
            except OSError:
                pass
        print("(Re)downloading prototxt...")
        if not _download(PROTOTXT_URL, SSD_PROTOTXT):
            # Try fallbacks
            for fb in PROTOTXT_FALLBACKS:
                if _download(fb, SSD_PROTOTXT):
                    break
        if not _validate_prototxt(SSD_PROTOTXT):
            raise FileNotFoundError("Failed to obtain valid MobileNetSSD_deploy.prototxt")

    need_model = force_redownload or not _validate_model(SSD_MODEL)
    if need_model:
        if os.path.exists(SSD_MODEL):
            try:
                os.remove(SSD_MODEL)
            except OSError:
                pass
        print("(Re)downloading caffemodel (this may take a while)...")
        if not _download(MODEL_URL, SSD_MODEL):
            for fb in MODEL_FALLBACKS:
                if _download(fb, SSD_MODEL):
                    break
        if not _validate_model(SSD_MODEL):
            raise FileNotFoundError("Failed to obtain valid MobileNetSSD_deploy.caffemodel")

    # Optional: display short hashes for reproducibility
    print("SSD files ready. Hashes:")
    print(f"  prototxt sha256: {sha256sum(SSD_PROTOTXT)[:16]}...")
    print(f"  caffemodel sha256: {sha256sum(SSD_MODEL)[:16]}...")


# === Отправка фото в Telegram ===
def send_photo(image_path):
    if OUTPUT_MODE != 'telegram':
        return
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram config missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        try:
            r = requests.post(url, data={'chat_id': CHAT_ID}, files={'photo': photo})
            if r.status_code == 200:
                print("Фото отправлено")
                logging.info('Photo sent: %s', image_path)
            else:
                print(f"Ошибка Telegram: {r.status_code} {r.text}")
                logging.error('Telegram photo error %s %s', r.status_code, r.text)
        except Exception as e:
            print(f"Ошибка отправки: {e}")
            logging.exception('Photo send failed: %s', image_path)


def send_video(video_path):
    if OUTPUT_MODE != 'telegram':
        return
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram config missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
    with open(video_path, 'rb') as video:
        try:
            r = requests.post(url, data={'chat_id': CHAT_ID}, files={'video': video})
            if r.status_code == 200:
                print("Видео отправлено")
                logging.info('Video sent: %s', video_path)
            else:
                print(f"Ошибка Telegram: {r.status_code} {r.text}")
                logging.error('Telegram video error %s %s', r.status_code, r.text)
        except Exception as e:
            print(f"Ошибка отправки видео: {e}")
            logging.exception('Video send failed: %s', video_path)


def initialize_model_runtime():
    global model, processor, MODEL_RUNTIME, DEVICE

    if model is not None:
        return

    DEVICE = "cuda" if (MODEL_TYPE == 'detr' and torch.cuda.is_available()) else "cpu"

    if MODEL_TYPE == 'detr':
        print("Загрузка DETR (тяжёлая модель)...")
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
        MODEL_RUNTIME = 'detr'
        return

    if MODEL_TYPE == 'ultralytics_pt':
        if YOLO is None:
            raise RuntimeError("Для model=ultralytics_pt требуется пакет ultralytics")
        pt_candidates = [YOLO_PT_PATH, 'yolov8n.pt', 'yolov5n.pt', 'yolov5s.pt']
        pt_path = next((c for c in pt_candidates if os.path.exists(c)), None)
        if not pt_path:
            raise FileNotFoundError("Не найден файл .pt для model=ultralytics_pt")
        print(f"Загрузка локальной PT модели: {pt_path}")
        model = YOLO(pt_path)
        MODEL_RUNTIME = 'ultralytics_pt'
        return

    try:
        ensure_ssd_files()
        print("Загрузка MobileNet-SSD (легче для CPU)...")
        try:
            model = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_MODEL)
        except cv2.error:
            print("Первичная загрузка не удалась. Попытка принудительного повторного скачивания...")
            ensure_ssd_files(force_redownload=True)
            model = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_MODEL)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        MODEL_RUNTIME = 'ssd'
    except Exception as e:
        print(f"Ошибка подготовки SSD файлов: {e}")
        print("SSD недоступен. Переключаюсь на локальный YOLO ONNX...")
        try:
            if not os.path.exists(YOLO_ONNX_PATH):
                raise FileNotFoundError(f"Не найден fallback-файл ONNX: {YOLO_ONNX_PATH}")
            model = cv2.dnn.readNetFromONNX(YOLO_ONNX_PATH)
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            MODEL_RUNTIME = 'yolo_onnx'
        except Exception as onnx_error:
            print(f"ONNX fallback недоступен: {onnx_error}")
            if YOLO is None:
                raise RuntimeError("Ultralytics не установлен, fallback на .pt невозможен")

            pt_candidates = [YOLO_PT_PATH, 'yolov8n.pt', 'yolov5n.pt', 'yolov5s.pt']
            pt_path = next((c for c in pt_candidates if os.path.exists(c)), None)
            if not pt_path:
                raise FileNotFoundError("Не найден fallback-файл .pt (ожидался yolov8n.pt/yolov5n.pt/yolov5s.pt)")

            print(f"Переключаюсь на локальный Ultralytics PT: {pt_path}")
            model = YOLO(pt_path)
            MODEL_RUNTIME = 'ultralytics_pt'


# === Детекция DETR ===
def detect_detr(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]]).to(DEVICE)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=CONFIDENCE_THRESHOLD
    )[0]
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()].lower()
        if label_name not in TARGET_CLASSES:
            continue
        box = box.to("cpu").numpy().astype(int)
        detections.append((box, float(score), label_name))
    return detections


# === Детекция SSD ===
def detect_ssd(frame):
    ssd_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    target_ids = {idx for idx, name in enumerate(ssd_classes) if name in TARGET_CLASSES}

    (h, w) = frame.shape[:2]
    # Уменьшаем ради скорости
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    raw_detections = model.forward()
    detections = []
    for i in range(raw_detections.shape[2]):
        conf = raw_detections[0, 0, i, 2]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        class_id = int(raw_detections[0, 0, i, 1])
        if class_id not in target_ids:
            continue
        box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype(int)
        label_name = ssd_classes[class_id]
        detections.append(((x1, y1, x2, y2), float(conf), label_name))
    return detections


def detect_yolo_onnx(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()

    # YOLOv8 ONNX often outputs [1, 84, N], convert to [1, N, 84].
    if len(outputs.shape) == 3 and outputs.shape[1] < outputs.shape[2]:
        outputs = np.transpose(outputs, (0, 2, 1))

    boxes = []
    confidences = []
    class_ids = []

    rows = outputs.shape[1]
    for i in range(rows):
        row = outputs[0][i]
        obj_conf = float(row[4])
        if obj_conf < CONFIDENCE_THRESHOLD:
            continue
        class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        cls_conf = float(class_scores[class_id])
        score = obj_conf * cls_conf
        if score < CONFIDENCE_THRESHOLD:
            continue
        if class_id not in YOLO_TARGET_IDS:
            continue

        cx, cy, bw, bh = row[0:4]
        x1 = int((cx - bw / 2) * w / YOLO_INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / YOLO_INPUT_SIZE)
        bw = int(bw * w / YOLO_INPUT_SIZE)
        bh = int(bh * h / YOLO_INPUT_SIZE)

        boxes.append([x1, y1, bw, bh])
        confidences.append(score)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.45)
    detections = []
    if len(indices) > 0:
        for idx in np.array(indices).flatten():
            x1, y1, bw, bh = boxes[int(idx)]
            x2 = x1 + bw
            y2 = y1 + bh
            label_name = YOLO_COCO_ID_TO_NAME.get(class_ids[int(idx)], str(class_ids[int(idx)]))
            detections.append(((x1, y1, x2, y2), float(confidences[int(idx)]), label_name))
    return detections


def detect_ultralytics_pt(frame):
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if not results:
        return []

    detections = []
    boxes = results[0].boxes
    if boxes is None:
        return detections

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id not in YOLO_TARGET_IDS:
            continue
        score = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        label_name = YOLO_COCO_ID_TO_NAME.get(cls_id, str(cls_id))
        detections.append(((x1, y1, x2, y2), score, label_name))

    return detections


def save_frame(frame, prefix='detected'):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOCAL_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"Сохранено: {path}")
    return path


def make_video_writer(frame, prefix='event'):
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(LOCAL_DIR, f'{prefix}_{ts}.mp4')
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError('Не удалось открыть VideoWriter для записи ролика')
    print(f'Запись видео: {path}')
    return writer, path


# === Главный цикл ===
def run_detector():
    initialize_model_runtime()
    logging.info('App started. model=%s runtime=%s output=%s capture=%s', MODEL_TYPE, MODEL_RUNTIME, OUTPUT_MODE, CAPTURE_MODE)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        logging.error('Camera open failed')
        return

    frame_count = 0
    presence_active = False
    last_detection_time = 0.0
    video_writer = None
    video_path = None
    video_started_at = 0.0

    print(f"Модель: {MODEL_TYPE} (runtime: {MODEL_RUNTIME}), режим вывода: {OUTPUT_MODE}, capture_mode: {CAPTURE_MODE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Кадр не получен")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Уменьшаем изображение для ускорения (только для обработки, рисуем на оригинале)
        proc_frame = frame

        if MODEL_RUNTIME == 'detr':
            detections = detect_detr(proc_frame)
        elif MODEL_RUNTIME == 'ultralytics_pt':
            detections = detect_ultralytics_pt(proc_frame)
        elif MODEL_RUNTIME == 'yolo_onnx':
            detections = detect_yolo_onnx(proc_frame)
        else:
            detections = detect_ssd(proc_frame)

        target_detected = len(detections) > 0
        now = time.time()

        for (box, score, label_name) in detections:
            (x1, y1, x2, y2) = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f'{label_name} {score:.2f}', (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        if target_detected:
            last_detection_time = now

        # Transition: object appeared on screen.
        if target_detected and not presence_active:
            presence_active = True
            print('Обнаружен новый вход объекта в кадр')
            if CAPTURE_MODE == 'photo':
                path = save_frame(frame, prefix='detected')
                if OUTPUT_MODE == 'telegram':
                    send_photo(path)
            else:
                if video_writer is None:
                    try:
                        video_writer, video_path = make_video_writer(frame)
                        video_started_at = now
                    except RuntimeError as e:
                        print(e)

        # Transition: object left screen for stable period.
        if (not target_detected) and presence_active and (now - last_detection_time >= ABSENCE_RESET_SECONDS):
            presence_active = False
            print('Объект вышел из кадра')

        if CAPTURE_MODE == 'video' and video_writer is not None:
            video_writer.write(frame)
            clip_too_long = (now - video_started_at) >= VIDEO_MAX_SECONDS
            no_detection_for_long = (not target_detected) and ((now - last_detection_time) >= VIDEO_SECONDS_AFTER_LOST)

            # Stop recording when scene is over or clip becomes too long.
            if clip_too_long or no_detection_for_long:
                video_writer.release()
                print(f'Видео сохранено: {video_path}')
                if OUTPUT_MODE == 'telegram' and video_path:
                    send_video(video_path)
                video_writer = None
                video_path = None

        cv2.imshow(f'Person Detection ({MODEL_RUNTIME})', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
        print(f'Видео сохранено: {video_path}')
        if OUTPUT_MODE == 'telegram' and video_path:
            send_video(video_path)

    cap.release()
    cv2.destroyAllWindows()


DEFAULT_CONFIG = {
    "telegram_token": "",
    "telegram_chat_id": "",
    "model": "ultralytics_pt",
    "prefer_light_model": True,
    "output_mode": "local",
    "capture_mode": "video",
    "local_save_dir": "captures",
    "cooldown": 10,
    "confidence": 0.5,
    "frame_skip": 3,
    "target_classes": ["person", "cat", "dog", "horse", "sheep", "cow", "bird"],
    "absence_reset_seconds": 1.0,
    "video_seconds_after_lost": 2.0,
    "video_max_seconds": 20,
    "video_fps": 10,
    "video_codec": "mp4v",
    "yolo_onnx_model": "yolov5n.onnx",
    "yolo_input_size": 640,
    "yolo_pt_model": "yolov8n.pt",
}


class DetectorGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Spy Control Panel")
        self.root.geometry("700x640")
        self.root.minsize(680, 620)

        self.app_dir = APP_DIR
        self.config_path = CONFIG_PATH
        self.process = None

        self.vars = {}
        self._build_ui()
        self.load_config_to_form()
        self._update_status("Готово")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Настройки детектора", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        self._add_combo(frame, "model", "Модель", ["ultralytics_pt", "ssd", "detr"], 1)
        self._add_check(frame, "prefer_light_model", "Предпочитать легкую модель", 2)
        self._add_combo(frame, "output_mode", "Режим вывода", ["local", "telegram"], 3)
        self._add_combo(frame, "capture_mode", "Режим захвата", ["video", "photo"], 4)

        self._add_entry(frame, "local_save_dir", "Папка сохранения", 5)
        self._add_entry(frame, "cooldown", "Cooldown (сек)", 6)
        self._add_entry(frame, "confidence", "Порог уверенности", 7)
        self._add_entry(frame, "frame_skip", "Frame skip", 8)
        self._add_entry(frame, "absence_reset_seconds", "Сброс после пропажи (сек)", 9)
        self._add_entry(frame, "video_seconds_after_lost", "Стоп видео после пропажи (сек)", 10)
        self._add_entry(frame, "video_max_seconds", "Макс длина видео (сек)", 11)
        self._add_entry(frame, "video_fps", "FPS записи", 12)
        self._add_entry(frame, "video_codec", "Кодек", 13)
        self._add_entry(frame, "yolo_pt_model", "PT модель", 14)
        self._add_entry(frame, "yolo_onnx_model", "ONNX модель", 15)
        self._add_entry(frame, "yolo_input_size", "Размер входа ONNX", 16)
        self._add_entry(frame, "target_classes", "Классы (через запятую)", 17)
        self._add_entry(frame, "telegram_token", "Telegram token", 18)
        self._add_entry(frame, "telegram_chat_id", "Telegram chat id", 19)

        btns = ttk.Frame(frame)
        btns.grid(row=20, column=0, columnspan=4, sticky="ew", pady=(12, 8))
        ttk.Button(btns, text="Сохранить", command=self.save_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Старт", command=self.start_detector).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Стоп", command=self.stop_detector).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(frame, textvariable=self.status_var, foreground="#0b5394")
        status.grid(row=21, column=0, columnspan=4, sticky="w")

        for col in range(4):
            frame.columnconfigure(col, weight=1)

    def _add_entry(self, parent, key, label, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
        self.vars[key] = var

    def _add_combo(self, parent, key, label, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        var = tk.StringVar()
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly").grid(row=row, column=1, columnspan=3, sticky="ew", pady=3)
        self.vars[key] = var

    def _add_check(self, parent, key, label, row):
        var = tk.BooleanVar()
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=4, sticky="w", pady=3)
        self.vars[key] = var

    def _update_status(self, text):
        self.status_var.set(text)

    def load_config_to_form(self):
        cfg = dict(DEFAULT_CONFIG)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg.update(json.load(f))
            except Exception as e:
                messagebox.showwarning("Внимание", f"Не удалось прочитать config.json: {e}")

        for key, var in self.vars.items():
            value = cfg.get(key, DEFAULT_CONFIG.get(key, ""))
            if key == "target_classes" and isinstance(value, list):
                value = ", ".join(value)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value))

    def _collect_config_from_form(self):
        cfg = {}
        cfg["telegram_token"] = self.vars["telegram_token"].get().strip()
        cfg["telegram_chat_id"] = self.vars["telegram_chat_id"].get().strip()
        cfg["model"] = self.vars["model"].get().strip()
        cfg["prefer_light_model"] = bool(self.vars["prefer_light_model"].get())
        cfg["output_mode"] = self.vars["output_mode"].get().strip()
        cfg["capture_mode"] = self.vars["capture_mode"].get().strip()
        cfg["local_save_dir"] = self.vars["local_save_dir"].get().strip() or "captures"
        cfg["cooldown"] = int(self.vars["cooldown"].get().strip())
        cfg["confidence"] = float(self.vars["confidence"].get().strip())
        cfg["frame_skip"] = int(self.vars["frame_skip"].get().strip())
        cfg["absence_reset_seconds"] = float(self.vars["absence_reset_seconds"].get().strip())
        cfg["video_seconds_after_lost"] = float(self.vars["video_seconds_after_lost"].get().strip())
        cfg["video_max_seconds"] = int(self.vars["video_max_seconds"].get().strip())
        cfg["video_fps"] = int(self.vars["video_fps"].get().strip())
        cfg["video_codec"] = self.vars["video_codec"].get().strip() or "mp4v"
        cfg["yolo_pt_model"] = self.vars["yolo_pt_model"].get().strip() or "yolov8n.pt"
        cfg["yolo_onnx_model"] = self.vars["yolo_onnx_model"].get().strip() or "yolov5n.onnx"
        cfg["yolo_input_size"] = int(self.vars["yolo_input_size"].get().strip())
        classes_raw = self.vars["target_classes"].get().strip()
        cfg["target_classes"] = [x.strip().lower() for x in classes_raw.split(",") if x.strip()]
        return cfg

    def save_config(self):
        try:
            cfg = self._collect_config_from_form()
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self._update_status("Настройки сохранены")
            return True
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить config.json: {e}")
            return False

    def start_detector(self):
        if self.process and self.process.poll() is None:
            self._update_status("Детектор уже запущен")
            return
        if not self.save_config():
            return

        try:
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--run-detector']
            else:
                cmd = [sys.executable, os.path.abspath(__file__), '--run-detector']

            self.process = subprocess.Popen(cmd, cwd=APP_DIR)
            self._update_status("Детектор запущен")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить детектор: {e}")

    def stop_detector(self):
        if not self.process or self.process.poll() is not None:
            self._update_status("Детектор не запущен")
            return

        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'], check=False, capture_output=True)
            else:
                self.process.terminate()
            self.process = None
            self._update_status("Детектор остановлен")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось остановить детектор: {e}")

    def _on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Выход", "Остановить детектор и закрыть окно?"):
                return
            self.stop_detector()
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:
        pass
    DetectorGui(root)
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-detector', action='store_true')
    args, _ = parser.parse_known_args()

    try:
        if args.run_detector:
            run_detector()
        else:
            run_gui()
    except Exception:
        logging.exception('Fatal error in main loop')
        raise
