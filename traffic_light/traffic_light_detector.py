#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
import cv2
import numpy as np
import threading
import time
import json
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

def nothing(x):
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'hsv_config.json')
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEFAULT_CONFIG = {
    "Brightness_Min_V": 100,
    "Saturation_Min_S": 80,
    "Red1_H_Max": 12,
    "Red2_H_Min": 165,
    "Yellow_H_Min": 13,
    "Yellow_H_Max": 35,
    "Green_H_Min": 40,
    "Green_H_Max": 90,
    "YOLO_Conf_Thresh": 25,
    "Min_Pixel_Count": 40
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"💾 [설정 자동 저장 완료] {CONFIG_PATH}")
    except Exception as e:
        print(f"⚠️ 설정 저장 실패: {e}")

class AsyncTrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')

        # 1. RealSense 카메라 영상 토픽 구독
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        
        # mission_manager 호환 QoS
        mission_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # 2. 토픽 발행기 (미션 매니저 연동 및 RViz2 디버그 전용)
        self.traf_pub = self.create_publisher(String, '/traffic_light', mission_qos)       # 소문자: red, yellow, green, none
        self.state_pub = self.create_publisher(String, '/traffic_state', 10)               # 대문자: RED, YELLOW, GREEN, NONE
        self.debug_img_pub = self.create_publisher(Image, '/traffic_light/debug_image', 10)# RViz2용 디버그 영상
        self.yolo_img_pub = self.create_publisher(Image, '/yolo_image', 1)                 # 기존 competition 토픽 호환
        self.box_area_pub = self.create_publisher(Float32, '/traffic_box_area', 10)        # YOLO 박스 면적 비율 퍼블리셔
        self.crop_img_pub = self.create_publisher(Image, '/traffic_light/cropped_image', 10)
        
        # [토픽 안정화(Debouncing) 로직]
        self.last_published_state = "none"
        self.none_counter = 0
        self.NONE_THRESHOLD = 45  # 45프레임(약 1.5초) 연속 NONE이어야만 진짜 NONE으로 인정 (직전 상태 유지)

        # 🚀 [4구간 신호차량 영구 유지(Latch) 로직]
        self.final_lane_state = None
        self.lane_detect_buffer = []

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # 3. YOLO 비동기 작업 변수
        self.yolo_model = None
        self.is_custom_model = False
        self.yolo_boxes_info = [] # (name, x1, y1, x2, y2, conf)
        self.yolo_lock = threading.Lock()
        self.yolo_conf_thresh = 0.25
        self.is_running = True

        # 4. 🧠 학습된 신호차 전용 모델 로드
        if YOLO_AVAILABLE:
            if os.path.exists(CUSTOM_MODEL_PATH):
                self.get_logger().info(f"🏆 [신호차 전용 커스텀 AI 모델 로드 성공!] {CUSTOM_MODEL_PATH}")
                self.yolo_model = YOLO(CUSTOM_MODEL_PATH)
                self.is_custom_model = True
            else:
                self.get_logger().info("🧠 [사전학습 YOLOv8n 모델 로드] (신호등 클래스 9번 탐지)")
                self.yolo_model = YOLO('yolov8n.pt')
                self.is_custom_model = False

        self.get_logger().info("=" * 60)
        self.get_logger().info("🚦 [신호차 자율주행 완성형 노드 가동!]")
        self.get_logger().info("📡 대회 미션매니저 연동 토픽: /traffic_light (red, yellow, green)")
        self.get_logger().info("=" * 60)

    def image_callback(self, msg):
        try:
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if msg.encoding in ['rgb8', 'RGB8']:
                cv_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            else:
                cv_img = im
            with self.frame_lock:
                self.latest_frame = cv_img
        except Exception:
            pass

    # ⚡ [백그라운드 스레드] YOLO AI 비동기 전용 추론 루프 (30 FPS 무지연)
    def yolo_worker_loop(self):
        while self.is_running and rclpy.ok():
            frame_to_predict = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    h, w, _ = self.latest_frame.shape
                    frame_to_predict = self.latest_frame[0:int(h * 0.7), :].copy()

            if frame_to_predict is None or self.yolo_model is None:
                time.sleep(0.02)
                continue

            try:
                # 커스텀 모델은 학습된 신호차 클래스 전체 탐지
                classes_to_detect = None if self.is_custom_model else [9]
                results = self.yolo_model.predict(
                    frame_to_predict,
                    conf=self.yolo_conf_thresh,
                    classes=classes_to_detect,
                    imgsz=256,
                    verbose=False,
                    device='cpu'
                )

                found_boxes_info = []
                for r in results:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        c = float(box.conf[0].cpu().numpy())
                        name = r.names[int(box.cls[0].cpu().numpy())]
                        area = (bx2 - bx1) * (by2 - by1)
                        if area > 100:
                            found_boxes_info.append((name, bx1, by1, bx2, by2, c))

                with self.yolo_lock:
                    self.yolo_boxes_info = found_boxes_info

            except Exception:
                pass

            time.sleep(0.01)

class NumberTuner:
    """★2026-09-12 통합: OpenCV 슬라이더 대신 숫자 입력칸 + 위아래(±1) 버튼 패널(Tkinter Spinbox).
    사용자 요청: 슬라이더가 불편 → 값을 직접 타이핑하거나 화살표로 1씩 조정. get(이름)으로 현재값 읽기, 매 루프 update() 호출."""
    def __init__(self, title, fields):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk(); self.root.title(title); self.root.resizable(False, False)
        self.vars = {}
        for row, (name, lo, hi, val) in enumerate(fields):
            tk.Label(self.root, text=name, anchor='w', width=20).grid(row=row, column=0, padx=6, pady=3, sticky='w')
            v = tk.IntVar(value=int(max(lo, min(hi, val))))
            sb = tk.Spinbox(self.root, from_=lo, to=hi, textvariable=v, width=6, increment=1, justify='right')
            sb.grid(row=row, column=1, padx=4, pady=3)
            tk.Label(self.root, text=f'({lo}~{hi})', fg='gray').grid(row=row, column=2, sticky='w')
            self.vars[name] = (v, lo, hi)
        self.save_requested = False
        tk.Button(self.root, text='저장 (s)', command=self._save).grid(row=len(fields), column=0, columnspan=3, pady=8, sticky='we')
        self.root.protocol('WM_DELETE_WINDOW', lambda: None)   # 창 닫기 무시(메인 루프가 관리)
    def _save(self):
        self.save_requested = True
    def get(self, name):
        v, lo, hi = self.vars[name]
        try:
            x = int(v.get())
        except Exception:
            x = lo
        x = max(lo, min(hi, x))
        return x
    def update(self):
        try:
            self.root.update_idletasks(); self.root.update()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AsyncTrafficLightDetector()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    yolo_thread = threading.Thread(target=node.yolo_worker_loop, daemon=True)
    yolo_thread.start()

    cfg = load_config()

    # 🎛️ 실시간 튜너 창
    tuner_win = "HSV / Color Tuner"
    tuner = NumberTuner(tuner_win, [
        ("Brightness (Min V)", 0, 255, cfg.get("Brightness_Min_V", 100)),
        ("Saturation (Min S)", 0, 255, cfg.get("Saturation_Min_S", 80)),
        ("Red1 H Max",         0, 30,  cfg.get("Red1_H_Max", 12)),
        ("Red2 H Min",         0, 180, cfg.get("Red2_H_Min", 165)),
        ("Yellow H Min",       0, 60,  cfg.get("Yellow_H_Min", 13)),
        ("Yellow H Max",       0, 60,  cfg.get("Yellow_H_Max", 35)),
        ("Green H Min",        0, 120, cfg.get("Green_H_Min", 40)),
        ("Green H Max",        0, 140, cfg.get("Green_H_Max", 90)),
        ("YOLO Conf (%)",      0, 100, cfg.get("YOLO_Conf_Thresh", 25)),
        ("Min Pixel Count",    0, 500, cfg.get("Min_Pixel_Count", 40)),
    ])
    tuner.update()                                  # 창 즉시 표시


    last_state = "NONE"
    smooth_box = None
    box_hold_count = 0
    import collections as _co
    recent_yolo = _co.deque(maxlen=30)  # ★(시각, 박스유무) 이력 — 시간 기준 확정 규칙용
    CONFIRM_WINDOW_S, CONFIRM_MIN, CONFIRM_RATIO = 0.6, 3, 0.5   # ★최근 0.6초의 YOLO 시도 중 절반 이상(최소 3회) 검출이면 인정
                                                                 #   (횟수 고정은 YOLO 속도에 따라 너무 빡빡/느슨해짐 → 비율로)
    USE_PLAN_B = False                       # ★박스 4등분 밝기 위치 판정(백화 대비). 우리 코스(3구/4구 혼재)엔 부적합 → 끔
    HOLD_S = 1.0                             # ★박스 유지 1.0초(놓침 사이를 잇는 시간. 0.5는 중간에 박스가 자주 사라짐)
    VOTE_S = 0.6                             # ★색 다수결 창 0.6초
    last_box_time = 0.0
    MAX_HOLD_FRAMES = 15          # ★12→30→15(≈0.5초, 2026-09-12): 놓침은 잇되 옛 위치를 오래 붙들지 않게
    SMOOTH_ALPHA = 1.0            # ★0.35→1.0: 평균 없이 최신 박스 그대로(접근 중 이동이 빨라 평균은 지연만 만듦)
    BOX_PAD_W, BOX_PAD_H = 0.30, 0.40   # ★색 판정용 박스 여유(가로 30%·세로 40%): 타이트한 YOLO 박스가 램프를 자르는 것 방지
    state_hist = _co.deque(maxlen=60)   # ★(시각, 색) 이력 — VOTE_S 초 안의 다수결
    USE_COLOR_FALLBACK = False    # ★YOLO 박스 없을 때 색 덩어리로 박스 대체하는 폴백. 대회장 콘·차량 오판 방지로 꺼둠

    zoom_win_opened = False
    current_cfg = cfg.copy()

    fps_time = time.time()
    fps = 0
    fps_counter = 0

    try:
        while rclpy.ok():
            tuner.update()                          # ★숫자 패널은 영상이 없어도 매 루프 갱신(안 하면 창이 안 그려짐)
            current_frame = None
            with node.frame_lock:
                if node.latest_frame is not None:
                    current_frame = node.latest_frame.copy()

            if current_frame is None:
                time.sleep(0.01)
                continue

            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            v_min = tuner.get("Brightness (Min V)")
            s_min = tuner.get("Saturation (Min S)")
            r1_max = tuner.get("Red1 H Max")
            r2_min = tuner.get("Red2 H Min")
            y_min = tuner.get("Yellow H Min")
            y_max = tuner.get("Yellow H Max")
            g_min = tuner.get("Green H Min")
            g_max = tuner.get("Green H Max")
            conf_val = tuner.get("YOLO Conf (%)")
            min_pixel = tuner.get("Min Pixel Count")

            node.yolo_conf_thresh = max(0.1, conf_val / 100.0)

            current_cfg = {
                "Brightness_Min_V": v_min,
                "Saturation_Min_S": s_min,
                "Red1_H_Max": r1_max,
                "Red2_H_Min": r2_min,
                "Yellow_H_Min": y_min,
                "Yellow_H_Max": y_max,
                "Green_H_Min": g_min,
                "Green_H_Max": g_max,
                "YOLO_Conf_Thresh": conf_val,
                "Min_Pixel_Count": min_pixel
            }

            h, w, _ = current_frame.shape
            upper_view = current_frame[0:int(h * 0.7), :]
            uh, uw, _ = upper_view.shape

            # HSV 마스크
            hsv_all = cv2.cvtColor(upper_view, cv2.COLOR_BGR2HSV)
            r_mask_all = (cv2.inRange(hsv_all, np.array([0, s_min, v_min]), np.array([r1_max, 255, 255])) |
                          cv2.inRange(hsv_all, np.array([r2_min, s_min, v_min]), np.array([180, 255, 255])))
            y_mask_all = cv2.inRange(hsv_all, np.array([y_min, s_min, v_min]), np.array([y_max, 255, 255]))
            g_mask_all = cv2.inRange(hsv_all, np.array([g_min, s_min, v_min]), np.array([g_max, 255, 255]))
            combined_all = r_mask_all | y_mask_all | g_mask_all
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_all = cv2.morphologyEx(combined_all, cv2.MORPH_CLOSE, kernel)

            # 백그라운드 YOLO 결과 확인
            target_box = None
            detect_source = "NONE"

            with node.yolo_lock:
                try:
                    current_boxes_info = node.yolo_boxes_info.copy()
                except AttributeError:
                    current_boxes_info = []

            curr_yolo_box = None
            curr_yolo_conf = 0.0
            best_area = 0
            
            for b in current_boxes_info:
                name, bx1, by1, bx2, by2, c = b
                if name in ("traffic_light", "traffic-light", "traffic-outdoor"):   # 모델 버전마다 클래스 이름이 다름(하이픈/밑줄) — 2026-09-12 통합 시 발견
                    area = (bx2 - bx1) * (by2 - by1)
                    if area > best_area:
                        best_area = area
                        curr_yolo_box = [bx1, by1, bx2, by2]
                        curr_yolo_conf = c

            # ★확정 규칙(2026-09-12): 이번 박스가 최근 5회 중 3회 이상과 겹칠 때만 인정 → 표지판·기둥 단발 오검출 차단
            _now = time.time()
            _sig = (tuple(round(v, 1) for v in curr_yolo_box) if curr_yolo_box is not None else None, len(current_boxes_info))
            if _sig != getattr(node, '_last_yolo_sig', object()):   # YOLO 결과가 바뀐 경우만 1회로 센다(화면 루프가 더 빠름)
                node._last_yolo_sig = _sig
                recent_yolo.append((_now, curr_yolo_box is not None))
            def _iou(a, b):
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                return inter / ua if ua > 0 else 0.0
            # 위치 조건은 두지 않는다 — 접근 중엔 박스가 프레임마다 크게 움직여 이전 박스와 안 겹침(지연의 원인이었음)
            _win_hits = [(_t, _has) for (_t, _has) in recent_yolo if _now - _t <= CONFIRM_WINDOW_S]
            _n_hit = sum(1 for (_t, _has) in _win_hits if _has)
            confirmed = (curr_yolo_box is not None and _n_hit >= CONFIRM_MIN and _n_hit >= CONFIRM_RATIO * max(1, len(_win_hits)))
            if confirmed:
                target_box = curr_yolo_box
                detect_source = f"YOLO ({curr_yolo_conf:.2f})"
            elif USE_COLOR_FALLBACK:   # ★2026-09-12 통합: 기본 꺼짐 — 신호등(YOLO 박스)을 찾았을 때만 색 판정(콘·차량 오판 방지)
                contours, _ = cv2.findContours(combined_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 80:
                        x, y, wb, hb = cv2.boundingRect(largest)
                        pad_x = int(wb * 1.5)
                        pad_y = int(hb * 1.5)
                        target_box = [max(0, x - pad_x), max(0, y - pad_y),
                                      min(uw, x + wb + pad_x), min(uh, y + hb + pad_y)]
                        detect_source = "COLOR"

            # 스무딩
            if target_box is not None:
                box_hold_count = MAX_HOLD_FRAMES
                last_box_time = _now
                if smooth_box is None:
                    smooth_box = [float(v) for v in target_box]
                else:
                    alpha = SMOOTH_ALPHA
                    for i in range(4):
                        smooth_box[i] = (1 - alpha) * smooth_box[i] + alpha * target_box[i]
            else:
                if _now - last_box_time > HOLD_S:   # ★시간 기준 유지: 0.5초 넘게 못 찾으면 박스 해제
                    smooth_box = None

            detected_state = "NONE"
            cropped_view = None
            hsv_info_text = ""

            # 색상 판별
            if smooth_box is not None:
                x1, y1, x2, y2 = [int(v) for v in smooth_box]
                _pw = int((x2 - x1) * BOX_PAD_W); _ph = int((y2 - y1) * BOX_PAD_H)   # ★박스 여유
                x1, y1 = max(0, x1 - _pw), max(0, y1 - _ph)
                x2, y2 = min(uw, x2 + _pw), min(uh, y2 + _ph)

                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    cropped_view = upper_view[y1:y2, x1:x2]
                    cropped_hsv = hsv_all[y1:y2, x1:x2]

                    # 🔍 [실시간 픽셀 분석 계측기] 가장 밝은 중심 영역의 실제 H, S, V 평균값 계산
                    ch, cw, _ = cropped_hsv.shape
                    cx1, cy1 = int(cw * 0.25), int(ch * 0.25)
                    cx2, cy2 = int(cw * 0.75), int(ch * 0.75)
                    center_crop = cropped_hsv[cy1:cy2, cx1:cx2]
                    
                    if center_crop.size > 0:
                        mean_h = int(np.mean(center_crop[:, :, 0]))
                        mean_s = int(np.mean(center_crop[:, :, 1]))
                        mean_v = int(np.mean(center_crop[:, :, 2]))
                        hsv_info_text = f"Live HSV -> H:{mean_h:2d} | S:{mean_s:3d} | V:{mean_v:3d}"

                    # --- [하이브리드 모드] 평소엔 HSV, 위기엔 Plan B ---
                    r_cnt = cv2.countNonZero(r_mask_all[y1:y2, x1:x2])
                    y_cnt = cv2.countNonZero(y_mask_all[y1:y2, x1:x2])
                    g_cnt = cv2.countNonZero(g_mask_all[y1:y2, x1:x2])
                    
                    max_c = max(r_cnt, y_cnt, g_cnt)
                    # ★2026-09-12: 박스는 있으면 항상 그린다(색 미확정이면 회색). 전엔 색 픽셀이 Min Pixel 미만인 순간
                    #   사각형을 안 그려 "박스가 사라진다"고 보였음(실제 박스·상태는 유지되고 있었음)
                    cv2.rectangle(upper_view, (x1, y1), (x2, y2), (160, 160, 160), 1)
                    cv2.putText(upper_view, f"box R:{r_cnt} Y:{y_cnt} G:{g_cnt} (min {min_pixel})", (x1, min(uh - 4, y2 + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

                    # 1차 시도: 기존 HSV 픽셀 카운트 방식
                    if max_c >= min_pixel:
                        if max_c == r_cnt:
                            detected_state = "RED"
                        elif max_c == y_cnt:
                            detected_state = "YELLOW"
                        elif max_c == g_cnt:
                            detected_state = "GREEN"

                        box_color = (0, 0, 255) if detected_state == "RED" else (0, 255, 255) if detected_state == "YELLOW" else (0, 255, 0) if detected_state == "GREEN" else (255, 255, 0)
                        cv2.rectangle(upper_view, (x1, y1), (x2, y2), box_color, 2)
                        
                        label_text = f"{detected_state} [{detect_source}] (R:{r_cnt} Y:{y_cnt} G:{g_cnt})"
                        cv2.putText(upper_view, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
                    
                    # 2차 시도: 픽셀 수가 미달(백화현상 등)이면 Plan B 비상 발동!
                    elif USE_PLAN_B:   # ★2026-09-12 기본 꺼짐: 3구 신호등엔 4등분이 안 맞고 여유 박스와 충돌 → 오판 원인
                        _bx1, _by1, _bx2, _by2 = [int(v) for v in smooth_box]       # Plan B는 여유 없는 원래 박스로
                        _bx1, _by1 = max(0, _bx1), max(0, _by1); _bx2, _by2 = min(uw, _bx2), min(uh, _by2)
                        hsv_crop = hsv_all[_by1:_by2, _bx1:_bx2]
                        v_channel = hsv_crop[:, :, 2] # 명도(밝기) 채널만 추출
                        
                        cw = x2 - x1
                        step = cw / 4.0
                        
                        brightness = []
                        for i in range(4):
                            col_start = int(i * step)
                            col_end = int((i + 1) * step) if i < 3 else cw
                            section = v_channel[:, col_start:col_end]
                            if section.size > 0:
                                top_pixels = np.percentile(section, 90)
                                mean_b = np.mean(section[section >= top_pixels]) if top_pixels > 0 else 0
                            else:
                                mean_b = 0
                            brightness.append(mean_b)
                        
                        max_idx = np.argmax(brightness)
                        max_b = brightness[max_idx]
                        
                        if max_b > 50:
                            if max_idx == 0: detected_state = "RED"
                            elif max_idx == 1: detected_state = "YELLOW"
                            elif max_idx == 2: detected_state = "GREEN"
                            elif max_idx == 3: detected_state = "GREEN"

                        box_color = (0, 0, 255) if detected_state == "RED" else (0, 255, 255) if detected_state == "YELLOW" else (0, 255, 0) if detected_state == "GREEN" else (255, 255, 0)
                        cv2.rectangle(upper_view, (x1, y1), (x2, y2), box_color, 2)
                        
                        for i in range(1, 4):
                            lx = x1 + int(i * step)
                            cv2.line(upper_view, (lx, y1), (lx, y2), (255, 255, 255), 1)
                        
                        label_text = f"[Plan B] {detected_state} (Zone:{max_idx+1})"
                        cv2.putText(upper_view, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # ------------------------------------------------------------------
            # ★다수결(2026-09-12): 최근 STATE_VOTE 프레임에서 가장 많은 색이 과반이면 그 색, 아니면 직전 확정 상태 유지
            state_hist.append((_now, detected_state))
            _win = [c for (_t, c) in state_hist if _now - _t <= VOTE_S]
            _cnt = _co.Counter(c for c in _win if c != "NONE")
            if _cnt:
                _top, _n = _cnt.most_common(1)[0]
                if _n * 2 > len(_win):
                    detected_state = _top
                elif last_state != "NONE" and _cnt.get(last_state, 0) > 0:
                    detected_state = last_state
                else:
                    detected_state = "NONE"
            else:
                detected_state = "NONE"

            # 📢 [상태 변경 로그 출력]
            # ------------------------------------------------------------------
            if detected_state != last_state:
                last_state = detected_state
                if detected_state == "RED":
                    node.get_logger().info("🔴 [RED] 신호 감지 -> mission_manager에 'red' 보고 (정지)")
                elif detected_state == "YELLOW":
                    node.get_logger().info("🟡 [YELLOW] 신호 감지 -> mission_manager에 'yellow' 보고 (감속/정지)")
                elif detected_state == "GREEN":
                    node.get_logger().info("🟢 [GREEN] 신호 감지 -> mission_manager에 'green' 보고 (출발)")
                else:
                    node.get_logger().info("⚪ [NONE] 신호차 없음")

            # 🚀 [4구간 신호차량 로직 추가]
            sign_boxes = [b for b in current_boxes_info if b[0] in ["sign_x", "sign_v", "sign_o"]]
            if len(sign_boxes) >= 3 and node.final_lane_state is None:
                # 가로(X) 좌표 순서로 정렬 (왼쪽부터 1번, 2번, 3번 전광판)
                sign_boxes.sort(key=lambda x: x[1])
                names = [b[0] for b in sign_boxes[:3]]
                
                # 초록/화살표(V)가 어디 있는지 파악
                if names[0] in ["sign_v", "sign_o"]:
                    node.lane_detect_buffer.append("lane_1")
                elif names[1] in ["sign_v", "sign_o"]:
                    node.lane_detect_buffer.append("lane_2")
                
                # 2초간 디버깅(확정 버퍼) - 약 30프레임 중 15번 이상이면 확정
                if len(node.lane_detect_buffer) > 30:
                    node.lane_detect_buffer.pop(0)
                
                if node.lane_detect_buffer.count("lane_1") >= 15:
                    node.final_lane_state = "lane_1"
                    node.get_logger().info("🔥 [신호차량 확정] 초록불이 1차선에 있습니다! 'lane_1' (직진) 무한 유지 시작!")
                elif node.lane_detect_buffer.count("lane_2") >= 15:
                    node.final_lane_state = "lane_2"
                    node.get_logger().info("🔥 [신호차량 확정] 초록불이 2차선에 있습니다! 'lane_2' (차선변경) 무한 유지 시작!")

            # 만약 신호차량 미션이 확정되었다면, 기존 색깔(detected_state) 다 무시하고 차선 강제 배정
            if node.final_lane_state is not None:
                detected_state = node.final_lane_state.upper()

            # [토픽 안정화(Debouncing) 로직 적용 - 1.5초 유지]
            if detected_state != "NONE":
                node.last_published_state = detected_state.lower()
                node.none_counter = 0
            else:
                node.none_counter += 1
                if node.none_counter >= node.NONE_THRESHOLD:
                    node.last_published_state = "none"

            # 1) 대회 미션매니저 연동 토픽 (/traffic_light : 소문자 red, yellow, green, none, lane_1, lane_2)
            traf_msg = String()
            traf_msg.data = node.last_published_state
            node.traf_pub.publish(traf_msg)

            # 2) 표준 상태 토픽 (/traffic_state : 대문자 RED, YELLOW, GREEN, NONE)
            state_msg = String()
            state_msg.data = node.last_published_state.upper()
            node.state_pub.publish(state_msg)

            # [Plan B] 신호등 박스 면적 비율(%) 퍼블리시 (0.0 ~ 100.0)
            area_msg = Float32()
            if smooth_box is not None:
                bx1, by1, bx2, by2 = [int(v) for v in smooth_box]
                box_area = float((bx2 - bx1) * (by2 - by1))
                frame_area = float(w * h)
                area_ratio = (box_area / frame_area) * 100.0
                area_msg.data = area_ratio
            else:
                area_msg.data = 0.0
            node.box_area_pub.publish(area_msg)

            # 3) RViz2 시각화 영상 토픽 발행 (/traffic_light/debug_image)
            debug_view = upper_view.copy()
            status_text = f"STATE: {detected_state} (FPS: {fps})"
            text_color = (0, 0, 255) if detected_state == "RED" else (0, 255, 255) if detected_state == "YELLOW" else (0, 255, 0) if detected_state == "GREEN" else (255, 100, 255) if "LANE" in detected_state else (200, 200, 200)
            cv2.putText(debug_view, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            # 🔍 화면 상단에 실시간 HSV 계측기 정보 출력!
            if hsv_info_text:
                cv2.putText(debug_view, hsv_info_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            try:
                if node.debug_img_pub.get_subscription_count() == 0 and node.yolo_img_pub.get_subscription_count() == 0:
                    raise RuntimeError('no debug subscribers')   # ★구독자 없으면 발행 생략(성능) — 아래 except가 삼킴
                debug_img_msg = Image()
                debug_img_msg.header.stamp = node.get_clock().now().to_msg()
                debug_img_msg.header.frame_id = "camera_color_frame"
                debug_img_msg.height, debug_img_msg.width, _ = debug_view.shape
                debug_img_msg.encoding = "bgr8"
                debug_img_msg.is_bigendian = 0
                debug_img_msg.step = debug_view.shape[1] * 3
                debug_img_msg.data = debug_view.tobytes()
                node.debug_img_pub.publish(debug_img_msg)
                node.yolo_img_pub.publish(debug_img_msg)
            except Exception:
                pass

            # 4) 화면 출력
            # ★표시용만 절반 크기(2026-09-12): 1280폭 창 3개를 매 프레임 그리면 그것만으로 100ms↑ → 6 FPS의 원인
            _dv = cv2.resize(debug_view, (debug_view.shape[1] // 2, debug_view.shape[0] // 2))
            _cm = cv2.resize(combined_all, (combined_all.shape[1] // 2, combined_all.shape[0] // 2))
            cv2.imshow("Full Camera (YOLO Detection)", _dv)
            cv2.imshow("HSV Mask (Debug)", _cm)

            if cropped_view is not None and cropped_view.size > 0:
                zoom_display = cv2.resize(cropped_view, (250, 250), interpolation=cv2.INTER_LINEAR)
                if hsv_info_text:
                    cv2.putText(zoom_display, hsv_info_text, (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                zoom_display = np.zeros((250, 250, 3), dtype=np.uint8)
                cv2.putText(zoom_display, "WAITING DETECT...", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            cv2.imshow("Traffic Light Zoom (Crop)", zoom_display)

            key = cv2.waitKey(1) & 0xFF
            tuner.update()                          # ★숫자 패널 이벤트 처리(매 루프)
            if key == ord('s') or tuner.save_requested:
                tuner.save_requested = False
                save_config(current_cfg)
            elif key == ord('p'):
                print("\n" + "=" * 50)
                print("📋 [현재 튜닝된 파라미터 값]")
                for k, v in current_cfg.items():
                    print(f"- {k:<20}: {v}")
                print("=" * 50 + "\n")
            elif key == ord('q') or key == 27:
                save_config(current_cfg)
                break

    except KeyboardInterrupt:
        save_config(current_cfg)
    finally:
        node.is_running = False
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
