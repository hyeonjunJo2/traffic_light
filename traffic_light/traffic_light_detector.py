#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
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
    "Brightness_Min_V": 120,
    "Saturation_Min_S": 100,
    "Red1_H_Max": 7,
    "Red2_H_Min": 170,
    "Yellow_H_Min": 8,
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

        # 2. 토픽 발행기 (미션 매니저 연동 및 RViz2 디버그 전용 - cmd_vel 충돌 방지 완료)
        self.traf_pub = self.create_publisher(String, '/traffic_light', mission_qos)       # 소문자: red, yellow, green, none
        self.state_pub = self.create_publisher(String, '/traffic_state', 10)               # 대문자: RED, YELLOW, GREEN, NONE
        self.debug_img_pub = self.create_publisher(Image, '/traffic_light/debug_image', 10)# RViz2용 디버그 영상
        self.yolo_img_pub = self.create_publisher(Image, '/yolo_image', 1)                 # 기존 competition 토픽 호환
        self.crop_img_pub = self.create_publisher(Image, '/traffic_light/cropped_image', 10)
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # 3. YOLO 비동기 작업 변수
        self.yolo_model = None
        self.is_custom_model = False
        self.yolo_box = None
        self.yolo_conf = 0.0
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
        self.get_logger().info("🛡️ [충돌 방지] /cmd_vel 직접 제어는 waypoints_follower에 일원화됨")
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

                found_box = None
                found_conf = 0.0
                best_area = 0

                for r in results:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        c = float(box.conf[0].cpu().numpy())
                        area = (bx2 - bx1) * (by2 - by1)
                        if area > best_area and area > 100:
                            best_area = area
                            found_box = [bx1, by1, bx2, by2]
                            found_conf = c

                with self.yolo_lock:
                    self.yolo_box = found_box
                    self.yolo_conf = found_conf

            except Exception:
                pass

            time.sleep(0.01)

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
    cv2.namedWindow(tuner_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tuner_win, 450, 480)

    cv2.createTrackbar("Brightness (Min V)", tuner_win, cfg.get("Brightness_Min_V", 120), 255, nothing)
    cv2.createTrackbar("Saturation (Min S)", tuner_win, cfg.get("Saturation_Min_S", 100), 255, nothing)
    cv2.createTrackbar("Red1 H Max", tuner_win, cfg.get("Red1_H_Max", 7), 30, nothing)
    cv2.createTrackbar("Red2 H Min", tuner_win, cfg.get("Red2_H_Min", 170), 180, nothing)
    cv2.createTrackbar("Yellow H Min", tuner_win, cfg.get("Yellow_H_Min", 8), 60, nothing)
    cv2.createTrackbar("Yellow H Max", tuner_win, cfg.get("Yellow_H_Max", 35), 60, nothing)
    cv2.createTrackbar("Green H Min", tuner_win, cfg.get("Green_H_Min", 40), 120, nothing)
    cv2.createTrackbar("Green H Max", tuner_win, cfg.get("Green_H_Max", 90), 140, nothing)
    cv2.createTrackbar("YOLO Conf (%)", tuner_win, cfg.get("YOLO_Conf_Thresh", 25), 100, nothing)
    cv2.createTrackbar("Min Pixel Count", tuner_win, cfg.get("Min_Pixel_Count", 40), 500, nothing)

    last_state = "NONE"
    smooth_box = None
    box_hold_count = 0
    MAX_HOLD_FRAMES = 12

    zoom_win_opened = False
    current_cfg = cfg.copy()

    fps_time = time.time()
    fps = 0
    fps_counter = 0

    try:
        while rclpy.ok():
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

            v_min = cv2.getTrackbarPos("Brightness (Min V)", tuner_win)
            s_min = cv2.getTrackbarPos("Saturation (Min S)", tuner_win)
            r1_max = cv2.getTrackbarPos("Red1 H Max", tuner_win)
            r2_min = cv2.getTrackbarPos("Red2 H Min", tuner_win)
            y_min = cv2.getTrackbarPos("Yellow H Min", tuner_win)
            y_max = cv2.getTrackbarPos("Yellow H Max", tuner_win)
            g_min = cv2.getTrackbarPos("Green H Min", tuner_win)
            g_max = cv2.getTrackbarPos("Green H Max", tuner_win)
            conf_val = cv2.getTrackbarPos("YOLO Conf (%)", tuner_win)
            min_pixel = cv2.getTrackbarPos("Min Pixel Count", tuner_win)

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
                curr_yolo_box = node.yolo_box
                curr_yolo_conf = node.yolo_conf

            if curr_yolo_box is not None:
                target_box = curr_yolo_box
                detect_source = f"YOLO ({curr_yolo_conf:.2f})"
            else:
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
                if smooth_box is None:
                    smooth_box = [float(v) for v in target_box]
                else:
                    alpha = 0.35
                    for i in range(4):
                        smooth_box[i] = (1 - alpha) * smooth_box[i] + alpha * target_box[i]
            else:
                if box_hold_count > 0:
                    box_hold_count -= 1
                else:
                    smooth_box = None

            detected_state = "NONE"
            cropped_view = None

            # 색상 판별
            if smooth_box is not None:
                x1, y1, x2, y2 = [int(v) for v in smooth_box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(uw, x2), min(uh, y2)

                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    cropped_view = upper_view[y1:y2, x1:x2]

                    r_cnt = cv2.countNonZero(r_mask_all[y1:y2, x1:x2])
                    y_cnt = cv2.countNonZero(y_mask_all[y1:y2, x1:x2])
                    g_cnt = cv2.countNonZero(g_mask_all[y1:y2, x1:x2])

                    max_c = max(r_cnt, y_cnt, g_cnt)

                    if max_c >= min_pixel:
                        if max_c == r_cnt:
                            detected_state = "RED"
                        elif max_c == y_cnt:
                            detected_state = "YELLOW"
                        elif max_c == g_cnt:
                            detected_state = "GREEN"

                    box_color = (0, 0, 255) if detected_state == "RED" else (0, 255, 255) if detected_state == "YELLOW" else (0, 255, 0) if detected_state == "GREEN" else (255, 255, 0)
                    cv2.rectangle(upper_view, (x1, y1), (x2, y2), box_color, 2)
                    
                    label_text = f"{detected_state} [{detect_source}]"
                    cv2.putText(upper_view, label_text, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

            # ------------------------------------------------------------------
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

            # 1) 대회 미션매니저 연동 토픽 (/traffic_light : 소문자 red, yellow, green, none)
            traf_msg = String()
            traf_msg.data = detected_state.lower()
            node.traf_pub.publish(traf_msg)

            # 2) 표준 상태 토픽 (/traffic_state : 대문자 RED, YELLOW, GREEN, NONE)
            state_msg = String()
            state_msg.data = detected_state
            node.state_pub.publish(state_msg)

            # 3) RViz2 시각화 영상 토픽 발행 (/traffic_light/debug_image)
            debug_view = upper_view.copy()
            status_text = f"STATE: {detected_state} (FPS: {fps})"
            text_color = (0, 0, 255) if detected_state == "RED" else (0, 255, 255) if detected_state == "YELLOW" else (0, 255, 0) if detected_state == "GREEN" else (200, 200, 200)
            cv2.putText(debug_view, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            try:
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
            cv2.imshow("Full Camera (YOLO Detection)", debug_view)
            cv2.imshow("HSV Mask (Debug)", combined_all)

            if cropped_view is not None and cropped_view.size > 0:
                zoom_display = cv2.resize(cropped_view, (250, 250), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Traffic Light Zoom (Crop)", zoom_display)
                zoom_win_opened = True
            else:
                if zoom_win_opened:
                    try:
                        cv2.destroyWindow("Traffic Light Zoom (Crop)")
                    except Exception:
                        pass
                    zoom_win_opened = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
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
