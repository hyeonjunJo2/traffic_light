#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob
import threading
import time
from datetime import datetime

class DualDatasetImageSaverNode(Node):
    def __init__(self):
        super().__init__('dual_image_saver_node')
        
        # RealSense 카메라 영상 토픽 구독
        self.image_topic = '/camera/camera/color/image_raw'
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        self.bridge = CvBridge()
        
        # 공유 변수 (스레드 안전)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 📁 [폴더 1] 3색 신호등 (빨강/주황/초록 원형) 저장 폴더
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.light_dir = os.path.join(base_dir, 'dataset_traffic_light')
        if not os.path.exists(self.light_dir):
            os.makedirs(self.light_dir)
            
        # 📁 [폴더 2] 화살표 / X 표시 (신호차 전광판) 저장 폴더
        self.arrow_dir = os.path.join(base_dir, 'dataset_arrow_sign')
        if not os.path.exists(self.arrow_dir):
            os.makedirs(self.arrow_dir)
            
        # 🛡️ 덮어쓰기 방지: 기존 파일 카운트
        light_files = glob.glob(os.path.join(self.light_dir, 'light_*.jpg'))
        arrow_files = glob.glob(os.path.join(self.arrow_dir, 'arrow_*.jpg'))
        self.light_count = len(light_files)
        self.arrow_count = len(arrow_files)
        self.last_saved_info = ""

        self.get_logger().info("=" * 65)
        self.get_logger().info("📸 [신호등 & 화살표 분리 수집기] 가동!")
        self.get_logger().info(f"📂 [1번 폴더] 3색 신호등: {self.light_dir} (현재 {self.light_count}장)")
        self.get_logger().info(f"📂 [2번 폴더] 화살표/X표시: {self.arrow_dir} (현재 {self.arrow_count}장)")
        self.get_logger().info("-" * 65)
        self.get_logger().info("👉 [1] 또는 [s] 키 : 🔴🟡🟢 3색 신호등 폴더에 저장")
        self.get_logger().info("👉 [2] 또는 [a] 키 : ⬅️⬆️❌ 화살표/X표시 폴더에 저장")
        self.get_logger().info("👉 [q] 또는 [ESC]  : 저장 종료")
        self.get_logger().info("=" * 65)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_img
        except Exception as e:
            self.get_logger().error(f"이미지 수신 에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DualDatasetImageSaverNode()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    fps_time = time.time()
    fps = 0
    frame_counter = 0

    try:
        while rclpy.ok():
            current_frame = None
            with node.frame_lock:
                if node.latest_frame is not None:
                    current_frame = node.latest_frame.copy()
            
            if current_frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_counter
                frame_counter = 0
                fps_time = time.time()

            # 화면 안내 HUD 오버레이
            display_img = current_frame.copy()
            
            # 상단 상태바 배경 박스
            cv2.rectangle(display_img, (0, 0), (640, 95), (0, 0, 0), -1)
            
            # 카운터 및 안내 출력
            cv2.putText(display_img, f"FPS: {fps} | [1] Light: {node.light_count} | [2] Arrow/X: {node.arrow_count}", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_img, "[1] or [s]: Save 3-Color Light  |  [2] or [a]: Save Arrow/X Sign", (15, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            if node.last_saved_info:
                cv2.putText(display_img, f"Saved: {node.last_saved_info}", (15, 82), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Dual Dataset Collector (Light vs Arrow)", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 🔴🟡🟢 1번 키 또는 's' 키: 3색 신호등 저장
            if key == ord('1') or key == ord('s'):
                node.light_count += 1
                file_name = f'light_{node.light_count:04d}_{timestamp}.jpg'
                full_path = os.path.join(node.light_dir, file_name)
                cv2.imwrite(full_path, current_frame)
                node.last_saved_info = f"[Light] {file_name}"
                node.get_logger().info(f"🔴🟡🟢 [3색 신호등 저장 #{node.light_count:04d}] {file_name}")

            # ⬅️⬆️❌ 2번 키 또는 'a' 키: 화살표 / X 표시 저장
            elif key == ord('2') or key == ord('a'):
                node.arrow_count += 1
                file_name = f'arrow_{node.arrow_count:04d}_{timestamp}.jpg'
                full_path = os.path.join(node.arrow_dir, file_name)
                cv2.imwrite(full_path, current_frame)
                node.last_saved_info = f"[Arrow/X] {file_name}"
                node.get_logger().info(f"⬅️⬆️❌ [화살표/X표시 저장 #{node.arrow_count:04d}] {file_name}")
                
            # 'q' 키 또는 ESC: 종료
            elif key == ord('q') or key == 27:
                node.get_logger().info(f"🛑 저장 종료: 3색 신호등 {node.light_count}장, 화살표 {node.arrow_count}장 수집 완료.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
