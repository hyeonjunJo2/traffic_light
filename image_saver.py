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

class SafeImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # RealSense 카메라 토픽 구독
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
        
        # 📁 저장 폴더 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, 'jb_light_dataset')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 🛡️ 덮어쓰기 방지: 기존 파일 카운트
        existing_files = glob.glob(os.path.join(self.save_dir, 'jb_light_*.jpg'))
        self.img_count = len(existing_files)
        self.last_saved_name = ""

        self.get_logger().info("=" * 60)
        self.get_logger().info("🚀 [초고속 무지연 모드] 데이터 수집 노드 가동!")
        self.get_logger().info(f"📂 저장 경로: {self.save_dir}")
        self.get_logger().info(f"🔢 기존 사진: {self.img_count}장 (이 번호부터 시작)")
        self.get_logger().info("👉 창 클릭 후 [s]: 저장 | [q] 또는 [ESC]: 종료")
        self.get_logger().info("=" * 60)

    def image_callback(self, msg):
        # 콜백에서는 변환 후 저장만 하고 0.001초 만에 즉시 리턴 (블로킹 완벽 방지)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_img
        except Exception as e:
            self.get_logger().error(f"이미지 수신 에러: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SafeImageSaverNode()
    
    # 1. ROS 2 통신을 별도 백그라운드 스레드에서 실행 (메시지 밀림 현상 100% 제거)
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

            # FPS 측정
            frame_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_counter
                frame_counter = 0
                fps_time = time.time()

            # 화면 안내 텍스트 표시
            display_img = current_frame.copy()
            cv2.putText(display_img, f"FPS: {fps} | Saved: {node.img_count}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_img, "Press 's' to Save, 'q' to Quit", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if node.last_saved_name:
                cv2.putText(display_img, f"Last: {node.last_saved_name}", (15, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("YOLO Dataset Collector (Ultra-Fast)", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 's' 키: 원본(글자 없는 깨끗한 사진) 저장
            if key == ord('s'):
                node.img_count += 1
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = f'jb_light_{node.img_count:04d}_{timestamp}.jpg'
                full_path = os.path.join(node.save_dir, file_name)
                
                cv2.imwrite(full_path, current_frame)
                node.last_saved_name = file_name
                node.get_logger().info(f"📸 [{node.img_count:04d}번째 저장 완료] {file_name}")
                
            # 'q' 키 또는 ESC: 종료
            elif key == ord('q') or key == 27:
                node.get_logger().info(f"🛑 총 {node.img_count}장 저장 완료. 프로그램을 종료합니다.")
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
