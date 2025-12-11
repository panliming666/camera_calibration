#!/usr/bin/env python3
import os
import time
import socket
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from camera_calibration_service.msg import ActionCalibration, CalibrationInformation
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

# 相机标定状态常量
UNCALIBRATED = 0
CALIBRATING = 1
CALIBRATED = 2
CALIBRATION_FAILED = 3

# 加载配置文件
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'camera_calibration_config.yaml')
def load_config():
    """加载配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_transform_matrix(R, t):
    """根据旋转矩阵 R 和平移向量 t 创建 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def invert_transform_matrix(T):
    """高效地计算 4x4 刚体变换矩阵的逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R.T @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)




class CameraState:
    """封装单个相机的所有状态信息"""
    def __init__(self, name):
        self.name = name
        self.camera_matrix = None
        self.dist_coeffs = None
        self.info_received = False
        self.frame = None
        self.success = False
        self.rvec_C_T = None
        self.tvec_C_T = None
        self.corners = None
        self.stable_count = 0
        self.last_calibrated_pose = None
        self.auto_calib_done = False
        self.auto_calib_in_progress = False

    def reset(self):
        """重置检测状态"""
        self.success = False
        self.rvec_C_T = None
        self.tvec_C_T = None
        self.corners = None


class ExtrinsicCalibratorOptimized(Node):
    def __init__(self):
        super().__init__('agv_extrinsic_calibrator_optimized')

        # 加载配置文件
        self.config = load_config()
        extrinsic_config = self.config['extrinsic_calibration']
        common_config = self.config['common']

        # === 输出目录配置 ===
        self.OUTPUT_DIR = common_config['output_dir']
        ensure_dir(self.OUTPUT_DIR)

        # === 内参和外参文件路径 ===
        # 内参文件路径（从内参标定读取）- 延迟检查，标定时动态读取
        self.INTRINSIC_CONFIG_FILE = os.path.join(self.OUTPUT_DIR, 'intrinsic_calibration.yaml')

        # 前相机外参文件路径
        self.FRONT_EXTRINSIC_CONFIG_FILE = os.path.join(self.OUTPUT_DIR, 'front_extrinsic_calibration.yaml')

        # 后相机外参文件路径
        self.REAR_EXTRINSIC_CONFIG_FILE = os.path.join(self.OUTPUT_DIR, 'rear_extrinsic_calibration.yaml')

        # === 文件路径 ===
        self.CALIBRATION_LOG_FILE = os.path.join('./calib_results', 'extrinsic_calibration_log.txt')

        # 初始化日志
        self.init_log_file()

        # === ROS话题配置 ===
        self.FRONT_IMAGE_TOPIC = extrinsic_config['front_image_topic']
        self.FRONT_CAMERA_INFO_TOPIC = extrinsic_config['front_camera_info_topic']
        self.REAR_IMAGE_TOPIC = extrinsic_config['rear_image_topic']
        self.CONTROL_TOPIC = extrinsic_config['control_topic']
        self.FRONT_CAMERA_OFFSET_TOPIC = extrinsic_config['front_camera_offset_topic']
        self.REAR_CAMERA_OFFSET_TOPIC = extrinsic_config['rear_camera_offset_topic']

        # === 棋盘格参数 ===
        self.SQUARES_X = extrinsic_config['board']['squares_x']
        self.SQUARES_Y = extrinsic_config['board']['squares_y']
        self.SQUARE_LENGTH = extrinsic_config['board']['square_size']

        # === 自动标定配置 ===
        self.ENABLE_AUTO_CALIBRATION = extrinsic_config['auto_calibration']['enable']
        self.AUTO_CALIB_STABLE_FRAMES = extrinsic_config['auto_calibration']['stable_frames']
        self.AUTO_CALIB_MIN_DISTANCE = extrinsic_config['auto_calibration']['min_distance']
        self.AUTO_CALIB_MIN_ROTATION = extrinsic_config['auto_calibration']['min_rotation']

        # === 图像显示配置 ===
        self.ENABLE_IMAGE_DISPLAY = extrinsic_config['display']['enable']
        self.DISPLAY_TIMER_INTERVAL = 0.033 if self.ENABLE_IMAGE_DISPLAY else 1.0

        # === 棋盘格位姿配置 ===
        self.FRONT_TRANSLATION_B_to_T = np.array(extrinsic_config['board_pose']['front']['translation'])
        self.FRONT_EULER_ANGLES_B_to_T = tuple(extrinsic_config['board_pose']['front']['rotation'])
        self.REAR_TRANSLATION_B_to_T = np.array(extrinsic_config['board_pose']['rear']['translation'])
        self.REAR_EULER_ANGLES_B_to_T = tuple(extrinsic_config['board_pose']['rear']['rotation'])

        # === 初始化棋盘格和位姿矩阵 ===
        self.board = self.init_board()
        self.T_B_to_T_front = self.calculate_T_B_T(
            self.FRONT_TRANSLATION_B_to_T, self.FRONT_EULER_ANGLES_B_to_T)
        self.T_B_to_T_rear = self.calculate_T_B_T(
            self.REAR_TRANSLATION_B_to_T, self.REAR_EULER_ANGLES_B_to_T)

        # === 初始化相机状态 ===
        self.front_camera = CameraState('front')
        self.rear_camera = CameraState('rear')

        # === 线程安全控制 ===
        self.calibration_lock = threading.Lock()  # 全局标定锁
        self.is_calibrating = False  # 全局标定状态标志

        # === 标定结果存储 ===
        self.cameras_calibrated = {}
        self.calibration_results = {'camera_params': {}}

        # === ROS 2 初始化 ===
        self.bridge = CvBridge()
        self.init_ros_components()

        # === 启动信息 ===
        self.start_time = datetime.now()
        self.log_to_file("=" * 80)
        self.log_to_file(f"标定开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_to_file("=" * 80)

        self.get_logger().info("--- 棋盘格双相机标定节点 (优化版) 已启动 ---")
        self.get_logger().info(f"等待 {self.FRONT_CAMERA_INFO_TOPIC} 上的前方相机内参...")
        self.get_logger().info(f"后置相机内参将在标定时动态读取: {self.INTRINSIC_CONFIG_FILE}")
        self.get_logger().info(f"监听 {self.CONTROL_TOPIC} 上的标定指令 (operation: 2=后相机, 3=前相机)...")

        if self.ENABLE_AUTO_CALIBRATION:
            self.get_logger().info("✅ 自动标定模式: 已启用")
        else:
            self.get_logger().info("⚠️  手动标定模式: 自动标定已禁用")

        # 创建定时器
        self.display_timer = self.create_timer(self.DISPLAY_TIMER_INTERVAL, self.display_frames)

    def init_log_file(self):
        """初始化日志文件"""
        ensure_dir(os.path.dirname(self.CALIBRATION_LOG_FILE))
        with open(self.CALIBRATION_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"棋盘格双相机标定日志文件\n")
            f.write(f"初始化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log_to_file(self, message):
        """将日志写入文件"""
        try:
            with open(self.CALIBRATION_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            self.get_logger().error(f"写入日志文件失败: {e}")

    def init_board(self):
        """初始化棋盘格世界坐标点 (3D object points)"""
        board = np.zeros((self.SQUARES_Y * self.SQUARES_X, 3), dtype=np.float32)
        for i in range(self.SQUARES_Y):
            for j in range(self.SQUARES_X):
                idx = i * self.SQUARES_X + j
                board[idx] = [j * self.SQUARE_LENGTH, i * self.SQUARE_LENGTH, 0]
        self.log_to_file(f"棋盘格世界坐标点已初始化: {self.SQUARES_X}x{self.SQUARES_Y}, 方格大小={self.SQUARE_LENGTH}m")
        return board

    def calculate_T_B_T(self, translation, euler_angles):
        """计算棋盘格的 T_B_to_T 矩阵"""
        r = Rotation.from_euler('xyz', euler_angles, degrees=True)
        R_B_to_T = r.as_matrix()
        return create_transform_matrix(R_B_to_T, translation)

    def init_ros_components(self):
        """初始化ROS组件"""
        qos_profile_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 前方相机订阅
        self.front_info_sub = self.create_subscription(
            CameraInfo, self.FRONT_CAMERA_INFO_TOPIC,
            self.front_info_callback, qos_profile_latched)
        self.front_image_sub = self.create_subscription(
            Image, self.FRONT_IMAGE_TOPIC,
            lambda msg: self.image_callback(msg, self.front_camera, 'front'), 10)

        # 后方相机订阅
        self.rear_image_sub = self.create_subscription(
            Image, self.REAR_IMAGE_TOPIC,
            lambda msg: self.image_callback(msg, self.rear_camera, 'rear'), 10)

        # 控制话题订阅
        self.control_sub = self.create_subscription(
            ActionCalibration, self.CONTROL_TOPIC,
            self.control_callback, 10)

        # 状态发布者
        self.front_camera_offset_pub = self.create_publisher(
            CalibrationInformation, self.FRONT_CAMERA_OFFSET_TOPIC, 10)
        self.rear_camera_offset_pub = self.create_publisher(
            CalibrationInformation, self.REAR_CAMERA_OFFSET_TOPIC, 10)

        # 初始化状态
        self._publish_camera_offset_state(self.front_camera, UNCALIBRATED)
        self._publish_camera_offset_state(self.rear_camera, UNCALIBRATED)

    def front_info_callback(self, msg):
        """处理前方相机 CameraInfo 消息"""
        if not self.front_camera.info_received:
            try:
                self.front_camera.camera_matrix = np.array(msg.k).reshape((3, 3))
                self.front_camera.dist_coeffs = np.array(msg.d)
                self.front_camera.info_received = True
                self.get_logger().info("成功接收到前方相机内参 (CameraInfo)！")
                self.log_to_file(f"[INFO] 成功接收到前方相机内参: {msg.width}x{msg.height}")

                camera_info_data = {
                    'width': msg.width, 'height': msg.height,
                    'camera_matrix': msg.k.tolist(), 'distortion_coefficients': msg.d.tolist(),
                    'distortion_model': msg.distortion_model,
                    'rectification_matrix': msg.r.tolist(),
                    'projection_matrix': msg.p.tolist()
                }
                self.calibration_results['camera_params']['front'] = camera_info_data
                self.destroy_subscription(self.front_info_sub)
            except Exception as e:
                error_msg = f"处理前相机CameraInfo消息失败: {str(e)}"
                self.get_logger().error(error_msg)
                self._burst_publish_extrinsic(
                    publisher=self.front_camera_offset_pub,
                    state=CALIBRATION_FAILED,
                    parameters=[],
                    description=error_msg
                )
                raise RuntimeError(error_msg)

    def _reprocess_rear_frame_for_detection(self):
        """内参就绪后，重新处理后置相机最新图像以检测棋盘格"""
        try:
            if self.rear_camera.frame is None:
                return

            # 转换为灰度图
            gray = cv2.cvtColor(self.rear_camera.frame, cv2.COLOR_BGR2GRAY)

            # 重置状态
            self.rear_camera.reset()

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

            if ret:
                # 亚像素精化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.rear_camera.corners = corners.copy()

                # 绘制角点
                display_frame = self.rear_camera.frame.copy()
                cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

                # 估计棋盘格位姿
                success, rvec, tvec = cv2.solvePnP(
                    self.board, corners, self.rear_camera.camera_matrix, self.rear_camera.dist_coeffs)

                if success:
                    self.rear_camera.success = True
                    self.rear_camera.rvec_C_T = rvec
                    self.rear_camera.tvec_C_T = tvec

                    # 绘制坐标轴
                    cv2.drawFrameAxes(display_frame, self.rear_camera.camera_matrix, self.rear_camera.dist_coeffs,
                                      self.rear_camera.rvec_C_T, self.rear_camera.tvec_C_T, 0.1)

                    # 更新显示帧
                    self.rear_camera.frame = display_frame

                    self.get_logger().info("✅ 后置相机重新检测到棋盘格！")
                else:
                    self.get_logger().info("后置相机位姿估计失败")
            else:
                self.get_logger().info("后置相机未检测到棋盘格")

        except Exception as e:
            self.get_logger().error(f"重新处理后置相机图像失败: {e}")
            self.log_to_file(f"[ERROR] 重新处理后置相机图像失败: {e}")

    def _load_rear_intrinsic_from_file(self):
        """从内参配置文件读取后置相机内参"""
        # 直接从内参配置文件读取
        config_file = self.INTRINSIC_CONFIG_FILE

        self.get_logger().info(f"正在从内参文件读取后置相机内参: {config_file}")
        self._load_rear_intrinsic_from_intrinsic_file(config_file)

    def _load_rear_intrinsic_from_intrinsic_file(self, intrinsic_file):
        """从内参文件中加载后内参记录"""
        try:
            with open(intrinsic_file, 'r', encoding='utf-8') as f:
                intrinsic_data = yaml.safe_load(f) or {}

            # 直接使用固定key
            rear_intrinsic_record = intrinsic_data.get('rear_intrinsic')
            if not rear_intrinsic_record:
                error_msg = "内参文件中未找到后相机内参记录"
                self.get_logger().error(error_msg)
                self._burst_publish_extrinsic(
                    publisher=self.rear_camera_offset_pub,
                    state=CALIBRATION_FAILED,
                    parameters=[],
                    description=error_msg
                )
                raise ValueError(error_msg)

            self.get_logger().info(f"找到后内参记录")

            # 解析内参矩阵（9位数完整相机矩阵）
            intrinsic_params = rear_intrinsic_record.get('intrinsic_matrix')
            if not intrinsic_params or len(intrinsic_params) != 9:
                error_msg = "内参数据格式不正确，需要9位数"
                self.get_logger().error(error_msg)
                self._burst_publish_extrinsic(
                    publisher=self.rear_camera_offset_pub,
                    state=CALIBRATION_FAILED,
                    parameters=[],
                    description=error_msg
                )
                raise ValueError(error_msg)

            # 直接转换为3x3相机矩阵
            camera_matrix = np.array(intrinsic_params, dtype=np.float64).reshape(3, 3)

            # 解析畸变系数
            dist_coeffs = rear_intrinsic_record.get('distortion_coefficients')
            if dist_coeffs:
                dist_coeffs = np.array(dist_coeffs, dtype=np.float64)

            self.rear_camera.camera_matrix = camera_matrix
            self.rear_camera.dist_coeffs = dist_coeffs
            self.rear_camera.info_received = True

            # 保存相机内参信息
            camera_info_data = {
                'width': 640, 'height': 480,
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist() if dist_coeffs is not None else None,
                'source': 'intrinsic_file',
                'file_path': intrinsic_file,
                'record_key': 'rear_intrinsic'
            }
            self.calibration_results['camera_params']['rear'] = camera_info_data

            self.get_logger().info("✅ 从内参文件加载后置相机内参成功！")
            self.log_to_file(f"[INFO] 从内参文件加载后置相机内参完成")

            # 重要：内参就绪后，如果已有图像在buffer中，重新处理一次以检测棋盘格
            if self.rear_camera.frame is not None:
                self.get_logger().info("内参已就绪，重新处理最新图像以检测棋盘格...")
                # 模拟图像回调，重新检测棋盘格
                temp_msg = None  # 我们不需要实际的ROS消息，只需要重新执行检测逻辑
                # 手动触发一次检测（使用现有的frame和gray图像）
                self._reprocess_rear_frame_for_detection()

        except Exception as e:
            error_msg = f"从内参文件读取后内参失败: {str(e)}"
            self.get_logger().error(error_msg)
            self.log_to_file(f"[ERROR] {error_msg}")
            self._burst_publish_extrinsic(
                publisher=self.rear_camera_offset_pub,
                state=CALIBRATION_FAILED,
                parameters=[],
                description=error_msg
            )
            raise RuntimeError(error_msg)

    def image_callback(self, msg, camera_state, camera_name):
        """统一的图像处理回调函数"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}相机 CvBridge 转换失败: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 只有在内参就绪时才进行棋盘格检测
        if camera_state.info_received:
            # 重置状态
            camera_state.reset()

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (self.SQUARES_X, self.SQUARES_Y), None)

            if ret:
                # 亚像素精化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                camera_state.corners = corners.copy()

                # 绘制角点
                cv2.drawChessboardCorners(display_frame, (self.SQUARES_X, self.SQUARES_Y), corners, ret)

                # 估计棋盘格位姿
                success, rvec, tvec = cv2.solvePnP(
                    self.board, corners, camera_state.camera_matrix, camera_state.dist_coeffs)

                if success:
                    camera_state.success = True
                    camera_state.rvec_C_T = rvec
                    camera_state.tvec_C_T = tvec

                    # 绘制坐标轴
                    cv2.drawFrameAxes(display_frame, camera_state.camera_matrix, camera_state.dist_coeffs,
                                      camera_state.rvec_C_T, camera_state.tvec_C_T, 0.1)

                    # 自动标定
                    if self.ENABLE_AUTO_CALIBRATION:
                        self.handle_auto_calibration(camera_state)
            else:
                camera_state.stable_count = 0

        # 保存图像帧（无论内参是否就绪）
        camera_state.frame = display_frame

    def control_callback(self, msg):
        """处理控制指令"""
        try:
            operation = msg.operation

            if operation == 2:
                # 后置相机标定
                self._calibrate_single_camera('rear')
            elif operation == 3:
                # 前置相机标定
                self._calibrate_single_camera('front')
            elif operation == 0:
                self.get_logger().info("收到标定指令: 停止标定")
                self.log_to_file("[INFO] 收到标定指令: 停止标定")
            else:
                self.get_logger().warn(f"收到未知标定指令: {operation}")
                self.log_to_file(f"[WARN] 未知标定指令: {operation}")

        except Exception as e:
            self.get_logger().error(f"处理标定指令失败: {e}")
            self.log_to_file(f"[ERROR] 处理标定指令失败: {e}")

    def _calibrate_single_camera(self, camera_name):
        """标定单个相机"""
        camera = self.front_camera if camera_name == 'front' else self.rear_camera
        publisher = self.front_camera_offset_pub if camera_name == 'front' else self.rear_camera_offset_pub

        self.get_logger().info(f"收到标定指令: 开始{camera_name}相机外参标定")
        self.log_to_file(f"[INFO] 收到标定指令: 开始{camera_name}相机外参标定")

        # 发送标定中状态
        for i in range(5):
            msg = CalibrationInformation()
            msg.state = 1  # 标定中
            msg.parameter = []
            msg.description = f"Starting {camera_name} camera extrinsic calibration"
            publisher.publish(msg)
            if i < 4:
                time.sleep(0.01)

        # 检查相机状态
        if not camera.info_received:
            # 对于后置相机，尝试动态读取内参文件
            if camera_name == 'rear' and not self.rear_camera.info_received:
                self.get_logger().info(f"[标定指令] {camera_name}相机内参未就绪，尝试从文件读取...")
                try:
                    self._load_rear_intrinsic_from_intrinsic_file(self.INTRINSIC_CONFIG_FILE)
                    self.get_logger().info(f"[标定指令] {camera_name}相机内参读取成功")
                except Exception as e:
                    failure_reason = f"{camera_name}相机内参文件读取失败: {str(e)}"
                    self.get_logger().warn(f"[标定指令] {failure_reason}")
                    self.log_to_file(f"[WARN] {failure_reason}")
                    self._burst_publish_extrinsic(
                        publisher=publisher,
                        state=CALIBRATION_FAILED,
                        parameters=[],
                        description=failure_reason
                    )
                    return
            else:
                failure_reason = f"{camera_name}相机内参未就绪（未收到CameraInfo或内参文件）"
                self.get_logger().warn(f"[标定指令] {failure_reason}")
                self.log_to_file(f"[WARN] {failure_reason}")
                self._burst_publish_extrinsic(
                    publisher=publisher,
                    state=CALIBRATION_FAILED,
                    parameters=[],
                    description=failure_reason
                )
                return

        # 使用全局锁检查标定状态（在内参检查之后）
        with self.calibration_lock:
            if self.is_calibrating:
                error_msg = f"标定正在进行中，忽略新的{camera_name}相机标定请求"
                self.get_logger().warn(error_msg)
                self.log_to_file(f"[WARN] {error_msg}")
                return

            # 立即设置状态，防止其他调用
            self.is_calibrating = True

        if not hasattr(camera, 'frame') or camera.frame is None:
            failure_reason = f"{camera_name}相机未收到图像数据"
            self.get_logger().warn(f"[标定指令] {failure_reason}")
            self.log_to_file(f"[WARN] {failure_reason}")
            self._burst_publish_extrinsic(
                publisher=publisher,
                state=CALIBRATION_FAILED,
                parameters=[],
                description=failure_reason
            )
            # 重置状态
            with self.calibration_lock:
                self.is_calibrating = False
            return

        if not (camera.success and camera.rvec_C_T is not None and camera.tvec_C_T is not None):
            failure_reason = f"{camera_name}相机未检测到棋盘格或位姿估计失败"
            self.get_logger().warn(f"[标定指令] {failure_reason}")
            self.log_to_file(f"[WARN] {failure_reason}")
            self._burst_publish_extrinsic(
                publisher=publisher,
                state=CALIBRATION_FAILED,
                parameters=[],
                description=failure_reason
            )
            # 重置状态
            with self.calibration_lock:
                self.is_calibrating = False
            return

        # 执行标定
        self.get_logger().info(f"[标定指令] {camera_name}相机检测到棋盘格，开始标定...")
        self.log_to_file(f"[INFO] {camera_name}相机标定开始")
        try:
            self.calibrate_camera(camera)
        finally:
            # 确保状态被重置，即使标定失败
            with self.calibration_lock:
                self.is_calibrating = False

    def _send_calibration_in_progress(self):
        """发送标定中状态"""
        for i in range(5):
            # 前相机
            msg = CalibrationInformation()
            msg.state = 1  # 标定中
            msg.parameter = []
            msg.description = "Starting extrinsic calibration"
            self.front_camera_offset_pub.publish(msg)

            # 后相机
            msg = CalibrationInformation()
            msg.state = 1  # 标定中
            msg.parameter = []
            msg.description = "Starting extrinsic calibration"
            self.rear_camera_offset_pub.publish(msg)

            if i < 4:  # 最后一帧不等待
                time.sleep(0.01)  # 10ms

    def _burst_publish_extrinsic(self, publisher, state, parameters, description):
        """连续发送5帧外参消息，每帧间隔10ms"""
        for i in range(5):
            msg = CalibrationInformation()
            msg.state = state
            msg.parameter = parameters
            msg.description = description
            publisher.publish(msg)
            if i < 4:  # 最后一帧不等待
                time.sleep(0.01)  # 10ms

    def handle_auto_calibration(self, camera_state):
        """简化的自动标定逻辑"""
        # 检查是否完成或正在标定
        if camera_state.auto_calib_done or camera_state.auto_calib_in_progress:
            return

        # 增加稳定计数
        camera_state.stable_count += 1

        if camera_state.stable_count >= self.AUTO_CALIB_STABLE_FRAMES:
            # 检查位姿变化
            current_pose = np.concatenate([camera_state.rvec_C_T.flatten(), camera_state.tvec_C_T.flatten()])

            if camera_state.last_calibrated_pose is not None:
                pose_changed = self.is_pose_significantly_changed(
                    camera_state.last_calibrated_pose, current_pose,
                    self.AUTO_CALIB_MIN_DISTANCE, self.AUTO_CALIB_MIN_ROTATION)
            else:
                pose_changed = True

            if pose_changed:
                # 使用全局锁检查标定状态，防止与手动标定冲突
                with self.calibration_lock:
                    if self.is_calibrating:
                        self.get_logger().info(f"[自动标定] {camera_state.name}相机检测到稳定棋盘格，但标定正在进行中，跳过本次自动标定")
                        camera_state.stable_count = 0  # 重置计数，避免持续触发
                        return

                    self.is_calibrating = True

                camera_state.auto_calib_in_progress = True
                self.get_logger().info(f"[自动标定] {camera_state.name}相机检测到稳定的棋盘格，开始自动标定...")
                self.log_to_file(f"[AUTO-CALIB] 开始{camera_state.name}相机自动标定")

                try:
                    self.calibrate_camera(camera_state)
                finally:
                    # 确保状态被重置
                    with self.calibration_lock:
                        self.is_calibrating = False

                camera_state.auto_calib_done = True
                camera_state.last_calibrated_pose = current_pose
                self.get_logger().info(f"[自动标定] {camera_state.name}相机自动标定完成！")
            else:
                camera_state.stable_count = 0

    def is_pose_significantly_changed(self, pose1, pose2, min_distance, min_rotation_deg):
        """检查位姿变化"""
        rvec1, tvec1 = pose1[:3], pose1[3:]
        rvec2, tvec2 = pose2[:3], pose2[3:]

        translation_change = np.linalg.norm(tvec2 - tvec1)

        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R_relative = R2 @ R1.T

        r = Rotation.from_matrix(R_relative)
        rotation_change_rad = np.abs(r.as_rotvec()).mean()
        rotation_change_deg = np.rad2deg(rotation_change_rad)

        return translation_change > min_distance or rotation_change_deg > min_rotation_deg

    def calibrate_camera(self, camera_state):
        """标定指定相机"""
        try:
            self._publish_camera_offset_state(camera_state, CALIBRATING)

            # 获取位姿
            rvec_C_T = camera_state.rvec_C_T
            tvec_C_T = camera_state.tvec_C_T
            camera_matrix = camera_state.camera_matrix
            dist_coeffs = camera_state.dist_coeffs
            T_B_to_T = self.T_B_to_T_front if camera_state.name == 'front' else self.T_B_to_T_rear

            # 计算变换矩阵
            R_C_to_T, _ = cv2.Rodrigues(rvec_C_T)
            T_C_to_T = create_transform_matrix(R_C_to_T, tvec_C_T)
            T_T_to_C = invert_transform_matrix(T_C_to_T)
            T_B_to_C = T_B_to_T @ T_T_to_C

            # 计算重投影误差
            reprojection_error = self.calculate_reprojection_error(
                rvec_C_T, tvec_C_T, camera_matrix, dist_coeffs, camera_state)

            # 打印和保存结果
            calibration_time = datetime.now()
            self.print_calibration_results(T_B_to_C, camera_state.name, calibration_time, reprojection_error)
            self.save_calibration_results(T_B_to_C, camera_state.name, calibration_time, reprojection_error)

            # 发布成功状态
            self._publish_camera_offset_state(camera_state, CALIBRATED, T_B_to_C, reprojection_error)

        except Exception as e:
            error_msg = f"{camera_state.name}相机标定失败: {str(e)}"
            self.get_logger().error(error_msg)
            self.log_to_file(f"[ERROR] {error_msg}")
            self._burst_publish_extrinsic(
                publisher=self.front_camera_offset_pub if camera_state.name == 'front' else self.rear_camera_offset_pub,
                state=CALIBRATION_FAILED,
                parameters=[],
                description=error_msg
            )
            raise RuntimeError(error_msg)

    def calculate_reprojection_error(self, rvec, tvec, camera_matrix, dist_coeffs, camera_state):
        """计算重投影误差"""
        try:
            if camera_state.corners is None:
                return None

            imgpoints, _ = cv2.projectPoints(
                self.board, rvec, tvec, camera_matrix, dist_coeffs)

            errors = []
            for i in range(len(camera_state.corners)):
                point_detected = camera_state.corners[i].ravel()
                point_projected = imgpoints[i].ravel()
                error = np.sqrt((point_detected[0] - point_projected[0])**2 +
                              (point_detected[1] - point_projected[1])**2)
                errors.append(error)

            errors = np.array(errors)

            return {
                'rms': float(np.sqrt(np.mean(errors**2))),
                'mean': float(np.mean(errors)),
                'max': float(np.max(errors)),
                'min': float(np.min(errors)),
                'std': float(np.std(errors)),
                'num_points': int(len(errors)),
                'all_errors': [float(e) for e in errors.tolist()]
            }

        except Exception as e:
            self.get_logger().error(f"计算重投影误差失败: {e}")
            return None

    def print_calibration_results(self, T_B_C, camera_name, calibration_time, reprojection_error):
        """打印标定结果"""
        R_B_C = T_B_C[:3, :3]
        t_B_C = T_B_C[:3, 3]

        r = Rotation.from_matrix(R_B_C)
        euler_xyz = r.as_euler('xyz', degrees=True)
        quat_xyzw = r.as_quat()

        np.set_printoptions(precision=4, suppress=True)
        camera_label = "前方" if camera_name == 'front' else "后方"

        self.get_logger().info(f"\n\n--- {camera_label}相机标定成功！---")
        self.get_logger().info(f"计算出的外参 T_B_{camera_name.upper()} (AGV 'base_link' -> '{camera_name}_camera_link'):\n")
        self.get_logger().info(f"--- 4x4 齐次变换矩阵 ---\n{T_B_C}\n")
        self.get_logger().info(f"--- 平移向量 (t) [x, y, z] (米) ---\n  {t_B_C}")
        self.get_logger().info(f"--- 旋转 (欧拉角) [roll, pitch, yaw] (度) ---\n  {euler_xyz}")
        self.get_logger().info(f"--- 旋转 (四元数) [x, y, z, w] ---\n  {quat_xyzw}\n")

        if reprojection_error is not None:
            self.get_logger().info(f"--- 重投影误差 (Reprojection Error) ---")
            self.get_logger().info(f"  RMS误差: %.4f 像素" % reprojection_error['rms'])
            self.get_logger().info(f"  平均误差: %.4f 像素" % reprojection_error['mean'])

        self.get_logger().info(f"--- 用于 static_transform_publisher (ROS 2) 的参数 ---")
        self.get_logger().info(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link")
        self.get_logger().info(f"--- {camera_label}相机标定结束 ---\n")

        # 记录到日志文件
        self.log_to_file(f"\n{'='*80}")
        self.log_to_file(f"【{camera_label}相机标定成功】")
        self.log_to_file(f"标定时间: {calibration_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_to_file(f"\n--- 4x4 齐次变换矩阵 ---\n{T_B_C}")
        self.log_to_file(f"\n--- 平移向量 (t) [x, y, z] (米) ---\n  {t_B_C}")
        self.log_to_file(f"\n--- 旋转 (欧拉角) [roll, pitch, yaw] (度) ---\n  {euler_xyz}")
        self.log_to_file(f"\n--- ROS 2 static_transform_publisher 命令 ---")
        self.log_to_file(f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link")
        self.log_to_file(f"{'='*80}\n")

    def save_calibration_results(self, T_B_C, camera_name, calibration_time, reprojection_error):
        """保存标定结果"""
        try:
            R_B_C = T_B_C[:3, :3]
            t_B_C = T_B_C[:3, 3]

            r = Rotation.from_matrix(R_B_C)
            euler_xyz = r.as_euler('xyz', degrees=True)
            quat_xyzw = r.as_quat()

            def to_python_type(obj):
                """转换为Python原生类型"""
                if isinstance(obj, np.ndarray):
                    return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in obj.tolist()]
                elif isinstance(obj, (np.floating, np.integer)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, list):
                    return [to_python_type(x) for x in obj]
                elif isinstance(obj, tuple):
                    return tuple(to_python_type(x) for x in obj)
                else:
                    return obj

            result_data = {
                'metadata': {
                    'hostname': socket.gethostname(),
                    'calibration_time': calibration_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'camera_name': camera_name,
                    'calibration_method': 'chessboard',
                    'board_squares_x': self.SQUARES_X,
                    'board_squares_y': self.SQUARES_Y,
                    'board_square_length': self.SQUARE_LENGTH
                },
                'transform_matrix': {
                    '4x4_matrix': to_python_type(T_B_C.tolist()),
                    'rotation_matrix': to_python_type(R_B_C.tolist()),
                    'translation': to_python_type(t_B_C.tolist())
                },
                'rotation': {
                    'euler_xyz_deg': to_python_type(euler_xyz.tolist()),
                    'quaternion_xyzw': to_python_type(quat_xyzw.tolist())
                },
                'quality_metrics': {
                    'reprojection_error': reprojection_error,
                    'quality_assessment': self.assess_calibration_quality(reprojection_error)
                },
                'ros2_command': {
                    'static_transform_publisher': f"ros2 run tf2_ros static_transform_publisher {t_B_C[0]} {t_B_C[1]} {t_B_C[2]} {quat_xyzw[0]} {quat_xyzw[1]} {quat_xyzw[2]} {quat_xyzw[3]} base_link {camera_name}_camera_link"
                }
            }

            self.cameras_calibrated[camera_name] = result_data
            self.get_logger().info(f"✅ {camera_name}相机标定结果已暂存！")
            self.get_logger().info(f"   已标定相机: {list(self.cameras_calibrated.keys())}")

            self.save_all_results_to_files()

        except Exception as e:
            error_msg = f"保存{camera_name}相机标定结果失败: {str(e)}"
            self.get_logger().error(error_msg)
            self.log_to_file(f"[ERROR] {error_msg}")
            self._burst_publish_extrinsic(
                publisher=self.front_camera_offset_pub if camera_name == 'front' else self.rear_camera_offset_pub,
                state=CALIBRATION_FAILED,
                parameters=[],
                description=error_msg
            )
            raise RuntimeError(error_msg)

    def assess_calibration_quality(self, reprojection_error):
        """评估标定质量"""
        if reprojection_error is None:
            return "无法评估（重投影误差计算失败）"

        rms = reprojection_error['rms']

        if rms < 0.3:
            return {'grade': '优秀', 'description': '重投影误差非常小，标定质量极佳', 'passed': True}
        elif rms < 0.5:
            return {'grade': '良好', 'description': '重投影误差较小，标定质量良好', 'passed': True}
        elif rms < 1.0:
            return {'grade': '可接受', 'description': '重投影误差在可接受范围内', 'passed': True}
        elif rms < 2.0:
            return {'grade': '警告', 'description': '重投影误差较大，建议重新标定', 'passed': False}
        else:
            return {'grade': '不合格', 'description': '重投影误差过大，标定结果不可靠，必须重新标定', 'passed': False}

    def save_all_results_to_files(self):
        """保存所有外参结果到独立yaml文件（精简版）"""
        try:
            # 为每个相机的标定结果保存精简信息（直接覆盖）
            for camera_name, camera_data in self.cameras_calibrated.items():
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 提取精简信息
                transform = camera_data['transform_matrix']
                rotation = camera_data['rotation']

                # 构建单个相机外参数据
                extrinsic_data = {
                    f'{camera_name}_extrinsic': {
                        'timestamp': timestamp,
                        'camera': camera_name,
                        'translation': transform['translation'],
                        'rotation_euler_xyz_deg': rotation['euler_xyz_deg'],
                        'rotation_quaternion_xyzw': rotation['quaternion_xyzw']
                    }
                }

                # 选择对应的文件路径
                if camera_name == 'front':
                    extrinsic_file = self.FRONT_EXTRINSIC_CONFIG_FILE
                elif camera_name == 'rear':
                    extrinsic_file = self.REAR_EXTRINSIC_CONFIG_FILE
                else:
                    continue

                # 保存到独立外参文件
                with open(extrinsic_file, 'w', encoding='utf-8') as f:
                    yaml.dump(extrinsic_data, f, default_flow_style=False, allow_unicode=True, indent=2)

                self.log_to_file(f"[INFO] {camera_name}相机外参结果已保存到: {extrinsic_file}")

            self.get_logger().info(f"🎉 所有外参结果已保存完成！")
            self.get_logger().info(f"   前相机外参: {self.FRONT_EXTRINSIC_CONFIG_FILE}")
            self.get_logger().info(f"   后相机外参: {self.REAR_EXTRINSIC_CONFIG_FILE}")
            self.get_logger().info(f"   已标定相机: {list(self.cameras_calibrated.keys())}")

        except Exception as e:
            error_msg = f"保存最终文件失败: {str(e)}"
            self.get_logger().error(error_msg)
            self.log_to_file(f"[ERROR] {error_msg}")
            # 为所有已标定的相机发送失败状态
            for camera_name in self.cameras_calibrated.keys():
                self._burst_publish_extrinsic(
                    publisher=self.front_camera_offset_pub if camera_name == 'front' else self.rear_camera_offset_pub,
                    state=CALIBRATION_FAILED,
                    parameters=[],
                    description=error_msg
                )
            raise RuntimeError(error_msg)

    def _publish_camera_offset_state(self, camera_state, state, T_B_C=None, reprojection_error=None, error_msg=None):
        """发布相机外参标定状态消息"""
        try:
            msg = CalibrationInformation()
            msg.state = state

            if T_B_C is not None:
                t_B_C = T_B_C[:3, 3]
                R_B_C = T_B_C[:3, :3]

                r = Rotation.from_matrix(R_B_C)
                euler_xyz = r.as_euler('xyz', degrees=False)

                parameters = [
                    float(t_B_C[0]), float(t_B_C[1]), float(t_B_C[2]),
                    float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
                ]
                msg.parameter = parameters
            else:
                parameters = []

            if error_msg:
                msg.description = error_msg
            else:
                msg.description = ""

            # 选择发布者
            publisher = self.front_camera_offset_pub if camera_state.name == 'front' else self.rear_camera_offset_pub

            # 如果是标定成功或失败状态，连续发送5帧
            if state in [CALIBRATED, CALIBRATION_FAILED]:
                self._burst_publish_extrinsic(publisher, state, parameters, msg.description)
                if state == CALIBRATED:
                    self.get_logger().info(f"[状态] {camera_state.name}相机: 标定成功! (连续发送5帧)")
                else:
                    self.get_logger().info(f"[状态] {camera_state.name}相机: 标定失败! (连续发送5帧)")
            else:
                # 其他状态正常发送
                publisher.publish(msg)
                # 记录日志
                if state == CALIBRATING:
                    self.get_logger().info(f"[状态] {camera_state.name}相机: 标定中...")
                elif state == CALIBRATED:
                    self.get_logger().info(f"[状态] {camera_state.name}相机: 标定成功!")

        except Exception as e:
            self.get_logger().error(f"发布{camera_state.name}相机状态消息失败: {e}")

    def display_frames(self):
        """显示图像并进行按键处理"""
        if not self.ENABLE_IMAGE_DISPLAY:
            return

        # 显示前方相机
        if hasattr(self.front_camera, 'frame') and self.front_camera.frame is not None:
            self.display_single_camera(self.front_camera, 'front')

        # 显示后方相机
        if hasattr(self.rear_camera, 'frame') and self.rear_camera.frame is not None:
            self.display_single_camera(self.rear_camera, 'rear')

        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("收到退出请求...")
            self.on_shutdown()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()
        elif key == ord('f'):
            if self.front_camera.success:
                with self.calibration_lock:
                    if not self.is_calibrating:
                        self.is_calibrating = True
                    else:
                        return
                try:
                    self.calibrate_camera(self.front_camera)
                finally:
                    with self.calibration_lock:
                        self.is_calibrating = False
        elif key == ord('r'):
            if self.rear_camera.success:
                with self.calibration_lock:
                    if not self.is_calibrating:
                        self.is_calibrating = True
                    else:
                        return
                try:
                    self.calibrate_camera(self.rear_camera)
                finally:
                    with self.calibration_lock:
                        self.is_calibrating = False

    def display_single_camera(self, camera_state, camera_name):
        """显示单个相机的图像"""
        frame = camera_state.frame.copy()
        label = "Front Camera" if camera_name == 'front' else "Rear Camera"

        cv2.putText(frame, f"{label} - Chessboard", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if camera_state.success:
            cv2.putText(frame, "Detected!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Place chessboard", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if self.ENABLE_AUTO_CALIBRATION:
            if camera_state.auto_calib_done:
                cv2.putText(frame, "Auto Calibrated!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            elif camera_state.auto_calib_in_progress:
                cv2.putText(frame, "Auto Calibrating...", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            elif camera_state.stable_count > 0:
                cv2.putText(frame, f"Stable: {camera_state.stable_count}/{self.AUTO_CALIB_STABLE_FRAMES}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Auto mode active", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Press '{camera_name[0]}' to calibrate", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(label, frame)

    def on_shutdown(self):
        """程序退出时保存汇总信息"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        if self.cameras_calibrated and len(self.cameras_calibrated) > 0:
            self.log_to_file("[INFO] 程序退出，正在保存最终标定结果...")
            self.save_all_results_to_files()

        try:
            self.log_to_file(f"\n{'='*80}")
            self.log_to_file("标定汇总信息")
            self.log_to_file(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_to_file(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_to_file(f"总耗时: {duration.total_seconds():.2f} 秒")
            self.log_to_file(f"输出目录: {self.OUTPUT_DIR}")
            self.log_to_file(f"已标定相机: {list(self.cameras_calibrated.keys())}")
            self.log_to_file(f"{'='*80}")

            self.get_logger().info(f"\n✅ 标定会话结束")
            self.get_logger().info(f"总耗时: {duration.total_seconds():.2f} 秒")
            if self.cameras_calibrated:
                self.get_logger().info(f"已标定相机: {', '.join(self.cameras_calibrated.keys())}")
            self.get_logger().info(f"前相机外参已保存到: {self.FRONT_EXTRINSIC_CONFIG_FILE}")
            self.get_logger().info(f"后相机外参已保存到: {self.REAR_EXTRINSIC_CONFIG_FILE}")

        except Exception as e:
            self.get_logger().error(f"保存汇总信息失败: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = ExtrinsicCalibratorOptimized()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"节点运行时发生未捕获异常: {e}")
        node.log_to_file(f"[FATAL] 节点运行时发生未捕获异常: {e}")
    finally:
        if rclpy.ok():
            node.on_shutdown()
            node.destroy_node()
            rclpy.shutdown()
        if node.ENABLE_IMAGE_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
