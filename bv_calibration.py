#!/usr/bin/env python3
import os
import time
import threading
import shutil
import subprocess
import tempfile
import glob
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
from camera_calibration_service.msg import ActionCalibration, CalibrationInformation
from cv_bridge import CvBridge
import cv2
import yaml
from datetime import datetime

# 加载配置文件
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'camera_calibration_config.yaml')
def load_config():
    """加载配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class FisheyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('fisheye_calibration_manager')

        # 加载配置文件
        self.config = load_config()
        intrinsic_config = self.config['intrinsic_calibration']
        common_config = self.config['common']

        # ---------------- 参数配置 ----------------
        self.declare_parameter('image_topic', intrinsic_config['image_topic'])
        self.declare_parameter('control_topic', intrinsic_config['control_topic'])
        self.declare_parameter('result_topic', intrinsic_config['result_topic'])

        # 结果保存的根目录
        self.declare_parameter('save_dir', common_config['output_dir'])

        # 内参标定结果文件路径
        self.intrinsic_result_file = os.path.join(self.get_parameter('save_dir').value, 'intrinsic_calibration.yaml')

        # C++ 可执行文件路径 (请修改为你实际的绝对路径)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.executable_path = os.path.join(script_dir, 'resource', 'bin', 'FisheyeCalibOnSingleImage')

        # ---------------- 内部状态 ----------------
        self.bridge = CvBridge()
        self.latest_image = None
        self.is_calibrating = False
        self.lock = threading.Lock() # 线程锁，防止图片读写冲突

        # ---------------- 通信接口 ----------------
        # 图像订阅
        self.img_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10
        )

        # 控制指令订阅
        self.ctrl_sub = self.create_subscription(
            ActionCalibration,
            self.get_parameter('control_topic').value,
            self.control_callback,
            10
        )

        # 结果发布
        self.result_pub = self.create_publisher(
            String,
            self.get_parameter('result_topic').value,
            10
        )

        # 后内参发布
        self.rear_intrins_pub = self.create_publisher(
            CalibrationInformation,
            intrinsic_config['rear_intrinsics_topic'],
            10
        )

        # 确保保存目录存在
        self.save_dir = self.get_parameter('save_dir').value
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info('Fisheye Calibration Manager Started.')
        self.get_logger().info(f'Listening on {self.get_parameter("control_topic").value} for calibration commands')
        self.get_logger().info('Using message type: ActionCalibration (operation: 1=intrinsic calibration)')

    def image_callback(self, msg):
        """实时更新最新的图像帧"""
        try:
            # 转换为 OpenCV 格式 (bgr8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {str(e)}')

    def control_callback(self, msg):
        """处理控制指令"""
        op = msg.operation

        if op == 1:
            # 连续发送5帧"标定中"状态，每帧间隔10ms
            self._burst_publish_rear_intrins(state=1, parameters=[], description="Starting intrinsic calibration")

            try:
                self.start_calibration()
            except RuntimeError as e:
                # 捕获异常但不退出，记录日志并发送错误状态
                error_msg = f'Calibration start failed: {str(e)}'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                # 注意：不重新抛出异常，让程序继续运行
            except Exception as e:
                # 捕获其他所有异常
                error_msg = f'Unexpected error in calibration: {str(e)}'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                import traceback
                traceback.print_exc()
        elif op == 0:
            self.stop_calibration()
        else:
            self.get_logger().warn(f'Unknown operation code: {op}')

    def _burst_publish_rear_intrins(self, state, parameters, description):
        """连续发送5帧消息，每帧间隔10ms"""
        for i in range(5):
            msg = CalibrationInformation()
            msg.state = state
            msg.parameter = parameters
            msg.description = description
            self.rear_intrins_pub.publish(msg)
            if i < 4:  # 最后一帧不等待
                time.sleep(0.01)  # 10ms

    def start_calibration(self):
        """
        开始标定流程。
        使用锁保护，确保在任何时候只有一个标定任务在执行。
        """
        with self.lock:
            # 检查是否已经在执行中
            if self.is_calibrating:
                error_msg = 'Calibration is already in progress. Ignoring new request.'
                self.get_logger().warn(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                raise RuntimeError(error_msg)

            # 检查是否有图像
            if self.latest_image is None:
                error_msg = 'Cannot start calibration: No image received yet. Please ensure the camera is publishing images.'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                raise RuntimeError(error_msg)

            # 立即设置状态为正在执行，防止其他线程并发执行
            self.is_calibrating = True
            self.get_logger().info('Starting calibration process...')

            # 拷贝一份当前图像用于处理，避免后续被覆盖
            process_image = self.latest_image.copy()

        # 在锁外开启新线程执行耗时任务，防止阻塞其他回调
        thread = threading.Thread(target=self._run_calibration_thread, args=(process_image,), daemon=True)
        thread.start()

    def stop_calibration(self):
        """
        注意：subprocess 启动的 C++ 进程很难在 Python 线程中优雅地强行终止。
        这里主要重置状态标志位。
        """
        if self.is_calibrating:
            self.get_logger().info('Stop command received. Resetting status flag.')
            # 在这里，实际的 C++ 进程可能还在跑，但我们不再关心它的结果了
            self.is_calibrating = False
        else:
            self.get_logger().info('Calibration is not running.')

    def _run_calibration_thread(self, cv_image):
        """在独立线程中运行的标定逻辑"""
        temp_dir = tempfile.mkdtemp(prefix='calib_task_')
        try:
            # 1. 准备环境 (Mock Directory Structure)
            # 因为 C++ 写死读 example_data/test/test.png
            input_dir = os.path.join(temp_dir, 'example_data', 'test')
            os.makedirs(input_dir, exist_ok=True)

            # C++ 写死输出到 result 文件夹，必须先创建
            os.makedirs(os.path.join(temp_dir, 'result'), exist_ok=True)

            # 保存图片为指定的名称
            img_path = os.path.join(input_dir, 'test.png')
            cv2.imwrite(img_path, cv_image)

            # 2. 执行 C++ 程序
            log_file = os.path.join(temp_dir, 'run.log')
            self.get_logger().info('Executing C++ binary...')

            with open(log_file, 'w') as f:
                result = subprocess.run(
                    [self.executable_path],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=temp_dir  # 关键：切换工作目录
                )

            # 3. 检查状态
            if not self.is_calibrating:
                self.get_logger().info('Calibration task finished but was cancelled by user.')
                return

            if result.returncode != 0:
                error_msg = f'Binary execution failed with return code {result.returncode}. Check logs.'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                raise RuntimeError(error_msg)

            # 4. 寻找结果文件
            # 递归搜索 XML 文件
            found_xmls = glob.glob(os.path.join(temp_dir, '**', '*.xml'), recursive=True)
            # 递归搜索所有图像文件
            found_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
                found_images.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))

            target_xml = None
            if found_xmls:
                # 优先找名字里带 calibration 的，或者直接取第一个
                target_xml = found_xmls[0]

            # 5. 保存结果文件
            saved_files = []

            # 保存XML文件并转换为YAML
            if target_xml:
                save_filename = f"intrinsic_calibration_result.yaml"
                final_path = os.path.join(self.save_dir, save_filename)

                # 解析XML内容并转换为YAML格式
                yaml_data = self._convert_xml_to_yaml(target_xml)

                # 保存为内参YAML文件
                with open(self.intrinsic_result_file, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2)

                saved_files.append(self.intrinsic_result_file)
                self.get_logger().info(f'Calibration YAML saved to: {self.intrinsic_result_file}')

                # 6. 保存并发布后内参信息
                self._save_and_publish_rear_intrinsics(yaml_data)

                # 7. 解析内容并发布 Topic
                content_str = self._read_xml_content(target_xml)
            else:
                error_msg = 'No output XML found in temp directory.'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                raise RuntimeError(error_msg)

            # 保存图像文件（硬编码路径）
            if found_images:
                images_dir = "./calib_results/images"
                os.makedirs(images_dir, exist_ok=True)

                for img_path in found_images:
                    # 获取文件名
                    img_name = os.path.basename(img_path)
                    # 添加时间戳避免重名
                    name, ext = os.path.splitext(img_name)
                    new_name = f"{name}{ext}"
                    dest_path = os.path.join(images_dir, new_name)
                    shutil.copy2(img_path, dest_path)
                    saved_files.append(dest_path)
                    self.get_logger().info(f'Image saved to: {dest_path}')
            else:
                self.get_logger().warn('No image files found in output')

            # 7. 发布结果消息
            files_str = "; ".join(saved_files) if saved_files else "No files saved"
            msg = String()
            msg.data = f"Success|Files:{files_str}|Data:{content_str}"
            self.result_pub.publish(msg)

        except Exception as e:
            error_msg = f'Error during calibration thread: {str(e)}'
            self.get_logger().error(error_msg)
            self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
            import traceback
            traceback.print_exc()
        finally:
            # 清理
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            self.is_calibrating = False

    def _read_xml_content(self, xml_path):
        """简单读取XML内容用于发布"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 这里可以根据需要提取 intrinsicMat 等具体数据
            # 暂时返回整个 XML 的字符串表示，或者只提取内参
            intrinsic = root.find('intrinsicMat/data')
            coeffs = root.find('coeffs/data')

            res = ""
            if intrinsic is not None:
                res += f"Intrinsic: {intrinsic.text.strip()}; "
            if coeffs is not None:
                res += f"Coeffs: {coeffs.text.strip()}"
            return res if res else "XML parsed but data not found"
        except:
            return "Error parsing XML"

    def _convert_xml_to_yaml(self, xml_path):
        """将XML文件转换为YAML格式"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            yaml_data = {}

            # 递归转换XML为字典
            def xml_to_dict(element):
                result = {}

                # 处理属性
                if element.attrib:
                    result['@attributes'] = element.attrib

                # 处理文本内容
                if element.text and element.text.strip():
                    # 尝试将文本转换为数字
                    text = element.text.strip()
                    try:
                        # 尝试转换为浮点数
                        if '.' in text or 'e' in text.lower():
                            result['#text'] = float(text)
                        else:
                            # 尝试转换为整数
                            result['#text'] = int(text)
                    except ValueError:
                        # 如果转换失败，保持为字符串
                        result['#text'] = text

                # 处理子元素
                for child in element:
                    child_data = xml_to_dict(child)
                    if child.tag in result:
                        # 如果已经存在同名元素，转换为列表
                        if not isinstance(result[child.tag], list):
                            result[child.tag] = [result[child.tag]]
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = child_data

                return result

            yaml_data = xml_to_dict(root)

            # 添加元数据
            yaml_data['calibration_metadata'] = {
                'calibration_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'format': 'YAML',
                'source_format': 'XML'
            }

            return yaml_data

        except Exception as e:
            error_msg = f'XML to YAML conversion failed: {str(e)}'
            self.get_logger().error(error_msg)
            self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
            raise RuntimeError(error_msg)

    def _save_and_publish_rear_intrinsics(self, yaml_data):
        """保存并发布后内参信息到统一yaml文件和/rear_camera_intrins话题"""
        try:
            # 解析内参矩阵
            intrinsic_data = None
            if 'intrinsicMat' in yaml_data and 'data' in yaml_data['intrinsicMat']:
                intrinsic_data = yaml_data['intrinsicMat']['data']
            elif 'intrinsic_matrix' in yaml_data:
                intrinsic_data = yaml_data['intrinsic_matrix']
            elif 'camera_matrix' in yaml_data:
                intrinsic_data = yaml_data['camera_matrix']

            if intrinsic_data and isinstance(intrinsic_data, dict) and '#text' in intrinsic_data:
                intrinsic_text = intrinsic_data['#text']
                intrinsic_values = []
                for line in intrinsic_text.strip().split('\n'):
                    line = line.strip()
                    if line:
                        values = line.split()
                        intrinsic_values.extend([float(v) for v in values])
                intrinsic_data = intrinsic_values

            # 解析畸变系数
            coeffs_data = None
            if 'coeffs' in yaml_data and 'data' in yaml_data['coeffs']:
                coeffs_data = yaml_data['coeffs']['data']
                if isinstance(coeffs_data, dict) and '#text' in coeffs_data:
                    coeffs_text = coeffs_data['#text']
                    coeffs_values = []
                    for line in coeffs_text.strip().split('\n'):
                        line = line.strip()
                        if line:
                            coeffs_values.extend([float(v) for v in line.split()])
                    coeffs_data = coeffs_values

            # 构建内参数组 [fx, fy, cx, cy]
            intrinsic_params = None
            if isinstance(intrinsic_data, list) and len(intrinsic_data) >= 4:
                # 相机矩阵: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                intrinsic_params = [intrinsic_data[0], intrinsic_data[4], intrinsic_data[2], intrinsic_data[5]]

            if intrinsic_params:
                # 构建完整的3x3相机矩阵 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                # 确保所有值都是浮点数（ROS 2消息要求）
                full_intrinsic_matrix = [
                    float(intrinsic_data[0]), 0.0, float(intrinsic_data[2]),
                    0.0, float(intrinsic_data[4]), float(intrinsic_data[5]),
                    0.0, 0.0, 1.0
                ]

                # 保存精简的内参信息到单独文件（使用rear_intrinsic键）
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                intrinsic_summary = {
                    'timestamp': timestamp,
                    'camera': 'rear',
                    'intrinsic_matrix': full_intrinsic_matrix,
                    'distortion_coefficients': coeffs_data if coeffs_data is not None else None
                }

                # 保存到文件（使用rear_intrinsic作为外层键）
                with open(self.intrinsic_result_file, 'w', encoding='utf-8') as f:
                    yaml.dump({'rear_intrinsic': intrinsic_summary}, f, default_flow_style=False, allow_unicode=True, indent=2)

                self.get_logger().info(f'已保存后内参到: {self.intrinsic_result_file}')

                # 连续发送5帧成功结果（使用9位数相机矩阵）
                self._burst_publish_rear_intrins(
                    state=2,
                    parameters=full_intrinsic_matrix,
                    description=""
                )
                self.get_logger().info(f'已连续发送5帧内参结果到/rear_camera_intrins: fx={intrinsic_data[0]:.2f}, fy={intrinsic_data[4]:.2f}, cx={intrinsic_data[2]:.2f}, cy={intrinsic_data[5]:.2f}')
            else:
                error_msg = '无法解析内参数据，跳过内参发布'
                self.get_logger().error(error_msg)
                self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f'保存/发布后内参失败: {str(e)}'
            self.get_logger().error(error_msg)
            self._burst_publish_rear_intrins(state=3, parameters=[], description=error_msg)
            raise RuntimeError(error_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FisheyeCalibrationNode()
    
    # 使用 MultiThreadedExecutor 是个好习惯，但在 Python 中
    # 我们上面已经手动使用了 threading.Thread 来处理耗时任务，
    # 所以这里用默认的 spin 也是可以的。
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()