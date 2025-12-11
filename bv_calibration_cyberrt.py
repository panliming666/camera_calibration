#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
import threading
import glob
import xml.etree.ElementTree as ET
import yaml
from datetime import datetime
import cv2

# CyberRT imports
from cyber.python.cyber_py3 import init, Node
from cyber.proto import raw_image_pb2  # 使用CyberRT的图像消息类型
from cyber.python import cyber

class FisheyeCalibrationNode:
    def __init__(self, node_name='fisheye_calibration_manager'):
        self.node = Node(node_name)

        # ---------------- 参数配置 ----------------
        self.image_topic = '/camera/image_raw'
        self.control_topic = '/action_rear_camera_para'
        self.result_topic = '/calibration/result'

        # 结果保存的根目录
        self.save_dir = os.path.join(os.getcwd(), 'calib_results')

        # C++ 可执行文件路径 (请修改为你实际的绝对路径)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.executable_path = os.path.join(script_dir, 'resource', 'bin', 'FisheyeCalibOnSingleImage')

        # ---------------- 内部状态 ----------------
        self.latest_image = None
        self.is_calibrating = False
        self.lock = threading.Lock()  # 线程锁，防止图片读写冲突

        # ---------------- 通信接口 ----------------
        # 图像订阅
        self.img_reader = self.node.create_reader(
            self.image_topic,
            raw_image_pb2.RawImage,
            self.image_callback
        )

        # 控制指令订阅
        # 注意：CyberRT需要先定义消息类型，这里用UInt32作为示例
        # 如果你有自定义的CalibrationControl消息，需要先编译
        try:
            from cyber.proto import std_msgs_pb2
            self.ctrl_reader = self.node.create_reader(
                self.control_topic,
                std_msgs_pb2.UInt32,
                self.control_callback
            )
        except:
            # 如果无法导入，使用RawMessage作为备选
            self.ctrl_reader = self.node.create_reader(
                self.control_topic,
                self._raw_message,
                self.control_callback
            )

        # 结果发布
        try:
            from cyber.proto import std_msgs_pb2
            self.result_writer = self.node.create_writer(
                self.result_topic,
                std_msgs_pb2.String
            )
        except:
            self.result_writer = self.node.create_writer(
                self.result_topic,
                self._raw_message
            )

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        print('[INFO] Fisheye Calibration Manager Started.')
        print(f'[INFO] Listening on {self.control_topic} for calibration commands')

    def _raw_message(self):
        """Raw message for fallback"""
        return type('RawMessage', (), {'data': None})()

    def image_callback(self, raw_image_msg):
        """实时更新最新的图像帧"""
        try:
            # CyberRT的RawImage消息格式转换为OpenCV格式
            # 这里需要根据实际的RawImage消息结构调整
            # 通常需要从原始数据重建图像
            import numpy as np

            # 假设RawImage有这些字段（需要根据实际消息结构调整）
            if hasattr(raw_image_msg, 'data'):
                # 将字节数据转换为numpy数组
                image_array = np.frombuffer(raw_image_msg.data, dtype=np.uint8)

                # 根据图像尺寸重新整形（需要根据实际消息格式调整）
                if hasattr(raw_image_msg, 'height') and hasattr(raw_image_msg, 'width'):
                    height = raw_image_msg.height
                    width = raw_image_msg.width
                    channels = raw_image_msg.channels if hasattr(raw_image_msg, 'channels') else 3
                    image_array = image_array.reshape((height, width, channels))
                else:
                    # 如果没有尺寸信息，需要其他方式处理
                    print('[WARN] No dimensions in RawImage message, cannot reshape')
                    return

                # 转换为OpenCV格式 (假设是BGR)
                cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.latest_image = cv_image
            else:
                print('[WARN] RawImage message has no data field')
        except Exception as e:
            print(f'[ERROR] Image conversion failed: {str(e)}')

    def control_callback(self, msg):
        """处理控制指令"""
        try:
            # 尝试从消息中获取操作码
            op = None
            if hasattr(msg, 'data'):
                op = msg.data
            elif hasattr(msg, 'operation'):
                op = msg.operation
            else:
                # 尝试检查消息的所有属性
                attrs = [attr for attr in dir(msg) if not attr.startswith('_')]
                print(f'[WARN] Message has attributes: {attrs}')
                return

            if op == 1:
                self.start_calibration()
            elif op == 0:
                self.stop_calibration()
            else:
                print(f'[WARN] Unknown operation code: {op}')
        except Exception as e:
            print(f'[ERROR] Error processing control callback: {str(e)}')

    def start_calibration(self):
        if self.is_calibrating:
            print('[WARN] Calibration is already in progress. Ignoring request.')
            return

        with self.lock:
            if self.latest_image is None:
                print('[ERROR] Cannot start calibration: No image received yet.')
                return
            # 拷贝一份当前图像用于处理，避免后续被覆盖
            process_image = self.latest_image.copy()

        print('[INFO] Starting calibration process...')
        self.is_calibrating = True

        # 开启新线程执行耗时任务，防止阻塞 CyberRT 回调
        thread = threading.Thread(target=self._run_calibration_thread, args=(process_image,))
        thread.start()

    def stop_calibration(self):
        """
        注意：subprocess 启动的 C++ 进程很难在 Python 线程中优雅地强行终止。
        这里主要重置状态标志位。
        """
        if self.is_calibrating:
            print('[INFO] Stop command received. Resetting status flag.')
            # 在这里，实际的 C++ 进程可能还在跑，但我们不再关心它的结果了
            self.is_calibrating = False
        else:
            print('[INFO] Calibration is not running.')

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
            print('[INFO] Executing C++ binary...')

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
                print('[INFO] Calibration task finished but was cancelled by user.')
                return

            if result.returncode != 0:
                print('[ERROR] Binary execution failed. Check logs.')
                return

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
                save_filename = f"calib_result.yaml"
                final_path = os.path.join(self.save_dir, save_filename)

                # 解析XML内容并转换为YAML格式
                yaml_data = self._convert_xml_to_yaml(target_xml)

                # 保存为YAML文件
                with open(final_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2)

                saved_files.append(final_path)
                print(f'[INFO] Calibration YAML saved to: {final_path}')

                # 6. 解析内容并发布 Topic
                content_str = self._read_xml_content(target_xml)
            else:
                print('[ERROR] No output XML found in temp directory.')
                content_str = "Error: No XML found"

            # 保存图像文件
            if found_images:
                images_dir = os.path.join(self.save_dir, 'images')
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
                    print(f'[INFO] Image saved to: {dest_path}')
            else:
                print('[WARN] No image files found in output')

            # 7. 发布结果消息
            files_str = "; ".join(saved_files) if saved_files else "No files saved"
            self._publish_result(f"Success|Files:{files_str}|Data:{content_str}")

        except Exception as e:
            print(f'[ERROR] Error during calibration thread: {str(e)}')
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
            print(f'[ERROR] XML to YAML conversion failed: {str(e)}')
            return {'error': f'Conversion failed: {str(e)}'}

    def _publish_result(self, message):
        """发布标定结果"""
        try:
            try:
                from cyber.proto import std_msgs_pb2
                result_msg = std_msgs_pb2.String()
                result_msg.data = message
                self.result_writer.write(result_msg)
            except:
                # 备选方案：使用RawMessage
                raw_msg = self._raw_message()
                raw_msg.data = message
                self.result_writer.write(raw_msg)
        except Exception as e:
            print(f'[ERROR] Failed to publish result: {str(e)}')

def main():
    # 初始化 CyberRT
    init()

    # 创建节点
    node = FisheyeCalibrationNode()

    # 启动节点
    node.node.start()

    # 保持运行
    try:
        cyber.spin()
    except KeyboardInterrupt:
        print('[INFO] Shutting down...')
    finally:
        node.node.stop()

if __name__ == '__main__':
    main()
