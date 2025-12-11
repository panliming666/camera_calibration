# 相机标定服务包

提供ROS 2框架下的鱼眼相机标定服务，支持内参和外参标定。

## 构建

### 1. 编译C++程序
```bash
cd resource
make clean && make
```

### 2. 编译ROS 2消息
```bash
colcon build --packages-select camera_calibration_service --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

## 使用

### 内参标定

**启动**：
```bash
python3 bv_calibration.py
```

**触发标定**：
```bash
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 1}"
```

**查看结果**：
```bash
ros2 topic echo /rear_camera_intrins
```

**结果文件**：`./calib_results/intrinsic_calibration.yaml`

---

### 外参标定

**启动**：
```bash
python3 chessboard_calibration_optimized.py
```

**注意**：后置相机内参将在标定时动态从 `./calib_results/intrinsic_calibration.yaml` 读取，无需提前准备。

**触发标定**：
```bash
# 标定后置相机
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 2}"

# 标定前置相机
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 3}"
```

**查看结果**：
```bash
ros2 topic echo /front_camera_offset
ros2 topic echo /rear_camera_offset
```

**结果文件**：
- `./calib_results/front_extrinsic_calibration.yaml`
- `./calib_results/rear_extrinsic_calibration.yaml`

## 常用命令

```bash
# 发送内参标定
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 1}"

# 发送外参标定（后置相机）
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 2}"

# 发送外参标定（前置相机）
ros2 topic pub --once /camera_calibration_req camera_calibration_service/msg/ActionCalibration "{operation: 3}"

# 监控状态
ros2 topic echo /front_camera_offset
ros2 topic echo /rear_camera_offset
ros2 topic echo /rear_camera_intrins
```

## 依赖

```bash
sudo apt install python3-opencv python3-numpy python3-scipy python3-yaml
sudo apt install ros-humble-rclpy ros-humble-std-msgs ros-humble-sensor-msgs ros-humble-cv-bridge
pip3 install scipy
```

## 消息格式

### 控制消息
**Topic**: `/camera_calibration_req`
```yaml
operation: 1  # 1=内参标定, 2=后相机外参标定, 3=前相机外参标定
```

### 结果消息
**Topics**: `/front_camera_offset`, `/rear_camera_offset`, `/rear_camera_intrins`

- `state`: 状态 (0=未标定, 1=标定中, 2=成功, 3=失败)
- `parameter`: 参数数组
  - 外参：[tx, ty, tz, rx, ry, rz] (平移: 米, 旋转: 弧度)
  - 内参：[fx, 0, cx, 0, fy, cy, 0, 0, 1] (3x3相机矩阵)
- `description`: 失败原因

## 文件结构

### 内参文件
```yaml
rear_intrinsic:
  timestamp: "2024-12-09 14:30:22"
  camera: rear
  intrinsic_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  distortion_coefficients: [k1, k2, p1, p2, k3]
```

### 外参文件
```yaml
front_extrinsic:
  timestamp: "2024-12-09 14:30:45"
  camera: front
  translation: [tx, ty, tz]
  rotation_euler_xyz_deg: [roll, pitch, yaw]
  rotation_quaternion_xyzw: [x, y, z, w]
```
