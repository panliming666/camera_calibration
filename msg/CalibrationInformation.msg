# 相机外参标定状态消息
# 用于报告相机外参标定的当前状态和结果

# === 标定状态 (0=未标定, 1=标定中, 2=标定成功, 3=标定失败) ===
uint8 state

# === 标定结果参数 ===
# 6-DOF外参: [tx, ty, tz, rx, ry, rz]  or 内参：[fx fy cx cy]
# 单位: 平移(米), 旋转(弧度)
float64[] parameter

# === 错误描述 (当标定失败时使用) ===
string description
