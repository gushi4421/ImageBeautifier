# 图像美化系统

数字图像处理技术课程大作业——一个集基础图像处理与 AI 增强功能于一体的综合图像美化平台。

## 功能概览

### 基础图像处理（手写算法，不依赖 OpenCV）

| 模块 | 功能 |
|------|------|
| **噪声** | 高斯噪声、椒盐噪声（基础版与优化版） |
| **空间域滤波** | 均值滤波、中值滤波、高斯模糊、双边滤波 |
| **频域滤波** | FFT 低通/高通/带通/带阻（理想/高斯/巴特沃斯） |
| **色调调整** | 灰度化、二值化、OTSU、伪彩色映射、亮度/对比度/饱和度、USM 锐化、智能补光、高光增强、通道置换假彩色、多光谱假彩色合成 |
| **几何变换** | 平移、旋转、缩放（双线性插值）、翻转、裁剪、加画框、拼接（水平/垂直） |
| **艺术效果** | 浮雕、毛玻璃、水面倒影 |

### AI 增强功能（基于深度学习）

| 模块 | 说明 |
|------|------|
| **图像拼接** | 经典 SIFT + RANSAC + 单应性线性融合；AI 拼接（UDIS++，ICCV 2023，TPS 变形 + 可微分融合） |
| **风格迁移** | 基于 VGG-19 的 Gatys et al. (CVPR 2016) 神经风格迁移 |
| **超分辨率** | EDSR ×4 超分辨率重建，支持去噪与锐化前后处理 |
| **旧照片修复** | 可组合修复管线：划痕修复、双边去噪、CLAHE、暗通道去雾、USM 锐化 |
| **曲线调整** | Catmull-Rom 样条非线性 LUT 调整（预设：S 曲线、提亮暗部、压缩高光、反相、褪色） |
| **智能锐化** | Sobel 边缘感知蒙版 + USM，仅锐化边缘区域 |

### 实时视频

- 摄像头实时处理（轮询快照模式）
- 视频文件逐帧处理与导出
- 支持实时滤镜应用

## 技术栈

**后端**
- Python 3.10+
- Flask（Web 框架，RESTful API）
- NumPy（核心计算）
- OpenCV（I/O、摄像头、部分算法）
- Pillow（图像加载）
- PyTorch / Torchvision（风格迁移 VGG-19、UDIS++ 拼接）
- PyYAML（配置管理）

**前端**
- 原生 HTML/CSS/JavaScript（无框架依赖）
- 深色主题，类 Photoshop 三栏布局
- 可视化操作历史时间线，支持撤销 / 回到原始

**桌面应用**
- pywebview（原生窗口封装）

## 项目结构

```
图像美化系统/
├── config.yaml              # 统一配置文件（输出目录等）
├── config.py                # 配置加载器
├── border.png               # 加画框特效模板
├── web/
│   └── app.py               # Flask Web 应用主入口
├── ui/
│   └── desktop_app.py       # 桌面应用入口（pywebview）
├── templates/
│   └── index.html           # 主页面 UI
├── static/
│   ├── css/style.css        # 样式表
│   └── js/
│       ├── api.js           # 前后端 API 通信
│       ├── main.js          # 核心前端逻辑
│       └── uiParamLoader.js # 动态参数面板渲染
├── src/
│   ├── utils.py             # 公共工具函数
│   ├── algorithm/           # 核心算法（手写实现）
│   │   ├── noise.py         # 噪声生成
│   │   ├── filter.py        # 空间域滤波
│   │   ├── frequency.py     # 频域滤波
│   │   ├── tone.py          # 色调/点操作
│   │   ├── geometry.py      # 几何变换
│   │   ├── artistic.py      # 艺术特效
│   │   └── stitch_classic.py # 经典 SIFT 拼接
│   ├── extensions/          # 扩展功能（含深度学习）
│   │   ├── stitch.py        # AI 拼接（UDIS++）
│   │   ├── style_transfer.py # 神经风格迁移
│   │   ├── super_resolution.py # 超分辨率
│   │   ├── photo_restoration.py # 旧照片修复
│   │   ├── curve_adjustment.py # 曲线调整
│   │   ├── smart_sharpen.py # 智能锐化
│   │   └── reflection.py    # 水面倒影
│   ├── adapters/
│   │   └── processor_adapter.py # 算法注册与调度适配器
│   └── io/
│       ├── image_io.py      # 图像读取/保存
│       └── video_io.py      # 视频 I/O
└── data/                    # 示例输出（各算法分类存放）
```

## 安装与运行

### 1. 安装依赖

```bash
pip install flask opencv-python opencv-contrib-python numpy Pillow pyyaml torch torchvision pywebview
```

### 2. 启动 Web 应用

```bash
cd 图像美化系统
python web/app.py
```

浏览器打开 `http://localhost:5000`。

### 3. 启动桌面应用

```bash
python ui/desktop_app.py
```

### 4. 作为 Python 库调用

```python
from src.algorithm.filter import mean_filter
import cv2

img = cv2.imread("photo.jpg")
result = mean_filter(img, kernel_size=5)
```

## 模块依赖说明

- **基础算法**（`src/algorithm/`）：仅依赖 NumPy，所有滤波/变换均由手写实现，适合学习图像处理基本原理
- **AI 增强**（`src/extensions/`）：依赖 PyTorch / OpenCV DNN，需下载预训练权重
  - UDIS++ 拼接：权重文件位于 `src/extensions/udis_stitching/Warp/model/` 和 `Composition/model/`
  - 风格迁移：自动下载 VGG-19 预训练权重
  - 超分辨率：首次运行自动下载 EDSR 权重

## 设计理念

- **学习优先**：基础算法全部手写，不依赖 OpenCV 高级 API，清晰展示算法原理
- **分层架构**：算法层 → 适配器层 → 展示层，职责分明，易于扩展
- **统一接口**：`ImageProcessorAdapter` 将全部处理功能注册为统一动作，支持单图/多图输入
- **多种交付**：同一套后端同时支持 Web、桌面、命令行三种使用方式
