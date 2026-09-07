# 红色区域时间戳扫描器（命令行版）

当前是第一版命令行原型，用于验证完整算法后再做本地界面。

## 任务名称

- `红点扫描`：扫描红色像素，生成原始时间戳，并按 10 秒归并生成待确认 txt。
- `红点切片`：读取归并后的时间戳 txt，为每段时间前后各 10 秒切视频，并导出该时间段全部音轨。

## GUI 界面

新版 `app.py` 使用 PySide6 暗色玻璃拟态 Dashboard，并加入扫描、归并、切片按钮。旧版 Tkinter 界面保留在 `app_tk.py`。

启动图形界面：

```powershell
py -3.10 app.py
```

GUI 已支持：

- 视频文件选择
- 预览帧加载与鼠标拖拽框选 ROI
- 红色通道灰度、色相范围、饱和度、最少红色像素滑块
- 扫描进度条与运行记录
- 扫描后生成归并 txt
- 按归并时间戳执行红点切片

首次在其它电脑运行时执行：

```powershell
py -3.10 -m pip install -r requirements.txt
```

## 功能

- 读取本地 mp4 视频，默认使用 CUDA 硬解码，失败时自动回退 CPU。
- 检测画面左上角为原点、`x=716, y=722, width=387, height=111` 的矩形区域。
- 每 30 帧检测一次。素材为 1920x1080 60fps 时，等价于每 0.5 秒检测一次。
- 红色判定：红色通道灰度值 `R > 180`，HSV 色相接近红色，默认饱和度不低于 60。
- 单个采样帧中至少有 400 个红色像素才记录事件。
- 连续命中的采样点合并为一个事件，只输出首次出现时间、估计持续时长、红色像素数。
- 输出 txt，时间从素材开头计算，格式为 `HH:MM:SS HH:MM:SS red_pixels=N`。

## 运行环境

本机当前环境已经可用：

- Python 3.10.11
- NumPy 2.2.6
- FFmpeg gyan 全量版，支持 CUDA/NVDEC
- NVIDIA RTX 4070

如果换机器，先执行：

```powershell
py -3.10 -m pip install -r requirements.txt
```

FFmpeg 需要能在 `PATH` 中找到，也可以通过环境变量 `FFMPEG_BIN` 和 `FFPROBE_BIN` 指定。

## 快速验证

生成一段包含已知红块事件的合成视频：

```powershell
py -3.10 make_test_video.py test_red_events.mp4
```

运行扫描：

```powershell
py -3.10 scan_red.py test_red_events.mp4
```

默认输出文件为 `test_red_events.red_timestamps.txt`。

合成视频包含两个红色块：1-3 秒和 5.5-7 秒，另外还有 4-5 秒橙色块与 8-9 秒白色块用于误报测试。当前验证结果只输出两个事件：

```text
00:00:01 00:00:02 red_pixels=15900
00:00:05 00:00:02 red_pixels=8600
```

## 扫描真实视频

```powershell
py -3.10 scan_red.py "C:\videos\input.mp4"
py -3.10 scan_red.py "C:\videos\input.mp4" -o "C:\videos\timestamps.txt"
```

## 按标记拼接片段

读取 `<视频名>.red_timestamps.txt` 中的标记时间，为每个标记截取前后各 10 秒，再按顺序拼接成一个视频：

```powershell
py -3.10 merge_segments.py "C:\videos\input.mp4"
```

输出为 `<视频名>_joined.mp4`，默认写到源视频所在目录。片段区间超出视频边界时会自动裁到可用范围内，不生成超过视频时长的片段。可通过 `--before` 和 `--after` 调整前后秒数。

## 参数

```text
--region x,y,width,height   检测区域，默认 716,722,387,111
--sample-every N            每 N 帧检测一次，默认 30
--red-gray-min 180          红色通道灰度值下限，旧写法 --r-min
--hue-window 10             距离红色色相的角度范围
--sat-min 60                HSV 饱和度下限
--min-pixels 400            单个采样帧最少红色像素数
--no-hardware               强制使用 CPU 解码
-o, --output                输出 txt 路径
```

## 输出口径

- 首次出现时间向下取整到整秒。
- 持续时长使用“最后一次命中 - 首次命中 + 0.5 秒”补齐采样误差，再向上取整到整秒。
- 如果视频结束时红块仍存在，不额外补 0.5 秒。
- `red_pixels` 是首次命中采样帧里的红色像素数量。
- 两个命中之间只要出现一次未命中，就按两个事件处理。

当前输出的时间粒度是整秒，事件边界最多有约 0.5 秒的量化误差，这是“每 30 帧检测一次”带来的预期误差。
