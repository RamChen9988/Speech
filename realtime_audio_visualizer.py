import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd

# 设置参数
samplerate = 44100  # 采样率
channels = 1        # 单声道
blocksize = 1024    # 每次采集的样本数
duration = 2        # 波形图显示的时长(秒)
num_points = int(samplerate * duration)  # 显示的总样本数


# 初始化音频数据缓冲区
audio_buffer = np.zeros(num_points)

# 设置matplotlib

plt.style.use('seaborn-v0_8')
# 简化字体设置，避免中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(np.arange(num_points), audio_buffer)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(0, num_points)
ax.set_title("实时音频波形图")
ax.set_xlabel(f"时间 ({duration}秒)")
ax.set_ylabel("振幅")

# 音频回调函数
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    
    # 更新缓冲区，移除旧数据，添加新数据
    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata.flatten()

# 动画更新函数
def update_plot(frame):
    line.set_ydata(audio_buffer)
    return line,

# 启动音频流
stream = sd.InputStream(
    samplerate=samplerate,
    channels=channels,
    blocksize=blocksize,
    callback=audio_callback
)

# 启动动画
ani = animation.FuncAnimation(
    fig, update_plot, interval=50, blit=True
)

try:
    with stream:
        plt.show()
except KeyboardInterrupt:
    print("\n程序已停止")
except Exception as e:
    print(f"发生错误: {e}")
