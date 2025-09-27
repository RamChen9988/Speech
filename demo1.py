import librosa
import matplotlib.pyplot as plt

# 读取音频文件
audio, sr = librosa.load('1.mp3', sr=None)

# 打印音频信息
print(f"采样率: {sr}Hz")
print(f"音频时长: {len(audio)/sr:.2f}秒")

# 绘制波形图
# 简化字体设置，避免中文显示问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 4))
plt.plot(audio)
plt.title("音频波形图")
plt.xlabel("样本数")
plt.ylabel("振幅")
plt.show()

# 将音频从44.1kHz转换为16kHz
audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# 保存转换后的音频
import soundfile as sf
sf.write('1.wav', audio_16k, 16000)

# 对比听感
print("播放原始音频...")
ipd.Audio(audio, rate=sr)
print("播放16kHz音频...")
ipd.Audio(audio_16k, rate=16000)