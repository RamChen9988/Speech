import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from scipy import signal
import os
from datetime import datetime
import IPython.display as ipd

class VoicePreprocessing:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_data = None
        self.preprocessed_data = None
        self.mfcc_features = None
        
    def record_audio(self, duration=3, sample_rate=22050):
        """录制音频"""
        print(f"开始录音，请说话... ({duration}秒)")
        self.sample_rate = sample_rate
        self.audio_data = sd.rec(int(duration * sample_rate), 
                                samplerate=sample_rate, 
                                channels=1)
        sd.wait()  # 等待录音完成
        self.audio_data = self.audio_data.flatten()
        print("录音完成!")
        return self.audio_data
    
    def load_example_audio(self):
        """加载示例音频（使用librosa自带的示例）"""
        try:
            # 使用librosa自带的示例音频
            file_path = librosa.example('trumpet')
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=self.sample_rate)
            print(f"加载示例音频成功，长度: {len(self.audio_data)/self.sample_rate:.2f}秒")
            return self.audio_data
        except:
            # 如果无法加载示例，生成一个简单的测试音频
            print("无法加载示例音频，生成测试音频...")
            t = np.linspace(0, 3, 3 * self.sample_rate)
            # 生成包含多个频率的测试信号
            self.audio_data = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
            self.audio_data = self.audio_data.astype(np.float32)
            return self.audio_data
    
    def preemphasis(self, signal, alpha=0.97):
        """预加重滤波"""
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])
    
    def framing(self, signal, frame_length=0.025, frame_step=0.01):
        """分帧处理"""
        frame_length = int(frame_length * self.sample_rate)
        frame_step = int(frame_step * self.sample_rate)
        
        signal_length = len(signal)
        frames = []
        
        for i in range(0, signal_length - frame_length, frame_step):
            frames.append(signal[i:i+frame_length])
        
        return np.array(frames)
    
    def apply_window(self, frames, window_type='hamming'):
        """应用窗函数"""
        frame_length = frames.shape[1]
        
        if window_type == 'hamming':
            window = np.hamming(frame_length)
        elif window_type == 'hann':
            window = np.hann(frame_length)
        else:  # rectangular
            window = np.ones(frame_length)
        
        return frames * window
    
    def extract_mfcc(self, audio, n_mfcc=13, n_fft=2048, hop_length=512):
        """提取MFCC特征"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, 
                                   n_mfcc=n_mfcc, n_fft=n_fft, 
                                   hop_length=hop_length)
        return mfccs
    
    def plot_waveform(self, audio, title="音频波形"):
        """绘制音频波形图"""
        plt.figure(figsize=(12, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.plot(audio)
        plt.title(title)
        plt.xlabel("样本数")
        plt.ylabel("振幅")
        plt.tight_layout()
        plt.show()
    
    def plot_spectrogram(self, audio, title="频谱图"):
        """绘制频谱图"""
        plt.figure(figsize=(12, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_mfcc(self, mfccs, title="MFCC特征"):
        """绘制MFCC特征图"""
        plt.figure(figsize=(12, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        librosa.display.specshow(mfccs, sr=self.sample_rate, x_axis='time')
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def compare_preemphasis(self):
        """比较预加重前后的效果"""
        if self.audio_data is None:
            print("请先加载或录制音频!")
            return
        
        # 应用预加重
        audio_pre = self.preemphasis(self.audio_data)
        
        # 绘制对比图
        plt.figure(figsize=(15, 10))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 原始音频波形
        plt.subplot(3, 2, 1)
        plt.plot(self.audio_data[:1000])
        plt.title("原始音频波形 (前1000个样本)")
        plt.xlabel("样本数")
        plt.ylabel("振幅")
        
        # 预加重后波形
        plt.subplot(3, 2, 2)
        plt.plot(audio_pre[:1000])
        plt.title("预加重后波形 (前1000个样本)")
        plt.xlabel("样本数")
        plt.ylabel("振幅")
        
        # 原始音频频谱
        plt.subplot(3, 2, 3)
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        librosa.display.specshow(D_orig, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title("原始音频频谱")
        
        # 预加重后频谱
        plt.subplot(3, 2, 4)
        D_pre = librosa.amplitude_to_db(np.abs(librosa.stft(audio_pre)), ref=np.max)
        librosa.display.specshow(D_pre, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title("预加重后频谱")
        
        # 频谱对比（高频区域）
        plt.subplot(3, 2, 5)
        f, Pxx_orig = signal.welch(self.audio_data, self.sample_rate, nperseg=1024)
        f, Pxx_pre = signal.welch(audio_pre, self.sample_rate, nperseg=1024)
        plt.semilogy(f, Pxx_orig, label='原始')
        plt.semilogy(f, Pxx_pre, label='预加重')
        plt.xlim(0, 5000)  # 聚焦在0-5kHz范围
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title('频谱对比 (0-5kHz)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return audio_pre
    
    def demonstrate_framing_window(self, audio):
        """演示分帧和加窗效果"""
        # 分帧
        frames = self.framing(audio)
        print(f"音频被分成 {len(frames)} 帧")
        
        # 应用汉明窗
        windowed_frames = self.apply_window(frames)
        
        # 绘制对比
        plt.figure(figsize=(15, 6))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 原始帧
        plt.subplot(2, 2, 1)
        plt.plot(frames[10])
        plt.title("第10帧 (原始)")
        plt.xlabel("样本数")
        plt.ylabel("振幅")
        
        # 加窗后的帧
        plt.subplot(2, 2, 2)
        plt.plot(windowed_frames[10])
        plt.title("第10帧 (加汉明窗后)")
        plt.xlabel("样本数")
        plt.ylabel("振幅")
        
        # 频谱对比
        plt.subplot(2, 2, 3)
        D_frame = librosa.amplitude_to_db(np.abs(librosa.stft(frames[10])), ref=np.max)
        librosa.display.specshow(D_frame, sr=self.sample_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title("原始帧频谱")
        
        plt.subplot(2, 2, 4)
        D_windowed = librosa.amplitude_to_db(np.abs(librosa.stft(windowed_frames[10])), ref=np.max)
        librosa.display.specshow(D_windowed, sr=self.sample_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title("加窗后帧频谱")
        
        plt.tight_layout()
        plt.show()
        
        return frames, windowed_frames
    
    def full_pipeline(self, use_recorded=True):
        """完整的处理流程"""
        if use_recorded:
            print("=== 开始录音 ===")
            self.record_audio(duration=3)
        else:
            print("=== 加载示例音频 ===")
            self.load_example_audio()
        
        print("=== 播放原始音频 ===")
        ipd.display(ipd.Audio(self.audio_data, rate=self.sample_rate))
        
        print("=== 绘制原始音频波形和频谱 ===")
        self.plot_waveform(self.audio_data, "原始音频波形")
        self.plot_spectrogram(self.audio_data, "原始音频频谱")
        
        print("=== 预加重处理 ===")
        audio_pre = self.compare_preemphasis()
        
        print("=== 分帧和加窗演示 ===")
        frames, windowed_frames = self.demonstrate_framing_window(audio_pre)
        
        print("=== 提取MFCC特征 ===")
        self.mfcc_features = self.extract_mfcc(audio_pre)
        print(f"MFCC特征形状: {self.mfcc_features.shape}")
        
        print("=== 显示MFCC特征 ===")
        self.plot_mfcc(self.mfcc_features, "MFCC特征")
        
        print("=== 处理完成! ===")
        
        return self.mfcc_features

# 创建可视化比较函数
def compare_different_voices():
    """比较不同语音的MFCC特征"""
    vp1 = VoicePreprocessing()
    vp2 = VoicePreprocessing()
    
    print("请录制第一段语音（例如说'啊'）")
    vp1.record_audio(duration=2)
    mfcc1 = vp1.extract_mfcc(vp1.audio_data)
    
    print("请录制第二段语音（例如说'咿'）")
    vp2.record_audio(duration=2)
    mfcc2 = vp2.extract_mfcc(vp2.audio_data)
    
    # 绘制对比
    plt.figure(figsize=(15, 6))
    # 简化字体设置，避免中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mfcc1, sr=vp1.sample_rate, x_axis='time')
    plt.colorbar()
    plt.title("第一段语音的MFCC ('啊')")
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfcc2, sr=vp2.sample_rate, x_axis='time')
    plt.colorbar()
    plt.title("第二段语音的MFCC ('咿')")
    
    plt.tight_layout()
    plt.show()
    
    # 播放两段音频用于对比
    print("播放第一段音频:")
    ipd.display(ipd.Audio(vp1.audio_data, rate=vp1.sample_rate))
    print("播放第二段音频:")
    ipd.display(ipd.Audio(vp2.audio_data, rate=vp2.sample_rate))

# 实时环境噪声分析
def analyze_environment_noise():
    """分析环境噪声"""
    vp = VoicePreprocessing()
    
    print("录制3秒环境噪声（请保持安静）...")
    noise = vp.record_audio(duration=3)
    
    print("录制3秒带语音的音频（请在安静后说话）...")
    speech = vp.record_audio(duration=3)
    
    # 分析噪声和语音的频谱差异
    plt.figure(figsize=(15, 8))
    # 简化字体设置，避免中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 噪声频谱
    plt.subplot(2, 2, 1)
    D_noise = librosa.amplitude_to_db(np.abs(librosa.stft(noise)), ref=np.max)
    librosa.display.specshow(D_noise, sr=vp.sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("环境噪声频谱")
    
    # 语音频谱
    plt.subplot(2, 2, 2)
    D_speech = librosa.amplitude_to_db(np.abs(librosa.stft(speech)), ref=np.max)
    librosa.display.specshow(D_speech, sr=vp.sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("带噪语言频谱")
    
    # MFCC对比
    plt.subplot(2, 2, 3)
    mfcc_noise = vp.extract_mfcc(noise)
    librosa.display.specshow(mfcc_noise, sr=vp.sample_rate, x_axis='time')
    plt.colorbar()
    plt.title("噪声MFCC")
    
    plt.subplot(2, 2, 4)
    mfcc_speech = vp.extract_mfcc(speech)
    librosa.display.specshow(mfcc_speech, sr=vp.sample_rate, x_axis='time')
    plt.colorbar()
    plt.title("带噪语音MFCC")
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 安装所需库的命令（在运行前需要安装）:
    # pip install numpy matplotlib librosa sounddevice soundfile scipy ipython
    
    print("语音信号预处理与MFCC特征提取项目")
    print("=" * 50)
    
    # 创建预处理对象
    vp = VoicePreprocessing()
    
    # 选择处理方式
    choice = input("请选择处理方式:\n1. 使用示例音频\n2. 实时录音\n请输入选择 (1或2): ")
    
    if choice == "1":
        # 使用示例音频
        vp.full_pipeline(use_recorded=False)
    else:
        # 实时录音
        vp.full_pipeline(use_recorded=True)
    
    # 提供额外功能选择
    extra_choice = input("\n是否运行额外演示?\n1. 比较不同语音\n2. 分析环境噪声\n3. 退出\n请输入选择 (1,2或3): ")
    
    if extra_choice == "1":
        compare_different_voices()
    elif extra_choice == "2":
        analyze_environment_noise()
    
    print("项目演示结束!")