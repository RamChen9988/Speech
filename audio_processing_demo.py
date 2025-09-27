import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from scipy import signal
import os
import requests
import tempfile
import IPython.display as ipd
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class AudioProcessingDemo:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.original_audio = None
        self.processed_audio = None
        self.audio_duration = 0
        
    def download_audio_from_url(self, url, max_duration=10):
        """从URL下载音频文件"""
        try:
            print(f"正在从URL下载音频: {url}")
            
            # 创建下载目录
            download_dir = "downloaded_audio"
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_audio.wav"
            download_path = os.path.join(download_dir, filename)
            
            # 下载文件
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"音频下载完成，保存到: {download_path}")
            
            # 加载音频
            audio, sr = librosa.load(download_path, sr=self.sample_rate, duration=max_duration)
            self.original_audio = audio
            self.sample_rate = sr
            self.audio_duration = len(audio) / sr
            
            print(f"音频加载成功: {self.audio_duration:.2f}秒, 采样率: {self.sample_rate}Hz")
            
            return audio
            
        except Exception as e:
            print(f"下载音频失败: {e}")
            print("使用内置示例音频...")
            return self.load_example_audio()
    
    def load_example_audio(self):
        """加载内置示例音频"""
        try:
            # 使用librosa自带的示例音频
            file_path = librosa.example('trumpet')
            self.original_audio, self.sample_rate = librosa.load(file_path, sr=self.sample_rate)
            self.audio_duration = len(self.original_audio) / self.sample_rate
            print(f"加载示例音频成功: {self.audio_duration:.2f}秒")
            return self.original_audio
        except Exception as e:
            print(f"加载示例音频失败: {e}")
            # 生成测试音频
            return self.generate_test_audio()
    
    def generate_test_audio(self):
        """生成测试音频"""
        duration = 3  # 3秒
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # 生成包含多个频率的复杂信号
        # 基频 + 谐波 + 噪声
        base_freq = 220  # A3
        self.original_audio = (
            0.7 * np.sin(2 * np.pi * base_freq * t) +           # 基频
            0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +       # 二次谐波
            0.2 * np.sin(2 * np.pi * base_freq * 3 * t) +       # 三次谐波
            0.1 * np.random.randn(len(t))                       # 噪声
        )
        self.original_audio = self.original_audio.astype(np.float32)
        self.audio_duration = duration
        
        print("生成测试音频成功")
        return self.original_audio
    
    def record_audio(self, duration=5):
        """录制音频"""
        print(f"开始录音，请说话... ({duration}秒)")
        audio_data = sd.rec(int(duration * self.sample_rate), 
                           samplerate=self.sample_rate, 
                           channels=1)
        sd.wait()
        self.original_audio = audio_data.flatten()
        self.audio_duration = duration
        print("录音完成!")
        return self.original_audio
    
    def apply_pitch_shift(self, audio, n_steps=4):
        """音高变换（产生明显听觉差异）"""
        print(f"应用音高变换: {n_steps} 个半音")
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def apply_time_stretch(self, audio, rate=1.5):
        """时间拉伸（产生明显听觉差异）"""
        print(f"应用时间拉伸: {rate}x 速度")
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def apply_reverb(self, audio, delay=0.1, decay=0.5):
        """添加混响效果"""
        print("添加混响效果")
        # 简单的混响实现
        delayed = np.zeros_like(audio)
        delay_samples = int(delay * self.sample_rate)
        
        if delay_samples < len(audio):
            delayed[delay_samples:] = audio[:-delay_samples] * decay
        
        return audio + delayed
    
    def apply_lowpass_filter(self, audio, cutoff_freq=1000):
        """应用低通滤波器（让声音变闷）"""
        print(f"应用低通滤波器: 截止频率 {cutoff_freq}Hz")
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def apply_highpass_filter(self, audio, cutoff_freq=2000):
        """应用高通滤波器（让声音变尖）"""
        print(f"应用高通滤波器: 截止频率 {cutoff_freq}Hz")
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def apply_distortion(self, audio, gain=5.0):
        """应用失真效果"""
        print("应用失真效果")
        # 简单的软削波失真
        distorted = np.tanh(gain * audio)
        return distorted / np.max(np.abs(distorted))
    
    def add_noise(self, audio, noise_level=0.1):
        """添加噪声（用于创建带噪语音）"""
        print(f"添加噪声，噪声水平: {noise_level}")
        noise = noise_level * np.random.randn(len(audio))
        return audio + noise
    
    def apply_noise_reduction(self, audio, reduction_strength=0.8):
        """应用噪声抑制（谱减法）"""
        print(f"应用噪声抑制，强度: {reduction_strength}")
        
        # 使用谱减法进行噪声抑制
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 估计噪声谱（使用前几帧）
        noise_frames = 10
        noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # 谱减法
        enhanced_magnitude = magnitude - reduction_strength * noise_magnitude
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.01 * magnitude)  # 避免负值
        
        # 重建信号
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def apply_voice_enhancement(self, audio, enhancement_factor=1.5):
        """应用语音增强（提升语音频率）"""
        print(f"应用语音增强，增强因子: {enhancement_factor}")
        
        # 使用带通滤波器增强语音频率范围（300-3400Hz）
        nyquist = self.sample_rate / 2
        low_freq = 300 / nyquist
        high_freq = 3400 / nyquist
        
        # 设计带通滤波器
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        # 增强语音频段
        enhanced_audio = audio + (enhancement_factor - 1) * filtered_audio
        
        # 归一化
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * np.max(np.abs(audio))
        
        return enhanced_audio
    
    def apply_compression(self, audio, threshold=0.5, ratio=4.0):
        """应用动态压缩"""
        print(f"应用动态压缩，阈值: {threshold}, 压缩比: {ratio}:1")
        
        # 简单的动态压缩实现
        compressed = np.copy(audio)
        
        # 对超过阈值的部分进行压缩
        mask = np.abs(audio) > threshold
        compressed[mask] = threshold + (audio[mask] - threshold) / ratio
        
        return compressed
    
    def create_dramatic_effect(self, audio):
        """创建戏剧性的听觉变化效果"""
        print("创建戏剧性听觉变化效果...")
        
        # 组合多种效果
        # 1. 先变调（提高音高）
        processed = self.apply_pitch_shift(audio, n_steps=6)
        
        # 2. 加速播放
        processed = self.apply_time_stretch(processed, rate=1.8)
        
        # 3. 添加失真
        processed = self.apply_distortion(processed, gain=8.0)
        
        # 4. 添加混响
        processed = self.apply_reverb(processed, delay=0.15, decay=0.7)
        
        return processed
    
    def create_noise_cleaning_effect(self, audio):
        """创建噪声清理和语音增强效果"""
        print("创建噪声清理和语音增强效果...")
        
        # 1. 先添加噪声（模拟嘈杂环境）
        noisy_audio = self.add_noise(audio, noise_level=0.15)
        
        # 2. 应用噪声抑制
        cleaned_audio = self.apply_noise_reduction(noisy_audio, reduction_strength=0.7)
        
        # 3. 应用语音增强
        enhanced_audio = self.apply_voice_enhancement(cleaned_audio, enhancement_factor=1.8)
        
        # 4. 应用动态压缩
        final_audio = self.apply_compression(enhanced_audio, threshold=0.3, ratio=3.0)
        
        return final_audio, noisy_audio
    
    def demonstrate_noise_cleaning(self):
        """演示噪声清理效果"""
        if self.original_audio is None:
            print("请先加载音频!")
            return
        
        print("\n" + "="*50)
        print("噪声清理和语音增强演示")
        print("="*50)
        
        # 创建带噪语音并清理
        enhanced_audio, noisy_audio = self.create_noise_cleaning_effect(self.original_audio)
        self.processed_audio = enhanced_audio
        
        # 保存带噪音频用于对比
        self.noisy_audio = noisy_audio
        
        print("\n🎵 播放原始音频...")
        ipd.display(ipd.Audio(self.original_audio, rate=self.sample_rate))
        
        print("🎵 播放带噪音频（模拟嘈杂环境）...")
        ipd.display(ipd.Audio(self.noisy_audio, rate=self.sample_rate))
        
        print("🎵 播放清理后的音频...")
        ipd.display(ipd.Audio(self.processed_audio, rate=self.sample_rate))
        
        # 分析效果
        self.analyze_noise_cleaning_effect()
        
        # 显示对比图表
        self.plot_noise_cleaning_comparison()
        
        # 保存选项
        save_choice = input("\n是否保存音频文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            self.save_noise_cleaning_files()
        
        print("\n噪声清理演示完成!")
    
    def analyze_noise_cleaning_effect(self):
        """分析噪声清理效果"""
        # 确保音频长度一致（截取到最短长度）
        min_length = min(len(self.original_audio), len(self.noisy_audio), len(self.processed_audio))
        orig_audio = self.original_audio[:min_length]
        noisy_audio = self.noisy_audio[:min_length]
        clean_audio = self.processed_audio[:min_length]
        
        orig_rms = np.sqrt(np.mean(orig_audio**2))
        noisy_rms = np.sqrt(np.mean(noisy_audio**2))
        clean_rms = np.sqrt(np.mean(clean_audio**2))
        
        # 计算信噪比改进
        noise_power = np.mean((noisy_audio - orig_audio)**2)
        signal_power = np.mean(orig_audio**2)
        original_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        clean_noise_power = np.mean((clean_audio - orig_audio)**2)
        clean_snr = 10 * np.log10(signal_power / clean_noise_power) if clean_noise_power > 0 else float('inf')
        
        print(f"\n🎯 噪声清理效果分析:")
        print(f"原始音频 RMS: {orig_rms:.4f}")
        print(f"带噪音频 RMS: {noisy_rms:.4f} (+{((noisy_rms/orig_rms)-1)*100:.1f}%)")
        print(f"清理后音频 RMS: {clean_rms:.4f} (+{((clean_rms/orig_rms)-1)*100:.1f}%)")
        
        if original_snr != float('inf'):
            print(f"带噪音频信噪比: {original_snr:.1f} dB")
            print(f"清理后音频信噪比: {clean_snr:.1f} dB")
            print(f"信噪比改进: {clean_snr - original_snr:.1f} dB")
        
        # 频谱质心对比
        orig_centroid = librosa.feature.spectral_centroid(y=orig_audio, sr=self.sample_rate)[0]
        noisy_centroid = librosa.feature.spectral_centroid(y=noisy_audio, sr=self.sample_rate)[0]
        clean_centroid = librosa.feature.spectral_centroid(y=clean_audio, sr=self.sample_rate)[0]
        
        print(f"原始音频频谱质心: {np.mean(orig_centroid):.1f} Hz")
        print(f"带噪音频频谱质心: {np.mean(noisy_centroid):.1f} Hz")
        print(f"清理后音频频谱质心: {np.mean(clean_centroid):.1f} Hz")
    
    def plot_noise_cleaning_comparison(self):
        """绘制噪声清理对比图"""
        # 确保音频长度一致（截取到最短长度）
        min_length = min(len(self.original_audio), len(self.noisy_audio), len(self.processed_audio))
        orig_audio = self.original_audio[:min_length]
        noisy_audio = self.noisy_audio[:min_length]
        clean_audio = self.processed_audio[:min_length]
        
        plt.figure(figsize=(18, 12))
        
        # 设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 波形对比
        plt.subplot(3, 3, 1)
        time_axis = np.arange(len(orig_audio)) / self.sample_rate
        plt.plot(time_axis, orig_audio, alpha=0.7, label='原始')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title('原始音频波形')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        plt.plot(time_axis, noisy_audio, alpha=0.7, color='orange', label='带噪')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title('带噪音频波形')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.plot(time_axis, clean_audio, alpha=0.7, color='green', label='清理后')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title('清理后音频波形')
        plt.grid(True, alpha=0.3)
        
        # 2. 频谱对比
        plt.subplot(3, 3, 4)
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig_audio)), ref=np.max)
        librosa.display.specshow(D_orig, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('原始音频频谱')
        plt.ylim(0, 8000)
        
        plt.subplot(3, 3, 5)
        D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
        librosa.display.specshow(D_noisy, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('带噪音频频谱')
        plt.ylim(0, 8000)
        
        plt.subplot(3, 3, 6)
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_audio)), ref=np.max)
        librosa.display.specshow(D_clean, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('清理后音频频谱')
        plt.ylim(0, 8000)
        
        # 3. 频谱包络对比
        plt.subplot(3, 3, 7)
        f, Pxx_orig = signal.welch(orig_audio, self.sample_rate, nperseg=1024)
        f, Pxx_noisy = signal.welch(noisy_audio, self.sample_rate, nperseg=1024)
        f, Pxx_clean = signal.welch(clean_audio, self.sample_rate, nperseg=1024)
        
        plt.semilogy(f, Pxx_orig, label='原始', alpha=0.8)
        plt.semilogy(f, Pxx_noisy, label='带噪', alpha=0.8)
        plt.semilogy(f, Pxx_clean, label='清理后', alpha=0.8)
        plt.xlim(0, 8000)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title('频谱包络对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 噪声谱对比
        plt.subplot(3, 3, 8)
        noise_spectrum = np.abs(librosa.stft(noisy_audio - orig_audio))
        librosa.display.specshow(librosa.amplitude_to_db(noise_spectrum, ref=np.max), 
                                sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('原始噪声谱')
        plt.ylim(0, 8000)
        
        plt.subplot(3, 3, 9)
        clean_noise_spectrum = np.abs(librosa.stft(clean_audio - orig_audio))
        librosa.display.specshow(librosa.amplitude_to_db(clean_noise_spectrum, ref=np.max), 
                                sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('残留噪声谱')
        plt.ylim(0, 8000)
        
        plt.tight_layout()
        plt.show()
    
    def save_noise_cleaning_files(self, prefix="noise_cleaning"):
        """保存噪声清理相关音频文件"""
        # 创建输出目录
        output_dir = "processed_audio"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 确保文件不会覆盖
        timestamp = np.random.randint(1000, 9999)
        orig_filename = os.path.join(output_dir, f"{prefix}_original_{timestamp}.wav")
        noisy_filename = os.path.join(output_dir, f"{prefix}_noisy_{timestamp}.wav")
        clean_filename = os.path.join(output_dir, f"{prefix}_cleaned_{timestamp}.wav")
        
        # 保存所有音频
        sf.write(orig_filename, self.original_audio, self.sample_rate)
        sf.write(noisy_filename, self.noisy_audio, self.sample_rate)
        sf.write(clean_filename, self.processed_audio, self.sample_rate)
        
        print(f"原始音频已保存: {orig_filename}")
        print(f"带噪音频已保存: {noisy_filename}")
        print(f"清理后音频已保存: {clean_filename}")
        print(f"所有音频文件已保存到 '{output_dir}' 目录中")
        
        return orig_filename, noisy_filename, clean_filename
    
    def play_audio_comparison(self):
        """播放原始和处理后的音频对比"""
        if self.original_audio is None or self.processed_audio is None:
            print("请先加载音频并应用处理!")
            return
        
        print("\n" + "="*50)
        print("音频对比播放")
        print("="*50)
        
        print("🎵 播放原始音频...")
        ipd.display(ipd.Audio(self.original_audio, rate=self.sample_rate))
        
        print("🎵 播放处理后的音频...")
        ipd.display(ipd.Audio(self.processed_audio, rate=self.sample_rate))
        
        # 计算一些音频特征用于对比
        self.analyze_audio_differences()
    
    def analyze_audio_differences(self):
        """分析音频差异"""
        orig_rms = np.sqrt(np.mean(self.original_audio**2))
        proc_rms = np.sqrt(np.mean(self.processed_audio**2))
        
        # 频谱对比
        orig_spectrum = np.abs(librosa.stft(self.original_audio))
        proc_spectrum = np.abs(librosa.stft(self.processed_audio))
        
        print(f"\n音频特征对比:")
        print(f"原始音频 RMS: {orig_rms:.4f}")
        print(f"处理后音频 RMS: {proc_rms:.4f}")
        print(f"音量变化: {proc_rms/orig_rms:.2f}x")
        
        # 频谱质心对比
        orig_centroid = librosa.feature.spectral_centroid(y=self.original_audio, sr=self.sample_rate)[0]
        proc_centroid = librosa.feature.spectral_centroid(y=self.processed_audio, sr=self.sample_rate)[0]
        
        print(f"原始音频频谱质心: {np.mean(orig_centroid):.1f} Hz")
        print(f"处理后音频频谱质心: {np.mean(proc_centroid):.1f} Hz")
    
    def plot_comprehensive_comparison(self):
        """绘制全面的音频对比图"""
        if self.original_audio is None or self.processed_audio is None:
            print("请先加载音频并应用处理!")
            return
        
        plt.figure(figsize=(16, 12))
        
        # 设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 波形对比
        plt.subplot(3, 2, 1)
        time_axis = np.arange(len(self.original_audio)) / self.sample_rate
        plt.plot(time_axis, self.original_audio, alpha=0.7, label='原始')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title('原始音频波形')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 2)
        time_axis_proc = np.arange(len(self.processed_audio)) / self.sample_rate
        plt.plot(time_axis_proc, self.processed_audio, alpha=0.7, color='red', label='处理后')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.title('处理后音频波形')
        plt.grid(True, alpha=0.3)
        
        # 2. 频谱对比
        plt.subplot(3, 2, 3)
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(self.original_audio)), ref=np.max)
        librosa.display.specshow(D_orig, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('原始音频频谱')
        plt.ylim(0, 8000)
        
        plt.subplot(3, 2, 4)
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(self.processed_audio)), ref=np.max)
        librosa.display.specshow(D_proc, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('处理后音频频谱')
        plt.ylim(0, 8000)
        
        # 3. 频谱包络对比
        plt.subplot(3, 2, 5)
        f, Pxx_orig = signal.welch(self.original_audio, self.sample_rate, nperseg=1024)
        f, Pxx_proc = signal.welch(self.processed_audio, self.sample_rate, nperseg=1024)
        
        plt.semilogy(f, Pxx_orig, label='原始', alpha=0.8)
        plt.semilogy(f, Pxx_proc, label='处理后', alpha=0.8)
        plt.xlim(0, 8000)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title('频谱包络对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. MFCC特征对比（分别显示）
        plt.subplot(3, 2, 6)
        mfcc_orig = librosa.feature.mfcc(y=self.original_audio, sr=self.sample_rate, n_mfcc=13)
        librosa.display.specshow(mfcc_orig, sr=self.sample_rate, x_axis='time')
        plt.colorbar()
        plt.title('原始音频MFCC')
        
        plt.tight_layout()
        plt.show()
    
    def save_audio_files(self, prefix="audio_comparison"):
        """保存音频文件"""
        if self.original_audio is None or self.processed_audio is None:
            print("请先加载音频并应用处理!")
            return
        
        # 创建输出目录
        output_dir = "processed_audio"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 确保文件不会覆盖
        timestamp = np.random.randint(1000, 9999)
        orig_filename = os.path.join(output_dir, f"{prefix}_original_{timestamp}.wav")
        proc_filename = os.path.join(output_dir, f"{prefix}_processed_{timestamp}.wav")
        
        # 保存原始音频
        sf.write(orig_filename, self.original_audio, self.sample_rate)
        print(f"原始音频已保存: {orig_filename}")
        
        # 保存处理后的音频
        sf.write(proc_filename, self.processed_audio, self.sample_rate)
        print(f"处理后音频已保存: {proc_filename}")
        
        print(f"所有音频文件已保存到 '{output_dir}' 目录中")
        return orig_filename, proc_filename
    
    def interactive_demo(self):
        """交互式演示"""
        print("🎵 音频处理演示程序")
        print("="*50)
        
        # 选择音频来源
        print("选择音频来源:")
        print("1. 从URL下载音频")
        print("2. 使用内置示例音频")
        print("3. 实时录音")
        print("4. 生成测试音频")
        
        source_choice = input("请输入选择 (1-4): ").strip()
        
        if source_choice == "1":
            url = input("请输入音频URL: ").strip()
            self.download_audio_from_url(url)
        elif source_choice == "2":
            self.load_example_audio()
        elif source_choice == "3":
            duration = float(input("请输入录音时长 (秒): ").strip() or "5")
            self.record_audio(duration=duration)
        else:
            self.generate_test_audio()
        
        # 选择处理效果
        print("\n选择处理效果:")
        print("1. 戏剧性变化 (推荐 - 音高变换+时间拉伸+失真+混响)")
        print("2. 音高变换")
        print("3. 时间拉伸")
        print("4. 低通滤波")
        print("5. 高通滤波")
        print("6. 失真效果")
        print("7. 混响效果")
        
        effect_choice = input("请输入选择 (1-7): ").strip()
        
        if effect_choice == "1":
            self.processed_audio = self.create_dramatic_effect(self.original_audio)
        elif effect_choice == "2":
            steps = int(input("请输入音高变化半音数 (正数提高, 负数降低): ").strip() or "4")
            self.processed_audio = self.apply_pitch_shift(self.original_audio, n_steps=steps)
        elif effect_choice == "3":
            rate = float(input("请输入时间拉伸比率 (>1加速, <1减速): ").strip() or "1.5")
            self.processed_audio = self.apply_time_stretch(self.original_audio, rate=rate)
        elif effect_choice == "4":
            cutoff = int(input("请输入低通滤波截止频率 (Hz): ").strip() or "1000")
            self.processed_audio = self.apply_lowpass_filter(self.original_audio, cutoff_freq=cutoff)
        elif effect_choice == "5":
            cutoff = int(input("请输入高通滤波截止频率 (Hz): ").strip() or "2000")
            self.processed_audio = self.apply_highpass_filter(self.original_audio, cutoff_freq=cutoff)
        elif effect_choice == "6":
            gain = float(input("请输入失真增益: ").strip() or "5.0")
            self.processed_audio = self.apply_distortion(self.original_audio, gain=gain)
        elif effect_choice == "7":
            delay = float(input("请输入混响延迟 (秒): ").strip() or "0.1")
            decay = float(input("请输入混响衰减: ").strip() or "0.5")
            self.processed_audio = self.apply_reverb(self.original_audio, delay=delay, decay=decay)
        else:
            print("使用默认戏剧性变化效果")
            self.processed_audio = self.create_dramatic_effect(self.original_audio)
        
        # 显示结果
        print("\n" + "="*50)
        print("处理完成!")
        print("="*50)
        
        # 播放对比
        self.play_audio_comparison()
        
        # 显示图表
        print("\n生成对比图表...")
        self.plot_comprehensive_comparison()
        
        # 保存选项
        save_choice = input("\n是否保存音频文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            self.save_audio_files()
        
        print("\n演示完成!")

# 示例URL列表（可以使用的音频资源）
EXAMPLE_URLS = [
    "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/ImperialMarch60.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"
]

def quick_demo_with_example_url():
    """使用示例URL快速演示"""
    demo = AudioProcessingDemo()
    
    print("🎵 快速音频处理演示")
    print("="*50)
    print("可用的示例音频:")
    for i, url in enumerate(EXAMPLE_URLS, 1):
        filename = os.path.basename(urlparse(url).path)
        print(f"{i}. {filename}")
    
    choice = int(input("请选择音频 (1-4): ").strip() or "1")
    url = EXAMPLE_URLS[choice-1]
    
    # 下载并处理音频
    demo.download_audio_from_url(url)
    demo.processed_audio = demo.create_dramatic_effect(demo.original_audio)
    
    # 播放对比
    demo.play_audio_comparison()
    demo.plot_comprehensive_comparison()
    
    # 保存文件
    demo.save_audio_files()

def noise_cleaning_demo():
    """噪声清理演示"""
    demo = AudioProcessingDemo()
    
    print("🎵 噪声清理和语音增强演示")
    print("="*50)
    print("可用的示例音频:")
    for i, url in enumerate(EXAMPLE_URLS, 1):
        filename = os.path.basename(urlparse(url).path)
        print(f"{i}. {filename}")
    
    choice = int(input("请选择音频 (1-4): ").strip() or "1")
    url = EXAMPLE_URLS[choice-1]
    
    # 下载音频
    demo.download_audio_from_url(url)
    
    # 演示噪声清理效果
    demo.demonstrate_noise_cleaning()

# 主程序
if __name__ == "__main__":
    # 安装所需库的命令:
    # pip install numpy matplotlib librosa sounddevice soundfile scipy requests ipython
    
    print("音频处理演示程序")
    print("="*50)
    print("功能特点:")
    print("• 支持从URL下载音频")
    print("• 多种音频处理效果")
    print("• 听觉明显不同的前后对比")
    print("• 可视化分析")
    print("• 音频文件保存")
    print("="*50)
    
    demo = AudioProcessingDemo()
    
    # 选择演示模式
    print("选择演示模式:")
    print("1. 交互式演示 (推荐)")
    print("2. 快速示例演示")
    print("3. 噪声清理演示")
    print("4. 自定义处理")
    
    mode_choice = input("请输入选择 (1-4): ").strip() or "1"
    
    if mode_choice == "1":
        demo.interactive_demo()
    elif mode_choice == "2":
        quick_demo_with_example_url()
    elif mode_choice == "3":
        noise_cleaning_demo()
    else:
        # 自定义处理模式
        print("自定义处理模式")
        url = input("请输入音频URL (留空使用示例音频): ").strip()
        
        if url:
            demo.download_audio_from_url(url)
        else:
            demo.load_example_audio()
        
        # 应用戏剧性效果
        demo.processed_audio = demo.create_dramatic_effect(demo.original_audio)
        
        # 显示结果
        demo.play_audio_comparison()
        demo.plot_comprehensive_comparison()
        
        save_choice = input("是否保存音频文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            demo.save_audio_files()
    
    print("\n程序执行完成!")
