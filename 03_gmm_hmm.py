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
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

class AcousticModel:
    def __init__(self, n_components=3, n_mfcc=13):
        self.n_components = n_components  # HMM状态数
        self.n_mfcc = n_mfcc  # MFCC特征维度
        self.models = {}  # 存储每个音素的HMM模型
        self.gmms = {}  # 存储每个音素的GMM模型
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, audio, sr=22050):
        """提取MFCC特征"""
        # 预加重
        audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=audio_pre, sr=sr, n_mfcc=self.n_mfcc, 
                                   n_fft=2048, hop_length=512)
        return mfccs.T  # 转置为 (时间帧数, 特征维度)
    
    def train_gmm(self, features, n_mixtures=3):
        """训练GMM模型"""
        gmm = GaussianMixture(n_components=n_mixtures, covariance_type='diag')
        gmm.fit(features)
        return gmm
    
    def train_hmm(self, features_list, n_components=3):
        """训练HMM模型"""
        # 计算所有特征序列的长度
        lengths = [len(features) for features in features_list]
        
        # 合并所有特征
        features_combined = np.vstack(features_list)
        
        # 创建并训练HMM模型
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        
        model.fit(features_combined, lengths)
        return model
    
    def train_models(self, training_data):
        """训练所有音素的GMM-HMM模型"""
        print("开始训练声学模型...")
        
        # 收集所有特征用于标准化
        all_features = []
        for phoneme, features_list in training_data.items():
            for features in features_list:
                all_features.append(features)
        
        # 标准化特征
        all_features_combined = np.vstack(all_features)
        self.scaler.fit(all_features_combined)
        
        # 为每个音素训练模型
        for phoneme, features_list in training_data.items():
            print(f"训练音素 '{phoneme}' 的模型...")
            
            # 标准化特征
            normalized_features_list = []
            for features in features_list:
                normalized_features = self.scaler.transform(features)
                normalized_features_list.append(normalized_features)
            
            # 训练GMM
            features_combined = np.vstack(normalized_features_list)
            self.gmms[phoneme] = self.train_gmm(features_combined)
            
            # 训练HMM
            self.models[phoneme] = self.train_hmm(normalized_features_list)
        
        self.is_trained = True
        print("模型训练完成!")
    
    def predict(self, audio, sr=22050):
        """使用训练好的模型进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_models方法")
        
        # 提取特征
        features = self.extract_features(audio, sr)
        features = self.scaler.transform(features)
        
        # 计算每个模型的分数
        scores = {}
        for phoneme, model in self.models.items():
            try:
                score = model.score(features)
                scores[phoneme] = score
            except:
                scores[phoneme] = -np.inf
        
        # 返回分数最高的音素
        best_phoneme = max(scores, key=scores.get)
        return best_phoneme, scores
    
    def save_models(self, filepath):
        """保存训练好的模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'gmms': self.gmms,
                'scaler': self.scaler,
                'n_components': self.n_components,
                'n_mfcc': self.n_mfcc,
                'is_trained': self.is_trained
            }, f)
        print(f"模型已保存到 {filepath}")
    
    def load_models(self, filepath):
        """加载训练好的模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.gmms = data['gmms']
        self.scaler = data['scaler']
        self.n_components = data['n_components']
        self.n_mfcc = data['n_mfcc']
        self.is_trained = data['is_trained']
        print(f"模型已从 {filepath} 加载")

class SpeechProject:
    def __init__(self):
        self.preprocessor = VoicePreprocessing()
        self.acoustic_model = AcousticModel()
        self.dataset_loaded = False
        
    def download_timit_dataset(self):
        """下载TIMIT数据集的小样本（由于完整数据集较大，这里使用简化版本）"""
        print("正在下载TIMIT示例数据集...")
        
        # 这里我们使用一个简化的方法：生成模拟数据
        # 在实际教学中，可以使用小型的语音命令数据集
        
        # 创建模拟数据目录
        os.makedirs('data/timit_sample', exist_ok=True)
        
        # 生成一些简单的音素数据用于演示
        self._generate_sample_data()
        
        print("示例数据准备完成!")
        self.dataset_loaded = True
    
    def _generate_sample_data(self):
        """生成用于演示的示例音素数据"""
        # 定义几个基本音素
        phonemes = ['aa', 'iy', 'uw', 'eh', 'ah']
        
        # 为每个音素生成一些示例音频
        sr = 22050
        duration = 0.5  # 0.5秒
        
        for i, phoneme in enumerate(phonemes):
            phoneme_dir = f'data/timit_sample/{phoneme}'
            os.makedirs(phoneme_dir, exist_ok=True)
            
            # 为每个音素生成5个变体
            for j in range(5):
                # 生成不同频率的音频来模拟不同音素
                base_freq = 200 + i * 100  # 每个音素有不同的基础频率
                t = np.linspace(0, duration, int(sr * duration))
                
                # 生成包含基频和泛音的音频
                audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
                audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
                audio += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
                
                # 添加一些噪声模拟真实语音
                noise = 0.05 * np.random.randn(len(audio))
                audio += noise
                
                # 保存音频文件
                filename = f'{phoneme_dir}/{phoneme}_{j:02d}.wav'
                sf.write(filename, audio, sr)
    
    def load_training_data(self):
        """加载训练数据"""
        if not self.dataset_loaded:
            self.download_timit_dataset()
        
        training_data = {}
        
        # 遍历所有音素目录
        phoneme_dirs = [d for d in os.listdir('data/timit_sample') 
                       if os.path.isdir(os.path.join('data/timit_sample', d))]
        
        for phoneme in phoneme_dirs:
            phoneme_path = f'data/timit_sample/{phoneme}'
            audio_files = [f for f in os.listdir(phoneme_path) if f.endswith('.wav')]
            
            features_list = []
            for audio_file in audio_files:
                filepath = os.path.join(phoneme_path, audio_file)
                audio, sr = librosa.load(filepath, sr=22050)
                
                # 提取特征
                features = self.acoustic_model.extract_features(audio, sr)
                features_list.append(features)
            
            training_data[phoneme] = features_list
        
        return training_data
    
    def demonstrate_gmm(self):
        """演示GMM的工作原理"""
        print("=== GMM演示 ===")
        
        # 生成模拟的MFCC特征数据（二维以便可视化）
        np.random.seed(42)
        
        # 生成三个不同的高斯分布
        n_samples = 300
        mean1, cov1 = [1, 1], [[0.5, 0], [0, 0.5]]
        mean2, cov2 = [4, 4], [[0.5, 0.1], [0.1, 0.5]]
        mean3, cov3 = [1, 4], [[0.3, -0.1], [-0.1, 0.3]]
        
        data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
        data2 = np.random.multivariate_normal(mean2, cov2, n_samples)
        data3 = np.random.multivariate_normal(mean3, cov3, n_samples)
        
        # 合并数据
        X = np.vstack([data1, data2, data3])
        
        # 训练GMM
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        # 可视化
        plt.figure(figsize=(15, 5))
        # 设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 原始数据
        plt.subplot(1, 3, 1)
        plt.scatter(data1[:, 0], data1[:, 1], alpha=0.6, label='类别1')
        plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label='类别2')
        plt.scatter(data3[:, 0], data3[:, 1], alpha=0.6, label='类别3')
        plt.title('原始数据分布')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        plt.legend()
        
        # GMM预测结果
        plt.subplot(1, 3, 2)
        labels = gmm.predict(X)
        colors = ['red', 'blue', 'green']
        for i in range(3):
            plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                       alpha=0.6, color=colors[i], label=f'GMM类别{i+1}')
        plt.title('GMM分类结果')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        plt.legend()
        
        # GMM概率分布
        plt.subplot(1, 3, 3)
        x = np.linspace(-1, 6, 100)
        y = np.linspace(-1, 6, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
        Z = gmm.score_samples(XX)
        Z = Z.reshape(X_grid.shape)
        
        plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(label='对数概率密度')
        plt.scatter(X[:, 0], X[:, 1], alpha=0.3, color='white', s=10)
        plt.title('GMM概率密度分布')
        plt.xlabel('MFCC系数1')
        plt.ylabel('MFCC系数2')
        
        plt.tight_layout()
        plt.show()
        
        print("GMM参数:")
        print(f"权重: {gmm.weights_}")
        print(f"均值: {gmm.means_}")
        print(f"协方差: {gmm.covariances_}")
    
    def demonstrate_hmm(self):
        """演示HMM的工作原理"""
        print("=== HMM演示 ===")
        
        # 生成模拟的语音特征序列
        np.random.seed(42)
        
        # 假设有3个状态（如：音素的开始、中间、结束）
        n_states = 3
        n_features = 2  # 简化特征维度
        
        # 生成训练数据：多个序列
        sequences = []
        lengths = []
        
        for _ in range(50):  # 50个序列
            seq_length = np.random.randint(20, 50)
            lengths.append(seq_length)
            
            # 生成序列数据，每个状态有不同的均值
            sequence = []
            for i in range(seq_length):
                # 模拟状态转移：简单的前向转移
                state = min(i // (seq_length // n_states), n_states - 1)
                mean = [state * 2, state * 2]  # 每个状态有不同的均值
                cov = [[0.5, 0], [0, 0.5]]
                sample = np.random.multivariate_normal(mean, cov)
                sequence.append(sample)
            
            sequences.append(np.array(sequence))
        
        # 合并所有序列
        X = np.vstack(sequences)
        
        # 训练HMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        model.fit(X, lengths)
        
        # 可视化一个序列的HMM状态解码
        test_sequence = sequences[0]
        decoded_states = model.predict(test_sequence)
        
        plt.figure(figsize=(15, 5))
        # 设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 原始序列
        plt.subplot(1, 2, 1)
        plt.plot(test_sequence[:, 0], test_sequence[:, 1], 'o-', alpha=0.7)
        plt.title('特征序列')
        plt.xlabel('特征维度1')
        plt.ylabel('特征维度2')
        
        # 状态序列
        plt.subplot(1, 2, 2)
        colors = ['red', 'blue', 'green']
        for i in range(n_states):
            mask = decoded_states == i
            plt.plot(np.where(mask)[0], test_sequence[mask, 0], 
                    'o', color=colors[i], label=f'状态{i+1}')
        plt.title('HMM解码状态序列')
        plt.xlabel('时间帧')
        plt.ylabel('特征维度1')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("HMM参数:")
        print(f"初始状态概率: {model.startprob_}")
        print(f"状态转移矩阵:\n{model.transmat_}")
        print(f"状态均值:\n{model.means_}")
    
    def train_demo_model(self):
        """训练演示用的声学模型"""
        print("=== 训练声学模型 ===")
        
        # 加载训练数据
        training_data = self.load_training_data()
        
        # 训练模型
        self.acoustic_model.train_models(training_data)
        
        # 保存模型
        self.acoustic_model.save_models('models/acoustic_model.pkl')
    
    def test_recognition(self):
        """测试语音识别"""
        if not self.acoustic_model.is_trained:
            print("模型未训练，正在加载预训练模型...")
            try:
                self.acoustic_model.load_models('models/acoustic_model.pkl')
            except:
                print("找不到预训练模型，请先运行训练...")
                return
        
        print("=== 语音识别测试 ===")
        print("请录制一个音素（aa, iy, uw, eh, ah 之一）")
        
        # 录制音频
        audio = self.preprocessor.record_audio(duration=2)
        
        # 播放录制的音频
        print("播放录制的音频:")
        ipd.display(ipd.Audio(audio, rate=22050))
        
        # 进行识别
        phoneme, scores = self.acoustic_model.predict(audio)
        
        print(f"识别结果: {phoneme}")
        print("所有音素的分数:")
        for p, score in scores.items():
            print(f"  {p}: {score:.2f}")
        
        # 可视化MFCC特征
        features = self.acoustic_model.extract_features(audio)
        plt.figure(figsize=(10, 4))
        # 设置中文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        librosa.display.specshow(features.T, sr=22050, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC特征 - 识别结果: {phoneme}')
        plt.tight_layout()
        plt.show()

# 整合之前的语音预处理类
class VoicePreprocessing:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_data = None
        
    def record_audio(self, duration=3, sample_rate=22050):
        """录制音频"""
        print(f"开始录音，请说话... ({duration}秒)")
        self.sample_rate = sample_rate
        self.audio_data = sd.rec(int(duration * sample_rate), 
                                samplerate=sample_rate, 
                                channels=1)
        sd.wait()
        self.audio_data = self.audio_data.flatten()
        print("录音完成!")
        return self.audio_data
    
    def load_audio(self, filepath):
        """加载音频文件"""
        self.audio_data, self.sample_rate = librosa.load(filepath, sr=self.sample_rate)
        return self.audio_data

# 主程序
def main():
    # 创建项目实例
    project = SpeechProject()
    
    while True:
        print("\n" + "="*50)
        print("语音识别声学模型项目")
        print("="*50)
        print("1. GMM原理演示")
        print("2. HMM原理演示")
        print("3. 训练声学模型")
        print("4. 语音识别测试")
        print("5. 退出")
        
        choice = input("请选择功能 (1-5): ").strip()
        
        if choice == '1':
            project.demonstrate_gmm()
        elif choice == '2':
            project.demonstrate_hmm()
        elif choice == '3':
            project.train_demo_model()
        elif choice == '4':
            project.test_recognition()
        elif choice == '5':
            print("退出程序，再见!")
            break
        else:
            print("无效选择，请重新输入!")

if __name__ == "__main__":
    # 安装所需库的命令:
    # pip install numpy matplotlib librosa sounddevice soundfile scipy hmmlearn scikit-learn ipython
    
    # 创建必要的目录
    os.makedirs('data/timit_sample', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 运行主程序
    main()