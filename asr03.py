# 完全离线的语音识别演示 - 国内网络环境下可稳定运行
import os
import wave
import numpy as np
from vosk import Model, KaldiRecognizer
import pyaudio
import threading
import time
import subprocess
import sys

def install_offline_dependencies():
    """安装离线语音识别所需的依赖"""
    print("🔧 安装离线语音识别组件...")
    
    packages = [
        "vosk",
        "pyaudio",
        "numpy"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📥 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")

def download_vosk_model():
    """下载Vosk中文语音识别模型"""
    model_path = "vosk-model-cn-0.22"
    model_url = "https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip"
    
    if os.path.exists(model_path):
        print(f"✅ Vosk中文模型已存在: {model_path}")
        return model_path
    
    print("📥 正在下载中文语音识别模型(约1.8GB)...")
    print("💡 首次下载需要较长时间，请耐心等待...")
    
    try:
        import urllib.request
        import zipfile
        
        # 下载模型文件
        zip_path = "vosk-model-cn-0.22.zip"
        urllib.request.urlretrieve(model_url, zip_path)
        
        # 解压模型
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # 删除压缩包
        os.remove(zip_path)
        print("✅ 中文语音模型下载完成！")
        return model_path
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("💡 请手动下载: https://alphacephei.com/vosk/models")
        print("💡 或使用备用的小模型")
        return None

def create_sample_audio():
    """创建示例音频文件用于演示"""
    print("🎵 创建示例音频文件...")
    
    # 创建一个简单的正弦波作为示例音频
    sample_rate = 16000
    duration = 3  # 3秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成440Hz的正弦波（A4音符）
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # 保存为WAV文件
    with wave.open("sample_audio.wav", 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    print("✅ 示例音频创建完成")
    return "sample_audio.wav"

def offline_speech_recognition(model_path):
    """
    离线实时语音识别
    完全在本地运行，无需网络连接
    """
    if not model_path or not os.path.exists(model_path):
        print("❌ 语音模型不存在，使用模拟模式")
        return simulate_recognition()
    
    print("🎤 初始化离线语音识别系统...")
    
    # 加载Vosk模型
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    
    # 初始化音频输入
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096
    )
    
    print("🔊 离线语音识别已启动!")
    print("💡 请对着麦克风说话...")
    print("⏹️ 按 Ctrl+C 停止识别")
    
    try:
        while True:
            # 读取音频数据
            data = stream.read(4096, exception_on_overflow=False)
            
            if recognizer.AcceptWaveform(data):
                # 获取识别结果
                result = recognizer.Result()
                result_json = eval(result)
                
                if 'text' in result_json and result_json['text']:
                    recognized_text = result_json['text']
                    print(f"✅ 识别结果: {recognized_text}")
                    
                    # 执行相应的命令
                    if not execute_offline_command(recognized_text):
                        break
            
            # 实时显示部分结果
            partial_result = recognizer.PartialResult()
            partial_json = eval(partial_result)
            if 'partial' in partial_json and partial_json['partial']:
                print(f"🔍 实时识别: {partial_json['partial']}", end='\r')
                
    except KeyboardInterrupt:
        print("\n\n👋 停止语音识别")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def simulate_recognition():
    """
    模拟语音识别（在没有真实模型时使用）
    """
    print("🎭 模拟语音识别模式（使用键盘输入测试）")
    print("💡 请输入指令来模拟语音输入:")
    
    demo_commands = [
        "打开灯光",
        "播放音乐", 
        "今天天气怎么样",
        "导航到学校",
        "退出"
    ]
    
    for i, cmd in enumerate(demo_commands, 1):
        print(f"{i}. {cmd}")
    
    while True:
        try:
            choice = input("\n请选择指令编号 (1-5): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 5:
                command_text = demo_commands[int(choice) - 1]
                print(f"🗣️ 模拟语音输入: {command_text}")
                
                if not execute_offline_command(command_text):
                    break
            else:
                print("❌ 请输入有效编号")
        except KeyboardInterrupt:
            print("\n👋 退出模拟模式")
            break

def execute_offline_command(command_text):
    """
    执行离线识别到的命令
    """
    command = command_text.lower()
    
    print(f"🤖 分析指令: {command}")
    
    # 智能家居控制
    if any(word in command for word in ['打开', '开启', '启动']):
        if '灯' in command:
            print("💡 执行: 打开灯光")
            print("✨ 客厅灯光已开启 - [智能家居系统响应]")
        elif '空调' in command:
            print("❄️ 执行: 打开空调")
            print("🌡️ 空调已启动，设定温度24℃ - [IoT设备响应]")
        elif '音乐' in command:
            print("🎵 执行: 播放音乐")
            print("🎶 正在播放推荐歌单... - [媒体系统响应]")
    
    # 查询功能
    elif any(word in command for word in ['天气', '温度']):
        print("🌤️ 执行: 查询天气")
        print("📊 今天晴转多云，25℃，适宜外出 - [天气服务响应]")
    
    # 导航功能
    elif any(word in command for word in ['导航', '去', '到']):
        print("🗺️ 执行: 路径规划")
        print("📍 已为您规划最优路线，预计用时15分钟 - [导航系统响应]")
    
    # 系统控制
    elif '退出' in command or '结束' in command:
        print("👋 退出语音识别系统")
        return False
    elif '你好' in command:
        print("👋 你好！我是离线语音助手")
    elif '谢谢' in command:
        print("😊 不客气，很高兴为您服务")
    else:
        print("💭 指令已收到，正在处理...")
    
    return True

def demonstrate_asr_workflow():
    """
    演示语音识别的完整工作流程
    """
    print("\n" + "="*60)
    print("🔬 语音识别技术流程详解")
    print("="*60)
    
    steps = [
        {
            "步骤": "1. 音频采集",
            "技术": "麦克风 → PCM数据",
            "类比": "用耳朵听声音"
        },
        {
            "步骤": "2. 预处理", 
            "技术": "降噪、分帧、加窗",
            "类比": "过滤背景噪音"
        },
        {
            "步骤": "3. 特征提取",
            "技术": "MFCC特征向量", 
            "类比": "提取声音指纹"
        },
        {
            "步骤": "4. 声学模型",
            "技术": "DNN/HMM识别音素",
            "类比": "识别发音单位"
        },
        {
            "步骤": "5. 语言模型", 
            "技术": "N-gram/Transformer",
            "类比": "理解语言规律"
        },
        {
            "步骤": "6. 解码输出",
            "技术": "维特比算法",
            "类比": "组合成完整句子"
        }
    ]
    
    for step in steps:
        print(f"\n{step['步骤']}: {step['技术']}")
        print(f"   🎯 {step['类比']}")

def file_based_recognition(model_path, audio_file=None):
    """
    基于文件的语音识别演示
    """
    if not audio_file:
        audio_file = create_sample_audio()
        print("💡 使用生成的示例音频进行识别演示")
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    print(f"📁 识别音频文件: {audio_file}")
    
    if not model_path or not os.path.exists(model_path):
        print("🔊 模拟文件识别结果: '这是一个测试音频文件'")
        return
    
    try:
        # 使用Vosk进行文件识别
        model = Model(model_path)
        wf = wave.open(audio_file, 'rb')
        
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("❌ 音频格式不支持，需要单声道16位PCM格式")
            return
        
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        print("🔍 正在识别音频文件内容...")
        while True:
            data = wf.readframes(4096)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        
        result = recognizer.FinalResult()
        result_json = eval(result)
        
        if 'text' in result_json:
            print(f"✅ 文件识别结果: {result_json['text']}")
        else:
            print("❌ 未能识别出有效内容")
            
    except Exception as e:
        print(f"❌ 文件识别失败: {e}")

# 主程序
if __name__ == "__main__":
    print("🎯 离线语音识别系统演示")
    print("=" * 50)
    print("💡 特点: 完全本地运行 · 无需网络 · 保护隐私")
    
    # 安装依赖
    install_offline_dependencies()
    
    # 技术流程演示
    demonstrate_asr_workflow()
    
    print("\n" + "="*50)
    print("🚀 选择识别模式:")
    print("1. 实时语音识别（需要麦克风）")
    print("2. 文件语音识别") 
    print("3. 模拟演示模式（无需麦克风）")
    print("4. 下载语音模型")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        model_path = None
        if choice in ["1", "2"]:
            print("\n📥 准备语音识别模型...")
            model_path = download_vosk_model()
        
        if choice == "1":
            print("\n🎤 启动实时语音识别...")
            offline_speech_recognition(model_path)
        elif choice == "2":
            file_path = input("请输入音频文件路径（留空使用示例）: ").strip()
            file_based_recognition(model_path, file_path if file_path else None)
        elif choice == "3":
            simulate_recognition()
        elif choice == "4":
            download_vosk_model()
            print("✅ 模型下载完成，请重新运行程序使用")
        else:
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print("💡 提示: 请检查麦克风权限和音频设备")
    
    print("\n" + "="*50)
    print("🎓 教学要点总结:")
    print("   - Vosk: 开源离线语音识别工具包")
    print("   - 声学模型: 将声音特征映射到音素") 
    print("   - 语言模型: 根据上下文预测最可能的文本")
    print("   - 实时识别: 流式处理，低延迟响应")
    print("🎉 离线语音识别体验完成！")