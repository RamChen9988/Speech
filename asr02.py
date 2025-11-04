# 真实可运行的语音识别演示
# 首次运行会自动安装依赖，无需手动下载

import speech_recognition as sr
import pyaudio
import requests
import os
import sys
import subprocess

def install_dependencies():
    """自动安装必要的依赖包"""
    print("🔧 检查并安装必要的依赖...")
    
    # 需要安装的包列表
    packages = [
        "SpeechRecognition",
        "pyaudio",
        "requests"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📥 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")

def real_time_speech_recognition():
    """
    实时语音识别演示
    类比：打造一个能听懂你说话的智能助手
    """
    print("🎤 初始化语音识别系统...")
    
    # 创建识别器实例
    recognizer = sr.Recognizer()
    
    # 使用麦克风作为音频源
    with sr.Microphone() as source:
        print("🔊 正在校准麦克风，请保持安静...")
        
        # 校准环境噪音（重要步骤！）
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        print("🎯 校准完成！请开始说话...")
        print("💡 提示：尝试说 '打开灯光'、'播放音乐' 或 '今天天气怎么样'")
        print("⏹️ 说完后请保持安静，系统会自动识别")
        
        while True:
            try:
                print("\n" + "="*40)
                print("🟢 正在聆听...")
                
                # 录制音频（超时时间5秒，最长录音10秒）
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("🔍 正在识别语音内容...")
                
                # 使用Google语音识别API
                text = recognizer.recognize_google(audio_data, language='zh-CN')
                
                print(f"✅ 识别结果: {text}")
                
                # 根据识别内容执行相应操作
                execute_command(text)
                
            except sr.WaitTimeoutError:
                print("⏰ 等待超时，没有检测到语音")
                continue
            except sr.UnknownValueError:
                print("❌ 无法理解语音内容，请重试")
                continue
            except sr.RequestError as e:
                print(f"🌐 网络错误: {e}")
                print("💡 请检查网络连接")
                break
            except KeyboardInterrupt:
                print("\n👋 感谢使用语音识别系统！")
                break

def execute_command(command_text):
    """
    根据识别到的文本执行相应操作
    类比：智能助手理解指令并执行任务
    """
    command = command_text.lower()
    
    print(f"🤖 分析指令: {command}")
    
    # 智能家居控制场景
    if any(word in command for word in ['打开', '开启', '启动']):
        if '灯' in command:
            print("💡 执行: 打开灯光")
            print("✨ 客厅灯光已开启")
        elif '空调' in command:
            print("❄️ 执行: 打开空调")
            print("🌡️ 空调已启动，设定温度24℃")
        elif '音乐' in command:
            print("🎵 执行: 播放音乐")
            print("🎶 正在播放推荐歌单...")
    
    # 查询场景
    elif any(word in command for word in ['天气', '温度']):
        print("🌤️ 执行: 查询天气")
        print("📊 今天晴转多云，25℃，适宜外出")
    
    # 导航场景
    elif any(word in command for word in ['导航', '去', '到']):
        print("🗺️ 执行: 路径规划")
        print("📍 已为您规划最优路线，预计用时15分钟")
    
    # 通用回应
    elif '你好' in command or '嗨' in command:
        print("👋 你好！我是您的语音助手")
    elif '谢谢' in command:
        print("😊 不客气，很高兴为您服务")
    elif '再见' in command or '退出' in command:
        print("👋 再见！期待再次为您服务")
        return False
    else:
        print("💭 已记录您的需求，正在学习中...")
    
    return True

def file_speech_recognition(audio_file_path=None):
    """
    文件语音识别演示
    类比：让系统"阅读"录音文件
    """
    print("\n📁 文件语音识别模式")
    
    recognizer = sr.Recognizer()
    
    # 如果没有提供文件路径，创建一个示例
    if not audio_file_path:
        print("💡 提示：您可以提供自己的WAV文件路径")
        print("📝 当前使用内置示例（需要网络下载）")
        
        # 下载示例音频文件
        example_url = "https://github.com/Uberi/speech_recognition/raw/master/examples/french.aiff"
        audio_file_path = "example_audio.wav"
        
        try:
            print("🌐 下载示例音频文件...")
            response = requests.get(example_url)
            with open(audio_file_path, "wb") as f:
                f.write(response.content)
            print("✅ 示例音频下载完成")
        except:
            print("❌ 下载失败，请检查网络连接")
            return
    
    try:
        # 加载音频文件
        with sr.AudioFile(audio_file_path) as source:
            print("🔊 读取音频文件...")
            audio_data = recognizer.record(source)
            
            print("🔍 正在识别文件内容...")
            # 识别英文音频
            text = recognizer.recognize_google(audio_data)
            
            print(f"📄 文件内容识别结果: {text}")
            
    except Exception as e:
        print(f"❌ 文件识别失败: {e}")

def demonstrate_asr_scenarios():
    """
    演示不同应用场景
    """
    print("\n" + "="*50)
    print("🚀 语音识别应用场景演示")
    print("="*50)
    
    scenarios = [
        {
            "场景": "智能家居",
            "指令": "'打开客厅灯光'",
            "系统响应": "💡 灯光已开启"
        },
        {
            "场景": "车载语音", 
            "指令": "'导航到最近的加油站'",
            "系统响应": "⛽ 已找到3个附近加油站"
        },
        {
            "场景": "语音输入",
            "指令": "'今天记得完成作业'", 
            "系统响应": "📝 文本已输入：今天记得完成作业"
        },
        {
            "场景": "智能客服",
            "指令": "'查询我的订单状态'",
            "系统响应": "📦 您的订单正在派送中"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['场景']}:")
        print(f"   👤 用户说: {scenario['指令']}")
        print(f"   🤖 系统: {scenario['系统响应']}")

# 主程序
if __name__ == "__main__":
    print("🎯 真实语音识别系统演示")
    print("=" * 50)
    
    # 自动安装依赖
    install_dependencies()
    
    # 演示应用场景
    demonstrate_asr_scenarios()
    
    print("\n" + "="*50)
    print("🎤 选择识别模式:")
    print("1. 实时语音识别（需要麦克风）")
    print("2. 文件语音识别")
    print("3. 退出")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 启动实时语音识别...")
            real_time_speech_recognition()
        elif choice == "2":
            file_path = input("请输入音频文件路径（留空使用示例）: ").strip()
            file_speech_recognition(file_path if file_path else None)
        elif choice == "3":
            print("👋 再见！")
        else:
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print("💡 提示: 请确保麦克风正常工作且已连接网络")
    
    print("\n" + "="*50)
    print("💡 技术要点总结:")
    print("   - 使用 SpeechRecognition 库简化开发")
    print("   - Google Speech API 提供准确的识别服务") 
    print("   - 噪音校准提升识别准确率")
    print("   - 支持中英文等多种语言")
    print("🎉 真实语音识别体验完成！")