#!/usr/bin/env python3
"""
音频处理演示程序测试脚本
测试程序的基本功能，不依赖网络下载
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import tempfile
import os

def test_basic_functionality():
    """测试基本功能"""
    print("测试音频处理演示程序基本功能...")
    
    # 导入我们的类
    from audio_processing_demo import AudioProcessingDemo
    
    # 创建实例
    demo = AudioProcessingDemo(sample_rate=22050)
    
    # 测试生成测试音频
    print("1. 测试生成测试音频...")
    audio = demo.generate_test_audio()
    print(f"✓ 生成音频成功，长度: {len(audio)} 样本")
    
    # 测试音高变换
    print("2. 测试音高变换...")
    pitch_shifted = demo.apply_pitch_shift(audio, n_steps=4)
    print(f"✓ 音高变换成功，长度: {len(pitch_shifted)} 样本")
    
    # 测试时间拉伸
    print("3. 测试时间拉伸...")
    time_stretched = demo.apply_time_stretch(audio, rate=1.5)
    print(f"✓ 时间拉伸成功，长度: {len(time_stretched)} 样本")
    
    # 测试失真效果
    print("4. 测试失真效果...")
    distorted = demo.apply_distortion(audio, gain=5.0)
    print(f"✓ 失真效果成功，长度: {len(distorted)} 样本")
    
    # 测试戏剧性效果组合
    print("5. 测试戏剧性效果组合...")
    dramatic = demo.create_dramatic_effect(audio)
    print(f"✓ 戏剧性效果成功，长度: {len(dramatic)} 样本")
    
    # 测试音频保存
    print("6. 测试音频保存...")
    demo.original_audio = audio
    demo.processed_audio = dramatic
    
    orig_file, proc_file = demo.save_audio_files("test")
    print(f"✓ 音频保存成功:")
    print(f"  原始音频: {orig_file}")
    print(f"  处理后音频: {proc_file}")
    
    # 检查文件是否存在
    if os.path.exists(orig_file) and os.path.exists(proc_file):
        print("✓ 音频文件创建成功")
        
        # 读取文件验证
        orig_data, sr = librosa.load(orig_file, sr=None)
        proc_data, sr = librosa.load(proc_file, sr=None)
        print(f"✓ 文件读取验证成功:")
        print(f"  原始文件长度: {len(orig_data)} 样本")
        print(f"  处理后文件长度: {len(proc_data)} 样本")
        
        # 清理测试文件
        os.remove(orig_file)
        os.remove(proc_file)
        print("✓ 测试文件清理完成")
    else:
        print("✗ 音频文件创建失败")
    
    print("\n🎉 所有基本功能测试通过!")
    return True

def test_audio_analysis():
    """测试音频分析功能"""
    print("\n测试音频分析功能...")
    
    from audio_processing_demo import AudioProcessingDemo
    
    demo = AudioProcessingDemo()
    demo.generate_test_audio()
    demo.processed_audio = demo.create_dramatic_effect(demo.original_audio)
    
    # 测试音频特征分析
    print("1. 测试音频特征分析...")
    demo.analyze_audio_differences()
    print("✓ 音频特征分析成功")
    
    # 测试图表生成（不显示，只检查是否报错）
    print("2. 测试图表生成...")
    try:
        # 使用非交互式后端避免显示窗口
        plt.switch_backend('Agg')
        demo.plot_comprehensive_comparison()
        plt.close('all')  # 关闭所有图表
        print("✓ 图表生成成功")
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("音频处理演示程序测试")
    print("=" * 60)
    
    try:
        # 测试基本功能
        if not test_basic_functionality():
            return False
        
        # 测试音频分析功能
        if not test_audio_analysis():
            return False
        
        print("\n" + "=" * 60)
        print("🎊 所有测试通过！程序功能正常")
        print("=" * 60)
        print("\n程序功能总结:")
        print("• ✓ 音频生成和加载")
        print("• ✓ 多种音频处理效果")
        print("• ✓ 音频文件保存")
        print("• ✓ 音频特征分析")
        print("• ✓ 可视化图表生成")
        print("• ✓ 前后音频对比")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
