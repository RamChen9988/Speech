#!/usr/bin/env python3
"""
音频文件清理脚本
用于清理音频处理演示程序生成的文件
"""

import os
import shutil
import argparse

def list_audio_files():
    """列出所有音频文件"""
    audio_dirs = ["downloaded_audio", "processed_audio"]
    audio_files = []
    
    print("当前项目中的音频文件:")
    print("=" * 50)
    
    for audio_dir in audio_dirs:
        if os.path.exists(audio_dir):
            print(f"\n📁 {audio_dir}/ 目录:")
            files = os.listdir(audio_dir)
            if files:
                for file in files:
                    file_path = os.path.join(audio_dir, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  📄 {file} ({file_size:.1f} KB)")
                    audio_files.append(file_path)
            else:
                print("  (空目录)")
        else:
            print(f"\n📁 {audio_dir}/ 目录不存在")
    
    return audio_files

def calculate_total_size(audio_files):
    """计算总文件大小"""
    total_size = 0
    for file_path in audio_files:
        total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # MB

def cleanup_audio_files(confirm=True):
    """清理音频文件"""
    audio_dirs = ["downloaded_audio", "processed_audio"]
    deleted_files = []
    total_size = 0
    
    for audio_dir in audio_dirs:
        if os.path.exists(audio_dir):
            files = os.listdir(audio_dir)
            for file in files:
                file_path = os.path.join(audio_dir, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                deleted_files.append(file_path)
    
    if not deleted_files:
        print("没有找到需要清理的音频文件")
        return
    
    total_size_mb = total_size / (1024 * 1024)
    
    print("将要删除的音频文件:")
    print("=" * 50)
    for file_path in deleted_files:
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"📄 {file_path} ({file_size:.1f} KB)")
    
    print(f"\n总计: {len(deleted_files)} 个文件, {total_size_mb:.2f} MB")
    
    if confirm:
        response = input(f"\n确定要删除这些文件吗? (y/n): ").strip().lower()
        if response != 'y':
            print("取消删除操作")
            return
    
    # 执行删除
    for audio_dir in audio_dirs:
        if os.path.exists(audio_dir):
            shutil.rmtree(audio_dir)
            print(f"🗑️  已删除目录: {audio_dir}/")
    
    print(f"\n✅ 清理完成! 释放了 {total_size_mb:.2f} MB 空间")

def cleanup_specific_directory(directory):
    """清理特定目录"""
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在")
        return
    
    files = os.listdir(directory)
    if not files:
        print(f"目录 '{directory}' 为空")
        return
    
    total_size = 0
    for file in files:
        file_path = os.path.join(directory, file)
        total_size += os.path.getsize(file_path)
    
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"将要删除 {directory}/ 目录中的文件:")
    print("=" * 50)
    for file in files:
        file_path = os.path.join(directory, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"📄 {file} ({file_size:.1f} KB)")
    
    response = input(f"\n确定要删除 {directory}/ 目录中的 {len(files)} 个文件吗? (y/n): ").strip().lower()
    if response != 'y':
        print("取消删除操作")
        return
    
    shutil.rmtree(directory)
    print(f"🗑️  已删除目录: {directory}/")
    print(f"✅ 清理完成! 释放了 {total_size_mb:.2f} MB 空间")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="音频文件清理工具")
    parser.add_argument("--list", action="store_true", help="列出所有音频文件")
    parser.add_argument("--clean", action="store_true", help="清理所有音频文件")
    parser.add_argument("--clean-downloaded", action="store_true", help="只清理下载的音频文件")
    parser.add_argument("--clean-processed", action="store_true", help="只清理处理后的音频文件")
    parser.add_argument("--force", action="store_true", help="强制删除，不确认")
    
    args = parser.parse_args()
    
    print("🎵 音频文件清理工具")
    print("=" * 50)
    
    if args.list:
        list_audio_files()
    
    elif args.clean:
        cleanup_audio_files(confirm=not args.force)
    
    elif args.clean_downloaded:
        cleanup_specific_directory("downloaded_audio")
    
    elif args.clean_processed:
        cleanup_specific_directory("processed_audio")
    
    else:
        # 交互模式
        audio_files = list_audio_files()
        
        if audio_files:
            total_size_mb = calculate_total_size(audio_files)
            print(f"\n📊 总计: {len(audio_files)} 个文件, {total_size_mb:.2f} MB")
            
            print("\n选择操作:")
            print("1. 清理所有音频文件")
            print("2. 只清理下载的音频文件")
            print("3. 只清理处理后的音频文件")
            print("4. 退出")
            
            choice = input("请输入选择 (1-4): ").strip()
            
            if choice == "1":
                cleanup_audio_files()
            elif choice == "2":
                cleanup_specific_directory("downloaded_audio")
            elif choice == "3":
                cleanup_specific_directory("processed_audio")
            else:
                print("退出")
        else:
            print("没有找到音频文件")

if __name__ == "__main__":
    main()
