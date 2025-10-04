import os
import sys
import time
import numpy as np
import torch
from indextts.infer_v2 import IndexTTS2

# 检查音频库可用性（延迟导入）
HAS_SOUNDDEVICE = False
HAS_PYGAME = False

def check_sounddevice():
    """检查sounddevice是否可用"""
    global HAS_SOUNDDEVICE
    try:
        import sounddevice as sd
        # 尝试列出音频设备，确保真的可用
        sd.query_devices()
        HAS_SOUNDDEVICE = True
        return True
    except Exception as e:
        print(f">> sounddevice不可用: {e}")
        HAS_SOUNDDEVICE = False
        return False

def check_pygame():
    """检查pygame是否可用"""
    global HAS_PYGAME
    try:
        import pygame
        # 尝试初始化mixer，确保真的可用
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.quit()
        HAS_PYGAME = True
        return True
    except Exception as e:
        print(f">> pygame不可用: {e}")
        HAS_PYGAME = False
        return False

def play_audio_stream_sounddevice(audio_generator, sample_rate=22050, buffer_size=1024):
    """
    使用sounddevice实时播放音频流
    """
    import sounddevice as sd
    print(">> 开始播放音频流 (sounddevice)...")
    
    # 创建音频播放流
    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=buffer_size,
        latency='low'
    )
    
    try:
        stream.start()
        
        for i, audio_chunk in enumerate(audio_generator):
            if torch.is_tensor(audio_chunk):
                # 转换torch tensor为numpy数组
                audio_data = audio_chunk.numpy().squeeze()
                
                # 检查空音频数据
                if len(audio_data) == 0:
                    print(f">> 跳过空音频段 {i+1}")
                    continue
                
                # 归一化到[-1, 1]范围
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                elif len(audio_data) > 0 and (audio_data.max() > 1.0 or audio_data.min() < -1.0):
                    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
                
                print(f">> 播放音频段 {i+1}: {len(audio_data)} 采样点, 时长 {len(audio_data)/sample_rate:.2f}秒")
                
                # 播放音频
                stream.write(audio_data)
                
                # 稍作延迟确保流畅播放
                time.sleep(0.01)
            else:
                print(f">> 跳过非音频数据: {type(audio_chunk)}")
        
        # 等待播放完成
        time.sleep(0.5)
        
    except Exception as e:
        print(f">> 播放出错: {e}")
    finally:
        stream.stop()
        stream.close()
        print(">> 音频播放完成")

def play_audio_stream_pygame(audio_generator, sample_rate=22050):
    """
    使用pygame播放音频流
    """
    import pygame
    print(">> 开始播放音频流 (pygame)...")
    
    try:
        pygame.mixer.quit()
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=1024)
        
        for i, audio_chunk in enumerate(audio_generator):
            if torch.is_tensor(audio_chunk):
                # 转换torch tensor为numpy数组
                audio_data = audio_chunk.numpy().squeeze()
                
                # 检查空音频数据
                if len(audio_data) == 0:
                    print(f">> 跳过空音频段 {i+1}")
                    continue
                
                # 转换为int16格式
                if audio_data.dtype != np.int16:
                    if len(audio_data) > 0 and audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)
                
                print(f">> 播放音频段 {i+1}: {len(audio_data)} 采样点, 时长 {len(audio_data)/sample_rate:.2f}秒")
                
                # 使用pygame播放
                sound = pygame.sndarray.make_sound(audio_data)
                sound.play()
                
                # 等待播放完成
                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                    
            else:
                print(f">> 跳过非音频数据: {type(audio_chunk)}")
        
    except Exception as e:
        print(f">> 播放出错: {e}")
    finally:
        pygame.mixer.quit()
        print(">> 音频播放完成")

def save_audio_chunks(audio_generator, sample_rate=22050, output_dir="temp_chunks"):
    """
    将音频块保存到文件（fallback方案）
    """
    print(">> 保存音频块到文件...")
    
    import torchaudio
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        all_chunks = []
        for i, audio_chunk in enumerate(audio_generator):
            if torch.is_tensor(audio_chunk):
                audio_data = audio_chunk.squeeze()
                
                # 检查空音频数据
                if len(audio_data) == 0:
                    print(f">> 跳过空音频段 {i+1}")
                    continue
                
                print(f">> 保存音频段 {i+1}: {len(audio_data)} 采样点, 时长 {len(audio_data)/sample_rate:.2f}秒")
                
                # 保存单个块
                chunk_path = os.path.join(output_dir, f"chunk_{i+1:03d}.wav")
                torchaudio.save(chunk_path, audio_data.unsqueeze(0).cpu().type(torch.int16), sample_rate)
                
                all_chunks.append(audio_data.cpu())
            else:
                print(f">> 跳过非音频数据: {type(audio_chunk)}")
        
        # 合并所有块
        if all_chunks:
            full_audio = torch.cat(all_chunks, dim=0)
            full_path = os.path.join(output_dir, "full_audio.wav")
            torchaudio.save(full_path, full_audio.unsqueeze(0).type(torch.int16), sample_rate)
            print(f">> 完整音频已保存到: {full_path}")
            print(f">> 音频块保存到目录: {output_dir}")
        
    except Exception as e:
        print(f">> 保存出错: {e}")

def play_audio_stream(audio_generator, sample_rate=22050, buffer_size=1024):
    """
    自动选择最佳的播放方法
    """
    # 首先尝试sounddevice
    if check_sounddevice():
        try:
            play_audio_stream_sounddevice(audio_generator, sample_rate, buffer_size)
            return
        except Exception as e:
            print(f">> sounddevice播放失败: {e}")
    
    # 然后尝试pygame
    if check_pygame():
        try:
            play_audio_stream_pygame(audio_generator, sample_rate)
            return
        except Exception as e:
            print(f">> pygame播放失败: {e}")
    
    # Fallback: 保存到文件
    print(">> 无法直接播放，将保存音频块到文件")
    save_audio_chunks(audio_generator, sample_rate)

def main():
    if len(sys.argv) != 5:
        print("Usage: python pt_tts.py <spk_audio> <pt_path> <text> <output_path>")
        sys.exit(1)

    spk_audio = sys.argv[1]
    pt_path = sys.argv[2]
    text = sys.argv[3]
    output_path = sys.argv[4]

    if not os.path.exists(spk_audio):
        print("Spk audio file does not exist.")
        sys.exit(1)

    print(">> 初始化TTS模型...")
    tts = IndexTTS2(use_fp16=True, use_cuda_kernel=True, use_deepspeed=True)

    # 生成pt文件（如果需要）
    # tts.infer(pt_gen_mode=True,spk_audio_prompt=spk_audio,save_spk_emb_path=pt_path,verbose=True)

    print(f">> 开始流式TTS生成: {text}")
    print(f">> 使用PT文件: {pt_path}")
    print(f">> 输出路径: {output_path}")

    # # 使用生成音频文件
    # audio_stream = tts.infer(
    #     spk_audio_prompt=pt_path,
    #     output_path=output_path,
    #     text=text,
    #     verbose=True,
    # )
    
    # 使用stream模式生成音频
    audio_stream = tts.infer(
        spk_audio_prompt=pt_path,
        text=text,
        verbose=True,
        stream_return=True
    )
    
    # 实时播放音频流
    play_audio_stream(audio_stream, sample_rate=22050)

if __name__ == "__main__":
    main()
