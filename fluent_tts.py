#!/usr/bin/env python3
"""
流畅TTS - 单生产者单消费者模式
目标：避免多线程CUDA错误，实现流畅的音频生成和播放

架构：
- 1个生产者线程：负责生成音频流
- 1个消费者线程：负责播放音频
- 1个共享缓冲区：存储音频块，保证顺序
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import torch
from collections import deque, namedtuple
from indextts.infer_v2 import IndexTTS2

# 设置CUDA环境变量，避免设备端断言错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

# =========================
# 配置参数
# =========================
CHUNK_SIZE = 30  # 文本块大小（字符）
SAMPLE_RATE = 22050
BUFFER_MAX_SIZE = 5  # 缓冲区最大块数
CONSUMER_WAIT_TIMEOUT = 2.0  # 消费者等待超时（秒）

# 音频块数据结构
AudioChunk = namedtuple('AudioChunk', [
    'index', 'audio_data', 'text_snippet', 
    'generation_time', 'audio_length', 'timestamp'
])

# 性能统计数据结构
PerformanceStats = namedtuple('PerformanceStats', [
    'total_chunks', 'generated_chunks', 'played_chunks', 'failed_chunks',
    'total_generation_time', 'total_play_time', 'total_audio_length',
    'avg_generation_time', 'generation_speed_ratio', 'play_speed_ratio'
])

# =========================
# 简单音频缓冲区
# =========================
class SimpleAudioBuffer:
    """
    简单的FIFO音频缓冲区
    单生产者单消费者，线程安全
    """
    
    def __init__(self, max_size=BUFFER_MAX_SIZE):
        self.max_size = max_size
        self.buffer = deque()  # 使用deque作为FIFO队列
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)  # 缓冲区不为空的条件
        self.not_full = threading.Condition(self.lock)   # 缓冲区不满的条件
        
        # 统计信息
        self.produced_count = 0
        self.consumed_count = 0
        self.failed_count = 0
        
        # 性能统计
        self.total_generation_time = 0.0
        self.total_play_time = 0.0
        self.total_audio_length = 0.0
        
        # 生产者状态标记 - 新增
        self.producer_working = True  # 生产者是否还在工作
        self.producer_finished = False  # 生产者是否已完成所有工作
        
    def put(self, audio_chunk, timeout=None):
        """生产者放入音频块"""
        with self.not_full:
            # 等待缓冲区不满
            while len(self.buffer) >= self.max_size:
                print(f">> [缓冲区] 已满({len(self.buffer)}/{self.max_size})，等待消费者...")
                if not self.not_full.wait(timeout):
                    return False  # 超时
            
            # 放入音频块
            self.buffer.append(audio_chunk)
            self.produced_count += 1
            
            # 更新统计
            self.total_generation_time += audio_chunk.generation_time
            self.total_audio_length += audio_chunk.audio_length
            
            buffered_audio_time = sum(chunk.audio_length for chunk in self.buffer)
            
            print(f">> [缓冲区] 放入音频块 {audio_chunk.index} "
                  f"(缓冲: {len(self.buffer)}/{self.max_size}, "
                  f"音频时长: {buffered_audio_time:.1f}s)")
            
            self.not_empty.notify()  # 通知消费者
            return True
    
    def get(self, timeout=CONSUMER_WAIT_TIMEOUT):
        """消费者取出音频块"""
        with self.not_empty:
            # 等待缓冲区不为空
            start_wait = time.time()
            while len(self.buffer) == 0:
                wait_result = self.not_empty.wait(timeout)
                if not wait_result:
                    # 超时前再检查一次
                    if len(self.buffer) > 0:
                        break
                    elapsed = time.time() - start_wait
                    print(f">> [缓冲区] 获取超时，等待了{elapsed:.1f}s，缓冲区大小：{len(self.buffer)}")
                    return None  # 超时
            
            # 取出音频块
            if len(self.buffer) == 0:
                print(f">> [缓冲区] 警告：条件满足但缓冲区为空")
                return None
                
            audio_chunk = self.buffer.popleft()
            self.consumed_count += 1
            
            buffered_audio_time = sum(chunk.audio_length for chunk in self.buffer)
            
            print(f">> [缓冲区] 取出音频块 {audio_chunk.index} "
                  f"(缓冲: {len(self.buffer)}/{self.max_size}, "
                  f"剩余音频: {buffered_audio_time:.1f}s)")
            
            self.not_full.notify()  # 通知生产者
            return audio_chunk
    
    def mark_failed(self):
        """标记失败"""
        with self.lock:
            self.failed_count += 1
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            total_chunks = self.produced_count + self.failed_count
            avg_gen_time = (self.total_generation_time / self.produced_count 
                           if self.produced_count > 0 else 0)
            
            # 生成速度比（实际音频时长 / 生成时间）
            gen_speed_ratio = (self.total_audio_length / self.total_generation_time 
                              if self.total_generation_time > 0 else 0)
            
            # 播放速度比（播放时间应该等于音频时长）
            play_speed_ratio = (self.total_audio_length / self.total_play_time 
                               if self.total_play_time > 0 else 1.0)
            
            return PerformanceStats(
                total_chunks=total_chunks,
                generated_chunks=self.produced_count,
                played_chunks=self.consumed_count,
                failed_chunks=self.failed_count,
                total_generation_time=self.total_generation_time,
                total_play_time=self.total_play_time,
                total_audio_length=self.total_audio_length,
                avg_generation_time=avg_gen_time,
                generation_speed_ratio=gen_speed_ratio,
                play_speed_ratio=play_speed_ratio
            )
    
    def update_play_time(self, play_time):
        """更新播放时间统计"""
        with self.lock:
            self.total_play_time += play_time
    
    def size(self):
        """获取当前缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def is_empty(self):
        """检查是否为空"""
        with self.lock:
            return len(self.buffer) == 0
    
    def set_producer_working(self, working):
        """设置生产者工作状态"""
        with self.lock:
            self.producer_working = working
            if not working:
                self.producer_finished = True
                # 通知消费者生产者已完成
                self.not_empty.notify_all()
    
    def is_producer_working(self):
        """检查生产者是否还在工作"""
        with self.lock:
            return self.producer_working
    
    def is_producer_finished(self):
        """检查生产者是否已完成"""
        with self.lock:
            return self.producer_finished

# =========================
# 音频播放功能 - 复用pt_tts.py的代码
# =========================
def check_pygame():
    """检查pygame是否可用"""
    try:
        import pygame
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f">> pygame不可用: {e}")
        return False

def play_audio_chunk_pygame(audio_data, sample_rate=SAMPLE_RATE):
    """使用pygame播放单个音频块"""
    try:
        import pygame
        
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=1024)
        
        if torch.is_tensor(audio_data):
            audio_data = audio_data.numpy().squeeze()
        
        if len(audio_data) == 0:
            return 0
        
        # 转换为int16格式
        if audio_data.dtype != np.int16:
            if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # 播放音频
        sound = pygame.sndarray.make_sound(audio_data)
        sound.play()
        
        # 等待播放完成
        while pygame.mixer.get_busy():
            time.sleep(0.01)
            
        return len(audio_data) / sample_rate  # 返回实际播放时长
        
    except Exception as e:
        print(f">> 播放失败: {e}")
        return 0

# =========================
# 文本处理
# =========================
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    """将文本分割成固定大小的块，并过滤无效内容"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        
        # 过滤无效的文本块
        if (chunk and 
            len(chunk) >= 5 and  # 至少5个字符
            not all(c in '-=_\n\r\t 《》[]()' for c in chunk) and  # 不全是分隔符
            not chunk.count('-') > len(chunk) * 0.8):  # 横线不超过80%
            
            chunks.append(chunk)
        else:
            print(f">> [文本处理] 跳过无效文本块: '{chunk[:30]}...'")
    
    return chunks

def read_text_file(file_path):
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f">> 读取文件失败: {e}")
        return None

# =========================
# 音频生产者线程
# =========================
def audio_producer_thread(tts, pt_path, text_chunks, audio_buffer, stop_event):
    """
    音频生产者线程
    基于pt_tts.py的推理代码生成音频流
    """
    print(f">> [生产者] 开始生成 {len(text_chunks)} 个音频块")
    
    try:
        for i, text_chunk in enumerate(text_chunks):
            # 检查停止信号
            if stop_event.is_set():
                print(f">> [生产者] 收到停止信号，退出")
                break
            
            start_time = time.time()
            
            try:
                print(f">> [生产者] 生成音频块 {i}: '{text_chunk[:40]}...'")
                
                # 使用pt_tts.py的推理代码生成音频流
                audio_stream = tts.infer(
                    spk_audio_prompt=pt_path,
                    text=text_chunk,
                    verbose=False,  # 减少日志输出
                    stream_return=True,
                    # 优化参数，减少生成时间
                    num_beams=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.8,
                    max_mel_tokens=400,
                )
                
                # 收集所有音频块
                audio_chunks = []
                stream_start = time.time()
                
                for audio_chunk in audio_stream:
                    if torch.is_tensor(audio_chunk):
                        # 立即移动到CPU避免GPU内存积累
                        audio_chunks.append(audio_chunk.cpu())
                
                stream_time = time.time() - stream_start
                
                if audio_chunks:
                    # 检查并修复音频块维度问题
                    fixed_chunks = []
                    for chunk in audio_chunks:
                        if len(chunk.shape) == 1:
                            chunk = chunk.unsqueeze(0)  # [length] -> [1, length]
                        elif chunk.shape[0] > 1:
                            chunk = chunk.mean(dim=0, keepdim=True)  # 多声道转单声道 [channels, length] -> [1, length]
                        fixed_chunks.append(chunk)
                    
                    # 合并音频块
                    combined_audio = torch.cat(fixed_chunks, dim=1)
                    generation_time = time.time() - start_time
                    audio_length = combined_audio.shape[1] / SAMPLE_RATE
                    
                    # 创建音频块对象
                    audio_chunk_obj = AudioChunk(
                        index=i,
                        audio_data=combined_audio,
                        text_snippet=text_chunk[:50],
                        generation_time=generation_time,
                        audio_length=audio_length,
                        timestamp=time.time()
                    )
                    
                    # 放入缓冲区，重试机制
                    max_put_retries = 3
                    put_success = False
                    
                    for retry in range(max_put_retries):
                        if stop_event.is_set():
                            print(f">> [生产者] 音频块 {i} 收到停止信号，停止放入")
                            break
                            
                        success = audio_buffer.put(audio_chunk_obj, timeout=15.0)
                        if success:
                            put_success = True
                            print(f">> [生产者] 完成音频块 {i}: "
                                  f"生成{generation_time:.1f}s, 流处理{stream_time:.1f}s, "
                                  f"音频{audio_length:.1f}s, "
                                  f"速度比{audio_length/generation_time:.2f}x")
                            break
                        else:
                            buffer_stats = audio_buffer.get_stats()
                            print(f">> [生产者] 音频块 {i} 放入缓冲区超时 (重试{retry+1}/{max_put_retries}), "
                                  f"缓冲区状态: {audio_buffer.size()}/5, 已消费: {buffer_stats.played_chunks}")
                            
                            if retry < max_put_retries - 1:
                                time.sleep(2)  # 等待2秒后重试
                    
                    if not put_success and not stop_event.is_set():
                        print(f">> [生产者] 音频块 {i} 多次重试后仍无法放入缓冲区，标记为失败")
                        audio_buffer.mark_failed()
                        
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
                else:
                    print(f">> [生产者] 音频块 {i} 生成为空")
                    audio_buffer.mark_failed()
                    
            except Exception as e:
                print(f">> [生产者] 音频块 {i} 生成失败: {e}")
                audio_buffer.mark_failed()
                
                # CUDA错误恢复
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    print(f">> [生产者] 检测到CUDA错误，执行恢复...")
                    torch.cuda.empty_cache()
                    time.sleep(1)
                
    except Exception as e:
        print(f">> [生产者] 线程异常: {e}")
    finally:
        # 标记生产者工作完成
        audio_buffer.set_producer_working(False)
        print(f">> [生产者] 标记工作完成，通知消费者")
    
    print(f">> [生产者] 线程结束")

# =========================
# 音频消费者线程
# =========================
def audio_consumer_thread(audio_buffer, total_chunks, stop_event):
    """音频消费者线程 - 智能等待播放音频"""
    print(f">> [消费者] 开始智能等待播放音频流")
    
    if not check_pygame():
        print(f">> [消费者] pygame不可用，无法播放音频")
        return
    
    played_count = 0
    
    # 第一阶段：等待40秒让生产者启动
    print(f">> [消费者] 等待生产者启动，等待40秒...")
    initial_wait_time = 40
    waited_time = 0
    
    while waited_time < initial_wait_time:
        if stop_event.is_set():
            print(f">> [消费者] 收到停止信号，退出初始等待")
            return
        
        # 检查是否已经有音频可播放
        if not audio_buffer.is_empty():
            print(f">> [消费者] 提前发现音频内容，开始播放（等待了{waited_time}s）")
            break
        
        time.sleep(2)  # 每2秒检查一次
        waited_time += 2
        
        if waited_time % 10 == 0:  # 每10秒报告一次
            buffer_stats = audio_buffer.get_stats()
            producer_working = audio_buffer.is_producer_working()
            print(f">> [消费者] 初始等待中...({waited_time}s/40s), "
                  f"已生成: {buffer_stats.generated_chunks}, "
                  f"生产者工作中: {producer_working}")
    
    if waited_time >= initial_wait_time and audio_buffer.is_empty():
        print(f">> [消费者] 初始等待40秒完成，开始智能检查模式")
    
    # 第二阶段：智能检查播放
    print(f">> [消费者] 进入智能检查播放模式")
    
    try:
        while True:
            # 检查停止信号
            if stop_event.is_set():
                print(f">> [消费者] 收到停止信号，退出")
                break
            
            # 检查缓冲区是否有内容
            if not audio_buffer.is_empty():
                # 有内容就获取播放
                audio_chunk = audio_buffer.get(timeout=1.0)  # 短超时
                
                if audio_chunk is not None:
                    # 播放音频块
                    play_start = time.time()
                    
                    print(f">> [消费者] 播放音频块 {audio_chunk.index}: "
                          f"'{audio_chunk.text_snippet}...', 时长 {audio_chunk.audio_length:.1f}s")
                    
                    actual_play_time = play_audio_chunk_pygame(audio_chunk.audio_data, SAMPLE_RATE)
                    
                    if actual_play_time > 0:
                        play_time = time.time() - play_start
                        audio_buffer.update_play_time(play_time)
                        
                        played_count += 1
                        
                        print(f">> [消费者] 完成播放音频块 {audio_chunk.index}: "
                              f"播放时间 {play_time:.1f}s, 音频时长 {audio_chunk.audio_length:.1f}s")
                    else:
                        print(f">> [消费者] 音频块 {audio_chunk.index} 播放失败")
                    
                    # 释放内存
                    del audio_chunk
                    continue
            
            # 缓冲区空，检查生产者状态
            producer_working = audio_buffer.is_producer_working()
            producer_finished = audio_buffer.is_producer_finished()
            buffer_stats = audio_buffer.get_stats()
            
            if producer_finished and audio_buffer.is_empty():
                print(f">> [消费者] 生产者已完成且缓冲区为空，播放结束")
                print(f">> [消费者] 总共播放了 {played_count} 个音频块")
                break
            
            if producer_working:
                print(f">> [消费者] 生产者还在工作中，等待2秒... "
                      f"(已播放: {played_count}, 已生成: {buffer_stats.generated_chunks}, "
                      f"失败: {buffer_stats.failed_chunks})")
                time.sleep(2)  # 等待2秒后再检查
            else:
                print(f">> [消费者] 生产者已停止工作但缓冲区为空，可能出现问题")
                # 再等待几秒看看是否有遗漏的音频块
                for i in range(3):
                    time.sleep(1)
                    if not audio_buffer.is_empty():
                        print(f">> [消费者] 发现遗漏的音频块，继续播放")
                        break
                else:
                    print(f">> [消费者] 确认无更多音频，停止播放")
                    break
            
    except Exception as e:
        print(f">> [消费者] 线程异常: {e}")
    
    print(f">> [消费者] 线程结束，已播放 {played_count}/{total_chunks} 个音频块")

# =========================
# 主要功能
# =========================
def fluent_tts_main(pt_path, text_input, is_file=True):
    """
    流畅TTS主函数
    单生产者单消费者模式
    """
    print("=== 流畅TTS系统启动 ===")
    print(f"架构: 1个生产者线程 + 1个消费者线程 + 共享缓冲区")
    print(f"配置: 文本块{CHUNK_SIZE}字符, 缓冲区{BUFFER_MAX_SIZE}块, 超时{CONSUMER_WAIT_TIMEOUT}s")
    
    start_time = time.time()
    
    try:
        # 1. 初始化TTS模型
        print(">> 初始化TTS模型...")
        tts = IndexTTS2(
            use_fp16=True,
            use_cuda_kernel=False,  # 禁用CUDA kernel避免错误
            use_deepspeed=False     # 禁用DeepSpeed避免多进程问题
        )
        
        # 2. 准备文本
        if is_file:
            print(f">> 读取文本文件: {text_input}")
            text = read_text_file(text_input)
            if text is None:
                return False
        else:
            text = text_input
        
        print(f">> 文本长度: {len(text)} 字符")
        
        # 3. 分割文本
        text_chunks = split_text_into_chunks(text, CHUNK_SIZE)
        total_chunks = len(text_chunks)
        print(f">> 文本分为 {total_chunks} 个块")
        
        # 显示前几个块的内容
        for i, chunk in enumerate(text_chunks[:3]):
            print(f"   块{i}: '{chunk[:60]}...'")
        
        # 4. 创建共享缓冲区
        audio_buffer = SimpleAudioBuffer(BUFFER_MAX_SIZE)
        stop_event = threading.Event()
        
        # 5. 同时启动生产者和消费者线程
        print(">> 启动生产者线程...")
        producer_thread = threading.Thread(
            target=audio_producer_thread,
            args=(tts, pt_path, text_chunks, audio_buffer, stop_event),
            daemon=True
        )
        producer_thread.start()
        
        print(">> 启动消费者线程...")
        consumer_thread = threading.Thread(
            target=audio_consumer_thread,
            args=(audio_buffer, total_chunks, stop_event),
            daemon=True
        )
        consumer_thread.start()
        
        # 7. 监控进度
        print(">> 开始监控进度...")
        
        last_stats_time = time.time()
        
        while (producer_thread.is_alive() or 
               consumer_thread.is_alive() or 
               not audio_buffer.is_empty()):
            
            time.sleep(2)  # 每2秒监控一次
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            stats = audio_buffer.get_stats()
            
            # 每10秒打印详细统计
            if current_time - last_stats_time >= 10:
                print(f"\n>> [监控] 运行时间: {elapsed:.0f}s")
                print(f"   生产: {stats.generated_chunks}/{stats.total_chunks}, "
                      f"失败: {stats.failed_chunks}")
                print(f"   播放: {stats.played_chunks}/{stats.total_chunks}")
                print(f"   缓冲区: {audio_buffer.size()} 个音频块")
                
                if stats.generated_chunks > 0:
                    print(f"   平均生成时间: {stats.avg_generation_time:.1f}s/块")
                    print(f"   生成速度比: {stats.generation_speed_ratio:.2f}x")
                    print(f"   总音频时长: {stats.total_audio_length:.1f}s")
                
                last_stats_time = current_time
            
            # 检查是否应该停止
            if (not producer_thread.is_alive() and 
                consumer_thread.is_alive() and
                audio_buffer.is_empty()):
                print(">> [监控] 生产者已完成且缓冲区为空，等待消费者完成...")
                time.sleep(2)  # 给消费者一些时间处理剩余任务
            
            if (not producer_thread.is_alive() and 
                not consumer_thread.is_alive()):
                break
        
        # 8. 等待线程结束
        stop_event.set()
        producer_thread.join(timeout=5)
        consumer_thread.join(timeout=5)
        
        # 9. 最终统计
        total_time = time.time() - start_time
        final_stats = audio_buffer.get_stats()
        
        print("\n=== 流畅TTS系统结束 ===")
        print(f"总运行时间: {total_time:.1f}s")
        print(f"成功生成: {final_stats.generated_chunks}/{final_stats.total_chunks}")
        print(f"成功播放: {final_stats.played_chunks}/{final_stats.total_chunks}")
        print(f"失败数量: {final_stats.failed_chunks}")
        
        if final_stats.generated_chunks > 0:
            print(f"总音频时长: {final_stats.total_audio_length:.1f}s")
            print(f"平均生成时间: {final_stats.avg_generation_time:.1f}s/块")
            print(f"生成速度比: {final_stats.generation_speed_ratio:.2f}x "
                  f"({'实时' if final_stats.generation_speed_ratio >= 1.0 else '慢于实时'})")
            
            # 计算效率
            if final_stats.total_audio_length > 0:
                efficiency = (final_stats.total_audio_length / total_time) * 100
                print(f"整体效率: {efficiency:.1f}% "
                      f"(音频时长/总时间 = {final_stats.total_audio_length:.1f}s/{total_time:.1f}s)")
        
        success_rate = final_stats.played_chunks / final_stats.total_chunks * 100
        print(f"播放成功率: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("✅ 系统运行良好")
            return True
        elif success_rate >= 70:
            print("⚠️ 系统基本正常，有少量失败")
            return True
        else:
            print("❌ 系统存在较多问题")
            return False
            
    except Exception as e:
        print(f">> 系统异常: {e}")
        return False
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(">> GPU缓存已清理")

# =========================
# 命令行接口
# =========================
def main():
    if len(sys.argv) < 3:
        print("Usage: python fluent_tts.py <pt_file_path> <text_file_path>")
        print("   or: python fluent_tts.py <pt_file_path> \"<text_content>\" --text")
        print("")
        print("流畅TTS - 单生产者单消费者模式")
        print("- 避免多线程CUDA错误")
        print("- 实现流畅的音频生成和播放")
        print("- 详细的性能监控和统计")
        sys.exit(1)
    
    pt_path = sys.argv[1]
    text_input = sys.argv[2]
    is_file = len(sys.argv) < 4 or sys.argv[3] != "--text"
    
    # 检查PT文件是否存在
    if not os.path.exists(pt_path):
        print(f"错误: PT文件不存在: {pt_path}")
        sys.exit(1)
    
    # 如果是文件模式，检查文本文件是否存在
    if is_file and not os.path.exists(text_input):
        print(f"错误: 文本文件不存在: {text_input}")
        sys.exit(1)
    
    print(f">> PT文件: {pt_path}")
    if is_file:
        print(f">> 文本文件: {text_input}")
    else:
        print(f">> 文本内容: {text_input[:50]}...")
    
    # 执行流畅TTS
    try:
        success = fluent_tts_main(pt_path, text_input, is_file)
        
        if success:
            print(f"\n✅✅✅ 流畅TTS成功完成 ✅✅✅")
            sys.exit(0)
        else:
            print(f"\n⚠️⚠️⚠️ 流畅TTS完成但有问题 ⚠️⚠️⚠️")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断程序")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 程序执行异常: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
