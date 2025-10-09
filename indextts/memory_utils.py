"""
内存优化工具模块 - 为 IndexTTS2 提供设备自适应功能
不侵入原有代码，通过猴子补丁（monkey patch）方式增强功能
"""

import torch
import gc


def get_model_device(model):
    """获取模型所在设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def ensure_tensor_device(tensor, target_device):
    """确保张量在目标设备上"""
    if tensor.device != target_device:
        return tensor.to(target_device)
    return tensor


def patch_semantic_codec_for_cpu(tts_model):
    """
    为 semantic_codec 添加 CPU 兼容性
    当 semantic_codec 在 CPU 时，自动处理设备转换
    
    这是一个非侵入性的补丁，原有代码无需修改
    """
    if not hasattr(tts_model, 'semantic_codec'):
        return
    
    semantic_codec = tts_model.semantic_codec
    codec_device = get_model_device(semantic_codec)
    
    # 只在 semantic_codec 在 CPU 时才需要打补丁
    if not str(codec_device).startswith('cpu'):
        print(f"[内存优化] semantic_codec 在 GPU，无需打补丁")
        return
    
    print(f"[内存优化] semantic_codec 在 CPU，应用设备适配补丁")
    
    # 检查是否已经打过补丁（避免重复打补丁）
    if hasattr(semantic_codec, '_cpu_patched'):
        print(f"[内存优化] semantic_codec 已打过补丁，跳过")
        return
    
    # 保存原始的 quantize 方法
    if hasattr(semantic_codec, 'quantize'):
        original_quantize = semantic_codec.quantize
        
        def quantize_with_device_adaptation(hidden_units):
            """包装 quantize 方法，自动处理设备转换"""
            # 将输入移到 CPU
            input_device = hidden_units.device
            if input_device != codec_device:
                hidden_units = hidden_units.to(codec_device)
            
            # 调用原始方法
            result = original_quantize(hidden_units)
            
            # 将结果移回原设备
            if isinstance(result, tuple):
                result = tuple(r.to(input_device) if torch.is_tensor(r) else r for r in result)
            elif torch.is_tensor(result):
                result = result.to(input_device)
            
            return result
        
        # 替换方法
        semantic_codec.quantize = quantize_with_device_adaptation
        print(f"[内存优化] ✓ 已包装 semantic_codec.quantize")
    
    # 处理 quantizer.vq2emb 方法（如果存在）
    if hasattr(semantic_codec, 'quantizer') and hasattr(semantic_codec.quantizer, 'vq2emb'):
        original_vq2emb = semantic_codec.quantizer.vq2emb
        
        def vq2emb_with_device_adaptation(codes):
            """包装 vq2emb 方法，自动处理设备转换"""
            input_device = codes.device
            if input_device != codec_device:
                codes = codes.to(codec_device)
            
            result = original_vq2emb(codes)
            
            if torch.is_tensor(result):
                result = result.to(input_device)
            
            return result
        
        semantic_codec.quantizer.vq2emb = vq2emb_with_device_adaptation
        print(f"[内存优化] ✓ 已包装 semantic_codec.quantizer.vq2emb")
    
    # 标记已打补丁
    semantic_codec._cpu_patched = True
    print(f"[内存优化] ✓ semantic_codec 设备适配补丁已应用")


def cleanup_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_optimal_inference_params(total_memory_gb):
    """
    根据 GPU 内存大小返回最优的推理参数
    
    Args:
        total_memory_gb: GPU 总内存（GB）
    
    Returns:
        dict: 推理参数配置
    """
    if total_memory_gb < 6:
        # 极小内存
        return {
            'diffusion_steps': 10,
            'max_mel_tokens': 250,
            'batch_size': 1,
            'cleanup_frequency': 'every_segment'
        }
    elif total_memory_gb < 8.5:
        # 小内存
        return {
            'diffusion_steps': 15,
            'max_mel_tokens': 300,
            'batch_size': 1,
            'cleanup_frequency': 'every_segment'
        }
    elif total_memory_gb < 10:
        # 中小内存
        return {
            'diffusion_steps': 20,
            'max_mel_tokens': 350,
            'batch_size': 1,
            'cleanup_frequency': 'every_2_segments'
        }
    else:
        # 正常内存
        return {
            'diffusion_steps': 25,
            'max_mel_tokens': 400,
            'batch_size': 1,
            'cleanup_frequency': 'end_only'
        }

