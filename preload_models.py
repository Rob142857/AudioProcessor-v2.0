#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.
This script downloads and caches models for all supported backends:
- Native OpenAI Whisper (large-v3, large-v3-turbo)
- Faster-Whisper (CTranslate2)
- Distil-Whisper (HuggingFace)
"""

import os
import sys
import logging


def preload_native_whisper():
    """Download and cache native OpenAI Whisper models."""
    print("\n" + "="*60)
    print("NATIVE WHISPER MODELS")
    print("="*60)
    
    try:
        import whisper
        import torch
        
        # Get available models
        try:
            avail = set(whisper.available_models())
        except Exception:
            avail = set()
        
        # Get cache directory
        cache_dir = torch.hub.get_dir()
        print(f"Cache directory: {cache_dir}\n")
        
        # Models to preload (in priority order)
        models_to_load = []
        if "large-v3-turbo" in avail:
            models_to_load.append("large-v3-turbo")
        if "large-v3" in avail:
            models_to_load.append("large-v3")
        
        if not models_to_load:
            print("⚠ No large models found in available models list")
            return False
        
        print(f"Preloading {len(models_to_load)} native Whisper model(s)...")
        
        success_count = 0
        for model_name in models_to_load:
            try:
                print(f"\n  [{success_count + 1}/{len(models_to_load)}] Loading {model_name}...")
                model = whisper.load_model(model_name)
                print(f"  ✓ Successfully cached: {model_name}")
                del model
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to preload {model_name}: {e}")
        
        print(f"\n✓ Native Whisper: {success_count}/{len(models_to_load)} models cached")
        return success_count > 0
        
    except ImportError:
        print("❌ whisper package not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def preload_faster_whisper():
    """Download and cache Faster-Whisper models (CTranslate2)."""
    print("\n" + "="*60)
    print("FASTER-WHISPER MODELS (CTranslate2)")
    print("="*60)
    
    try:
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For CUDA, try compute types in order of compatibility
        # int8 works on all CUDA GPUs (including older ones like GTX 1070 Ti)
        # float16 requires Volta+ (RTX 20xx, 30xx, 40xx)
        if device == "cuda":
            # Check GPU compute capability
            try:
                capability = torch.cuda.get_device_capability(0)
                # Volta and newer (compute capability >= 7.0) support efficient FP16
                if capability[0] >= 7:
                    compute_type = "float16"
                    print(f"GPU: {torch.cuda.get_device_name(0)} (Compute {capability[0]}.{capability[1]})")
                    print(f"Using float16 (efficient on this GPU)")
                else:
                    compute_type = "int8"
                    print(f"GPU: {torch.cuda.get_device_name(0)} (Compute {capability[0]}.{capability[1]})")
                    print(f"Using int8 (optimal for Pascal/Maxwell GPUs)")
            except:
                compute_type = "int8"  # Safe default for older GPUs
                print(f"Using int8 compute type (safe default)")
        else:
            compute_type = "int8"
            print(f"Device: CPU, using int8")
        
        print(f"\nDevice: {device}, Compute type: {compute_type}\n")
        
        models_to_load = ["large-v3"]  # Start with just large-v3
        success_count = 0
        
        for model_name in models_to_load:
            try:
                print(f"  [{success_count + 1}/{len(models_to_load)}] Loading {model_name}...")
                print(f"  📥 Downloading model (this may take a few minutes)...")
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
                print(f"  ✓ Successfully cached: {model_name}")
                del model
                success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to preload {model_name}: {e}")
        
        print(f"\n✓ Faster-Whisper: {success_count}/{len(models_to_load)} models cached")
        return success_count > 0
        
    except ImportError:
        print("⚠ faster-whisper not installed (optional)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def preload_distil_whisper():
    """Download and cache Distil-Whisper model (HuggingFace)."""
    print("\n" + "="*60)
    print("DISTIL-WHISPER MODEL (HuggingFace)")
    print("="*60)
    
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Device: {device}, Dtype: {torch_dtype}\n")
        
        model_id = "distil-whisper/distil-large-v3"
        
        try:
            print(f"  Loading {model_id}...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_id)
            print(f"  ✓ Successfully cached: {model_id}")
            del model, processor
            print(f"\n✓ Distil-Whisper: model cached")
            return True
        except Exception as e:
            print(f"  ❌ Failed to preload {model_id}: {e}")
            return False
        
    except ImportError:
        print("⚠ transformers not installed (optional)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_system_info():
    """Display system information."""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM: {props.total_memory / (1024**3):.1f} GB")
    except Exception as e:
        print(f"Could not get PyTorch info: {e}")
    
    print()


def preload_all_models():
    """Download and cache Faster-Whisper large-v3 model only."""
    show_system_info()
    
    results = {}
    
    # Native Whisper - COMMENTED OUT (not needed, using faster-whisper)
    # results['native'] = preload_native_whisper()
    
    # Faster-Whisper - PRIMARY MODEL
    results['faster'] = preload_faster_whisper()
    
    # Distil-Whisper - COMMENTED OUT (not needed for now)
    # results['distil'] = preload_distil_whisper()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for backend, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {backend}")
    
    return results.get('faster', False)  # Return faster-whisper status


if __name__ == "__main__":
    success = preload_all_models()
    if not success:
        print("\n❌ Failed to preload faster-whisper model")
        sys.exit(1)
    print("\n✓ Model preloading complete!")
