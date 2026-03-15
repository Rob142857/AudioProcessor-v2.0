#!/usr/bin/env python3
"""
Preload Whisper models to avoid first-run download delays.

Downloads and caches the two models offered in the GUI:
  1. Faster-Whisper large-v3 (CTranslate2, GPU int8) — recommended
  2. Native OpenAI Whisper large-v3 — fallback
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
        
        # Only preload the model offered in the GUI dropdown
        models_to_load = []
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
        
        # Preload models offered in the GUI dropdown
        models_to_load = [
            "large-v3",           # Best accuracy
            "large-v3-turbo",     # 2x faster, half the VRAM
        ]
        success_count = 0
        
        for idx, model_name in enumerate(models_to_load, start=1):
            try:
                print(f"  [{idx}/{len(models_to_load)}] Loading {model_name}...")
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


def preload_punctuation_model():
    """Download and cache the punctuation restoration BERT model."""
    print("\n" + "="*60)
    print("PUNCTUATION RESTORATION MODEL")
    print("="*60)
    
    model_name = "oliverguhr/fullstop-punctuation-multilang-large"
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        print(f"Model: {model_name}")
        print(f"📥 Downloading punctuation model (~2.4GB)...")
        
        # Download tokenizer and model (will cache in ~/.cache/huggingface/hub/)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        print(f"✓ Model cached successfully")
        print(f"  Labels: {list(model.config.id2label.values())}")
        
        # Quick test
        import torch
        text = "hello world this is a test"
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Quick test: ✓ Model inference working")
        
        del model, tokenizer
        return True
        
    except ImportError:
        print("⚠ transformers not installed, trying deepmultilingualpunctuation...")
        try:
            from deepmultilingualpunctuation import PunctuationModel
            print(f"📥 Downloading via deepmultilingualpunctuation...")
            model = PunctuationModel(model=model_name)
            print(f"✓ Model cached successfully")
            del model
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to preload punctuation model: {e}")
        return False


def preload_paragraph_model():
    """Download and cache the semantic paragraph segmentation model."""
    print("\n" + "="*60)
    print("SEMANTIC PARAGRAPH MODEL")
    print("="*60)
    
    model_name = "all-MiniLM-L6-v2"
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Model: {model_name}")
        print(f"📥 Downloading semantic model (~80MB)...")
        
        model = SentenceTransformer(model_name)
        
        print(f"✓ Model cached successfully")
        
        # Quick test
        test_sentences = ["Hello world.", "This is a test."]
        embeddings = model.encode(test_sentences)
        print(f"  Quick test: ✓ Generated {len(embeddings)} embeddings")
        
        del model
        return True
        
    except ImportError:
        print("⚠ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Failed to preload paragraph model: {e}")
        return False


def preload_all_models():
    """Download and cache the two models offered in the GUI."""
    show_system_info()
    
    results = {}
    
    # Faster-Whisper large-v3 — PRIMARY (GPU, recommended)
    results['faster-whisper'] = preload_faster_whisper()
    
    # Native Whisper large-v3 — FALLBACK
    results['native-whisper'] = preload_native_whisper()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for backend, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {backend}")
    
    all_success = all(results.values())
    return all_success


if __name__ == "__main__":
    success = preload_all_models()
    if not success:
        print("\n❌ Failed to preload faster-whisper model")
        sys.exit(1)
    print("\n✓ Model preloading complete!")
