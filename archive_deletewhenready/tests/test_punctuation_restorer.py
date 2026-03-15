"""
Quick test script for the new punctuation restoration module.
Tests the oliverguhr/fullstop-punctuation-multilang-large model integration.
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_punctuation_restorer():
    """Test the punctuation restorer with sample transcription text."""
    
    # Sample text from your example (Whisper output with some issues)
    test_text = """What can be said about the difference between a system of development and a way of development? Now people like yourselves who are interested in doing something about your spiritual life, sooner or later will come to contact with a group, with maybe a school, maybe. A teacher, maybe somebody who has some knowledge of these things.
And it's very important in the first few months to try and decide whether that person or group or school is teaching a system or is indicating a way of development. There is a very big difference between them, and the results obtained also are very, very different. We have various systems of yoga. We have, within the sufi framework, particular and very narrow systems. The system arises when there are specific teachings, fixed and sometimes very rigid rules in the group or school that must be obeyed, and the doctrines mustn't be. Departed from."""

    print("="*80)
    print("TESTING PUNCTUATION RESTORER")
    print("="*80)
    print("\nOriginal Whisper Output:")
    print("-"*80)
    print(test_text)
    print()
    
    try:
        from punctuation_restorer import create_punctuation_restorer
        
        print("\nInitializing punctuation restorer...")
        print("-"*80)
        restorer = create_punctuation_restorer()
        
        print("\nRestoring punctuation (preserve Whisper hints mode)...")
        print("-"*80)
        refined_text = restorer.refine_transcription(test_text, aggressive=False)
        
        print("\nRefined Output:")
        print("-"*80)
        print(refined_text)
        print()
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Compare lengths
        print(f"\nOriginal length: {len(test_text)} characters")
        print(f"Refined length:  {len(refined_text)} characters")
        print(f"Difference:      {len(refined_text) - len(test_text):+d} characters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggressive_mode():
    """Test aggressive mode that fully re-predicts punctuation."""
    
    # Text without punctuation
    test_text = """what can be said about the difference between a system of development and a way of development now people like yourselves who are interested in doing something about your spiritual life sooner or later will come to contact with a group with maybe a school maybe a teacher maybe somebody who has some knowledge of these things"""
    
    print("\n" + "="*80)
    print("TESTING AGGRESSIVE MODE (Full Re-prediction)")
    print("="*80)
    print("\nUnpunctuated Input:")
    print("-"*80)
    print(test_text)
    print()
    
    try:
        from punctuation_restorer import create_punctuation_restorer
        
        restorer = create_punctuation_restorer()
        
        print("\nRestoring punctuation (aggressive mode)...")
        print("-"*80)
        refined_text = restorer.refine_transcription(test_text, aggressive=True)
        
        print("\nFully Restored Output:")
        print("-"*80)
        print(refined_text)
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🔤 Punctuation Restoration Test Suite\n")
    
    # Test 1: Preserve Whisper hints
    success1 = test_punctuation_restorer()
    
    # Test 2: Aggressive mode
    success2 = test_aggressive_mode()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Preserve hints mode: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Aggressive mode:     {'✅ PASS' if success2 else '❌ FAIL'}")
    print()
    
    if success1 and success2:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)
