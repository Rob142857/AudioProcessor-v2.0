"""Test kredor/punctuate-all model."""

from punctuation_restorer import create_punctuation_restorer

test_text = """And we start off just using the fingertips as lightly as possible just lightly touching the forehead stroking over the head as lightly as possible then going down behind the ears bringing the fingers down over the chest and then two or three circular movements very lightly over the heart and solar plexus region and take our fingers over the outside of the thighs to the knees and then we bring our hands back underneath the thighs as far as possible towards the genital region and then repeat the whole cycle over again just a matter of doing that for five or seven minutes to the accompaniment of some sort of music which you find pleasant to yourself but the lighter the touch the better it is"""

print("Testing kredor/punctuate-all model...")
print("="*80)

restorer = create_punctuation_restorer()
result = restorer.restore_punctuation(test_text, preserve_whisper_hints=False)

print("\nResult:")
print("="*80)
print(result)
