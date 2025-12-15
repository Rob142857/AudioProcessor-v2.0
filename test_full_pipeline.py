"""
Full pipeline test: Punctuation + Paragraph Segmentation

Tests the complete enhancement pipeline on lecture transcripts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_full_pipeline(text: str):
    """Test both punctuation restoration and paragraph segmentation."""
    
    print("="*80)
    print("FULL PIPELINE TEST: Punctuation + Paragraph Segmentation")
    print("="*80)
    
    print("\n1️⃣  ORIGINAL WHISPER OUTPUT:")
    print("-"*80)
    print(text[:500] + "...\n")
    
    # Step 1: Punctuation Restoration
    try:
        from punctuation_restorer import create_punctuation_restorer
        
        print("\n2️⃣  APPLYING PUNCTUATION RESTORATION...")
        print("-"*80)
        restorer = create_punctuation_restorer()
        punctuated = restorer.refine_transcription(text, aggressive=False)
        print(f"✅ Punctuation complete\n")
        
    except Exception as e:
        print(f"❌ Punctuation failed: {e}")
        punctuated = text
    
    # Step 2: Semantic Paragraph Segmentation
    try:
        from paragraph_segmenter import create_paragraph_segmenter
        
        print("\n3️⃣  APPLYING SEMANTIC PARAGRAPH SEGMENTATION...")
        print("-"*80)
        segmenter = create_paragraph_segmenter(similarity_threshold=0.50)
        segmented = segmenter.segment_paragraphs(punctuated, verbose=True)
        
        # Refine paragraph lengths
        final = segmenter.refine_paragraphs(segmented, min_paragraph_length=150, max_paragraph_length=800)
        print(f"✅ Paragraph segmentation complete\n")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        final = punctuated
    
    # Results
    print("\n4️⃣  FINAL RESULT:")
    print("="*80)
    print(final)
    print("\n" + "="*80)
    
    # Statistics
    orig_paras = text.count('\n\n') + 1
    final_paras = final.count('\n\n') + 1
    
    print("\n📊 STATISTICS:")
    print("-"*80)
    print(f"Original paragraphs: {orig_paras}")
    print(f"Final paragraphs:    {final_paras}")
    print(f"Change:              {final_paras - orig_paras:+d}")
    
    return final


if __name__ == "__main__":
    # Your actual lecture sample
    sample_text = """Last week, we made an initial look at the role of colour in life, and we pointed out that various colourful molecules fill our world with all sorts of stimulating factors. That touch us and affect us. They also touch animals and they affect the surroundings. Colour is a radiation of various wavelengths into the environment. And even the environment is affected to a certain extent by colour. And tonight we're going to extend this a little more fully. It's a subject that we would need a hundred lectures and lots of experiments and observations to fully cover.
But it's necessary for us to try and understand eventually tonight where colour comes from, what its psycho-spiritual function is, and what happens to colours in the Spiritual world. It's very important for us to try and get this broad, overall understanding of why colour exists in the world at all.
Now once we were looking at colours last week they were confined mainly to plants what we forget is in the natural physical world a lot of substances physical substances. Inert substances are brightly coloured a lot of the chemical elements are coloured for example chlorine has a green colour brownine is red copper is a sort of brownish. Red colour gold is yellow bismuth is white antimony no, it's not antimony, it's SP is black and then we find manganese metal, it's silvery, with a reddish tinge. And then there's the rare metal, praiseodiamium, just forgotten the colour of that, that has a yellow tinge. Now, these are just a few of the 104-105 known chemical elements. Most of the metals are silvery. But we notice that manganese not only is silvery, it has a reddish tinge. Other metals have a bluish tinge. Osmium, for example, has a distinct bluish tinge to it."""
    
    result = test_full_pipeline(sample_text)
    
    # Optionally save
    with open('test_full_pipeline_output.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\n💾 Saved to: test_full_pipeline_output.txt")
