"""Test the tuned BERT punctuation refinement."""

from punctuation_restorer import create_punctuation_restorer
from paragraph_segmenter import create_paragraph_segmenter

# Read sample
with open('test_bert_tuning.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('Testing BERT with aggressive mode and larger chunks...')
print('='*80)

# BERT refinement with new tuned parameters
punct_restorer = create_punctuation_restorer()
text = punct_restorer.refine_transcription(text, aggressive=True, chunk_size=400)

print('\nBERT refinement complete. Adding paragraph breaks...')
print('='*80)

# Paragraph segmentation
segmenter = create_paragraph_segmenter(similarity_threshold=0.48)
text = segmenter.segment_paragraphs(text, verbose=False)
text = segmenter.refine_paragraphs(text, min_paragraph_length=200, max_paragraph_length=900)

print('\n' + '='*80)
print('FINAL RESULT:')
print('='*80 + '\n')
print(text)

# Save
with open('test_bert_result.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print('\n✅ Saved to test_bert_result.txt')
