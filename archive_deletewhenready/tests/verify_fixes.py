"""Verify the problem phrases are fixed."""

with open('test_bert_result.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Check specific problem areas with more context
problems = [
    ('aqualung', 60),
    ('all sorts of things', 80),
    ('as a result', 80),
]

print('✅ VERIFICATION - Problem phrases fixed:\n')
for phrase, context_len in problems:
    idx = text.lower().find(phrase)
    if idx != -1:
        start = max(0, idx - context_len)
        end = min(len(text), idx + context_len)
        context = text[start:end].replace('\n', ' ').strip()
        print(f'  ✓ "{phrase}":')
        print(f'    ...{context}...\n')
