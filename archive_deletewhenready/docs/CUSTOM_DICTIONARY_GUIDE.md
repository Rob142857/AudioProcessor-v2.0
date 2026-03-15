# Custom Word Dictionary System

The Audio Transcription system now includes a powerful custom word dictionary feature that automatically corrects specialized terms, proper names, and technical vocabulary during the transcription process.

## How It Works

The custom dictionary system is integrated into the Ultra Text Processor as **Pass 2** (between punctuation restoration and sentence segmentation). This ensures that word corrections are applied early in the processing pipeline for maximum effectiveness.

## Files

- **`custom_dictionary.md`** - The main dictionary file containing word substitutions
- **`custom_dictionary.py`** - The dictionary processing engine
- **`dict_manager.py`** - Command-line tool for managing the dictionary

## Dictionary Format

The dictionary uses a simple `incorrect_word -> correct_word` format in markdown code blocks:

```markdown
## Proper Names
```
gurdief -> Gurdjieff
wyspensky -> Ouspensky
shakespeare -> Shakespeare
```
```

## Adding New Words

### Method 1: Command Line Tool (Recommended)
```powershell
# Add a single word
python dict_manager.py add "gurdief" "Gurdjieff"

# Add a word to a specific section
python dict_manager.py add "nietzsche" "Nietzsche" --section "## Philosophers"

# Test if a word would be substituted
python dict_manager.py test "gurdief"

# Test dictionary on sample text
python dict_manager.py test-text "gurdief was a teacher"

# List all current substitutions
python dict_manager.py list
```

### Method 2: Direct File Editing
Edit `custom_dictionary.md` directly and add entries in the appropriate sections.

## Dictionary Features

### Case-Insensitive Matching
- Dictionary entries use lowercase for the "incorrect" word
- Matching is case-insensitive during processing
- Proper capitalization is applied via the "correct" word

### Word Boundaries
- Uses word boundary matching to avoid partial replacements
- "christ" matches "christ" but not "christmas"

### Longest Match First
- Longer phrases are processed before shorter ones
- "the new testament" is matched before "new" or "testament"

### Processing Order
1. **Pass 1**: Basic punctuation restoration
2. **Pass 2**: **Custom dictionary substitutions** ← NEW
3. **Pass 3**: Advanced sentence segmentation
4. **Pass 4**: Capitalization and proper nouns
5. **Pass 5**: Grammar and style improvements
6. **Pass 6**: Final cleanup and formatting

## Pre-loaded Dictionary

The system comes with 40+ pre-configured substitutions including:

### Proper Names
- `gurdief` → `Gurdjieff`
- `wyspensky` → `Ouspensky`
- `shakespeare` → `Shakespeare`
- `christ` → `Christ`
- `pythagoras` → `Pythagoras`

### Religious/Spiritual Terms
- `the new testament` → `The New Testament`
- `the old testament` → `The Old Testament`
- `the gospels` → `The Gospels`
- `the bible` → `The Bible`

### Technical Terms
- `i ching` → `I Ching`
- `feng shui` → `Feng Shui`
- `synchronicity` → `synchronicity`
- `eschatology` → `eschatology`

### Places
- `stonehenge` → `Stonehenge`
- `isle of man` → `Isle of Man`
- `oxford` → `Oxford`

## Usage Examples

### Testing the System
```python
from custom_dictionary import CustomDictionary

dict_system = CustomDictionary()
text = "gurdief and wyspensky studied the new testament"
result = dict_system.apply_substitutions(text)
print(result)  # "Gurdjieff and Ouspensky studied The New Testament"
```

### Integration with Transcription
The dictionary is automatically used during transcription. No additional setup required!

## Best Practices

### Adding Words
1. **Use lowercase** for the "incorrect" word (left side)
2. **Use proper capitalization** for the "correct" word (right side)
3. **Test additions** before using in production
4. **Be specific** - avoid common words that might have multiple meanings
5. **Organize by sections** for easier maintenance

### Common Use Cases
- **Proper names** that Whisper consistently gets wrong
- **Technical terminology** specific to your domain
- **Foreign words** or phrases
- **Brand names** and trademarked terms
- **Biblical/religious terms** with specific capitalization
- **Historical figures** and places

### Testing Workflow
1. Add new words using `dict_manager.py add`
2. Test with `dict_manager.py test-text "sample phrase"`
3. Run a small transcription to verify
4. Adjust if needed

## Performance Impact

- **Minimal overhead** - dictionary processing is very fast
- **Early processing** - corrections happen before other text processing
- **Parallel compatible** - works with multi-threaded text processing
- **Memory efficient** - dictionary is loaded once and reused

## Troubleshooting

### Word Not Being Substituted
1. Check spelling in dictionary entry
2. Verify word boundaries (whole words only)
3. Test with `dict_manager.py test "word"`
4. Check for conflicting longer phrases

### Unwanted Substitutions
1. Use more specific dictionary entries
2. Add word boundaries if needed
3. Remove problematic entries
4. Test with sample text first

### Dictionary Not Loading
- Ensure `custom_dictionary.md` exists in the project root
- Check file formatting (proper `->` syntax)
- Verify no syntax errors in markdown

## Statistics

Use the dictionary manager to get statistics:
```powershell
python dict_manager.py list
# Shows total count and lists all substitutions
```

The system reports substitution counts during processing:
```
📝 Applied 8 custom word substitutions
```

## Future Enhancements

- **Domain-specific dictionaries** - Load different dictionaries for different content types
- **Context-aware substitutions** - Different corrections based on surrounding text
- **Learning mode** - Suggest new dictionary entries based on transcription results
- **Pronunciation-based matching** - Match words that sound similar
- **Regular expression patterns** - Support for pattern-based substitutions