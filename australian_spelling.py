"""
Australian/British Spelling Converter and Text Normalizer

Converts American spellings to Australian/British spellings and fixes common formatting issues.
"""

import re
from typing import Dict


# Comprehensive American to Australian/British spelling mappings
SPELLING_MAP: Dict[str, str] = {
    # -ize to -ise
    "realize": "realise", "realizes": "realises", "realized": "realised", "realizing": "realising",
    "recognize": "recognise", "recognizes": "recognises", "recognized": "recognised", "recognizing": "recognising",
    "organize": "organise", "organizes": "organises", "organized": "organised", "organizing": "organising",
    "analyze": "analyse", "analyzes": "analyses", "analyzed": "analysed", "analyzing": "analysing",
    "visualize": "visualise", "visualizes": "visualises", "visualized": "visualised", "visualizing": "visualising",
    "minimize": "minimise", "minimizes": "minimises", "minimized": "minimised", "minimizing": "minimising",
    "maximize": "maximise", "maximizes": "maximises", "maximized": "maximised", "maximizing": "maximising",
    "crystallize": "crystallise", "crystallizes": "crystallises", "crystallized": "crystallised",
    "emphasize": "emphasise", "emphasizes": "emphasises", "emphasized": "emphasised", "emphasizing": "emphasising",
    "summarize": "summarise", "summarizes": "summarises", "summarized": "summarised", "summarizing": "summarising",
    "characterize": "characterise", "characterizes": "characterises", "characterized": "characterised",
    "symbolize": "symbolise", "symbolizes": "symbolises", "symbolized": "symbolised", "symbolizing": "symbolising",
    "utilize": "utilise", "utilizes": "utilises", "utilized": "utilised", "utilizing": "utilising",
    "specialize": "specialise", "specializes": "specialises", "specialized": "specialised", "specializing": "specialising",
    "generalize": "generalise", "generalizes": "generalises", "generalized": "generalised", "generalizing": "generalising",
    "memorize": "memorise", "memorizes": "memorises", "memorized": "memorised", "memorizing": "memorising",
    "stabilize": "stabilise", "stabilizes": "stabilises", "stabilized": "stabilised", "stabilizing": "stabilising",
    "apologize": "apologise", "apologizes": "apologises", "apologized": "apologised", "apologizing": "apologising",
    "criticize": "criticise", "criticizes": "criticises", "criticized": "criticised", "criticizing": "criticising",
    "prioritize": "prioritise", "prioritizes": "prioritises", "prioritized": "prioritised", "prioritizing": "prioritising",
    "theorize": "theorise", "theorizes": "theorises", "theorized": "theorised", "theorizing": "theorising",
    
    # -or to -our
    "color": "colour", "colors": "colours", "colored": "coloured", "coloring": "colouring",
    "favor": "favour", "favors": "favours", "favored": "favoured", "favoring": "favouring",
    "honor": "honour", "honors": "honours", "honored": "honoured", "honoring": "honouring",
    "labor": "labour", "labors": "labours", "labored": "laboured", "laboring": "labouring",
    "neighbor": "neighbour", "neighbors": "neighbours", "neighboring": "neighbouring",
    "flavor": "flavour", "flavors": "flavours", "flavored": "flavoured", "flavoring": "flavouring",
    "harbor": "harbour", "harbors": "harbours", "harbored": "harboured", "harboring": "harbouring",
    "vigor": "vigour", "vigorous": "vigorous",  # vigorous stays the same
    "behavior": "behaviour", "behaviors": "behaviours", "behavioral": "behavioural",
    "rumor": "rumour", "rumors": "rumours", "rumored": "rumoured",
    "tumor": "tumour", "tumors": "tumours",
    "endeavor": "endeavour", "endeavors": "endeavours", "endeavored": "endeavoured",
    
    # -er to -re
    "center": "centre", "centers": "centres", "centered": "centred", "centering": "centring",
    "theater": "theatre", "theaters": "theatres",
    "meter": "metre", "meters": "metres",
    "fiber": "fibre", "fibers": "fibres",
    "caliber": "calibre", "calibers": "calibres",
    "liter": "litre", "liters": "litres",
    "saber": "sabre", "sabers": "sabres",
    "somber": "sombre",
    
    # -og to -ogue
    "analog": "analogue", "analogs": "analogues",
    "catalog": "catalogue", "catalogs": "catalogues", "cataloged": "catalogued",
    "dialog": "dialogue", "dialogs": "dialogues",
    
    # -ense to -ence
    "defense": "defence", "defenses": "defences", "defensive": "defensive",  # defensive stays
    "offense": "offence", "offenses": "offences", "offensive": "offensive",  # offensive stays
    "license": "licence", "licenses": "licences", "licensed": "licensed",  # verb form stays -se
    "pretense": "pretence", "pretenses": "pretences",
    
    # -ll-
    "traveling": "travelling", "traveled": "travelled", "traveler": "traveller", "travelers": "travellers",
    "modeling": "modelling", "modeled": "modelled", "modeler": "modeller",
    "labeled": "labelled", "labeling": "labelling",
    "canceled": "cancelled", "canceling": "cancelling", "cancellation": "cancellation",  # -llation stays
    "signaling": "signalling", "signaled": "signalled",
    
    # misc
    "program": "programme", "programs": "programmes",  # in non-computing contexts
    "gray": "grey", "grays": "greys",
    "check": "cheque",  # only for bank cheques, not "check" verb
    "tire": "tyre", "tires": "tyres",  # vehicle tyres only
    "sulfur": "sulphur",
    "aluminum": "aluminium",
    "mold": "mould", "molds": "moulds", "molded": "moulded", "molding": "moulding",
    "plow": "plough", "plows": "ploughs", "plowed": "ploughed", "plowing": "ploughing",
    "skeptic": "sceptic", "skeptical": "sceptical", "skepticism": "scepticism",
}


def convert_to_australian_spelling(text: str) -> str:
    """
    Convert American spellings to Australian/British spellings.
    
    Args:
        text: Input text with American spellings
        
    Returns:
        Text with Australian/British spellings
    """
    result = text
    
    # Sort by length (longest first) to handle compound words correctly
    for american, australian in sorted(SPELLING_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        # Case-insensitive replacement with case preservation
        # Match whole words only (with word boundaries)
        pattern = r'\b' + re.escape(american) + r'\b'
        
        def replace_with_case(match):
            matched_text = match.group(0)
            # Preserve case
            if matched_text.isupper():
                return australian.upper()
            elif matched_text[0].isupper():
                return australian.capitalize()
            else:
                return australian
        
        result = re.sub(pattern, replace_with_case, result, flags=re.IGNORECASE)
    
    return result


def fix_number_formatting(text: str) -> str:
    """
    Fix number formatting issues like '2, 500' -> '2,500' or '2500'.
    
    Removes spaces around commas in numbers and ensures proper thousand separators.
    
    Args:
        text: Input text with malformed numbers
        
    Returns:
        Text with corrected number formatting
    """
    # Fix numbers with spaces after commas: "2, 500" -> "2,500"
    # Match digit(s), comma, space(s), digit(s)
    text = re.sub(r'(\d+),\s+(\d{3})', r'\1,\2', text)
    
    # Fix numbers with spaces before commas: "2 ,500" -> "2,500"
    text = re.sub(r'(\d+)\s+,(\d{3})', r'\1,\2', text)
    
    # Fix standalone comma-separated numbers with spaces: "1 , 000" -> "1,000"
    text = re.sub(r'(\d+)\s*,\s*(\d{3})', r'\1,\2', text)
    
    return text


def normalize_text(text: str, use_australian_spelling: bool = True, fix_numbers: bool = True) -> str:
    """
    Apply all text normalization: Australian spelling, number formatting, etc.
    
    Args:
        text: Input text to normalize
        use_australian_spelling: Whether to convert to Australian spelling
        fix_numbers: Whether to fix number formatting
        
    Returns:
        Normalized text
    """
    result = text
    
    if fix_numbers:
        result = fix_number_formatting(result)
    
    if use_australian_spelling:
        result = convert_to_australian_spelling(result)
    
    return result


if __name__ == "__main__":
    # Test the converter
    test_text = """
    I realize that the center of the theater has 2, 500 colors organized in a catalog.
    The neighbor's behavior was characterized by skeptical analysis and gray modeling.
    We recognize the defense requires specialized labor and traveling to analyze the fiber.
    """
    
    print("Original:")
    print(test_text)
    print("\nConverted:")
    print(normalize_text(test_text))
