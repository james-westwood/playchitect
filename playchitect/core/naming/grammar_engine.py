"""Grammar engine for natural-sounding playlist names.

Implements the Royal Order of adjectives and provides synonym replacement
to avoid repetitive naming. Uses pure Python with no external dependencies.
"""

# Royal Order of adjectives (standard English ordering)
ROYAL_ORDER: list[str] = [
    "Opinion",
    "Size",
    "Age",
    "Shape",
    "Color",
    "Origin",
    "Material",
    "Purpose",
]

# Word-to-category mapping for known adjectives
WORD_CATEGORIES: dict[str, str] = {
    # Opinion
    "Dark": "Opinion",
    "Ethereal": "Opinion",
    "Minimal": "Opinion",
    "Deep": "Opinion",
    "Intense": "Opinion",
    "Subtle": "Opinion",
    "Bright": "Opinion",
    "Raw": "Opinion",
    "Polished": "Opinion",
    "Gritty": "Opinion",
    "Smooth": "Opinion",
    "Rough": "Opinion",
    "Clean": "Opinion",
    "Pure": "Opinion",
    "Harsh": "Opinion",
    # Size
    "Vast": "Size",
    "Dense": "Size",
    "Massive": "Size",
    "Compact": "Size",
    "Expansive": "Size",
    "Intimate": "Size",
    "Wide": "Size",
    "Narrow": "Size",
    # Age
    "Late": "Age",
    "Ancient": "Age",
    "New": "Age",
    "Fresh": "Age",
    "Vintage": "Age",
    "Modern": "Age",
    "Classic": "Age",
    # Shape
    "Geometric": "Shape",
    "Flowing": "Shape",
    "Angular": "Shape",
    "Curved": "Shape",
    "Round": "Shape",
    "Linear": "Shape",
    "Jagged": "Shape",
    # Color
    "Midnight": "Color",
    "Amber": "Color",
    "Azure": "Color",
    "Crimson": "Color",
    "Golden": "Color",
    "Silver": "Color",
    "Obsidian": "Color",
    # Origin
    "Nordic": "Origin",
    "Urban": "Origin",
    "Tribal": "Origin",
    "Industrial": "Origin",
    "Tropical": "Origin",
    "Cosmic": "Origin",
    "Digital": "Origin",
    "Analog": "Origin",
    # Material
    "Metallic": "Material",
    "Organic": "Material",
    "Synthetic": "Material",
    "Wooden": "Material",
    "Crystal": "Material",
    # Purpose
    "Driving": "Purpose",
    "Hypnotic": "Purpose",
    "Dancing": "Purpose",
    "Focusing": "Purpose",
    "Relaxing": "Purpose",
    "Energizing": "Purpose",
}

# Synonym mappings to avoid repetitive naming
SYNONYMS: dict[str, list[str]] = {
    "Dark": ["Shadowed", "Dim", "Nocturnal", "Obscure", "Gloomy"],
    "Bright": ["Luminous", "Radiant", "Brilliant", "Vivid", "Gleaming"],
    "Deep": ["Profound", "Bottomless", "Submerged", "Abyssal", "Low"],
    "Intense": ["Extreme", "Fierce", "Severe", "Powerful", "Strong"],
    "Smooth": ["Sleek", "Velvety", "Silky", "Polished", "Refined"],
    "Vast": ["Immense", "Boundless", "Endless", "Infinite", "Expansive"],
    "Driving": ["Propulsive", "Pulsing", "Thrusting", "Pushing", "Moving"],
    "Ethereal": ["Celestial", "Heavenly", "Airy", "Delicate", "Light"],
    "Minimal": ["Sparse", "Stripped", "Reduced", "Simple", "Bare"],
    "Geometric": ["Structured", "Patterned", "Angular", "Regular", "Ordered"],
    "Nordic": ["Scandinavian", "Arctic", "Northern", "Polar", "Baltic"],
    "Urban": ["Metropolitan", "City", "Street", "Downtown", "Concrete"],
    "Organic": ["Natural", "Living", "Biological", "Earthy", "Raw"],
    "Hypnotic": ["Mesmerizing", "Trance", "Enchanting", "Spellbinding", "Magical"],
    "Industrial": ["Mechanical", "Factory", "Steel", "Machine", "Synthetic"],
}


def sort_by_royal_order(words: list[str]) -> list[str]:
    """Sort adjectives according to the Royal Order.

    Words not in WORD_CATEGORIES are placed at the end, preserving
    their relative order among themselves.

    Args:
        words: List of adjective words to sort.

    Returns:
        Sorted list with words ordered by Royal Order category.

    Example:
        >>> sort_by_royal_order(['Driving', 'Dark', 'Nordic'])
        ['Dark', 'Nordic', 'Driving']
    """
    known: list[str] = []
    unknown: list[str] = []

    for word in words:
        if word in WORD_CATEGORIES:
            known.append(word)
        else:
            unknown.append(word)

    # Sort known words by their category's position in ROYAL_ORDER
    def _category_index(word: str) -> int:
        category = WORD_CATEGORIES[word]
        return ROYAL_ORDER.index(category)

    known.sort(key=_category_index)
    return known + unknown


def pick_synonym(word: str, used: set[str]) -> str:
    """Pick a synonym for a word that hasn't been used yet.

    If the original word is already in `used`, attempts to find an unused
    synonym. Returns the original word if it's not used or if no synonyms
    are available.

    Args:
        word: The original word to check.
        used: Set of words already used in the playlist name.

    Returns:
        An unused synonym if the original is used, otherwise the original word.

    Example:
        >>> pick_synonym('Dark', {'Shadowed'})
        'Dim'  # Returns a synonym not in the used set
    """
    # If word is not already used, keep it
    if word not in used:
        return word

    # Word is used, try to find a synonym that's not used
    if word not in SYNONYMS:
        return word

    for synonym in SYNONYMS[word]:
        if synonym not in used:
            return synonym

    return word


def indefinite_article(word: str) -> str:
    """Return the appropriate indefinite article for a word.

    Args:
        word: The word to determine the article for.

    Returns:
        'an' if the word starts with a vowel (A, E, I, O, U),
        'a' otherwise.

    Example:
        >>> indefinite_article('Intense')
        'an'
        >>> indefinite_article('Driving')
        'a'
    """
    if not word:
        return "a"

    first_char = word[0].upper()
    return "an" if first_char in "AEIOU" else "a"


def generate_name(
    descriptors: list[str],
    noun: str,
    used_names: set[str] | None = None,
) -> str:
    """Generate a natural-sounding playlist name from descriptors and noun.

    Sorts descriptors by Royal Order, applies synonym substitution to avoid
    repetition, and joins them with the noun in title case.

    Args:
        descriptors: List of adjective descriptors.
        noun: The head noun for the playlist name.
        used_names: Set of words already used in other playlist names.
            Defaults to empty set if None.

    Returns:
        A title-cased playlist name string.

    Example:
        >>> generate_name(['Dark', 'Driving'], 'Journey', set())
        'Dark Driving Journey'
    """
    if used_names is None:
        used_names = set()

    # Sort descriptors by Royal Order
    sorted_descriptors = sort_by_royal_order(descriptors)

    # Apply synonym substitution to avoid repetition
    final_descriptors: list[str] = []
    for desc in sorted_descriptors:
        final_desc = pick_synonym(desc, used_names)
        final_descriptors.append(final_desc)
        used_names.add(final_desc)

    # Join descriptors with noun
    all_words = final_descriptors + [noun]
    return " ".join(all_words).title()
