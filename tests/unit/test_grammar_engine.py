"""Unit tests for grammar_engine module."""

from playchitect.core.naming.grammar_engine import (
    ROYAL_ORDER,
    SYNONYMS,
    WORD_CATEGORIES,
    generate_name,
    indefinite_article,
    pick_synonym,
    sort_by_royal_order,
)


class TestRoyalOrder:
    """Test ROYAL_ORDER constant."""

    def test_royal_order_has_eight_categories(self) -> None:
        """ROYAL_ORDER should contain exactly 8 categories."""
        assert len(ROYAL_ORDER) == 8

    def test_royal_order_opinion_first(self) -> None:
        """Opinion should be the first category in ROYAL_ORDER."""
        assert ROYAL_ORDER[0] == "Opinion"

    def test_royal_order_purpose_last(self) -> None:
        """Purpose should be the last category in ROYAL_ORDER."""
        assert ROYAL_ORDER[-1] == "Purpose"


class TestWordCategories:
    """Test WORD_CATEGORIES dictionary."""

    def test_word_categories_has_minimum_entries(self) -> None:
        """WORD_CATEGORIES should have at least 40 entries."""
        assert len(WORD_CATEGORIES) >= 40

    def test_word_categories_opinion_entries(self) -> None:
        """Opinion category should have expected words."""
        opinion_words = [
            "Dark",
            "Ethereal",
            "Minimal",
            "Deep",
            "Intense",
            "Subtle",
        ]
        for word in opinion_words:
            assert WORD_CATEGORIES.get(word) == "Opinion"

    def test_word_categories_size_entries(self) -> None:
        """Size category should have expected words."""
        size_words = ["Vast", "Dense", "Massive", "Compact", "Expansive"]
        for word in size_words:
            assert WORD_CATEGORIES.get(word) == "Size"

    def test_word_categories_age_entries(self) -> None:
        """Age category should have expected words."""
        age_words = ["Late", "Ancient", "New", "Fresh", "Vintage"]
        for word in age_words:
            assert WORD_CATEGORIES.get(word) == "Age"

    def test_word_categories_shape_entries(self) -> None:
        """Shape category should have expected words."""
        shape_words = ["Geometric", "Flowing", "Angular", "Curved"]
        for word in shape_words:
            assert WORD_CATEGORIES.get(word) == "Shape"

    def test_word_categories_color_entries(self) -> None:
        """Color category should have expected words."""
        color_words = ["Midnight", "Amber", "Azure", "Crimson", "Golden"]
        for word in color_words:
            assert WORD_CATEGORIES.get(word) == "Color"

    def test_word_categories_origin_entries(self) -> None:
        """Origin category should have expected words."""
        origin_words = ["Nordic", "Urban", "Tribal", "Industrial", "Tropical"]
        for word in origin_words:
            assert WORD_CATEGORIES.get(word) == "Origin"

    def test_word_categories_material_entries(self) -> None:
        """Material category should have expected words."""
        material_words = ["Metallic", "Organic", "Synthetic", "Wooden"]
        for word in material_words:
            assert WORD_CATEGORIES.get(word) == "Material"

    def test_word_categories_purpose_entries(self) -> None:
        """Purpose category should have expected words."""
        purpose_words = ["Driving", "Hypnotic", "Dancing", "Focusing"]
        for word in purpose_words:
            assert WORD_CATEGORIES.get(word) == "Purpose"


class TestSynonyms:
    """Test SYNONYMS dictionary."""

    def test_synonyms_has_minimum_entries(self) -> None:
        """SYNONYMS should have at least 15 entries."""
        assert len(SYNONYMS) >= 15

    def test_synonyms_dark_has_alternatives(self) -> None:
        """Dark should have multiple synonyms."""
        assert "Dark" in SYNONYMS
        assert len(SYNONYMS["Dark"]) >= 3
        assert "Shadowed" in SYNONYMS["Dark"]

    def test_synonyms_bright_has_alternatives(self) -> None:
        """Bright should have multiple synonyms."""
        assert "Bright" in SYNONYMS
        assert len(SYNONYMS["Bright"]) >= 3

    def test_synonyms_deep_has_alternatives(self) -> None:
        """Deep should have multiple synonyms."""
        assert "Deep" in SYNONYMS
        assert len(SYNONYMS["Deep"]) >= 3


class TestSortByRoyalOrder:
    """Test sort_by_royal_order function."""

    def test_sort_by_royal_order_basic(self) -> None:
        """sort_by_royal_order sorts descriptors correctly."""
        result = sort_by_royal_order(["Driving", "Dark", "Nordic"])
        # Opinion (Dark) < Origin (Nordic) < Purpose (Driving)
        assert result == ["Dark", "Nordic", "Driving"]

    def test_sort_by_royal_order_single_word(self) -> None:
        """sort_by_royal_order handles single word."""
        result = sort_by_royal_order(["Dark"])
        assert result == ["Dark"]

    def test_sort_by_royal_order_empty_list(self) -> None:
        """sort_by_royal_order handles empty list."""
        result = sort_by_royal_order([])
        assert result == []

    def test_sort_by_royal_order_unknown_words_last(self) -> None:
        """Unknown words are placed at the end."""
        result = sort_by_royal_order(["Unknown", "Dark", "Mystery"])
        # Dark comes first (known, Opinion), then unknown words in order
        assert result[0] == "Dark"
        assert result[1:] == ["Unknown", "Mystery"]

    def test_sort_by_royal_order_multiple_same_category(self) -> None:
        """Words in same category maintain relative order."""
        result = sort_by_royal_order(["Intense", "Dark", "Bright"])
        # All are Opinion, maintain input order within category
        assert result == ["Intense", "Dark", "Bright"]

    def test_sort_by_royal_order_all_categories(self) -> None:
        """Test sorting across all eight categories."""
        words = [
            "Driving",  # Purpose
            "Dark",  # Opinion
            "Vast",  # Size
            "Ancient",  # Age
            "Geometric",  # Shape
            "Midnight",  # Color
            "Nordic",  # Origin
            "Metallic",  # Material
        ]
        result = sort_by_royal_order(words)
        expected = [
            "Dark",  # Opinion
            "Vast",  # Size
            "Ancient",  # Age
            "Geometric",  # Shape
            "Midnight",  # Color
            "Nordic",  # Origin
            "Metallic",  # Material
            "Driving",  # Purpose
        ]
        assert result == expected


class TestPickSynonym:
    """Test pick_synonym function."""

    def test_pick_synonym_returns_original_when_no_synonyms(self) -> None:
        """pick_synonym returns original word when no synonyms defined."""
        result = pick_synonym("NonExistent", set())
        assert result == "NonExistent"

    def test_pick_synonym_returns_original_when_not_used(self) -> None:
        """pick_synonym returns original word when it's not in used set."""
        result = pick_synonym("Dark", set())
        assert result == "Dark"

    def test_pick_synonym_returns_unused_synonym_when_original_used(self) -> None:
        """pick_synonym returns an unused synonym when original is used."""
        result = pick_synonym("Dark", {"Dark", "Shadowed"})
        assert result != "Dark"
        assert result != "Shadowed"
        assert result in SYNONYMS["Dark"]

    def test_pick_synonym_returns_original_when_all_synonyms_used(self) -> None:
        """pick_synonym returns original when all synonyms are used."""
        all_synonyms = set(SYNONYMS["Dark"])
        result = pick_synonym("Dark", all_synonyms)
        # Since all synonyms are used, and Dark is not in SYNONYMS,
        # we get Dark back (it's not in SYNONYMS, so we can't pick from there)
        assert result == "Dark"

    def test_pick_synonym_returns_first_available_synonym(self) -> None:
        """pick_synonym returns first available synonym when original is used."""
        used = {"Dark"} | set(SYNONYMS["Dark"][1:])  # Dark + all but first synonym
        result = pick_synonym("Dark", used)
        assert result == SYNONYMS["Dark"][0]

    def test_pick_synonym_returns_original_when_used_set_empty(self) -> None:
        """pick_synonym returns original word when used set is empty."""
        result = pick_synonym("Dark", set())
        assert result == "Dark"


class TestIndefiniteArticle:
    """Test indefinite_article function."""

    def test_indefinite_article_vowel_start(self) -> None:
        """Words starting with vowels get 'an'."""
        assert indefinite_article("Intense") == "an"
        assert indefinite_article("Ethereal") == "an"
        assert indefinite_article("Ancient") == "an"
        assert indefinite_article("Organic") == "an"
        assert indefinite_article("Urban") == "an"

    def test_indefinite_article_consonant_start(self) -> None:
        """Words starting with consonants get 'a'."""
        assert indefinite_article("Driving") == "a"
        assert indefinite_article("Dark") == "a"
        assert indefinite_article("Nordic") == "a"
        assert indefinite_article("Metallic") == "a"
        assert indefinite_article("Geometric") == "a"

    def test_indefinite_article_empty_string(self) -> None:
        """Empty string returns 'a'."""
        assert indefinite_article("") == "a"

    def test_indefinite_article_lowercase_vowel(self) -> None:
        """Lowercase vowels also get 'an'."""
        assert indefinite_article("intense") == "an"
        assert indefinite_article("ethereal") == "an"


class TestGenerateName:
    """Test generate_name function."""

    def test_generate_name_basic(self) -> None:
        """generate_name returns non-empty title-case string."""
        result = generate_name(["Dark", "Driving"], "Journey", set())
        assert result
        assert isinstance(result, str)
        assert result == result.title()

    def test_generate_name_single_descriptor(self) -> None:
        """generate_name works with single descriptor."""
        result = generate_name(["Dark"], "Journey", set())
        assert result == "Dark Journey"

    def test_generate_name_applies_royal_order(self) -> None:
        """generate_name sorts descriptors by Royal Order."""
        result = generate_name(["Driving", "Dark", "Nordic"], "Mix", set())
        # Should be sorted: Dark (Opinion), Nordic (Origin), Driving (Purpose)
        assert result == "Dark Nordic Driving Mix"

    def test_generate_name_applies_synonyms(self) -> None:
        """generate_name uses synonyms to avoid repetition."""
        used = {"Shadowed"}
        result = generate_name(["Dark"], "Mix", used)
        # Should not be Shadowed since it's in used
        assert "Shadowed" not in result

    def test_generate_name_tracks_used_words(self) -> None:
        """generate_name adds chosen words to used set."""
        used: set[str] = set()
        generate_name(["Dark", "Driving"], "Mix", used)
        # The chosen descriptors should be in used
        assert len(used) >= 2

    def test_generate_name_empty_descriptors(self) -> None:
        """generate_name works with empty descriptors."""
        result = generate_name([], "Journey", set())
        assert result == "Journey"

    def test_generate_name_title_case(self) -> None:
        """generate_name returns title-cased result."""
        result = generate_name(["dark", "driving"], "journey", set())
        assert result == "Dark Driving Journey"

    def test_generate_name_default_used_names(self) -> None:
        """generate_name works when used_names is not provided."""
        result = generate_name(["Dark", "Driving"], "Journey")
        assert result == "Dark Driving Journey"

    def test_generate_name_unknown_words_preserved(self) -> None:
        """Unknown words are preserved and placed at the end."""
        result = generate_name(["Mystery", "Dark"], "Mix", set())
        # Dark comes first (known), Mystery at the end
        assert result == "Dark Mystery Mix"
