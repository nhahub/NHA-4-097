"""
utils/language_detector.py
Detects Arabic vs English (and other languages) in user input.
Bonus feature: multilingual support (AR/EN).
"""

from __future__ import annotations
import re
import unicodedata


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")


def detect_language(text: str) -> str:
    """
    Returns ISO 639-1 language code.
    'ar' for Arabic, 'en' for English, 'mixed' if both detected.
    """
    text = text.strip()
    if not text:
        return "en"

    arabic_chars = len(ARABIC_PATTERN.findall(text))
    latin_chars = len(re.findall(r"[a-zA-Z]+", text))

    total = arabic_chars + latin_chars
    if total == 0:
        return "en"

    arabic_ratio = arabic_chars / total

    if arabic_ratio > 0.6:
        return "ar"
    elif arabic_ratio > 0.2:
        return "mixed"
    else:
        return "en"


def is_arabic(text: str) -> bool:
    return detect_language(text) == "ar"


def get_direction(lang: str) -> str:
    """RTL for Arabic, LTR for others."""
    return "rtl" if lang == "ar" else "ltr"


def translate_label(label: str, lang: str) -> str:
    """
    Simple label translation for UI elements.
    For full translation, use the LLM response agent.
    """
    translations = {
        "ar": {
            "Upload": "رفع الملفات",
            "Clear Memory": "مسح الذاكرة",
            "Chat": "المحادثة",
            "Dashboard": "لوحة التحكم",
            "Settings": "الإعدادات",
            "Search": "بحث",
            "Analyze": "تحليل",
            "Answer": "الإجابة",
            "Sources": "المصادر",
            "Loading": "جارٍ التحميل...",
            "Error": "خطأ",
            "No data": "لا توجد بيانات",
            "Rows": "صفوف",
            "Columns": "أعمدة",
        }
    }
    lang_dict = translations.get(lang, {})
    return lang_dict.get(label, label)


def clean_for_embedding(text: str) -> str:
    """
    Normalize Arabic/English text before embedding.
    Removes diacritics, normalizes whitespace.
    """
    # Remove Arabic diacritics (harakat)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Normalize alef variants
    text = re.sub(r"[إأٱآا]", "ا", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text