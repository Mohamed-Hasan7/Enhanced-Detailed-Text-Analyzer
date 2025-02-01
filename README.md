# Enhanced-Detailed-Text-Analyzer

## Overview

The **Enhanced Detailed Text Analyzer** is a Python-based desktop application for in-depth analysis of creative writing. It offers two modes:

1. **Comparative Analysis Mode:** Compare two texts side-by-side with a detailed, line-by-line report covering lexical, syntactic, readability, sentiment, narrative, and more metrics.
2. **Single Text Analysis Mode:** Analyze one text in depth with additional breakdowns and commentary.


## Features

- **Advanced Text Analysis:**
  - **Lexical Metrics:** Total words, unique words, lexical diversity, average word length, syllable counts, and vocabulary sophistication.
  - **Syntactic & POS Analysis:** Parts-of-speech frequencies, dependency tree depth, passive voice detection, and ratios (adjective/noun, adverb/verb).
  - **Sentence & Readability Metrics:** Sentence count, average/median sentence length, variability, Flesch Reading Ease, Flesch–Kincaid Grade Level, and SMOG index.
  - **Additional Metrics:** Paragraph analysis (count and average words per paragraph), sentiment analysis (using VADER), pronoun usage (first vs. third person), dialogue detection, cliché detection, and transition word frequency.
  - **Overall Style Grading:** A computed overall grade with detailed commentary on strengths and areas for improvement.

- **Modern GUI:**
  - **Tabbed Interface:** Easily switch between Comparative Analysis and Single Text Analysis modes.
  - **Dedicated Report Window:** Analysis results open in a separate, scrollable window for clear, focused reading.
  - **Theme Support:** Built‑in dark and light modes with an appearance mode toggle for a customizable experience.
  - **Enhanced Controls:** Buttons for comparing texts, single text analysis, saving reports, copying reports to clipboard, and resetting input fields.

## Installation

### Prerequisites

- Python 3.6 or later

The application requires the following Python packages:

- **[spaCy](https://spacy.io/):** 
- **[VADER Sentiment](https://github.com/cjhutto/vaderSentiment):**
- **[Pyperclip](https://github.com/asweigart/pyperclip):**
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter):**

You can install these dependencies using pip along with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
