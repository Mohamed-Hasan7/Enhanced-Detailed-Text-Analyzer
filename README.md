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

```pip install -r requirements.txt ```
```python -m spacy download en_core_web_sm```


### Running the Application
To launch the application, run:
```python main.py```

### Usage
Comparative Analysis Mode
Switch to the Comparative Analysis tab.
Enter your first text in the left text box and the second text in the right text box.
Click the "Compare Analysis" button.
A new report window will display the detailed side-by-side analysis.
Single Text Analysis Mode
Switch to the Single Text Analysis tab.
Paste or type the text you wish to analyze in the large text box.
Click the "Analyze Text" button.
A new report window will display the detailed analysis of your text.
Additional Controls
Save Report: Click the "Save Report" button (located in the bottom control frame) to save the current report as a text file.
Copy Report: Use the "Copy Report" button to copy the report to your clipboard.
Reset: The "Reset" button clears all input fields.
Toggle Dark/Light Mode: Use the "Toggle Dark/Light" button to switch the appearance of the application.
Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to modify.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements
CustomTkinter for the modern UI components.
spaCy for natural language processing.
VADER Sentiment for sentiment analysis.
Pyperclip for clipboard functionality.
Special thanks to all contributors and users for their feedback and support.
