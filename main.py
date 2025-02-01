import customtkinter as ctk
import spacy
import re
import statistics
import math
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyperclip



def count_syllables(word):
    """Heuristic syllable counter."""
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for char in word:
        if char in vowels:
            if not prev_vowel:
                syllables += 1
            prev_vowel = True
        else:
            prev_vowel = False
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return syllables if syllables > 0 else 1

def get_common_tokens(doc, pos_tag, top=5):
    """Returns the most common tokens for a given POS."""
    return Counter(token.text.lower() for token in doc if token.pos_ == pos_tag and token.is_alpha).most_common(top)

def get_tree_depth(token):
    """Recursively computes maximum dependency tree depth from a token."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_depth(child) for child in children)

def analyze_sentiment(text):
    """Returns the average compound sentiment score using VADER."""
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return 0
    scores = [sentiment_analyzer.polarity_scores(sent.text)["compound"] for sent in sentences]
    return statistics.mean(scores)

def count_pronouns(doc):
    """Counts first-person and third-person pronouns."""
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    third_person = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
    fp_count = sum(1 for token in doc if token.text.lower() in first_person)
    tp_count = sum(1 for token in doc if token.text.lower() in third_person)
    return fp_count, tp_count

def analyze_paragraphs(text):
    """Splits text into paragraphs and returns paragraph count, average words per paragraph, and list of word counts."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    if not paragraphs:
        return 0, 0, []
    word_counts = [len([w for w in p.split() if w.isalpha()]) for p in paragraphs]
    avg = statistics.mean(word_counts) if word_counts else 0
    return len(paragraphs), avg, word_counts

def count_dialogue(text):
    """Counts dialogue lines based on quotation mark usage."""
    return sum(1 for line in text.splitlines() if '"' in line or '“' in line or '”' in line)

def detect_cliches(text):
    """Searches for common clichés in the text."""
    cliche_list = [
        "at the end of the day", "in the blink of an eye", "once in a lifetime",
        "last but not least", "in a nutshell", "better late than never", 
        "easy as pie", "when all is said and done"
    ]
    found = {}
    lower_text = text.lower()
    for phrase in cliche_list:
        count = lower_text.count(phrase)
        if count:
            found[phrase] = count
    return found

def count_transition_words(text):
    """Counts occurrences of common transition words."""
    transitions = [
        "however", "moreover", "therefore", "meanwhile",
        "consequently", "furthermore", "in addition", "on the other hand",
        "nevertheless", "nonetheless"
    ]
    counts = {}
    lower_text = text.lower()
    for word in transitions:
        c = lower_text.count(word)
        if c:
            counts[word] = c
    return counts

def detailed_analysis(text):
    """
    Performs an in-depth analysis of the text.
    Returns a dictionary with metrics covering lexical, syntactic, sentence,
    paragraph, sentiment, dialogue, cliché, and transition aspects.
    """
    analysis = {}
    doc = nlp(text)
    
    # Lexical Metrics
    word_tokens = [token for token in doc if token.is_alpha]
    total_words = len(word_tokens)
    unique_words = set(token.text.lower() for token in word_tokens)
    unique_word_count = len(unique_words)
    lexical_diversity = unique_word_count / total_words if total_words > 0 else 0
    avg_word_length = sum(len(token.text) for token in word_tokens) / total_words if total_words > 0 else 0
    syllable_counts = [count_syllables(token.text) for token in word_tokens]
    total_syllables = sum(syllable_counts)
    avg_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
    complex_words = sum(1 for s in syllable_counts if s >= 3)
    sophistication_ratio = complex_words / total_words if total_words > 0 else 0

    # POS Frequencies and Ratios
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)
    adj_noun_ratio = pos_counts.get("ADJ", 0) / pos_counts.get("NOUN", 1)
    adv_verb_ratio = pos_counts.get("ADV", 0) / pos_counts.get("VERB", 1)
    punct_counts = dict(Counter(token.text for token in doc if token.is_punct))
    
    # Sentence Metrics
    sentences = list(doc.sents)
    sentence_word_counts = []
    passive_sentences = 0
    dependency_depths = []
    for sent in sentences:
        sent_words = [t for t in sent if t.is_alpha]
        sentence_word_counts.append(len(sent_words))
        if any(token.dep_ == "auxpass" for token in sent):
            passive_sentences += 1
        dependency_depths.append(get_tree_depth(sent.root))
    total_sentences = len(sentences)
    if sentence_word_counts:
        avg_sentence_length = statistics.mean(sentence_word_counts)
        median_sentence_length = statistics.median(sentence_word_counts)
        sentence_variance = statistics.variance(sentence_word_counts) if len(sentence_word_counts) > 1 else 0
        std_sentence_length = statistics.stdev(sentence_word_counts) if len(sentence_word_counts) > 1 else 0
        sentence_cv = std_sentence_length / avg_sentence_length if avg_sentence_length > 0 else 0
    else:
        avg_sentence_length = median_sentence_length = sentence_variance = std_sentence_length = sentence_cv = 0
    passive_ratio = passive_sentences / total_sentences if total_sentences > 0 else 0
    if dependency_depths:
        avg_dep_depth = statistics.mean(dependency_depths)
        median_dep_depth = statistics.median(dependency_depths)
        dep_depth_std = statistics.stdev(dependency_depths) if len(dependency_depths) > 1 else 0
    else:
        avg_dep_depth = median_dep_depth = dep_depth_std = 0

    # Readability Metrics
    if total_sentences > 0 and total_words > 0:
        flesch_reading_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        flesch_kincaid_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    else:
        flesch_reading_ease = flesch_kincaid_grade = 0
    if total_sentences >= 30:
        smog_index = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291
    else:
        smog_index = None

    # Common Tokens by POS
    common_adjectives = get_common_tokens(doc, "ADJ")
    common_adverbs = get_common_tokens(doc, "ADV")
    common_nouns = get_common_tokens(doc, "NOUN")
    common_verbs = get_common_tokens(doc, "VERB")
    
    # Additional Features
    para_count, avg_para_words, _ = analyze_paragraphs(text)
    avg_sentiment = analyze_sentiment(text)
    fp_count, tp_count = count_pronouns(doc)
    dialogue_count = count_dialogue(text)
    cliches_found = detect_cliches(text)
    transitions_found = count_transition_words(text)
    
    analysis.update({
        "total_words": total_words,
        "unique_word_count": unique_word_count,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "total_syllables": total_syllables,
        "avg_syllables_per_word": avg_syllables_per_word,
        "complex_words": complex_words,
        "sophistication_ratio": sophistication_ratio,
        "pos_counts": dict(pos_counts),
        "adj_noun_ratio": adj_noun_ratio,
        "adv_verb_ratio": adv_verb_ratio,
        "punct_counts": punct_counts,
        "total_sentences": total_sentences,
        "avg_sentence_length": avg_sentence_length,
        "median_sentence_length": median_sentence_length,
        "sentence_length_variance": sentence_variance,
        "std_sentence_length": std_sentence_length,
        "sentence_length_cv": sentence_cv,
        "passive_sentence_count": passive_sentences,
        "passive_ratio": passive_ratio,
        "avg_dep_depth": avg_dep_depth,
        "median_dep_depth": median_dep_depth,
        "dep_depth_std": dep_depth_std,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "smog_index": smog_index,
        "common_adjectives": common_adjectives,
        "common_adverbs": common_adverbs,
        "common_nouns": common_nouns,
        "common_verbs": common_verbs,
        "paragraph_count": para_count,
        "avg_para_words": avg_para_words,
        "avg_sentiment": avg_sentiment,
        "first_person_count": fp_count,
        "third_person_count": tp_count,
        "dialogue_count": dialogue_count,
        "cliches_found": cliches_found,
        "transitions_found": transitions_found,
    })
    return analysis

def calculate_overall_grade(analysis):
    """
    Computes an overall style grade from weighted metrics.
    Returns a tuple: (grade, list_of_comments).
    """
    score = 0
    comments = []
    
    # Lexical Diversity
    ld = analysis["lexical_diversity"]
    if ld >= 0.5:
        score += 2
        comments.append("Excellent vocabulary diversity.")
    elif ld < 0.3:
        score -= 1
        comments.append("Vocabulary diversity is low; consider enriching your word choice.")
    else:
        score += 1
        comments.append("Moderate vocabulary diversity.")
    
    # Vocabulary Sophistication
    sr = analysis["sophistication_ratio"]
    if sr > 0.4:
        score += 1
        comments.append("High proportion of complex words indicates advanced vocabulary.")
    elif sr < 0.2:
        score -= 1
        comments.append("Low use of complex words; consider using a richer vocabulary.")
    else:
        score += 0.5
        comments.append("Balanced use of complex words.")
    
    # Passive Voice Usage
    pr = analysis["passive_ratio"]
    if pr > 0.5:
        score -= 2
        comments.append("Excessive passive voice usage may weaken clarity.")
    elif pr > 0.3:
        score -= 1
        comments.append("Moderate passive voice usage; try to favor active constructions.")
    else:
        score += 1
        comments.append("Limited passive voice usage; good clarity.")
    
    # Sentence Balance (CV)
    cv = analysis["sentence_length_cv"]
    if cv > 1.0:
        score -= 1
        comments.append("High sentence length variation may affect readability.")
    else:
        score += 1
        comments.append("Sentence lengths are well balanced.")
    
    # Syntactic Complexity
    if analysis["avg_dep_depth"] > 8:
        score -= 1
        comments.append("Very complex syntactic structures detected.")
    else:
        score += 1
        comments.append("Syntactic structures are clear and manageable.")
    
    # Sentiment (Neutral or Slightly Positive is Ideal)
    sent = analysis["avg_sentiment"]
    if sent < -0.2:
        score -= 1
        comments.append("The overall tone is quite negative; consider a more balanced tone.")
    elif sent > 0.5:
        score -= 0.5
        comments.append("The tone is very positive; ensure it doesn’t become saccharine.")
    else:
        score += 1
        comments.append("The tone is balanced.")
    
    # Pronoun Usage (Balanced narrative perspective)
    fp = analysis["first_person_count"]
    tp = analysis["third_person_count"]
    if fp > tp * 1.5:
        score -= 0.5
        comments.append("Excessive first-person narration detected.")
    elif tp > fp * 1.5:
        score -= 0.5
        comments.append("Excessive third-person narration detected.")
    else:
        score += 0.5
        comments.append("Balanced narrative perspective.")
    
    if score >= 6:
        overall_grade = "Excellent"
    elif score >= 3:
        overall_grade = "Good"
    elif score >= 0:
        overall_grade = "Average"
    else:
        overall_grade = "Needs Improvement"
    
    return overall_grade, comments

def generate_detailed_report(text1, text2):
    """Generates a detailed comparative report for two texts."""
    a1 = detailed_analysis(text1)
    a2 = detailed_analysis(text2)
    
    grade1, comments1 = calculate_overall_grade(a1)
    grade2, comments2 = calculate_overall_grade(a2)
    
    report_lines = []
    report_lines.append("DETAILED COMPARATIVE ANALYSIS")
    report_lines.append("=" * 60)
    
    # Report for Text 1.
    report_lines.append("\n=== TEXT 1 ===")
    report_lines.append(f"Total Words               : {a1['total_words']}")
    report_lines.append(f"Unique Words              : {a1['unique_word_count']} (Lexical Diversity: {a1['lexical_diversity']:.2f})")
    report_lines.append(f"Average Word Length       : {a1['avg_word_length']:.2f} characters")
    report_lines.append(f"Total Syllables           : {a1['total_syllables']}")
    report_lines.append(f"Avg Syllables/Word        : {a1['avg_syllables_per_word']:.2f}")
    report_lines.append(f"Complex Words (≥3 syl)    : {a1['complex_words']} (Sophistication Ratio: {a1['sophistication_ratio']:.2f})")
    report_lines.append(f"Adjective/Noun Ratio      : {a1['adj_noun_ratio']:.2f}")
    report_lines.append(f"Adverb/Verb Ratio         : {a1['adv_verb_ratio']:.2f}")
    report_lines.append(f"Sentence Count            : {a1['total_sentences']}")
    report_lines.append(f"Avg Sentence Length       : {a1['avg_sentence_length']:.2f} words")
    report_lines.append(f"Median Sentence Length    : {a1['median_sentence_length']:.2f} words")
    report_lines.append(f"Sentence Length Variance  : {a1['sentence_length_variance']:.2f}")
    report_lines.append(f"Sentence Length Std Dev   : {a1['std_sentence_length']:.2f}")
    report_lines.append(f"Sentence Length CV        : {a1['sentence_length_cv']:.2f}")
    report_lines.append(f"Passive Voice Ratio       : {a1['passive_ratio']*100:.1f}%")
    report_lines.append(f"Avg Dependency Depth      : {a1['avg_dep_depth']:.2f}")
    report_lines.append(f"Median Dependency Depth   : {a1['median_dep_depth']:.2f}")
    report_lines.append(f"Dependency Depth Std Dev  : {a1['dep_depth_std']:.2f}")
    report_lines.append(f"Flesch Reading Ease       : {a1['flesch_reading_ease']:.2f}")
    report_lines.append(f"Flesch–Kincaid Grade Level: {a1['flesch_kincaid_grade']:.2f}")
    if a1["smog_index"] is not None:
        report_lines.append(f"SMOG Index                : {a1['smog_index']:.2f}")
    else:
        report_lines.append("SMOG Index                : Not enough sentences (30+) for a reliable estimate")
    report_lines.append("Common Adjectives         : " + ", ".join(f"{w} ({c})" for w, c in a1["common_adjectives"]))
    report_lines.append("Common Adverbs            : " + ", ".join(f"{w} ({c})" for w, c in a1["common_adverbs"]))
    report_lines.append("Common Nouns              : " + ", ".join(f"{w} ({c})" for w, c in a1["common_nouns"]))
    report_lines.append("Common Verbs              : " + ", ".join(f"{w} ({c})" for w, c in a1["common_verbs"]))
    report_lines.append(f"Paragraph Count           : {a1['paragraph_count']} (Avg. {a1['avg_para_words']:.2f} words/para)")
    report_lines.append(f"Avg Sentiment Score       : {a1['avg_sentiment']:.2f}")
    report_lines.append(f"First-Person Pronouns     : {a1['first_person_count']}")
    report_lines.append(f"Third-Person Pronouns     : {a1['third_person_count']}")
    report_lines.append(f"Dialogue Lines            : {a1['dialogue_count']}")
    if a1["cliches_found"]:
        report_lines.append("Clichés Detected          : " + ", ".join(f"{ph} ({cnt})" for ph, cnt in a1["cliches_found"].items()))
    else:
        report_lines.append("Clichés Detected          : None")
    if a1["transitions_found"]:
        report_lines.append("Transition Words          : " + ", ".join(f"{w} ({c})" for w, c in a1["transitions_found"].items()))
    else:
        report_lines.append("Transition Words          : None")
    report_lines.append(f"\nOverall Style Grade (Text 1): {grade1}")
    report_lines.append("Comments:")
    for comment in comments1:
        report_lines.append("  - " + comment)
    
    # Report for Text 2.
    report_lines.append("\n=== TEXT 2 ===")
    report_lines.append(f"Total Words               : {a2['total_words']}")
    report_lines.append(f"Unique Words              : {a2['unique_word_count']} (Lexical Diversity: {a2['lexical_diversity']:.2f})")
    report_lines.append(f"Average Word Length       : {a2['avg_word_length']:.2f} characters")
    report_lines.append(f"Total Syllables           : {a2['total_syllables']}")
    report_lines.append(f"Avg Syllables/Word        : {a2['avg_syllables_per_word']:.2f}")
    report_lines.append(f"Complex Words (≥3 syl)    : {a2['complex_words']} (Sophistication Ratio: {a2['sophistication_ratio']:.2f})")
    report_lines.append(f"Adjective/Noun Ratio      : {a2['adj_noun_ratio']:.2f}")
    report_lines.append(f"Adverb/Verb Ratio         : {a2['adv_verb_ratio']:.2f}")
    report_lines.append(f"Sentence Count            : {a2['total_sentences']}")
    report_lines.append(f"Avg Sentence Length       : {a2['avg_sentence_length']:.2f} words")
    report_lines.append(f"Median Sentence Length    : {a2['median_sentence_length']:.2f} words")
    report_lines.append(f"Sentence Length Variance  : {a2['sentence_length_variance']:.2f}")
    report_lines.append(f"Sentence Length Std Dev   : {a2['std_sentence_length']:.2f}")
    report_lines.append(f"Sentence Length CV        : {a2['sentence_length_cv']:.2f}")
    report_lines.append(f"Passive Voice Ratio       : {a2['passive_ratio']*100:.1f}%")
    report_lines.append(f"Avg Dependency Depth      : {a2['avg_dep_depth']:.2f}")
    report_lines.append(f"Median Dependency Depth   : {a2['median_dep_depth']:.2f}")
    report_lines.append(f"Dependency Depth Std Dev  : {a2['dep_depth_std']:.2f}")
    report_lines.append(f"Flesch Reading Ease       : {a2['flesch_reading_ease']:.2f}")
    report_lines.append(f"Flesch–Kincaid Grade Level: {a2['flesch_kincaid_grade']:.2f}")
    if a2["smog_index"] is not None:
        report_lines.append(f"SMOG Index                : {a2['smog_index']:.2f}")
    else:
        report_lines.append("SMOG Index                : Not enough sentences (30+) for a reliable estimate")
    report_lines.append("Common Adjectives         : " + ", ".join(f"{w} ({c})" for w, c in a2["common_adjectives"]))
    report_lines.append("Common Adverbs            : " + ", ".join(f"{w} ({c})" for w, c in a2["common_adverbs"]))
    report_lines.append("Common Nouns              : " + ", ".join(f"{w} ({c})" for w, c in a2["common_nouns"]))
    report_lines.append("Common Verbs              : " + ", ".join(f"{w} ({c})" for w, c in a2["common_verbs"]))
    report_lines.append(f"Paragraph Count           : {a2['paragraph_count']} (Avg. {a2['avg_para_words']:.2f} words/para)")
    report_lines.append(f"Avg Sentiment Score       : {a2['avg_sentiment']:.2f}")
    report_lines.append(f"First-Person Pronouns     : {a2['first_person_count']}")
    report_lines.append(f"Third-Person Pronouns     : {a2['third_person_count']}")
    report_lines.append(f"Dialogue Lines            : {a2['dialogue_count']}")
    if a2["cliches_found"]:
        report_lines.append("Clichés Detected          : " + ", ".join(f"{ph} ({cnt})" for ph, cnt in a2["cliches_found"].items()))
    else:
        report_lines.append("Clichés Detected          : None")
    if a2["transitions_found"]:
        report_lines.append("Transition Words          : " + ", ".join(f"{w} ({c})" for w, c in a2["transitions_found"].items()))
    else:
        report_lines.append("Transition Words          : None")
    grade2, comments2 = calculate_overall_grade(a2)
    report_lines.append(f"\nOverall Style Grade (Text 2): {grade2}")
    report_lines.append("Comments:")
    for comment in comments2:
        report_lines.append("  - " + comment)
    
    # Comparative Summary.
    report_lines.append("\n=== COMPARATIVE SUMMARY ===")
    if a1['lexical_diversity'] > a2['lexical_diversity']:
        report_lines.append("Text 1 employs a more diverse vocabulary.")
    elif a1['lexical_diversity'] < a2['lexical_diversity']:
        report_lines.append("Text 2 employs a more diverse vocabulary.")
    else:
        report_lines.append("Both texts display similar vocabulary diversity.")
    
    if a1['avg_sentence_length'] > a2['avg_sentence_length']:
        report_lines.append("Text 1 features longer sentences on average.")
    elif a1['avg_sentence_length'] < a2['avg_sentence_length']:
        report_lines.append("Text 2 features longer sentences on average.")
    else:
        report_lines.append("Both texts have similar sentence lengths.")
    
    if a1['sophistication_ratio'] > a2['sophistication_ratio']:
        report_lines.append("Text 1 exhibits a higher proportion of complex vocabulary.")
    elif a1['sophistication_ratio'] < a2['sophistication_ratio']:
        report_lines.append("Text 2 exhibits a higher proportion of complex vocabulary.")
    else:
        report_lines.append("Both texts exhibit similar vocabulary sophistication.")
    
    if a1['passive_ratio'] > a2['passive_ratio']:
        report_lines.append("Text 1 employs passive constructions more frequently.")
    elif a1['passive_ratio'] < a2['passive_ratio']:
        report_lines.append("Text 2 employs passive constructions more frequently.")
    else:
        report_lines.append("Both texts use passive constructions at similar rates.")
    
    if a1['avg_dep_depth'] > a2['avg_dep_depth']:
        report_lines.append("Text 1 has higher syntactic complexity (dependency tree depth).")
    elif a1['avg_dep_depth'] < a2['avg_dep_depth']:
        report_lines.append("Text 2 has higher syntactic complexity (dependency tree depth).")
    else:
        report_lines.append("Both texts exhibit similar syntactic complexity.")
    
    return "\n".join(report_lines)

def generate_single_text_report(text):
    """
    Generates a detailed analysis report for a single text.
    Provides an in-depth breakdown of all metrics along with overall style grade and commentary.
    """
    a = detailed_analysis(text)
    grade, comments = calculate_overall_grade(a)
    
    report_lines = []
    report_lines.append("DETAILED SINGLE TEXT ANALYSIS")
    report_lines.append("=" * 60)
    
    # Lexical & Syntactic Overview
    report_lines.append("\n--- Lexical & Syntactic Metrics ---")
    report_lines.append(f"Total Words               : {a['total_words']}")
    report_lines.append(f"Unique Words              : {a['unique_word_count']} (Lexical Diversity: {a['lexical_diversity']:.2f})")
    report_lines.append(f"Avg. Word Length          : {a['avg_word_length']:.2f} characters")
    report_lines.append(f"Total Syllables           : {a['total_syllables']}")
    report_lines.append(f"Avg. Syllables/Word       : {a['avg_syllables_per_word']:.2f}")
    report_lines.append(f"Complex Words (≥3 syl)    : {a['complex_words']} (Sophistication Ratio: {a['sophistication_ratio']:.2f})")
    report_lines.append(f"Adjective/Noun Ratio      : {a['adj_noun_ratio']:.2f}")
    report_lines.append(f"Adverb/Verb Ratio         : {a['adv_verb_ratio']:.2f}")
    
    # Sentence & Readability Metrics
    report_lines.append("\n--- Sentence & Readability Metrics ---")
    report_lines.append(f"Sentence Count            : {a['total_sentences']}")
    report_lines.append(f"Avg Sentence Length       : {a['avg_sentence_length']:.2f} words")
    report_lines.append(f"Median Sentence Length    : {a['median_sentence_length']:.2f} words")
    report_lines.append(f"Sentence Length Variance  : {a['sentence_length_variance']:.2f}")
    report_lines.append(f"Passive Voice Ratio       : {a['passive_ratio']*100:.1f}%")
    report_lines.append(f"Avg Dependency Depth      : {a['avg_dep_depth']:.2f}")
    report_lines.append(f"Flesch Reading Ease       : {a['flesch_reading_ease']:.2f}")
    report_lines.append(f"Flesch–Kincaid Grade Level: {a['flesch_kincaid_grade']:.2f}")
    if a["smog_index"] is not None:
        report_lines.append(f"SMOG Index                : {a['smog_index']:.2f}")
    else:
        report_lines.append("SMOG Index                : Not enough sentences (30+) for a reliable estimate")
    
    # Additional Features
    report_lines.append("\n--- Additional Features ---")
    report_lines.append(f"Paragraph Count           : {a['paragraph_count']} (Avg. {a['avg_para_words']:.2f} words/para)")
    report_lines.append(f"Avg Sentiment Score       : {a['avg_sentiment']:.2f}")
    report_lines.append(f"First-Person Pronouns     : {a['first_person_count']}")
    report_lines.append(f"Third-Person Pronouns     : {a['third_person_count']}")
    report_lines.append(f"Dialogue Lines            : {a['dialogue_count']}")
    if a["cliches_found"]:
        report_lines.append("Clichés Detected          : " + ", ".join(f"{ph} ({cnt})" for ph, cnt in a["cliches_found"].items()))
    else:
        report_lines.append("Clichés Detected          : None")
    if a["transitions_found"]:
        report_lines.append("Transition Words          : " + ", ".join(f"{w} ({c})" for w, c in a["transitions_found"].items()))
    else:
        report_lines.append("Transition Words          : None")
    
    # Common Tokens Summary
    report_lines.append("\n--- Common Tokens ---")
    report_lines.append("Adjectives: " + ", ".join(f"{w} ({c})" for w, c in a["common_adjectives"]))
    report_lines.append("Adverbs   : " + ", ".join(f"{w} ({c})" for w, c in a["common_adverbs"]))
    report_lines.append("Nouns     : " + ", ".join(f"{w} ({c})" for w, c in a["common_nouns"]))
    report_lines.append("Verbs     : " + ", ".join(f"{w} ({c})" for w, c in a["common_verbs"]))
    
    # Overall Grade & Comments
    report_lines.append("\n--- Overall Style Grade ---")
    report_lines.append(f"Overall Grade             : {grade}")
    report_lines.append("Comments:")
    for comment in comments:
        report_lines.append("  - " + comment)
    
    return "\n".join(report_lines)


class ModernProseAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Detailed Prose Analyzer")
        self.geometry("1100x800")
        self.resizable(False, False)
        
        # Create a tab view for switching modes.
        self.tabview = ctk.CTkTabview(self, width=1080, height=700)
        self.tabview.pack(pady=20, padx=10)
        self.tabview.add("Comparative Analysis")
        self.tabview.add("Single Text Analysis")
        
        # Build Comparative Analysis tab.
        self.build_comparative_tab()
        # Build Single Text Analysis tab.
        self.build_single_tab()
        
        # Bottom frame for common controls.
        self.bottom_frame = ctk.CTkFrame(self, width=1080, height=50)
        self.bottom_frame.pack(pady=(0, 10))
        self.save_button = ctk.CTkButton(self.bottom_frame, text="Save Report", command=self.save_report)
        self.save_button.pack(side="left", padx=10)
        self.copy_button = ctk.CTkButton(self.bottom_frame, text="Copy Report", command=self.copy_report)
        self.copy_button.pack(side="left", padx=10)
        self.reset_button = ctk.CTkButton(self.bottom_frame, text="Reset", command=self.reset_fields)
        self.reset_button.pack(side="left", padx=10)
        # Appearance mode switcher.
        self.mode_switch = ctk.CTkButton(self.bottom_frame, text="Toggle Dark/Light", command=self.toggle_mode)
        self.mode_switch.pack(side="right", padx=10)
    
    def build_comparative_tab(self):
        tab = self.tabview.tab("Comparative Analysis")
        # Create frames for the two input texts.
        self.comp_frame = ctk.CTkFrame(tab)
        self.comp_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.text1_label = ctk.CTkLabel(self.comp_frame, text="Enter Text 1:")
        self.text1_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.text1_box = ctk.CTkTextbox(self.comp_frame, width=500, height=250)
        self.text1_box.grid(row=1, column=0, padx=10, pady=5)
        
        self.text2_label = ctk.CTkLabel(self.comp_frame, text="Enter Text 2:")
        self.text2_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.text2_box = ctk.CTkTextbox(self.comp_frame, width=500, height=250)
        self.text2_box.grid(row=1, column=1, padx=10, pady=5)
        
        self.compare_button = ctk.CTkButton(self.comp_frame, text="Compare Analysis", command=self.compare_analysis)
        self.compare_button.grid(row=2, column=0, columnspan=2, pady=10)
    
    def build_single_tab(self):
        tab = self.tabview.tab("Single Text Analysis")
        self.single_frame = ctk.CTkFrame(tab)
        self.single_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.single_label = ctk.CTkLabel(self.single_frame, text="Enter Text for Analysis:")
        self.single_label.pack(pady=5)
        self.single_box = ctk.CTkTextbox(self.single_frame, width=1020, height=400)
        self.single_box.pack(padx=10, pady=5)
        
        self.single_analyze_button = ctk.CTkButton(self.single_frame, text="Analyze Text", command=self.single_analysis)
        self.single_analyze_button.pack(pady=10)
    
    def compare_analysis(self):
        text1 = self.text1_box.get("0.0", "end").strip()
        text2 = self.text2_box.get("0.0", "end").strip()
        if not text1 or not text2:
            ctk.CTkMessagebox(title="Input Needed", message="Please enter text in both boxes for comparative analysis.")
            return
        report = generate_detailed_report(text1, text2)
        self.show_report(report)
    
    def single_analysis(self):
        text = self.single_box.get("0.0", "end").strip()
        if not text:
            ctk.CTkMessagebox(title="Input Needed", message="Please enter text for analysis.")
            return
        report = generate_single_text_report(text)
        self.show_report(report)
    
    def show_report(self, report):
        # Create a new window for the report.
        self.report_window = ctk.CTkToplevel(self)
        self.report_window.title("Analysis Report")
        self.report_window.geometry("1000x600")
        self.report_text = ctk.CTkTextbox(self.report_window, width=980, height=580)
        self.report_text.pack(pady=10, padx=10)
        self.report_text.insert("0.0", report)
    
    def save_report(self):
        # Save the report from the current report window if open; else show error.
        try:
            report = self.report_text.get("0.0", "end").strip()
        except Exception:
            ctk.CTkMessagebox(title="No Report", message="Run an analysis first to generate a report.")
            return
        file_path = ctk.filedialog.asksaveasfilename(defaultextension=".txt",
                                                     filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                                                     title="Save Report As")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            ctk.CTkMessagebox(title="Report Saved", message=f"Report saved to:\n{file_path}")
    
    def copy_report(self):
        try:
            report = self.report_text.get("0.0", "end").strip()
        except Exception:
            ctk.CTkMessagebox(title="No Report", message="Run an analysis first to generate a report.")
            return
        pyperclip.copy(report)
        ctk.CTkMessagebox(title="Copied", message="Report copied to clipboard.")
    
    def reset_fields(self):
        self.text1_box.delete("0.0", "end")
        self.text2_box.delete("0.0", "end")
        self.single_box.delete("0.0", "end")
        try:
            self.report_text.delete("0.0", "end")
        except Exception:
            pass
    
    def toggle_mode(self):
        current = ctk.get_appearance_mode()
        new_mode = "Dark" if current == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)

if __name__ == "__main__":
    # Load spaCy model and VADER once.
    nlp = spacy.load("en_core_web_sm")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    app = ModernProseAnalyzerApp()
    app.mainloop()
