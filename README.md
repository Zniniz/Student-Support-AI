# Student Support AI

## Overview
An intelligent conversational agent built to assist students by providing accurate, context-aware answers to queries. The AI combines vector-based semantic search with real-time sentiment analysis, allowing it to dynamically evaluate user frustration and automatically escalate complex or emotionally charged issues to a human advisor.

## Key Features
* **Semantic Search Pipeline:** Embeds knowledge base questions using the `all-MiniLM-L6-v2` SentenceTransformer and retrieves answers using cosine similarity matching.
* **Real-Time Sentiment Analysis:** Integrates a pre-trained RoBERTa model to continuously analyze user prompt sentiment (Positive, Neutral, Negative).
* **Automated Human Escalation:** Features smart intervention logic that automatically flags conversations for human review if the negative sentiment confidence exceeds a strict 90% threshold.
* **Stateful Session Analytics:** Tracks the complete conversation lifecycle, generating a detailed summary of questions, sentiment scores, and escalation states upon exit.

## Tech Stack
* Python
* Hugging Face Transformers
* Sentence-Transformers
* Pandas & NumPy
* Scikit-Learn (Cosine Similarity)

## Installation & How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Zniniz/Student-Support-AI.git](https://github.com/Zniniz/Student-Support-AI.git)
   cd student-support-ai
   ```
2. Install the required dependencies:
   ```
   pip install pandas numpy sentence-transformers transformers scikit-learn
   ```
3. Ensure your custom `knowledge_base.csv` is populated with `question` and `answer` columns and located in the root directory.
4. Run the application
   ```
   python main.py
   ```
