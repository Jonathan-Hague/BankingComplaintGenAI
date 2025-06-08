import nltk
from nltk.corpus import stopwords
import itertools
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd
from src.config.config import TFIDF_PARAMS, CUSTOM_STOP_WORDS, NUM_THEMES

class ThemeExtractor:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = self._prepare_stop_words()
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            **TFIDF_PARAMS
        )
        self.nmf_model = NMF(n_components=NUM_THEMES, random_state=42)

    def _prepare_stop_words(self):
        """Prepare stop words list."""
        nltk_stop_words = stopwords.words('english')
        nltk_stop_words.extend(CUSTOM_STOP_WORDS)
        
        # Add two and three letter combinations
        three_letter_words = [''.join(comb) for comb in itertools.product(string.ascii_lowercase, repeat=3)]
        two_letter_words = [''.join(comb) for comb in itertools.product(string.ascii_lowercase, repeat=2)]
        
        nltk_stop_words.extend(three_letter_words)
        nltk_stop_words.extend(two_letter_words)
        
        return nltk_stop_words

    def extract_themes(self, documents):
        """Extract themes using TF-IDF and NMF."""
        # Compute TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Apply NMF
        nmf_features = self.nmf_model.fit_transform(tfidf_matrix)
        
        # Get top words for each theme
        top_words = self._get_top_words(10)
        
        # Assign documents to dominant themes
        dominant_themes = nmf_features.argmax(axis=1)
        
        return {
            'top_words': top_words,
            'dominant_themes': dominant_themes,
            'theme_docs': self._group_docs_by_theme(documents, dominant_themes)
        }

    def _get_top_words(self, n_top_words):
        """Get top words for each theme."""
        top_words = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_words[f"Theme #{topic_idx+1}"] = [
                feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]
            ]
        
        return pd.DataFrame(top_words)

    def _group_docs_by_theme(self, documents, dominant_themes):
        """Group documents by their dominant theme."""
        theme_docs = {f"Theme #{i+1}": [] for i in range(NUM_THEMES)}
        
        for doc, theme in zip(documents, dominant_themes):
            theme_docs[f"Theme #{theme+1}"].append(doc)
        
        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in theme_docs.items()])) 