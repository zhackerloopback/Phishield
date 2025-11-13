# data_preprocessing.py
from urllib.parse import urlparse
import re

class PhishingDataProcessor:
    def __init__(self):
        # Define the feature order that matches your trained model
        self.features = [
            'has_https', 'len_hostname', 'len_path', 
            'count_at', 'count_hyphen', 'count_digits'
        ]
    
    def extract_features(self, url):
        """Extract features from URL"""
        try:
            # Ensure URL has scheme for parsing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urlparse(url)
            
            features = {
                'has_https': 1 if url.startswith('https://') else 0,
                'len_hostname': len(parsed.hostname or ''),
                'len_path': len(parsed.path or ''),
                'count_at': url.count('@'),
                'count_hyphen': (parsed.hostname or '').count('-'),
                'count_digits': sum(c.isdigit() for c in url)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            return None
    
    def prepare_features(self, url):
        """Extract features and return as vector in correct order"""
        features = self.extract_features(url)
        if features is None:
            return None
        
        return [features[feature] for feature in self.features]
