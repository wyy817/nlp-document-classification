"""
Document Classification System (DCS)
DTS305TC Natural Language Processing - Coursework 2
Implementation of document classification using Machine Learning and Deep Learning algorithms
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TextCNN(nn.Module):
    """1D CNN model for document classification"""
    
    def __init__(self, vocab_size, embedding_dim, num_classes, max_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        x = self.conv1(x)  # (batch_size, 128, seq_length)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, 128, 1)
        x = x.squeeze(2)  # (batch_size, 128)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TextDataset(Dataset):
    """Data preparation for 1D CNN training"""
    
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class DocumentClassificationSystem:
    """Main Document Classification System"""
    
    def __init__(self):
        """Initialize the system"""
        self.documents = []
        self.labels = []
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model1 = None  # Naive Bayes + TF-IDF
        self.model2 = None  # 1D CNN
        self.tfidf_vectorizer = None
        self.vocab = None 
        self.word2idx = None
        self.max_length = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_mapping = {}
        self.reverse_label_mapping = {}
    
    def main(self):
        """
        (1) Main function: control the startup and flow of the entire document classification system
        """
        print("="*80)
        print("Document Classification System")
        print("DTS305TC Natural Language Processing - Coursework 2")
        print("="*80)
        
        # Step 1: Load dataset
        print("\nStep 1: Loading dataset...")
        self.database_input()
        
        # Step 2: Remove duplicates
        print("\nStep 2: Checking for duplicates...")
        self.remove_duplicates()
        
        # Step 3: Preprocess text
        print("\nStep 3: Preprocessing text...")
        self.text_preprocessing()
        
        # Step 4: Train models
        print("\nStep 4: Training classification models...")
        print("  - Training Algorithm 1: Naive Bayes + TF-IDF ...")
        self.classification_algorithm_1()
        print("  - Training Algorithm 2: 1D CNN ...")
        self.classification_algorithm_2()
        
        # Step 5: Evaluate models
        print("\nStep 5: Evaluating models...")
        self.classification_performance()
        
        # Step 6: Test with predefined examples
        print("\nStep 6: Testing with predefined examples...")
        self.test_with_examples()

        # Step 7: User interaction
        print("\nStep 7: Ready for user input classification...")
        self.user_input()
        
    def user_input(self):
        """
        (2) User input function: allow continuous user input of text for classification via the console
        """
        print("\n" + "="*80)
        print("Interactive Classification Mode")
        print("="*80)
        print("Enter text to classify (or type 'quit' to exit)")
        print("-"*80)
        
        while True:
            user_text = input("\nEnter text: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("\nExiting system. Goodbye!")
                break
                
            if not user_text:
                print("Please enter valid text.")
                continue
                
            # Classify user input
            self.output_classification(user_text)
    
    def database_input(self):
        """
        (3) Database input function: read a local text library from a document folder
        """
        # Check if dataset folder exists
        dataset_folder = "Dataset/Dataset"
        
        if not os.path.exists(dataset_folder):
            print(f"Dataset folder '{dataset_folder}' not found. Creating sample dataset...")
            self._create_sample_dataset(dataset_folder)
        
        # Read documents from folder
        category_count = defaultdict(int)
        
        for category in os.listdir(dataset_folder):
            category_path = os.path.join(dataset_folder, category)
            
            if os.path.isdir(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(category_path, filename)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                self.documents.append(content)
                                self.labels.append(category)
                                category_count[category] += 1
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} documents from {len(category_count)} categories:")
        for category, count in category_count.items():
            print(f"  - {category}: {count} documents")
            
        # Create label mapping
        unique_labels = sorted(list(set(self.labels)))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        return self.documents, self.labels
    
    def remove_duplicates(self):
        """
        Duplicate detection: check and remove duplicate documents from the dataset
        """
        original_count = len(self.documents)
        
        # Create a dictionary to track unique documents
        seen = {}
        unique_documents = []
        unique_labels = []
        duplicates_info = []
        
        for idx, (doc, label) in enumerate(zip(self.documents, self.labels)):
            # Preprocess document for comparison
            processed_doc = self._preprocess_text(doc)
            
            # Check if there are any duplicates
            if processed_doc in seen:
                duplicates_info.append({
                    'duplicate_index': idx,
                    'duplicate_label': label,
                    'original_index': seen[processed_doc]['index'],
                    'original_label': seen[processed_doc]['label']
                })
            else:
                # New unique document
                seen[processed_doc] = {'index': idx, 'label': label}
                unique_documents.append(doc)
                unique_labels.append(label)
        
        # Update documents and labels
        self.documents = unique_documents
        self.labels = unique_labels
        
        duplicates_count = original_count - len(self.documents)
        
        if duplicates_count > 0:
            print(f"  Found {duplicates_count} duplicate document(s)")
            print(f"  Original: {original_count} documents")
            print(f"  After removing duplicates: {len(self.documents)} documents")
            
            # Show details of duplicates
            if duplicates_info:
                print("\n  Duplicate details:")
                for i, dup in enumerate(duplicates_info[:3]): 
                    print(f"    - Document #{dup['duplicate_index']} (label: {dup['duplicate_label']}) "
                          f"is duplicate of #{dup['original_index']} (label: {dup['original_label']})")
                if len(duplicates_info) > 3:
                    print(f"    ... and {len(duplicates_info) - 3} more duplicates")
        else:
            print(f" No duplicates found. All {original_count} documents are unique.")
        
        return self.documents, self.labels
    
    def text_preprocessing(self):
        """
        (4) Text preprocessing function: preprocess the read documents 
        and use 80% as training samples and 20% as validation samples
        """
        # Convert labels to numeric
        numeric_labels = [self.label_mapping[label] for label in self.labels]
        
        # Split dataset: 80% training, 20% validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.documents, 
            numeric_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=numeric_labels
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def _preprocess_text(self, text):
        """Helper function to preprocess a single text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def classification_algorithm_1(self):
        """
        (5) Classification algorithm 1: train Model 1 on the training samples using 
        Naive Bayes with TF-IDF features
        """
        # Preprocess training data
        X_train_processed = [self._preprocess_text(doc) for doc in self.X_train]
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_processed)
        
        # Train Naive Bayes classifier
        self.model1 = MultinomialNB(alpha=1.0)
        self.model1.fit(X_train_tfidf, self.y_train)
        
        print(f"  Model 1 trained: Naive Bayes with {X_train_tfidf.shape[1]} TF-IDF features")
        
        return self.model1
    
    def classification_algorithm_2(self):
        """
        (6) Classification algorithm 2: train Model 2 on the training samples using 
        1D CNN for text classification
        """
        # Preprocess training data
        X_train_processed = [self._preprocess_text(doc) for doc in self.X_train]
        
        # Build vocabulary
        max_words = 5000
        all_words = []
        for text in X_train_processed:
            all_words.extend(text.split())
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(max_words - 2)  # Reserve 2 for special tokens
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common):
            self.word2idx[word] = idx + 2
        
        self.vocab = self.word2idx
        vocab_size = len(self.word2idx)
        
        # Create dataset and dataloader
        train_dataset = TextDataset(X_train_processed, self.y_train, self.word2idx, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Build model
        num_classes = len(self.label_mapping)
        embedding_dim = 128
        
        self.model2 = TextCNN(vocab_size, embedding_dim, num_classes, self.max_length)
        self.model2 = self.model2.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model2.parameters(), lr=0.001)
        
        # Train model
        num_epochs = 20
        self.model2.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for texts, labels in train_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model2(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        print(f"  Model 2 trained: 1D CNN with {vocab_size} vocabulary size")
        
        return self.model2
    
    def classification_performance(self):
        """
        (7) Classification algorithm performance: output the model metrics of Model 1 and 
        Model 2 on the validation samples
        """
        # Preprocess validation data
        X_val_processed = [self._preprocess_text(doc) for doc in self.X_val]
        
        # Model 1 predictions (Naive Bayes + TF-IDF)
        X_val_tfidf = self.tfidf_vectorizer.transform(X_val_processed)
        y_pred_model1 = self.model1.predict(X_val_tfidf)
        
        # Model 2 predictions (1D CNN)
        self.model2.eval()
        val_dataset = TextDataset(X_val_processed, self.y_val, self.word2idx, self.max_length)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        y_pred_model2 = []
        with torch.no_grad():
            for texts, _ in val_loader:
                texts = texts.to(self.device)
                outputs = self.model2(texts)
                _, predicted = torch.max(outputs, 1)
                y_pred_model2.extend(predicted.cpu().numpy())
        
        y_pred_model2 = np.array(y_pred_model2)
        
        # Calculate metrics
        metrics_data = []
        
        # Model 1 metrics
        precision_m1 = precision_score(self.y_val, y_pred_model1, average='weighted')
        recall_m1 = recall_score(self.y_val, y_pred_model1, average='weighted')
        f1_m1 = f1_score(self.y_val, y_pred_model1, average='weighted')
        
        metrics_data.append({
            'Model': 'Naive Bayes + TF-IDF',
            'Precision': precision_m1,
            'Recall': recall_m1,
            'F1-Score': f1_m1
        })
        
        # Model 2 metrics
        precision_m2 = precision_score(self.y_val, y_pred_model2, average='weighted')
        recall_m2 = recall_score(self.y_val, y_pred_model2, average='weighted')
        f1_m2 = f1_score(self.y_val, y_pred_model2, average='weighted')
        
        metrics_data.append({
            'Model': '1D CNN',
            'Precision': precision_m2,
            'Recall': recall_m2,
            'F1-Score': f1_m2
        })
        
        # Display results
        print("\n" + "="*80)
        print("Model Performance on Validation Set")
        print("="*80)
        
        metrics_df = pd.DataFrame(metrics_data)
        print(metrics_df.to_string(index=False))
        
        # Save metrics to CSV files
        self._save_metrics_to_csv(y_pred_model1, y_pred_model2)
        
        return metrics_data
    
    def test_with_examples(self):
        """
        Test the system with 10 new text examples with predefined labels
        """
        print("\n" + "="*80)
        print("Testing with 10 New Text Examples")
        print("="*80)
        
        # 10 test examples (5 sport, 5 tech), each with at least 2 sentences (requirements in email)
        test_examples = [
            {
                'text': "The championship game was an exciting finale to the season. The home team secured victory with a last-minute goal that thrilled the crowd.",
                'expected_label': 'sport'
            },
            {
                'text': "Artificial intelligence is transforming healthcare through innovative diagnostic tools. Machine learning algorithms can now detect diseases from medical images with remarkable accuracy.",
                'expected_label': 'tech'
            },
            {
                'text': "The Olympic athlete broke the world record in swimming. Her performance demonstrated years of dedicated training and exceptional technique.",
                'expected_label': 'sport'
            },
            {
                'text': "The latest smartphone features cutting-edge camera technology. Users can capture professional-quality photos with advanced computational photography.",
                'expected_label': 'tech'
            },
            {
                'text': "The basketball tournament featured intense competition between rival teams. Fans packed the stadium to witness the thrilling playoff matches.",
                'expected_label': 'sport'
            },
            {
                'text': "Quantum computing represents a breakthrough in processing power. Researchers are developing new algorithms that could revolutionize cryptography and drug discovery.",
                'expected_label': 'tech'
            },
            {
                'text': "The tennis player dominated the grand slam tournament with flawless serves. Her aggressive playing style overwhelmed opponents throughout the competition.",
                'expected_label': 'sport'
            },
            {
                'text': "Cloud computing platforms are enabling businesses to scale operations efficiently. Companies can now deploy applications globally with minimal infrastructure investment.",
                'expected_label': 'tech'
            },
            {
                'text': "The football league announced new player safety protocols. These regulations aim to reduce injuries and protect athletes' long-term health.",
                'expected_label': 'sport'
            },
            {
                'text': "Autonomous vehicles are being tested in urban environments. Advanced sensors and neural networks allow these cars to navigate complex traffic situations safely.",
                'expected_label': 'tech'
            }
        ]
        
        # Store results for analysis
        results = []
        
        for i, example in enumerate(test_examples, 1):
            print(f"\n--- Test Example {i} ---")
            print(f"Text: {example['text']}")
            print(f"Expected Label: {example['expected_label']}")
            
            # Classify the text
            processed_text = self._preprocess_text(example['text'])
            
            # Model 1 Naive Bayes + TF-IDF prediction
            text_tfidf = self.tfidf_vectorizer.transform([processed_text])
            pred1 = self.model1.predict(text_tfidf)[0]
            pred1_label = self.reverse_label_mapping[pred1]
            
            # Model 2 1D CNN prediction
            self.model2.eval()
            words = processed_text.split()
            indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
            
            if len(indices) < self.max_length:
                indices = indices + [self.word2idx['<PAD>']] * (self.max_length - len(indices))
            else:
                indices = indices[:self.max_length]
            
            text_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                output = self.model2(text_tensor)
                pred2 = torch.argmax(output, dim=1).item()
                pred2_label = self.reverse_label_mapping[pred2]
            
            print(f"Model 1 Prediction: {pred1_label}")
            print(f"Model 2 Prediction: {pred2_label}")
            
            results.append({
                'example_id': i,
                'text': example['text'],
                'expected': example['expected_label'],
                'model1_pred': pred1_label,
                'model2_pred': pred2_label
            })
        
        # Calculate metrics on test examples
        # Convert expected labels to numeric
        y_true = [self.label_mapping.get(ex['expected_label'], -1) for ex in test_examples]
        
        # Get predictions
        y_pred_m1 = [self.label_mapping[r['model1_pred']] for r in results]
        y_pred_m2 = [self.label_mapping[r['model2_pred']] for r in results]
        
        # Calculate metrics
        print("\n" + "="*80)
        print("Performance Metrics on Test Examples")
        print("="*80)
        
        # Model 1 metrics
        precision_m1 = precision_score(y_true, y_pred_m1, average='weighted', zero_division=0)
        recall_m1 = recall_score(y_true, y_pred_m1, average='weighted', zero_division=0)
        f1_m1 = f1_score(y_true, y_pred_m1, average='weighted', zero_division=0)
        
        # Model 2 metrics
        precision_m2 = precision_score(y_true, y_pred_m2, average='weighted', zero_division=0)
        recall_m2 = recall_score(y_true, y_pred_m2, average='weighted', zero_division=0)
        f1_m2 = f1_score(y_true, y_pred_m2, average='weighted', zero_division=0)
        
        # Display metrics
        test_metrics_data = [
            {
                'Model': 'Naive Bayes + TF-IDF',
                'Precision': precision_m1,
                'Recall': recall_m1,
                'F1-Score': f1_m1
            },
            {
                'Model': '1D CNN',
                'Precision': precision_m2,
                'Recall': recall_m2,
                'F1-Score': f1_m2
            }
        ]
        
        test_metrics_df = pd.DataFrame(test_metrics_data)
        print(test_metrics_df.to_string(index=False))
        
        # Save test metrics
        test_metrics_df.to_csv('test_examples_metrics.csv', index=False)
        print("\nTest metrics saved to: test_examples_metrics.csv")

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('test_examples_results.csv', index=False)
        print("\n" + "="*80)
        print("Test results saved to: test_examples_results.csv")
        print("="*80)
        
        return results

    def _save_metrics_to_csv(self, y_pred_model1, y_pred_model2):
        """Save detailed metrics to CSV files"""
        
        # Precision
        precision_m1 = precision_score(self.y_val, y_pred_model1, average='weighted')
        precision_m2 = precision_score(self.y_val, y_pred_model2, average='weighted')
        
        precision_df = pd.DataFrame({
            'Algorithm': ['Naive Bayes + TF-IDF', '1D CNN'],
            'Precision': [precision_m1, precision_m2]
        })
        precision_df.to_csv('precision.csv', index=False)
        
        # Recall
        recall_m1 = recall_score(self.y_val, y_pred_model1, average='weighted')
        recall_m2 = recall_score(self.y_val, y_pred_model2, average='weighted')
        
        recall_df = pd.DataFrame({
            'Algorithm': ['Naive Bayes + TF-IDF', '1D CNN'],
            'Recall': [recall_m1, recall_m2]
        })
        recall_df.to_csv('recall.csv', index=False)
        
        # F1
        f1_m1 = f1_score(self.y_val, y_pred_model1, average='weighted')
        f1_m2 = f1_score(self.y_val, y_pred_model2, average='weighted')
        
        f1_df = pd.DataFrame({
            'Algorithm': ['Naive Bayes + TF-IDF', '1D CNN'],
            'F1': [f1_m1, f1_m2]
        })
        f1_df.to_csv('F1.csv', index=False)
        
        print("\nMetrics saved to: precision.csv, recall.csv, F1.csv")
    
    def output_classification(self, text):
        """
        (8) Output function: output the classification results of Model 1 and Model 2 for user input
        """
        # Preprocess input text
        processed_text = self._preprocess_text(text)
        
        # Model 1 prediction (Naive Bayes)
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        pred1 = self.model1.predict(text_tfidf)[0]
        pred1_proba = self.model1.predict_proba(text_tfidf)[0]
        
        # Model 2 prediction (1D CNN)
        self.model2.eval()
        
        # Convert text to indices
        words = processed_text.split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        text_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model2(text_tensor)
            pred2_proba = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred2 = np.argmax(pred2_proba)
        
        # Display results
        print("\n" + "-"*80)
        print("Classification Results:")
        print("-"*80)
        print(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print()
        print(f"Algorithm 1 (Naive Bayes + TF-IDF):")
        print(f"  Predicted Category: {self.reverse_label_mapping[pred1]}")
        print(f"  Confidence: {pred1_proba[pred1]:.4f}")
        print()
        print(f"Algorithm 2 (1D CNN):")
        print(f"  Predicted Category: {self.reverse_label_mapping[pred2]}")
        print(f"  Confidence: {pred2_proba[pred2]:.4f}")
        print("-"*80)


if __name__ == "__main__":
    # Create and run the document classification system
    dcs = DocumentClassificationSystem()
    dcs.main()