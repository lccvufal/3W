"""
Binary Classification Framework for 3W Dataset

This module contains PyTorch-based binary classifiers and utility functions
for implementing one-vs-all classification strategy on time series data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter


def convert_windowed_dfs_to_arrays(dfs, classes):
    """
    Convert already windowed DataFrames to numpy arrays suitable for neural networks
    
    Parameters:
    dfs (list): List of windowed DataFrames (already processed)
    classes (list): Corresponding class labels
    
    Returns:
    X (np.array): Feature arrays [n_samples, n_timesteps, n_features]
    y (np.array): Class labels
    """
    if not dfs or len(dfs) == 0:
        print("❌ No windowed DataFrames provided")
        return np.array([]), np.array([])
    
    sequences = []
    labels = []
    
    print(f"Converting {len(dfs)} windowed sequences to arrays...")
    
    for i, (df, class_label) in enumerate(zip(dfs, classes)):
        try:
            # Remove class column if present and get numeric features
            feature_cols = [col for col in df.columns if 'class' not in col.lower()]
            df_features = df[feature_cols].select_dtypes(include=[np.number])
            
            if len(df_features.columns) == 0:
                print(f"Warning: No numeric features in window {i}")
                continue
            
            # Convert to numpy array
            sequence_array = df_features.values
            
            if len(sequence_array) > 0:
                sequences.append(sequence_array)
                labels.append(class_label)
        
        except Exception as e:
            print(f"Error converting window {i}: {e}")
            continue
    
    if sequences:
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"✅ Successfully converted to arrays:")
        print(f"   • Shape: {X.shape} (samples, timesteps, features)")
        print(f"   • Features: {X.shape[2]} per timestep")
        print(f"   • Sequence length: {X.shape[1]} timesteps")
        
        return X, y
    else:
        print("❌ No sequences could be converted")
        return np.array([]), np.array([])


class BinaryLSTMClassifier(nn.Module):
    """
    PyTorch LSTM-based binary classifier architecture adapted from OTC notebook
    """
    def __init__(self, input_size, lstm_units=128, dense_units=64, dropout_rate=0.3, num_layers=2):
        super(BinaryLSTMClassifier, self).__init__()
        
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.bn1 = nn.BatchNorm1d(lstm_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(lstm_units // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units // 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.dropout4 = nn.Dropout(dropout_rate / 2)
        
        # Output layer
        self.fc_out = nn.Linear(dense_units // 2, 1)
        
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm1(x)
        # Apply batch norm to last timestep
        lstm_out_last = lstm_out[:, -1, :]  # Take last timestep
        lstm_out_last = self.bn1(lstm_out_last)
        lstm_out_last = self.dropout1(lstm_out_last)
        
        # Expand for second LSTM
        lstm_out_expanded = lstm_out_last.unsqueeze(1)  # Add timestep dimension back
        lstm_out2, _ = self.lstm2(lstm_out_expanded)
        lstm_out2 = lstm_out2[:, -1, :]  # Take last timestep
        lstm_out2 = self.bn2(lstm_out2)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Dense layers
        x = F.relu(self.fc1(lstm_out2))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        
        # Output layer with sigmoid
        output = torch.sigmoid(self.fc_out(x))
        
        return output


def create_binary_lstm_classifier(input_shape, lstm_units=128, dense_units=64, dropout_rate=0.3, device='cpu'):
    """
    Create PyTorch LSTM-based binary classifier
    
    Parameters:
    input_shape (tuple): Shape of input sequences (timesteps, features)
    lstm_units (int): Number of LSTM units
    dense_units (int): Number of dense layer units
    dropout_rate (float): Dropout rate for regularization
    device (str): Device to run the model on ('cpu' or 'cuda')
    
    Returns:
    model: PyTorch model for binary classification
    """
    input_size = input_shape[1]  # Number of features
    
    model = BinaryLSTMClassifier(
        input_size=input_size,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    return model


def prepare_binary_data(X, y, target_class, balance_classes=True, samples_per_class=None):
    """
    Prepare data for binary classification (one vs all)
    
    Parameters:
    X (np.array): Feature sequences
    y (np.array): Class labels  
    target_class (int): Class to distinguish from all others
    balance_classes (bool): Whether to balance positive/negative classes
    samples_per_class (int): Specific number of samples per class (if None, uses all available)
    
    Returns:
    X_binary (np.array): Same feature sequences
    y_binary (np.array): Binary labels (1 for target class, 0 for others)
    indices (np.array): Indices of selected samples
    """
    # Create binary labels
    y_binary = (y == target_class).astype(int)
    
    if balance_classes:
        # Balance classes using undersampling and oversampling to meet exact requirements
        pos_indices = np.where(y_binary == 1)[0]
        neg_indices = np.where(y_binary == 0)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            if samples_per_class is not None:
                # Use specific number of samples per class with under/oversampling
                target_samples = samples_per_class
                
                print(f"   Requested: {target_samples} per class")
                print(f"   Available: {len(pos_indices)} positive, {len(neg_indices)} negative")
                
                # Randomly sample indices with replacement if needed
                np.random.seed(42)
                
                # Handle positive samples
                if len(pos_indices) >= target_samples:
                    # Undersample: we have enough, randomly select without replacement
                    selected_pos = np.random.choice(pos_indices, target_samples, replace=False)
                    pos_strategy = "undersampling"
                else:
                    # Oversample: we need more, randomly select with replacement
                    selected_pos = np.random.choice(pos_indices, target_samples, replace=True)
                    pos_strategy = "oversampling"
                
                # Handle negative samples
                if len(neg_indices) >= target_samples:
                    # Undersample: we have enough, randomly select without replacement
                    selected_neg = np.random.choice(neg_indices, target_samples, replace=False)
                    neg_strategy = "undersampling"
                else:
                    # Oversample: we need more, randomly select with replacement
                    selected_neg = np.random.choice(neg_indices, target_samples, replace=True)
                    neg_strategy = "oversampling"
                
                print(f"   Strategy: Positive ({pos_strategy}), Negative ({neg_strategy})")
                print(f"   Final: {len(selected_pos)} positive, {len(selected_neg)} negative samples")
                
            else:
                # Use minimum of the two class sizes, but cap at reasonable number
                min_size = min(len(pos_indices), len(neg_indices))
                max_samples = min(min_size, 1000)  # Cap at 1000 samples per class
                
                # Randomly sample indices without replacement (classic approach)
                np.random.seed(42)
                selected_pos = np.random.choice(pos_indices, max_samples, replace=False)
                selected_neg = np.random.choice(neg_indices, max_samples, replace=False)
            
            # Combine indices
            selected_indices = np.concatenate([selected_pos, selected_neg])
            np.random.shuffle(selected_indices)
            
            return X[selected_indices], y_binary[selected_indices], selected_indices
    
    return X, y_binary, np.arange(len(X))


def train_pytorch_model(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32, learning_rate=0.001, patience=10, device='cpu'):
    """
    Train a PyTorch binary classifier using separate validation data
    
    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data (test set)
    input_shape: Shape of input sequences
    epochs: Maximum number of epochs
    batch_size: Batch size for training
    learning_rate: Learning rate for optimizer
    patience: Early stopping patience
    device: Device to run training on
    
    Returns:
    model: Trained PyTorch model
    history: Training history
    """
    # Create model
    model = create_binary_lstm_classifier(input_shape, device=device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch + 1} (validation loss didn't improve)")
            break
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_binary_classifier(model, X_test, y_test, device='cpu'):
    """
    Evaluate a binary classifier on test data
    
    Parameters:
    model: Trained PyTorch model
    X_test: Test features
    y_test: Test labels
    device: Device to run evaluation on
    
    Returns:
    results: Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_prob = model(X_test_tensor).cpu().numpy().flatten()
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred_binary,
        'probabilities': y_pred_prob,
        'true_labels': y_test
    }


def train_binary_classifiers(X_train, y_train, X_test, y_test, selected_classes, device='cpu', 
                           epochs=50, batch_size=32, learning_rate=0.001, patience=10,
                           train_samples_per_class=3000, val_samples_per_class=200):
    """
    Train binary classifiers for multiple classes using one-vs-all strategy with test set as validation
    
    Parameters:
    X_train: Training features
    y_train: Training labels
    X_test: Test features (used as validation)
    y_test: Test labels (used as validation)
    selected_classes: List of classes to train binary classifiers for
    device: Device to run training on
    epochs: Maximum number of epochs per classifier
    batch_size: Batch size for training
    learning_rate: Learning rate for optimizer
    patience: Early stopping patience
    train_samples_per_class: Number of samples per class for training (3000 each positive/negative)
    val_samples_per_class: Number of samples per class for validation (200 each positive/negative)
    
    Returns:
    binary_classifiers: Dictionary of trained models
    training_results: Dictionary of training metrics
    training_histories: Dictionary of training histories
    """
    binary_classifiers = {}
    training_results = {}
    training_histories = {}
    
    print(f"🚀 TRAINING BINARY CLASSIFIERS - ONE VS ALL (PyTorch)")
    print(f"Using test set as validation for early stopping")
    print(f"Training samples per class: {train_samples_per_class}")
    print(f"Validation samples per class: {val_samples_per_class}")
    print(f"Using device: {device}")
    print("=" * 60)
    
    for class_idx, target_class in enumerate(selected_classes):
        print(f"📊 Training Binary Classifier {class_idx + 1}/{len(selected_classes)}")
        print(f"Target Class: {target_class} vs All Others")
        print("-" * 40)
        
        try:
            # Prepare binary training data (balanced, 3000 samples each)
            X_train_binary, y_train_binary, train_indices = prepare_binary_data(
                X_train, y_train, target_class, balance_classes=True,
                samples_per_class=train_samples_per_class
            )
            
            # Prepare binary validation data (balanced, 200 samples each)
            X_val_binary, y_val_binary, val_indices = prepare_binary_data(
                X_test, y_test, target_class, balance_classes=True,
                samples_per_class=val_samples_per_class
            )
            
            print(f"Binary training data prepared:")
            print(f"   • Training samples: {len(X_train_binary)}")
            print(f"   • Training positive (class {target_class}): {np.sum(y_train_binary)} samples")
            print(f"   • Training negative (others): {len(y_train_binary) - np.sum(y_train_binary)} samples")
            print(f"   • Validation samples: {len(X_val_binary)}")
            print(f"   • Validation positive (class {target_class}): {np.sum(y_val_binary)} samples")
            print(f"   • Validation negative (others): {len(y_val_binary) - np.sum(y_val_binary)} samples")
            
            if len(X_train_binary) < 10:  # Minimum samples needed
                print(f"   ⚠️ Insufficient training samples for class {target_class}, skipping...")
                continue
                
            if len(X_val_binary) < 5:  # Minimum validation samples needed
                print(f"   ⚠️ Insufficient validation samples for class {target_class}, skipping...")
                continue
            
            # Create and train model
            input_shape = (X_train_binary.shape[1], X_train_binary.shape[2])
            print(f"   • Model input shape: {input_shape}")
            print(f"   • Training started...")
            
            model, history = train_pytorch_model(
                X_train_binary, y_train_binary,
                X_val_binary, y_val_binary,
                input_shape, 
                epochs=epochs, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                patience=patience,
                device=device
            )
            
            # Store model and results
            binary_classifiers[target_class] = model
            training_histories[target_class] = history
            
            # Calculate training metrics
            train_results = evaluate_binary_classifier(model, X_train_binary, y_train_binary, device)
            val_results = evaluate_binary_classifier(model, X_val_binary, y_val_binary, device)
            
            # Store training results
            training_results[target_class] = {
                'train_accuracy': train_results['accuracy'],
                'train_precision': train_results['precision'],
                'train_recall': train_results['recall'],
                'train_f1': train_results['f1_score'],
                'val_accuracy': val_results['accuracy'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'val_f1': val_results['f1_score'],
                'train_samples': len(X_train_binary),
                'val_samples': len(X_val_binary),
                'positive_samples': np.sum(y_train_binary),
                'epochs_trained': len(history['train_loss'])
            }
            
            print(f"   ✅ Training completed")
            print(f"   • Epochs: {len(history['train_loss'])}")
            print(f"   • Final Training Accuracy: {train_results['accuracy']:.3f}")
            print(f"   • Final Training F1-Score: {train_results['f1_score']:.3f}")
            print(f"   • Final Validation Accuracy: {val_results['accuracy']:.3f}")
            print(f"   • Final Validation F1-Score: {val_results['f1_score']:.3f}")
            print()
            
        except Exception as e:
            print(f"   ❌ Training failed for class {target_class}: {str(e)}")
            print()
            continue
    
    print("🎯 BINARY CLASSIFIERS TRAINING SUMMARY")
    print("=" * 45)
    print(f"Successfully trained: {len(binary_classifiers)} out of {len(selected_classes)} classifiers")
    print(f"Trained classes: {list(binary_classifiers.keys())}")
    
    return binary_classifiers, training_results, training_histories


def evaluate_binary_classifiers(binary_classifiers, X_test, y_test, device='cpu', 
                              samples_per_class=200, balance_classes=True):
    """
    Evaluate all binary classifiers on test data
    
    Parameters:
    binary_classifiers: Dictionary of trained models
    X_test: Test features
    y_test: Test labels
    device: Device to run evaluation on
    samples_per_class: Number of samples per class for evaluation (200 each positive/negative)
    balance_classes: Whether to balance the evaluation data
    
    Returns:
    evaluation_results: Dictionary of evaluation metrics for each classifier
    """
    evaluation_results = {}
    
    print("🔍 EVALUATING BINARY CLASSIFIERS ON TEST DATA (PyTorch)")
    if balance_classes:
        print(f"Using balanced evaluation with {samples_per_class} samples per class")
    else:
        print("Using all available test data (unbalanced)")
    print("=" * 60)
    
    if not binary_classifiers:
        print("❌ No trained classifiers available.")
        return evaluation_results
    
    for class_num in sorted(binary_classifiers.keys()):
        print(f"📊 Evaluating Binary Classifier for Class {class_num}")
        print("-" * 35)
        
        model = binary_classifiers[class_num]
        
        try:
            # Prepare binary test data (balanced or unbalanced based on parameters)
            if balance_classes:
                X_test_binary, y_test_binary, test_indices = prepare_binary_data(
                    X_test, y_test, class_num, balance_classes=True,
                    samples_per_class=samples_per_class
                )
            else:
                X_test_binary, y_test_binary, test_indices = prepare_binary_data(
                    X_test, y_test, class_num, balance_classes=False
                )
            
            print(f"Test data prepared:")
            print(f"   • Total samples: {len(X_test_binary)}")
            print(f"   • Positive class (class {class_num}): {np.sum(y_test_binary)} samples")
            print(f"   • Negative class (others): {len(y_test_binary) - np.sum(y_test_binary)} samples")
            
            if len(X_test_binary) == 0:
                print(f"   ⚠️ No test samples for class {class_num}, skipping...")
                continue
            
            # Evaluate model
            results = evaluate_binary_classifier(model, X_test_binary, y_test_binary, device)
            
            # Store additional info
            results.update({
                'test_samples': len(X_test_binary),
                'positive_samples': np.sum(y_test_binary)
            })
            
            evaluation_results[class_num] = results
            
            print(f"   ✅ Evaluation Results:")
            print(f"   • Accuracy: {results['accuracy']:.3f}")
            print(f"   • Precision: {results['precision']:.3f}")
            print(f"   • Recall: {results['recall']:.3f}")
            print(f"   • F1-Score: {results['f1_score']:.3f}")
            print()
            
        except Exception as e:
            print(f"   ❌ Evaluation failed for class {class_num}: {str(e)}")
            print()
            continue
    
    # Summary of all evaluations
    if evaluation_results:
        print("🎯 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"Classes evaluated: {list(evaluation_results.keys())}")
        print()
        
        # Create summary table
        print("Performance Summary:")
        print("-" * 70)
        print(f"{'Class':<6} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10} {'Samples':<8}")
        print("-" * 70)
        
        for class_num in sorted(evaluation_results.keys()):
            results = evaluation_results[class_num]
            print(f"{class_num:<6} {results['accuracy']:<10.3f} {results['precision']:<11.3f} "
                  f"{results['recall']:<8.3f} {results['f1_score']:<10.3f} {results['test_samples']:<8}")
        
        print("-" * 70)
        
        # Calculate average performance
        avg_accuracy = np.mean([r['accuracy'] for r in evaluation_results.values()])
        avg_precision = np.mean([r['precision'] for r in evaluation_results.values()])
        avg_recall = np.mean([r['recall'] for r in evaluation_results.values()])
        avg_f1 = np.mean([r['f1_score'] for r in evaluation_results.values()])
        
        print(f"{'AVG':<6} {avg_accuracy:<10.3f} {avg_precision:<11.3f} "
              f"{avg_recall:<8.3f} {avg_f1:<10.3f}")
        print("-" * 70)
    else:
        print("❌ No evaluation results available.")
    
    return evaluation_results