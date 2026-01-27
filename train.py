
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import gc
from contextlib import contextmanager

@contextmanager
def joblib_resource_manager():
    """Context manager to properly clean up joblib resources"""
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()
        # Clear any remaining parallel backends
        try:
            from joblib import parallel_backend
            with parallel_backend('threading', n_jobs=1):
                pass
        except:
            pass

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Original function - kept for backward compatibility
    But now includes warnings about the limitations
    """
    print("WARNING: Using simple train-test split. For IEEE journal standards, use train_with_nested_cv() instead.")
    
    # Convert string labels to numeric for models that require it
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # XGBoost requires numeric labels
        if name == 'XGBoost':
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        classes = label_encoder.classes_ if name == 'XGBoost' else np.unique(y_test)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes,
                   yticklabels=classes)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
    
    # Compare model performance
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    return results

class XGBoostWrapper:
    """
    Enhanced wrapper for XGBoost to handle string labels automatically
    Updated for better external dataset compatibility
    """
    def __init__(self, **params):
        # Set default parameters if not provided
        default_params = {
            'eval_metric': 'logloss',
            'random_state': 42
        }
        default_params.update(params)
        
        self.xgb_model = XGBClassifier(**default_params)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the model with automatic label encoding"""
        # Encode labels for XGBoost
        y_encoded = self.label_encoder.fit_transform(y)
        self.xgb_model.fit(X, y_encoded)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict with automatic label decoding"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        y_pred_encoded = self.xgb_model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.xgb_model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get XGBoost parameters"""
        params = self.xgb_model.get_params(deep)
        return params
    
    def set_params(self, **params):
        """Set XGBoost parameters"""
        self.xgb_model.set_params(**params)
        return self
    
    @property
    def feature_importances_(self):
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances")
        return self.xgb_model.feature_importances_
    
    @property
    def classes_(self):
        """Get class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing classes")
        return self.label_encoder.classes_
    
    @property
    def n_features_in_(self):
        """Get number of features used during training"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing n_features_in_")
        return self.xgb_model.n_features_in_

def train_with_nested_cv(X, y, random_state=42, outer_cv_folds=5, inner_cv_folds=3):
    """
    ENHANCED: IEEE Journal Standard Training with Nested Cross-Validation
    Updated for better external dataset compatibility
    
    Parameters:
    -----------
    X : numpy.ndarray
        All features (not split into train/test)
    y : numpy.ndarray
        All labels (not split into train/test)
    random_state : int
        Random seed for reproducibility
    outer_cv_folds : int
        Number of outer CV folds
    inner_cv_folds : int
        Number of inner CV folds for hyperparameter tuning
    
    Returns:
    --------
    dict
        Enhanced results with statistical validation
    """
    print("=== NESTED CROSS-VALIDATION FOR IEEE JOURNAL STANDARDS ===")
    print(f"Using {outer_cv_folds}-fold outer CV and {inner_cv_folds}-fold inner CV")
    print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")
    
    # Define models with enhanced hyperparameter grids
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'XGBoost': {
            'model': XGBoostWrapper(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=random_state),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(max_iter=1000, random_state=random_state),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'solver': ['adam', 'lbfgs']
            }
        }
    }
    
    # Set up cross-validation
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state)
    
    results = {}
    all_predictions = {}  # Store predictions for statistical analysis
    
    # Process each model
    for name, config in models.items():
        print(f"\nProcessing {name} with nested cross-validation...")
        
        model = config['model']
        param_grid = config['params']
        
        outer_scores = []
        feature_importances_all = []
        best_params_all = []
        fold_predictions = []
        fold_true_labels = []
        
        # Outer cross-validation loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"  Outer fold {fold_idx + 1}/{outer_cv_folds}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner cross-validation for hyperparameter tuning
            try:
                with joblib_resource_manager():
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=inner_cv,
                        scoring='accuracy',
                        n_jobs=2,  # FIXED: Limit to 2 jobs instead of -1 to prevent resource leaks
                        verbose=0,
                        error_score='raise'
                    )
                    
                    # Fit on outer training set
                    grid_search.fit(X_train_outer, y_train_outer)
                
                # Get best model from inner CV
                best_model = grid_search.best_estimator_
                best_params_all.append(grid_search.best_params_)
                
                # Evaluate on outer test set
                y_pred_fold = best_model.predict(X_test_outer)
                fold_score = accuracy_score(y_test_outer, y_pred_fold)
                outer_scores.append(fold_score)
                
                # Store predictions for later analysis
                fold_predictions.extend(y_pred_fold)
                fold_true_labels.extend(y_test_outer)
                
                # Store feature importances if available
                if hasattr(best_model, 'feature_importances_'):
                    feature_importances_all.append(best_model.feature_importances_)
                
                print(f"    Fold {fold_idx + 1} accuracy: {fold_score:.4f}")
                print(f"    Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"    ERROR in fold {fold_idx + 1} for {name}: {str(e)}")
                # Skip this fold but continue
                continue
        
        # Skip model if no successful folds
        if not outer_scores:
            print(f"  SKIPPING {name} - no successful folds")
            continue
        
        # Calculate statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        # Calculate 95% confidence interval
        n = len(outer_scores)
        se = std_score / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_score, scale=se)
        
        # Store comprehensive results
        results[name] = {
            'scores': outer_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'confidence_interval_95': ci_95,
            'best_params_per_fold': best_params_all,
            'feature_importances': feature_importances_all,
            'all_predictions': fold_predictions,
            'all_true_labels': fold_true_labels,
            'best_model': grid_search.best_estimator_ if 'grid_search' in locals() else None
        }
        
        print(f"  {name} Results:")
        print(f"    Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"    95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"    Successful folds: {len(outer_scores)}/{outer_cv_folds}")
        
        # Store for statistical comparison
        all_predictions[name] = {
            'predictions': fold_predictions,
            'true_labels': fold_true_labels
        }
    
    # Check if we have any successful models
    if not results:
        print("ERROR: No models completed successfully!")
        return None
    
    # Statistical significance testing between models
    print("\n=== STATISTICAL SIGNIFICANCE TESTING ===")
    model_names = list(results.keys())
    n_models = len(model_names)
    
    if n_models > 1:
        # Pairwise statistical tests
        significance_matrix = np.ones((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    scores1 = results[model1]['scores']
                    scores2 = results[model2]['scores']
                    
                    # Only compare if both models have same number of successful folds
                    min_folds = min(len(scores1), len(scores2))
                    if min_folds >= 2:  # Need at least 2 samples for t-test
                        scores1_trim = scores1[:min_folds]
                        scores2_trim = scores2[:min_folds]
                        
                        # Paired t-test (since same CV folds)
                        try:
                            t_stat, p_value = stats.ttest_rel(scores1_trim, scores2_trim)
                            significance_matrix[i, j] = p_value
                            
                            if p_value < 0.05:
                                print(f"  {model1} vs {model2}: p = {p_value:.4f} (SIGNIFICANT)")
                            else:
                                print(f"  {model1} vs {model2}: p = {p_value:.4f}")
                        except Exception as e:
                            print(f"  {model1} vs {model2}: statistical test failed - {e}")
                    else:
                        print(f"  {model1} vs {model2}: insufficient data for comparison")
        
        # Visualize statistical significance
        plt.figure(figsize=(10, 8))
        mask = np.eye(n_models, dtype=bool)
        sns.heatmap(significance_matrix, 
                   xticklabels=model_names,
                   yticklabels=model_names,
                   annot=True, 
                   fmt='.4f',
                   cmap='RdYlBu_r',
                   mask=mask,
                   vmin=0, vmax=0.1,
                   center=0.05)
        plt.title('Statistical Significance Testing (p-values)\nRed: p < 0.05 (significant difference)')
        plt.tight_layout()
        plt.savefig('statistical_significance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add statistical test results to results
        results['statistical_tests'] = {
            'significance_matrix': significance_matrix,
            'model_names': model_names
        }
    
    # Enhanced performance comparison visualization
    plt.figure(figsize=(16, 12))
    
    # Box plot of cross-validation scores
    plt.subplot(2, 3, 1)
    scores_data = [results[name]['scores'] for name in model_names]
    box_plot = plt.boxplot(scores_data, labels=model_names, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
    
    plt.title('Cross-Validation Score Distribution')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Mean scores with confidence intervals
    plt.subplot(2, 3, 2)
    means = [results[name]['mean_score'] for name in model_names]
    ci_lows = [results[name]['confidence_interval_95'][0] for name in model_names]
    ci_highs = [results[name]['confidence_interval_95'][1] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = plt.bar(x_pos, means, 
                   yerr=[np.array(means) - np.array(ci_lows), 
                         np.array(ci_highs) - np.array(means)], 
                   capsize=5, color=colors[:len(means)], alpha=0.7)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(x_pos, model_names, rotation=45)
    plt.title('Mean Accuracy with 95% Confidence Intervals')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Feature importance stability (if available)
    plt.subplot(2, 3, 3)
    importance_plotted = False
    for i, name in enumerate(model_names):
        if results[name]['feature_importances'] and len(results[name]['feature_importances']) > 1:
            importances = np.array(results[name]['feature_importances'])
            if len(importances.shape) > 1 and importances.shape[0] > 1:
                std_importances = np.std(importances, axis=0)
                mean_importances = np.mean(importances, axis=0)
                top_10_indices = np.argsort(mean_importances)[-10:]
                
                plt.plot(range(10), std_importances[top_10_indices], 
                        label=f'{name}', marker='o', color=colors[i])
                importance_plotted = True
    
    if importance_plotted:
        plt.title('Feature Importance Stability (Top 10 Features)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Standard Deviation of Importance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Feature Importance\nStability Analysis\n(Not available)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance Stability')
    
    # Model performance comparison with rankings
    plt.subplot(2, 3, 4)
    performace_data = [(name, results[name]['mean_score'], results[name]['std_score']) 
                      for name in model_names]
    performace_data.sort(key=lambda x: x[1], reverse=True)
    
    ranks = [f"#{i+1}" for i in range(len(performace_data))]
    names_ranked = [item[0] for item in performace_data]
    scores_ranked = [item[1] for item in performace_data]
    stds_ranked = [item[2] for item in performace_data]
    
    y_pos = np.arange(len(names_ranked))
    plt.barh(y_pos, scores_ranked, xerr=stds_ranked, alpha=0.7, 
            color=colors[:len(names_ranked)])
    
    plt.yticks(y_pos, [f"{rank} {name}" for rank, name in zip(ranks, names_ranked)])
    plt.xlabel('Mean Accuracy')
    plt.title('Model Performance Ranking')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Cross-validation stability analysis
    plt.subplot(2, 3, 5)
    cv_coefficients = [results[name]['std_score'] / results[name]['mean_score'] 
                      for name in model_names]
    
    bars = plt.bar(model_names, cv_coefficients, color=colors[:len(model_names)], alpha=0.7)
    plt.ylabel('Coefficient of Variation')
    plt.title('Model Stability Analysis')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line for good stability threshold
    plt.axhline(y=0.05, color='red', linestyle='--', label='Good Stability Threshold (CV < 0.05)')
    plt.legend()
    
    # Add value labels on bars
    for bar, cv in zip(bars, cv_coefficients):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Model comparison summary
    plt.subplot(2, 3, 6)
    summary_text = "MODEL COMPARISON SUMMARY\n\n"
    
    for i, (name, mean, std) in enumerate(performace_data):
        ci = results[name]['confidence_interval_95']
        summary_text += f"{i+1}. {name}\n"
        summary_text += f"   Accuracy: {mean:.3f} ± {std:.3f}\n"
        summary_text += f"   95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]\n\n"
    
    plt.text(0.05, 0.95, summary_text, ha='left', va='top', transform=plt.gca().transAxes,
             fontfamily='monospace', fontsize=10)
    plt.title('Performance Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== NESTED CROSS-VALIDATION COMPLETED ===")
    print("Key improvements over simple train-test split:")
    print("1. ✓ Unbiased performance estimates")
    print("2. ✓ Statistical significance testing")  
    print("3. ✓ 95% Confidence intervals")
    print("4. ✓ Feature importance stability analysis")
    print("5. ✓ Protection against overfitting")
    print("6. ✓ Enhanced hyperparameter optimization")
    print("7. ✓ Model stability assessment")
    
    gc.collect()
    
    return results

def get_best_model_final(nested_cv_results, X, y):
    """
    Train the final model on all data using the best performing algorithm
    Enhanced for better external dataset compatibility
    
    Parameters:
    -----------
    nested_cv_results : dict
        Results from nested cross-validation
    X : numpy.ndarray
        All features
    y : numpy.ndarray
        All labels
    
    Returns:
    --------
    tuple
        (Final trained model, model name)
    """
    if not nested_cv_results:
        raise ValueError("No valid results from nested cross-validation")
    
    # Find best model based on mean CV score
    valid_models = {k: v for k, v in nested_cv_results.items() 
                   if isinstance(v, dict) and 'mean_score' in v}
    
    if not valid_models:
        raise ValueError("No valid model results found")
    
    best_model_name = max(valid_models, key=lambda x: valid_models[x]['mean_score'])
    best_result = nested_cv_results[best_model_name]
    
    print(f"\nTraining final model: {best_model_name}")
    print(f"Expected performance: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")
    
    # Get the best parameters from cross-validation
    if 'best_params_per_fold' in best_result and best_result['best_params_per_fold']:
        # Use the most common parameter values across folds
        all_params = best_result['best_params_per_fold']
        best_params = {}
        
        # For each parameter, find the most common value across folds
        for param_name in all_params[0].keys():
            param_values = [params[param_name] for params in all_params if param_name in params]
            # For numeric values, take the median; for categorical, take the mode
            if param_values:
                if isinstance(param_values[0], (int, float)):
                    best_params[param_name] = np.median(param_values)
                    if isinstance(param_values[0], int):
                        best_params[param_name] = int(best_params[param_name])
                else:
                    # Take most common value
                    best_params[param_name] = max(set(param_values), key=param_values.count)
        
        print(f"Using optimized parameters: {best_params}")
    else:
        best_params = {}
        print("No parameter optimization results found, using default parameters")
    
    # Create final model with best parameters
    if best_model_name == 'XGBoost':
        final_model = XGBoostWrapper(**best_params)
    elif best_model_name == 'Random Forest':
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif best_model_name == 'SVM':
        final_model = SVC(probability=True, **best_params, random_state=42)
    elif best_model_name == 'Neural Network':
        final_model = MLPClassifier(max_iter=1000, **best_params, random_state=42)
    else:
        # Fallback to the stored model
        if 'best_model' in best_result and best_result['best_model'] is not None:
            final_model = best_result['best_model']
        else:
            raise ValueError(f"Unknown model name: {best_model_name}")
    
    # Train on all data
    print("Training final model on complete dataset...")
    final_model.fit(X, y)
    
    print(f"✓ Final {best_model_name} model trained successfully")
    
    # Display model characteristics
    if hasattr(final_model, 'feature_importances_'):
        print(f"  Model has feature importances (useful for interpretation)")
    if hasattr(final_model, 'predict_proba'):
        print(f"  Model supports probability predictions")
    if hasattr(final_model, 'n_features_in_'):
        print(f"  Model expects {final_model.n_features_in_} features")
    
    return final_model, best_model_name

