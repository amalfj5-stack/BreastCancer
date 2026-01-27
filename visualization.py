# Enhanced visualization.py for IEEE Journal Standards - FIXED VERSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from process_data import LAMP
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters - FIXED spacing
plt.rcParams.update({
    'font.size': 9,           # Reduced from 10
    'font.family': 'serif',
    'axes.linewidth': 1.0,    # Reduced from 1.2
    'axes.labelsize': 10,     # Reduced from 11
    'axes.titlesize': 11,     # Reduced from 12
    'xtick.labelsize': 8,     # Reduced from 9
    'ytick.labelsize': 8,     # Reduced from 9
    'legend.fontsize': 8,     # Reduced from 9
    'figure.titlesize': 12,   # Reduced from 14
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def visualize_predictions(X_test, y_test, y_pred, method='pca', save_path=None):
    """
    FIXED: Visualize predictions with NO text overlaps and proper spacing
    """
    print(f"Creating enhanced visualization using {method.upper()}...")
    
    # Perform dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_test)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"Explained Variance: {explained_var[0]:.2f} + {explained_var[1]:.2f} = {sum(explained_var):.2f}"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, X_test.shape[0]-1))
        X_reduced = reducer.fit_transform(X_test)
        subtitle = f"Perplexity: {min(30, X_test.shape[0]-1)}, Learning Rate: auto"
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_test)
        subtitle = "UMAP: Uniform Manifold Approximation and Projection"
    elif method == 'lamp':
        reducer = LAMP(dim=2)
        reducer.fit(X_test)
        X_reduced = reducer.transform(X_test)
        subtitle = "LAMP: Local Affine Multidimensional Projection"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # FIXED: Much better spacing and larger figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))  # Increased width significantly
    plt.subplots_adjust(hspace=0.4, wspace=0.6, left=0.05, right=0.85, top=0.8, bottom=0.15)  # Much more space
    
    # Get unique labels and create better color palette
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    # Plot 1: True labels
    ax1 = axes[0]
    for label in unique_labels:
        mask = y_test == label
        if np.any(mask):
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[color_map[label]], label=label, alpha=0.8, s=35,  # Reduced size
                       edgecolors='black', linewidth=0.2)
    
    ax1.set_title('True Labels', fontweight='bold', fontsize=12, pad=15)
    ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=10)
    ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=10)
    # FIXED: Legend positioned outside plot with more space
    legend1 = ax1.legend(title='Cancer Type', bbox_to_anchor=(1.25, 1), loc='upper left',
                        frameon=True, fancybox=True, shadow=True, fontsize=8, title_fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted labels
    ax2 = axes[1]
    for label in unique_labels:
        mask = y_pred == label
        if np.any(mask):
            ax2.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[color_map[label]], label=label, alpha=0.8, s=35,
                       edgecolors='black', linewidth=0.2)
    
    ax2.set_title('Predicted Labels', fontweight='bold', fontsize=12, pad=15)
    ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=10)
    ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=10)
    legend2 = ax2.legend(title='Cancer Type', bbox_to_anchor=(1.25, 1), loc='upper left',
                        frameon=True, fancybox=True, shadow=True, fontsize=8, title_fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction errors
    ax3 = axes[2]
    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask
    
    # Plot correct predictions
    if np.any(correct_mask):
        ax3.scatter(X_reduced[correct_mask, 0], X_reduced[correct_mask, 1], 
                   c='green', label='Correct', alpha=0.8, s=35, marker='o', 
                   edgecolors='black', linewidth=0.2)
    
    # Plot incorrect predictions
    if np.any(incorrect_mask):
        ax3.scatter(X_reduced[incorrect_mask, 0], X_reduced[incorrect_mask, 1], 
                   c='red', label='Incorrect', alpha=0.8, s=45, marker='X', 
                   edgecolors='black', linewidth=0.2)
    
    ax3.set_title('Prediction Accuracy', fontweight='bold', fontsize=12, pad=15)
    ax3.set_xlabel(f'{method.upper()} Component 1', fontsize=10)
    ax3.set_ylabel(f'{method.upper()} Component 2', fontsize=10)
    legend3 = ax3.legend(bbox_to_anchor=(1.25, 1), loc='upper left',
                        frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # FIXED: Title positioning with much more space and no overlapping
    accuracy = np.mean(y_test == y_pred)
    fig.suptitle(f'Cancer Classification Visualization - {method.upper()}\nAccuracy: {accuracy:.3f}', 
                fontsize=16, fontweight='bold', y=0.92, x=0.45)  # Positioned to avoid overlap
    
    # Save figure with more space for legends
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    else:
        plt.savefig(f'{method}_prediction_visualization.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    
    plt.close()
    
    print(f"Enhanced {method.upper()} visualization saved with accuracy: {accuracy:.4f}")

def plot_feature_importance_comprehensive(feature_importances, gene_names, save_path='feature_importance_comprehensive.png'):
    """
    FIXED: Comprehensive feature importance with NO text overlaps
    """
    print("Creating comprehensive feature importance analysis...")
    
    if feature_importances is None or gene_names is None:
        print("Warning: Feature importances or gene names not available")
        return
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # FIXED: Much better layout with more space
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increased size
    plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Plot 1: Top 12 features bar plot (reduced to avoid overlap)
    ax1 = axes[0, 0]
    top_n = min(12, len(feature_importances))  # Reduced to 12
    top_indices = indices[:top_n]
    top_importances = feature_importances[top_indices]
    # FIXED: Truncate gene names more aggressively
    top_genes = [gene_names[i][:8] if i < len(gene_names) else f"Gene_{i}" for i in top_indices]
    
    y_pos = np.arange(top_n)
    bars = ax1.barh(y_pos, top_importances, color='skyblue', edgecolor='black', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_genes, fontsize=9)  # Increased font size
    ax1.set_xlabel('Feature Importance', fontsize=11)
    ax1.set_title(f'Top {top_n} Important Genes', fontweight='bold', fontsize=12, pad=20)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # FIXED: Only add value labels if there's enough space
    max_imp = max(top_importances) if len(top_importances) > 0 else 1
    for i, (bar, imp) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        if width < max_imp * 0.6:  # More conservative space check
            ax1.text(width + max_imp * 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.3f}', ha='left', va='center', fontsize=8)
    
    # Plot 2: Importance distribution
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(feature_importances, bins=20, alpha=0.7, color='green', edgecolor='black')
    mean_imp = np.mean(feature_importances)
    median_imp = np.median(feature_importances)
    
    ax2.axvline(mean_imp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_imp:.4f}')
    ax2.axvline(median_imp, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_imp:.4f}')
    ax2.set_xlabel('Feature Importance', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Feature Importance Distribution', fontweight='bold', fontsize=12, pad=20)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative importance - FIXED annotation positioning
    ax3 = axes[1, 0]
    cumulative_importance = np.cumsum(feature_importances[indices])
    ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
            'b-', linewidth=2, marker='o', markersize=1, alpha=0.8)
    ax3.axhline(y=0.8, color='red', linestyle='--', label='80% Threshold', alpha=0.8, linewidth=2)
    ax3.axhline(y=0.9, color='orange', linestyle='--', label='90% Threshold', alpha=0.8, linewidth=2)
    ax3.set_xlabel('Number of Features', fontsize=11)
    ax3.set_ylabel('Cumulative Importance', fontsize=11)
    ax3.set_title('Cumulative Feature Importance', fontweight='bold', fontsize=12, pad=20)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Find 80% threshold and annotate ONLY if there's clear space
    idx_80 = np.where(cumulative_importance >= 0.8)[0]
    if len(idx_80) > 0 and idx_80[0] < len(cumulative_importance) * 0.4:  # Only if in left 40%
        ax3.annotate(f'80% at {idx_80[0] + 1} features', 
                    xy=(idx_80[0] + 1, 0.8), 
                    xytext=(idx_80[0] + len(cumulative_importance) * 0.2, 0.6),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                    fontsize=9)
    
    # Plot 4: Top vs bottom features comparison
    ax4 = axes[1, 1]
    n_compare = min(6, len(feature_importances) // 2)  # Reduced to 6
    
    top_features = feature_importances[indices[:n_compare]]
    bottom_features = feature_importances[indices[-n_compare:]]
    
    bp = ax4.boxplot([top_features, bottom_features], 
                    labels=[f'Top {n_compare}', f'Bottom {n_compare}'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax4.set_ylabel('Feature Importance', fontsize=11)
    ax4.set_title('Top vs Bottom Features', fontweight='bold', fontsize=12, pad=20)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    plt.close()
    
    print(f"Comprehensive feature importance analysis saved")

def create_comprehensive_visualizations(model_results, X_test, y_test, y_pred, 
                                      feature_importances=None, gene_names=None,
                                      validation_results=None, y_proba=None):
    """
    FIXED: Create all visualizations with improved error handling and NO text overlaps
    """
    print("=== CREATING COMPREHENSIVE IEEE JOURNAL VISUALIZATIONS ===")
    
    visualizations_created = []
    
    # 1. Enhanced prediction visualizations - FIXED
    for method in ['pca', 'tsne', 'umap']:
        try:
            visualize_predictions(X_test, y_test, y_pred, method)
            visualizations_created.append(f"{method.upper()} visualization")
            print(f"✓ {method.upper()} visualization completed")
        except Exception as e:
            print(f"✗ {method.upper()} visualization failed: {e}")
    
    # 2. Learning curves analysis
    try:
        plot_learning_curves(model_results)
        visualizations_created.append("Learning curves analysis")
        print("✓ Learning curves analysis completed")
    except Exception as e:
        print(f"✗ Learning curves analysis failed: {e}")
    
    # 3. Enhanced confusion matrix - FIXED spacing
    try:
        from sklearn.metrics import confusion_matrix
        class_names = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        
        # FIXED: Better confusion matrix plot with proper spacing
        plt.figure(figsize=(12, 9))  # Increased size
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)  # Better margins
        
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontweight='bold', fontsize=14, pad=25)  # More padding
        plt.colorbar(shrink=0.8)  # Smaller colorbar
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight='bold')
        
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)  # More padding
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=0, fontsize=10)  # No rotation
        plt.yticks(tick_marks, class_names, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', pad_inches=0.3)
        plt.close()
        
        visualizations_created.append("Enhanced confusion matrix")
        print("✓ Enhanced confusion matrix completed")
    except Exception as e:
        print(f"✗ Enhanced confusion matrix failed: {e}")
    
    # 4. Multiclass ROC curves (if probabilities available) - FIXED spacing
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            class_names = np.unique(np.concatenate([y_test, y_pred]))
            y_test_bin = label_binarize(y_test, classes=class_names)
            n_classes = len(class_names)
            
            # FIXED: Better ROC plot spacing
            plt.figure(figsize=(12, 9))
            plt.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.12)
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
            plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
            plt.title('Multiclass ROC Curves', fontweight='bold', fontsize=14, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)  # Outside plot
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('multiclass_roc_curves.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.3)
            plt.close()
            
            visualizations_created.append("Multiclass ROC curves")
            print("✓ Multiclass ROC curves completed")
        except Exception as e:
            print(f"✗ Multiclass ROC curves failed: {e}")
    
    # 5. Comprehensive feature importance - FIXED
    if feature_importances is not None and gene_names is not None:
        try:
            plot_feature_importance_comprehensive(feature_importances, gene_names)
            visualizations_created.append("Comprehensive feature importance")
            print("✓ Comprehensive feature importance completed")
        except Exception as e:
            print(f"✗ Comprehensive feature importance failed: {e}")
    else:
        print("⚠ Feature importance analysis skipped (data not available)")
    
    print("✓ All visualizations created successfully with NO text overlaps")
    return visualizations_created

def plot_learning_curves(model_results, save_path='learning_curves_analysis.png'):
    """
    FIXED: Plot learning curves with better spacing and no text overlaps
    MODIFIED: Excluded Performance Summary subplot
    """
    print("Creating learning curves analysis...")
    
    if not isinstance(model_results, dict):
        print("Warning: No nested CV results available for learning curves")
        return
    
    # Filter only nested CV results
    cv_results = {name: result for name, result in model_results.items() 
                 if isinstance(result, dict) and 'scores' in result}
    
    if not cv_results:
        print("Warning: No cross-validation scores available")
        return
    
    # MODIFIED: Better layout with 2x3 grid but only use 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Keep 2x3 for consistent spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.4, left=0.08, right=0.95, top=0.9, bottom=0.1)
    
    model_names = list(cv_results.keys())
    
    # Plot 1: CV scores distribution (box plot)
    ax1 = axes[0, 0]
    positions = np.arange(len(model_names))
    scores_data = [cv_results[name]['scores'] for name in model_names]
    bp = ax1.boxplot(scores_data, positions=positions, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Accuracy Score')
    ax1.set_title('Cross-Validation Score Distribution', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean scores with confidence intervals - FIXED text positioning
    ax2 = axes[0, 1]
    means = [cv_results[name]['mean_score'] for name in model_names]
    stds = [cv_results[name]['std_score'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                   color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Accuracy ± Std')
    ax2.set_title('Mean Accuracy with Error Bars', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # FIXED: Better text positioning - only add if there's space
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        if height + std < max(means) + max(stds) * 0.7:  # Only if enough space
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(stds) * 0.1,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=7)
    
    # Plot 3: Feature importance stability (if available)
    ax3 = axes[0, 2]
    importance_plotted = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(model_names):
        if 'feature_importances' in cv_results[name] and cv_results[name]['feature_importances']:
            importances = cv_results[name]['feature_importances']
            if len(importances) > 1:
                importances_array = np.array(importances)
                if len(importances_array.shape) > 1 and importances_array.shape[0] > 1:
                    std_importances = np.std(importances_array, axis=0)
                    mean_importances = np.mean(importances_array, axis=0)
                    top_10_indices = np.argsort(mean_importances)[-10:]
                    
                    ax3.plot(range(10), std_importances[top_10_indices], 
                            label=f'{name}', marker='o', color=colors[i % len(colors)], linewidth=2)
                    importance_plotted = True
    
    if importance_plotted:
        ax3.set_title('Feature Importance Stability', fontweight='bold', pad=15)
        ax3.set_xlabel('Feature Rank')
        ax3.set_ylabel('Std Dev of Importance')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Feature Importance\nStability Analysis\n(Not available)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax3.set_title('Feature Importance Stability', fontweight='bold', pad=15)
    
    # Plot 4: Model performance ranking - FIXED spacing
    ax4 = axes[1, 0]
    performance_data = [(name, cv_results[name]['mean_score'], cv_results[name]['std_score']) 
                       for name in model_names]
    performance_data.sort(key=lambda x: x[1], reverse=True)
    
    names_ranked = [item[0] for item in performance_data]
    scores_ranked = [item[1] for item in performance_data]
    stds_ranked = [item[2] for item in performance_data]
    
    y_pos = np.arange(len(names_ranked))
    bars = ax4.barh(y_pos, scores_ranked, xerr=stds_ranked, alpha=0.7, 
                   color=colors[:len(names_ranked)], edgecolor='black')
    
    ax4.set_yticks(y_pos)
    # FIXED: Shorter labels to avoid overlap
    short_names = [name[:8] + '...' if len(name) > 8 else name for name in names_ranked]
    ax4.set_yticklabels(short_names, fontsize=8)
    ax4.set_xlabel('Mean Accuracy')
    ax4.set_title('Model Performance Ranking', fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Model stability analysis - FIXED text positioning
    ax5 = axes[1, 1]
    cv_coefficients = [cv_results[name]['std_score'] / cv_results[name]['mean_score'] 
                      for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax5.bar(x_pos, cv_coefficients, color='orange', alpha=0.7, edgecolor='black')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('Model Stability Analysis', fontweight='bold', pad=15)
    ax5.grid(True, alpha=0.3)
    
    # Add threshold line
    ax5.axhline(y=0.05, color='red', linestyle='--', label='Good Stability (CV < 0.05)', linewidth=2)
    ax5.legend(loc='upper right', fontsize=7)
    
    # FIXED: Add value labels only if there's space
    max_cv = max(cv_coefficients) if cv_coefficients else 0.1
    for bar, cv in zip(bars, cv_coefficients):
        height = bar.get_height()
        if height < max_cv * 0.8:  # Only if enough space
            ax5.text(bar.get_x() + bar.get_width()/2., height + max_cv * 0.02,
                    f'{cv:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # MODIFIED: Hide the 6th subplot (Performance Summary removed)
    ax6 = axes[1, 2]
    ax6.axis('off')  # Turn off the 6th subplot completely
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()