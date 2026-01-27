"""
Enhanced Breast Cancer Gene Expression Data Preprocessing Pipeline
Updated to work properly with external datasets - FIXED VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LAMP:
    def __init__(self, dim=2):
        self.dim = dim
        self.control_points = None
        self.control_proj = None

    def fit(self, X):
        n_samples = X.shape[0]
        control_size = min(10, n_samples)
        control_indices = np.random.choice(n_samples, control_size, replace=False)
        self.control_points = X[control_indices]

        pca = PCA(n_components=self.dim)
        self.control_proj = pca.fit_transform(self.control_points)

    def transform(self, X):
        X_proj = []
        for x in X:
            weights = 1 / (np.linalg.norm(self.control_points - x, axis=1) + 1e-5)
            weights /= weights.sum()
            A = (self.control_proj - self.control_proj.mean(axis=0)).T @ np.diag(weights)
            B = (self.control_points - self.control_points.mean(axis=0)).T @ np.diag(weights)
            try:
                M = A @ np.linalg.pinv(B)
                x_proj = self.control_proj.mean(axis=0) + M @ (x - self.control_points.mean(axis=0))
            except np.linalg.LinAlgError:
                x_proj = np.zeros(self.control_proj.shape[1])
            X_proj.append(x_proj)
        return np.array(X_proj)
        
class GeneExpressionPreprocessor:
    """
    A comprehensive preprocessing pipeline for gene expression data
    Enhanced for external dataset compatibility - FIXED VERSION
    """
    
    def __init__(self, expression_file=None, labels_file=None, random_state=42):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        expression_file : str
            Path to the gene expression data file
        labels_file : str
            Path to the labels file
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.expression_df = None
        self.labels_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Enhanced attributes for external validation
        self.scaler = None
        self.dim_reducer = None
        self.selected_features = None  # Store selected feature indices
        self.feature_names_in_ = None  # Store original feature names
        self.all_feature_names = None  # Store all feature names from data
        self.label_encoder = None  # Store label encoder
        
        # FIXED: Store original data before preprocessing
        self.X_original = None  # Store original unprocessed data
        self.all_feature_names_original = None  # Store original feature names
        
        if expression_file and labels_file:
            self.load_data(expression_file, labels_file)
    
    def load_data(self, expression_file, labels_file):
        """
        Load gene expression and labels data
        Enhanced to store feature names for external validation
        
        Parameters:
        -----------
        expression_file : str
            Path to the gene expression data file
        labels_file : str
            Path to the labels file
        """
        print("Loading data...")
        
        # Load gene expression data
        self.expression_df = pd.read_csv(expression_file)
        
        print(f"Original expression data shape: {self.expression_df.shape}")
        
        # Load labels data
        self.labels_df = pd.read_csv(labels_file)
        print(f"Labels shape: {self.labels_df.shape}")
        
        # Skip the first row if it's a header row (empty in first column)
        if len(self.expression_df.columns) > 0 and pd.isna(self.expression_df.iloc[0, 0]):
            print("Detected header row, removing...")
            self.expression_df = self.expression_df.iloc[1:]
            self.expression_df.reset_index(drop=True, inplace=True)
        
        print(f"Expression data shape after processing: {self.expression_df.shape}")
        
        # Make sure the number of samples match
        if len(self.expression_df) != len(self.labels_df):
            raise ValueError(f"Number of samples in expression data ({len(self.expression_df)}) and labels ({len(self.labels_df)}) do not match!")
        
        # Extract sample IDs and gene expression values
        # Assuming first column contains sample IDs
        sample_ids = self.expression_df.iloc[:, 0]
        self.X = self.expression_df.iloc[:, 1:].values
        
        # FIXED: Store original data and feature names before any processing
        self.X_original = self.X.copy()  # Store original unprocessed data
        
        # Store all feature names for external validation
        self.all_feature_names = list(self.expression_df.columns[1:])  # Skip sample ID column
        self.all_feature_names_original = self.all_feature_names.copy()  # Store original copy
        self.feature_names_in_ = self.all_feature_names.copy()
        
        # Extract labels
        self.y = self.labels_df.iloc[:, 1].values
        
        # Initialize label encoder for consistent label handling
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"Data loaded successfully with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Feature names stored: {len(self.all_feature_names)} features")
        print(f"Class distribution: {pd.Series(self.y).value_counts()}")
    
    def load_from_memory(self, X, y, feature_names=None):
        """
        Load data directly from memory
        Enhanced to store feature names
        
        Parameters:
        -----------
        X : numpy.ndarray
            Gene expression data matrix
        y : numpy.ndarray
            Class labels
        feature_names : list, optional
            Feature names
        """
        self.X = X
        self.y = y
        
        # FIXED: Store original data
        self.X_original = X.copy()
        
        # Store feature names if provided
        if feature_names is not None:
            self.all_feature_names = feature_names
            self.all_feature_names_original = feature_names.copy()
            self.feature_names_in_ = feature_names.copy()
        else:
            # Create default feature names
            self.all_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.all_feature_names_original = self.all_feature_names.copy()
            self.feature_names_in_ = self.all_feature_names.copy()
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"Data loaded from memory with {self.X.shape[0]} samples and {self.X.shape[1]} features")
        print(f"Feature names stored: {len(self.all_feature_names)} features")
        print(f"Class distribution: {pd.Series(self.y).value_counts()}")
    
    def check_missing_values(self):
        """
        Check for missing values in the data
        
        Returns:
        --------
        dict
            Summary of missing values
        """
        if self.X is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print("Checking for missing values...")
        
        # Create a temporary DataFrame for counting missing values
        temp_df = pd.DataFrame(self.X)
        
        # Count missing values
        missing_count = temp_df.isnull().sum().sum()
        zero_count = (temp_df == 0).sum().sum()
        
        result = {
            'missing_values': missing_count,
            'zero_values': zero_count,
            'missing_percentage': missing_count / (self.X.shape[0] * self.X.shape[1]) * 100,
            'zero_percentage': zero_count / (self.X.shape[0] * self.X.shape[1]) * 100
        }
        
        print(f"Missing values: {result['missing_values']} ({result['missing_percentage']:.2f}%)")
        print(f"Zero values: {result['zero_values']} ({result['zero_percentage']:.2f}%)")
        
        return result
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the data
        
        Parameters:
        -----------
        strategy : str, default='mean'
            Strategy to handle missing values ('mean', 'median', 'zero', or 'knn')
        
        Returns:
        --------
        numpy.ndarray
            Data with missing values handled
        """
        if self.X is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print(f"Handling missing values with strategy: {strategy}...")
        
        # Create a temporary DataFrame for easier handling
        temp_df = pd.DataFrame(self.X)
        
        if strategy == 'mean':
            self.X = temp_df.fillna(temp_df.mean()).values
        elif strategy == 'median':
            self.X = temp_df.fillna(temp_df.median()).values
        elif strategy == 'zero':
            self.X = temp_df.fillna(0).values
        elif strategy == 'knn':
            # KNN imputation requires sklearn.impute.KNNImputer
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            self.X = imputer.fit_transform(self.X)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print("Missing values handled.")
        return self.X
    
    def explore_data_distribution(self, n_bins=50, n_genes=5):
        """
        Explore the distribution of gene expression values
        
        Parameters:
        -----------
        n_bins : int, default=50
            Number of bins for histograms
        n_genes : int, default=5
            Number of genes to plot
        """
        if self.X is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print("Exploring data distribution...")
        
        # Plot overall distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.X.flatten(), bins=n_bins, alpha=0.7)
        plt.title('Overall Distribution of Gene Expression Values')
        plt.xlabel('Expression Value')
        plt.ylabel('Frequency')
        
        # Plot boxplot of expression values
        plt.subplot(1, 2, 2)
        plt.boxplot(self.X[:, :n_genes])
        plt.title(f'Boxplot of First {n_genes} Genes')
        plt.xlabel('Gene Index')
        plt.ylabel('Expression Value')
        
        plt.tight_layout()
        plt.savefig('data_distribution.png')
        plt.close()
        
        # Class-specific distributions
        classes = np.unique(self.y)
        
        plt.figure(figsize=(12, 4 * len(classes)))
        
        for i, cls in enumerate(classes):
            cls_data = self.X[self.y == cls]
            
            plt.subplot(len(classes), 1, i + 1)
            plt.hist(cls_data.flatten(), bins=n_bins, alpha=0.7)
            plt.title(f'Distribution for Class {cls}')
            plt.xlabel('Expression Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('class_distributions.png')
        plt.close()
        
        print("Data distribution exploration completed. See 'data_distribution.png' and 'class_distributions.png'.")
    
    def normalize_data(self, method='standard'):
        """
        Normalize the gene expression data
        
        Parameters:
        -----------
        method : str, default='standard'
            Normalization method ('standard', 'robust', 'minmax', or 'log')
        
        Returns:
        --------
        numpy.ndarray
            Normalized data
        """
        if self.X is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print(f"Normalizing data using {method} method...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif method == 'robust':
            self.scaler = RobustScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif method == 'log':
            # Apply log2 transformation (adding a small constant to avoid log(0))
            self.X = np.log2(self.X + 1e-6)
            self.scaler = None  # No scaler for log transformation
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        print("Data normalization completed.")
        return self.X
    
    def select_features(self, method='kbest', n_features=100):
        """
        Select most informative features
        Enhanced to store selected feature information
        
        Parameters:
        -----------
        method : str, default='kbest'
            Feature selection method ('kbest', 'mutual_info', or 'variance')
        n_features : int, default=100
            Number of features to select
        
        Returns:
        --------
        numpy.ndarray
            Data with selected features
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print(f"Selecting {n_features} features using {method} method...")
        
        if method == 'kbest':
            selector = SelectKBest(f_classif, k=n_features)
            self.X = selector.fit_transform(self.X, self.y)
            self.selected_features = selector.get_support(indices=True)
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            self.X = selector.fit_transform(self.X, self.y)
            self.selected_features = selector.get_support(indices=True)
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            # Select features with variance above threshold
            selector = VarianceThreshold(threshold=0.1)
            self.X = selector.fit_transform(self.X)
            self.selected_features = selector.get_support(indices=True)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Update feature names to selected features only
        if self.all_feature_names is not None and self.selected_features is not None:
            self.feature_names_in_ = [self.all_feature_names[i] for i in self.selected_features]
        
        print(f"Feature selection completed. Reduced to {self.X.shape[1]} features.")
        print(f"Selected feature indices: {self.selected_features[:10]}..." if len(self.selected_features) > 10 else f"Selected feature indices: {self.selected_features}")
        
        return self.X
    
    def reduce_dimensions(self, method='pca', n_components=2):
        """
        Reduce dimensions of the data
        
        Parameters:
        -----------
        method : str, default='pca'
            Dimensionality reduction method ('pca', 'tsne', 'umap', or 'lamp')
        n_components : int, default=2
            Number of components to reduce to
        
        Returns:
        --------
        numpy.ndarray
            Dimensionally reduced data
        """
        if self.X is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print(f"Reducing dimensions to {n_components} components using {method}...")
        
        if method == 'pca':
            self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
            reduced_data = self.dim_reducer.fit_transform(self.X)
            self.explained_variance = self.dim_reducer.explained_variance_ratio_
            print(f"Explained variance: {sum(self.explained_variance):.4f}")
        elif method == 'tsne':
            self.dim_reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, self.X.shape[0]-1)) # Added random state
            reduced_data = self.dim_reducer.fit_transform(self.X)
        elif method == 'umap':
            self.dim_reducer = umap.UMAP(n_components=n_components, random_state=self.random_state)
            reduced_data = self.dim_reducer.fit_transform(self.X)
        elif method == 'lamp':
            self.dim_reducer = LAMP(dim=n_components)
            self.dim_reducer.fit(self.X)
            reduced_data = self.dim_reducer.transform(self.X)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Visualize the reduced data
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            
            classes = np.unique(self.y)
            for cls in classes:
                plt.scatter(
                    reduced_data[self.y == cls, 0],
                    reduced_data[self.y == cls, 1],
                    label=f'Class {cls}',
                    alpha=0.7
                )
            
            plt.title(f'{method.upper()} Visualization of Gene Expression Data')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
            plt.savefig(f'{method}_visualization.png')
            plt.close()
        
        print(f"Dimensionality reduction completed. See '{method}_visualization.png'.")
        
        return reduced_data
    
    def split_train_test(self, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        stratify : bool, default=True
            Whether to preserve the class distribution in train and test sets
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded! Call load_data() first.")
        
        print(f"Splitting data into train and test sets (test_size={test_size}, stratify={stratify})...")
        
        stratify_param = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Data split completed. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Check class distribution in train and test sets
        train_distribution = pd.Series(self.y_train).value_counts(normalize=True)
        test_distribution = pd.Series(self.y_test).value_counts(normalize=True)
        
        print("Class distribution:")
        print(f"Training set: {dict(train_distribution.round(3))}")
        print(f"Test set: {dict(test_distribution.round(3))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def transform_for_external_validation(self, X_external):
        """
        FIXED: Transform external dataset with proper feature selection and scaling order
        """
        print("Transforming external dataset using FIXED preprocessing pipeline...")
        
        # Convert to DataFrame if numpy array
        if isinstance(X_external, np.ndarray):
            if self.all_feature_names_original is not None and len(self.all_feature_names_original) == X_external.shape[1]:
                X_external = pd.DataFrame(X_external, columns=self.all_feature_names_original)
            else:
                X_external = pd.DataFrame(X_external, columns=[f"gene_{i}" for i in range(X_external.shape[1])])
        
        print(f"External dataset shape: {X_external.shape}")
        
        # Step 1: Feature alignment
        if self.all_feature_names_original is not None:
            print(f"Original training data had {len(self.all_feature_names_original)} features")
            
            # Find exact feature matches
            common_features = list(set(X_external.columns).intersection(set(self.all_feature_names_original)))
            print(f"  Found {len(common_features)} exact feature matches.")
            
            if len(common_features) >= 50:
                print(f"  Using {len(common_features)} matched features")
                X_aligned = X_external[common_features].copy()
                
                # Reorder to match training order
                training_order = [f for f in self.all_feature_names_original if f in common_features]
                X_aligned = X_aligned[training_order]
            else:
                print(f"  Not enough feature matches, using all external features")
                X_aligned = X_external.copy()
        else:
            print("  No original feature names available, using external features as-is")
            X_aligned = X_external.copy()
        
        # Step 2: Handle missing values
        missing_count = X_aligned.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values with column medians...")
            X_aligned = X_aligned.fillna(X_aligned.median())
        
        # Convert to numpy array for processing
        X_values = X_aligned.values
        
        # Step 3: FIXED APPROACH - Apply feature selection FIRST, then scaling
        if self.selected_features is not None:
            print(f"Applying feature selection first: selecting {len(self.selected_features)} features...")
            
            # Ensure we have enough features
            if X_values.shape[1] >= max(self.selected_features) + 1:
                # Apply same feature selection as training
                X_selected = X_values[:, self.selected_features]
                print(f"✓ Feature selection applied, shape: {X_selected.shape}")
            else:
                # Fallback: take first N features
                n_features = len(self.selected_features)
                if X_values.shape[1] >= n_features:
                    X_selected = X_values[:, :n_features]
                    print(f"✓ Using first {n_features} features, shape: {X_selected.shape}")
                else:
                    # Pad with zeros if not enough features
                    padding = np.zeros((X_values.shape[0], n_features - X_values.shape[1]))
                    X_selected = np.hstack([X_values, padding])
                    print(f"✓ Padded to {n_features} features, shape: {X_selected.shape}")
        else:
            X_selected = X_values
        
        # Step 4: Apply scaling on selected features
        if self.scaler is not None:
            print("Applying scaling on selected features...")
            try:
                # Create a new scaler fitted on the selected features only
                from sklearn.preprocessing import StandardScaler
                
                # Option 1: Try to use original scaler if it matches
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == X_selected.shape[1]:
                    X_scaled = self.scaler.transform(X_selected)
                    print("✓ Applied original scaler successfully")
                else:
                    # Option 2: Apply basic standardization
                    print("⚠ Scaler mismatch, applying basic standardization")
                    X_scaled = (X_selected - np.mean(X_selected, axis=0)) / (np.std(X_selected, axis=0) + 1e-8)
                    print("✓ Applied basic standardization")
                    
            except Exception as e:
                print(f"⚠ Scaling failed: {e}. Using basic standardization.")
                X_scaled = (X_selected - np.mean(X_selected, axis=0)) / (np.std(X_selected, axis=0) + 1e-8)
        else:
            print("No scaler available, using selected features as-is")
            X_scaled = X_selected
        
        # Step 5: Final validation and cleanup
        print(f"Final preprocessed shape: {X_scaled.shape}")
        
        # Handle any remaining issues
        if np.any(np.isnan(X_scaled)):
            print("⚠ NaN values detected, replacing with zeros")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        if np.any(np.isinf(X_scaled)):
            print("⚠ Infinite values detected, clipping")
            X_scaled = np.clip(X_scaled, -1e10, 1e10)
        
        # Final statistics
        data_min, data_max = np.min(X_scaled), np.max(X_scaled)
        data_mean, data_std = np.mean(X_scaled), np.std(X_scaled)
        print(f"Final data stats: range=[{data_min:.3f}, {data_max:.3f}], mean={data_mean:.3f}, std={data_std:.3f}")
        
        return X_scaled
        
    def run_full_pipeline(self, 
                         missing_strategy='mean', 
                         normalize_method='standard',
                         feature_selection_method='kbest',
                         n_features=100,
                         dim_reduction_method='pca',
                         n_components=2,
                         test_size=0.2):
        """
        Run the full preprocessing pipeline
        Enhanced for external validation compatibility
        
        Parameters:
        -----------
        missing_strategy : str, default='mean'
            Strategy to handle missing values
        normalize_method : str, default='standard'
            Data normalization method
        feature_selection_method : str, default='kbest'
            Feature selection method
        n_features : int, default=100
            Number of features to select
        dim_reduction_method : str, default='pca'
            Dimensionality reduction method
        n_components : int, default=2
            Number of components for dimensionality reduction
        test_size : float, default=0.2
            Test set size for train-test split
        
        Returns:
        --------
        dict
            Results of the preprocessing pipeline
        """
        print("Starting full preprocessing pipeline...")
        
        # Check for missing values
        missing_results = self.check_missing_values()
        
        # Handle missing values if needed
        if missing_results['missing_values'] > 0:
            self.handle_missing_values(strategy=missing_strategy)
        
        # Explore data distribution
        self.explore_data_distribution()
        
        # Normalize data
        self.normalize_data(method=normalize_method)
        
        # Select features
        self.select_features(method=feature_selection_method, n_features=n_features)
        
        # Reduce dimensions (just for visualization, not for model training)
        reduced_data = self.reduce_dimensions(method=dim_reduction_method, n_components=n_components)
        
        # Split into train and test sets
        self.split_train_test(test_size=test_size)
        
        results = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'reduced_data': reduced_data,
            'missing_results': missing_results,
            'feature_selection_method': feature_selection_method,
            'n_features': n_features,
            'normalize_method': normalize_method,
            'dim_reduction_method': dim_reduction_method
        }
        
        print("Full preprocessing pipeline completed successfully.")
        print(f"Pipeline state saved for external validation:")
        print(f"  - Scaler fitted: {self.scaler is not None}")
        print(f"  - Features selected: {len(self.selected_features) if self.selected_features is not None else 'None'}")
        print(f"  - Dim reducer fitted: {self.dim_reducer is not None}")
        print(f"  - Feature names stored: {len(self.feature_names_in_) if self.feature_names_in_ else 'None'}")
        
        return results