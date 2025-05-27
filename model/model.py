import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MLModel:
    """
    Model wrapper class that encapsulates model-specific logic
    and provides a consistent interface for the training script.
    """
    
    def __init__(self, model_type="random_forest", **kwargs):
        """
        Initialize model based on specified type
        
        Args:
            model_type (str): Type of model to create
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.create_model(**kwargs)
    
    def create_model(self, **kwargs):
        """Create underlying model instance"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return self.model
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Return feature importances if supported by model"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_model_params(self):
        """Get model parameters"""
        return self.model.get_params()
    
    def save(self, path):
        """Save model to disk - defer to calling code for persistence"""
        pass
    
    @classmethod
    def load(cls, path):
        """Load model from disk - defer to calling code for loading"""
        pass

class CustomModel(BaseEstimator, ClassifierMixin):
    """
    Example of a custom model implementation that follows scikit-learn conventions.
    Useful when you need to implement custom models or ensembles.
    """
    
    def __init__(self, base_estimator=None, feature_transformer=None):
        self.base_estimator = base_estimator or RandomForestClassifier()
        self.feature_transformer = feature_transformer
        
    def fit(self, X, y):
        """Fit model to training data"""
        if self.feature_transformer:
            X = self.feature_transformer.fit_transform(X)
        
        self.base_estimator.fit(X, y)
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        """Make predictions"""
        if self.feature_transformer:
            X = self.feature_transformer.transform(X)
        
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        if self.feature_transformer:
            X = self.feature_transformer.transform(X)
            
        return self.base_estimator.predict_proba(X)
