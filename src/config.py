from dataclasses import dataclass, field
from typing import List

@dataclass
class ProjectConfig:
    """Configuration for the Credit Risk Model pipeline."""
    raw_data_path: str = "data/raw/data.csv"
    processed_data_path: str = "data/processed/train_labeled.csv"
    model_save_path: str = "src/api/model.pkl"
    mlflow_tracking_uri: str = "file://mlruns"
    
    n_clusters: int = 3
    random_state: int = 42
    
    cat_cols: List = field(default_factory=lambda:)
    cols_to_drop: List = field(default_factory=lambda:)
    
    rf_n_estimators: int = 50
    rf_max_depth: int = 10
    lr_max_iter: int = 1000

config = ProjectConfig()
