"""兼容层：导出 HeteroFeatureEncoder。"""
from .feature_store import HeteroFeatureEncoder, FeatureDims  # re-export

__all__ = ["HeteroFeatureEncoder", "FeatureDims"]
