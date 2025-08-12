from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame
from tabprep.cat_engineering import CategoricalFeatureEngineer
from autogluon.features.generators import CategoryFeatureGenerator

class CatEngineeringAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.detector = CategoricalFeatureEngineer(        
                target_type=target_type,
                execution_mode='all',
                max_order=3,
                use_mvp=True, 
            )
        
    # def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None):
    #     self.interaction_detector.fit(X_out, y)
    #     X_out = self.interaction_detector.transform(X_out)
    #     return super()._fit_transform_custom(X_out=X_out, type_group_map_special=None, y=y)
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        self.detector.fit(X, y)
        X_out = self.detector.transform(X)
        X_out = super().fit_transform(X_out, y, feature_metadata_in=None, **kwargs)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        X_out = self.detector.transform(X)
        X_out = super().transform(X_out)
        return X_out
    
    # def _get_category_feature_generator(self):
    #     return CategoryFeatureGenerator()

