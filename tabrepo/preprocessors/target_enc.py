from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame

import sys
sys.path.append('/home/ubuntu/cat_detection')
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from autogluon.features.generators import CategoryFeatureGenerator

class CatTEAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.detector = LeaveOneOutEncoder(
           handle_unknown='value',
           handle_missing='indicator',
            )
        
    # def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None):
    #     self.interaction_detector.fit(X_out, y)
    #     X_out = self.interaction_detector.transform(X_out)
    #     return super()._fit_transform_custom(X_out=X_out, type_group_map_special=None, y=y)
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        # X_out = X.copy()
        # # self.detector.fit(X, y)
        # for col in X.columns:
        #     X_out[col+'_num'] = X[col].astype(float)
        # # X_out = self.detector.transform(X)
        
        X_out = X[['MGR_ID', "RESOURCE"]].copy()
        X_out = super().fit_transform(X_out, y, feature_metadata_in=None, **kwargs)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        # X_out = X.copy()
        # # X_out = self.detector.transform(X)
        # for col in X.columns:
        #     X_out[col+'_num'] = X[col].astype(float)
        X_out = X[['MGR_ID', "RESOURCE"]].copy()
        X_out = super().transform(X_out)
        return X_out
    
    # def _get_category_feature_generator(self):
    #     return CategoryFeatureGenerator()
