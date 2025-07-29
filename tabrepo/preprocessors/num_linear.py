from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame

import sys
sys.path.append('/home/ubuntu/cat_detection')
from sklearn.linear_model import LinearRegression,LogisticRegression
from autogluon.features.generators import CategoryFeatureGenerator

class NumLinearAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.target_type = target_type
        if target_type == 'regression':
            self.linear_model = LinearRegression()
        else:
            self.linear_model = LogisticRegression()
    
        
    # def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None):
    #     self.interaction_detector.fit(X_out, y)
    #     X_out = self.interaction_detector.transform(X_out)
    #     return super()._fit_transform_custom(X_out=X_out, type_group_map_special=None, y=y)
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X_out = X.copy()
        X_num = X.select_dtypes(include=['number'])
        self.linear_model.fit(X_num, y)
        if self.target_type == 'regression':
            X_out['linear'] = self.linear_model.predict(X_num)
        else:
            X_out['linear'] = self.linear_model.predict_proba(X_num)[:, 1]
        X_out = super().fit_transform(X_out, y, feature_metadata_in=None, **kwargs)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        X_out = X.copy()
        X_num = X.select_dtypes(include=['number'])
        if self.target_type == 'regression':
            X_out['linear'] = self.linear_model.predict(X_num)
        else:
            X_out['linear'] = self.linear_model.predict_proba(X_num)[:, 1]
        X_out = super().transform(X_out)
        return X_out
    
    # def _get_category_feature_generator(self):
    #     return CategoryFeatureGenerator()


class FromCSVCatIntAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

    
        self.interaction_detector = CategoricalInteractionDetector(
            target_type=target_type,
            execution_mode='reduce'
            )

    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        self.interaction_detector.fit(X, y)
        X = self.interaction_detector.transform(X)
        return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        X = self.interaction_detector.transform(X)
        return super().transform(X)

     


# class FromCSVAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
#     def __init__(self):
#         super().__init__()
    
#     def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
#         X = OpenMLTaskWrapper.to_csv_format(X)
#         return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
#     def transform(self, X: DataFrame) -> DataFrame:
#         X = OpenMLTaskWrapper.to_csv_format(X)
#         return super().transform(X)

     