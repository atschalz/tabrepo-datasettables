from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame, Series
from autogluon.features.generators import CategoryFeatureGenerator

import sys
sys.path.append('/home/ubuntu/cat_detection')
from tabprep.irrelevant_cat_detection import IrrelevantCatDetector


# class CategoryFeatureGeneratorWithCatRes(CategoryFeatureGenerator):
#     """
#     A custom feature generator that extends the CategoryFeatureGenerator to include cat resolution detection.
#     This generator is used to handle categorical features with potential resolution issues.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.detector = CatResolutionDetector(
#             target_type='classification',  # or 'regression' based on your use case
#             operation_mode='sequential',  # 'sequential' or 'full'
#             max_to_test=100,  # maximum number of unique values to test
#             drop_unique=False, cv_method='regular'
#         )
#     def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> (DataFrame, dict):
#         self.detector.fit(X, y)
#         X_out = self.detector.transform(X)
#         return super()._fit_transform(X_out, **kwargs)
        

class CatIrrelevantAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.detector = IrrelevantCatDetector(
                target_type=target_type, 
                method='CV',  # 'CV' or 'LOO'
                n_folds=5, cv_method='regular',
                use_mvp=False,
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

class FromCSVCatIrrelevantAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detector = IrrelevantCatDetector(
            target_type=target_type,
            method='LOO'
            )

    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        self.detector.fit(X, y)
        X = self.detector.transform(X)
        return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        X = self.detector.transform(X)
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

     