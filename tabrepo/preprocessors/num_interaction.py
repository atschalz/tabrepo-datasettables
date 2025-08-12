from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame
from tabprep.num_interaction import NumericalInteractionDetector
from autogluon.features.generators import CategoryFeatureGenerator

class NumIntAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.detector = NumericalInteractionDetector(
                target_type=target_type, 
                max_order=2, num_operations='all',
                use_mvp=False,
                corr_thresh=.95,
                select_n_candidates=2000,

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

     