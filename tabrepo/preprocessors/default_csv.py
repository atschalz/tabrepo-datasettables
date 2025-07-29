from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame

class FromCSVAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        enable_numeric_features=True,
        enable_categorical_features=True,
        enable_datetime_features=True,
        enable_text_special_features=True,
        enable_text_ngram_features=True,
        enable_raw_text_features=False,
        enable_vision_features=True,
        vectorizer=None,
        text_ngram_params=None,
        **kwargs,
                 ):
        super().__init__(
            enable_numeric_features=enable_numeric_features,
            enable_categorical_features=enable_categorical_features,
            enable_datetime_features=enable_datetime_features,
            enable_text_special_features=enable_text_special_features,
            enable_text_ngram_features=enable_text_ngram_features,
            enable_raw_text_features=enable_raw_text_features,
            enable_vision_features=enable_vision_features,
            vectorizer=vectorizer,
            text_ngram_params=text_ngram_params,
            **kwargs,
        )
        # self.preprocessing_adapted = False
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X_out = OpenMLTaskWrapper.to_csv_format(X)
        # if (X_out != X).any().any():
        #     self.preprocessing_adapted = True
        # else:
        #     raise ValueError("The preprocessing did not change the DataFrame.")
        X_out = super().fit_transform(X_out, y, feature_metadata_in, **kwargs)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        # if self.preprocessing_adapted:
        X = OpenMLTaskWrapper.to_csv_format(X)
        return super().transform(X)

     