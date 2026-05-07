from __future__ import annotations

from autogluon.common.space import Categorical, Real, Int

from tabarena.benchmark.models.prep_ag.prep_tabicl.prep_tabicl_model import PrepTabICLv2Model
from tabarena.utils.config_utils import PrepConfigGenerator

# name = "TabICLv2"
manual_configs = [
    # Default config with refit after cross-validation.
    {"ag_args_ensemble": {"refit_folds": True}},
]

# Unofficial search space
search_space = {
    # "checkpoint_version": Categorical("tabicl-classifier-v1.1-0506.ckpt", "tabicl-classifier-v1-0208.ckpt"),
    # "norm_methods": Categorical("none", "power", "robust", "quantile_rtdl", ["none", "power"]),
    # # just in case, tuning between TabICL and TabPFN defaults
    # "outlier_threshold": Real(4.0, 12.0),
    # "average_logits": Categorical(False, True),
    # # if average_logits=True this is equivalent to temperature scaling
    # "softmax_temperature": Real(0.999, 1.0),
    "random_state": Int(0,100000, default=0),
    # # Hack to integrate refitting into the search space
    "ag_args_ensemble": Categorical({"refit_folds": True}),
}


prep_manual_configs = [
    {
        "use_random_noise": True,
        # "use_cat_fe": True,
        # "use_groupby": True,
        # "use_rstafc": True,
        # "use_select_spearman": True,
    }]

prep_search_space = {
        # Preprocessing hyperparameters
        "use_random_noise": Categorical(True),
        # "use_cat_fe": Categorical(True),
        # "use_rstafc": Categorical(True),
        # "use_groupby": Categorical(True), 
        # "use_select_spearman": Categorical(True), # Might rather tune no. of features, i.e. in {1000, 1500, 2000}

        # "arithmetic_max_feats": Categorical(2000, 1000, 50),
        # "arithmetic_random_state": Categorical(42,84,168,336,672),

        # "cat_fe_max_feats": Categorical(100, 500),
        # "cat_fe_random_state": Categorical(42,84,168,336,672),

        # "rstafc_n_subsets": Categorical(50,100, 1),
        # "rstafc_random_state": Categorical(42,84,168,336,672),

        # "oofte_random_state": Categorical(42,84,168,336,672),

        # "groupby_max_feats": Categorical(500, 100, 1000), 

        # "spearman_max_feats": Categorical(2000),
}       

gen_tabicl = PrepConfigGenerator(
    model_cls=PrepTabICLv2Model,
    search_space=search_space,
    manual_configs=manual_configs,
    prep_search_space=prep_search_space,
    prep_manual_configs=prep_manual_configs,
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_tabicl.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
