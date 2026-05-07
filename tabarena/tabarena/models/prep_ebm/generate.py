from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.prep_ag.prep_ebm.prep_ebm_model import PrepEBMModel
from tabarena.models.ebm.generate import manual_configs as base_manual_configs
from tabarena.models.ebm.generate import search_space as base_search_space
from tabarena.utils.config_utils import PrepConfigGenerator

name = "PrepEBM"
manual_configs = [dict(config) for config in base_manual_configs] or [{}]

# Base EBM hyperparameters are defined in tabarena.models.ebm.generate.
# Here we extend that space with the standard TabArena preprocessing knobs.
search_space = dict(base_search_space)

prep_manual_configs = [
    {
        "use_arithmetic_preprocessor": True,
        "use_cat_fe": True,
        "use_rstafc": True,
        "use_groupby": True,
        "use_select_spearman": True,
    }
]

prep_search_space = {
    "use_arithmetic_preprocessor": Categorical(True),
    "use_cat_fe": Categorical(True),
    "use_rstafc": Categorical(True),
    "use_groupby": Categorical(True),
    "use_select_spearman": Categorical(True),
    "arithmetic_max_feats": Categorical(2000, 1000),
    "arithmetic_random_state": Categorical(42, 84, 168, 336, 672),
    "cat_fe_max_feats": Categorical(100, 500),
    "cat_fe_random_state": Categorical(42, 84, 168, 336, 672),
    "rstafc_n_subsets": Categorical(50, 100, 1),
    "rstafc_random_state": Categorical(42, 84, 168, 336, 672),
    "oofte_random_state": Categorical(42, 84, 168, 336, 672),
    "groupby_max_feats": Categorical(500, 100, 1000),
    "spearman_max_feats": Categorical(2000),
}

gen_prep_ebm = PrepConfigGenerator(
    name=name,
    model_cls=PrepEBMModel,
    search_space=search_space,
    manual_configs=manual_configs,
    prep_search_space=prep_search_space,
    prep_manual_configs=prep_manual_configs,
)


def generate_configs_prep_ebm(num_random_configs=200):
    return gen_prep_ebm.generate_all_configs_lst(num_random_configs=num_random_configs)


if __name__ == "__main__":
    print(generate_configs_prep_ebm(3))
