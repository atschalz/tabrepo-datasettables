from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import pandas as pd
import numpy as np

from tabarena.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from bencheval.website_format import format_leaderboard

cat_num_datasets = [
    'anneal', 'credit-g', 'qsar-biodeg', 'healthcare_insurance_expenses', 'website_phishing', 'Fitness_Club', 'airfoil_self_noise',
    'Another-Dataset-on-used-Fiat-500', 'MIC', 'Is-this-a-good-customer', 'Marketing_Campaign', 'seismic-bumps', 'students_dropout_and_academic_success',
    'churn', 'coil2000_insurance_policies', 'Bank_Customer_Churn', 'E-CommereShippingData', 'online_shoppers_intention', 'in_vehicle_coupon_recommendation',
    'HR_Analytics_Job_Change_of_Data_Scientists', 'credit_card_clients_default', 'bank-marketing', 'Food_Delivery_Time', 
    # 'kddcup09_appetency',
    'diamonds', 'Diabetes130US', 'SDSS17', 'customer_satisfaction_in_airline'
    ]



if __name__ == '__main__':
    # exp_name = 'icml_experimental'
    # exp_name = 'neighbor_interaction'
    # exp_name = 'test_newpreps'
    # exp_name = 'current_with_rstafc'
    # exp_name = 'current_with_linear_feature'
    # exp_name = 'groupby_experiments'
    # exp_name = 'new_search_space'
    # exp_name = 'icml_ablation'
    exp_name = "icml_final"
    exp_name = 'NEW_TEST_3'

    expname = f'/ceph/atschalz/auto_prep/experiments/{exp_name}'  # folder location to save all experiment artifacts
    eval_dir = f'/ceph/atschalz/auto_prep/eval/{exp_name}'

    ignore_cache = False   # set to True to overwrite existing caches and re-run experiments from scratch
    debug = False
    model_name = "EBM" # "PrepTABICL-V2_noArithmetic" # "TABICL-V2" #"GBM-ablation"  #"OpenFELGBModel" # 'AutoFeatLinearModel'  # 'LR', 'GBM' 'CAT', 'TABM', or 'REALTABPFN-V2.5' or 'GBM-ablation'
    start_config = 0
    n_configs = 1
    n_datasets = 2
    datasets = None # ["Amazon_employee_access", "diamonds", "kddcup09_appetency", "SDSS17", "APSFailure", "polish_companies_bankruptcy"] 
    filter_datasets = None #['Bioresponse', 'hiva_agnostic', 'splice', 'QSAR_fish_toxicity', 'anneal', 'MIC', 'QSAR-TID-11']
    
    #["polish_companies_bankruptcy"] # Amazon_employee_access'] #None #["superconductivity"] #["credit-g", "website_phishing", "maternal_health_risk", "Fitness_Club", "Another-Dataset-on-used-Fiat-500", "bank-marketing"]
                #"maternal_health_risk", 
                # "houses", "diamonds", 
                # "E-CommereShippingData", "airfoil_self_noise",
                # "Another-Dataset-on-used-Fiat-500", "Fitness_Club", "bank-marketing",
                # "GiveMeSomeCredit", "physiochemical_protein",  
                # "website_phishing", "coil2000_insurance_policies",
                # "Diabetes130US", 
                # "kddcup09_appetency", # "SDSS17", "APSFailure"
                # ]
    # datasets = ["credit-g"] #['miami_housing'] #[
        # 'blood-transfusion-service-center', 'diabetes', 'Fitness_Club', 'Another-Dataset-on-used-Fiat-500', 'maternal_health_risk', 'NATICUSdroid', 
                # 'bank-marketing',
                # # "GiveMeSomeCredit",
                # # "physiochemical_protein",  
                # 'website_phishing', #'coil2000_insurance_policies', 'customer_satisfaction_in_airline'] # None #['physiochemical_protein', "superconductivity", "concrete_compressive_strength"
                # ]  
    raise_on_failure = True
    fold = 0
    adjust_time_limit = 7200
    
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Sample for a quick demo
    # datasets = ["anneal", "credit-g", "diabetes"]  # 
    if datasets is None:
        datasets = task_metadata.sort_values('n_samples_train_per_fold').name.tolist()
        if n_datasets is not None:
            datasets = datasets[:n_datasets]

    if filter_datasets is not None:
        datasets = [d for d in datasets if d not in filter_datasets]

                # ]
    folds = [fold]

    if model_name=='GBM':
        from tabarena.models.prep_lgb.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='GBM-refit':
        from tabarena.models.prep_lgb.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
        for i in range(len(methods)):
            methods[i].method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["refit_folds"] = True
    elif model_name=='REALTABPFN-V2.5':
        from tabarena.models.prep_tabpfnv2_5.generate import gen_realtabpfnv25
        methods = gen_realtabpfnv25.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='EBM':
        from tabarena.models.prep_ebm.generate import gen_prep_ebm
        methods = gen_prep_ebm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='BaseLGB':
        from tabarena.models.lightgbm.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='BaseLR':
        from tabarena.models.lr.generate import gen_linear
        methods = gen_linear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='LR':
        from tabarena.models.prep_lr.generate import gen_linear
        methods = gen_linear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='TABM':
        from tabarena.models.prep_tabm.generate import gen_tabm
        methods = gen_tabm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='CAT':
        from tabarena.models.prep_catboost.generate import gen_catboost
        methods = gen_catboost.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='REALMLP':
        from tabarena.models.prep_realmlp.generate import gen_realmlp
        methods = gen_realmlp.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "AutoFeatLinearModel":
        from tabarena.models.autofeat.generate import gen_autofeatlinear
        methods = gen_autofeatlinear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "OpenFELGBModel":
        from tabarena.models.openfe.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "GBM-ablation":
        from tabarena.models.prep_lgb.generate_ablation import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "TABICL-V2":
        from tabarena.models.tabiclv2.generate import gen_tabicl
        methods = gen_tabicl.generate_all_bag_experiments(num_random_configs=1)
    elif model_name == "PrepTABICL-V2":
        from tabarena.models.prep_tabiclv2.generate import gen_tabicl
        methods = gen_tabicl.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "PrepTABICL-V2_noArithmetic":
        from tabarena.models.prep_tabiclv2.generate_2 import gen_tabicl
        methods = gen_tabicl.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "TABICL-V2-noise":
        from tabarena.models.prep_tabiclv2.generate_noise import gen_tabicl
        methods = gen_tabicl.generate_all_bag_experiments(num_random_configs=1)    
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


    for i in range(len(methods)):
        methods[i].method_kwargs['model_hyperparameters']['ag_args_ensemble']['model_random_seed'] = 0
        methods[i].method_kwargs['model_hyperparameters']['ag_args_ensemble']['vary_seed_across_folds'] = True

        if model_name == "GBM-ablation":
            methods[i].name = methods[i].name.replace("prep_LightGBM", "prep_LightGBM-ablation")
        # if model_name == "GBM-ablation":
        #     methods[i].name = methods[i].name.replace("prep_LightGBM", "prep_LightGBM-ablation")
        # if model_name == "GBM-ablation":
        #     methods[i].name = methods[i].name.replace("prep_LightGBM", "prep_LightGBM-ablation")

        # if adjust_time_limit is not None:
        #     methods[i].method_kwargs['time_limit'] = adjust_time_limit

        # methods[i].method_kwargs['model_hyperparameters']['ag.prep_params'][0] = [
        #     ["RandomSubsetTAFC", {
        #         # 'n_subsets': 50, 
        #         # 'round_numerical': 2, 
        #         # 'subset_size': None, 
        #         # 'min_subset_size': 4, 
        #         # 'max_subset_size': None,
        #         # 'only_cat': True,
        #         'n_subsets': 50, 
        #         'round_numerical': 0, 
        #         'subset_size': None, 
        #         'min_subset_size': 2, 
        #         'max_subset_size': None,
        #         'only_cat': False,
        #         }]]
        # methods[i].method_kwargs['model_hyperparameters']['ag.prep_params'][0] = [["RandomSubsetTAFC", {'n_subsets': 50,  # Numeric features
        #                                                                                                 'round_numerical': 2, 
        #                                                                                                 'subset_size': None, 
        #                                                                                                 'min_subset_size': 1, 
        #                                                                                                 'max_subset_size': 4,
        #                                                                                                 'only_cat': False,
        #                                                                                                 }]]
        # try:
        #     methods[i].method_kwargs['model_hyperparameters']['ag.prep_params'][0][1] = [
        #         "RandomSubsetTAFC", {
        #             # 'n_subsets': 50, 
        #             # 'round_numerical': 2, 
        #             # 'subset_size': None, 
        #             # 'min_subset_size': 1, 
        #             # 'max_subset_size': 4,
        #             # 'only_cat': False,
        #     }]
        # except:
        #     continue

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    if debug == True:
        for i in range(len(methods)):
            methods[i].method_kwargs['model_hyperparameters']['ag_args_ensemble']["fold_fitting_strategy"] = "sequential_local"
    methods = methods[start_config:n_configs+start_config]

    for m in methods:
        if 'ag.prep_params' in m.method_kwargs['model_hyperparameters']:
            # print(f"Method: {m.name}, Hyperparameters: {[(k,v) for k, v in m.method_kwargs['model_hyperparameters'].items() if k in ['C', 'C_scale', 'scaler', 'penalty', 'proc.skew_threshold', 'proc.impute_strategy']]}")
            print(f"Method:{m.name}, Prep Params: {m.method_kwargs['model_hyperparameters']['ag.prep_params']}")

#########################
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
        raise_on_failure=raise_on_failure,
    )

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_misc import gbm_aio_0808_metadata
    extra_methods = [gbm_aio_0808_metadata]

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
        use_model_results=True,  # If False: Will instead use the ensemble/HPO results
        new_result_prefix="Demo_",
        tabarena_context_kwargs={'extra_methods': extra_methods},
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
#########################
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    tabarena_context = TabArenaContext(extra_methods=extra_methods)
    hpo_results = pd.concat([tabarena_context.load_hpo_results(method=m) for m in tabarena_context.methods if m not in ['AutoGluon_v140_bq_4h8c', 'AutoGluon_v140_eq_4h8c', 'AutoGluon_v150_eq_4h8c']])
    hpo_results = hpo_results.loc[hpo_results.fold==fold].reset_index(drop=True)
    
    for dat in end_to_end_results.hpo_results.dataset.unique():
        hpo_results_dat = hpo_results.loc[hpo_results.dataset==dat]
        best_model = hpo_results_dat.loc[hpo_results_dat['metric_error'].idxmin(),'config_type']

        print('---' * 10 + dat + '---' * 10)
        print(f"New Model Results: {end_to_end_results.model_results.loc[end_to_end_results.model_results.dataset==dat, ['fold', 'method', 'metric_error', 'metric_error_val']]}")
        print('---' * 20)
        print(f"New HPO Results: {end_to_end_results.hpo_results.loc[end_to_end_results.hpo_results.dataset==dat, ['fold', 'method', 'metric_error', 'metric_error_val']]}")
        print('---' * 20)
        print(f"Old {model_name} Results: {hpo_results_dat.loc[hpo_results_dat.method.apply(lambda x: model_name in x), ['method', 'metric_error']]}")
        print('---' * 20)
        print(f"Best Model Results: {hpo_results_dat.loc[hpo_results_dat.method.apply(lambda x: best_model in x), ['method', 'metric_error']]}")
    


