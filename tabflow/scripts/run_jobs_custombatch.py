import os
import sys
from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
# from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
"""
bash ./tabrepo/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
 
aws s3 cp --recursive "s3://prateek-ag/neerick-exp-3/" ../data/neerick-exp-3/ --exclude "*.log"
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs 

"""

docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/andrej-test:test-csv"
# docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/pmdesai:mlflow-tabrepo"
sagemaker_role = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"

##########################
# def get_benchmark_metadata(benchmark: str, subset: list[str]=None) -> pd.DataFrame:
#     if benchmark == "TabZilla":
#         from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
#         tids, dids = get_benchmark_dataIDs("TabZilla")
#         metadata = get_metadata_df(tids, dids)    
#     elif benchmark == "Grinsztajn":
#         from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
#         tids, dids = get_benchmark_dataIDs("Grinsztajn")
#         metadata = get_metadata_df(tids, dids)    
#     elif benchmark == "TabArena":
#         metadata = load_task_metadata()        
#     else:
#         raise ValueError(f"Unknown benchmark: {benchmark}")

#     if subset is not None:
#         metadata = metadata[metadata["name"].isin(subset)]

#     return metadata

# def get_model(model_name: str) -> AbstractModel:
#     return tabrepo_model_register.key_to_cls(model_name)  
##########################

# batch_sizes = {'APSFailure': 4, 
#  'Amazon_employee_access': 4, # CHANGED FOR CATINT
#  'Another-Dataset-on-used-Fiat-500': 256,
#  'Bank_Customer_Churn': 256,
#  'Bioresponse': 32,
#  'Diabetes130US': 8, # CHANGED FOR CATINT
#  'E-CommereShippingData': 256,
#  'Fitness_Club': 256,
#  'Food_Delivery_Time': 128,
#  'GiveMeSomeCredit': 128,
#  'HR_Analytics_Job_Change_of_Data_Scientists': 256,
#  'Is-this-a-good-customer': 256,
#  'MIC': 256,
#  'Marketing_Campaign': 256,
#  'NATICUSdroid': 128,
#  'QSAR-TID-11': 128,
#  'QSAR_fish_toxicity': 256,
#  'SDSS17': 4,
#  'airfoil_self_noise': 256,
#  'anneal': 256,
#  'bank-marketing': 256,
#  'blood-transfusion-service-center': 256,
#  'churn': 256,
#  'coil2000_insurance_policies': 256,
#  'concrete_compressive_strength': 256,
#  'credit-g': 256,
#  'credit_card_clients_default': 128,
#  'customer_satisfaction_in_airline': 8,
#  'diabetes': 256,
#  'diamonds': 128,
#  'hazelnut-spread-contaminant-detection': 256,
#  'healthcare_insurance_expenses': 256,
#  'heloc': 256,
#  'hiva_agnostic': 128,
#  'houses': 32, #Original value was 128, but reduced to 64 as it took too long with arrithmetic interactions
#  'in_vehicle_coupon_recommendation': 8, # CHANGED FOR CATINT
#  'jm1': 64,
#  'kddcup09_appetency': 4, # CHANGED FOR CATINT
#  'maternal_health_risk': 256,
#  'miami_housing': 128,
#  'online_shoppers_intention': 256,
#  'physiochemical_protein': 4,
#  'polish_companies_bankruptcy': 128,
#  'qsar-biodeg': 256,
#  'seismic-bumps': 256,
#  'splice': 128,
#  'students_dropout_and_academic_success': 128,
#  'superconductivity': 32, # Might better reduce to 4 for ftd
#  'taiwanese_bankruptcy_prediction': 128,
#  'website_phishing': 256,
#  'wine_quality': 64}

batch_sizes = {'APSFailure': 4, 'Amazon_employee_access': 25, 'Another-Dataset-on-used-Fiat-500': 48, 
               'Bank_Customer_Churn': 89, 'Bioresponse': 14, 'Diabetes130US': 5, 
               'E-CommereShippingData': 20, 'Fitness_Club': 102, 'Food_Delivery_Time': 6, 
               'GiveMeSomeCredit': 3, 'HR_Analytics_Job_Change_of_Data_Scientists': 10, 
               'Is-this-a-good-customer': 76, 'MIC': 15, 'Marketing_Campaign': 15, 'NATICUSdroid': 305, 
               'QSAR-TID-11': 161, 'QSAR_fish_toxicity': 70, 'SDSS17': 2, 'airfoil_self_noise': 24, 
               'anneal': 33, 'bank-marketing': 17, 'blood-transfusion-service-center': 366, 'churn': 13, 
               'coil2000_insurance_policies': 30, 'concrete_compressive_strength': 4, 'credit-g': 35, 
               'credit_card_clients_default': 7, 'customer_satisfaction_in_airline': 3, 'diabetes': 24, 
               'diamonds': 4, 'hazelnut-spread-contaminant-detection': 16, 'healthcare_insurance_expenses': 446, 
               'heloc': 13, 'hiva_agnostic': 68, 'houses': 3, 'in_vehicle_coupon_recommendation': 47, 'jm1': 11, 
               'kddcup09_appetency': 5, 'maternal_health_risk': 56, 'miami_housing': 2, 'online_shoppers_intention': 11, 
               'physiochemical_protein': 0, 'polish_companies_bankruptcy': 15, 'qsar-biodeg': 14, 'seismic-bumps': 21,
               'splice': 35, 'students_dropout_and_academic_success': 9, 'superconductivity': 3, 
               'taiwanese_bankruptcy_prediction': 18, 'website_phishing': 470, 'wine_quality': 4
               
               }

batch_sizes = {'APSFailure': 8, 'Amazon_employee_access': 50, 'Another-Dataset-on-used-Fiat-500': 97, 
               'Bank_Customer_Churn': 178, 'Bioresponse': 29, 'Diabetes130US': 11, 
               'E-CommereShippingData': 40, 'Fitness_Club': 204, 'Food_Delivery_Time': 12, 
               'GiveMeSomeCredit': 7, 'HR_Analytics_Job_Change_of_Data_Scientists': 21, 
               'Is-this-a-good-customer': 152, 'MIC': 31, 'Marketing_Campaign': 31, 'NATICUSdroid': 610, 
               'QSAR-TID-11': 322, 'QSAR_fish_toxicity': 140, 'SDSS17': 4, 'airfoil_self_noise': 49, 
               'anneal': 66, 'bank-marketing': 35, 'blood-transfusion-service-center': 733, 'churn': 27, 
               'coil2000_insurance_policies': 61, 'concrete_compressive_strength': 8, 'credit-g': 70, 
               'credit_card_clients_default': 14, 'customer_satisfaction_in_airline': 6, 'diabetes': 49, 
               'diamonds': 8, 'hazelnut-spread-contaminant-detection': 32, 'healthcare_insurance_expenses': 892, 
               'heloc': 26, 'hiva_agnostic': 136, 'houses': 6, 'in_vehicle_coupon_recommendation': 95, 'jm1': 22, 
               'kddcup09_appetency': 11, 'maternal_health_risk': 112, 'miami_housing': 4, 
               'online_shoppers_intention': 23, 'physiochemical_protein': 1, 'polish_companies_bankruptcy': 30, 
               'qsar-biodeg': 28, 'seismic-bumps': 42, 'splice': 71, 'students_dropout_and_academic_success': 18, 
               'superconductivity': 7, 'taiwanese_bankruptcy_prediction': 36, 'website_phishing': 941,
               'wine_quality': 8
                 }

if __name__ == "__main__":
    task_metadata = load_task_metadata()

    experiment_name = "cb_default"
    max_concurrent_jobs = 10000
    batch_size = 16
    batch_scaling_factor = 16
    min_per_batch = 1
    wait = False
    s3_bucket = "prateek-ag"
    region_name = "us-west-2"
    instance_type = "ml.m6i.2xlarge" #"ml.g6.2xlarge" 
    filter_cache = True

    methods_file = f"configs_all_{experiment_name}.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])

    # raise AssertionError("No datasets found")
   
    # toy run
    # done = [#'Fitness_Club', 'Is-this-a-good-customer', 'Marketing_Campaign', 'credit-g', 'diabetes', 'blood-transfusion-service-center',
    #       'anneal', 'airfoil_self_noise', 'website_phishing'
    #     ]
    
    # done = ['Amazon_employee_access', 
    #    'Bank_Customer_Churn', 'Diabetes130US', 'Fitness_Club',
    #    'Is-this-a-good-customer', 'Marketing_Campaign', 'airfoil_self_noise',
    #    'anneal', 'bank-marketing', 'blood-transfusion-service-center', 'churn',
    #    'coil2000_insurance_policies', 'concrete_compressive_strength',
    #    'credit-g', 'diabetes', 'healthcare_insurance_expenses',
    #    'in_vehicle_coupon_recommendation', 'qsar-biodeg', 'seismic-bumps',
    #    'website_phishing']
    # in_progress =  ['wine_quality', 'houses', 'jm1', 'coil2000_insurance_policies', 'blood-transfusion-service-center', 'physiochemical_protein', 'taiwanese_bankruptcy_prediction', 'Another-Dataset-on-used-Fiat-500']
    # lin_datasets = ['blood-transfusion-service-center', 'Amazon_employee_access', 'diabetes', 'Fitness_Club', 'credit-g', 'seismic-bumps', 'Marketing_Campaign', 'Diabetes130US', 'qsar-biodeg','Is-this-a-good-customer']
    # datasets = [dataset for dataset in datasets if dataset not in done+lin_datasets+in_progress]
    folds = list(range(3))
    repeats = list(range(3))
    methods = methods[:1]

    print(datasets)
    print()

    job_manager = JobManager(
        experiment_name=experiment_name,
        task_metadata=task_metadata,
        methods_file=methods_file,
        max_concurrent_jobs=max_concurrent_jobs,
        s3_bucket=s3_bucket,
        wait=wait,
        instance_type=instance_type,
        batch_size=batch_size,
        sagemaker_role=sagemaker_role,
        docker_image_uri=docker_image_uri,
    )

    tasks_batch_combined = []
    for dataset_name in datasets:
        tasks_dense = job_manager.get_tasks_dense(
            datasets=[dataset_name],
            repeats=repeats,
            folds=folds,
            methods=methods,
        )
        if filter_cache:
            tasks_dense = job_manager.filter_to_only_uncached_tasks(tasks=tasks_dense, verbose=True)

        tasks_batch = job_manager.batch_tasks(tasks=tasks_dense, batch_size=max([min_per_batch, int(batch_sizes[dataset_name]/batch_scaling_factor)]))
        tasks_batch_combined.extend(tasks_batch)

    uncached_tasks_batched = tasks_batch_combined

    # raise AssertionError

    job_manager.run_tasks_batched(task_batch_lst=uncached_tasks_batched, check_cache=False)