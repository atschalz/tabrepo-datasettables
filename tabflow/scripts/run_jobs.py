import sys
sys.path.append('/home/ubuntu/cat_detection/')
from tabflow.cli.launch_jobs import JobManager
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabprep.utils import get_benchmark_dataIDs, get_metadata_df


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

if __name__ == "__main__":
    task_metadata = load_task_metadata(subset="TabPFNv2")

    experiment_name = "test_csv2"
    max_concurrent_jobs = 100
    batch_size = 8
    wait = True
    s3_bucket = "prateek-ag"
    region_name = "us-west-2"
    instance_type = "ml.m6i.2xlarge" #"ml.g6.2xlarge" 

    methods_file = "configs_all_new.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)

    datasets = list(task_metadata["name"])

    # raise AssertionError("No datasets found")

    # toy run
    datasets = datasets[:1]
    folds = [0]
    repeats = [0]
    # methods = methods[:1]

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

    tasks_dense = job_manager.get_tasks_dense(
        datasets=datasets,
        repeats=repeats,
        folds=folds,
        methods=methods,
    )
    # uncached_tasks = job_manager.filter_to_only_uncached_tasks(tasks=tasks_dense, verbose=True)

    tasks_batch = job_manager.batch_tasks(tasks=tasks_dense, batch_size=batch_size)
    tasks_batch_combined = tasks_batch

    uncached_tasks_batched = tasks_batch_combined

    # raise AssertionError

    job_manager.run_tasks_batched(task_batch_lst=uncached_tasks_batched, check_cache=False)