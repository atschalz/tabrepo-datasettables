from tabrepo.nips2025_utils.artifacts import tabarena_method_metadata_map
from tabrepo.nips2025_utils.artifacts.tabarena51_artifact_loader import TabArena51ArtifactLoader
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == '__main__':
    loader = TabArena51ArtifactLoader()
    # tabarena_context = TabArenaContext()
    # 
    # methods = list(tabarena_method_metadata_map.keys())
    # 
    # loader.download_raw()  # download raw data for all methods, very large (1 TB)
    # loader.download_processed()  # download processed data, much smaller (100 GB)
    # loader.download_results()  # download results data (<100 MB)
    # 
    # # loader._download_raw_method("TabM")  # raw for just a single method
    loader._download_processed_method("RandomForest")  # processed for just a single method
    # # loader._download_results_method("TabM")  # results for just a single method
    # 
    # for method in methods:
    #     path_to_repo_artifact = tabarena_context.generate_repo(method=method)  # convert raw to processed
    # 
    # for method in methods:
    #     results, config_results = tabarena_context.simulate_repo(method=method)  # convert processed to results