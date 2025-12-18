import pandas as pd
import asreview
from asreview.learner import ActiveLearningCycle
from asreview.models.queriers import TopDown
from asreview.models.stoppers import IsFittable
from argparse import ArgumentParser
from tqdm import tqdm
import os
from utils import model_configurations, n_query_extreme, get_abstract_length

def main():
    # Define the argument parser
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="improve", help="The benchmark to use (improve or synergy)")
    parser.add_argument("--dataset", type=str, default="macular_degeneration", help="The dataset to use")
    parser.add_argument("--n_pos_priors", type=int, default=1, help="The number of positive priors")
    parser.add_argument("--n_neg_priors", type=int, default=1, help="The number of negative priors")
    parser.add_argument("--initial_random_seed", type=int, default=42, help="The initial random seed for reproducibility")
    parser.add_argument("--model", type=str, default="elas_u4", help="The ASReview model to use")
    parser.add_argument("--n_random_cycles", type=int, default=10, help="The number of random cycles to run")
    parser.add_argument("--transformer_batch_size", type=int, default=32, help="Batch size for transformer encoding")
    parser.add_argument("--oa_status", type=str, default=None, choices=[None, "True", "False"], help="Filter by open access status (synergy only): True, False, or None for no filter")
    parser.add_argument("--abstract_min_length", type=int, default=None, help="Minimum abstract length (words for space-separated languages, characters otherwise)")
    args = parser.parse_args()
    benchmark = args.benchmark
    dataset = args.dataset
    n_pos_priors = args.n_pos_priors
    n_neg_priors = args.n_neg_priors
    initial_seed = args.initial_random_seed
    model = args.model
    n_random_cycles = args.n_random_cycles
    transformer_batch_size = args.transformer_batch_size
    oa_status = args.oa_status
    abstract_min_length = args.abstract_min_length

    # Determine OA status folder name
    if oa_status is None:
        oa_folder = "all"
    elif oa_status == "True":
        oa_folder = "open"
    else:  # "False"
        oa_folder = "closed"

    # Check if simulation is already done
    save_folder_path = f"./results/{benchmark}/{dataset}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)/{oa_folder}"
    abs_suffix = f"abs{abstract_min_length}" if abstract_min_length is not None else "abs_all"
    df_results_path = f"{save_folder_path}/{model}_{abs_suffix}_results.csv"
    if os.path.exists(df_results_path):
        print(f"Simulation already done. Results saved in {df_results_path}")
        return

    # Define global variables
    global model_configurations
    valid_models = list(model_configurations.keys())

    # Read the data file
    if benchmark == "synergy":
        file_path = f"./data/{benchmark}/{dataset}/labels.csv"
        df_full = pd.read_csv(file_path)
    else:  # improve
        file_path = f"./data/{benchmark}/{dataset}/{dataset}.xlsx"
        df_full = pd.read_excel(file_path)
    
    # Store original indices before any filtering
    df_full = df_full.reset_index(drop=True)
    
    # Define model configurations   
    try:
        model_config = model_configurations[args.model]
    except KeyError:
        raise ValueError(f"Invalid model name. Choose from {valid_models}")
    
    # Check if embeddings are needed and available for h3 or l2 models
    skip_feature_extraction = False
    X_full = df_full[["title", "abstract"]]
    
    if model in ["elas_h3", "elas_l2"]:
        # Determine embedding file path - store in embeddings directory (without filters)
        embedding_path = f"./embeddings/{benchmark}/{dataset}/{model}_embeddings.parquet"
        
        if os.path.exists(embedding_path):
            # Load pre-computed embeddings for all records
            print(f"Loading pre-computed embeddings from {embedding_path}")
            embeddings_df = pd.read_parquet(embedding_path)
            X_full = embeddings_df
            skip_feature_extraction = True
        else:
            # Compute embeddings for ALL records and save them
            print(f"Computing embeddings for {model} for all {len(df_full)} records. This may take a while...")
            from sentence_transformers import SentenceTransformer
            
            # Select the appropriate model
            if model == "elas_h3":
                transformer_model = "mixedbread-ai/mxbai-embed-large-v1"
            else:  # elas_l2
                transformer_model = "intfloat/multilingual-e5-large"
            
            # Load the sentence transformer model
            encoder = SentenceTransformer(transformer_model)
            
            # Combine title and abstract for ALL records
            X_text = df_full[["title", "abstract"]].fillna("")
            texts = (X_text["title"] + " " + X_text["abstract"]).tolist()
            
            # Compute embeddings for ALL records
            embeddings = encoder.encode(texts, 
                                        batch_size=transformer_batch_size,
                                        normalize_embeddings=True,
                                        show_progress_bar=True)
            
            # Save embeddings as parquet
            embeddings_df = pd.DataFrame(embeddings)
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            embeddings_df.to_parquet(embedding_path)
            print(f"Embeddings for all records saved to {embedding_path}")
            
            X_full = embeddings_df
            skip_feature_extraction = True
    
    # Apply filters: abstract length and oa_status
    # Start with all records included
    filter_mask = pd.Series([True] * len(df_full), index=df_full.index)
    
    # Apply abstract length filter if specified
    if abstract_min_length is not None:
        df_full["_abstract_length"] = df_full.apply(get_abstract_length, axis=1)
        filter_mask = filter_mask & (df_full["_abstract_length"] >= abstract_min_length)
    
    # Apply oa_status filter if specified (for synergy benchmark only)
    if benchmark == "synergy" and oa_status is not None:
        oa_filter = oa_status == "True"
        # Ensure is_open_access is boolean type
        df_full["is_open_access"] = df_full["is_open_access"].astype(bool)
        # Combine with existing filter mask
        filter_mask = filter_mask & (df_full["is_open_access"] == oa_filter)
    
    # Apply combined filter
    initial_count = len(df_full)
    df = df_full[filter_mask].reset_index(drop=True)
    X = X_full[filter_mask].reset_index(drop=True)
    filtered_count = len(df)
    
    # Clean up temporary column if it exists
    if "_abstract_length" in df.columns:
        df = df.drop(columns=["_abstract_length"])
    
    print(f"Filtered from {initial_count} to {filtered_count} records (abstract_min_length={abstract_min_length}, oa_status={oa_status})")
    
    Y = df["label_included"].fillna(0)
    ids = df["openalex_id"]
    total_n_records = len(df)
    
    # Check if there are any inclusions after filtering
    n_inclusions = int(Y.sum())
    n_exclusions = int((Y == 0).sum())
    n_priors = n_pos_priors + n_neg_priors
    
    if total_n_records <= n_priors:
        print(f"ERROR: Not enough records for simulation!")
        print(f"Dataset has {total_n_records} records, but need more than {n_priors} (n_pos_priors={n_pos_priors} + n_neg_priors={n_neg_priors})")
        print(f"A simulation requires at least {n_priors + 1} records to have priors plus at least one record to screen.")
        return
    
    if n_inclusions == 0:
        print(f"ERROR: No inclusions found after filtering!")
        print(f"Dataset has {total_n_records} records (0 inclusions, {n_exclusions} exclusions)")
        print(f"Cannot run simulation without any positive labels.")
        return
    
    if n_inclusions < n_pos_priors:
        print(f"ERROR: Not enough inclusions for the requested priors!")
        print(f"Found {n_inclusions} inclusions, but need {n_pos_priors} positive priors")
        return
    
    if n_exclusions < n_neg_priors:
        print(f"ERROR: Not enough exclusions for the requested priors!")
        print(f"Found {n_exclusions} exclusions, but need {n_neg_priors} negative priors")
        return
    
    print(f"Dataset: {total_n_records} records ({n_inclusions} inclusions, {n_exclusions} exclusions)")
    
    # Set up the active learning cycle
    alc = [
        ActiveLearningCycle(
            querier=TopDown(),
            stopper=IsFittable(),
            n_query=lambda results: n_query_extreme(results, X.shape[0])
        ),
        ActiveLearningCycle.from_meta(model_config),
    ]
    
    # Generate a sequence of random seeds based on the number of random cycles and the initial seed
    seeds = [initial_seed + i for i in range(n_random_cycles)]

    # Run the simulation for each random seed
    df_results_list = []
    for seed in tqdm(seeds):
        # Set up the list of priors for the simulation
        pos_priors = ids[Y == 1].sample(n_pos_priors, random_state=seed).index.tolist()
        neg_priors = ids[Y == 0].sample(n_neg_priors, random_state=seed).index.tolist()
        priors = pos_priors + neg_priors
        
        # Run the simulation
        simulate = asreview.Simulate(
            X = X,
            labels=Y,
            cycles=alc,
            skip_transform=skip_feature_extraction,
        )

        simulate.label(priors)
        simulate.review()

        # Save the results to a CSV file
        df_results = simulate._results.dropna(axis=0,
                                              subset="training_set")
        df_results = df_results.copy() 
        df_results["total_n_records"] = total_n_records
        df_results["seed"] = seed
        df_results["total_n_relevant_records"] = df_results["label"].sum()
        df_results_list.append(df_results[["record_id", "label", "seed", "total_n_records", "training_set", "total_n_relevant_records"]])

    df_results_final = pd.concat(df_results_list)
    df_results_final.rename(columns={"training_set": "n_training_records"}, inplace=True)

    # make output folder if it doesn't exist
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    df_results_final.to_csv(df_results_path, index=False)

if __name__ == "__main__":
    main()