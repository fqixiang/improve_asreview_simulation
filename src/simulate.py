import pandas as pd
import asreview
from asreview.learner import ActiveLearningCycle
from asreview.models.queriers import TopDown
from asreview.models.stoppers import IsFittable
from argparse import ArgumentParser
from tqdm import tqdm
import os
from utils import model_configurations, n_query_extreme

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

    # Determine OA status folder name
    if oa_status is None:
        oa_folder = "all"
    elif oa_status == "True":
        oa_folder = "open"
    else:  # "False"
        oa_folder = "closed"

    # Check if simulation is already done
    save_folder_path = f"./results/{benchmark}/{dataset}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)/{oa_folder}"
    df_results_path = f"{save_folder_path}/{model}_results.csv"
    if os.path.exists(df_results_path):
        print(f"Simulation already done. Results saved in {df_results_path}")
        return

    # Define global variables
    global model_configurations
    valid_models = list(model_configurations.keys())

    # Read the data file
    if benchmark == "synergy":
        file_path = f"./data/{benchmark}/{dataset}/labels.csv"
        df = pd.read_csv(file_path)
        # Filter by open access status if specified
        if oa_status is not None:
            oa_filter = oa_status == "True"
            # Ensure is_open_access is boolean type
            df["is_open_access"] = df["is_open_access"].astype(bool)
            df = df[df["is_open_access"] == oa_filter].reset_index(drop=True)
            print(f"Filtered to {len(df)} records with is_open_access={oa_filter}")
    else:  # improve
        file_path = f"./data/{benchmark}/{dataset}/{dataset}.xlsx"
        df = pd.read_excel(file_path)
    
    X = df[["title", "abstract"]]
    Y = df["label_included"].fillna(0)
    ids = df["openalex_id"]
    total_n_records = len(df)

    # Define model configurations   
    try:
        model_config = model_configurations[args.model]
    except KeyError:
        raise ValueError(f"Invalid model name. Choose from {valid_models}")
    
    # Check if embeddings are needed and available for h3 or l2 models
    skip_feature_extraction = False
    if model in ["elas_h3", "elas_l2"]:
        # Determine embedding file path - store in embeddings directory
        embedding_path = f"./embeddings/{benchmark}/{dataset}/{model}_embeddings.parquet"
        
        if os.path.exists(embedding_path):
            # Load pre-computed embeddings
            print(f"Loading pre-computed embeddings from {embedding_path}")
            embeddings_df = pd.read_parquet(embedding_path)
            X = embeddings_df
            skip_feature_extraction = True
        else:
            # Compute embeddings and save them
            print(f"Computing embeddings for {model}. This may take a while...")
            from sentence_transformers import SentenceTransformer
            
            # Select the appropriate model
            if model == "elas_h3":
                transformer_model = "mixedbread-ai/mxbai-embed-large-v1"
            else:  # elas_l2
                transformer_model = "intfloat/multilingual-e5-large"
            
            # Load the sentence transformer model
            encoder = SentenceTransformer(transformer_model)
            
            # Combine title and abstract
            X_text = df[["title", "abstract"]].fillna("")
            texts = (X_text["title"] + " " + X_text["abstract"]).tolist()
            
            # Compute embeddings
            embeddings = encoder.encode(texts, 
                                        batch_size=transformer_batch_size,
                                        normalize_embeddings=True,
                                        show_progress_bar=True)
            
            # Save embeddings as parquet
            embeddings_df = pd.DataFrame(embeddings)
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            embeddings_df.to_parquet(embedding_path)
            print(f"Embeddings saved to {embedding_path}")
            
            X = embeddings_df
            skip_feature_extraction = True
    
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