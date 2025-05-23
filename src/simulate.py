import pandas as pd
import asreview
from asreview.learner import ActiveLearningCycle
from asreview.models.queriers import TopDown
from asreview.models.stoppers import IsFittable
from argparse import ArgumentParser
from tqdm import tqdm
import os
from utils import model_configurations

def main():
    # Define the argument parser
    parser = ArgumentParser()
    parser.add_argument("--topic", type=str, default="macular_degeneration", help="The topic of the dataset")
    parser.add_argument("--n_pos_priors", type=int, default=1, help="The number of positive priors")
    parser.add_argument("--n_neg_priors", type=int, default=1, help="The number of negative priors")
    parser.add_argument("--initial_random_seed", type=int, default=42, help="The initial random seed for reproducibility")
    parser.add_argument("--model", type=str, default="elas_u4", help="The model to use")
    parser.add_argument("--n_random_cycles", type=int, default=10, help="The number of random cycles to run")
    args = parser.parse_args()
    topic = args.topic
    n_pos_priors = args.n_pos_priors
    n_neg_priors = args.n_neg_priors
    initial_seed = args.initial_random_seed
    model = args.model
    n_random_cycles = args.n_random_cycles

    # Check if simulation is already done
    save_folder_path = f"./results/{topic}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)"
    df_results_path = f"{save_folder_path}/{model}_results.csv"
    if os.path.exists(df_results_path):
        print(f"Simulation already done. Results saved in {df_results_path}")
        return

    # Define global variables
    global model_configurations
    valid_models = list(model_configurations.keys())

    # Read the Excel file
    file_path = f"./data/{topic}.xlsx"
    df = pd.read_excel(file_path)
    X = df[["title", "abstract"]]
    Y = df["label_included"].fillna(0)
    ids = df["paper_id"]
    total_n_records = len(df)

    # Define model configurations   
    try:
        model_config = model_configurations[args.model]
    except KeyError:
        raise ValueError(f"Invalid model name. Choose from {valid_models}")
    
    # Set up the active learning cycle
    alc = [
        ActiveLearningCycle(
            querier=TopDown(),
            stopper=IsFittable(),
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