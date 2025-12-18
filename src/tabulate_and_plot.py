import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from asreview.metrics import loss
from asreview.metrics import ndcg
from utils import pad_labels

def main():
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="improve", help="The benchmark to use (improve or synergy)")
    parser.add_argument("--dataset", type=str, default="macular_degeneration", help="The dataset to use")
    parser.add_argument("--model", type=str, default="elas_u4", help="The ASReview model to use")
    parser.add_argument("--n_pos_priors", type=int, default=1, help="The number of positive priors")
    parser.add_argument("--n_neg_priors", type=int, default=1, help="The number of negative priors")
    parser.add_argument("--oa_status", type=str, default=None, choices=[None, "True", "False"], help="Filter by open access status (synergy only): True, False, or None for no filter")
    parser.add_argument("--abstract_min_length", type=int, default=None, help="Minimum abstract length (words for space-separated languages, characters otherwise)")

    args = parser.parse_args()
    benchmark = args.benchmark
    dataset = args.dataset
    model = args.model
    n_pos_priors = args.n_pos_priors
    n_neg_priors = args.n_neg_priors
    oa_status = args.oa_status
    abstract_min_length = args.abstract_min_length

    # Determine OA status folder name
    if oa_status is None:
        oa_folder = "all"
    elif oa_status == "True":
        oa_folder = "open"
    else:  # "False"
        oa_folder = "closed"

    # Read the CSV file
    save_folder_path = f"./results/{benchmark}/{dataset}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)/{oa_folder}"
    abs_suffix = f"abs{abstract_min_length}" if abstract_min_length is not None else "abs_all"
    df = pd.read_csv(f"{save_folder_path}/{model}_{abs_suffix}_results.csv")

    # define some variables 
    total_n_records = int(df["total_n_records"].iloc[0])
    seeds = df["seed"].unique()
    n_iterations = len(seeds)
    n_priors = n_pos_priors + n_neg_priors
    total_n_relevant_records = int(df["total_n_relevant_records"].iloc[0])

    plot_df_list = []
    n_records_needed_for_full_recall_list = []
    losses_ls = []
    ndcgs_ls = []
    for seed in seeds:
        df_seed = df[df["seed"] == seed]
        padded_labels = pad_labels(
            labels=df_seed["label"].tolist(), 
            n_priors=n_priors,
            total_n_records=total_n_records
        )
        losses_ls.append(loss(padded_labels))
        ndcgs_ls.append(ndcg(padded_labels))

        n_recall = np.cumsum(padded_labels)
        relative_recall = n_recall / n_recall.max()
        prop_records = np.arange(1, len(padded_labels) + 1) / len(padded_labels)

        # find the index of the first record where recall is 1
        n_records_needed_for_full_recall = df_seed["n_training_records"].max()
        n_records_needed_for_full_recall_list.append(n_records_needed_for_full_recall)

        # use sns to make a plot where x is the proportion of records and y is the recall
        plot_df = pd.DataFrame({
            "proportion_of_records": prop_records,
            "recall": relative_recall,
            "seed": seed
        })
        plot_df_list.append(plot_df)

    plot_df_final = pd.concat(plot_df_list)
    # calculate the average number of records needed for full recall
    ave_n_records_needed_for_full_recall = int(np.mean(n_records_needed_for_full_recall_list))
    # calculate the average loss and ndcg
    ave_loss = np.mean(losses_ls)
    ave_ndcg = np.mean(ndcgs_ls)
    # calculate the standard deviation of the loss and ndcg
    std_loss = np.std(losses_ls)
    std_ndcg = np.std(ndcgs_ls)

    # make a table with the following columns
    summary_df = pd.DataFrame({
        "dataset": [dataset],
        "model": [model],
        "oa_status": [oa_status if oa_status is not None else "all"],
        "min_abstract_words": [abstract_min_length],
        "total_n_records": [total_n_records],
        "total_n_inclusions": [total_n_relevant_records],
        "loss_mean": [ave_loss],
        "loss_sd": [std_loss],
        "ndcg_mean": [ave_ndcg],
        "ndcg_sd": [std_ndcg],
        "n_seeds": [n_iterations]
    })
    # check if summary table already exists
    if os.path.exists(f"./results/simulation_summary.csv"):
        # read the existing summary table
        summary_df_existing = pd.read_csv(f"./results/simulation_summary.csv")
        # append the new summary table to the existing one if the dataset, model, oa_status, and min_abstract_words are not already in the summary table
        match_condition = (
            (summary_df_existing["dataset"] == dataset) & 
            (summary_df_existing["model"] == model) &
            (summary_df_existing["oa_status"] == (oa_status if oa_status is not None else "all")) &
            (summary_df_existing["min_abstract_words"] == abstract_min_length)
        )
        if not match_condition.any():
            # append the new summary table to the existing one
            print("Summary table already exists, appending new results to it.")
            # Ensure consistent dtypes before concatenation
            for col in summary_df.columns:
                if col in summary_df_existing.columns:
                    summary_df[col] = summary_df[col].astype(summary_df_existing[col].dtype)
            summary_df_existing = pd.concat([summary_df_existing, summary_df], ignore_index=True)
        else:
            # if the combination already exists in the summary table, update the existing row with the new values
            print("Summary table already exists, updating existing results.")
            summary_df_existing.loc[
                match_condition,
                ["total_n_records", "total_n_inclusions", "loss_mean", "loss_sd", "ndcg_mean", "ndcg_sd", "n_seeds"]
            ] = [total_n_records, total_n_relevant_records, ave_loss, std_loss, ave_ndcg, std_ndcg, n_iterations]
        # save the updated summary table to a csv file
        print("Saving updated summary table to simulation_summary.csv")
        summary_df_existing.to_csv(f"./results/simulation_summary.csv", index=False)
    else:
        # save the completely new summary table to a csv file
        print("Summary table does not exist, creating new one.")
        summary_df.to_csv(f"./results/simulation_summary.csv", index=False)

    # use sns to make a plot where x is the proportion of records and y is the recall, group/colour by seed
    sns.lineplot(data=plot_df_final, x="proportion_of_records", y="recall", hue="seed")
    # add a line starting at (0,0) and ending at (1,1)
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("Proportion of Records")
    plt.ylabel("Recall")
    # remove legend
    plt.legend().remove()
    # title text
    title = f"Recall vs Proportion of Records Screened for '{dataset}'"
    oa_status_display = oa_status if oa_status is not None else "all"
    abstract_min_display = abstract_min_length if abstract_min_length is not None else "all"
    subtitle = f"""Model: {model} | OA Status: {oa_status_display} | Min Abstract Length: {abstract_min_display}
    {n_pos_priors} positive prior(s), {n_neg_priors} negative prior(s), {len(seeds)} random cycles (differently colored lines)
    All {total_n_relevant_records} inclusions retrieved after ~{ave_n_records_needed_for_full_recall} records screened
    Loss: {ave_loss:.3f} ± {std_loss:.3f} | NDCG: {ave_ndcg:.3f} ± {std_ndcg:.3f}"""
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.title(subtitle, fontsize=10)
    # set plot size
    plt.gcf().set_size_inches(10, 9)

    # Save plot to the same folder as results
    plt.savefig(f"{save_folder_path}/{model}_{abs_suffix}_recall_vs_proportion.png")
    # plt.show()

if __name__ == "__main__":
    main()