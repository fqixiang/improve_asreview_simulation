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
    parser.add_argument("--topic", type=str, default="macular_degeneration", help="The topic of the dataset")
    parser.add_argument("--model", type=str, default="elas_u4", help="The model to use")
    parser.add_argument("--n_pos_priors", type=int, default=1, help="The number of positive priors")
    parser.add_argument("--n_neg_priors", type=int, default=1, help="The number of negative priors")

    args = parser.parse_args()
    topic = args.topic
    model = args.model
    n_pos_priors = args.n_pos_priors
    n_neg_priors = args.n_neg_priors

    # Read the CSV file
    df = pd.read_csv(f"./results/{topic}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)/{model}_results.csv")

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

    # make a table with the following columns:
    # topic, model, n_iterations, average loss, std loss, average ndcg, std ndcg
    summary_df = pd.DataFrame({
        "topic": [topic],
        "model": [model],
        "n_iterations": [n_iterations],
        "average_loss": [ave_loss],
        "std_loss": [std_loss],
        "average_ndcg": [ave_ndcg],
        "std_ndcg": [std_ndcg]
    })
    # check if summary table already exists
    if os.path.exists(f"./results/simulation_summary.csv"):
        # read the existing summary table
        summary_df_existing = pd.read_csv(f"./results/simulation_summary.csv")
        # append the new summary table to the existing one if the topic and model are not already in the summary table
        if not ((summary_df_existing["topic"] == topic) & (summary_df_existing["model"] == model)).any():
            # append the new summary table to the existing one
            print("Summary table already exists, appending new results to it.")
            summary_df_existing = pd.concat([summary_df_existing, summary_df], ignore_index=True)
        else:
            # if the topic and model are already in the summary table, update the existing row with the new values
            print("Summary table already exists, updating existing results.")
            summary_df_existing.loc[
                (summary_df_existing["topic"] == topic) & (summary_df_existing["model"] == model),
                ["n_iterations", "average_loss", "std_loss", "average_ndcg", "std_ndcg"]
            ] = [n_iterations, ave_loss, std_loss, ave_ndcg, std_ndcg]
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
    title = f"Recall vs Proportion of Records Screened for '{topic}'"
    subtitle = f"""With model '{model}', {n_pos_priors} positive prior(s) and {n_neg_priors} negative prior(s),
    {len(seeds)} random sampling cycles (corresponding to differently colored lines).
    All {total_n_relevant_records} remaining relevant records retrieved after screening ~{ave_n_records_needed_for_full_recall} records.
    Average loss: {ave_loss:.3f}; Average NDCG: {ave_ndcg:.3f}"""
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.title(subtitle, fontsize=10)
    # set plot size
    plt.gcf().set_size_inches(10, 9)

    # make output folder if it doesn't exist
    save_folder_path = f"./results/{topic}/{n_pos_priors}_pos_prior(s)/{n_neg_priors}_neg_prior(s)"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    plt.savefig(f"{save_folder_path}/{model}_recall_vs_proportion.png")
    # plt.show()


if __name__ == "__main__":
    main()