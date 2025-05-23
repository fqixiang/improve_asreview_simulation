#!/bin/bash
#SBATCH --job-name=parallel_simulations
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=q.fang@uu.nl
#SBATCH --chdir=/projects/0/prjs1241/ASReview_Simulation

# Load necessary modules
module load 2022
module load CUDA/11.7.0
source /projects/0/prjs1241/ASReview_Simulation/.venv/bin/activate

models=("elas_u4" "elas_u3" "elas_l2" "elas_h3")
topics=("chronic_rhinosinusitis" "macular_degeneration" "multiple_sclerosis" "severe_aortic_stenosis" "atrial_fibrillation" "cervical_cancer" "coronary_artery_disease" "heart_failure" "neck_and_head_cancer" "prostate_cancer")
n_neg_priors=11

# Generate all combinations
tasks=()
for model in "${models[@]}"; do
  for topic in "${topics[@]}"; do
    tasks+=("$model $topic")
  done
done

# Limit to N concurrent jobs
max_parallel=4
running=0

for task in "${tasks[@]}"; do
  model=$(echo $task | awk '{print $1}')
  topic=$(echo $task | awk '{print $2}')

  echo "Launching task for model=$model, topic=$topic"

  # Run simulate + plot in sequence within one srun call
  srun --exclusive -N1 -n1 bash -c "
    echo 'Simulating $model $topic'
    python ./src/simulate.py --model '$model' --topic '$topic' --n_neg_priors '$n_neg_priors'

    echo 'Plotting $model $topic'
    python ./src/tabulate_and_plot.py --model '$model' --topic '$topic' --n_neg_priors '$n_neg_priors'
  " &

  ((running+=1))

  # Wait if max parallelism reached
  if (( running >= max_parallel )); then
    wait
    running=0
  fi
done

# Wait for any remaining background tasks
wait
echo "All simulations completed."