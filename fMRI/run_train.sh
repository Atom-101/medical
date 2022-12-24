#!/bin/bash
sbatch << EOT
#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 32:00:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu         # Partition to submit to
#SBATCH --gres=gpu:1        # Number of GPUs
#SBATCH --mem=64000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out/$1.out  # %j inserts jobid
#SBATCH -e out/$1.err  # %j inserts jobid

# source activate snemi_new
/n/home11/abanerjee/.conda/envs/snemi_new/bin/python clip_distill_train.py $1
EOT
