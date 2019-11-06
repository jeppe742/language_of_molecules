#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J language_of_molecules
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s180213@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o joblogs/gpu-%J.out
#BSUB -e joblogs/gpu_%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.0
#module load python3


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/zhome/9e/8/130993/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/zhome/9e/8/130993/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/zhome/9e/8/130993/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/zhome/9e/8/130993/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate rdkit
python3 train.py --num_epochs=100  --scaffold=True  --num_layers=8 --num_heads=6  --name_postfix=_db=zinc
