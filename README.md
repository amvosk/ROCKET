# ROCKET
ROCKET provides a SOTA solution in time-series classification. But how can we adapt it for uncertainty estimation? 

## Packages
numpy
scipy
matplotlib
h5py
torch
einops
tqdm

## Evaluation
You can run evaluation.ipynb, which compares accuracy of the original model (taken from ROCKET paper, in accuracy.txt) and of the reproduced models. Also, it compares different metrics of uncertancy estimate and plots results.

To run actual experiments, you need to download dataset from https://drive.google.com/file/d/1I0YFYwOxlPKLDO0x5DQCZhNnPRQTOV8N/view?usp=share_link and put it into the "ROCKET/" directory. \\
Then, you can run "python evaluationRC.py" to get accuracy of the ridge regression classifier (about 2 hours on gpu), or run "python evaluation.py" to get uncertancy estimates (maybe about 15+ gpu time). All this result are precomputed in the "results" directory.

