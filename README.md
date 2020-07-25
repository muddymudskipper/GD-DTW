
DTW script for GD project

Subsequence DTW for different recordings of one concert segmented into individual tracks.
Uses shared_memory, so may need a cleanup if the script exist improperly. 
Needs vamp tuning difference plugin.
Multiprocessing variables set for 12 CPU cores and 64GB RAM.
May need increase of resource limit (ulimit -n)

Prerequisites:

Python 3.8,
Jython,
Vamp plugins: nnls-chroma, tuning-difference, match-subsequence

