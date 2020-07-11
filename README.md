
DTW script for GD project

Subsequence DTW for different recordings of one concert segmented into individual tracks.
Uses shared_memory, so may need a cleanup if the script exist improperly. 
Needs vamp tuning difference plugin.
All audio is loaded into memory, tested on 12 CPU cores and 64GB RAM.
may need increase of resource limit, e.g. "ulimit -n 30000"
