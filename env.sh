# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.

module load languages/intel/2020-u4

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=socket