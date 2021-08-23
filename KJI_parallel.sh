#PBS -N pythonmpi
#PBS -l nodes=2:ppn=8
#PBS -l pmem=8gb
#PBS -l walltime=05:00:00
#PBS -q N10C80
#PBS -j oe
#PBS -o kji.out

# Load modules

module load py3-mpi4py

# module load py3-scipy

module load py3-numpy

# Run for multiple no of processors (p), various size of matrix A (size) and both block and shuffle methods

cd $PBS_O_WORKDIR

for p in 6 8 10 12 16
do
for size in 16 32 48
do
for m in 0 1
do
mpirun -np $p python3 KJI_parallel.py $m $size 0
done
done
done
