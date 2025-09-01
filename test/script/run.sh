#!/bin/sh
mpirun -np 8 ../../test_f \
--process_rows 2 --process_columns 2 --process_slices 2 \
--size 65 --hamiltonian data-0-H --overlap data-0-S \
--number_of_electrons 10 --threshold 1e-6 \
--converge_overlap 1e-3 --converge_density 1e-5 \
--density DM.mtx
