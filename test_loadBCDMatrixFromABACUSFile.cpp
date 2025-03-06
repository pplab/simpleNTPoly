#include <iostream>
#include <string>
#include <fstream>
#include <mpi.h>
#include <ProcessGrid.h>
#include <PSMatrix.h>
#include <TripletList.h>
#include <Triplet.h>
#include <Permutation.h>
#include <SolverParameters.h>
#include <SquareRootSolvers.h>
#include <DensityMatrixSolvers.h>
#include "simple_ntpoly.h"
#include "utils.hpp"
#include "Cblacs.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    const std::string filename=argv[1];
    // set matrix parameters
    int nFull;
    int nblk;
    if(myid==0)
    {
        std::fstream fin(filename);
        fin>>nFull;
        fin.close();
    }
    MPI_Bcast(&nFull, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(nFull<=4)
    {
        nblk=1;
    }
    else if(nFull<=16)
    {
        nblk=2;
    }
    else
    {
        nblk=4;
    }
    // init cblacs
    int blacs_ctxt;
    int desc[9];
    int narows, nacols;
    initBlacsGrid(MPI_COMM_WORLD, nFull, nblk, blacs_ctxt, narows, nacols, desc);

    // call loadBCDMatrixFromABACUSFile to load matrix from abacus file
    double* H = new double[narows * nacols];
    loadBCDMatrixFromABACUSFile(filename, MPI_COMM_WORLD, desc, H);

    // save matrix to file
    saveLocalMatrixToFile(narows, nacols, H, "H_"+std::to_string(myid)+".dat");

    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H, "H_final.dat");

    // transform BCD Matrix to NTPoly PSMatrix
    
    // init default process grid
    int process_slice=1;
    NTPoly::ConstructGlobalProcessGrid(MPI_COMM_WORLD, process_slice);
    NTPoly::Matrix_ps Hamiltonian(nFull);
    double threshold=1e-10;
    ntpoly::constructPSMatrixFromBCD(Hamiltonian, MPI_COMM_WORLD, desc, narows, nacols, H, threshold);
    ntpoly::constructBCDFromPSMatrix(Hamiltonian, MPI_COMM_WORLD, desc, narows, nacols, H);
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H, "H_final2.dat");
    MPI_Finalize();
    return 0;
}