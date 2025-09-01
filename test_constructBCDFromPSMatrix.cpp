#include <iostream>
#include <mpi.h>
#include <ProcessGrid.h>
#include <PSMatrix.h>
#include <TripletList.h>
#include <Triplet.h>
#include "utils.hpp"
#include "simple_ntpoly.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int nFull, n_process_slice;
    double threshold;
    if (myid == 0)
    {
        std::cout<<"input nFull, n_process_slice and threshold:"<<std::endl;
        std::cin>>nFull>>n_process_slice>>threshold;
    }
    MPI_Bcast(&nFull, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_process_slice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    outlog("parameters are broadcasted");
    outlog("nFull", nFull);
    outlog("threshold", threshold);
    outlog("n_process_slice", n_process_slice);

    int blacs_ctxt;
    int narows, nacols;
    int desc[9];
    initBlacsGrid(MPI_COMM_WORLD, nFull, 2, blacs_ctxt, narows, nacols, desc);

    double* H = new double[narows * nacols];
    outlog("start loading H");
    loadBCDMatrixFromABACUSFile("data-0-H", MPI_COMM_WORLD, desc, H);
    outlog("H and S are loaded");
    saveLocalMatrixToFile(narows, nacols, H, "H_"+std::to_string(myid)+".dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H, "H_save.dat");

    
    NTPoly::ConstructGlobalProcessGrid(MPI_COMM_WORLD, n_process_slice);    
    NTPoly::Matrix_ps Hamiltonian(nFull);
    ntpoly::constructPSMatrixFromBCD(Hamiltonian, MPI_COMM_WORLD, desc, narows, nacols, H, threshold);
    Hamiltonian.WriteToMatrixMarket("Hamiltonian.mtx");

    double* S = new double[narows * nacols];
    ntpoly::constructBCDFromPSMatrix(Hamiltonian, MPI_COMM_WORLD, 'C', desc, narows, nacols, S);
    MPI_Barrier(MPI_COMM_WORLD);
    outlog("BCM matrix S is constructed from PSMatrix Hamiltonian");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, S, "S.dat");
    MPI_Barrier(MPI_COMM_WORLD);
    outlog("BCM matrix S is saved to file");
    delete[] H;
    delete[] S;
    MPI_Finalize();
    return 0;
}
