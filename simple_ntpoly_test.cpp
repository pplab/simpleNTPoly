#include <iostream>
#include <mpi.h>
#include "utils.hpp"
#include "simple_ntpoly.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int nFull, nelec, nspin;
    double converge_density, converge_overlap, threshold;
    if (myid == 0)
    {
        loadParametersFromFile("parameters.txt", nFull, nelec, nspin, converge_density, converge_overlap, threshold);
        // std::cout<<"nFull: "<<nFull<<"\n";
        // std::cout<<"nelec: "<<nelec<<"\n";
        // std::cout<<"nspin: "<<nspin<<"\n";
        // std::cout<<"converge_density: "<<converge_density<<"\n";
        // std::cout<<"converge_overlap: "<<converge_overlap<<"\n";
        // std::cout<<"threshold: "<<threshold<<"\n";
    }
    MPI_Bcast(&nFull, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nelec, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nspin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // std::cout<<"myid="<<myid<<" nFull="<<nFull<<" nelec="<<nelec<<" nspin="<<nspin<<"\n";
    MPI_Bcast(&converge_density, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&converge_overlap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // std::cout<<"myid="<<myid<<" converge_density="<<converge_density
    //          <<" converge_overlap="<<converge_overlap
    //          <<" threshold="<<threshold<<"\n";
    outlog("parameters are broadcasted");
    outlog("nFull", nFull);
    outlog("nelec", nelec);
    outlog("nspin", nspin);
    outlog("converge_density", converge_density);
    outlog("converge_overlap", converge_overlap);
    outlog("threshold", threshold);

    int blacs_ctxt;
    int narows, nacols;
    int desc[9];
    initBlacsGrid(MPI_COMM_WORLD, nFull, 2, blacs_ctxt, narows, nacols, desc);

    double* H = new double[narows * nacols];
    double* S = new double[narows * nacols];
    outlog("start loading H");
    loadBCDMatrixFromABACUSFile("data-0-H", MPI_COMM_WORLD, desc, H);
    outlog("start loading S");
    loadBCDMatrixFromABACUSFile("data-0-S", MPI_COMM_WORLD, desc, S);
    outlog("H and S are loaded");
    saveLocalMatrixToFile(narows, nacols, H, "H_"+std::to_string(myid)+".dat");
    saveLocalMatrixToFile(narows, nacols, S, "S_"+std::to_string(myid)+".dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H, "H_save.dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, S, "S_save.dat");

    double* DM = new double[narows * nacols];
    double* EDM = new double[narows * nacols];
    double energy, chemical_potential;
    outlog("start ntpoly solving");
    ntpoly::simple_ntpoly(MPI_COMM_WORLD, desc, 
                narows, nacols,
                converge_density, converge_overlap, threshold, 
                nelec, nspin, H, S, 
                DM, EDM, energy, chemical_potential);
    outlog("ntpoly solving finished");
    saveLocalMatrixToFile(narows, nacols, DM, "DM_"+std::to_string(myid)+".dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, DM, "DM.dat");
    delete[] H;
    delete[] S;
    delete[] DM;
    delete[] EDM;
    MPI_Finalize();
    return 0;
}
