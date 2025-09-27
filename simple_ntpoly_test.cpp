#include <iostream>
#include <mpi.h>
#include "utils.hpp"
#include "simple_ntpoly.h"
#include "timer.hpp"
#include <vector>  // 引入 vector 头文件

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int nFull, nelec, nspin;
    double converge_density, converge_overlap, threshold;    
    int verbose_level=0;
    MPITimer timer;
    if (myid == 0)
    {
        loadParametersFromFile("parameters.txt", nFull, nelec, nspin, 
                converge_density, converge_overlap, threshold, verbose_level);
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
    MPI_Bcast(&verbose_level, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // std::cout<<"myid="<<myid<<" converge_density="<<converge_density
    //          <<" converge_overlap="<<converge_overlap
    //          <<" threshold="<<threshold<<"\n";
    if(verbose_level>0)
    {
        outlog("parameters are broadcasted");
        outlog("nFull", nFull);
        outlog("nelec", nelec);
        outlog("nspin", nspin);
        outlog("converge_density", converge_density);
        outlog("converge_overlap", converge_overlap);
        outlog("threshold", threshold);
        outlog("verbose_level", verbose_level);
    }
    int blacs_ctxt;
    int narows, nacols;
    int desc[9];
    initBlacsGrid(MPI_COMM_WORLD, 'R', nFull, 2, blacs_ctxt, narows, nacols, desc);

    // 使用 std::vector<double> 替代 double*
    std::vector<double> H(narows * nacols);
    std::vector<double> S(narows * nacols);
    if(verbose_level>0)
    {
        outlog("start loading H");
        timer.start();
    }
    loadBCDMatrixFromABACUSFile("data-0-H", MPI_COMM_WORLD, desc, H.data());
    if(verbose_level>0)
    {
        outlog("H loading time", timer.stop());
    }
    if(verbose_level>0)
    {
        outlog("start loading S");
        timer.start();
    }
    loadBCDMatrixFromABACUSFile("data-0-S", MPI_COMM_WORLD, desc, S.data());
    if(verbose_level>0)
    {
        outlog("S loading time", timer.stop()); 
    }
    if(verbose_level>2)
    {
        saveLocalMatrixToFile(narows, nacols, H.data(), "H_"+std::to_string(myid)+".dat");
        saveLocalMatrixToFile(narows, nacols, S.data(), "S_"+std::to_string(myid)+".dat");
        saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H.data(), "H_save.dat");
        saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, S.data(), "S_save.dat");
    }

    // 使用 std::vector<double> 替代 double*
    std::vector<double> DM(narows * nacols);
    std::vector<double> EDM(narows * nacols);
    double energy, chemical_potential;

    if(verbose_level>0)
    {
        outlog("start ntpoly solving");
        timer.start();
    }
    ntpoly::simple_ntpoly(MPI_COMM_WORLD, 'R', desc, 
                narows, nacols,
                converge_density, converge_overlap, threshold, 
                nelec, nspin, H.data(), S.data(), 
                DM.data(), EDM.data(), energy, chemical_potential, verbose_level);

    if(verbose_level>0)
    {
        outlog("simple_ntpoly solving time", timer.stop());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(verbose_level>0)
    {
        outlog("ntpoly solving finished");
        timer.start();
    }
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, DM.data(), "DM.dat");
    if(verbose_level>0)
    {
        outlog("DM saving time", timer.stop());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        std::cout<<"energy="<<energy<<"\n";
        std::cout<<"chemical_potential="<<chemical_potential<<"\n";
    }
    
    if(verbose_level>0)
    {
        outlog("finished");
    }
    MPI_Finalize();
    return 0;
}
