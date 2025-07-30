#include <iostream>
#include <mpi.h>
#include "utils.hpp"
#include "simple_ntpoly.h"
#include <vector>  // 引入 vector 头文件

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

    // 使用 std::vector<double> 替代 double*
    std::vector<double> H(narows * nacols);
    std::vector<double> S(narows * nacols);
    outlog("start loading H");
    loadBCDMatrixFromABACUSFile("data-0-H", MPI_COMM_WORLD, desc, H.data());
    outlog("start loading S");
    loadBCDMatrixFromABACUSFile("data-0-S", MPI_COMM_WORLD, desc, S.data());
    outlog("H and S are loaded");
    saveLocalMatrixToFile(narows, nacols, H.data(), "H_"+std::to_string(myid)+".dat");
    saveLocalMatrixToFile(narows, nacols, S.data(), "S_"+std::to_string(myid)+".dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, H.data(), "H_save.dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, S.data(), "S_save.dat");

    // 使用 std::vector<double> 替代 double*
    std::vector<double> DM(narows * nacols);
    std::vector<double> EDM(narows * nacols);
    double energy, chemical_potential;

    // 打印调用前的指针地址，使用 outlog 替代 std::cout
    outlog("Before ntpoly::simple_ntpoly - DM address: ", DM.data());
    outlog("Before ntpoly::simple_ntpoly - EDM address: ", EDM.data());

    outlog("start ntpoly solving");
    ntpoly::simple_ntpoly(MPI_COMM_WORLD, desc, 
                narows, nacols,
                converge_density, converge_overlap, threshold, 
                nelec, nspin, H.data(), S.data(), 
                DM.data(), EDM.data(), energy, chemical_potential);

    // 打印调用后的指针地址，使用 outlog 替代 std::cout
    outlog("After ntpoly::simple_ntpoly - DM address: ", DM.data());
    outlog("After ntpoly::simple_ntpoly - EDM address: ", EDM.data());

    MPI_Barrier(MPI_COMM_WORLD);
    outlog("ntpoly solving finished");
    //saveLocalMatrixToFile(narows, nacols, DM.data(), "DM_"+std::to_string(myid)+".dat");
    //saveLocalMatrixToFile(narows, nacols, EDM.data(), "EDM_"+std::to_string(myid)+".dat");
    saveBCDMatrixToFile(MPI_COMM_WORLD, desc, narows, nacols, DM.data(), "DM.dat");
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        std::cout<<"energy="<<energy<<"\n";
        std::cout<<"chemical_potential="<<chemical_potential<<"\n";
    }
    // 无需手动 delete[]，vector 会自动管理内存

    MPI_Barrier(MPI_COMM_WORLD);
    // ntpoly::cleanupMPIResources();
    outlog("finished");
    MPI_Finalize();
    return 0;
}
