include make.inc.aocc

simple_ntpoly_test: simple_ntpoly_test.o simple_ntpoly.o utils.o
	$(CXX) $(CXXFLAGS)  -o simple_ntpoly_test simple_ntpoly_test.o simple_ntpoly.o utils.o \
		-L$(NTPOLY_DIR) $(NTPOLY_LIB) \
		-L$(LIB_DIR) $(SCALAPACK_LIB) $(LAPACK_LIB) $(BLAS_LIB) $(MATH_LIBS) \
		$(OPENMPI_F_LIB) $(FORTRAN_LIB)

test_init_blacs: test_init_blacs.o
	$(CXX) $(CXXFLAGS) -o test_init_blacs test_init_blacs.o \
		-L$(LIB_DIR) $(SCALAPACK_LIB) $(LAPACK_LIB) $(BLAS_LIB) $(MATH_LIBS) \
		$(OPENMPI_F_LIB) $(FORTRAN_LIB)

test_init_blacs.o: test_init_blacs.cpp
	$(CXX) $(CXXFLAGS) -c test_init_blacs.cpp

test_loadBCDMatrixFromABACUSFile: test_loadBCDMatrixFromABACUSFile.o simple_ntpoly.o utils.o
	$(CXX) $(CXXFLAGS) -o test_loadBCDMatrixFromABACUSFile test_loadBCDMatrixFromABACUSFile.o simple_ntpoly.o utils.o \
		-L$(NTPOLY_DIR) $(NTPOLY_LIB) \
		-L$(LIB_DIR) $(SCALAPACK_LIB) $(LAPACK_LIB) $(BLAS_LIB) $(MATH_LIBS) \
		$(OPENMPI_F_LIB) $(FORTRAN_LIB)

test_constructBCDFromPSMatrix: test_constructBCDFromPSMatrix.o simple_ntpoly.o utils.o
	$(CXX) $(CXXFLAGS) -o test_constructBCDFromPSMatrix test_constructBCDFromPSMatrix.o simple_ntpoly.o utils.o \
		-L$(NTPOLY_DIR) $(NTPOLY_LIB) \
		-L$(LIB_DIR) $(SCALAPACK_LIB) $(LAPACK_LIB) $(BLAS_LIB) $(MATH_LIBS) \
		$(OPENMPI_F_LIB) $(FORTRAN_LIB)

simple_ntpoly_test.o: simple_ntpoly_test.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -c simple_ntpoly_test.cpp

simple_ntpoly.o: simple_ntpoly.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -D__NTPOLY -c simple_ntpoly.cpp

test_loadBCDMatrixFromABACUSFile.o: test_loadBCDMatrixFromABACUSFile.cpp
	$(CXX) $(CXXFLAGS) -c test_loadBCDMatrixFromABACUSFile.cpp

test_constructBCDFromPSMatrix.o: test_constructBCDFromPSMatrix.cpp
	$(CXX) $(CXXFLAGS) -c test_constructBCDFromPSMatrix.cpp

utils.o: utils.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -c utils.cpp

clean:
	rm -f *.o simple_ntpoly_test
