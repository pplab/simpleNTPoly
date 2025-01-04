include make.inc.aocc

simple_ntpoly_test: simple_ntpoly_test.o simple_ntpoly.o utils.o
	$(CXX) $(CXXFLAGS)  -o simple_ntpoly_test simple_ntpoly_test.o simple_ntpoly.o utils.o \
		-L$(NTPOLY_DIR) $(NTPOLY_LIB) \
		-L$(LIB_DIR) $(SCALAPACK_LIB) $(LAPACK_LIB) $(BLAS_LIB) $(MATH_LIBS) \
		$(OPENMPI_F_LIB) $(FORTRAN_LIB)

simple_ntpoly_test.o: simple_ntpoly_test.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -c simple_ntpoly_test.cpp

simple_ntpoly.o: simple_ntpoly.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -D__NTPOLY -c simple_ntpoly.cpp

utils.o: utils.cpp
	$(CXX) $(CXXFLAGS) -I$(NTPOLY_INC) -c utils.cpp

clean:
	rm -f *.o simple_ntpoly_test
