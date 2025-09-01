#!/bin/awk -f
BEGIN{
	print "%%MatrixMarket matrix coordinate real general";
	print "%";
}
NR==1{
	print NF,NF,NF*NF;
}
{
	for(i=1; i<=NF; ++i)
	{
		if($i != 0) printf("%d %d %12e\n", NR,i,$i);
	}
}
