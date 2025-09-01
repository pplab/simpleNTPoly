#!/bin/awk -f
BEGIN{
    print "%%MatrixMarket matrix coordinate real general";
    print "%";
}
NR==1{
    N=$1;
    print N, N, N*N;
    for(i=NR; i<=N; ++i)
    {
        val=$(i+1);
        #if(val != 0) printf("%d %d %12e\n", NR,i,val);
        if(val>Threshold || val< -Threshold ) 
        {
            printf("%d %d %12e\n", NR,i,val);
			if(i != NR) printf("%d %d %12e\n", i,NR,val);
        }
    }
    next;
}
NR>1{
    for(i=NR; i<=N; ++i)
    {
        val=$(i-NR+1);
        #if(val != 0) printf("%d %d %12e\n", NR,i,val);
        if(val>Threshold || val< -Threshold ) # printf("%d %d %12e\n", NR,i,val);
        {
            printf("%d %d %12e\n", NR,i,val);
            if(i != NR) printf("%d %d %12e\n", i,NR,val);
        }
    }
}
