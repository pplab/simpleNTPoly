#!/bin/sh
# compare matrix of different formats
# input matrix formats are:
# 1. data-0-H or data-0-S dumped from ABACUS
# 2. full matrix saved from simpleNTPoly: H_save.dat or S_save.dat

usage() {
    echo "Usage: $0 data-0-H H_save.dat or $0 data-0-S S_save.dat"
    exit 1
}

# check input
if [ $# -ne 2 ]; then
    usage
fi
if [ ! -f $1 ]; then
    echo "File $1 does not exist"
    usage
fi
if [ ! -f $2 ]; then
    echo "File $2 does not exist"
    usage
fi

# compare matrix
awk 'BEGIN {is_diff=0;}
NR==1{
    N=$1;
    for(i=1;i<=N;i++)
    {
        idx=1","i;
        H[idx]=$(i+1);
    }
    next;
}
NR==FNR{
    for(i=FNR;i<=N;i++)
    {
        idx=FNR","i;
        H[idx]=$(i-FNR+1);
        # print i-FNR+1, $(i-FNR+1), idx,H[idx];
    }
}
NR>FNR{
    for(i=1;i<=N;i++)
    {
        if(i<=FNR)
            idx=i","FNR;
        else
            idx=FNR","i;
        H_in=H[idx];
        H_out=$i;
        diff=H_in-H_out;
        if(diff>1e-6 || diff<-1e-6)
        {
            printf("H_in[%s]=%f, H_out[%s]=%f, diff=%f\n",idx,H_in,idx,H_out,diff);
            is_diff=1;
        }        
    }
}
END{
if(is_diff==0)
    print "two matrices are same";
}' $1 $2