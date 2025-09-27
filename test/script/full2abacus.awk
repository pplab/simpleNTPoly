#!/usr/bin/awk -f
NR==1{
    printf NF;
}
{
    for(i=NR;i<=NF;++i)
  		printf " "$i;
    printf "\n";
}

