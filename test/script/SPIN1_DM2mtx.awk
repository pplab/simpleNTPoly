#!/usr/bin/awk -f
# 此 AWK 脚本用于将ABACUS输出的SPIN_DM格式的文件转换为 MatrixMarket 格式的矩阵文件。
# 具体流程如下：
# 1. 识别包含 "fermi energy" 的行，其后一行包含矩阵的大小。
# 2. 读取矩阵大小后，跳过一行分隔符，开始处理矩阵元素。
# 3. 收集所有非零元素，并记录其位置和值。
# 4. 最后输出符合 MatrixMarket 规范的矩阵文件，包含矩阵的维度和非零元素信息。

# 使用方法：
# 假设输入文件名为SPIN1_DM，可在终端中运行以下命令：
# awk -f /path/to/SPIN1_DM2mtx.awk SPIN1_DM > output.mtx
# 其中 output.mtx 为生成的 MatrixMarket 格式文件。

/fermi energy/ {next_is_size=1; next}
next_is_size == 1 {
    N=$1;
    next_is_size=2;
    next;
}
next_is_size == 2 {
    next_is_size=0;
    next_is_value=1;
    idx=0;
    nz=0;
    print "%%MatrixMarket matrix coordinate real general";
    print "%";
    next; 
}
next_is_value == 1 {
    for(i=1; i<=NF; ++i)
    {
        if($i > 0 || $i < 0)
        {
            ix=idx%N;
            iy=int(idx/N);
            nz_idx=ix+1"\t"iy+1;
            nz_val[nz_idx]=$i;
            nz++;
        }        
        idx++;
    }
}
END{
    print N, N, nz;
    for(ix=1; ix<=N; ++ix)
    {
        for(iy=1; iy<=N; ++iy)
        {
            nz_idx=ix"\t"iy;
            print nz_idx"\t"nz_val[nz_idx];
        }
    }
}