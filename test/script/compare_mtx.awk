#!/usr/bin/awk -f
# 此 AWK 脚本用于比较两个 MatrixMarket 格式的矩阵文件，输出对应元素的差值。
# 具体流程如下：
# 1. 从两个输入文件的第 3 行读取矩阵的大小。
# 2. 从第 4 行开始，收集两个文件中矩阵非零元素的位置和值。
# 3. 检查两个矩阵的大小是否一致，若不一致则输出错误信息并退出。
# 4. 遍历矩阵的每个位置，计算相同位置元素的差值并输出。

# 使用方法：
# 假设两个 MatrixMarket 格式的输入文件分别为 matrix1.mtx 和 matrix2.mtx，
# 可在终端中运行以下命令：
# awk -f /path/to/compare_mtx.awk matrix1.mtx matrix2.mtx
# 脚本执行后，会在终端输出每个对应位置元素差值的绝对值，以及最大差值和平均差值。
# 若两个矩阵大小不一致，脚本会输出错误信息并终止运行。

# Define the abs function
function abs(x) {
    return x < 0 ? -x : x
}
# readin the first file
NR == FNR {
    if (FNR == 3) {
        N1 = $1;
    }
    if (FNR > 3) {
        idx1 = $1 "\t" $2;
        val1[idx1] = $3;
    }
}
# readin the second file
NR != FNR {
    if (FNR == 3) {
        N2 = $1;
    }
    if (FNR > 3) {
        idx2 = $1 "\t" $2;
        val2[idx2] = $3;
    }
}
END {
    # Check if N1 and N2 are initialized
    if (!N1 || !N2) {
        print "Error: Matrix size information not found in input files!";
        exit 1;
    }
    # check the size
    if (N1 != N2) {
        print "Error: the size of the two files are not the same!";
        exit 1;
    }
    max_diff=0;
    avg_diff=0;
    NZ=0;
    # check the value
    for (ix = 1; ix <= N1; ++ix) {
        for (iy = 1; iy <= N1; ++iy) {
            idx = ix "\t" iy;
            # if the idx is exist in val1 or val2, check the value difference
            if (idx in val1 || idx in val2) {
                # Use 0 if the index is not present in one of the arrays
                diff = (idx in val1 ? val1[idx] : 0) - (idx in val2 ? val2[idx] : 0);
                print idx "\t" diff;
                NZ++;
                avg_diff+=abs(diff);
                if(abs(diff)>max_diff)
                {
                    max_diff=abs(diff);
                }
            }
        }
    }
    avg_diff=avg_diff/NZ;
    print "max_diff =",max_diff;
    print "avg_diff =",avg_diff
}