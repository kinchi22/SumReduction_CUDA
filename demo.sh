make

NumElements=500000000
./SumReduction $NumElements cpu
./SumReduction $NumElements cpu_ur
./SumReduction $NumElements g_atom
./SumReduction $NumElements s_atom
./SumReduction $NumElements binary
./SumReduction $NumElements shfl_a
./SumReduction $NumElements shfl_s
./SumReduction $NumElements shfl_q
