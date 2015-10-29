# LokiCNN

[!This version is very similar, all the unit test passes. However, compiled network does not run due to some concurrency bug. (CPU version runs). For Loki version, please go to the directory:
``
/local/scratch/chw43/LokiCNN
``
]


For module-specific documentation, view html/index.html for a doxygen style UI

The overview is provided in the report

# Compilation

The three models provided are:
-- mnist
-- cifar
-- german

They can be compiled with
``
	make host-[model_name]
	make loki-[model_name]
``
for CPU and LOKI compilation respectively

# FLAGS

in src/setting.h

1. the number of cores used in each module can be specified
2. the algorithm used in Convolutional layer can be
specified
3. the datatype can be specified
4. whether to print the debugging info can be specified

# Old Modules

in src/lokicnn.h

By default, matrix_vector algorithm is used for
parallelizing fully_connected layer, so
fully_connected_layer.h is included, which includes
"gemv_loki.h".

However, in old versions, each core takes a layer
of fully connected layer, so one can include
"fully_connected_layer_MultiCore.h" instead as well

In math_functions.h, one can specify which version
of gemm_loki to be used. There are multiple options


# GEMM Test

Matrix multiplication by itself can be used
as a benchmark.

Compile the unit test module with the following:
``
make loki-gemm-test
``

Afterwards, one can run the test with:
``
lokisim -run build/loki-gemm-test --args M N K
``
where M N K specifies the size of the matrices

one can modify the include section in test/gemm_test.c to change which module to test with

There are following options:

1. gemm_loki.h this is the most optimized module. Although there is still a bubble in line 242 due to
a simulator bug (discussed with Alex Chadwick)

2. gemm_loki_limited_stable.h this is a more stable
version

3. gemm_loki_macro_p.h this version is expected to
never return errors, while the other two, in rare cases may return invalid_memory_access. (In general, the error shall not happen as discussed in the report)

4. gemm_loki_forward.h this version have core_0 to fetch A, and then forward the data to the other cores

5. gemm_loki_multitile.h this version enables multiple tiles. Note, in some cases, this version reports a lot of error entries. This is apparently because the remote tile does not do the loop condition in the outermost loop correctly. Adding an assert there will fix the problem.

When running LokiCNN, makes sure one uncomment the 16 srai to turn matrix multiplication to FIX8

When running gemm test, makes sure one comment these out to run integer matrix multiplication. 