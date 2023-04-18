# OCCL
OneFlow Collecitve Communication Library.
A Deadlock-free Library for GPU Collective Communication.


## Compiling

```shell
make -j<n>
```
> Using `-gencode=arch=compute_86,code=sm_86` for `NVCC_GENCODE` by default. Set the `NVCC_GENCODE` environment variable when needed.
