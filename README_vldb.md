This documentation illustrates how experiments in the submitted paper to VLDB are reproduced.

OCCL is implemented based on NCCL 2.12.12, please refer the [NCCL repo](https://github.com/NVIDIA/nccl) for compiling and installing. 

---

Experiments conducted in Sec. 5.2 rely on our [modified NCCL-Tests](https://github.com/Panlichen/nccl-tests), which are implemented based on official NCCL-Tests, too, please refer to the [official NCCL-Tests repo](https://github.com/NVIDIA/nccl-tests)  for compiling.

Two scripts are used to run experiments in our modified NCCL-Tests:

```shell
bash nccl_tests.sh <NUM_CARDS> <COLL_FUNC> <BUFFER_SIZE>
bash ofccl_tests.sh <NUM_CARDS> <COLL_FUNC> <BUFFER_SIZE>
```

The above commands evaluates NCCL and OCCL, and reports the bandwidth and end-to-end latency. By running these two commands properly we get results reported in Fig. 6.

To get the time of "Read SQE", "Extra Overheads", and "Write CQE" as shown in Fig. 5, we need to uncomment the `DEBUG_CLOCK` and `DEBUG_CLOCK_IO` macros in [collectives_ofccl.h](/src/include/collectives_ofccl.h), recompile OCCL, and run the ofccl_tests.sh. 

The Core Execution Time of OCCL reported in Fig. 7 can also be got in this way. 

The Core Execution Time of NCCL is got via Nsight System and running nccl_tests.sh after recompiling when uncommenting the `NCCL_DEBUG_CLOCK` macro in [collectives.h](/src/include/collectives.h) and [common.h](https://github.com/Panlichen/nccl-tests/blob/master/src/common.h).

----

Experiments for 5.3 rely on our [modified OneFlow](https://github.com/Oneflow-Inc/oneflow/tree/ofccl_dev), please refer to the [official documentation of OneFlow](https://github.com/Oneflow-Inc/oneflow) for compiling and installing.

---

Training Resnet50 rely on the [models repo](https://github.com/Panlichen/models) that is forked from the official [Oneflow-Inc/models](https://github.com/Oneflow-Inc/models). 

Results in Fig. 8 can be reproduced with [train_ofccl_graph_distributed_fp32.sh](https://github.com/Panlichen/models/blob/test_ofccl/Vision/classification/image/resnet50/examples/train_ofccl_graph_distributed_fp32.sh):

```shell
bash train_ofccl_graph_distributed_fp32.sh <NUM_CARDS>
```

Results in Fig.9 were recorded when the stickiness adjustment scheme was not implemented well. The number of context switch and the task queue length can be reported when we uncomment the `DEBUG_CLOCK`, `DEBUG_CLOCK_3D`, and `DEBUG_CLOCK_3D_HOST` macros in [collectives_ofccl.h](/src/include/collectives_ofccl.h), recompile OCCL and OneFlow, and then run train_ofccl_graph_distributed_fp32.sh.

----

Training Vision Transformer rely on the [libai repo](https://github.com/Panlichen/libai) that is forked from the official [Oneflow-Inc/libai](https://github.com/Oneflow-Inc/libai).

Please refer to the official [libai documentation](https://libai.readthedocs.io/en/latest/index.html) for installing libai and configuring which parallel DNN training method to use.

Results in Fig. 10 can be reproduced with [tools/train.sh](https://github.com/Panlichen/libai/blob/main/tools/train.sh) and other python files:

```shell
bash tools/train.sh tools/train_net.py configs/vit_imagenet.py <NUM_CARDS>
```

---

All the figures are drawn with python scripts in [occl_figure repo](https://github.com/Panlichen/occl_figure), which also include the raw data of the figures.
