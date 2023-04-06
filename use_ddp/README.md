[TOC]

# 参考

- [「新生手册」：PyTorch分布式训练](https://zhuanlan.zhihu.com/p/360405558)
- [PyTorch分布式训练简明教程(2022更新版)](https://zhuanlan.zhihu.com/p/113694038)
- [从 PyTorch DDP 到 Accelerate 到 Trainer，轻松掌握分布式训练](https://www.cnblogs.com/huggingface/p/17126220.html)
- [DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/master/distributed.html)
- [⭐transformers 的性能指南](https://huggingface.co/docs/transformers/v4.27.2/en/performance)

# 问题

单GPU没问题. 多卡运行的时候会报错, 但我本地没有设备复现

```
RuntimeError: Invalid mt19937 state
```

- https://github.com/huggingface/accelerate/issues/934
- https://github.com/huggingface/accelerate/issues/190
- https://github.com/huggingface/accelerate/issues/1209
- https://github.com/pytorch/pytorch/issues/1637

