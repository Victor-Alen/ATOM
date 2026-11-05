# 启动说明

默认是三个优化点全部开启的单卡 BF16 版本，MTP 关闭。

```bash
bash scripts/start_server.sh
```

脚本完成依赖/源码校验、加载与 Graph 预热，并在真实生成 8 个 token 后打印 READY。
完整开关、累计消融、设备映射、日志位置与停止方式见 [交付说明](../README-CSIG.md)。
