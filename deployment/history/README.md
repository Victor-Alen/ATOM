# 外部算子仓库的增量提交历史

每个 bundle 保留对应冻结基点之后的提交及父子关系，不 squash；`index.json` 给出基点、分支头与来源。复原时先取得相应 ROCm 上游仓库中的基点，再用 `git fetch <bundle> HEAD` 导入增量历史。`patch-series` 可直接审阅逐次修改。工作区尚未提交的 AITER 差异另存于上一层的 `aiter.working-tree.patch`，最终部署源码快照及每个文件的 SHA256 列于 `runtime-manifest.json`。

ATOM 主仓库使用自身完整提交历史；此处用于保留外部 AITER/CK 内核工作，不将它们压成无历史的单个补丁。
