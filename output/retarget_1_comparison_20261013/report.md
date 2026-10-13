# retarget_1.json 求解对比运行记录

文件已上传至 `tests/retarget_1.json`，SHA-256 校验一致。

当前工作目录和历史 `main~2 = adde3216860a76b70f3bfda462ab48e50ab9cba4` 均以 `ex001_wuji` 尝试求解同一份、未改动的 JSON。历史代码由 Git 导出至独立目录，没有回退生产 checkout。

## 结论

**没有生成两份完整轨迹，因此无法判断轨迹是否完全一致。**

- 当前版本：batch summary 为 skipped，完成轨迹数为 0。第 131–248 帧共 118 帧无法关联原始 EMF，连续缺失超过 5 帧，求解器主动跳过。批处理退出码 0 不能视为求解成功。
- 历史版本：求解失败，报 `No camera frame has exact left/right EMF payload matches`。上传文件缺少旧版要求的左右手 `glove_emf_poses` 字段。
- 上传 JSON 有 249 帧、跨度 12.4 秒；原始同名采集 JSON 有 195 帧、跨度 6.527677952 秒。上传文件使用标准化 source_mano21 字段及重排后的时间戳，不能直接作为旧 EMF pipeline 的原始输入。

## 命令适配

原始参数组合 `--config ... --test_data tests/retarget_1.json` 报错：两项不能并用。batch 输入要求 collection 目录。测试将原 JSON 链接到独立 episode 输入目录，配置副本只替换输入集合与输出目录。原始 EMF、TF 与视频取自上传 JSON 的 name 所指向的原始采集目录。

系统 `python3` 缺少 warp，实际运行使用已有 pi0 环境 Python。原 YAML、求解器文件和已有轨迹未改动。用于保存中间数组的包装脚本没有更改求解逻辑。

详细路径和状态见 `comparison_report.json`；运行日志为 `current.log`、`historical.log`。`historical_native.npz` 只是原采集 SDK 手指中间结果，不是 retarget_1.json 的完整轨迹。

后续要完成对比，需改用含原始 EMF 字段与原始时间戳的采集 JSON，或适配两版 pipeline 以标准化 source_mano21 数据直接求解。
