# NUS 病理 AI 实习代码

只放自己写的代码；模型、数据和实验结果放在服务器 `/data/nas_tmp`，不上传。

## aspect_eval/
PathoVernier 评测相关脚本，依赖官方 ASPECT 代码（github.com/ChyaZhang/ASPECT），运行时用 `PYTHONPATH` 指向它。

- `qwen_baseline_eval.py`：按 ASPECT 论文附录 D.4 的基线提示词评测原版 Qwen3-VL-8B-Instruct，支持断点续跑；`--full-question` 保留题目里的计数键名说明（论文基线设置）。
- `diagnose_coverage.py`：只读已有评测结果，按题型统计计数部分的格式问题（非 JSON、漏结尾标签、没写、缺键、被截断），排查基线计数覆盖率偏低的原因。
- `check_conic_lizard.py`、`check_consep.py`：检查数据集图像重建是否正确。

## roi_benchmark/
用医生阅片轨迹构建 ROI 定位基准。
- `dwell_heatmap.py`：从一条阅片轨迹算停留热力图（按时间摊到视野面积上，去掉长时间空档），和医生画的框一起叠加到切片缩略图上，并统计停留时间落在框内的比例。

## server_setup/
- `3_autodl_setup.sh`、`4_server_setup.sh`：服务器环境搭建脚本（token 不写进脚本，自己登录）。
