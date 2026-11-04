# NUS 病理 AI 实习代码

只放自己写的代码；模型、数据和实验结果放在服务器 `/data/nas_tmp`，不上传。

## aspect_eval/
PathoVernier 评测相关脚本，依赖官方 ASPECT 代码（github.com/ChyaZhang/ASPECT），运行时用 `PYTHONPATH` 指向它。

- `qwen_baseline_eval.py`：按 ASPECT 论文附录 D.4 的基线提示词评测原版 Qwen3-VL-8B-Instruct，支持断点续跑；`--full-question` 保留题目里的计数键名说明（论文基线设置）。
- `diagnose_coverage.py`：只读已有评测结果，按题型统计计数部分的格式问题（非 JSON、漏结尾标签、没写、缺键、被截断），排查基线计数覆盖率偏低的原因。
- `check_conic_lizard.py`、`check_consep.py`：检查数据集图像重建是否正确。

## roi_benchmark/
用医生阅片轨迹构建 ROI 定位基准。
- `viewing_heatmap.py`：按 Ghezloo et al., J Imaging Inform Med 2024（及其公开代码 fGhezloo/ROI-Localization-melanoma）从阅片轨迹提取看片重要区域（放大峰、注视 >2 秒、慢速平移），生成时长热力图，并与医生画的框比较（IoU/precision/recall/F1、CC、NSS）。每一步的出处和我们自己的适配写在文件开头。
- `batch_stats.py`：对某个部位的全部轨迹批量运行上面的分析，输出 CSV 和中位数汇总，不读切片。
- 旧版 `dwell_heatmap.py`（把时间按视野面积平摊，无文献依据）已删除，见 git 历史。

## server_setup/
- `3_autodl_setup.sh`、`4_server_setup.sh`：服务器环境搭建脚本（token 不写进脚本，自己登录）。
