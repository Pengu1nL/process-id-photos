---
name: process-id-photos
description: 在完全本地环境中使用 MODNet ONNX 对 JPG/JPEG 证件照做人脸参考裁切、连续 Alpha Matting、纯色换底、目标尺寸与 JPEG 体积约束，并生成规格 QA、视觉质检联系表和机器可读报告。用于批量制作、换底、统一构图、压缩及验收证件照，尤其适合要求不上传照片、不使用生成式模型、保持原文件名且不得覆盖原图的任务。
---

# 本地批量处理证件照

## 坚守边界

- 仅使用本地、确定性代码和用户提供的本地 MODNet ONNX 模型。不得调用生图、生成式填充、第三方图像服务或远程推理。
- 使用连续 Alpha Matting。不得改用 GrabCut，不得硬二值化 Alpha，不得腐蚀人物，不得以主连通域清理人物；这些操作会误删细发、马尾和颈侧长发。
- 始终显式指定源目录、全新输出目录和报告路径。不得覆盖源图；不得令源目录与输出目录相同或互相嵌套。
- 保持 JPG/JPEG 文件名不变，剥离源元数据。若输入含 PNG、TIFF 等格式，先停止并确认转换与改名方案。
- 不引入上传、授权、表格 token 或业务系统逻辑，不把照片、姓名、凭据或业务路径写进本 Skill。

## 执行流程

1. 确认目标宽高、背景 RGB、JPEG 字节上下限、数量和构图参考。未指定时可建议 `240x320`、`66,142,217`、`15360–70000`，但必须作为可修改默认值。
2. 选择已安装 `numpy`、`Pillow`、`opencv-python`、`onnxruntime` 的本地 Python，并取得本地 MODNet ONNX 路径。确认源、输出、报告、QA 和模型路径位于非云同步的本地存储；脚本不发起网络请求，但无法识别所有映射盘或同步客户端。若依赖或模型缺失，停止；不要擅自安装服务或上传照片。
3. 先运行单张样片。为样片使用全新的输出目录与报告路径，并传 `--limit 1`：

```powershell
& <python> "<skill-dir>\scripts\process_id_photos.py" `
  --source-dir "<source-dir>" `
  --output-dir "<new-sample-output-dir>" `
  --report "<new-sample-report.json>" `
  --model "<local-modnet.onnx>" `
  --target-size 240x320 `
  --background-rgb 66,142,217 `
  --min-bytes 15360 `
  --max-bytes 70000 `
  --background-tolerance 12 `
  --modnet-short-edge 512 `
  --limit 1
```

4. 对样片做独立前向与视觉质检；`qa-dir` 必须是全新目录：

```powershell
& <python> "<skill-dir>\scripts\qa_id_photos.py" `
  --source-dir "<source-dir>" `
  --output-dir "<new-sample-output-dir>" `
  --batch-report "<new-sample-report.json>" `
  --model "<local-modnet.onnx>" `
  --qa-dir "<new-sample-qa-dir>"
```

5. 检查 `qa_report.json` 和全部联系表，重点放大头发边缘、马尾、颈侧长发、肩颈、衣物内部与人物近边缘背景。自动标记只是分诊信号；不得仅凭“0 个标记”跳过人工目检。
6. 样片通过后，用另一套全新输出目录、报告路径和 QA 目录运行全批。已知数量时传 `--expected-count`；不要在正式批次沿用 `--limit`。
7. 仅在处理报告为 `pass`、QA 无规格失败、视觉标记已人工处理且联系表已逐张检查后交付。任一脚本非零退出时停止，不得把部分目录当成成功结果。

## 构图与异常覆盖

- 默认构图按目标尺寸等比例缩放已验证的脸框参考；用 `--composition-face-box x,y,w,h` 修改目标脸框。
- Haar 未检出、误检或构图需人工批准时，提供 `--overrides-json`。仅在必要时读取 [quality-gates.md](references/quality-gates.md) 中的通用 schema；不要把某次任务的 override 文件放入 Skill。
- 保持 `--modnet-short-edge 512` 为默认。仅在样片验证和人工复核后尝试更高值；`768/1024` 曾造成衣物误抠洞。
- 需要解释 JPEG 体积策略、报告状态、视觉指标或限制时，读取 [quality-gates.md](references/quality-gates.md)。

## 解释退出状态

- `process_id_photos.py`：`0` 表示所有选中照片处理、规格验证并整批提交；`1` 表示失败，正式输出不应视为可交付。
- `qa_id_photos.py`：`0` 表示自动规格门禁通过，但仍需人工检查联系表；`1` 表示规格或执行失败；`2` 表示视觉标记要求人工复核。
- 仅在用户明确接受具体视觉标记后使用 `--allow-visual-flags`。即使退出 `0`，报告仍保留标记和人工复核要求。

## 交付记录

报告实际参数、处理/失败数量、输出规格、背景容差、模型哈希、文件哈希、QA 状态和人工复核结论。明确说明全程本地、未使用生成式模型、未调用外部图像服务，并列出仍需人工判断的限制。报告含原文件名、哈希和构图数据，联系表含照片派生画面；把它们视为敏感数据，仅存于本地并按用户的数据保留要求清理。
