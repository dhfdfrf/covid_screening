$ErrorActionPreference = "Stop"

$Root = "D:\covid_screening"
$BasePpt = Join-Path $Root "docs\covid_screening_project_v18aware_detailed_v4.pptx"
$DefenseOut = Join-Path $Root "docs\covid_screening_project_v18aware_defense_code_redone.pptx"
$StudyOut = Join-Path $Root "docs\covid_screening_project_v18aware_study_code_walkthrough.pptx"

function RGB($r, $g, $b) {
    return [int]($r + ($g * 256) + ($b * 65536))
}

function Add-BlankSlide($presentation) {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $bg = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
    $bg.Fill.ForeColor.RGB = RGB 244 249 251
    $bg.Line.Visible = 0
    $bg.ZOrder(1) | Out-Null
    $accent = $slide.Shapes.AddShape(1, 0, 34, 960, 4)
    $accent.Fill.ForeColor.RGB = RGB 14 116 144
    $accent.Line.Visible = 0
    return $slide
}

function Add-TopBar($slide, $section) {
    $bar = $slide.Shapes.AddShape(1, 0, 0, 960, 34)
    $bar.Fill.ForeColor.RGB = RGB 8 47 73
    $bar.Line.Visible = 0
    $mark = $slide.Shapes.AddShape(1, 0, 0, 12, 34)
    $mark.Fill.ForeColor.RGB = RGB 220 38 38
    $mark.Line.Visible = 0
    $t = $slide.Shapes.AddTextbox(1, 28, 7, 760, 20)
    $t.TextFrame.TextRange.Text = $section
    $t.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $t.TextFrame.TextRange.Font.Size = 10
    $t.TextFrame.TextRange.Font.Color.RGB = RGB 229 231 235
}

function Add-Title($slide, $title, $subtitle = "") {
    $line = $slide.Shapes.AddShape(1, 42, 55, 6, 45)
    $line.Fill.ForeColor.RGB = RGB 14 116 144
    $line.Line.Visible = 0
    $box = $slide.Shapes.AddTextbox(1, 58, 50, 850, 55)
    $box.TextFrame.TextRange.Text = $title
    $box.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $box.TextFrame.TextRange.Font.Size = 27
    $box.TextFrame.TextRange.Font.Bold = -1
    $box.TextFrame.TextRange.Font.Color.RGB = RGB 15 23 42
    if ($subtitle -ne "") {
        $sub = $slide.Shapes.AddTextbox(1, 60, 104, 835, 38)
        $sub.TextFrame.TextRange.Text = $subtitle
        $sub.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
        $sub.TextFrame.TextRange.Font.Size = 13
        $sub.TextFrame.TextRange.Font.Color.RGB = RGB 71 85 105
    }
}

function Add-Text($slide, $text, $left, $top, $width, $height, $size = 15, $color = $null, $bold = $false) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = [single]$size
    if ($null -eq $color) { $color = RGB 51 65 85 }
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = -1 }
    return $shape
}

function Add-Card($slide, $left, $top, $width, $height, $title, $body, $fillColor) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = $fillColor
    $rect.Line.ForeColor.RGB = RGB 148 163 184
    $rect.Line.Weight = 1.2
    Add-Text $slide $title ($left + 12) ($top + 10) ($width - 24) 24 13 (RGB 15 23 42) $true | Out-Null
    Add-Text $slide $body ($left + 12) ($top + 38) ($width - 24) ($height - 44) 10.5 (RGB 51 65 85) $false | Out-Null
}

function Add-Code($slide, $code, $left, $top, $width, $height, $size = 9) {
    $rect = $slide.Shapes.AddShape(1, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = RGB 11 31 51
    $rect.Line.ForeColor.RGB = RGB 14 116 144
    $rect.Line.Weight = 1.5
    $box = $slide.Shapes.AddTextbox(1, $left + 12, $top + 10, $width - 24, $height - 20)
    $box.TextFrame.TextRange.Text = $code
    $box.TextFrame.TextRange.Font.Name = "Consolas"
    $box.TextFrame.TextRange.Font.Size = [single]$size
    $box.TextFrame.TextRange.Font.Color.RGB = RGB 226 232 240
}

function Add-Table($slide, $rows, $left, $top, $width, $height, $fontSize = 10) {
    $rowCount = $rows.Count
    $colCount = $rows[0].Count
    $table = $slide.Shapes.AddTable($rowCount, $colCount, $left, $top, $width, $height).Table
    for ($r = 1; $r -le $rowCount; $r++) {
        for ($c = 1; $c -le $colCount; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.Shape.TextFrame.TextRange.Text = $rows[$r-1][$c-1]
            $cell.Shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
            $cell.Shape.TextFrame.TextRange.Font.Size = [single]$fontSize
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 30 41 59
            if ($r -eq 1) {
                $cell.Shape.Fill.ForeColor.RGB = RGB 8 47 73
                $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 255 255 255
                $cell.Shape.TextFrame.TextRange.Font.Bold = -1
            }
        }
    }
}

function Add-Arrow($slide, $x1, $y1, $x2, $y2, $color) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = 2
    $line.Line.EndArrowheadStyle = 3
}

function Add-TalkNote($slide, $text, $left = 70, $top = 475, $width = 820, $height = 42) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = RGB 255 247 237
    $rect.Line.ForeColor.RGB = RGB 251 146 60
    $rect.Line.Weight = 1.2
    Add-Text $slide $text ($left + 14) ($top + 10) ($width - 28) ($height - 16) 12.5 (RGB 124 45 18) $true | Out-Null
}

function Prepare-Presentation($ppt, $outPath) {
    Copy-Item -LiteralPath $BasePpt -Destination $outPath -Force
    $presentation = $ppt.Presentations.Open($outPath, $false, $false, $false)
    for ($i = $presentation.Slides.Count; $i -ge 28; $i--) {
        $presentation.Slides.Item($i).Delete()
    }
    return $presentation
}

function Add-Defense-CodeSlides($presentation) {
    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "26 代码展示总览"
    Add-Title $slide "答辩时先讲代码怎么串起来" "按请求入口、模型加载、模型前向、门控选择、结果返回五步讲，不陷入细枝末节。"
    Add-Card $slide 45 155 150 95 "1. Web入口" "app_v12_web.py`n接收上传图片和参数" (RGB 219 234 254)
    Add-Card $slide 225 155 150 95 "2. 推理封装" "src/v12_predictor.py`n读图、归一化、调用模型" (RGB 220 252 231)
    Add-Card $slide 405 155 150 95 "3. 模型构建" "model_factory.py`n按名称创建 v18 或 LowDice" (RGB 254 243 199)
    Add-Card $slide 585 155 150 95 "4. 门控选择" "eval_lowdice_expert_gate.py`n按错误类型选专家" (RGB 254 226 226)
    Add-Card $slide 765 155 150 95 "5. 可视化返回" "mask / overlay / metrics`n返回网页展示" (RGB 237 233 254)
    Add-Arrow $slide 195 202 225 202 (RGB 14 116 144)
    Add-Arrow $slide 375 202 405 202 (RGB 14 116 144)
    Add-Arrow $slide 555 202 585 202 (RGB 14 116 144)
    Add-Arrow $slide 735 202 765 202 (RGB 14 116 144)
    Add-Table $slide @(
        @("答辩说法", "重点"),
        @("不是只贴代码", "说明系统从上传到输出的完整链路。"),
        @("不是平均融合", "最终策略是错误类型感知的多专家门控。"),
        @("不是单模型硬堆", "v18 提供稳定基线，LowDice 专家处理困难样本。")
    ) 90 300 780 120 11
    Add-TalkNote $slide "讲法：先用这一页把评委带到主线，后面每页只讲一个关键代码点。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "27 关键代码 1"
    Add-Title $slide "v18 基线模型：稳定分割主干" "v18 负责大多数常规样本，是后续专家门控的基准和回退输出。"
    $code = @'
prior_feat, prior_logits = self.prior_branch(x)
s1 = self.adapter1(self.enc1(x), prior_feat, prior_logits)
s2 = self.adapter2(self.enc2(down(s1)), prior_feat, prior_logits)
s3 = self.adapter3(self.enc3(down(s2)), prior_feat, prior_logits)
s4 = self.adapter4(self.enc4(down(s3)), prior_feat, prior_logits)
tokens = self.transformer(flatten(down(s4)), h, w)
d1 = decode_with_spatial_gates(tokens, [s4, s3, s2, s1])
seg = coarse_out(d1) + refine(d1, coarse_out(d1))
return {"seg": seg, "prior": prior_logits, "boundary": boundary_head(d1)}
'@
    Add-Code $slide $code 42 142 560 260 9
    Add-Card $slide 630 145 260 64 "prior_branch" "从灰度、低频、高频、边缘和 Laplace 响应中提取病灶先验。" (RGB 219 234 254)
    Add-Card $slide 630 222 260 64 "adapter1-4" "把先验注入不同尺度特征，增强模型对病灶边界和纹理的关注。" (RGB 220 252 231)
    Add-Card $slide 630 299 260 64 "transformer" "在瓶颈层建模全局上下文，补足卷积局部感受野不足。" (RGB 254 243 199)
    Add-Card $slide 630 376 260 64 "输出字典" "seg 是分割结果，prior 是先验监督，boundary 是边界辅助输出。" (RGB 254 226 226)
    Add-TalkNote $slide "讲法：v18 的创新点是频域先验残差注入，但最终策略不是只靠 v18，而是把它作为稳定基线。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "28 关键代码 2"
    Add-Title $slide "LowDice 专家：专门处理困难样本" "低分样本通常是小病灶、低对比度、边界错位或大面积漏检。"
    $code = @'
gray = x[:, :1]
low  = avg_pool(gray, kernel=9)
high = abs(gray - low)
edge = sqrt(sobel_x(gray)^2 + sobel_y(gray)^2)
lap  = abs(laplace(gray))
xx, yy = normalized_coord(gray.shape)
prior = concat([gray, low, high, edge, lap, xx, yy])
feat = self.encoder(self.stem(concat([x, prior])))
feat = self.aspp(feat)
d1 = self.local_refine(concat([self.decoder(feat), prior]))
return {"seg": self.seg_head(d1)}
'@
    Add-Code $slide $code 42 142 560 300 9
    Add-Card $slide 630 145 260 58 "low / high" "low 表示整体亮度趋势，high 表示局部纹理差异。" (RGB 219 234 254)
    Add-Card $slide 630 214 260 58 "edge / lap" "用 Sobel 和 Laplace 强化边界信息，减少边缘错位。" (RGB 220 252 231)
    Add-Card $slide 630 283 260 58 "xx / yy" "坐标先验告诉模型左右肺和上下位置，减少位置混淆。" (RGB 254 243 199)
    Add-Card $slide 630 352 260 58 "aspp + refine" "多尺度上下文加局部细化，针对困难病灶修正形状。" (RGB 254 226 226)
    Add-TalkNote $slide "讲法：LowDice 专家不是替代所有样本，而是对 v18 已经明显失败的样本做专项纠错。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "29 关键代码 3"
    Add-Title $slide "v18-aware Gate：按错误类型选专家" "这页是最终优化策略的核心，说明为什么不是简单平均融合。"
    $rows = @(
        @("错误类型", "优先专家", "答辩解释"),
        @("over_segmented_fp", "actual_fp / precision", "预测太大，优先用抑制误检的专家。"),
        @("under_segmented_fn", "actual_fn / recall", "真实病灶没覆盖够，优先用提高召回的专家。"),
        @("boundary_or_shift_error", "basev18 / boundary", "大体位置对但边界偏，先保留稳定基线或边界专家。"),
        @("wrong_location_no_overlap", "actual_boundary / precision", "位置错得明显，需要更强的边界和保守预测。"),
        @("missed_all", "actual_all / actual_fn", "完全漏检时，用整体低分专家或漏检专家补救。")
    )
    Add-Table $slide $rows 52 140 856 255 10
    $code = @'
for preferred in routing[failure_type]:
    if preferred in available:
        return preferred
return "basev18"
'@
    Add-Code $slide $code 110 420 330 70 10
    Add-Text $slide "这几行代码的含义：按候选顺序逐个找可用专家，找到就返回；如果都不可用，就回退到 basev18，保证系统稳定。" 480 420 380 62 12 (RGB 51 65 85) $true | Out-Null

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "30 关键代码 4"
    Add-Title $slide "网页端推理：从上传图片到返回结果" "答辩时只讲输入、模型调用、后处理、输出四件事。"
    $code = @'
@app.post("/api/predict")
async def predict(file, checkpoint, threshold=0.25, min_area=32):
    image_path = save_upload(file)
    result = segmenter.segment_file(
        image_path,
        checkpoint=checkpoint,
        threshold=threshold,
        min_area=min_area,
    )
    metrics = evaluate_if_gt_exists(image_path, result.mask)
    return {
        "original": url(result.original_png),
        "mask": url(result.mask_png),
        "overlay": url(result.overlay_png),
        "metrics": metrics,
    }
'@
    Add-Code $slide $code 42 130 570 320 9
    Add-Card $slide 640 132 245 66 "threshold" "概率阈值，控制模型输出是否判为感染区域。" (RGB 219 234 254)
    Add-Card $slide 640 208 245 66 "min_area" "最小连通域面积，用于过滤孤立噪声点。" (RGB 220 252 231)
    Add-Card $slide 640 284 245 66 "evaluate_if_gt_exists" "如果能找到真值 mask，就自动计算 Dice、IoU、Precision、Recall。" (RGB 254 243 199)
    Add-Card $slide 640 360 245 66 "return JSON" "把原图、mask、overlay 和指标返回给前端页面展示。" (RGB 254 226 226)
    Add-TalkNote $slide "讲法：Web 端证明系统不是只停留在训练脚本，而是能把模型输出可视化给用户。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "31 总结"
    Add-Title $slide "答辩版代码展示收束" "最后把代码落到创新点和结果上。"
    Add-Card $slide 70 150 250 220 "模型层面" "v18 提供稳定基线；LowDice 专家补足低分样本；边界、频域和坐标先验增强困难病灶表达。" (RGB 219 234 254)
    Add-Card $slide 355 150 250 220 "策略层面" "v18-aware gate 按错误类型选专家，不做盲目平均融合，避免专家在不适合的样本上拖低结果。" (RGB 254 243 199)
    Add-Card $slide 640 150 250 220 "系统层面" "网页端完成上传、推理、后处理、可视化和指标计算，形成完整展示闭环。" (RGB 220 252 231)
    Add-TalkNote $slide "结尾说法：最终提升来自错误类型分析 + 专家化模型 + 可视化系统，不是单一代码片段。"
}

function Add-Study-CodeSlides($presentation) {
    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "26 学习版导读"
    Add-Title $slide "这版 PPT 怎么学习代码" "先理解名词，再看调用链，最后逐行看关键代码。"
    Add-Card $slide 65 145 245 210 "第一步：名词" "把 tensor、logits、mask、threshold、Dice、FP/FN 等词弄清楚。" (RGB 219 234 254)
    Add-Card $slide 357 145 245 210 "第二步：调用链" "看清楚从网页入口到模型前向，再到后处理和返回 JSON 的顺序。" (RGB 220 252 231)
    Add-Card $slide 650 145 245 210 "第三步：逐行代码" "每页左边放关键代码，右边说明每行做什么、为什么这样做。" (RGB 254 243 199)
    Add-TalkNote $slide "学习建议：先不要背代码，先把数据流记住：image -> prob -> mask -> overlay -> metrics。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "27 名词解释 1"
    Add-Title $slide "模型推理里常见名词是什么意思？" "这些词后面代码页会反复出现。"
    Add-Table $slide @(
        @("名词", "意思", "在本项目中的作用"),
        @("tensor", "张量，多维数组", "图片进入 PyTorch 后的表示，常见形状为 B×C×H×W。"),
        @("logits", "未经过 sigmoid 的模型原始输出", "数值可正可负，用来继续计算损失或转为概率。"),
        @("prob", "概率图", "sigmoid(logits) 后得到，每个像素是感染概率。"),
        @("mask", "二值掩码", "prob > threshold 后得到，1 表示感染，0 表示背景。"),
        @("threshold", "概率阈值", "控制预测保守或激进，阈值越高通常误检越少。"),
        @("min_area", "最小连通域面积", "过滤很小的噪声预测块。")
    ) 50 140 860 315 10
    Add-TalkNote $slide "记忆方式：logits 是模型生输出，prob 是概率，mask 是最终黑白分割图。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "28 名词解释 2"
    Add-Title $slide "评价指标和门控名词是什么意思？" "这页用于理解为什么要分 precision / recall / boundary 专家。"
    Add-Table $slide @(
        @("名词", "意思", "对应错误"),
        @("TP", "预测感染且真实也是感染", "正确区域。"),
        @("FP", "预测感染但真实是背景", "误检，蓝色 error map。"),
        @("FN", "真实感染但模型没预测", "漏检，绿色 error map。"),
        @("Precision", "TP / (TP + FP)", "低说明误检多。"),
        @("Recall", "TP / (TP + FN)", "低说明漏检多。"),
        @("Expert", "专家模型", "只针对某类困难样本训练。"),
        @("Gate", "门控选择器", "按错误类型选择哪个专家输出。")
    ) 50 132 860 330 10
    Add-TalkNote $slide "低分样本不是一种错误，所以才要拆成 FP、FN、boundary、wrong location 等类别。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "29 文件职责地图"
    Add-Title $slide "哪个文件负责哪一段？" "先看文件职责，再看代码跳转。"
    Add-Table $slide @(
        @("文件", "职责", "什么时候被调用"),
        @("app_v12_web.py", "网页接口，接收上传文件", "用户点击预测时。"),
        @("src/v12_predictor.py", "推理封装，读图、加载模型、后处理", "Web 接口内部调用。"),
        @("src/models/model_factory.py", "根据模型名称创建模型", "训练和推理都可能调用。"),
        @("src/models/transunet2d_v18.py", "v18 基线模型结构", "model=transunet2d_v18 时调用。"),
        @("src/models/lowdice_refinenet2d.py", "LowDice 专家模型结构", "训练或评估专家时调用。"),
        @("scripts/eval_lowdice_expert_gate.py", "离线评估门控策略", "测试 v18-aware gate 时调用。"),
        @("src/train_unet2d_qata.py", "训练入口和损失函数", "训练模型权重时调用。")
    ) 40 130 880 342 9.5
    Add-TalkNote $slide "代码跳转主线：Web -> v12_predictor -> model_factory -> model.forward -> postprocess -> Web 返回。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "30 调用链总览"
    Add-Title $slide "整体如何运作？" "这张图就是从用户上传到网页显示结果的代码跳转顺序。"
    Add-Card $slide 45 160 140 78 "用户上传" "浏览器提交图片和参数。" (RGB 219 234 254)
    Add-Card $slide 215 160 140 78 "app_v12_web" "接收请求，保存临时文件。" (RGB 220 252 231)
    Add-Card $slide 385 160 140 78 "V12Segmenter" "加载模型，完成推理。" (RGB 254 243 199)
    Add-Card $slide 555 160 140 78 "model.forward" "输出 logits / prior / boundary。" (RGB 254 226 226)
    Add-Card $slide 725 160 140 78 "postprocess" "阈值化、过滤小连通域、叠加图。" (RGB 237 233 254)
    Add-Arrow $slide 185 199 215 199 (RGB 14 116 144)
    Add-Arrow $slide 355 199 385 199 (RGB 14 116 144)
    Add-Arrow $slide 525 199 555 199 (RGB 14 116 144)
    Add-Arrow $slide 695 199 725 199 (RGB 14 116 144)
    Add-Table $slide @(
        @("跳转", "含义"),
        @("app_v12_web.py -> src/v12_predictor.py", "Web 只负责接收请求，真正推理交给 Segmenter。"),
        @("src/v12_predictor.py -> model_factory.py", "根据 checkpoint 或 model_name 选择模型结构。"),
        @("model_factory.py -> transunet2d_v18.py", "创建 v18 网络，然后加载权重。"),
        @("model.forward -> postprocess", "模型输出概率图，后处理转成可展示结果。")
    ) 80 310 800 145 10.5
    Add-TalkNote $slide "先背这条线，后面每段代码都能挂到这条线上。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "31 Web 代码逐行"
    Add-Title $slide "app_v12_web.py：用户点击预测后发生什么？" "左边是答辩用伪代码，右边逐行解释。"
    $code = @'
@app.post("/api/predict")
async def predict(file, checkpoint, threshold=0.25, min_area=32):
    image_path = save_upload(file)
    cfg = resolve_config(checkpoint)
    result = segmenter.segment_file(
        image_path, checkpoint=cfg.ckpt,
        threshold=threshold, min_area=min_area, tta=cfg.tta
    )
    metrics = evaluate_if_gt_exists(image_path, result.mask)
    return {"original": url(result.original_png),
            "mask": url(result.mask_png),
            "overlay": url(result.overlay_png),
            "metrics": metrics}
'@
    Add-Code $slide $code 35 125 475 350 8.5
    Add-Table $slide @(
        @("代码行", "解释"),
        @("@app.post", "声明网页后端接口，前端向 /api/predict 发送请求。"),
        @("predict(...)", "接收图片、模型权重、阈值和最小面积。"),
        @("save_upload", "把上传文件保存成本地临时图片。"),
        @("resolve_config", "根据选择的 checkpoint 找到配置。"),
        @("segment_file", "跳到 src/v12_predictor.py 做真正推理。"),
        @("evaluate_if_gt_exists", "如果有真值 mask，就计算指标。"),
        @("return", "把图片路径和指标返回给前端。")
    ) 535 125 385 350 8.5

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "32 推理封装逐行"
    Add-Title $slide "src/v12_predictor.py：segment_file 做了什么？" "它是 Web 与模型之间的中间层。"
    $code = @'
def segment_file(path, checkpoint, threshold, min_area, tta=False):
    image = read_grayscale(path)
    x = preprocess(image).to(device)
    model = load_model(checkpoint).to(device)
    logits = model(x)["seg"]
    prob = sigmoid(logits)
    if tta:
        prob = average_tta_predictions(model, x)
    mask = remove_small_components(prob > threshold, min_area)
    overlay = build_overlay(image, mask)
    return SegmentResult(mask=mask, overlay=overlay, prob=prob)
'@
    Add-Code $slide $code 35 132 475 330 9
    Add-Table $slide @(
        @("代码行", "解释"),
        @("read_grayscale", "胸片按灰度图读取。"),
        @("preprocess", "缩放、归一化、转 tensor。"),
        @("load_model", "根据权重加载 v18 或其他模型。"),
        @("model(x)[seg]", "调用模型 forward，拿分割 logits。"),
        @("sigmoid", "把 logits 转为 0-1 概率。"),
        @("prob > threshold", "概率图转二值 mask。"),
        @("build_overlay", "把红色 mask 叠加回原图。")
    ) 535 132 385 330 8.8
    Add-TalkNote $slide "这里最关键的跳转是 load_model -> model_factory -> 具体模型类。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "33 模型构建逐行"
    Add-Title $slide "model_factory.py：模型名字如何变成代码对象？" "训练和推理都需要先构建模型结构，再加载权重。"
    $code = @'
def build_model(name, in_channels=1, out_channels=1):
    name = normalize_model_name(name)
    if name == "transunet2d_v18":
        from src.models.transunet2d_v18 import build_transunet2d_v18
        return build_transunet2d_v18(in_channels, out_channels)
    if name == "lowdice_refinenet2d":
        from src.models.lowdice_refinenet2d import build_lowdice_refinenet2d
        return build_lowdice_refinenet2d(in_channels, out_channels)
    raise ValueError("unknown model")
'@
    Add-Code $slide $code 50 135 520 280 9
    Add-Table $slide @(
        @("代码行", "解释"),
        @("normalize_model_name", "把别名统一成标准模型名。"),
        @("if transunet2d_v18", "选择 v18 基线模型。"),
        @("import build_transunet2d_v18", "跳到 src/models/transunet2d_v18.py。"),
        @("if lowdice_refinenet2d", "选择 LowDice 专家模型。"),
        @("raise ValueError", "输入未知模型名时直接报错，避免加载错模型。")
    ) 610 135 295 280 9
    Add-TalkNote $slide "调用链：segment_file 需要模型时，会通过 model_factory 找到具体模型文件。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "34 v18 代码逐行 1"
    Add-Title $slide "v18 的 FrequencyPriorExtractor 每行做什么？" "这一段把胸片变成频域/边界先验。"
    $code = @'
gray = image.mean(dim=1, keepdim=True)
low = avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
high = gray - low
gx = conv2d(gray, sobel_x, padding=1)
gy = conv2d(gray, sobel_y, padding=1)
edge = sqrt(gx.square() + gy.square() + 1e-6)
lap = conv2d(gray, laplace, padding=1).abs()
feat = encoder(cat([gray, low, high.abs(), edge, lap], dim=1))
prior_logits = head(feat)
'@
    Add-Code $slide $code 35 128 445 310 9
    Add-Table $slide @(
        @("行", "意思"),
        @("gray", "把输入转为灰度通道。胸片本身就是灰度信息为主。"),
        @("low", "平均池化得到低频亮度趋势。"),
        @("high", "原图减低频，得到局部纹理变化。"),
        @("gx / gy", "Sobel 横纵梯度，检测边缘方向。"),
        @("edge", "合成边缘强度图。"),
        @("lap", "Laplace 响应，强调边界和突变。"),
        @("feat", "把这些先验拼接后送入小 encoder。"),
        @("prior_logits", "输出一个辅助病灶先验图。")
    ) 505 128 415 310 8.2

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "35 v18 代码逐行 2"
    Add-Title $slide "v18 的 residual adapter 每行做什么？" "adapter 的作用是把先验注入不同尺度特征。"
    $code = @'
prior_feat = interpolate(prior_feat, size=feat.shape[-2:])
prior_prob = interpolate(sigmoid(prior_logits), size=feat.shape[-2:])
delta = self.net(cat([feat, prior_feat, prior_prob], dim=1))
out = feat + tanh(self.gamma) * delta
return out
'@
    Add-Code $slide $code 55 150 440 170 10
    Add-Table $slide @(
        @("行", "意思"),
        @("interpolate prior_feat", "把先验特征缩放到和当前 encoder 特征一样大小。"),
        @("sigmoid prior_logits", "把先验 logits 变成 0-1 概率。"),
        @("cat", "把原特征、先验特征、先验概率拼在一起。"),
        @("delta", "经过小网络生成要补充的残差信息。"),
        @("gamma", "可学习强度，初始化为 0，避免一开始破坏基线模型。"),
        @("feat + ...", "保留原特征，只加一小部分先验修正。")
    ) 535 125 370 290 9
    Add-TalkNote $slide "这就是 v18 稳定的原因：不是强行替换特征，而是残差式轻量注入。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "36 v18 代码逐行 3"
    Add-Title $slide "v18 forward：一张图如何走完整个模型？" "这是 v18 从输入到输出的主路径。"
    $code = @'
prior_feat, prior_logits = self.prior_branch(x)
s1 = adapter1(enc1(x), prior_feat, prior_logits)
s2 = adapter2(enc2(down(s1)), prior_feat, prior_logits)
s3 = adapter3(enc3(down(s2)), prior_feat, prior_logits)
s4 = adapter4(enc4(down(s3)), prior_feat, prior_logits)
tokens = transformer(flatten(down(s4)), h, w)
d1 = decoder(tokens, skip=[s4, s3, s2, s1])
coarse = coarse_out(d1)
seg_logits = coarse + refine(d1, coarse)
return {"seg": seg_logits, "prior": prior_logits, "boundary": boundary_head(d1)}
'@
    Add-Code $slide $code 35 122 500 345 8.8
    Add-Table $slide @(
        @("阶段", "解释"),
        @("prior_branch", "先算胸片先验。"),
        @("s1-s4", "四层 encoder 特征，并逐层注入先验。"),
        @("transformer", "瓶颈层建模全局上下文。"),
        @("decoder", "上采样恢复空间分辨率，并使用跳跃连接。"),
        @("coarse + refine", "粗分割加边界细化。"),
        @("输出字典", "seg 用于最终 mask，prior/boundary 用于辅助监督。")
    ) 560 122 360 300 9
    Add-TalkNote $slide "模型代码跳转：forward 内部先去 prior_branch，再走 encoder/adapter，再走 transformer/decoder，最后返回字典。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "37 LowDice 代码逐行 1"
    Add-Title $slide "LowDicePrior：为什么要额外加坐标和边界？" "低分样本常见位置偏移、小病灶和边界模糊。"
    $code = @'
gray = x[:, :1]
low  = avg_pool(gray, kernel=9)
high = abs(gray - low)
edge = sqrt(sobel_x(gray)^2 + sobel_y(gray)^2)
lap  = abs(laplace(gray))
xx, yy = normalized_coord(gray.shape)
prior = concat([gray, low, high, edge, lap, xx, yy])
'@
    Add-Code $slide $code 45 140 450 250 9.5
    Add-Table $slide @(
        @("行", "解释"),
        @("gray", "保留原始胸片灰度。"),
        @("low", "看整体肺野亮度趋势。"),
        @("high", "看局部纹理异常。"),
        @("edge", "看 Sobel 边缘。"),
        @("lap", "看 Laplace 边界突变。"),
        @("xx / yy", "告诉模型像素在图中的位置。"),
        @("concat", "把所有先验合成新的输入特征。")
    ) 535 130 370 285 9
    Add-TalkNote $slide "v18 先验偏稳定，LowDice 先验更强调困难样本的细节和位置。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "38 LowDice 代码逐行 2"
    Add-Title $slide "LowDiceRefineNet forward：专家模型如何输出 mask？" "这段说明专家模型不是单纯 U-Net。"
    $code = @'
prior = self.prior(x)
x0 = self.stem(cat([x, prior], dim=1))
feat = self.encoder(x0)
feat = self.aspp(feat)
d1 = self.decoder(feat)
d1 = self.local_refine(cat([d1, prior], dim=1))
seg = self.seg_head(d1)
return {"seg": seg}
'@
    Add-Code $slide $code 50 135 460 250 10
    Add-Table $slide @(
        @("行", "解释"),
        @("self.prior", "生成低分样本先验特征。"),
        @("cat([x, prior])", "把原图和先验一起送入网络。"),
        @("stem", "最初的卷积特征提取。"),
        @("encoder", "提取深层语义。"),
        @("aspp", "多尺度上下文，兼顾小病灶和大片病灶。"),
        @("decoder", "恢复空间分辨率。"),
        @("local_refine", "回到高分辨率位置细化边界。"),
        @("seg_head", "输出最终分割 logits。")
    ) 535 120 385 320 8.6

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "39 Gate 代码逐行"
    Add-Title $slide "门控代码：错误类型如何变成专家选择？" "最终策略靠这段逻辑把低分样本分流。"
    $code = @'
def gate_choice(failure_type, available):
    routing = {
        "over_segmented_fp": ("actual_fp", "actual_all", "precision", "broad"),
        "under_segmented_fn": ("actual_fn", "boundary", "recall", "broad"),
        "mostly_missed_fn": ("recall", "actual_fn", "boundary", "broad"),
        "boundary_or_shift_error": ("basev18", "actual_all", "broad"),
    }.get(str(failure_type), ("broad",))
    for preferred in routing:
        if preferred in available:
            return preferred
    return "basev18"
'@
    Add-Code $slide $code 35 122 515 340 8.6
    Add-Table $slide @(
        @("行", "解释"),
        @("failure_type", "当前样本属于哪类错误。"),
        @("available", "当前已经有预测结果的专家集合。"),
        @("routing 字典", "为每类错误设置候选专家顺序。"),
        @("get(... broad)", "未知错误时默认用 broad 专家。"),
        @("for preferred", "按优先级逐个尝试专家。"),
        @("if in available", "该专家存在就选它。"),
        @("return basev18", "所有专家不可用时回退到 v18。")
    ) 575 122 345 320 8.4
    Add-TalkNote $slide "这段代码就是论文里的「错误类型感知多专家门控」。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "40 后处理逐行"
    Add-Title $slide "后处理：概率图如何变成网页里的红色区域？" "后处理决定最终展示是否干净。"
    $code = @'
prob = sigmoid(logits)
pred = prob > threshold
pred = remove_small_components(pred, min_area)
area_ratio = pred.sum() / pred.size
overlay = build_overlay(image, pred, alpha=0.35)
save_png(original, mask, overlay)
'@
    Add-Code $slide $code 55 150 390 185 10.5
    Add-Table $slide @(
        @("行", "解释"),
        @("sigmoid", "logits 转成 0-1 概率。"),
        @("prob > threshold", "概率大于阈值的像素设为感染。"),
        @("remove_small_components", "去掉太小的孤立噪声块。"),
        @("area_ratio", "计算感染面积占比。"),
        @("build_overlay", "把 mask 用红色半透明叠加到原图。"),
        @("save_png", "保存前端需要展示的图片。")
    ) 505 130 385 260 9
    Add-TalkNote $slide "网页端显示的 Original、Mask、Overlay，本质上就是这几步产生的。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "41 训练流程逐行"
    Add-Title $slide "训练脚本：权重是怎么训练出来的？" "推理之前必须先得到 checkpoint。"
    $code = @'
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        outputs = model(x)
        loss = compute_loss(outputs, y, loss_mode)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_dice = evaluate(model, val_loader)
    if val_dice > best:
        save_checkpoint(model)
'@
    Add-Code $slide $code 45 132 445 300 9.2
    Add-Table $slide @(
        @("行", "解释"),
        @("for epoch", "按轮次训练。"),
        @("model.train", "进入训练模式。"),
        @("for x,y", "读取一批图像和掩码。"),
        @("outputs=model(x)", "前向传播。"),
        @("compute_loss", "计算 Dice/Focal/Boundary 等损失。"),
        @("backward/step", "反向传播并更新参数。"),
        @("evaluate", "验证集评估。"),
        @("save_checkpoint", "保存验证 Dice 最好的权重。")
    ) 520 120 390 320 8.7

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "42 从代码到结果"
    Add-Title $slide "代码如何对应最终结果？" "把工程实现和论文结果连起来。"
    Add-Table $slide @(
        @("代码模块", "产生的论文结果"),
        @("transunet2d_v18.py", "v18 baseline，全测试集 Dice 0.7740。"),
        @("lowdice_refinenet2d.py", "LowDice broad / precision / recall / boundary 专家。"),
        @("build_qata_v18_error_expert_manifests.py", "把低分样本拆成误检、漏检、边界偏移等类别。"),
        @("eval_lowdice_expert_gate.py", "得到 v18-aware gate，全测试集 Dice 0.7959。"),
        @("app_v12_web.py + v12_predictor.py", "形成网页端 Original / Mask / Overlay / metrics 展示。")
    ) 70 145 820 245 11
    Add-TalkNote $slide "学习时要把代码和实验表格对应起来，否则只看代码很容易不知道为什么写这一段。"

    $slide = Add-BlankSlide $presentation
    Add-TopBar $slide "43 学习版总结"
    Add-Title $slide "整套代码的主线" "最后用一句话串起来。"
    Add-Card $slide 65 145 250 210 "训练阶段" "train_unet2d_qata.py 读取数据、构建模型、计算损失、保存 checkpoint。" (RGB 219 234 254)
    Add-Card $slide 355 145 250 210 "离线评估阶段" "eval_lowdice_expert_gate.py 比较 v18 和专家输出，得到最终门控 Dice。" (RGB 254 243 199)
    Add-Card $slide 645 145 250 210 "网页展示阶段" "app_v12_web.py 调用 segmenter，把概率图转成 mask 和 overlay 返回页面。" (RGB 220 252 231)
    Add-TalkNote $slide "最终记忆链路：数据 -> 训练 -> checkpoint -> 推理 -> 后处理 -> 可视化 -> 指标。"
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
try {
    $defense = Prepare-Presentation $ppt $DefenseOut
    Add-Defense-CodeSlides $defense
    $defense.SaveAs($DefenseOut)
    $defense.Close()

    $study = Prepare-Presentation $ppt $StudyOut
    Add-Study-CodeSlides $study
    $study.SaveAs($StudyOut)
    $study.Close()
}
finally {
    $ppt.Quit()
}

Write-Output "defense=$DefenseOut"
Write-Output "study=$StudyOut"


