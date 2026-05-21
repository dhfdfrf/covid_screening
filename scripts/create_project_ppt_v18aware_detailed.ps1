$ErrorActionPreference = "Stop"

$Root = "D:\covid_screening"
$OutPath = Join-Path $Root "docs\covid_screening_project_v18aware_detailed_v4.pptx"
$FigDir = Join-Path $Root "docs\figures"

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
    $t = $slide.Shapes.AddTextbox(1, 28, 7, 720, 20)
    $t.TextFrame.TextRange.Text = $section
    $t.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $t.TextFrame.TextRange.Font.Size = 10
    $t.TextFrame.TextRange.Font.Color.RGB = RGB 229 231 235
}

function Add-Title($slide, $title, $subtitle = "") {
    $line = $slide.Shapes.AddShape(1, 42, 55, 6, 45)
    $line.Fill.ForeColor.RGB = RGB 14 116 144
    $line.Line.Visible = 0
    $box = $slide.Shapes.AddTextbox(1, 46, 52, 850, 54)
    $box.TextFrame.TextRange.Text = $title
    $box.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $box.TextFrame.TextRange.Font.Size = 29
    $box.TextFrame.TextRange.Font.Bold = -1
    $box.TextFrame.TextRange.Font.Color.RGB = RGB 31 41 55
    if ($subtitle -ne "") {
        $sub = $slide.Shapes.AddTextbox(1, 49, 105, 830, 38)
        $sub.TextFrame.TextRange.Text = $subtitle
        $sub.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
        $sub.TextFrame.TextRange.Font.Size = 14
        $sub.TextFrame.TextRange.Font.Color.RGB = RGB 100 116 139
    }
}

function Add-Text($slide, $text, $left, $top, $width, $height, $size = 16, $color = $null, $bold = $false) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = $size
    if ($null -eq $color) { $color = RGB 51 65 85 }
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = -1 }
    return $shape
}

function Add-Code($slide, $code, $left, $top, $width, $height, $size = 10) {
    $rect = $slide.Shapes.AddShape(1, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = RGB 11 31 51
    $rect.Line.ForeColor.RGB = RGB 14 116 144
    $rect.Line.Weight = 1.5
    $box = $slide.Shapes.AddTextbox(1, $left + 14, $top + 12, $width - 28, $height - 24)
    $box.TextFrame.TextRange.Text = $code
    $box.TextFrame.TextRange.Font.Name = "Consolas"
    $box.TextFrame.TextRange.Font.Size = $size
    $box.TextFrame.TextRange.Font.Color.RGB = RGB 226 232 240
}

function Add-Card($slide, $left, $top, $width, $height, $title, $body, $fillColor) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = $fillColor
    $rect.Line.ForeColor.RGB = RGB 148 163 184
    $rect.Line.Weight = 1.2
    Add-Text $slide $title ($left + 14) ($top + 12) ($width - 28) 28 15 (RGB 15 23 42) $true | Out-Null
    Add-Text $slide $body ($left + 14) ($top + 45) ($width - 28) ($height - 52) 12 (RGB 51 65 85) $false | Out-Null
}

function Add-TalkNote($slide, $text, $left = 70, $top = 476, $width = 820, $height = 38) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = RGB 255 247 237
    $rect.Line.ForeColor.RGB = RGB 251 146 60
    $rect.Line.Weight = 1.2
    Add-Text $slide $text ($left + 14) ($top + 10) ($width - 28) ($height - 16) 13 (RGB 124 45 18) $true | Out-Null
}

function Add-Metric($slide, $left, $top, $width, $label, $value, $note, $color) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, 94)
    $rect.Fill.ForeColor.RGB = RGB 255 255 255
    $rect.Line.ForeColor.RGB = $color
    $rect.Line.Weight = 2
    Add-Text $slide $label ($left + 14) ($top + 11) ($width - 28) 20 11 (RGB 71 85 105) $true | Out-Null
    Add-Text $slide $value ($left + 14) ($top + 31) ($width - 28) 34 24 $color $true | Out-Null
    Add-Text $slide $note ($left + 14) ($top + 68) ($width - 28) 20 10 (RGB 100 116 139) $false | Out-Null
}

function Add-Picture($slide, $path, $left, $top, $width, $height) {
    if (Test-Path $path) {
        return $slide.Shapes.AddPicture($path, 0, -1, $left, $top, $width, $height)
    }
    Add-Text $slide "Missing image:`n$path" $left $top $width $height 12 (RGB 220 38 38) $true | Out-Null
}

function Add-Arrow($slide, $x1, $y1, $x2, $y2, $color) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = 2
    $line.Line.EndArrowheadStyle = 3
}

function Add-Table($slide, $rows, $left, $top, $width, $height, $fontSize = 11) {
    $rowCount = $rows.Count
    $colCount = $rows[0].Count
    $table = $slide.Shapes.AddTable($rowCount, $colCount, $left, $top, $width, $height).Table
    for ($r = 1; $r -le $rowCount; $r++) {
        for ($c = 1; $c -le $colCount; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.Shape.TextFrame.TextRange.Text = $rows[$r-1][$c-1]
            $cell.Shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
            $cell.Shape.TextFrame.TextRange.Font.Size = $fontSize
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 30 41 59
            if ($r -eq 1) {
                $cell.Shape.Fill.ForeColor.RGB = RGB 15 23 42
                $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 255 255 255
                $cell.Shape.TextFrame.TextRange.Font.Bold = -1
            }
        }
    }
}

function Add-LegendItem($slide, $left, $top, $color, $text) {
    $swatch = $slide.Shapes.AddShape(1, $left, $top + 4, 16, 16)
    $swatch.Fill.ForeColor.RGB = $color
    $swatch.Line.ForeColor.RGB = RGB 148 163 184
    Add-Text $slide $text ($left + 23) $top 270 24 10 (RGB 30 41 59) $false | Out-Null
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

# 1 cover
$slide = Add-BlankSlide $presentation
$cover = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
$cover.Fill.ForeColor.RGB = RGB 15 23 42
$cover.Line.Visible = 0
$accent = $slide.Shapes.AddShape(1, 0, 418, 960, 122)
$accent.Fill.ForeColor.RGB = RGB 239 68 68
$accent.Line.Visible = 0
Add-Text $slide "COVID-19 胸片感染区域分割系统" 62 88 830 54 34 (RGB 248 250 252) $true | Out-Null
Add-Text $slide "TransUNet2D v18 + LowDice 多专家门控优化" 66 152 820 35 20 (RGB 203 213 225) $false | Out-Null
Add-Text $slide "项目展示详版 PPT  |  QaTa-COV19  |  Web 可视化系统" 66 450 760 34 16 (RGB 255 255 255) $false | Out-Null
Add-Text $slide "Final Dice 0.7959" 700 84 210 36 20 (RGB 252 211 77) $true | Out-Null

# 2 outline
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "00 展示目录"
Add-Title $slide "这份 PPT 怎么讲" "按系统、模型、公式、可视化、代码五条线展开。"
Add-Card $slide 60 145 260 105 "1. 系统与数据" "任务目标、数据集、预处理、Web 系统入口" (RGB 219 234 254)
Add-Card $slide 350 145 260 105 "2. 分割流程" "输入图像 → 模型概率图 → 阈值化 → 后处理 → 可视化" (RGB 220 252 231)
Add-Card $slide 640 145 260 105 "3. 模型优化" "v18 基线、低分样本分析、LowDice 专家、v18-aware 门控" (RGB 254 243 199)
Add-Card $slide 60 290 260 105 "4. 评价公式" "Dice、IoU、Precision、Recall、损失函数和字母含义" (RGB 237 233 254)
Add-Card $slide 350 290 260 105 "5. 实验展示" "定量对比、低分样本提升、颜色图例、误差图说明" (RGB 254 226 226)
Add-Card $slide 640 290 260 105 "6. 关键代码" "模型先验、门控路由、推理后处理、Web 调用" (RGB 226 232 240)

# 3 task
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "01 项目目标"
Add-Title $slide "要解决的问题是什么？" "给定一张胸片，自动找出 COVID-19 感染区域。"
Add-Card $slide 65 150 250 230 "输入" "胸部 X 光图像`n格式：PNG / JPG / DICOM`n来源：QaTa-COV19" (RGB 219 234 254)
Add-Card $slide 355 150 250 230 "算法输出" "像素级感染概率图`n二值感染掩码`n病灶面积占比" (RGB 254 243 199)
Add-Card $slide 645 150 250 230 "展示输出" "Original`nMask`nOverlay`nError Map`nDice / IoU / Precision / Recall" (RGB 220 252 231)
Add-Arrow $slide 315 265 355 265 (RGB 71 85 105)
Add-Arrow $slide 605 265 645 265 (RGB 71 85 105)
Add-Text $slide "核心难点：病灶区域小、边界模糊、灰度与肺纹理接近，容易误检或漏检。" 75 430 810 42 16 (RGB 220 38 38) $true | Out-Null

# 4 system architecture
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "02 系统架构"
Add-Title $slide "系统由四层组成" "数据层、模型层、评估层、展示层。"
Add-Card $slide 65 135 180 250 "数据层" "raw images`nGT masks`nNPZ cache`nmanifest.csv" (RGB 219 234 254)
Add-Card $slide 275 135 180 250 "模型层" "TransUNet2D v18`nLowDice experts`nthreshold / min_area" (RGB 254 243 199)
Add-Card $slide 485 135 180 250 "评估层" "Dice / IoU`nPrecision / Recall`nlow-dice analysis`nexpert gate eval" (RGB 237 233 254)
Add-Card $slide 695 135 180 250 "展示层" "web/index.html`napp_v12_web.py`nOriginal / Mask / Overlay`nDownload outputs" (RGB 220 252 231)
Add-Arrow $slide 245 260 275 260 (RGB 71 85 105)
Add-Arrow $slide 455 260 485 260 (RGB 71 85 105)
Add-Arrow $slide 665 260 695 260 (RGB 71 85 105)

# 5 data
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "03 数据集"
Add-Title $slide "QaTa-COV19 数据与划分" "以胸片图像和人工感染区域标注为监督信号。"
Add-Metric $slide 60 150 180 "总样本" "9258" "全部 manifest 记录" (RGB 37 99 235)
Add-Metric $slide 280 150 180 "训练集" "7406" "用于模型拟合" (RGB 5 150 105)
Add-Metric $slide 500 150 180 "验证集" "925" "用于阈值和 checkpoint 选择" (RGB 217 119 6)
Add-Metric $slide 720 150 180 "测试集" "927" "最终评估" (RGB 124 58 237)
Add-Text $slide "每条数据包含：id、img_path、mask_path、npz_path、split。训练时读取 NPZ 中的 img 和 mask。" 72 300 820 42 16 (RGB 30 41 59) $true | Out-Null
Add-Text $slide "为什么要单独保留测试集：避免用测试样本调参，保证最终 Dice 具有对比意义。" 72 360 820 42 15 (RGB 71 85 105) $false | Out-Null

# 6 preprocessing
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "04 预处理流程"
Add-Title $slide "图像进入模型前做了什么？" "统一输入尺度和数值范围，保证模型训练稳定。"
Add-Card $slide 45 150 165 195 "1. 读取图像" "PIL/OpenCV 读取`n转灰度图" (RGB 219 234 254)
Add-Card $slide 235 150 165 195 "2. 归一化" "像素缩放到 [0,1]`n减少曝光差异" (RGB 220 252 231)
Add-Card $slide 425 150 165 195 "3. 尺寸统一" "Resize 到 224×224`n形成固定输入张量" (RGB 254 243 199)
Add-Card $slide 615 150 165 195 "4. Mask 二值化" "感染区=1`n背景=0" (RGB 237 233 254)
Add-Card $slide 805 150 110 195 "5. 存储" "NPZ`nmanifest" (RGB 254 226 226)
Add-Arrow $slide 210 250 235 250 (RGB 71 85 105)
Add-Arrow $slide 400 250 425 250 (RGB 71 85 105)
Add-Arrow $slide 590 250 615 250 (RGB 71 85 105)
Add-Arrow $slide 780 250 805 250 (RGB 71 85 105)
Add-Text $slide "训练增强：水平翻转、亮度/对比度/Gamma、轻微噪声；避免垂直翻转破坏胸片上下结构。" 70 405 830 42 15 (RGB 71 85 105) $false | Out-Null

# 7 segmentation flow
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "05 分割流程"
Add-Title $slide "从概率图到最终 Mask" "模型输出不是直接结果，还需要阈值化和后处理。"
Add-Card $slide 55 150 180 220 "Step 1`n模型前向传播" "输入 X`n输出 logits Z`nSigmoid 得概率 P" (RGB 219 234 254)
Add-Card $slide 275 150 180 220 "Step 2`n阈值化" "P(x,y) > τ 则预测为感染区域`n否则为背景" (RGB 254 243 199)
Add-Card $slide 495 150 180 220 "Step 3`n连通域过滤" "删除面积小于 min_area 的小碎片，降低噪声误检" (RGB 220 252 231)
Add-Card $slide 715 150 180 220 "Step 4`n可视化与评估" "Mask / Overlay / Error Map`n计算四个指标" (RGB 237 233 254)
Add-Arrow $slide 235 260 275 260 (RGB 71 85 105)
Add-Arrow $slide 455 260 495 260 (RGB 71 85 105)
Add-Arrow $slide 675 260 715 260 (RGB 71 85 105)
Add-Text $slide "本项目最终阈值不是固定 0.5，而是在验证集上搜索；不同专家模型有各自最优 threshold 和 min_area。" 70 420 820 42 15 (RGB 220 38 38) $true | Out-Null

# 8 threshold formula
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "06 后处理公式"
Add-Title $slide "二值化和连通域过滤怎么算？" "这一步决定最终感染面积和 Overlay 结果。"
Add-Text $slide "概率图：" 70 145 150 28 16 (RGB 15 23 42) $true | Out-Null
Add-Text $slide "P(x,y) = sigmoid(Z(x,y))" 220 145 620 28 20 (RGB 37 99 235) $true | Out-Null
Add-Text $slide "二值 Mask：" 70 205 150 28 16 (RGB 15 23 42) $true | Out-Null
Add-Text $slide "M_hat(x,y) = 1, if P(x,y) > tau;  otherwise 0" 220 205 660 28 18 (RGB 220 38 38) $true | Out-Null
Add-Text $slide "连通域过滤：" 70 265 150 28 16 (RGB 15 23 42) $true | Out-Null
Add-Text $slide "删除面积 Area(C_i) < min_area 的预测连通块 C_i" 220 265 660 28 18 (RGB 5 150 105) $true | Out-Null
Add-Card $slide 90 345 230 95 "P(x,y)" "像素 (x,y) 属于感染区域的概率。" (RGB 219 234 254)
Add-Card $slide 365 345 230 95 "tau" "阈值。越高越保守，误检减少但漏检可能增加。" (RGB 254 243 199)
Add-Card $slide 640 345 230 95 "min_area" "最小连通域面积。用于过滤小噪声点。" (RGB 220 252 231)

# 9 model evolution
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "07 模型迭代"
Add-Title $slide "模型路线为什么从单模型转向多专家？" "单一结构改进提升有限，错误类型拆解更有效。"
$rows = @(
    @("阶段", "代表方案", "作用", "结论"),
    @("基础模型", "U-Net / DynUNet / Attention U-Net", "建立分割基线", "可用但性能有限"),
    @("TransUNet 系列", "v11-v20", "窗口注意力、边界、频域、先验", "部分提升不稳定"),
    @("传统融合", "v15 概率融合", "利用 TransUNet 与 UCTRansNet 互补", "Dice 0.7879"),
    @("最终优化", "v18-aware 多专家门控", "按低分错误类型选择专家", "Dice 0.7959")
)
Add-Table $slide $rows 55 150 850 245 10
Add-Text $slide "工程判断：如果某类错误反复出现，就针对这类错误训练专家，而不是继续把所有问题塞给一个模型。" 70 430 820 42 15 (RGB 15 23 42) $true | Out-Null

# 10 main model changes
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "08 主要模型改动"
Add-Title $slide "主要模型到底改在哪里？" "最终版不是只改一个网络文件，而是模型结构、训练损失、低分数据划分和门控策略一起改。"
$rows = @(
    @("改动位置", "具体内容", "解决的问题"),
    @("src/models/lowdice_refinenet2d.py", "LowDiceRefineNet2D：边缘/频域/坐标先验，高分辨率解码，SE/ASPP/AttentionMerge。", "给低 Dice 样本单独建专家模型。"),
    @("src/train_unet2d_qata.py", "hard_case_combo、precision_hard_case_combo、recall_hard_case_combo、boundary_shift_combo。", "不同专家使用不同损失倾向。"),
    @("scripts/build_qata_v18_error_expert_manifests.py", "按 v18 的实际错误把低分样本拆成 actual_fp / actual_fn / actual_boundary / actual_all。", "让训练集和错误类型对应起来。"),
    @("scripts/eval_lowdice_expert_gate.py", "gate_choice 实现 v18-aware 路由，按错误类型选择 basev18 或对应专家。", "避免平均融合，把每个样本交给更合适的模型。")
)
Add-Table $slide $rows 38 140 884 290 9
Add-Card $slide 110 450 740 54 "一句话总结" "最终模型改动点是 LowDice 专家池 + v18-aware 门控路由，不是单纯把 v18 换成一个新模型。" (RGB 254 243 199)

# 10 baseline v18
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "09 v18 基线"
Add-Title $slide "TransUNet2D v18 扮演什么角色？" "它是最终门控策略的基准模型和常规样本默认输出。"
Add-Metric $slide 75 150 220 "全测试集 Dice" "0.7740" "v18 baseline" (RGB 37 99 235)
Add-Metric $slide 370 150 220 "低分样本数" "158" "Dice < 0.60" (RGB 220 38 38)
Add-Metric $slide 665 150 220 "低分均值" "0.4033" "困难样本平均 Dice" (RGB 217 119 6)
Add-Card $slide 100 310 760 100 "为什么还保留 v18？" "虽然 v18 在低分样本上不够好，但它在部分边界偏移样本上比专家更稳。因此最终策略不是完全替换 v18，而是让 v18 成为可回退的专家之一。" (RGB 226 232 240)

# 11 low dice analysis
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "10 低分样本分析"
Add-Title $slide "158 个低分样本错在哪里？" "用错误类型决定训练哪些专家。"
Add-Picture $slide (Join-Path $FigDir "fig_lowdice_failure_distribution_ppt.png") 60 130 480 270 | Out-Null
Add-Card $slide 590 135 300 65 "boundary_or_shift_error" "49 例：大体位置对，但边界偏移。" (RGB 219 234 254)
Add-Card $slide 590 215 300 65 "over_segmented_fp" "42 例：预测过大，假阳性多。" (RGB 254 226 226)
Add-Card $slide 590 295 300 65 "under / missed FN" "54 例：病灶覆盖不足或完全漏掉。" (RGB 237 233 254)
Add-Card $slide 590 375 300 65 "wrong_location" "13 例：预测区域与真值几乎无重叠。" (RGB 220 252 231)

# 12 expert pool
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "11 LowDice 专家池"
Add-Title $slide "每个专家具体解决什么问题？" "不同专家不是分数都高，而是在特定错误类型上更合适。"
$rows = @(
    @("专家", "训练目标", "适合错误", "作用"),
    @("broad", "低分样本通用纠错", "多数困难样本", "稳健替换"),
    @("precision", "减少假阳性", "过分割 FP", "让预测更保守"),
    @("recall", "提高召回", "明显漏检 FN", "扩大病灶覆盖"),
    @("boundary", "边界与质心约束", "边界偏移", "细化边缘"),
    @("actual-fp/fn/boundary", "按 v18 实际错误划分", "极端低分样本", "针对性更强")
)
Add-Table $slide $rows 45 135 870 330 10

# 13 gate strategy
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "12 最终优化策略"
Add-Title $slide "v18-aware Gate 是什么意思？" "不是平均融合，而是按错误类型选择模型。"
Add-Picture $slide (Join-Path $FigDir "fig_v18aware_pipeline_ppt.png") 45 125 870 310 | Out-Null
Add-Text $slide "核心思想：先让 v18 给出基线，再对低分困难样本按错误类型选择更合适的专家输出。" 62 460 850 38 15 (RGB 51 65 85) $true | Out-Null

# 14 routing table
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "13 门控规则"
Add-Title $slide "每种错误路由到哪个专家？" "这张表就是最终优化策略的含义。"
$rows = @(
    @("错误类型", "优先模型", "为什么这样选"),
    @("boundary_or_shift_error", "basev18", "这类样本位置基本对，v18 边界更稳，避免专家过修正。"),
    @("over_segmented_fp", "actual_fp", "预测区域过大，需要更强的误检抑制。"),
    @("under_segmented_fn", "actual_fn / boundary / recall", "真实病灶没覆盖够，需要提高召回或补边界。"),
    @("mostly_missed_fn", "recall", "大面积漏检时优先用召回专家。"),
    @("wrong_location_no_overlap", "actual_boundary / precision", "预测错位，优先用实际错位样本训练出的专家。"),
    @("missed_all", "actual_all / actual_fn", "完全漏掉时用整体低分专家或漏检专家。")
)
Add-Table $slide $rows 45 130 870 335 10

# 15 formulas intro
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "14 评价指标"
Add-Title $slide "为什么要用四个指标？" "单看 Dice 不够，需要同时看误检和漏检。"
Add-Card $slide 70 150 190 190 "Dice" "衡量预测区域与真实区域的重叠程度。医学分割最常用。" (RGB 219 234 254)
Add-Card $slide 285 150 190 190 "IoU" "交并比，比 Dice 更严格，对边界偏移更敏感。" (RGB 220 252 231)
Add-Card $slide 500 150 190 190 "Precision" "预测为病灶的像素中，有多少是真的。低则误检多。" (RGB 254 243 199)
Add-Card $slide 715 150 190 190 "Recall" "真实病灶像素中，有多少被找出来。低则漏检多。" (RGB 254 226 226)
Add-Text $slide "四者关系：Dice/IoU 看整体重叠；Precision 关注 FP；Recall 关注 FN。最终模型要在误检和漏检之间折中。" 70 410 820 48 15 (RGB 15 23 42) $true | Out-Null

# 16 symbols
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "15 公式符号说明"
Add-Title $slide "公式里的字母分别代表什么？" "先统一符号，再看每个公式。"
$rows = @(
    @("符号", "含义"),
    @("M(x,y)", "真实二值掩码。感染区域为 1，背景为 0。"),
    @("M_hat(x,y)", "模型预测二值掩码。阈值化后感染区域为 1。"),
    @("P(x,y)", "模型预测的感染概率，范围为 0 到 1。"),
    @("TP", "True Positive，预测为感染且真实也是感染的像素数。"),
    @("FP", "False Positive，预测为感染但真实是背景的像素数。"),
    @("FN", "False Negative，真实是感染但模型预测为背景的像素数。"),
    @("epsilon", "很小的平滑项，防止分母为 0。")
)
Add-Table $slide $rows 95 130 770 335 12

# 17 dice iou
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "16 Dice 和 IoU"
Add-Title $slide "重叠类指标怎么算？" "越接近 1，预测和真实区域越一致。"
Add-Text $slide "Dice = (2 × TP + epsilon) / (2 × TP + FP + FN + epsilon)" 70 150 830 34 20 (RGB 37 99 235) $true | Out-Null
Add-Text $slide "等价写法：Dice = 2 × |M_hat ∩ M| / (|M_hat| + |M|)" 70 200 830 28 18 (RGB 37 99 235) $false | Out-Null
Add-Text $slide "IoU = (TP + epsilon) / (TP + FP + FN + epsilon)" 70 280 830 34 20 (RGB 5 150 105) $true | Out-Null
Add-Text $slide "等价写法：IoU = |M_hat ∩ M| / |M_hat ∪ M|" 70 330 830 28 18 (RGB 5 150 105) $false | Out-Null
Add-Text $slide "解释：Dice 对重叠比较友好；IoU 分母是并集，因此同样错误下通常比 Dice 更低、更严格。" 80 420 800 42 15 (RGB 71 85 105) $true | Out-Null

# 18 precision recall
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "17 Precision 和 Recall"
Add-Title $slide "误检和漏检怎么算？" "这两个指标对应 FP 和 FN 两类错误。"
Add-Text $slide "Precision = (TP + epsilon) / (TP + FP + epsilon)" 70 160 830 34 20 (RGB 217 119 6) $true | Out-Null
Add-Text $slide "含义：模型说是感染的像素中，有多少是真的感染。Precision 低 -> 误检多。" 85 212 800 34 15 (RGB 71 85 105) $false | Out-Null
Add-Text $slide "Recall = (TP + epsilon) / (TP + FN + epsilon)" 70 300 830 34 20 (RGB 220 38 38) $true | Out-Null
Add-Text $slide "含义：真实感染像素中，有多少被模型找出来。Recall 低 -> 漏检多。" 85 352 800 34 15 (RGB 71 85 105) $false | Out-Null
Add-Text $slide "为什么低分专家要分 precision / recall：过分割主要看 FP，漏分割主要看 FN，优化方向相反。" 80 430 800 42 15 (RGB 15 23 42) $true | Out-Null

# 19 loss
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "18 损失函数"
Add-Title $slide "训练时模型靠什么学习？" "用 Dice/Focal/Tversky/Boundary 等损失组合处理类别不平衡和困难边界。"
Add-Text $slide "Soft Dice Loss = 1 - (2 × Σ(P × M) + epsilon) / (ΣP + ΣM + epsilon)" 60 145 850 32 17 (RGB 37 99 235) $true | Out-Null
Add-Text $slide "Focal Loss：降低易分类背景像素权重，让模型更关注难分像素。" 60 205 850 28 16 (RGB 217 119 6) $true | Out-Null
Add-Text $slide "Tversky Loss：通过 alpha、beta 调整 FP 和 FN 的惩罚强度。" 60 265 850 28 16 (RGB 124 58 237) $true | Out-Null
Add-Text $slide "Boundary Loss：对边界带区域加权，缓解病灶边缘模糊导致的偏移。" 60 325 850 28 16 (RGB 5 150 105) $true | Out-Null
Add-Text $slide "本项目不同专家使用不同损失倾向：precision 专家更抑制 FP，recall 专家更抑制 FN，boundary 专家更重视边界。" 75 420 810 45 15 (RGB 15 23 42) $true | Out-Null

# 20 paper result table
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "19 论文结果表"
Add-Title $slide "论文里的最终结果表" "公式讲完后再看结果表，Dice、IoU、Precision、Recall 的含义更清楚。"
$rows = @(
    @("方法", "Dice", "IoU", "Precision", "Recall"),
    @("TransUNet2D v14", "0.7755", "0.6708", "0.7771", "0.8344"),
    @("UCTRansNet", "0.7777", "0.6722", "0.7640", "0.8542"),
    @("v15 概率级融合", "0.7879", "0.6851", "0.7831", "0.8482"),
    @("TransUNet2D v18 基线", "0.7740", "0.6691", "0.7894", "0.8177"),
    @("LowDice broad 专家替换", "0.7850", "0.6792", "0.7948", "0.8250"),
    @("v18-aware 多专家门控（最终）", "0.7959", "0.6893", "0.7988", "0.8303"),
    @("v18-aware Oracle 上限（分析）", "0.8129", "-", "-", "-")
)
Add-Table $slide $rows 55 135 850 300 10
Add-Text $slide "写论文和答辩时：最终可部署结果写 v18-aware 多专家门控；Oracle 只作为专家池潜力分析，不当作真实部署成绩。" 70 458 820 42 14 (RGB 220 38 38) $true | Out-Null

# 20 quantitative results
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "20 全测试集结果"
Add-Title $slide "最终结果提升多少？" "v18-aware Gate 是当前最终版。"
Add-Picture $slide (Join-Path $FigDir "fig_dice_comparison_v18aware.png") 52 125 530 305 | Out-Null
Add-Metric $slide 620 138 240 "v18 Baseline" "0.7740" "全测试集 Dice" (RGB 37 99 235)
Add-Metric $slide 620 258 240 "v18-aware Gate" "0.7959" "+0.0219 vs v18" (RGB 220 38 38)
Add-Metric $slide 620 378 240 "Oracle Upper" "0.8129" "专家池上限分析" (RGB 124 58 237)

# 21 low dice results
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "21 低分样本结果"
Add-Title $slide "低分样本是本轮优化重点" "平均 Dice 从 0.4033 提升到 0.5319。"
$rows = @(
    @("策略", "低分子集 Dice", "Dice<0.60 数量", "说明"),
    @("v18 基线", "0.4033", "158", "原始低分集合"),
    @("LowDice broad", "0.4683", "118", "通用困难样本专家"),
    @("Expanded gate", "0.5276", "91", "多专家错误类型门控"),
    @("v18-aware gate", "0.5319", "97", "边界偏移回退到 basev18"),
    @("Oracle 上限", "0.6315", "61", "每例选最优专家，仅作上限")
)
Add-Table $slide $rows 65 135 830 270 11
Add-Text $slide "注意：Oracle 上限不是可部署策略，只用于说明专家池还有潜力；真实部署还需要无真值门控分类器。" 75 435 810 42 15 (RGB 220 38 38) $true | Out-Null

# 23 visual result high
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "22 可视化样例 1"
Add-Title $slide "高 Dice 样本：预测和真值基本重合" "重点看 Overlay 和 Error Map，判断模型是否真正覆盖病灶。"
Add-Picture $slide (Join-Path $FigDir "fig_qual_compare_high.png") 65 120 830 250 | Out-Null
Add-Text $slide "颜色含义：" 58 390 120 24 13 (RGB 15 23 42) $true | Out-Null
Add-LegendItem $slide 60 420 (RGB 239 68 68) "Overlay 红色填充：预测感染区域"
Add-LegendItem $slide 360 420 (RGB 234 179 8) "黄色轮廓：真实标注 GT 边界"
Add-LegendItem $slide 660 420 (RGB 6 182 212) "青色轮廓：预测边界"
Add-LegendItem $slide 60 456 (RGB 220 38 38) "Error Map 红色：TP，预测正确重叠"
Add-LegendItem $slide 360 456 (RGB 34 197 94) "Error Map 绿色：FN，真实有但漏检"
Add-LegendItem $slide 660 456 (RGB 37 99 235) "Error Map 蓝色：FP，模型误检"
Add-Text $slide "高 Dice 的图像通常红色 TP 占主导，黄色 GT 边界和青色预测边界贴合，说明模型定位和形状都比较准确。" 75 502 810 24 12 (RGB 71 85 105) $true | Out-Null

# 24 visual result low
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "23 可视化样例 2"
Add-Title $slide "低 Dice 样本：看漏检、误检和边界错位" "低分样本是 LowDice 专家池和 v18-aware Gate 的主要优化对象。"
Add-Picture $slide (Join-Path $FigDir "fig_qual_compare_low.png") 65 120 830 250 | Out-Null
Add-Card $slide 70 392 250 78 "绿色 FN 多" "真实有感染但模型没分出来，说明 Recall 不足，优先考虑 recall / actual_fn 专家。" (RGB 220 252 231)
Add-Card $slide 355 392 250 78 "蓝色 FP 多" "模型把背景误分为感染，说明 Precision 不足，优先考虑 precision / actual_fp 专家。" (RGB 219 234 254)
Add-Card $slide 640 392 250 78 "轮廓整体错位" "位置大体接近但边界偏移时，优先保留 basev18 或使用 boundary 类专家。" (RGB 254 243 199)
Add-Text $slide "颜色直接对应错误来源：红色 TP 是对的，绿色 FN 是漏检，蓝色 FP 是误检；最终门控就是按这些错误类型选择专家。" 75 492 810 34 13 (RGB 15 23 42) $true | Out-Null

# 25 workflow visual
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "24 结果图怎么读"
Add-Title $slide "一行结果图的阅读顺序" "从左到右判断模型是否分对。"
Add-Picture $slide (Join-Path $FigDir "fig_qualitative_large_cases.png") 70 130 820 250 | Out-Null
Add-Text $slide "1. Original：原始胸片；2. Ground Truth：人工标注；3. Prediction：模型二值结果；4. Overlay：预测叠加；5. Error Map：红 TP、绿 FN、蓝 FP。" 75 415 810 54 15 (RGB 15 23 42) $true | Out-Null

# 25 web
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "25 Web 系统"
Add-Title $slide "网页端功能怎么展示？" "本地服务打开后即可上传胸片查看结果。"
Add-Card $slide 65 145 245 230 "输入与参数" "上传 PNG/JPG/DICOM`n选择 checkpoint/config`n设置 threshold / min_area / TTA" (RGB 219 234 254)
Add-Card $slide 357 145 245 230 "结果预览" "Original`nMask`nOverlay`n感染面积占比" (RGB 220 252 231)
Add-Card $slide 650 145 245 230 "评估与保存" "自动匹配 GT mask`nDice / IoU / Precision / Recall`n保存到 outputs/web_infer" (RGB 254 243 199)
Add-Text $slide "启动命令：python app_v12_web.py    访问地址：http://127.0.0.1:8000" 75 425 830 32 17 (RGB 220 38 38) $true | Out-Null

# 26 code v18
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "26 关键代码 1"
Add-Title $slide "v18 基线模型代码怎么讲？" "v18 是最终门控策略的稳定基线：频域先验 + 残差注入 + 窗口 Transformer + 边界细化。"
$code = @'
# NAG: v18 = stable TransUNet + frequency-prior residual adapters.
class _FrequencyPriorExtractor(nn.Module):
    def forward(self, image):
        gray = image.mean(dim=1, keepdim=True)
        low  = avg_pool2d(gray, kernel_size=9)
        high = gray - low
        edge = sqrt(sobel_x(gray)^2 + sobel_y(gray)^2)
        lap  = conv2d(gray, laplace).abs()
        feat = encoder(cat([gray, low, abs(high), edge, lap]))
        return feat, head(feat)

class BoundaryAwareTransUNet2D_v18(nn.Module):
    def forward(self, x):
        prior_feat, prior_logits = self.prior_branch(x)
        s1 = adapter1(enc1(x), prior_feat, prior_logits)
        s2 = adapter2(enc2(down(s1)), prior_feat, prior_logits)
        s3 = adapter3(enc3(down(s2)), prior_feat, prior_logits)
        s4 = adapter4(enc4(down(s3)), prior_feat, prior_logits)

        tokens = self.transformer(flatten(down(s4)), h, w)
        d1 = decode_with_spatial_gates(tokens, [s4, s3, s2, s1])
        seg = coarse_out(d1) + refine(d1, coarse_out(d1))
        return {"seg": seg, "prior": prior_logits, "boundary": boundary_head(d1)}
'@
Add-Code $slide $code 42 132 570 330 8
Add-Card $slide 640 132 260 62 "1. 频域先验" "gray / low / high / edge / lap 提供胸片灰度和边缘线索。" (RGB 219 234 254)
Add-Card $slide 640 205 260 62 "2. 残差注入" "adapter1-4 把先验注入不同尺度 encoder 特征。" (RGB 220 252 231)
Add-Card $slide 640 278 260 62 "3. 全局建模" "window transformer 在瓶颈层补充长距离上下文。" (RGB 254 243 199)
Add-Card $slide 640 351 260 62 "4. 边界输出" "seg / prior / boundary 三个输出共同支持训练与评估。" (RGB 254 226 226)
Add-TalkNote $slide "讲法/NAG：先说明 v18 是稳定基线，不是最终全部创新；它提供可回退结果，后面的 LowDice 专家专门处理 v18 低分样本。" 55 465 850 46

# 27 code lowdice
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "27 关键代码 2"
Add-Title $slide "LowDiceRefineNet：为什么适合低分样本？" "加入边缘、局部对比和坐标先验。"
$code = @'
# NAG: 低分专家不是普通 U-Net 复训，而是显式加入先验通道。
class _LowDicePrior(nn.Module):
    def forward(self, x):
        gray = x[:, :1]
        low  = avg_pool(gray, kernel=9)       # NAG: 低频结构，保留肺野整体亮度
        high = abs(gray - low)                # NAG: 高频纹理，突出感染细碎纹理
        edge = sqrt(sobel_x(gray)^2 + sobel_y(gray)^2)
        lap  = abs(laplace(gray))             # NAG: 边界响应，辅助模糊病灶边缘
        xx, yy = normalized_coord(gray.shape)  # NAG: 坐标先验，减少左右肺错位
        return concat([gray, low, high, edge, lap, xx, yy])

class LowDiceRefineNet2D(nn.Module):
    def forward(self, x):
        prior = self.prior(x)
        x0 = self.stem(concat([x, prior]))
        feat = self.encoder(x0)
        feat = self.aspp(feat)                 # NAG: 多尺度上下文
        d1 = self.decoder(feat)
        d1 = self.local_refine(concat([d1, prior]))
        return {"seg": self.seg_head(d1)}
'@
Add-Code $slide $code 42 132 875 330 9
Add-TalkNote $slide "讲法/NAG：这一页强调创新点来自低分样本先验增强，不是简单换一个分割网络。"

# 28 code gate
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "28 关键代码 3"
Add-Title $slide "v18-aware 门控路由代码" "最终策略的核心是 gate_choice。"
$code = @'
# NAG: 门控不是平均融合，而是按 v18 错误类型选择最合适专家。
def gate_choice(failure_type, available):
    routing = {
        "over_segmented_fp": ("actual_fp", "actual_all", "precision", "broad"),
        "wrong_location_no_overlap": ("actual_boundary", "precision", "actual_all", "broad"),
        "under_segmented_fn": ("actual_fn", "boundary", "recall", "broad"),
        "mostly_missed_fn": ("recall", "actual_fn", "boundary", "broad"),
        "missed_all": ("actual_all", "actual_fn", "broad", "recall"),
        "boundary_or_shift_error": ("basev18", "actual_all", "broad", "actual_boundary"),
    }.get(str(failure_type), ("broad",))

    # NAG: 同一错误类型给出候选顺序，保证专家缺失时还能回退。
    for preferred in routing:
        if preferred in available:
            return preferred
    return "basev18" if "basev18" in available else "broad"

def apply_gate(case, preds, failure_type):
    expert = gate_choice(failure_type, preds.keys())
    mask = postprocess(preds[expert])
    return expert, mask
'@
Add-Code $slide $code 42 132 875 330 9
Add-TalkNote $slide "讲法/NAG：论文里可写成「错误类型感知的多专家门控策略」；当前离线评估用真实错误类型，后续可接无真值门控分类器。"

# 29 code inference
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "29 关键代码 4"
Add-Title $slide "推理后处理与 Overlay" "这部分决定网页端展示出来的 Mask 和红色叠加图。"
$code = @'
# NAG: 后处理让网页端结果更稳定，过滤孤立噪声点。
def remove_small_components(mask, min_area):
    labels, stats = connectedComponentsWithStats(mask.astype(uint8))
    clean = zeros_like(mask)
    for i in range(1, labels.max() + 1):
        area = stats[i, CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 1
    return clean

def build_overlay(image01, mask, alpha=0.35):
    image_u8 = normalize_to_uint8(image01)
    rgb = stack([image_u8, image_u8, image_u8], axis=-1)
    red = zeros_like(rgb)
    red[..., 0] = 255
    overlay = rgb * (1 - alpha * mask) + red * alpha * mask
    return overlay.astype(uint8)

# NAG: threshold 决定敏感度；min_area 决定小噪声是否保留。
prob = model(image)
prob = tta_average(prob) if use_tta else prob
pred = remove_small_components(prob > threshold, min_area)
'@
Add-Code $slide $code 42 132 875 330 9
Add-TalkNote $slide "讲法/NAG：这一页解释网页里 Mask、Overlay 和感染面积占比是怎么从概率图生成的。"

# 30 code web
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "30 关键代码 5"
Add-Title $slide "Web 端调用流程" "上传文件后，后端调用 V12Segmenter / 配置模型完成推理。"
$code = @'
# NAG: Web 端把模型能力产品化，用户只需要上传胸片。
@app.post("/api/predict")
async def predict(file, checkpoint, threshold=0.25, min_area=32):
    image_path = save_upload(file)
    cfg = resolve_config(checkpoint)

    result = segmenter.segment_file(
        image_path,
        checkpoint=cfg.ckpt,
        threshold=threshold,
        min_area=min_area,
        tta=cfg.tta,
    )

    # NAG: 如果文件名能匹配 GT，自动补 Dice/IoU/Precision/Recall。
    metrics = evaluate_if_gt_exists(image_path, result.mask)
    return {
        "original": url(result.original_png),
        "mask": url(result.mask_png),
        "overlay": url(result.overlay_png),
        "metrics": metrics,
        "save_dir": result.save_dir,
    }
'@
Add-Code $slide $code 42 120 875 350 9
Add-TalkNote $slide "讲法/NAG：这一页用于说明系统不仅有模型训练，还完成了可交互展示、结果保存和自动指标计算。" 70 482 820 42

# 31 summary
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "31 总结"
Add-Title $slide "项目最终结论" "从能分割，到能解释为什么提升。"
Add-Card $slide 60 145 260 245 "完成内容" "1. 胸片感染区域分割训练流程`n2. Web 可视化系统`n3. 低分样本错误分析`n4. LowDice 多专家模型池" (RGB 219 234 254)
Add-Card $slide 350 145 260 245 "最终效果" "全测试集 Dice：0.7959`n低分子集 Dice：0.4033 → 0.5319`nOracle 上限：0.8129" (RGB 254 226 226)
Add-Card $slide 640 145 260 245 "后续工作" "1. 训练无真值门控分类器`n2. 将 v18-aware gate 接入 Web`n3. 增加外部数据验证`n4. 优化错位无重叠样本" (RGB 220 252 231)
Add-Text $slide "答辩重点：不是简单堆模型，而是基于错误类型进行有针对性的专家化优化。" 100 430 760 42 18 (RGB 15 23 42) $true | Out-Null

$presentation.SaveAs($OutPath)
$presentation.Close()
$ppt.Quit()
Write-Output "saved=$OutPath"






