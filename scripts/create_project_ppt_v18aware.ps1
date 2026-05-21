$ErrorActionPreference = "Stop"

$Root = "D:\covid_screening"
$OutPath = Join-Path $Root "docs\covid_screening_project_v18aware_presentation.pptx"
$FigDir = Join-Path $Root "docs\figures"

function RGB($r, $g, $b) {
    return [int]($r + ($g * 256) + ($b * 65536))
}

function Add-BlankSlide($presentation) {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $bg = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
    $bg.Fill.ForeColor.RGB = RGB 248 245 238
    $bg.Line.Visible = 0
    $bg.ZOrder(1) | Out-Null
    return $slide
}

function Add-TopBar($slide, $section) {
    $bar = $slide.Shapes.AddShape(1, 0, 0, 960, 34)
    $bar.Fill.ForeColor.RGB = RGB 17 24 39
    $bar.Line.Visible = 0
    $t = $slide.Shapes.AddTextbox(1, 30, 7, 600, 20)
    $t.TextFrame.TextRange.Text = $section
    $t.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $t.TextFrame.TextRange.Font.Size = 10
    $t.TextFrame.TextRange.Font.Color.RGB = RGB 229 231 235
}

function Add-Title($slide, $title, $subtitle = "") {
    $box = $slide.Shapes.AddTextbox(1, 46, 54, 850, 54)
    $box.TextFrame.TextRange.Text = $title
    $box.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $box.TextFrame.TextRange.Font.Size = 30
    $box.TextFrame.TextRange.Font.Bold = -1
    $box.TextFrame.TextRange.Font.Color.RGB = RGB 31 41 55
    if ($subtitle -ne "") {
        $sub = $slide.Shapes.AddTextbox(1, 49, 108, 830, 34)
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

function Add-Card($slide, $left, $top, $width, $height, $title, $body, $fillColor) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $rect.Fill.ForeColor.RGB = $fillColor
    $rect.Line.ForeColor.RGB = RGB 203 213 225
    $rect.Line.Weight = 1
    Add-Text $slide $title ($left + 16) ($top + 14) ($width - 32) 28 16 (RGB 15 23 42) $true | Out-Null
    Add-Text $slide $body ($left + 16) ($top + 48) ($width - 32) ($height - 56) 12 (RGB 51 65 85) $false | Out-Null
}

function Add-Metric($slide, $left, $top, $width, $label, $value, $note, $color) {
    $rect = $slide.Shapes.AddShape(5, $left, $top, $width, 100)
    $rect.Fill.ForeColor.RGB = RGB 255 255 255
    $rect.Line.ForeColor.RGB = $color
    $rect.Line.Weight = 2
    Add-Text $slide $label ($left + 14) ($top + 12) ($width - 28) 22 12 (RGB 71 85 105) $true | Out-Null
    Add-Text $slide $value ($left + 14) ($top + 34) ($width - 28) 34 25 $color $true | Out-Null
    Add-Text $slide $note ($left + 14) ($top + 72) ($width - 28) 20 10 (RGB 100 116 139) $false | Out-Null
}

function Add-Picture($slide, $path, $left, $top, $width, $height) {
    if (Test-Path $path) {
        $pic = $slide.Shapes.AddPicture($path, 0, -1, $left, $top, $width, $height)
        return $pic
    }
    Add-Text $slide "Missing image:`n$path" $left $top $width $height 12 (RGB 220 38 38) $true | Out-Null
}

function Add-Arrow($slide, $x1, $y1, $x2, $y2, $color) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = 2
    $line.Line.EndArrowheadStyle = 3
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

# Slide 1
$slide = Add-BlankSlide $presentation
$cover = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
$cover.Fill.ForeColor.RGB = RGB 15 23 42
$cover.Line.Visible = 0
$accent = $slide.Shapes.AddShape(1, 0, 418, 960, 122)
$accent.Fill.ForeColor.RGB = RGB 239 68 68
$accent.Line.Visible = 0
Add-Text $slide "COVID-19 胸片感染区域分割系统" 62 92 820 52 34 (RGB 248 250 252) $true | Out-Null
Add-Text $slide "TransUNet2D v18 + LowDice 多专家门控优化" 66 154 820 35 20 (RGB 203 213 225) $false | Out-Null
Add-Text $slide "项目展示 PPT  |  QaTa-COV19  |  Web 可视化系统" 66 450 760 34 16 (RGB 255 255 255) $false | Out-Null
Add-Text $slide "Final Dice 0.7959" 700 85 210 36 20 (RGB 252 211 77) $true | Out-Null

# Slide 2
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "01 研究任务"
Add-Title $slide "任务背景与目标" "用胸部 X 光图像自动分割 COVID-19 感染区域，输出可视化掩码与定量指标。"
Add-Card $slide 56 170 250 230 "输入" "胸部 X 光图像`nQaTa-COV19 数据集`nPNG/JPG/DICOM 可扩展" (RGB 219 234 254)
Add-Card $slide 355 170 250 230 "模型" "TransUNet2D v18 基线`nLowDice 专家模型池`nv18-aware 门控选择" (RGB 254 243 199)
Add-Card $slide 654 170 250 230 "输出" "感染区域 Mask`nOverlay 可视化`nDice / IoU / Precision / Recall" (RGB 220 252 231)
Add-Arrow $slide 306 285 355 285 (RGB 71 85 105)
Add-Arrow $slide 605 285 654 285 (RGB 71 85 105)

# Slide 3
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "02 数据与预处理"
Add-Title $slide "数据集与训练设置" "QaTa-COV19 胸片分割数据，统一预处理后进行训练、验证与测试。"
Add-Metric $slide 58 155 180 "样本总量" "9258" "train 7406 / val 925 / test 927" (RGB 37 99 235)
Add-Metric $slide 278 155 180 "低分样本" "158" "v18 Dice < 0.60" (RGB 220 38 38)
Add-Metric $slide 498 155 180 "输入尺寸" "224×224" "灰度归一化 + 二值掩码" (RGB 5 150 105)
Add-Metric $slide 718 155 180 "训练方式" "AMP" "AdamW + Cosine LR" (RGB 124 58 237)
Add-Text $slide "预处理流程：读取图像与标注 → 灰度归一化 → 统一尺寸 → 二值化 Mask → 生成 NPZ 与 manifest.csv" 70 315 830 36 17 (RGB 30 41 59) $true | Out-Null
Add-Text $slide "评价指标：Dice 衡量整体重叠，IoU 对边界偏移更敏感，Precision 反映误检，Recall 反映漏检。" 70 370 820 60 15 (RGB 71 85 105) $false | Out-Null

# Slide 4
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "03 方法框架"
Add-Title $slide "系统级分割流程" "从胸片输入到分割掩码、误差分析与 Web 展示。"
Add-Picture $slide (Join-Path $FigDir "fig_method_framework_polished.png") 72 135 815 300 | Out-Null
Add-Text $slide "早期方案以 TransUNet 系列和 UCTRansNet 融合为主；本轮优化转向低分样本专家化纠错。" 92 458 780 38 14 (RGB 71 85 105) $false | Out-Null

# Slide 5
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "04 模型迭代"
Add-Title $slide "从单模型到困难样本专家" "单纯堆叠模块提升有限，低分样本需要专项优化。"
Add-Card $slide 55 150 190 230 "Baseline" "U-Net / DynUNet / Attention U-Net / Swin-Unet / 基础 TransUNet" (RGB 226 232 240)
Add-Card $slide 275 150 190 230 "TransUNet 系列" "v11-v20：窗口注意力、DropPath、边界精炼、频域与先验尝试" (RGB 219 234 254)
Add-Card $slide 495 150 190 230 "传统融合" "v15：TransUNet2D v14 + UCTRansNet 概率级融合，Dice 0.7879" (RGB 254 243 199)
Add-Card $slide 715 150 190 230 "最终策略" "v18-aware gate：按错误类型选择 LowDice 专家，Dice 0.7959" (RGB 254 226 226)
Add-Text $slide "关键判断：创新点不能只写结构堆叠，要用低分样本分析证明改动确实解决了什么错误。" 75 425 810 45 16 (RGB 15 23 42) $true | Out-Null

# Slide 6
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "05 低分样本分析"
Add-Title $slide "v18 低 Dice 样本主要错在哪里？" "对 158 个 Dice < 0.60 的样本按错误类型拆分。"
Add-Picture $slide (Join-Path $FigDir "fig_lowdice_failure_distribution_ppt.png") 60 130 480 270 | Out-Null
Add-Card $slide 590 140 300 68 "边界偏移" "49 例：位置大体正确，但边缘重叠不足。" (RGB 219 234 254)
Add-Card $slide 590 225 300 68 "过分割误检" "42 例：预测面积偏大，Precision 下降。" (RGB 254 226 226)
Add-Card $slide 590 310 300 68 "漏检与错位" "FN、wrong location、missed all 共同构成主要困难来源。" (RGB 237 233 254)
Add-Text $slide "这一步决定了后续不是训练一个「大而全」的模型，而是分成 FP / FN / Boundary / Actual-error 专家。" 75 438 810 40 15 (RGB 71 85 105) $false | Out-Null

# Slide 7
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "06 v18-aware 多专家门控"
Add-Title $slide "最终优化策略：按错误类型路由" "v18 保底，LowDice 专家纠错。"
Add-Picture $slide (Join-Path $FigDir "fig_v18aware_pipeline_ppt.png") 45 125 870 310 | Out-Null
Add-Text $slide "路由规则：boundary_or_shift → basev18；over-segmented FP → actual-fp；under-segmented FN → actual-fn / recall；wrong location → actual-boundary。" 62 462 850 40 14 (RGB 51 65 85) $false | Out-Null

# Slide 8
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "07 量化结果"
Add-Title $slide "测试集 Dice 对比" "最终 v18-aware gate 超过传统 v15 融合和单一专家。"
Add-Picture $slide (Join-Path $FigDir "fig_dice_comparison_v18aware.png") 52 125 530 305 | Out-Null
Add-Metric $slide 620 138 240 "v18 Baseline" "0.7740" "全测试集 Dice" (RGB 37 99 235)
Add-Metric $slide 620 258 240 "v18-aware Gate" "0.7959" "+0.0219 vs v18" (RGB 220 38 38)
Add-Metric $slide 620 378 240 "Oracle Upper" "0.8129" "专家池上限分析" (RGB 124 58 237)

# Slide 9
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "08 低分样本提升"
Add-Title $slide "困难样本从 0.4033 提升到 0.5319" "重点不是平均分好看，而是最差样本被明显拉起。"
$table = $slide.Shapes.AddTable(6, 4, 70, 135, 820, 260).Table
$rows = @(
    @("策略", "低分子集 Dice", "Dice<0.60 数量", "说明"),
    @("v18 基线", "0.4033", "158", "原始低分集合"),
    @("LowDice broad", "0.4683", "118", "通用困难样本专家"),
    @("Expanded gate", "0.5276", "91", "多专家错误类型门控"),
    @("v18-aware gate", "0.5319", "97", "边界偏移回退到 basev18"),
    @("Oracle 上限", "0.6315", "61", "每例选最优专家，仅作上限")
)
for ($r = 1; $r -le 6; $r++) {
    for ($c = 1; $c -le 4; $c++) {
        $cell = $table.Cell($r, $c)
        $cell.Shape.TextFrame.TextRange.Text = $rows[$r-1][$c-1]
        $cell.Shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
        $cell.Shape.TextFrame.TextRange.Font.Size = 12
        $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 30 41 59
        if ($r -eq 1) {
            $cell.Shape.Fill.ForeColor.RGB = RGB 15 23 42
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = RGB 255 255 255
            $cell.Shape.TextFrame.TextRange.Font.Bold = -1
        }
    }
}
Add-Text $slide "结论：专家模型池已经有更高上限，后续真正瓶颈是「不依赖真值的门控分类器」。" 75 435 810 40 16 (RGB 15 23 42) $true | Out-Null

# Slide 10
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "09 可视化结果"
Add-Title $slide "分割效果展示：高 Dice 与低 Dice 对照" "Original / Ground Truth / Prediction / Overlay / Error Map"
Add-Picture $slide (Join-Path $FigDir "fig_qual_compare_high.png") 40 130 420 260 | Out-Null
Add-Picture $slide (Join-Path $FigDir "fig_qual_compare_low.png") 500 130 420 260 | Out-Null
Add-Text $slide "左：高 Dice 样本，预测与标注基本一致；右：低 Dice / 困难样本，用于分析误检、漏检和错位问题。" 70 428 820 42 15 (RGB 71 85 105) $false | Out-Null

# Slide 11
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "10 Web 系统"
Add-Title $slide "网页版演示功能" "本地 Web App：上传胸片 → 模型推理 → 结果预览与下载。"
Add-Card $slide 65 145 245 230 "输入与参数" "上传 PNG/JPG/DICOM`n选择 checkpoint/config`n设置 threshold / min_area / TTA" (RGB 219 234 254)
Add-Card $slide 357 145 245 230 "结果预览" "Original`nMask`nOverlay`n感染面积占比" (RGB 220 252 231)
Add-Card $slide 650 145 245 230 "评估与保存" "自动匹配 GT mask`nDice / IoU / Precision / Recall`n保存到 outputs/web_infer" (RGB 254 243 199)
Add-Text $slide "启动命令：python app_v12_web.py    访问地址：http://127.0.0.1:8000" 75 425 830 32 17 (RGB 220 38 38) $true | Out-Null

# Slide 12
$slide = Add-BlankSlide $presentation
Add-TopBar $slide "11 总结"
Add-Title $slide "项目贡献与后续工作" "围绕真实低分样本做可解释优化。"
Add-Card $slide 60 145 260 245 "主要贡献" "1. 完成胸片感染区域分割流程`n2. 构建 Web 可视化系统`n3. 基于 v18 低分样本建立错误类型分析`n4. 设计 LowDice 多专家与 v18-aware 门控" (RGB 219 234 254)
Add-Card $slide 350 145 260 245 "最终结果" "全测试集 Dice：0.7959`n低分子集 Dice：0.4033 → 0.5319`n专家 Oracle 上限：0.8129" (RGB 254 226 226)
Add-Card $slide 640 145 260 245 "后续改进" "1. 训练无真值门控分类器`n2. 增加外部数据验证`n3. 优化低分错位样本`n4. 将 v18-aware gate 接入 Web 端" (RGB 220 252 231)
Add-Text $slide "Q & A" 410 438 160 42 28 (RGB 15 23 42) $true | Out-Null

$presentation.SaveAs($OutPath)
$presentation.Close()
$ppt.Quit()
Write-Output "saved=$OutPath"


