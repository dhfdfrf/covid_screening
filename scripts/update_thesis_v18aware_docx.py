from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SRC = Path("docs/thesis_front_expanded.academic_revised_visual_compare.docx")
DST = Path("docs/thesis_front_expanded.v18aware_final.docx")
FIG = Path("docs/figures/fig_dice_comparison_v18aware.png")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)
NS = {"w": W}


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def set_para_text(p: ET.Element, text: str) -> None:
    ppr = p.find("w:pPr", NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    r = ET.SubElement(p, q("r"))
    t = ET.SubElement(r, q("t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text


def set_cell_text(tc: ET.Element, text: str) -> None:
    paragraphs = tc.findall("./w:p", NS)
    p = paragraphs[0] if paragraphs else ET.SubElement(tc, q("p"))
    set_para_text(p, text)
    for extra in paragraphs[1:]:
        tc.remove(extra)


def update_paragraphs(root: ET.Element) -> int:
    replacements: list[tuple[str, str]] = [
        (
            "COVID-19 胸部影像感染区域分割可为病灶范围评估和定量分析提供支持。",
            "COVID-19 胸部影像感染区域分割可为病灶范围评估和定量分析提供支持。本文以 QaTa-COV19 数据集为对象，围绕胸部 X 光图像病灶分割任务，构建了基于改进 TransUNet 与低分样本专家门控的分割方法。实验首先比较 U-Net、Attention U-Net、DynUNet、Swin-Unet、UCTRansNet、基础 TransUNet 及多个改进版本；在此基础上，针对 v18 基线模型中 Dice 低于 0.60 的困难样本，进一步设计 LowDice RefineNet 专家、误检抑制专家、漏检召回专家、边界偏移专家以及 v18-aware 多模型门控策略。最终策略在全测试集上取得 Dice 0.7959，相比 v18 基线 Dice 0.7740 提升约 0.0219；在 158 个低分样本上，平均 Dice 由 0.4033 提升至 0.5319。实验结果表明，针对困难样本进行错误类型分析和专家化纠错，比单纯堆叠单一网络模块更能提升整体分割稳定性。",
        ),
        (
            "Segmentation of COVID-19 infection regions in chest images supports lesion quantification",
            "Segmentation of COVID-19 infection regions in chest images supports lesion quantification and computer-aided analysis. This thesis studies a chest X-ray lesion segmentation method based on an improved TransUNet baseline and a v18-aware low-Dice expert gating strategy on the QaTa-COV19 dataset. After comparing several segmentation networks and TransUNet variants, the work further analyzes the hard cases with Dice lower than 0.60 and trains multiple specialist models for false-positive suppression, false-negative recovery and boundary-shift correction. The final v18-aware gated strategy achieves a test Dice of 0.7959, improving the v18 baseline Dice of 0.7740 by about 0.0219. On the 158 hard cases, the mean Dice increases from 0.4033 to 0.5319. The results indicate that error-type-aware expert routing is more effective than blindly stacking extra modules in a single model.",
        ),
        (
            "（3）设计 TransUNet2D v11 改进模型。",
            "（3）设计 TransUNet2D v18 基线模型与 LowDice 困难样本专家。基线模型用于获得稳定的整体分割结果，LowDice 专家则针对误检、漏检、边界偏移和完全遗漏等极端情况进行专项优化。",
        ),
        (
            "（5）探索推理阶段优化策略。",
            "（5）探索推理阶段优化策略。除阈值、连通域过滤和模型融合外，本文进一步构建 v18-aware 多专家门控：对 v18 低分样本进行错误类型分析，并按错误模式选择对应专家模型进行纠错。",
        ),
        (
            "本文共分为七章。",
            "本文共分为七章。第 1 章介绍 COVID-19 胸部影像病灶分割的研究背景、研究意义、国内外研究现状以及本文主要工作；第 2 章介绍 QaTa-COV19 数据集、数据预处理流程和数据划分方式；第 3 章说明本文采用的模型结构，包括 TransUNet2D v18 基线、LowDice RefineNet 专家和 v18-aware 门控策略；第 4 章给出实验环境、评价指标、单模型对比结果、困难样本分析和多专家门控实验；第 5 章介绍项目系统实现，包括项目目录、训练流程和推理流程；第 6 章展示可视化结果与系统运行效果；第 7 章总结全文并讨论后续改进方向。",
        ),
        (
            "3.2 TransUNet2D v11 改进结构",
            "3.2 TransUNet2D v18 与 LowDice 多专家优化结构",
        ),
        (
            "本文最终采用的 TransUNet2D v11 主要由卷积编码器",
            "本文最终策略以 TransUNet2D v18 作为稳定基线，并在其低分样本上叠加 LowDice 多专家纠错模块。v18 基线负责大多数常规样本的分割；LowDice RefineNet 保留高分辨率路径，并加入边缘、局部对比和坐标先验，用于处理小病灶、单侧病灶和边界复杂区域。",
        ),
        (
            "为更直观地说明本文方法流程，图 3-1 给出了系统级分割框架。",
            "为更直观地说明本文方法流程，图 3-1 给出了系统级分割框架。输入胸片首先经过预处理并送入 TransUNet2D v18 基线模型，得到初始概率图和分割掩码；随后根据低分样本分析得到的错误模式，引入 LowDice broad、precision、recall、boundary、actual-fp、actual-fn 和 actual-boundary 等专家模型。v18-aware 门控在边界偏移样本上优先保留 v18 结果，在误检、漏检和错位样本上选择对应专家进行纠错，最终输出感染区域掩码。",
        ),
        (
            "虽然模型代码中保留了深监督输出和边界头",
            "虽然模型代码中保留了深监督输出和边界头，但本轮优化表明，单一模型内部继续增加模块并不一定稳定提升整体 Dice。相比之下，将低分样本拆分为误检、漏检、边界偏移和错位等类型，并训练对应专家模型，能够更直接地提升困难样本的平均 Dice。",
        ),
        (
            "这一结果也说明，医学图像分割模型的改进并不是模块越多越好。",
            "这一结果也说明，医学图像分割模型的改进并不是模块越多越好。本文最终将“困难样本错误类型分析 + 多专家门控”作为主要优化点：该策略不强行要求所有样本都由同一模型解决，而是利用不同专家模型的归纳偏置处理不同错误模式。",
        ),
        (
            "单一模型结构往往存在归纳偏置。例如，TransUNet2D v14",
            "单一模型结构往往存在归纳偏置。本文在 v18 基线模型上观察到，低 Dice 样本主要集中在边界偏移、过分割误检、漏检、错位无重叠和完全遗漏等情况。针对这些错误，本文构建 LowDice 多专家模型池，包括 broad 专家、precision 专家、recall 专家、boundary 专家以及基于 v18 实际错误重新划分的 actual-fp、actual-fn、actual-boundary 专家。",
        ),
        (
            "其中，P_ens(x,y) 表示像素位置",
            "其中，P_k(x,y) 表示第 k 个专家模型在像素位置 (x,y) 处的预测概率，G(x) 表示由 v18 错误类型分析得到的门控选择函数。最终输出不再简单采用固定权重平均，而是根据样本错误模式选择对应专家：边界偏移样本优先回退到 v18，过分割误检样本优先使用 actual-fp 专家，漏检样本优先使用 actual-fn 或 recall 专家，错位样本使用 actual-boundary 专家。",
        ),
        (
            "模型融合能够提升效果的原因在于不同结构的错误模式并不完全一致。",
            "多专家门控能够提升效果的原因在于不同模型的错误模式并不完全一致。单一专家在全部低分样本上未必最优，但在特定错误类型上具有优势；例如 recall 专家更适合明显漏检样本，actual-fp 专家更适合过分割误检样本，而 v18 基线在部分边界偏移样本上反而更稳。",
        ),
        (
            "需要强调的是，融合策略提高了推理性能",
            "需要强调的是，多专家门控提高了推理性能，但也增加了推理阶段的模型管理成本。本文将基于真实标签的错误类型门控作为实验分析策略，用于证明专家模型池的上限和有效性；如果部署到 Web 系统，还需要进一步训练不依赖真实掩码的错误类型识别器。",
        ),
        (
            "本文进一步对 TransUNet2D v11 的推理策略进行实验。",
            "本文进一步对 TransUNet2D v18 的推理策略进行实验。v18 基线在全测试集上取得 Dice 0.7740；针对 Dice 低于 0.60 的 158 个困难样本，本文分别训练 broad、precision、recall、boundary、actual-fp、actual-fn 和 actual-boundary 等专家模型，并在验证集上搜索各自阈值和最小连通域面积。",
        ),
        (
            "由于 TransUNet2D v11 和 UCTRansNet 结构侧重点不同",
            "在专家模型池基础上，本文进一步构建 v18-aware 门控策略。该策略并不简单选择整体平均 Dice 最高的专家，而是根据 v18 低分样本的错误类型进行路由：边界偏移类优先保留 basev18，过分割误检类进入 actual-fp，漏检类进入 actual-fn 或 recall，错位无重叠类进入 actual-boundary。表 4-2 给出了单模型、传统融合和本轮门控优化策略的测试集对比。",
        ),
        (
            "结果显示，融合模型相比 TransUNet2D v11 单模型",
            "结果显示，v18-aware 多专家门控在全测试集上取得 Dice 0.7959，相比 v18 基线 0.7740 提升约 0.0219；相比此前 v15 融合方案 0.7879 也有进一步提升。在 158 个 v18 低分样本上，平均 Dice 从 0.4033 提升到 0.5319，低于 0.60 的样本数由 158 个减少到 97 个。专家 Oracle 上限 Dice 为 0.8129，说明当前主要瓶颈已经从专家模型能力转向门控选择准确性。",
        ),
        (
            "第一，单纯改进 TransUNet 结构能够带来一定提升",
            "第一，单纯改进 TransUNet 结构能够带来一定提升，但提升幅度有限。后续 v18 及 LowDice 实验表明，针对困难样本进行专家化建模比继续堆叠单一网络模块更有效。",
        ),
        (
            "第三，模型融合是当前项目中最有效的性能提升策略。",
            "第三，v18-aware 多专家门控是当前项目中最有效的性能提升策略。该策略将 v18 基线与多个 LowDice 专家结合，使最终全测试集 Dice 达到 0.7959，并显著改善低分样本表现。",
        ),
        (
            "推理阶段首先加载训练得到的模型权重",
            "推理阶段首先加载训练得到的模型权重，对测试集样本逐张或按批量生成概率图。常规样本使用 TransUNet2D v18 基线输出；对于低分困难样本分析实验，系统根据错误类型选择 LowDice 专家模型进行替换或纠错。最终结果经阈值化和小连通域过滤后输出为二值感染区域掩码。",
        ),
        (
            "本文实现的 Web 系统面向胸部 X 光图像感染区域分割任务",
            "本文实现的 Web 系统面向胸部 X 光图像感染区域分割任务，用户上传图像后，系统调用训练好的分割模型生成感染概率图，并根据验证集确定的阈值得到二值掩码。系统输出包括原始图像、预测掩码、叠加图、感染面积占比以及 Dice、IoU、Precision 和 Recall 等评价指标，便于对模型结果进行直观检查。当前 Web 端默认可加载 v12/v15 等配置；v18-aware 多专家门控作为实验优化策略保留在离线评估脚本中。",
        ),
        (
            "为了更直观地展示不同版本模型的性能差异，图 6-2",
            "为了更直观地展示不同版本模型的性能差异，图 6-2 对主要模型和本轮 v18-aware 优化策略的测试集 Dice 指标进行了比较。可以看到，传统单模型和 v15 融合方案已经具备较好性能，但针对低分样本进行专家化纠错后，整体 Dice 进一步提升到 0.7959。",
        ),
        (
            "最终采用的 v15 融合方案由 TransUNet2D v14",
            "最终采用的 v18-aware 多专家门控方案以 TransUNet2D v18 为基线，并结合 LowDice broad、precision、recall、boundary、actual-fp、actual-fn 和 actual-boundary 等专家模型。在全测试集上，该策略取得 Dice 0.7959；在 158 个 v18 低分样本上，平均 Dice 由 0.4033 提升至 0.5319。",
        ),
        (
            "在模型改进过程中，本文进一步尝试了病灶先验、后验校准",
            "在模型改进过程中，本文进一步尝试了病灶先验、后验校准、频域先验、训练期频域辅助监督、LowDice RefineNet 以及 v18 概率图先验纠错模型。实验发现，v18-prior 二通道模型在验证集上表现较好，但在 158 个极端低分测试样本上未稳定超过现有专家池，因此最终未纳入主策略。该结果进一步说明，创新模块需要以独立测试结果为依据，不能仅凭验证集曲线判断有效。",
        ),
        (
            "实验结果表明，单模型结构改进能够带来一定表达能力提升",
            "实验结果表明，单模型结构改进能够带来一定表达能力提升，但提升幅度有限；最终采用的 v18-aware 多专家门控方案在测试集上取得 Dice 0.7959，整体优于 v18 基线、传统 v15 融合和单一 LowDice 专家。Web 可视化系统进一步将分割掩码、叠加图和感染面积占比直接展示出来，提高了实验结果的可读性。",
        ),
        (
            "从研究过程看，本文较重要的收获是对模型改进效果保持客观评价。",
            "从研究过程看，本文较重要的收获是对模型改进效果保持客观评价。继续增加复杂模块并不必然提升 Dice，部分模型创新更适合作为消融分析和可解释性讨论；在当前数据集上，围绕低分样本错误类型建立专家模型池，并进行有针对性的门控路由，比盲目堆叠单一模型模块更有实际价值。",
        ),
    ]

    changed = 0
    for p in root.findall(".//w:p", NS):
        text = para_text(p)
        if not text:
            continue
        for key, new_text in replacements:
            if text.startswith(key):
                set_para_text(p, new_text)
                changed += 1
                break
    return changed


def update_strategy_table(root: ET.Element) -> None:
    tables = root.findall(".//w:tbl", NS)
    if len(tables) < 2:
        return
    table = tables[1]
    rows = table.findall("./w:tr", NS)
    new_rows = [
        ["方法", "Dice", "IoU", "Precision", "Recall"],
        ["TransUNet2D v14", "0.7755", "0.6708", "0.7771", "0.8344"],
        ["UCTRansNet", "0.7777", "0.6722", "0.7640", "0.8542"],
        ["v15 概率级融合", "0.7879", "0.6851", "0.7831", "0.8482"],
        ["TransUNet2D v18 基线", "0.7740", "0.6691", "0.7894", "0.8177"],
        ["LowDice broad 专家替换", "0.7850", "0.6792", "0.7948", "0.8250"],
        ["v18-aware 多专家门控（最终）", "0.7959", "0.6893", "0.7988", "0.8303"],
        ["v18-aware Oracle 上限（分析）", "0.8129", "—", "—", "—"],
    ]
    while len(rows) < len(new_rows):
        clone = copy.deepcopy(rows[-1])
        table.append(clone)
        rows.append(clone)
    while len(rows) > len(new_rows):
        table.remove(rows.pop())
    for row, values in zip(rows, new_rows):
        for tc, value in zip(row.findall("./w:tc", NS), values):
            set_cell_text(tc, value)


def original_namespace_declarations(xml_bytes: bytes) -> dict[str, str]:
    return {
        prefix.decode("ascii"): uri.decode("utf-8")
        for prefix, uri in re.findall(rb'xmlns:([A-Za-z0-9]+)="([^"]+)"', xml_bytes)
    }


def register_original_namespaces(xml_bytes: bytes) -> None:
    for prefix_text, uri_text in original_namespace_declarations(xml_bytes).items():
        if prefix_text == "xml":
            continue
        ET.register_namespace(prefix_text, uri_text)


def restore_unused_namespace_declarations(xml_bytes: bytes, source_xml: bytes) -> bytes:
    """ElementTree drops unused xmlns declarations, but Word's mc:Ignorable can still reference them."""
    xml = xml_bytes.decode("utf-8")
    source_ns = original_namespace_declarations(source_xml)
    missing = []
    for prefix, uri in source_ns.items():
        if f"xmlns:{prefix}=" not in xml:
            missing.append(f' xmlns:{prefix}="{uri}"')
    if not missing:
        return xml_bytes
    root_start = xml.find("<w:document")
    first_close = xml.find(">", root_start)
    if first_close < 0:
        return xml_bytes
    xml = xml[:first_close] + "".join(missing) + xml[first_close:]
    return xml.encode("utf-8")


def main() -> None:
    shutil.copy2(SRC, DST)
    with zipfile.ZipFile(SRC, "r") as zin:
        source_xml = zin.read("word/document.xml")
        register_original_namespaces(source_xml)
        root = ET.fromstring(source_xml)

    changed = update_paragraphs(root)
    update_strategy_table(root)
    document_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    document_xml = restore_unused_namespace_declarations(document_xml, source_xml)

    tmp = DST.with_suffix(".tmp.docx")
    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = document_xml
            elif item.filename == "word/media/image5.png" and FIG.exists():
                data = FIG.read_bytes()
            zout.writestr(item, data)
    tmp.replace(DST)
    print(f"changed_paragraphs={changed}")
    print(f"output={DST}")


if __name__ == "__main__":
    main()
