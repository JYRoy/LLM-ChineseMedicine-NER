import pandas as pd
import json
import random


def get_label(texts, labels, sample_id, label_choice, input, label2id):

    results = {}
    results["input"] = input
    results["task_type"] = "ner"
    results["task_dataset"] = "custom_data"
    results["sample_id"] = sample_id
    results["answer_choices"] = list(label_choice)
    # 获取 id 到标签的反向映射
    id2label = {v: k for k, v in label2id.items()}

    # 存储最终提取的结果
    extracted_spans = []  # 存 (lable, text) 的 set

    # 临时存储当前标签和对应文本
    current_label = None
    current_text = []

    for char, label_id in zip(texts, labels):
        label = id2label[label_id]
        if label != "O":
            # 如果遇到新标签或者不同的 "B-" 开头的标签，保存之前的结果
            if (
                current_label is None
                or label.startswith("B-")
                or (label != current_label and not label.startswith("I-"))
            ):
                if current_text:
                    extracted_spans.append((current_label, "".join(current_text)))
                    current_text = []
                current_label = label

            # 将字符添加到当前文本中
            current_text.append(char)
        else:
            # 当遇到 'O' 标签时，保存之前的结果并重置
            if current_text:
                extracted_spans.append((current_label[2:], "".join(current_text)))
                current_text = []
            current_label = None

    # 处理最后的文本块
    if current_text:
        extracted_spans.append((current_label[2:], "".join(current_text)))

    # 打印提取结果
    output = "上述句子中的实体包含：\n"
    answer = []
    for label, text_span in extracted_spans:
        answer.append(f"{label}实体：{text_span}")
    output += "\n".join(answer)
    results["target"] = output
    return results


def read_data(file_path):
    """
    从指定文件路径读取数据，并返回文本、标签、标签到ID的映射和标签选项。

    参数:
        file_path (str): 数据文件的路径。

    返回:
        tuple: 包含文本列表、标签ID列表、标签到ID的映射和标签选项列表的元组。
    """

    # 初始化变量
    texts = []
    labels = []
    current_text = []  # 存储当前的句子
    current_labels = []  # 存储当前的句子的标签

    # 打开文件并读取内容
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和多余的空格

            if line == "":
                # 如果遇到空行，代表一个文本段落结束
                if current_text and current_labels:
                    texts.append("".join(current_text))
                    labels.append(" ".join(current_labels))
                    current_text = []
                    current_labels = []
            else:
                # 非空行，继续收集文本和标签
                parts = line.split()  # 按空格分割文本和标签，数据格式为 "word label"
                if len(parts) == 2:
                    word, label = parts
                    current_text.append(word)
                    current_labels.append(label)

        # 处理文件末尾的最后一段（如果没有空行结尾）
        # 如果有空行结尾，这段代码不会执行，current_text 已经在上面的循环中处理了
        if current_text and current_labels:
            texts.append("".join(current_text))
            labels.append(" ".join(current_labels))

    # 统计所有标签
    labels_id = set()
    for label in labels:
        for l in label.split():
            labels_id.add(l)
    # 建立标签到 id 的映射
    label2id = {label: idx for idx, label in enumerate(labels_id)}
    id2label = {idx: label for label, idx in label2id.items()}
    # 统计所有标签
    label_all_id = []
    for label in labels:
        label_all_id.append([label2id[l] for l in label.split(" ")])

    # 统计所有标签的具体描述
    # 例如：“B-西医诊断”，a[2:] 处理后就是“西医诊断”
    label_choice = set()
    for a in label2id.keys():
        if len(a) > 1:
            label_choice.add(a[2:])
    return texts, label_all_id, label2id, list(label_choice)


def create_data(texts, label_all_id, label_choice, label2id, output_name):
    """
    根据给定的文本、标签、标签选项和标签到ID的映射，生成一个包含实体抽取任务的JSON文件。

    参数:
        texts (list): 包含文本的列表。
        label_all_id (list): 包含标签ID的列表。
        label_choice (set): 所有可能的标签选项。
        label2id (dict): 标签到ID的映射。
        output_name (str): 输出文件的名称。

    返回:
        None
    """
    template = [
        "找出指定的实体：\\n[INPUT_TEXT]\\n类型选项：[LIST_LABELS]\\n答：",
        "找出指定的实体：\\n[INPUT_TEXT]\\n实体类型选项：[LIST_LABELS]\\n答：",
        "找出句子中的[LIST_LABELS]实体：\\n[INPUT_TEXT]\\n答：",
        "[INPUT_TEXT]\\n问题：句子中的[LIST_LABELS]实体是什么？\\n答：",
        "生成句子中的[LIST_LABELS]实体：\\n[INPUT_TEXT]\\n答：",
        "下面句子中的[LIST_LABELS]实体有哪些？\\n[INPUT_TEXT]\\n答：",
        "实体抽取：\\n[INPUT_TEXT]\\n选项：[LIST_LABELS]\\n答：",
        "医学实体识别：\\n[INPUT_TEXT]\\n实体选项：[LIST_LABELS]\\n答：",
        "在下面的文本中找出医疗命名实体：\n[INPUT_TEXT]\n请从以下类型中选择医疗命名实体：[LIST_LABELS]\n答：",
        "请从以下文本中提取医疗命名实体：\n[INPUT_TEXT]\n医疗命名实体的类型包括：[LIST_LABELS]\n答：",
        "识别以下文本中的医疗命名实体：\n[INPUT_TEXT]\n请标注以下类型的医疗命名实体：[LIST_LABELS]\n答：",
        "请从以下句子中找出[LIST_LABELS]实体：\n[INPUT_TEXT]\n答：",
        "[INPUT_TEXT]中包含了哪些[LIST_LABELS]实体？请列举出来。\n答：",
        "在给定的文本[INPUT_TEXT]中，请标出所有的[LIST_LABELS]实体。\n答：",
        "[INPUT_TEXT]\n问题：请标出句子中的[LIST_LABELS]实体。\n答：",
        "[INPUT_TEXT]\n问题：你能识别出句子中的哪些[LIST_LABELS]实体吗？\n答：",
        "[INPUT_TEXT]\n问题：根据句子内容，找出其中的[LIST_LABELS]实体。\n答：",
        "请识别以下医学文本中的实体：\\n[INPUT_TEXT]\\n实体类型：[LIST_LABELS]\\n答：",
        "在下述文本中标记出医学实体：\n[INPUT_TEXT]\n可识别的实体有：[LIST_LABELS]\n答：",
        "医学命名实体识别任务：\n请从下面的文本中提取医疗实体：\n[INPUT_TEXT]\n实体类型包括：[LIST_LABELS]\n答：",
        "请标出以下句子中的医疗实体类型为[LIST_LABELS]的实体：\n[INPUT_TEXT]\n答：",
        "请识别并列举出以下文本中属于[LIST_LABELS]类型的医疗实体：\\n[INPUT_TEXT]\\n答：",
        "在下面的句子中找出所有关于[LIST_LABELS]的医疗实体：\\n[INPUT_TEXT]\\n答：",
    ]
    # 初始化一个空列表，用于存储所有的结果
    results_all = []
    # 遍历文本和标签列表
    for i, (text, label) in enumerate(zip(texts, label_all_id)):
        # 随机选择一个模板
        index = random.randint(0, 7)
        # 使用选定的模板生成输入提示
        input = (
            template[index]
            .replace("[INPUT_TEXT]", text)
            .replace("[LIST_LABELS]", "，".join(list(label_choice)))
        )  # "，".join(list(label_choice)) 为 '其他治疗，西医治疗，临床表现，中医证候，中医治则，中医治疗，中药，方剂，中医诊断，西医诊断'
        # 调用get_label函数获取标签结果
        results = get_label(
            text, label, f"f{output_name}_{i}", label_choice, input, label2id
        )
        results_all.append(results)
    with open(f"data/{output_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(results_all, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    for data_str in ["medical.dev", "medical.train", "medical.test"]:
        texts, label_all_id, label2id, label_choice = read_data(f"data/{data_str}")
        create_data(texts, label_all_id, label_choice, label2id, data_str[8:])
