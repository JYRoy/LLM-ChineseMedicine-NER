from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
import datasets
from itertools import chain
from tqdm.auto import tqdm
import torch
import json
import numpy as np
import re
from collections import defaultdict


def token_map(exapmle):
    """
    将输入和输出示例转换为模型输入和标签的格式。

    参数:
    exapmle (dict): 包含输入和输出的示例字典。

    返回:
    dict: 包含模型输入和标签的字典。
    """
    input = exapmle["input"]
    output = exapmle["target"]
    # 对输入进行格式化处理
    # 形成：<|im_start|>问：\n XXXXXXXXXX \n答：\n
    input = "<|im_start|>问：\n{}\n答：\n".format(
        input.strip().replace("答：", "").strip()
    )
    # 对输出进行格式化处理，添加 <|im_end|> 标记
    output = "{}\n<|im_end|>".format(output.strip())
    # 对输入部分进行分词（不添加特殊标记），并计算输入部分的长度
    input_len = len(tokenizer(input, add_special_tokens=False)["input_ids"])
    # 对输入和输出拼接后进行分词（不添加特殊标记）
    results = tokenizer(text=input + output, add_special_tokens=False)
    # 生成 labels 列表，将输入部分的 labels 设为 -100，输出部分为对应的 input_ids
    results["labels"] = [-100] * input_len + results["input_ids"][input_len:]
    # 返回包含 input_ids 和 labels 的字典
    return {"input_ids": results["input_ids"], "labels": results["labels"]}


def get_entity(text, pattern, entity_types):
    """
    从给定的文本中提取指定类型的实体。

    参数:
    text (str): 要提取实体的文本。
    pattern (re.Pattern): 用于匹配实体的正则表达式模式。
    entity_types (list): 要提取的实体类型列表。

    返回:
    list: 包含提取的实体的列表，每个实体是一个字典，键为实体类型，值为实体列表。
    """
    # 初始化一个空列表，用于存储结果
    result_list = []
    # 使用正则表达式模式在文本中查找所有匹配项
    matches = pattern.findall(text)
    # 初始化一个字典，用于存储每个实体类型的实体列表
    entity_dict = {etype: [] for etype in entity_types}

    # 遍历所有匹配项
    for entity_type, entity in matches:
        # 去除实体类型中的 "实体" 后缀
        entity_type = entity_type.replace("实体", "")
        # 如果实体类型在实体类型列表中，则将实体添加到相应的列表中
        if entity_type in entity_dict:
            entity_dict[entity_type].append(entity)

    # 遍历实体字典
    for etype, entities in entity_dict.items():
        # 如果实体列表不为空，则将其添加到结果列表中
        if entities:
            result_list.append({etype: entities})
    # 返回结果列表
    return result_list


def calculate_f1(answer, pred):
    """
    计算预测结果与标准答案之间的F1分数。

    参数:
    answer (list): 标准答案列表。
    pred (list): 预测结果列表。

    返回:
    float: F1分数。
    """
    # 计算预测结果和标准答案的交集，即真正例的数量
    true_positive = len(set(answer) & set(pred))
    # 计算精确率，即真正例数量除以预测结果数量
    precision = true_positive / len(pred) if pred else 0
    # 计算召回率，即真正例数量除以标准答案数量
    recall = true_positive / len(answer) if answer else 0
    # 如果精确率和召回率都为0，则返回0.0
    if precision + recall == 0:
        return 0.0
    # 计算F1分数，即2倍的精确率乘以召回率除以精确率加召回率
    return 2 * (precision * recall) / (precision + recall)


def calculate_f1_scores(data):
    """
    计算每个实体类型的平均F1分数。

    参数:
    data (list): 包含预测结果和标准答案的列表。

    返回:
    dict: 每个实体类型的平均F1分数。
    """
    # 初始化一个字典，用于存储每个实体类型的F1分数列表
    f1_scores = defaultdict(list)

    # 遍历数据列表中的每个元素
    for item in data:
        # 初始化两个字典，用于存储标准答案和预测结果的实体
        answer_dict = defaultdict(list)
        pred_dict = defaultdict(list)

        # 遍历标准答案列表中的每个元素
        for ans in item["answer"]:
            # 遍历标准答案字典中的每个键值对
            for key, value in ans.items():
                # 将实体添加到相应的列表中
                answer_dict[key].extend(value)

        # 遍历预测结果列表中的每个元素
        for prd in item["pred"]:
            # 遍历预测结果字典中的每个键值对
            for key, value in prd.items():
                # 将实体添加到相应的列表中
                pred_dict[key].extend(value)

        # 遍历所有实体类型
        for entity_type in set(answer_dict.keys()).union(set(pred_dict.keys())):
            # 获取标准答案中的实体列表
            answer_entities = answer_dict[entity_type]
            # 获取预测结果中的实体列表
            pred_entities = pred_dict[entity_type]
            # 计算F1分数
            f1 = calculate_f1(answer_entities, pred_entities)
            # 将F1分数添加到相应的列表中
            f1_scores[entity_type].append(f1)

    # 计算每个实体类型的平均F1分数
    average_f1_scores = {
        etype: sum(scores) / len(scores) for etype, scores in f1_scores.items()
    }

    # 返回平均F1分数字典
    return average_f1_scores


def get_results():
    """
    生成并保存模型的预测结果，并计算每个实体类型的平均F1分数。

    该函数执行以下步骤：
    1. 将模型设置为评估模式。
    2. 遍历测试数据集的每个批次。
    3. 对每个批次进行推理，生成预测结果。
    4. 将预测结果和标签解码为文本。
    5. 从解码后的文本中提取查询、答案和预测实体。
    6. 将结果保存到JSON文件中。
    7. 计算并打印每个实体类型的平均F1分数。
    """
    # 将模型设置为评估模式
    model.eval()
    # 初始化一个空列表，用于存储所有结果
    results_all = []
    # 定义要提取的实体类型列表
    entity_types = [
        "临床表现",
        "中医治则",
        "方剂",
        "西医诊断",
        "其他治疗",
        "西医治疗",
        "中医证候",
        "中药",
        "中医治疗",
        "中医诊断",
    ]
    # 编译正则表达式模式，用于匹配实体
    # \u4e00-\u9fa5 表示 unicode 中所有的中文字符
    pattern = re.compile(r"(\w+实体)：([\u4e00-\u9fa5]+)")
    # 使用tqdm显示进度条，遍历测试数据集的每个批次
    for batch in tqdm(test_data, total=len(test_data)):
        # 禁用梯度计算，加快推理速度
        with torch.no_grad():
            # 将输入和标签移动到GPU上
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            # 对输入进行推理，生成输出
            outputs = model(input_ids=input_ids, labels=labels)
            # 获取预测结果
            preds = outputs.logits.argmax(-1)
            # 将预测结果和标签移动到CPU上，并转换为numpy数组
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            # 对预测结果进行解码，忽略特殊标记
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            # 将标签中的-100替换为填充标记的ID
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # 对标签进行解码，忽略特殊标记
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # 将结果添加到列表中
            results_all.append(
                {
                    "query": labels[0].split("答：")[0].strip(),
                    "answer": get_entity(
                        labels[0].split("答：")[1].strip(), pattern, entity_types
                    ),
                    "pred": get_entity(
                        decoded_preds[0].split("答：")[1].strip(), pattern, entity_types
                    ),
                }
            )
    # 将结果保存到JSON文件中
    with open(f"ruslts.json", "w", encoding="utf-8") as json_file:
        json.dump(results_all, json_file, ensure_ascii=False, indent=4)
    # 计算每个实体类型的平均F1分数
    f1_score = calculate_f1_scores(results_all)
    # 打印平均F1分数
    print(f1_score)


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(
        "pretrain_model", trust_remote_code=True
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("pretrain_model", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.decode(151645)
    test_data = datasets.Dataset.from_json("data/test.json")
    test_data = test_data.map(
        token_map,
        remove_columns=[
            "input",
            "target",
            "answer_choices",
            "task_dataset",
            "task_type",
            "sample_id",
        ],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    test_data = DataLoader(test_data, batch_size=1, collate_fn=data_collator)
    get_results()
