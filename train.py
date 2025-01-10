import os
import threading

import numpy as np
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from peft import TaskType
from torch.nn import CrossEntropyLoss
import datasets
from transformers import AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from itertools import chain

from tqdm.auto import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from rouge_chinese import Rouge

max_source_length = 128
max_target_length = 64
max_seq_length = max_source_length + max_target_length


def token_map_with_cut(examples):
    """
    训练效果不好，会 cut 掉一些有用信息
    """

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples["input"])):
        input = examples["input"][i]
        output = examples["target"][i]
        # 对输入进行格式化处理
        # 形成：<|im_start|>问：\n XXXXXXXXXX \n答：\n
        input = "<|im_start|>问：\n{}\n答：\n".format(
            input.strip().replace("答：", "").strip()
        )
        # 对输出进行格式化处理，添加 <|im_end|> 标记
        output = "{}\n<|im_end|>".format(output.strip())
        a_ids = tokenizer.encode(text=input, add_special_tokens=False)
        # 对答案进行编码，不添加特殊标记
        b_ids = tokenizer.encode(text=output, add_special_tokens=False)
        # 如果提示长度超过最大源长度，截断
        if len(a_ids) > max_source_length - 1:
            a_ids = a_ids[: max_source_length - 1]
        # 如果答案长度超过最大目标长度，截断
        if len(b_ids) > max_target_length - 2:
            b_ids = b_ids[: max_target_length - 2]
        # 构建包含特殊标记的输入 id，[CLS]和[SEP]
        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        input_len = len(a_ids)
        # 构建标签，上下文部分为 -100，其余部分为输入 id
        mask_position = input_len - 1
        labels = [-100] * input_len + input_ids[mask_position + 1 :]
        # 计算填充长度
        pad_len = max_seq_length - len(input_ids)
        # 对输入 id 进行填充
        input_ids = input_ids + [151645] * pad_len
        # 对标签进行填充
        labels = labels + [-100] * pad_len
        model_inputs["input_ids"].append(input_ids)
        # 将处理好的标签添加到标签列表
        model_inputs["labels"].append(labels)
    return model_inputs


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


def group_texts(examples):
    """
    将输入的文本示例分组为固定大小的块。

    参数:
    examples (dict): 包含文本示例的字典，键为特征名称，值为特征列表。

    返回:
    dict: 分组后的文本示例字典，键为特征名称，值为分块后的特征列表。
    """
    # 设置每个块的大小为128
    block_size = 128
    # 将所有示例的特征连接成一个长列表
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # 计算连接后的文本总长度
    total_length = len(concatenated_examples["input_ids"])

    # 如果总长度大于等于块大小，则将总长度调整为块大小的倍数
    if total_length >= block_size:
        total_length = (total_length // block_size + 1) * block_size
    result = {}
    for k, t in concatenated_examples.items():
        # 如果总长度大于当前特征的长度，则进行填充
        if total_length > len(t):
            # 如果是输入ID，则用结束标记填充
            if k == "input_ids":
                t = t + [151645] * (total_length - len(t))
            # 否则用-100填充
            else:
                t = t + [-100] * (total_length - len(t))

        # 将填充后的特征分块
        truncs = [t[i : i + block_size] for i in range(0, total_length, block_size)]
        # 将分块后的特征添加到结果字典中
        result[k] = truncs

    return result


def compute_metrics(eval_preds):
    """
    计算预测结果的评估指标。

    参数:
    eval_preds (tuple): 包含预测结果和标签的元组。

    返回:
    dict: 包含评估指标的字典。
    """
    preds, labels = eval_preds
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    # 使用 tokenizer 对预测结果进行解码，忽略特殊标记
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # 将标签中的 -100 替换为 151645
    labels = np.where(labels != -100, labels, 151645)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        hypothesis = " ".join(hypothesis)
        if not hypothesis:
            hypothesis = "-"
        scores = rouge.get_scores(hypothesis, " ".join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu(
            [list(label)], list(pred), smoothing_function=SmoothingFunction().method3
        )
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True
    )
    # 列：['input', 'task_type', 'task_dataset', 'sample_id', 'answer_choices', 'target']
    train_data = datasets.Dataset.from_json("data/train.json")
    # map 完成后的列：['input_ids', 'labels']
    train_data = train_data.map(
        token_map,
        remove_columns=[
            "input",
            "target",
            "answer_choices",
            "task_dataset",
            "sample_id",
            "task_type",
        ],
    )
    # 使用 map 方法对训练数据集应用 group_texts 函数进行分组处理，batched=True 表示按批次处理
    train_data = train_data.map(group_texts, batched=True)

    test_data = datasets.Dataset.from_json("data/test.json")
    test_data = test_data.map(
        token_map,
        remove_columns=[
            "input",
            "target",
            "answer_choices",
            "task_dataset",
            "sample_id",
            "task_type",
        ],
    )
    test_data = test_data.map(group_texts, batched=True)

    tokenizer.pad_token = tokenizer.decode(151645)
    # 配置 LoraConfig，设置相关参数，例如秩为 8，任务类型为因果语言模型等
    config = LoraConfig(
        r=8,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True
    )
    print(model)

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    train_data = DataLoader(train_data, batch_size=4, collate_fn=data_collator)
    test_data = DataLoader(test_data, batch_size=4, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 创建 CrossEntropyLoss 损失函数，忽略索引为 -100 的元素
    criterion = CrossEntropyLoss(ignore_index=-100)
    train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {train_param}")
    # 创建 Accelerator 对象，设置梯度累积步数为 8
    accelerator = Accelerator(gradient_accumulation_steps=8)
    epochs = 4
    # 创建 CosineAnnealingLR 学习率调度器，根据优化器、训练数据长度和 epoch 数等调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_data) * epochs / 8
    )

    (
        model,
        optimizer,
        train_data,
        test_data,
        scheduler,
    ) = accelerator.prepare(model, optimizer, train_data, test_data, scheduler)

    total_loss = 0
    for epoch in range(epochs):
        for step_id, batch in tqdm(
            enumerate(train_data), desc=f"Epoch {epoch}", total=len(train_data)
        ):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                outputs = model(input_ids=input_ids, labels=labels)
                loss = criterion(
                    outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)
                )
                accelerator.backward(loss)
                total_loss += loss.item()
                if (
                    threading.current_thread().getName() == "MainThread"
                    and step_id % 100 == 0
                ):
                    print(f"train_loss: {loss.item()}")
                loss_main = loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for batch in tqdm(test_data, desc=f"Epoch {epoch}", total=len(test_data)):
            with torch.no_grad():
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                outputs = model(input_ids=input_ids, labels=labels)
                preds = outputs.logits.argmax(-1)
                score_dict = compute_metrics((preds, labels))
                if threading.current_thread().getName() == "MainThread":
                    print(score_dict)

    output_dir = "pretrain_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
