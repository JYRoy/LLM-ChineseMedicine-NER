import datasets
from transformers import AutoTokenizer
from collections import Counter

def token_map_with_cut(examples):
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
        model_inputs["input_ids"].append(a_ids)
        # 将处理好的标签添加到标签列表
        model_inputs["labels"].append(b_ids)
    return model_inputs



if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True
    )
    # 列：['input', 'task_type', 'task_dataset', 'sample_id', 'answer_choices', 'target']
    train_data = datasets.Dataset.from_json("data/train.json")
    # map 完成后的列：['input_ids', 'labels']
    train_data = train_data.map(
        token_map_with_cut,
        batched=True,
        remove_columns=[
            "input",
            "target",
            "answer_choices",
            "task_dataset",
            "sample_id",
            "task_type",
        ],
    )
    input_length = [len(i) for i in train_data["input_ids"]]
    label_length = [len(i) for i in train_data["labels"]]
    print(max(input_length))
    print(max(label_length))
    import matplotlib.pyplot as plt

    # 示例数据
    # 使用 Counter 统计相同元素的个数
    element_counts = Counter(input_length)


    # 获取元素和对应的计数
    elements = element_counts.keys()
    counts = element_counts.values()

    # 绘制柱状图
    plt.bar(elements, counts)
    # 添加标签和标题
    plt.xlabel('Length')
    plt.ylabel('Numbers')
    plt.title('Count of Same Length in the List')


    # 保存图像
    plt.savefig('input.png')
    
    plt.close()
    
    # 使用 Counter 统计相同元素的个数
    element_counts = Counter(label_length)


    # 获取元素和对应的计数
    elements = element_counts.keys()
    counts = element_counts.values()

    # 绘制柱状图
    plt.bar(elements, counts)
    # 添加标签和标题
    plt.xlabel('Length')
    plt.ylabel('Numbers')
    plt.title('Count of Same Length in the List')


    # 保存图像
    plt.savefig('label.png')