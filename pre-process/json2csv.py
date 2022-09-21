import csv
import os
import json
import pandas as pd

input_file = "data/Musical_Instruments_5.json"
input_json = open(input_file, "r", encoding="utf-8")
output_file = "data/Musical_Instruments_5.csv"

#先判断是否为空文件
def is_empty_file(file_path:str):
    assert isinstance(file_path,str), f"file_path参数类型不是字符串类型:{type(file_path)}"
    assert os.path.isfile(file_path), f"file_path不是一个文件：{file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(0)
        if not first_char:
            return True
    return False

if is_empty_file(input_file):
    print("transfering...")
    with open(output_file, "w", encoding="utf-8") as output_csv:
        csv_writer = csv.writer(output_csv)
        write_line = 0
        for line in input_json.readlines():
            dic = json.loads(line)
            #第一行写表头
            if write_line == 0:
                csv_writer.writerow(dic)
                write_line = 1
            #其他正常写values
            csv_writer.writerow(dic.values())

#选取reviewerID,asin,reviewText,overall生成新文件
data = pd.read_csv(output_file,encoding='utf-8')
data.loc[:,['reviewerID','asin','reviewText','overall']].to_csv('data/Musical_Instruments.csv')

print("Done")

