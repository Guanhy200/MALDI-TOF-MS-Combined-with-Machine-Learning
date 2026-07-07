# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:13:28 2026

@author: LENOVO
"""

import pandas as pd
from pathlib import Path
from decimal import Decimal, InvalidOperation


# =========================
# 1. 文件路径：按实际文件名修改
# =========================
file1_path = r"文件1.xlsx"          # 特征峰、均值、标准差
file2_path = r"文件2.xlsx"          # 样本号 + 各特征峰峰强度
output_path = r"文件2_Zscore.xlsx"  # 输出文件


# =========================
# 2. 自动读取 Excel 或 CSV
# =========================
def read_table(file_path, sheet_name=0):
    """根据文件后缀读取 Excel 或 CSV 文件。"""
    suffix = Path(file_path).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, sheet_name=sheet_name)

    elif suffix == ".csv":
        return pd.read_csv(file_path)

    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


# =========================
# 3. 统一特征峰名称格式
#    避免 100、100.0、'100 ' 等格式导致无法匹配
# =========================
def normalize_feature_name(x):
    """
    将特征峰名称统一为字符串。
    数值型峰位会尽可能转为标准数值字符串，例如：
    100.0 -> '100'
    100.123000 -> '100.123'
    """
    if pd.isna(x):
        return None

    text = str(x).strip()

    try:
        value = Decimal(text)
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError):
        return text


# =========================
# 4. 读取文件1：特征峰、均值、标准差
# =========================
param_df = read_table(file1_path)

# 默认文件1前三列依次为：特征峰、均值、标准差
param_df = param_df.iloc[:, :3].copy()
param_df.columns = ["feature_peak", "mean", "std"]

# 特征峰名称标准化
param_df["feature_key"] = param_df["feature_peak"].apply(normalize_feature_name)

# 均值、标准差转换为数值
param_df["mean"] = pd.to_numeric(param_df["mean"], errors="coerce")
param_df["std"] = pd.to_numeric(param_df["std"], errors="coerce")

# 检查文件1是否存在重复特征峰
duplicate_features = param_df.loc[
    param_df["feature_key"].duplicated(keep=False),
    "feature_peak"
].tolist()

if duplicate_features:
    raise ValueError(
        "文件1中存在重复特征峰，无法确定应使用哪一组均值和标准差：\n"
        f"{duplicate_features}"
    )

# 检查均值或标准差缺失
invalid_param = param_df[
    param_df["mean"].isna() |
    param_df["std"].isna()
]

if not invalid_param.empty:
    print("警告：以下特征峰的均值或标准差为空，无法进行 Z-score：")
    print(invalid_param[["feature_peak", "mean", "std"]])

# 检查标准差是否为 0
zero_std = param_df[param_df["std"] == 0]

if not zero_std.empty:
    print("警告：以下特征峰的标准差为 0，无法进行 Z-score：")
    print(zero_std[["feature_peak", "mean", "std"]])

# 建立“特征峰 -> 均值、标准差”的索引表
param_map = param_df.set_index("feature_key")[["mean", "std"]]


# =========================
# 5. 读取文件2：样本号 + 峰强度矩阵
# =========================
data_df = read_table(file2_path)

# 第一列默认为样本号
sample_id_col = data_df.columns[0]

# 复制结果表，保留样本号
zscore_df = data_df.copy()

# 文件2除第一列以外均视为特征峰列
feature_columns = data_df.columns[1:]

matched_features = []
unmatched_features = []
invalid_std_features = []

# =========================
# 6. 按特征峰匹配并进行 Z-score
# =========================
for col in feature_columns:
    feature_key = normalize_feature_name(col)

    # 文件1中没有该特征峰
    if feature_key not in param_map.index:
        unmatched_features.append(col)
        continue

    mean_value = param_map.loc[feature_key, "mean"]
    std_value = param_map.loc[feature_key, "std"]

    # 均值、标准差缺失或标准差为0时不计算
    if pd.isna(mean_value) or pd.isna(std_value) or std_value == 0:
        invalid_std_features.append(col)
        zscore_df[col] = pd.NA
        continue

    # 转为数值后进行 Z-score
    peak_intensity = pd.to_numeric(data_df[col], errors="coerce")
    zscore_df[col] = (peak_intensity - mean_value) / std_value

    matched_features.append(col)


# =========================
# 7. 输出匹配情况
# =========================
print(f"文件2中特征峰总数：{len(feature_columns)}")
print(f"成功完成 Z-score 的特征峰数：{len(matched_features)}")

if unmatched_features:
    print("\n以下特征峰未在文件1中找到均值和标准差，已保留原始值：")
    print(unmatched_features)

if invalid_std_features:
    print("\n以下特征峰标准差缺失或为0，结果已设为缺失值：")
    print(invalid_std_features)


# =========================
# 8. 导出结果
# =========================
zscore_df.to_excel(output_path, index=False)

print(f"\nZ-score 标准化完成，结果已保存至：{output_path}")