# 跨域泛化实验设计

## 实验目的
验证模型在不同地理区域的迁移能力

## 实验设置

### 实验1: USA → Kenya 迁移
- 训练: SatBird-USA-summer (624 species)
- 测试: SatBird-Kenya (527 species)  
- 挑战: 不同大陆、不同物种组成、不同环境

### 实验2: Summer → Winter 迁移
- 训练: SatBird-USA-summer
- 测试: SatBird-USA-winter (670 species)
- 挑战: 同一地区、不同季节、部分物种重叠

### 实验3: Zero-shot 物种预测
- 训练集排除某些物种
- 测试集包含这些物种
- 验证模型对未见物种的泛化能力

## 评估指标
- mAP (mean Average Precision)
- Top-K accuracy
- Per-species F1
- 迁移性能下降比例
