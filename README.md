# Medical Text Classification with BERT 🏥

基于BERT的医疗文本多标签分类系统，适用于医学报告自动标注任务。使用PyTorch+Transformers实现，支持混合精度训练和梯度检查点优化。

## 功能特点 ✨
- ✅ 中文医疗文本多标签分类（支持17个解剖区域）
- 🧠 使用`bert-base-chinese`预训练模型
- ⚙️ 支持动态池化策略（mean/cls）
- 🔥 混合精度训练（AMP）加速
- 📈 梯度检查点节省显存
- 📊 输出mLogLoss评估指标
- 📦 自动生成Kaggle格式提交文件

## 安装指南 🛠️
```bash
# 克隆仓库
git clone https://github.com/yourname/medical-bert.git
cd medical-bert

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install torch transformers pandas numpy scikit-learn

# 下载BERT预训练模型
mkdir -p bert-base-chinese
# 从HuggingFace下载中文BERT模型到该目录
```

## 数据准备 📁
1. 将训练数据放入项目根目录：
   - `track1_round1_train_20210222.csv`
   - `track1_round1_testB.csv`
2. 数据格式要求：
   ```csv
   report_ID,description,label
   "12345","患者主诉：胸痛、呼吸困难","0 3 5"
   ```

## 使用方法 🚀
### 训练模型
```bash
python train.py
# 自动保存最佳模型到 ./Project/best_model_mlogloss.pth
# 生成提交文件 ./Project/submission_bert.csv
```

### 模型参数配置
在`train.py`顶部修改以下参数：
```python
# 批量大小/BATCH_SIZE
# 学习率/LEARNING_RATE 
# 最大序列长度/MAX_LENGTH
# 池化策略（'mean'或'cls'）/POOLING_STRATEGY
```

## 项目结构 📐
```
medical-bert/
├── bert-base-chinese/      # BERT预训练模型
├── data/
│   ├── track1_round1_train_20210222.csv
│   └── track1_round1_testB.csv
├── Project/
│   ├── best_model_mlogloss.pth   # 最佳模型
│   └── submission_bert.csv       # 提交文件
├── train.py                  # 训练脚本
└── README.md
```

## 评估指标 📊
使用`mLogLoss`作为主要评估指标：
- 自动在验证集上早停（5个epoch无提升）
- 学习率预热+线性衰减调度

## 贡献指南 🤝
欢迎提交PR！请遵循以下步骤：
1. Fork仓库
2. 创建新分支 (`git checkout -b feature/new`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送分支 (`git push origin feature/new`)
5. 创建Pull Request

## 许可证 📄
详见 [LICENSE](LICENSE) 文件

## 代码片段示例 🧩
```python
# 动态池化策略实现
if self.pooling_strategy == 'mean':
    masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
    pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
else:
    pooled_output = outputs.pooler_output
```

## 技术指标 📋
| 参数 | 值 |
|------|-----|
| 最大序列长度 | 256 |
| 学习率 | 2e-5 |
| 批量大小 | 64 |
| 早停耐心值 | 5 |
| 混合精度训练 | ✅ 启用 |
