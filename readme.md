# Kivy AI 应用示例 🤖✨

## 简介 📝

井字棋学习神经网络和强化学习。

参考：[图解神经网络和强化学习：400 行 C 代码训练一个井字棋高手（2025）](https://arthurchiao.art/blog/reinforcement-learning-400-lines-of-c-code-zh/?continueFlag=27570a6e45cc66abbea69288530aea45)

## 核心功能 🚀

* ✨ 基于 Kivy 的图形用户界面 (GUI)
* 🧠 集成并使用 `best_model.pkl` AI 模型
* 🔤 从 `assets/fonts/` 加载并使用自定义字体
* 📦 包含 Android (`.apk`) 和 桌面 (Windows/Linux) 打包说明

## 准备工作 🛠️

```bash
# 创建虚拟环境
uv venv .venv python=3.8

# 安装依赖
uv pip install -r requirements.txt

# 训练模型
python train.py

# 训练模型（Google Gemini 优化，效果更好）
python train-gemini.py

# 运行
python main.py
```
