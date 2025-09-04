# 🤖 自动测试时对齐系统 (Automatic Test-Time Alignment System)

这个系统实现了完全自动化的偏好对齐，无需用户手动指定任何偏好设置。

## 🎯 核心功能

### 1. **自动偏好检测**
- 分析用户 prompt 中的隐含偏好信号
- 基于关键词、模式和上下文自动识别偏好类型
- 支持 8 种偏好类型：concise, verbose, creative, formal, pleasant, complex, uplifting, sycophantic

### 2. **智能偏好生成**
- 根据用户意图自动生成相应的偏好提示词
- 如："Write a brief summary" → "Your answer should be concise as much as possible"
- 如："Tell me a creative story" → "Your answer should be creative as much as possible"

### 3. **自动验证提示生成**
- 生成 4 种类型的验证器：preference_alignment, style_consistency, preference_completeness, answer_relevance
- 每个验证器都包含完整的验证逻辑和示例

## 🚀 使用方法

### 基础使用

```python
from prompts.auto_alignment_system import create_auto_alignment_system

# 创建系统实例
system = create_auto_alignment_system()

# 用户输入
user_question = "Write a brief summary of quantum computing"
ai_response = "Quantum computing uses quantum mechanics..."

# 自动生成验证提示
verification_prompts = system.generate_self_rewarding_prompts(user_question, ai_response)

# 输出：4个验证器的完整提示词
for verifier_name, prompt in verification_prompts.items():
    print(f"{verifier_name}: {prompt}")
```

### 与现有 vera_prompts 集成

```python
from prompts.vera_prompts import get_vera_prompt

# 现在支持自动模式（不需要手动指定 preference）
verification_prompt = get_vera_prompt("preference_alignment", user_question, ai_response)
# 系统会自动检测并生成相应的偏好设置
```

### 完整评估流程

```python
# 包含偏好分析和验证结果的完整评估
results = system.evaluate_alignment(user_question, ai_response, evaluator_function)

# 结果包含：
# - preference_analysis: 偏好检测详情  
# - verification_prompts: 生成的验证提示
# - verification_results: 验证结果（如果提供评估函数）
# - alignment_score: 总体对齐分数
```

## 🧠 偏好检测机制

### 1. **关键词检测**
- 识别用户 prompt 中的偏好指示词
- 例如："brief", "detailed", "creative", "formal" 等

### 2. **模式匹配**
- 使用正则表达式识别特定的问题模式
- 例如："how to...step by step" → verbose preference

### 3. **上下文分析**
- 基于问题类型推断偏好
- 例如："tutorial" → verbose, "summary" → concise

### 4. **启发式规则**
- 基于问题长度、标点符号等特征
- 例如：问号 + "how" → 倾向于详细回答

## 📊 支持的偏好类型

| 偏好类型 | 描述 | 自动检测关键词示例 |
|---------|------|------------------|
| concise | 简洁性 | "brief", "short", "summarize" |
| verbose | 详细性 | "detailed", "explain", "step by step" |
| creative | 创意性 | "creative", "story", "imaginative" |
| formal | 正式性 | "formal", "professional", "business" |
| pleasant | 友好性 | "friendly", "kind", "polite" |
| complex | 复杂性 | "technical", "advanced", "complex" |
| uplifting | 振奋性 | "motivational", "encouraging", "inspire" |
| sycophantic | 奉承性 | "praise", "compliment", "admire" |

## 🔧 配置选项

### 置信度阈值
```python
# 调整偏好检测的置信度阈值
detected_pref = generator.detect_dominant_preference(user_prompt, threshold=0.3)
```

### 多偏好模式
```python
# 生成多个偏好选项进行综合评估
prompts = system.generate_self_rewarding_prompts(
    user_question, ai_response, use_multiple_preferences=True
)
```

### 偏好建议
```python
# 获取偏好检测的详细分析
suggestions = system.get_preference_suggestions(user_question, top_k=3)
# 返回：[(preference_type, preference_prompt, confidence_score), ...]
```

## 🎯 与 LatentSeek 集成

这个自动对齐系统完美集成到现有的 LatentSeek self-rewarding 框架中：

1. **无缝替换**：现有的 `get_vera_prompt` 函数现在支持自动模式
2. **向后兼容**：仍然支持手动指定偏好的传统模式
3. **多验证器**：提供 4 种不同角度的偏好对齐验证
4. **评分机制**：自动计算对齐分数用于 self-rewarding

## 🚦 运行演示

```bash
python demo_auto_alignment.py
```

该演示展示了系统如何：
- 自动检测不同类型问题的偏好
- 生成相应的验证提示词
- 与现有系统无缝集成

## 🎖️ 主要优势

1. **零配置**：用户无需指定任何偏好设置
2. **智能检测**：基于多种启发式规则自动识别用户意图
3. **全面验证**：从 4 个不同维度验证偏好对齐
4. **易于集成**：与现有 vera_prompts 系统完全兼容
5. **可扩展性**：容易添加新的偏好类型和检测规则

这个系统真正实现了"test-time alignment"的目标，让 AI 能够在测试时自动适应用户的隐含偏好，无需任何额外的用户输入。
