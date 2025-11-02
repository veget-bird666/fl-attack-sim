import data_utils
import model

# 获取模型参数
NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
print(f"用户数: {NUM_USERS}, 电影数: {NUM_MOVIES}")

# 创建模型
test_model = model.create_model(NUM_USERS, NUM_MOVIES)

# 打印所有层的信息
print("\n=== 模型层信息 ===")
for i, layer in enumerate(test_model.layers):
    print(f"{i}: {layer.name} - {type(layer).__name__}")

# 打印权重信息
print("\n=== 模型权重信息 ===")
for i, weight in enumerate(test_model.get_weights()):
    print(f"{i}: 形状 {weight.shape}, 类型 {type(weight)}")

# 特别检查嵌入层
print("\n=== 嵌入层详细信息 ===")
for layer in test_model.layers:
    if 'embedding' in layer.name.lower():
        print(f"嵌入层: {layer.name}")
        weights = layer.get_weights()
        if weights:
            print(f"  权重形状: {weights[0].shape}")