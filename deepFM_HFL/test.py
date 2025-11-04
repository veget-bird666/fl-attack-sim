import data_utils
import model

NUM_USERS, NUM_MOVIES = data_utils.get_data_parameters()
print(f"Users: {NUM_USERS}, Movies: {NUM_MOVIES}")

test_model = model.create_model(NUM_USERS, NUM_MOVIES)

print("\n=== Layers ===")
for i, layer in enumerate(test_model.layers):
    print(f"{i}: {layer.name} - {type(layer).__name__}")

print("\n=== Weights ===")
for i, weight in enumerate(test_model.get_weights()):
    print(f"{i}: shape {weight.shape}, type {type(weight)}")

print("\n=== Embeddings ===")
for layer in test_model.layers:
    if 'embedding' in layer.name.lower():
        print(f"Layer: {layer.name}")
        weights = layer.get_weights()
        if weights:
            print(f"  Weight shape: {weights[0].shape}")