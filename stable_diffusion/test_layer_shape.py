from model_loader import load_from_standard_weights
from encoder import VAE_Encoder
from decoder import VAE_decoder
from diffusion import Diffusion
from clip import CLIP

device = 'cpu'
state_dict = load_from_standard_weights(r"D:\Data Science stuffs\Stable Diffusion from scratch\data\v1-5-pruned-emaonly.ckpt", device)

# # View loaded model's encoder layers size
# print("Model's encoder architecture:")
# for name, param in state_dict['encoder'].items():
#     print(name, param.shape)
# print("="*30)
# # View our model encoder layers size:
# print("Our encoder layers' size:")
# encoder = VAE_Encoder().to(device)
# for name, param in encoder.state_dict().items():
#     print(name, param.shape)

# # Test Encoder shape
# print(f"{'Loaded Model Name':<30} {'Loaded Shape':<30} {'Our Model Name':<30} {'Our Shape':<30}")
# print("=" * 120)

# # Compare layers
# loaded_layers = state_dict['encoder']
# encoder = VAE_Encoder().to(device)
# our_layers = encoder.state_dict()

# for (loaded_name, loaded_param), (our_name, our_param) in zip(loaded_layers.items(), our_layers.items()):
#     print(f"{loaded_name:<30} {str(loaded_param.shape):<30} {our_name:<30} {str(our_param.shape):<30}")

# # Test Decoder shape
# print(f"{'Loaded Model Name':<30} {'Loaded Shape':<30} {'Our Model Name':<30} {'Our Shape':<30}")
# print("=" * 120)

# # Compare layers
# loaded_layers = state_dict['decoder']
# decoder  = VAE_decoder().to(device)
# our_layers = decoder.state_dict()

# # Sort layers by loaded model name
# sorted_loaded_layers = sorted(state_dict['decoder'].items(), key=lambda x: x[0])
# sorted_our_layers = sorted(decoder.state_dict().items(), key=lambda x: x[0])

# for (loaded_name, loaded_param), (our_name, our_param) in zip(sorted_loaded_layers, sorted_our_layers):
#     print(f"{loaded_name:<30} {str(loaded_param.shape):<30} {our_name:<30} {str(our_param.shape):<30}")

# # Test Diffusion shape
# print(f"{'Loaded Model Name':<30} {'Loaded Shape':<30} {'Our Model Name':<30} {'Our Shape':<30}")
# print("=" * 120)

# # Compare layers
# loaded_layers = state_dict['diffusion']
# diffusion  = Diffusion().to(device)
# our_layers = diffusion.state_dict()

# # Sort layers by loaded model name
# sorted_loaded_layers = sorted(state_dict['diffusion'].items(), key=lambda x: x[0])
# sorted_our_layers = sorted(diffusion.state_dict().items(), key=lambda x: x[0])

# for (loaded_name, loaded_param), (our_name, our_param) in zip(sorted_loaded_layers, sorted_our_layers):
#     print(f"{loaded_name:<30} {str(loaded_param.shape):<30} {our_name:<30} {str(our_param.shape):<30}")

# Test CLIP shape
print(f"{'Loaded Model Name':<30} {'Loaded Shape':<30} {'Our Model Name':<30} {'Our Shape':<30}")
print("=" * 120)

# Compare layers
loaded_layers = state_dict['clip']
clip  = CLIP().to(device)
our_layers = clip.state_dict()

# Sort layers by loaded model name
sorted_loaded_layers = sorted(state_dict['clip'].items(), key=lambda x: x[0])
sorted_our_layers = sorted(clip.state_dict().items(), key=lambda x: x[0])

for (loaded_name, loaded_param), (our_name, our_param) in zip(sorted_loaded_layers, sorted_our_layers):
    print(f"{loaded_name:<30} {str(loaded_param.shape):<30} {our_name:<30} {str(our_param.shape):<30}")