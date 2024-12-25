# import torch
# import numpy as np
# from tqdm import tqdm
# from ddpm import DDPMSampler

# WIDTH = 512
# HEIGHT = 512
# LATENTS_WIDTH = WIDTH // 8
# LATENTS_HEIGHT = HEIGHT // 8

# def generate(prompt: str, 
#              uncond_prompt: str, 
#              input_image=None, 
#              strength = 0.8, 
#              do_cfg=True, 
#              cfg_scale=7.5, 
#              sampler_name="ddpm", 
#              n_inference_steps=50, 
#              models={}, 
#              seed=None,
#              device=None,
#              idle_device=None,
#              tokenizer=None
#             ):
#     """_summary_

#     Args:
#         prompt (str): Input prompt
#         uncond_prompt (str): Negative prompt to except unwanted elements
#         input_image (_type_, optional): In case we want it to be img2img. Defaults to None.
#         strength (float, optional): How much attention we want it to focus on input image. Defaults to 0.8.
#         do_cfg (bool, optional): Classifier free guidance. Defaults to True.
#         cfg_scale (float, optional): How much focus on the prompt we want w.r.t unconditioned prompt, ranging from 1-40 . Defaults to 7.5.
#         sampler_name (str, optional): _description_. Defaults to "ddpm".
#         n_inference_steps (int, optional): Number of inference steps. Defaults to 50.
#         models (dict, optional): _description_. Defaults to {}.
#         seed (_type_, optional): _description_. Defaults to None.
#         device (_type_, optional): Device CUDA or CPU. Defaults to None.
#         idle_device (_type_, optional): Move to idle device if don't need. Defaults to None.
#         tokenizer (_type_, optional): Tokenizer. Defaults to None.
#     """
#     with torch.inference_mode():
#         if not (0 < strength <= 1):
#             raise ValueError("Strength must be between 0 and 1")
        
#         if idle_device:
#             to_idle = lambda x: x.to(idle_device)
#         else:
#             to_idle = lambda x: x
        
#         generator = torch.Generator(device=device)
#         if seed is None:
#             generator.seed()
#         else:
#             generator.manual_seed(seed)

#         clip = models["clip"]
#         clip.to(device)

#         if do_cfg:
#             # Convert the prompt into tokens using the tokenizer
#             cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
#             # B, seq_len
#             cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
#             # B, seq_len -> B, seq_len, dim
#             cond_context = clip(cond_tokens)

#             uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
#             uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
#             # B, seq_len -> B, seq_len, dim
#             uncond_context = clip(uncond_tokens)

#             # (2, seq_len, dim) = (2, 77, 768)
#             context = torch.cat([cond_context, uncond_context])
#         else:
#             # Convert it into a list of tokens
#             tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
#             tokens = torch.tensor(tokens, dtype=torch.long, device=device)
#             context = clip(tokens)
#         to_idle(clip)

#         if sampler_name == "ddpm":
#             sampler = DDPMSampler(generator)
#             sampler.set_inference_timesteps(n_inference_steps)
#         else: 
#             raise ValueError(f"Unknown Sampler {sampler_name}")
        
#         latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

#         if input_image:
#             encoder = models["encoder"]
#             encoder.to(device)

#             input_image_tensor = input_image.resize((WIDTH, HEIGHT))
#             input_image_tensor = np.array(input_image_tensor)
#             # H, W, C
#             input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
#             input_image_tensor = rescale(input_image_tensor, dtype=torch.float32)
#             # H, W, C -> B, H, W, C
#             input_image_tensor = input_image_tensor.unsqueeze(0)
#             # B, H, W, C -> B, C, H, W
#             input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

#             encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
#             # run the image through the encoder of the VAE
#             latents = encoder(input_image_tensor, encoder_noise)

#             sampler.set_strength(strength=strength)
#             latents = sampler.add_noise(latents, sampler.timesteps[0])

#             to_idle(encoder)
#         else:
#             # If we are doing text2img, start with random noise N(0, 1)
#             latents = torch.randn(latents_shape, generator=generator, device=device)

#         # 999 ... 0
#         # 1000 980 960 940 920 ... 0
#         diffusion = models["diffusion"]
#         diffusion.to(device)

#         timesteps = tqdm(sampler.timesteps)
#         for i, timestep in enumerate(timesteps):
#             # (1, 320)
#             time_embedding = get_time_embedding(timestep).to(device)

#             # B, 4, LH, LW
#             model_input = latents

#             if do_cfg:
#                 # B, 4, LH, LW -> 2*B, 4, LH, LW
#                 model_input = model_input.repeat(2, 1, 1, 1) # Repeating to use the Batch size twice, one with prompt and one without
#             # model_output is the predicted noise by the UNET
#             model_output = diffusion(model_input, context, time_embedding)

#             if do_cfg:
#                 output_cond, output_uncond = model_output.chunk(2)
#                 model_output = cfg_scale * (output_cond-output_uncond) + output_uncond
            
#             # Remove noise predicted by the UNET
#             latents = sampler.step(timestep, latents, model_output)

#             to_idle(diffusion)

#             decoder = models["decoder"]
#             decoder.to(device)

#             images = decoder(latents)
#             to_idle(decoder)

#             images = rescale(images, (-1, 1), (0, 255), clamp=True)
#             # B, C, H, W -> B, H, W, C
#             images = images.permute(0, 2, 3, 1)
#             images = images.to("cpu", torch.uint8).numpy()
#             return images[0]
        
# def rescale(x, old_range, new_range, clamp=False):
#     old_min, old_max = old_range
#     new_min, new_max = new_range
#     x -= old_min
#     x *= (new_max-new_min) / (old_max-old_min)
#     x += new_min
#     if clamp: 
#         x = x.clamp(new_min, new_max)
#     return x

# def get_time_embedding(timestep):
#     # Math: PE(pos, 2i) = sin\frac{pos}{10000^{\frac{2i}{d_{model}}}}
#     # Math: PE(pos, 2i+1) = cos\frac{pos}{10000^{\frac{2i}{d_{model}}}}
#     # (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
#     # (1, 1)@(1, 160) = (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] # freqs[None]: (160,) -> (1, 160)
#     # (1, 320)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


### TRY CODE ###
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)