import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import os
from natsort import natsorted
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from optim_utils import set_random_seed
import torchvision.transforms.v2.functional as F
import open_clip
from tqdm import tqdm
import argparse

# Setup all adversarial attacked images here.
# Three pixel-level different attacks - Gaussian Noise, Gaussian Blur, and JPEG Compression. 
# Three geometric attacks (Also pixel-level) - Cropping, Rotation_warping, and Aspect Ratio Distortion
# We will setup Semantic attacks in a bit.


semantic_edit_attacks = {
    "object_swap": "Modify the main subject or the center of the image into a chrome robot.",
    "bg_shift": "Change the background to a minimalist, stark white studio",
    "style_art": "Transform this into a cyberpunk glitch art style",
    "atmos_shift": "Add a heavy, thick fog and a blinding snowstorm."
}

# valid_commands = ["gaussian_blur", "gaussian_noise", "jpeg", "crop", "rotate", "distort", "brightness", "gamma", "equalize", "semantic", "pez"]
valid_commands = ["semantic", "pez"]
# valid_commands = ["gaussian_blur", "gaussian_noise", "jpeg", "crop", "rotate", "distort", "brightness", "gamma", "equalize"]


def optimize_pez(target_embedding, model, device, num_tokens=32, iterations=3000, weight_decay = 0.1, lr=1e-1):
    model.to(device)

    if num_tokens > 75:
        num_tokens = 75

    embedding_layer = model.token_embedding
    weight_mat = embedding_layer.weight

    mask = torch.zeros(model.vocab_size).to(device)
    for idx in range(model.vocab_size):
        try:
            text = open_clip.decode(torch.tensor([idx])).strip()
            if text.isalnum() and len(text) > 2:
                mask[idx] = 1
        except:
            continue

    init_tokens = torch.randperm(model.vocab_size)[:num_tokens].unsqueeze(0).to(device)
    soft_embeddings = nn.Parameter(embedding_layer(init_tokens).clone())
    optimizer = torch.optim.Adam([soft_embeddings], lr=lr, weight_decay=weight_decay)    

    for i in range(iterations):
        with torch.no_grad():
            logits = soft_embeddings @ weight_mat.T 
            logits = logits + (mask.log() * 1e6) 
            
            final_indices = torch.zeros((1, num_tokens), dtype=torch.long, device=device)
            temp_logits = logits.clone()
            for t in range(num_tokens):
                idx = temp_logits[0, t].argmax()
                final_indices[0, t] = idx
                temp_logits[0, :, idx] = -1e9 
                
            hard_embeddings = embedding_layer(final_indices)          

        optimized_embeds = soft_embeddings + (hard_embeddings - soft_embeddings).detach()
        
        pooled_embed = optimized_embeds.mean(dim=1)
        pooled_embed = pooled_embed / pooled_embed.norm(dim=-1, keepdim=True)

        token_sim = optimized_embeds[0] @ optimized_embeds[0].T
        diversity_loss = (token_sim.sum() - num_tokens) / (num_tokens**2) 
        
        loss = (1 - torch.sum(pooled_embed * target_embedding)) + (0.1 * diversity_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    raw_text = open_clip.decode(final_indices[0]) 

    return raw_text.replace("<start_of_text>", "").replace("<end_of_text>", "").strip()

def get_target_embedding(image, model, preprocess, device):

    img_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(img_input)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
    return image_features


def apply_pixel_level_attacks(attack_type, image, img_name, output_dir):
    if attack_type == "gaussian_blur":
        transform = v2.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 2.0))
    elif attack_type == "gaussian_noise":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.GaussianNoise(mean=0.0, sigma=0.2),
            v2.ToPILImage()
        ])        
    elif attack_type == "jpeg":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.JPEG(quality=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToPILImage()
        ])       
    elif attack_type == "crop":
        transform = v2.RandomResizedCrop(
            size=(512, 512), 
            scale=(0.5, 0.8),
            ratio=(1.0, 1.0),
            interpolation=v2.InterpolationMode.BILINEAR
        )             
    elif attack_type == "rotate":
        transform = v2.RandomRotation(
            degrees=15, 
            interpolation=v2.InterpolationMode.BILINEAR,
            expand=False
        )     
    elif attack_type == "distort":
        transform = v2.Compose([
            v2.Resize((512, 384)),
            v2.Resize((512, 512))  
        ])
    elif attack_type == "brightness":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(brightness=0.5),
            v2.ToPILImage()
        ])
    elif attack_type == "gamma":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: F.adjust_gamma(x, gamma=2.0)),
            v2.ToPILImage()
        ])
    elif attack_type == "equalize":
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomEqualize(p=1),
            v2.ToPILImage()
        ])

    transform_img = transform(image)
    transform_img.save(os.path.join(output_dir, f"{img_name}_{attack_type}.png"))        

def apply_semantic_attack(ip2p, image, edit_type, img_name, output_dir, index, pez_prompt=""):
        if(edit_type not in semantic_edit_attacks and edit_type != "pez"):
            print("Invalid edit type. Aborting...")
            return
        if edit_type == "background":
            igs = 1.15
            gs = 13.0
            infer_steps = 50
        elif edit_type == "atmos_shift":
            igs = 1.3
            gs = 11.0
            infer_steps = 50
        elif edit_type == "pez":
            igs = 1.05
            gs = 14.5
            infer_steps = 50
        else:
            igs = 1.25
            gs = 11.0
            infer_steps = 50

        seed = index
        set_random_seed(seed)

        if not pez_prompt:
            prompt = semantic_edit_attacks[edit_type]
        else:
            prompt = pez_prompt

        edited_image = ip2p(
            prompt=prompt, 
            num_images_per_prompt=1,
            image=image, 
            num_inference_steps=infer_steps, 
            image_guidance_scale=igs, 
            guidance_scale=gs
        ).images[0]

        edited_image.save(os.path.join(output_dir, f"{img_name}_{edit_type}.png"))

def setup_attacks(command, file_dir, output_dir, device, ip2p=None, edit_type=""):
    if command not in valid_commands:
        print("Invalid command. Please try again.")
        return

    if command == "pez":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        model.to(device)

    for i, file in enumerate(natsorted(os.listdir(file_dir))):
        filepath = os.path.join(file_dir, file)
        if not os.path.isfile(filepath):
            print("Not a file, skipping...")
            continue
        
        if i > 99:
            print(f"Attack generation stops at {i} images for time constraints.")
            break

        print(f"Applying {command} attack on {filepath}!")
        image = Image.open(filepath)
        if command != "semantic" and command != "pez":
            apply_pixel_level_attacks(command, image, os.path.splitext(file)[0], output_dir)
        elif command == "semantic" and ip2p is not None and edit_type is not None:
            apply_semantic_attack(ip2p, image, edit_type, os.path.splitext(file)[0], output_dir, i)
        elif command == "pez" and ip2p is not None:
            target_embedding = get_target_embedding(image, model, preprocess, device)
            pez_prompt = optimize_pez(target_embedding, model, device)
            apply_semantic_attack(ip2p, image, "pez", os.path.splitext(file)[0], output_dir, i, pez_prompt)
        else:
            print(f"Failed to apply {command} attacks! Please try again with proper commands.")
            return
    
    print(f"Finished applying {command} attacks!")

def main(args):
    if "semantic" in valid_commands or "pez" in valid_commands:
        device = 'mps' if torch.mps.is_available() else 'cpu'
        ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", torch_dtype=torch.float32
        )
        
        ip2p.to(device)

        ip2p.scheduler = EulerAncestralDiscreteScheduler.from_config(ip2p.scheduler.config)

        ip2p.safety_checker = None
        ip2p.feature_extractor = None
    else:
        device = None
        ip2p = None

    for command in valid_commands:

        if command != "semantic":
        
            # For future work, this should be refactor to account for different file paths.
            if args.wm_file_path == 'data/wm' and args.no_wm_file_path == 'data/no_wm':
                wm_att_dir = f"data/attack/{command}/wm"
                no_wm_att_dir = f"data/attack/{command}/no_wm"
            else:
                wm_att_dir = f"data/user_custom/attack/{command}/wm"
                no_wm_att_dir = f"data/user_custom/attack/{command}/no_wm"                

            os.makedirs(wm_att_dir, exist_ok=True)
            os.makedirs(no_wm_att_dir, exist_ok=True)

            setup_attacks(command, args.wm_file_path, wm_att_dir, device, ip2p)
            setup_attacks(command, args.no_wm_file_path, no_wm_att_dir, device, ip2p)
        else:
            for edit_type in semantic_edit_attacks:

                if args.wm_file_path == 'data/wm' and args.no_wm_file_path == 'data/no_wm':
                    wm_att_dir = f"data/attack/{command}/{edit_type}/wm"
                    no_wm_att_dir = f"data/attack/{command}/{edit_type}/no_wm"
                else:
                    wm_att_dir = f"data/user_custom/attack/{command}/{edit_type}/wm"
                    no_wm_att_dir = f"data/user_custom/attack/{command}/{edit_type}/no_wm"                    

                os.makedirs(wm_att_dir, exist_ok=True)
                os.makedirs(no_wm_att_dir, exist_ok=True)

                setup_attacks(command, args.wm_file_path, wm_att_dir, device, ip2p, edit_type)
                setup_attacks(command, args.no_wm_file_path, no_wm_att_dir, device, ip2p, edit_type)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TR_attack_creator')
    parser.add_argument('--wm_file_path', default='data/wm')
    parser.add_argument('--no_wm_file_path', default='data/no_wm')

    args = parser.parse_args()

    main(args)

