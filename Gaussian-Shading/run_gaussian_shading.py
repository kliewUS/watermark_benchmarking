import transformers
import diffusers
import argparse
import copy
import wandb
from tqdm import tqdm
# import torch
# from transformers import CLIPModel, CLIPTokenizer


if not hasattr(transformers, 'CLIPFeatureExtractor'):
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor


if 'CLIPFeatureExtractor' not in dir(transformers):
    setattr(transformers, 'CLIPFeatureExtractor', transformers.CLIPImageProcessor)

def manual_numpy_to_pil(self, images):
    """
    Injecting the missing conversion logic required by the 
    ModifiedStableDiffusionPipeline safety checker.
    """
    if images.ndim == 3:
        images = images[None, ...]
    # Standard Stable Diffusion normalization (0-1 float to 0-255 uint8)
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]

transformers.CLIPImageProcessor.numpy_to_pil = manual_numpy_to_pil

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *



def main(args):

    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_gaussian_shading_watermark', name='Gaussian_Shading_Exp', tags=['gaussian_shading_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_w', 'w_clip_score', 'prompt', 'w_acc_metric'])
    
    device = 'mps' if torch.mps.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            # torch_dtype=torch.float16,
            torch_dtype=torch.float32,
            # revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Scores
    clip_scores = []

    #test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]


        # set_random_seed(seed)
        # init_latents_no_w = pipe.get_random_latents()
        # outputs_no_w = pipe(
        #     current_prompt,
        #     num_images_per_prompt=args.num_images,
        #     guidance_scale=args.guidance_scale,
        #     num_inference_steps=args.num_inference_steps,
        #     height=args.image_length,
        #     width=args.image_length,
        #     latents=init_latents_no_w,
        #     )
        # image_no_w = outputs_no_w.images[0]        

        #generate with watermark
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]

        # distortion
        # image_no_w_distortion = image_distortion(image_no_w, seed, args)
        image_w_distortion = image_distortion(image_w, seed, args)

        # img_no_w = transform_img(image_no_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        # image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        # reversed_latents_no_w = pipe.forward_diffusion(
        #     latents=image_latents_no_w,
        #     text_embeddings=text_embeddings,
        #     guidance_scale=1,
        #     num_inference_steps=args.test_num_inference_steps,
        # )        

        # reverse img
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        #acc metric
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        acc.append(acc_metric)

        #CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        table.add_data(wandb.Image(image_w), clip_socre, current_prompt, acc_metric)
        clip_scores.append(clip_socre)

    #tpr metric
    tpr_detection, tpr_traceability = watermark.get_tpr()

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'w_clip_score_mean': mean(clip_scores), 'w_clip_score_std': stdev(clip_scores),
                   'acc_mean':mean(acc), 'acc_std': stdev(acc), 
                   'tpr_traceability': (tpr_traceability/args.num), 'tpr_detection': (tpr_detection/args.num)})

    #save metrics
    # save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')

    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)


    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
