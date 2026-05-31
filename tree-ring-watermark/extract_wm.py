
import torch
from optim_utils import * # Import the official method
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import numpy as np
import argparse
import os
from PIL import Image
from natsort import natsorted
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
import matplotlib.pyplot as plt

semantic_edit_attacks = {
    "object_swap": "Modify the main subject or the center of the image into a chrome robot.",
    "bg_shift": "Change the background to a minimalist, stark white studio",
    "style_art": "Transform this into a cyberpunk glitch art style",
    "atmos_shift": "Add a heavy, thick fog and a blinding snowstorm."
}

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("mps")
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to("mps")
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to("mps")

def preprocess_for_metrics(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0).to("mps") 

def compute_fidelity(img_wm, img_edit):

    img_wm_tensor = preprocess_for_metrics(img_wm)
    img_edit_tensor = preprocess_for_metrics(img_edit)    

    score_psnr = psnr_metric(img_edit_tensor, img_wm_tensor).item()
    score_ssim = ssim_metric(img_edit_tensor, img_wm_tensor).item()

    if score_psnr == float('inf') or score_psnr > 100.0:
        score_psnr = 100.0

    img_wm_lpips =  (img_wm_tensor * 2) - 1
    img_edit_lpips = (img_edit_tensor * 2) - 1
    score_lpips = lpips_metric(img_edit_lpips, img_wm_lpips).item()
    
    return score_psnr, score_lpips, score_ssim

@torch.no_grad()
def get_clip_metrics(ref_model, ref_tokenizer, ref_preprocess, 
                     img_wm, img_edit, prompt_orig, prompt_edit):

    img_wm_input = ref_preprocess(img_wm).unsqueeze(0).to('mps')
    img_edit_input = ref_preprocess(img_edit).unsqueeze(0).to('mps')
    
    img_wm_feat = ref_model.encode_image(img_wm_input)
    img_edit_feat = ref_model.encode_image(img_edit_input)
    
    txt_orig_input = ref_tokenizer([prompt_orig]).to('mps')
    txt_edit_input = ref_tokenizer([prompt_edit]).to('mps')
    
    txt_orig_feat = ref_model.encode_text(txt_orig_input)
    txt_edit_feat = ref_model.encode_text(txt_edit_input)
    
    img_wm_feat /= img_wm_feat.norm(dim=-1, keepdim=True)
    img_edit_feat /= img_edit_feat.norm(dim=-1, keepdim=True)
    txt_orig_feat /= txt_orig_feat.norm(dim=-1, keepdim=True)
    txt_edit_feat /= txt_edit_feat.norm(dim=-1, keepdim=True)
    
    attack_clip = (img_edit_feat @ txt_edit_feat.T).item()
    
    img_dir = img_edit_feat - img_wm_feat
    txt_dir = txt_edit_feat - txt_orig_feat
    dir_sim = F.cosine_similarity(img_dir, txt_dir, dim=-1).item()
    
    return attack_clip, dir_sim

# Code is mostly borrowed from Tree-Ring-Watermark. I only modified the code to remove the generation part, so I don't have regenerate all the images again.
def main(args):
    #If no watermark or non-watermark directories are specfied, then don't anything.
    if args.wm_dir is None or args.no_wm_dir is None:
        print("Please specfic a watermark and a non-watermark directory with images on each directory.")
        return

    num_images = len([entry for entry in os.listdir(args.wm_dir) if os.path.isfile(os.path.join(args.wm_dir, entry))])
    num_no_wm_imgs = len([entry for entry in os.listdir(args.no_wm_dir) if os.path.isfile(os.path.join(args.no_wm_dir, entry))])

    if(num_images != num_no_wm_imgs):
        print("Number of watermarked and non-watermarked images are not equal. Check to see if you are missing any images in either directory.")
        return

    os.makedirs(args.output_path, exist_ok=True)

    device = 'mps' if torch.mps.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        )  
    
    pipe.to(device) 

    if args.dataset is not None:
        dataset, prompt_key = get_dataset(args)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)    

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    results = []

    clip_scores = []
    clip_scores_w = []

    no_w_metrics = []
    w_metrics = []
    
    no_w_p_values = []
    w_p_values = []

    no_w_psnr_scores = []
    w_psnr_scores = []    

    no_w_lpips_scores = []
    w_lpips_scores = []

    no_w_ssim_scores = []
    w_ssim_scores = []

    no_w_attack_clip_scores = []
    no_w_dir_sim_scores = []

    w_attack_clip_scores = []
    w_dir_sim_scores = []

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)    

    for i, file in tqdm(enumerate(natsorted(os.listdir(args.wm_dir)))):

        # if i > 9:
        if i > 99:
            print("Testing over. Computing final score.")
            break

        seed = i + args.gen_seed

        if args.dataset is not None:
            current_prompt = dataset[i][prompt_key]
        else:
            txt = open(os.path.join(args.prompt_dir, f"prompt_{i}.txt")) 
            current_prompt = txt.read()

        set_random_seed(seed)
        init_latents_w = pipe.get_random_latents()

        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        no_wm_image = Image.open(os.path.join(args.no_wm_dir, file))

        img_no_w = transform_img(no_wm_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        wm_image = Image.open(os.path.join(args.wm_dir, file))

        # reverse img with watermarking
        img_w = transform_img(wm_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args) 
        no_w_p_value, w_p_value = get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args) 

        if args.wm_ground_dir is not None and args.no_wm_ground_dir is not None:

            gt_file = f"img_{i}.png"

            no_wm_image_ground = Image.open(os.path.join(args.no_wm_ground_dir, gt_file))
            wm_image_ground = Image.open(os.path.join(args.wm_ground_dir, gt_file))

            no_w_psnr_score, no_w_lpips_score, no_w_ssim_score = compute_fidelity(no_wm_image_ground, no_wm_image)
            w_psnr_score, w_lpips_score, w_ssim_score = compute_fidelity(wm_image_ground, wm_image)


            if args.attack_type in semantic_edit_attacks or args.attack_type == "pez":
                no_w_attack_clip_score, no_w_dir_sim_score = get_clip_metrics(
                        ref_model, ref_tokenizer, ref_clip_preprocess,
                        no_wm_image_ground, no_wm_image, current_prompt, args.attack_type
                    )        

                w_attack_clip_score, w_dir_sim_score = get_clip_metrics(
                        ref_model, ref_tokenizer, ref_clip_preprocess,
                        wm_image_ground, wm_image, current_prompt, args.attack_type
                    )  


        if args.reference_model is not None:
            sims = measure_similarity([no_wm_image, wm_image], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })   

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        no_w_p_values.append(no_w_p_value)          
        w_p_values.append(w_p_value)

        if args.wm_ground_dir is not None and args.no_wm_ground_dir is not None:
            no_w_psnr_scores.append(no_w_psnr_score)         
            w_psnr_scores.append(w_psnr_score)        

            no_w_lpips_scores.append(no_w_lpips_score)         
            w_lpips_scores.append(w_lpips_score)

            no_w_ssim_scores.append(no_w_ssim_score)         
            w_ssim_scores.append(w_ssim_score)  

            if args.attack_type in semantic_edit_attacks or args.attack_type == "pez":
                no_w_attack_clip_scores.append(no_w_attack_clip_score)         
                w_attack_clip_scores.append(w_attack_clip_score)   

                no_w_dir_sim_scores.append(no_w_dir_sim_score)         
                w_dir_sim_scores.append(w_dir_sim_score)             

        if args.with_tracking:
            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)        

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    print(f'clip_score_mean: {mean(clip_scores):.4f}')
    print(f'w_clip_score_mean: {mean(clip_scores_w):.4f}')

    # print(f'p_values_mean: {mean(no_w_p_values):.4f}')
    # print(f'w_p_values_mean: {mean(w_p_values):.4f}')

    # if args.wm_ground_dir is not None and args.no_wm_ground_dir is not None:

    #     print(f'no_w_psnr_scores_mean: {mean(no_w_psnr_scores):.4f}')
    #     print(f'w_psnr_scores_mean: {mean(w_psnr_scores):.4f}')

    #     print(f'ssim_mean: {mean(no_w_ssim_scores):.4f}')
    #     print(f'w_ssim_mean: {mean(w_ssim_scores):.4f}')      

    #     print(f'lpips_scores_mean: {mean(no_w_lpips_scores):.4f}')
    #     print(f'w_lpips_scores_mean: {mean(w_lpips_scores):.4f}')

    #     if args.attack_type in semantic_edit_attacks:
    #         print(f'no_w_attack_clip_scores_mean: {mean(no_w_attack_clip_scores):.4f}')
    #         print(f'w_attack_clip_scores_mean: {mean(w_attack_clip_scores):.4f}')
    #         print(f'no_w_dir_sim_scores_mean: {mean(no_w_dir_sim_scores):.4f}')
    #         print(f'w_dir_sim_scores_mean: {mean(w_dir_sim_scores):.4f}')           

    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')

    names = {
        'jpeg_ratio': "Jpeg.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'resize_ratio': "Resize.txt",
        'random_rotate_ratio': "RandomRotate.txt",
        'brightness_factor': "Brightness.txt",
        'gamma': "Gamma.txt",
        'equalize': "Equalize.txt",
        'object_swap': "SemanticObjectSwap.txt",
        'bg_shift': "SemanticBGShift.txt",
        'style_art': "SemanticStyle.txt",
        'atmos_shift': "SemanticAtmosShift.txt",
        'pez': "Pez.txt"
    }
    filename = "Identity.txt"
    # print(f"Attack_Type: {args.attack_type}")
    if args.attack_type in names:
        filename = names[args.attack_type]


    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axvline(x=0.01, color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if args.attack_type in names:
        plt.title(f'{args.attack_type} ROC Curve with TPR @ 1% FPR')
    else:
        plt.title(f'Identity ROC Curve with TPR @ 1% FPR') 
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if args.attack_type in names:
        plt.savefig(os.path.join(args.output_path, f"plot_{args.attack_type}.png")) 
    else:
        plt.savefig(os.path.join(args.output_path, f"plot_identity.png"))          

    if args.reference_model is not None and (args.attack_type in semantic_edit_attacks or args.attack_type == "pez") and args.wm_ground_dir is not None and args.no_wm_ground_dir is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       'mean_w_clip_score:' + f"{mean(clip_scores_w):.4f}" + '      ' + 'std_w_clip_score:' + f"{stdev(clip_scores_w):.4f}" + '      ' +
                       'mean_p_value:' + f"{mean(no_w_p_values):.4f}" + '      ' + 'std_p_value:' + f"{stdev(no_w_p_values):.4f}" + '      ' +
                       'mean_w_p_value:' + f"{mean(w_p_values):.4f}" + '      ' + 'std_w_p_value:' + f"{stdev(w_p_values):.4f}" + '      ' +
                       'mean_psnr_scores:' + f"{mean(no_w_psnr_scores):.4f}" + '      ' + 'std_psnr_scores:' + f"{stdev(no_w_psnr_scores):.4f}" + '      ' +
                       'mean_w_psnr_scores:' + f"{mean(w_psnr_scores):.4f}" + '      ' + 'std_w_psnr_scores:' + f"{stdev(w_psnr_scores):.4f}" + '      ' +
                       'mean_ssim_scores:' + f"{mean(no_w_ssim_scores):.4f}" + '      ' + 'std_ssim_scores:' + f"{stdev(no_w_ssim_scores):.4f}" + '      ' +
                       'mean_w_ssim_scores:' + f"{mean(w_ssim_scores):.4f}" + '      ' + 'std_w_ssim_scores:' + f"{stdev(w_ssim_scores):.4f}" + '      ' +
                       'mean_lpips_scores:' + f"{mean(no_w_lpips_scores):.4f}" + '      ' + 'std_lpips_scores:' + f"{stdev(no_w_lpips_scores):.4f}" + '      ' +
                       'mean_w_lpips_scores:' + f"{mean(w_lpips_scores):.4f}" + '      ' + 'std_w_lpips_scores:' + f"{stdev(w_lpips_scores):.4f}" + '      ' +
                       'mean_attack_clip_scores:' + f"{mean(no_w_attack_clip_scores):.4f}" + '      ' + 'std_attack_clip_scores:' + f"{stdev(no_w_attack_clip_scores):.4f}" + '      ' +
                       'mean_w_attack_clip_scores:' + f"{mean(w_attack_clip_scores):.4f}" + '      ' + 'std_w_attack_clip_scores:' + f"{stdev(w_attack_clip_scores):.4f}" + '      ' +
                       'mean_no_w_dir_sim_scores:' + f"{mean(no_w_dir_sim_scores):.4f}" + '      ' + 'std_no_w_dir_sim_scores:' + f"{stdev(no_w_dir_sim_scores):.4f}" + '      ' +
                       'mean_w_dir_sim_scores:' + f"{mean(w_dir_sim_scores):.4f}" + '      ' + 'std_w_dir_sim_scores:' + f"{stdev(w_dir_sim_scores):.4f}" + '      ' +
                       '\n')
    elif args.reference_model is not None and args.wm_ground_dir is not None and args.no_wm_ground_dir is not None: 
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       'mean_w_clip_score:' + f"{mean(clip_scores_w):.4f}" + '      ' + 'std_w_clip_score:' + f"{stdev(clip_scores_w):.4f}" + '      ' +
                       'mean_p_value:' + f"{mean(no_w_p_values):.4f}" + '      ' + 'std_p_value:' + f"{stdev(no_w_p_values):.4f}" + '      ' +
                       'mean_w_p_value:' + f"{mean(w_p_values):.4f}" + '      ' + 'std_w_p_value:' + f"{stdev(w_p_values):.4f}" + '      ' +
                       'mean_psnr_scores:' + f"{mean(no_w_psnr_scores):.4f}" + '      ' + 'std_psnr_scores:' + f"{stdev(no_w_psnr_scores):.4f}" + '      ' +
                       'mean_w_psnr_scores:' + f"{mean(w_psnr_scores):.4f}" + '      ' + 'std_w_psnr_scores:' + f"{stdev(w_psnr_scores):.4f}" + '      ' +
                       'mean_ssim_scores:' + f"{mean(no_w_ssim_scores):.4f}" + '      ' + 'std_ssim_scores:' + f"{stdev(no_w_ssim_scores):.4f}" + '      ' +
                       'mean_w_ssim_scores:' + f"{mean(w_ssim_scores):.4f}" + '      ' + 'std_w_ssim_scores:' + f"{stdev(w_ssim_scores):.4f}" + '      ' +
                       'mean_lpips_scores:' + f"{mean(no_w_lpips_scores):.4f}" + '      ' + 'std_lpips_scores:' + f"{stdev(no_w_lpips_scores):.4f}" + '      ' +
                       'mean_w_lpips_scores:' + f"{mean(w_lpips_scores):.4f}" + '      ' + 'std_w_lpips_scores:' + f"{stdev(w_lpips_scores):.4f}" + '      ' +
                       '\n')
    elif args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       'mean_w_clip_score:' + f"{mean(clip_scores_w):.4f}" + '      ' + 'std_w_clip_score:' + f"{stdev(clip_scores_w):.4f}" + '      ' +
                       'mean_p_value:' + f"{mean(no_w_p_values):.4f}" + '      ' + 'std_p_value:' + f"{stdev(no_w_p_values):.4f}" + '      ' +
                       'mean_w_p_value:' + f"{mean(w_p_values):.4f}" + '      ' + 'std_w_p_value:' + f"{stdev(w_p_values):.4f}" + '      ' +
                       '\n')
    else:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                        'mean_p_value:' + f"{mean(no_w_p_values):.4f}" + '      ' + 'std_p_value:' + f"{stdev(no_w_p_values):.4f}" + '      ' +
                       'mean_w_p_value:' + f"{mean(w_p_values):.4f}" + '      ' + 'std_w_p_value:' + f"{stdev(w_p_values):.4f}" + '      ' +
                       '\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark_extractor')
    parser.add_argument('--run_name', default='test')
    # parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--dataset', default=None)
    # parser.add_argument('--start', default=0, type=int)
    # parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    # parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    parser.add_argument('--attack_type', default=None)
    parser.add_argument('--wm_dir', default=None)
    parser.add_argument('--no_wm_dir', default=None)
    parser.add_argument('--wm_ground_dir', default=None)
    parser.add_argument('--no_wm_ground_dir', default=None)
    parser.add_argument('--prompt_dir', default=None)
    parser.add_argument('--output_path', default='./output/')
    
    # for image distortion
    # parser.add_argument('--r_degree', default=None, type=float)
    # parser.add_argument('--jpeg_ratio', default=None, type=int)
    # parser.add_argument('--crop_scale', default=None, type=float)
    # parser.add_argument('--crop_ratio', default=None, type=float)
    # parser.add_argument('--gaussian_blur_r', default=None, type=int)
    # parser.add_argument('--gaussian_std', default=None, type=float)
    # parser.add_argument('--brightness_factor', default=None, type=float)
    # parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)        