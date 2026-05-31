
import torch# Import the official method
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
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
from pytorch_fid.fid_score import *
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

    if args.wm_dir is None or args.no_wm_dir is None:
        print("Please specfic a watermark and a non-watermark directory with images on each directory.")
        return    

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
    if args.dataset_path is not None:
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
    no_w_acc = []
    #CLIP Scores
    clip_scores = []

    psnr_scores = []
    lpips_scores = []
    ssim_scores = []

    attack_clip_scores = []
    dir_sim_scores = []


    #test
    for i, file in tqdm(enumerate(natsorted(os.listdir(args.wm_dir)))):

        # if i > 9:
        if i > 99:
            print("Testing over. Computing final score.")
            break

        seed = i + args.gen_seed

        if args.dataset_path is not None:
            current_prompt = dataset[i][prompt_key]
        else:
            txt = open(os.path.join(args.prompt_dir, f"prompt_{i}.txt")) 
            current_prompt = txt.read()
   
        #generate with watermark
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()

        no_wm_filename = f"img_{i}.png"

        no_wm_image = Image.open(os.path.join(args.no_wm_dir, no_wm_filename))

        img_no_w = transform_img(no_wm_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )        

        image_w = Image.open(os.path.join(args.wm_dir, file))

        # distortion
        image_w_distortion = image_distortion(image_w, seed, args)

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

        no_w_acc_metric = watermark.eval_watermark(reversed_latents_no_w)
        no_w_acc.append(no_w_acc_metric)

        if args.wm_ground_dir is not None:

            gt_file = f"img_{i}.png"

            wm_image_ground = Image.open(os.path.join(args.wm_ground_dir, gt_file))

            psnr_score, lpips_score, ssim_score = compute_fidelity(wm_image_ground, image_w)

            if args.attack_type in semantic_edit_attacks or args.attack_type == "pez":
                attack_clip_score, dir_sim_score = get_clip_metrics(
                        ref_model, ref_tokenizer, ref_clip_preprocess,
                        wm_image_ground, image_w, current_prompt, args.attack_type
                    )  


        #CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        
        clip_scores.append(clip_socre)

        if args.wm_ground_dir is not None:
            psnr_scores.append(psnr_score)                
            lpips_scores.append(lpips_score)         
            ssim_scores.append(ssim_score)         

            if args.attack_type in semantic_edit_attacks or args.attack_type == "pez":
                attack_clip_scores.append(attack_clip_score)         
                dir_sim_scores.append(dir_sim_score)             

    #tpr metric
    # tpr_detection, tpr_traceability = watermark.get_tpr()

    # roc
    preds = no_w_acc +  acc
    t_labels = [0] * len(no_w_acc) + [1] * len(acc)
    # print(f"Accuracy/BER scores: {acc}")
    # print(f"No_W Accuracy/BER scores: {no_w_acc}")

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    bal_acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]   

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

    if args.reference_model is not None and (args.attack_type in semantic_edit_attacks or args.attack_type == "pez") and args.wm_ground_dir is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(bal_acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       'mean_psnr_scores:' + f"{mean(psnr_scores):.4f}" + '      ' + 'std_psnr_scores:' + f"{stdev(psnr_scores):.4f}" + '      ' +
                       'mean_ssim_scores:' + f"{mean(ssim_scores):.4f}" + '      ' + 'std_ssim_scores:' + f"{stdev(ssim_scores):.4f}" + '      ' +
                       'mean_lpips_scores:' + f"{mean(lpips_scores):.4f}" + '      ' + 'std_lpips_scores:' + f"{stdev(lpips_scores):.4f}" + '      ' +
                       'mean_attack_clip_scores:' + f"{mean(attack_clip_scores):.4f}" + '      ' + 'std_attack_clip_scores:' + f"{stdev(attack_clip_scores):.4f}" + '      ' +
                       'mean_dir_sim_scores:' + f"{mean(dir_sim_scores):.4f}" + '      ' + 'std_dir_sim_scores:' + f"{stdev(dir_sim_scores):.4f}" + '      ' +
                       '\n')
    elif args.reference_model is not None and args.wm_ground_dir is not None: 
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(bal_acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       'mean_psnr_scores:' + f"{mean(psnr_scores):.4f}" + '      ' + 'std_psnr_scores:' + f"{stdev(psnr_scores):.4f}" + '      ' +
                       'mean_ssim_scores:' + f"{mean(ssim_scores):.4f}" + '      ' + 'std_ssim_scores:' + f"{stdev(ssim_scores):.4f}" + '      ' +
                       'mean_lpips_scores:' + f"{mean(lpips_scores):.4f}" + '      ' + 'std_lpips_scores:' + f"{stdev(lpips_scores):.4f}" + '      ' +
                       '\n')
    elif args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(bal_acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       'mean_clip_score:' + f"{mean(clip_scores):.4f}" + '      ' + 'std_clip_score:' + f"{stdev(clip_scores):.4f}" + '      ' +
                       '\n')
    else:
        with open(args.output_path + filename, "a") as file:
            file.write('auc:' + str(auc) + '      ' +
                       'bal_acc:' + str(bal_acc) + '      ' +
                       f'TPR@1%FPR:' + str(low) + '      ' +
                       '\n')

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
    # parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')

    parser.add_argument('--attack_type', default=None)
    parser.add_argument('--wm_dir', default=None)
    parser.add_argument('--no_wm_dir', default=None)
    parser.add_argument('--wm_ground_dir', default=None)
    parser.add_argument('--prompt_dir', default=None)

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