import os
import sys
import pickle
sys.path.append('/home/dahee/T2I') 
sys.path.append('/home/dahee/T2I/ovam/')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import torch
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForText2Image
from ovam_turbo.stable_diffusion import StableDiffusionHooker
from ovam_turbo.utils import set_seed
import accelerate
import numpy as np
from matplotlib import gridspec
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.nn import PairwiseDistance
import argparse

torch.cuda.empty_cache()

# seed, prompt, mask_attribute, nth att, name
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=None, type=int, help="Random Seed")
    parser.add_argument("--device", type=int, default=1, help="GPU num")

    parser.add_argument("--prompt", default="A red apple and a green apple.", type=str, help="image generating prompt")
    parser.add_argument("--att_prompt", default="A red apple and a green apple.", type= str, help="Attribute descriptions")
    parser.add_argument("--att_idx", default=2,type=int, help="Which attribute you want to extract among att_prompt.")
    parser.add_argument("--thres", default=0.9,type=float, help="Threshold for binarize mask.")
    parser.add_argument("--n_mask_ratio", type=float, default=0.05, help="channel saving name")
    parser.add_argument("--step", type=int, default=30, help="evaluating specific timestep")
    parser.add_argument("--n_samples", type=int, default=1, help="number of samples")
    
    return parser.parse_args()

# Function to normalize each [b, b] matrix in the tensor
def normalize_tensor(tensor):
    if len(tensor.shape) == 2:
        a,b = tensor.shape
    if len(tensor.shape) == 3:
        a, b, _ = tensor.shape
    normalized_tensor = np.zeros_like(tensor)
    
    for i in range(a):
        matrix = tensor[i]
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        range_val = max_val - min_val
        
        if range_val > 0:  # To avoid division by zero
            normalized_matrix = (matrix - min_val) / range_val
        else:
            normalized_matrix = matrix - min_val  # All values are the same
        
        normalized_tensor[i] = normalized_matrix
        
    return normalized_tensor

def sigmoid(z,k=1):
    return 1/(1 + np.exp(-k*z))

def find_ranges(ranges, queries):
    results = []
    channel_num = [[] for _ in range(len(ranges))]
    pair_results = []
    
    for n,query in enumerate(queries):
        # If the query is less than the first range start or more than the last
        if query > ranges[-1]:
            results.append(None)  # Query out of range boundaries
            continue

        # Check within the ranges defined
        if query < ranges[0]:
            results.append(0)
            channel_num[0].append(query)
            layer_ = 0
            channel_ = query
            found = True
            pair_results.append((layer_,channel_))
            continue 
        
        found = False
        for i in range(len(ranges) - 1):
            if ranges[i] <= query < ranges[i + 1]:
                results.append(i + 1)  # Append the range number (1-indexed)
                channel_num[i+1].append(query-ranges[i])
                layer_ = i+1
                channel_ = query - ranges[i]
                found = True
                break
        
        # If the query is within the last specified range
        if not found and query >= ranges[-1]:
            results.append(len(ranges))
            channel_num[len(ranges)].append(query-ranges[-1])
            layer_ = len(ranges)
            channel_ = query - ranges[-1]
        pair_results.append((layer_,channel_))    
    
    return results,channel_num, pair_results

def channel_scoring(down,mid,up,mask):
    # resized_mask = cv2.resize(mask.astype(float), (512,512), interpolation = cv2.INTER_AREA).astype(bool)

    down0,down1,down2 = down[0], down[1], down[2]
    up0,up1,up2 = up[0] ,up[1] ,up[2]
    mid0 = mid[0]
    # layer_list = [down0,down1,down2,down3,down4,down5,down6,down7,down8,down9,down10,down11,mid0,up0,up1,up2,up3]
    layer_list = [down0,down1,down2,mid0,up0,up1,up2]
    scores = []
    
    dist_outside = distance_transform_edt(1 - mask)
    dist_outside /= dist_outside.max()

    # Apply a sigmoid function to smooth the transition
    smoothed_outside = np.power(dist_outside, 0.2)
    mask = 1-smoothed_outside
    
    
    k=4
    for att in layer_list:
        # ch , res = org_layer.shape[1],org_layer.shape[2]
        # att = cv2.resize(att.astype(float), (128,128), interpolation = cv2.INTER_AREA)#.astype(bool)

        # att = cv2.resize(att, (128, 128), interpolation=cv2.INTER_LINEAR)
        # att = np.expand_dims(att, axis=0)
        att = F.interpolate(
                            torch.tensor(att)[None,...],
                            size=(128,128),
                            mode="bilinear",
                        )[0]
        shape = att.shape
        att = att.reshape(shape[0],-1)
        att = np.abs(att)
        # min-max normalize
        att = (att-att.min(axis=1)[0][..., np.newaxis])
        att = att / att.max(axis=1)[0][..., np.newaxis]
        #softmax
        att = att - (att.min(axis=1)[0][...,np.newaxis]+att.max(axis=1)[0][...,np.newaxis])/2 
        # print("before sigmoid: ", att.min(axis=1)[0][:5], att.max(axis=1)[0][:5])
        att = sigmoid(att, k)

        ma = np.zeros_like(att)
        ma[np.where(att>0.4)] = 1
            
        tp = (ma*mask.flatten()).sum(axis=1)
        fp = (ma*(1-mask.flatten())).sum(axis=1)
        fn = mask.sum()-tp
        # tn = (~mask).sum() - fn
        
        tv_score = []
        for i in range(len(tp)):
            tversky_s = np.sum(tp[i]+1e-10)/(np.sum(tp[i])+2*np.sum(fp[i])+1*np.sum(fn[i])+1e-10)            
            tv_score.append(tversky_s)

        scores.append(tv_score)  
    # if step % 10 == 0:
        # print(f"Scoring {step}-steps are done!")
    return scores
    
def channel_extract(scores,n_mask_ratio=0.1):
    score_length = [len(scores[i]) for i in range(len(scores))]
    score_length_cum = np.cumsum(score_length)
    n_mask = int(len(scores)*n_mask_ratio)
    top_channels = np.argsort(np.hstack(scores))[::-1][:n_mask]
    _,_,top_pairs = find_ranges(score_length_cum,top_channels)
    
    top_dict = {}
    # Iterate over the list of tuples
    for key, value in top_pairs:
        # Check if the key is already in the dictionary
        if key in top_dict:
            # Append the value to the existing list
            top_dict[key].append(value)
        else:
            # Create a new list for the key
            top_dict[key] = [value]

    return top_dict
    
def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.seed is None:
        if args.n_samples > 1:
            seed = list(np.arange(0,args.n_samples,1))
        else:
            seed = [0]
    else:
        seed = [args.seed]
    prompt = args.prompt
    threshold = args.thres
    att_prompt = args.att_prompt
    att_idx = args.att_idx
    n_mask_ratio = args.n_mask_ratio
    
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to("cuda")
    
    channel_list = []
    img_list = []
    for s in seed:
        with StableDiffusionHooker(pipe) as hooker:
            set_seed(int(s))
            out, down, mid, up = pipe(prompt) #613
            img_list.append(out.images[0])
            
        ovam_evaluator = hooker.get_ovam_callable(
            expand_size=(128, 128)
        )  # You can configure OVAM here (aggregation, activations, size, ...)

        with torch.no_grad():
            attention_maps = ovam_evaluator(att_prompt)
            attention_maps = attention_maps[0].cpu().numpy() # (8, 512, 512)

        # Get maps for monkey, hat and mouth
        att = attention_maps[att_idx+1]
        
        attn_maps = att/ att.max()
        thres = np.quantile(attn_maps,threshold)
        mask = (attn_maps > thres).astype(int)
        # print("Start scoring layers.....")
        scores = channel_scoring(down,mid,up,mask)
        # print("================================================")
        # print("Start extracting channels with high score......")
        top_channels = channel_extract(scores,n_mask_ratio)
        channel_list.append(top_channels)
        hooker.clear()
        # ovam_evaluator.clear()
        if s % 10 == 0:
            percent_ = np.round((s/len(seed))*100,2)
            print(f"====================== {percent_}% of process is done! ==========================")         
        del out,down, mid, up, ovam_evaluator
         
    print("================================================")
    print("Saving dictionaries....")

    save_folder = '../channel_results/'+ f'{prompt}/{att_prompt}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = save_folder + f'edt_att{att_idx}_nsamples{args.n_samples}_mask{n_mask_ratio*100}%'+ '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(channel_list, f)
    save_path = save_folder + f'edt_att{att_idx}_nsamples{args.n_samples}_mask{n_mask_ratio*100}%_imgs'+ '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(img_list, f)
    print("Done!")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()