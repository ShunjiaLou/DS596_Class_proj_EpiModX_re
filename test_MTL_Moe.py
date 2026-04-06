import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from einops import rearrange
import sys
from einops.layers.torch import Rearrange

# [C2] Baseline benchmark models — not included in repo, import commented out
# from sei import *
# from pretrain_multihead import *
# from DeepHistone import NetDeepHistone
# from ablution_Study import CNN_BLSTM

from Pretrain_Moe import *
from utils import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
import pickle


def main(args):

    Histone = args.histone
    # Histone =  "H3K27me3"
    # Histone = "H3K27ac"

    model_name = "LLM_Moe"
    task_dict = {"task1":6,"task2":5,"task3":4,"task4":4,"task5":3}
    model_path = "./models/%s_%s.pt" %(Histone,model_name)
    batch_size = 8
    seq_length = 4096

    print("load datasets...")
    # parallel_experts (MoE) uses custom kernels incompatible with MPS;
    # use CUDA when available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_module = ADDataModule("./Datasets/%s_all_data.csv" %Histone,["chr10"], ["chr8","chr9"],seq_length, batch_size,pretrain=True)
    test_loader = data_module.test_dataloader()
    model = Pretrain_Moe(task_dict).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    prediction_all = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch['sequence'], batch['label']
            data, labels = data.to(device), labels.to(device)
            # [B3] model() returns (outputs_dict, aux_loss) — unpack correctly
            outputs, _ = model(data)
            outputs = torch.concat([v for k, v in outputs.items()], dim=1)
            logits = torch.sigmoid(outputs)
            prediction_all.append(logits.cpu().detach().numpy())

    os.makedirs("./test_results", exist_ok=True)
    with open("./test_results/%s_%s_test_result" %(Histone,model_name),"wb") as f:
        pickle.dump(prediction_all,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test EpiModX model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # [B2] Changed short flag from -h to -t to avoid conflict with argparse's built-in --help
    parser.add_argument("-t", "--histone", type=str, default="H3K27ac",
                        help="Histone type (H3K27ac, H3K4me3, H3K27me3)")
    parser.add_argument('--save_model', type=bool, default=False,
                        help='For Saving the current Model')
    # [B1] Added --seed argument (was referenced on line 71 but never defined)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(args)
