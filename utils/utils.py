import torch
import os
import numpy as np
import sys
import pysam
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC



class ADDataModule():
    def __init__(self, data_dir: str = "path/to/dir",vali_set = ["chr10"], test_set=["chr8","chr9"], seq_length = 4096,batch_size: int = 64, pretrain = False):
        super().__init__()

        # [C3] Read reference genome path from environment variable.
        # Set before running: export REFERENCE_GENOME_PATH=/path/to/GRCh38.fasta
        faste_path = os.environ.get(
            "REFERENCE_GENOME_PATH",
            "Path to reference genome"  # fallback placeholder — must be replaced
        )
        self.fasta_file =pysam.Fastafile(faste_path)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vali_set = vali_set
        self.test_set = test_set
        self.seq_length = seq_length
        df_data = pd.read_csv(self.data_dir)
        self.pretrain = pretrain
        
        self.train_data =HistoneDataset(df_data[(~df_data["chrom"].isin(self.vali_set))&(~df_data["chrom"].isin(self.test_set))],self.fasta_file, self.seq_length, self.pretrain)
        self.vali_data =  HistoneDataset(df_data[df_data["chrom"].isin(self.vali_set)],self.fasta_file, self.seq_length,self.pretrain)
        self.test_data = HistoneDataset(df_data[df_data["chrom"].isin(self.test_set)],self.fasta_file, self.seq_length,self.pretrain)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.vali_data, batch_size=self.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,shuffle=False)


class HistoneDataset(Dataset):
    def __init__(self, datafiles, fasta_file,seq_length, pretrain = False, transform=None, target_transform=None):
        self.data_file = datafiles
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.pretrain = pretrain
        if self.pretrain:
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __len__(self):
        return self.data_file.shape[0]

    def __getitem__(self, idx):
        chr_temp, start_chr, end_chr = self.data_file.iloc[idx,:3]
        label = self.data_file.iloc[idx,3:]
        start_position = int((start_chr+end_chr)/2)-int(self.seq_length/2)
        seq = self.fasta_file.fetch(chr_temp,start_position, start_position+self.seq_length)
        if self.pretrain:
            encoded_sequence = self.tokenizer(seq, return_tensors="pt")['input_ids'][0]
        else:
            encoded_sequence = one_hot_encode_dna(seq)

        sample = {'sequence': encoded_sequence,
                 'label': torch.tensor(label.to_list(),dtype=torch.float)}
        # sample.update(label.to_dict())
        
        return  sample

class mutationDataset(Dataset):
    def __init__(self,temp_chr, snp,seq_length, test_length,pretrain = False, transform=None, target_transform=None):
        self.snp = snp
        self.chr = temp_chr
        # [C4] Replaced original author's hardcoded absolute path with env var
        faste_path = os.environ.get(
            "REFERENCE_GENOME_PATH",
            "Path to reference genome"  # fallback placeholder — must be replaced
        )
        self.fasta_file =pysam.Fastafile(faste_path)
        self.seq_length = seq_length
        self.pretrain = pretrain
        self.test_length = test_length
        self.mutation = ["A","G","C","T"]
        if self.pretrain:
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __len__(self):
        return len(self.snp)*4*self.test_length

    def __getitem__(self, idx):
        snp_idx = idx//(self.test_length*4)
        pos_idx = (idx-snp_idx*(self.test_length*4))%self.test_length
        new_mutation = self.mutation[(idx-snp_idx*(self.test_length*4))//self.test_length]
        start_chr, end_chr = int(self.snp[snp_idx]-self.seq_length/2), int(self.snp[snp_idx]+self.seq_length/2)
        seq = self.fasta_file.fetch(self.chr,start_chr, end_chr)
        loc = int(self.seq_length/2-self.test_length/2+pos_idx)
        new_seq = seq[:loc] + new_mutation + seq[loc+1:]
        # print(loc, new_mutation,start_chr, end_chr)
       
        if self.pretrain:
            encoded_sequence = self.tokenizer(new_seq, return_tensors="pt")['input_ids'][0]
        else:
            encoded_sequence = one_hot_encode_dna(new_seq)
        
        return  encoded_sequence

def one_hot_encode_dna(seq):
    seq_len = len(seq)
    seq = seq.upper()
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    seq_code = torch.zeros((seq_len, 4))

    for i, base in enumerate(seq):
        if base in base_map:
            index = base_map[base]
            seq_code[i, index] = 1
    
    return seq_code.transpose(0, 1)

class multiperformance():

    def __init__(self):
        self.metric_dict = { "accuracy": BinaryAccuracy(),"AUC":BinaryAUROC(),"F1":BinaryF1Score(),"PRC": BinaryAUPRC()}

    def update(self, inputs, target):
        for name, m in self.metric_dict.items():
            m.update(inputs, target)

    def compute(self):
        return { name:m.compute() for name,m in self.metric_dict.items()}

    def reset(self):
        for name, m in self.metric_dict.items():
            m.reset()
