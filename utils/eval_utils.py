import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}

    if args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, n_classes = args.n_classes, act = 'gelu', dropout = args.drop_out)
    elif args.model_type == 'dsmil':
        from models.DSMIL import FCLayer, BClassifier, MILNet
        i_classifier = FCLayer(in_size= args.in_dim, out_size= args.n_classes)
        b_classifier = BClassifier(input_size= args.in_dim, output_class= args.n_classes, dropout_v=args.drop_out)
        model = MILNet(i_classifier, b_classifier) 
    elif args.model_type == 'wikgmil':
        from models.WiKGMIL import WiKG
        model = WiKG(dim_in=args.in_dim, dim_hidden=512, topk=2, n_classes=args.n_classes, agg_type='bi-interaction', dropout=args.drop_out, pool='attn').cuda()
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
    elif args.model_type=='rrtmil':
        model_params = {
            'input_dim': args.in_dim,
            'n_classes': args.n_classes,
            'dropout': args.drop_out,
            'act': 'gelu',
            'region_num': 32,
            'pos': None,
            'pos_pos': 0,
            'pool': 'attn',
            'peg_k': 7,
            'drop_path': 0.,
            'n_layers': 2,
            'n_heads': 8,
            'attn':'rmsa',
            'da_act': 'tanh',
            'trans_dropout': 0.1,
            'ffn': True,
            'mlp_ratio': 4.,
            'trans_dim': 64,
            'epeg': False,
            'min_region_num': 0,
            'qkv_bias': False,
            'epeg_k': 13,
            'epeg_2d': True,
            'epeg_bias': False,
            'epeg_type': 'attn',
            'region_attn': 'native',
            'peg_1d': True,
            'cr_msa': False,
            'crmsa_k': 3,
            'all_shortcut': True,
            'crmsa_mlp':True,
            'crmsa_heads':1,
         }
        from modules.rrt import RRTMIL
        model = RRTMIL(**model_params).cuda()
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def eval_survival(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader_survival(dataset)
    patient_results, c_index = summary_survival(model, loader, args)
    #print('test_error: ', test_error)
    print('c-index: ', c_index)
    return patient_results, c_index

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        #print(slide_id)
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(tqdm(loader)):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            # hazards, survival, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic)
            hazards, survival, Y_hat, _, _ = model(data_WSI)
            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)
        # risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy()) 
        # risk = np.ndarray.item(-torch.sum(survival, dim=1).cpu().numpy())
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        # event_time = np.asscalar(event_time)
        event_time = np.ndarray.item(event_time)
        # c = np.asscalar(c)
        c = np.ndarray.item(c.cpu().numpy())
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index