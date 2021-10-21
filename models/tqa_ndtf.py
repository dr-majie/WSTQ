from transformers import RobertaForSequenceClassification
import torch

def build_model_ndtf(args):
    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
    if args.pretrainings != "":
        model.roberta = torch.load(args.pretrainings).roberta
    else:
        model.roberta = torch.load("./checkpoints/txt_matching_e2.pth").roberta

    if args.device == "gpu":
        device = torch.device("cuda:0")
        model.cuda(device)
    if args.device == "cpu":
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    return device, model