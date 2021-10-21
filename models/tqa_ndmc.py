from transformers import RobertaForMultipleChoice
import torch


def build_model_ndmc(args):
    model = RobertaForMultipleChoice.from_pretrained("roberta-large")
    if args.pretrainings != "":
        model.roberta = torch.load(args.pretrainings).roberta
    else:
        print('txt_matching adopted')
        model.roberta = torch.load("checkpoints/txt_matching_e2.pth").roberta

    if args.device == "gpu":
        device = torch.device("cuda:0")
        model.cuda(device)
    if args.device == "cpu":
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()
    return device, model
