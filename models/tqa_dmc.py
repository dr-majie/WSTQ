from transformers import RobertaModel
import numpy as np
import torch
import torchvision.models as models
from utils.tools import get_rois,get_relation_label
from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer

def build_model_dmc(args):
    if args.pretrainings != "":
        model = torch.load(args.pretrainings)
    else:
        model = RobertaRelationDetection()
        model.roberta = torch.load("./checkpoints/txt_matching_e2.pth").roberta
    for param in model.resnet.parameters():
        param.requires_grad = False
    device = None
    if args.device == "gpu":
        device = torch.device("cuda:0")
        model.cuda(device=device)
    if args.device == "cpu":
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    return device, model



class RobertaRelationDetection(torch.nn.Module):
    def __init__(self):
        super(RobertaRelationDetection, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-large")

        self.resnet = models.resnet101(pretrained=True)

        self.feats = torch.nn.Sequential(torch.nn.Linear(1000, 1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))

        self.boxes = torch.nn.Sequential(torch.nn.Linear(4, 1024), torch.nn.LayerNorm(1024, eps=1e-12))

        self.dropout = torch.nn.Dropout(0.1)
        self.rel_classifier = torch.nn.Linear(1024, 1)
        self.classifier = torch.nn.Linear(1024, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images=None, coords=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b = []
        coord_b = []
        for img_q, coord_q in zip(images, coords):
            for img_file, coord in zip(img_q, coord_q):
                img_b.append(img_file)
                coord_b.append(coord)

        roi_b = []
        rel_label_list = []
        rel_logit_list = []
        device = out_roberta.device
        for image, coord, roberta_b in zip(img_b, coord_b, out_roberta):
            img_v = get_rois(image, coord[:32], device)
            coord_v = torch.tensor(coord[:32]).cuda(device)
            # obtain relation label
            rel_label = get_relation_label(coord[:32], 50)
            rel_label_list.extend(rel_label)
            # over
            out_boxes = self.boxes(coord_v)
            out_resnet = self.resnet(img_v)
            out_resnet = self.feats(out_resnet)
            out_resnet = self.feats2(out_resnet)
            out_resnet = out_resnet.view(-1, 1024)
            out_roi = (out_resnet + out_boxes) / 2
            # predict the relation according to the features
            num_roi = out_roi.shape[0]
            roi_feat0 = out_roi.repeat(1, num_roi).view(num_roi, num_roi, -1)
            roi_feat1 = out_roi.repeat(num_roi, 1).view(num_roi, num_roi, -1)
            rel_feats = self.dropout(roi_feat0 * roi_feat1)
            rel_logit = self.rel_classifier(rel_feats).squeeze(-1)
            rel_logit = rel_logit.view(num_roi * num_roi)
            rel_logit_list.append(rel_logit)
            # over
            out_roi = torch.sum(out_roi, dim=0)
            roi_b.append(out_roi)

        out_visual = torch.stack(roi_b, dim=0)
        final_out = out_roberta * out_visual

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        qa_logits = (reshaped_logits,)

        rel_labels = torch.from_numpy(np.array(rel_label_list)).cuda(device)
        rel_logits = torch.cat(rel_logit_list, dim=0)
        outputs = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss0 = loss_fct(reshaped_logits, labels)

            # if this is effective, considering focal loss
            loss_rel = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.5]).cuda(device))
            loss1 = loss_rel(rel_logits, rel_labels.cuda(device))

            loss = loss0 + 0.1 * loss1
            outputs = (loss,) + (loss0,) + (loss1,) + qa_logits + (rel_logits,) + (rel_labels,)
        return outputs

