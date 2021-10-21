from utils.tools import get_txt_match_data, flat_accuracy
import torch
from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import random
from tqdm import tqdm
import numpy as np

def train_txt_matching_with_default_settings(devices):
    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    device = None
    if devices == "gpu":
        device = torch.device("cuda:0")
        model.cuda(device=device)
    if devices == "cpu":
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    batch_size = 1
    max_len = 180
    lr = 1e-6
    epochs = 4
    save_model = True

    data = get_txt_match_data(tokenizer, max_len)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(data[-1]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    training_txt_matching(model, data, optimizer, scheduler, epochs, batch_size, device, save_model)


def training_txt_matching(model, data, optimizer, scheduler, epochs, batch_size, device, save_model):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        input_ids_list, att_mask_list, labels_list = data

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        dataset_ids = list(range(len(labels_list)))
        random.shuffle(dataset_ids)
        batched_ids = [dataset_ids[k:k + batch_size] for k in range(0, len(dataset_ids), batch_size)]
        pbar = tqdm(batched_ids)
        for batch_ids in pbar:
            model.train()
            b_input_ids = torch.tensor([x for y, x in enumerate(input_ids_list) if y in batch_ids]).to(device)
            b_input_mask = torch.tensor([x for y, x in enumerate(att_mask_list) if y in batch_ids]).to(device)
            b_labels = torch.tensor([x for y, x in enumerate(labels_list) if y in batch_ids]).to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, labels=b_labels)

            loss, logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors, _ = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points / (total_points + total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description(
                "txt_matching accuracy: {0:.4f} loss: {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/txt_matching" + "_e" + str(epoch_i + 1) + ".pth")

    print("")
    print("Training complete!")