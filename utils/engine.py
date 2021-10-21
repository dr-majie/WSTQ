import numpy as np
import json, math
from tqdm import tqdm
import torch
from models.tqa_ndmc import build_model_ndmc
from models.tqa_ndtf import build_model_ndtf
import random
import os
from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from utils.tools import get_data_dmc, flat_accuracy, flat_accuracy_det, get_acc_of_type, \
    get_data_ndmc, process_data_ndmc, get_data_ndtf
from models.tqa_dmc import build_model_dmc
from models.txt_matching import train_txt_matching_with_default_settings


class Engine(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def args_phrase(self, args, model_type):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        batch_size = args.batchsize
        max_len = args.maxlen
        lr = args.lr
        epochs = args.epochs
        retrieval_solver = args.retrieval
        save_model = args.save

        if not os.path.exists("checkpoints/txt_matching_e2.pth"):
            train_txt_matching_with_default_settings(args.device)

        if model_type == "dmc":
            device, model = build_model_dmc(args)
            raw_data_train = get_data_dmc("train", retrieval_solver, tokenizer, max_len)
            raw_data_val = get_data_dmc("val", retrieval_solver, tokenizer, max_len)
            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
            total_steps = len(raw_data_train[-1]) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            self.training_dmc_rel(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size,
                                  retrieval_solver,
                                  device, save_model)
        elif model_type == "ndmc":
            device, model = build_model_ndmc(args)
            dataset_name = 'ndmc'
            raw_data_train = get_data_ndmc(dataset_name, "train", retrieval_solver, tokenizer, max_len)
            raw_data_val = get_data_ndmc(dataset_name, "val", retrieval_solver, tokenizer, max_len)
            train_dataloader = process_data_ndmc(raw_data_train, batch_size, "train")
            val_dataloader = process_data_ndmc(raw_data_val, batch_size, "val")

            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            self.training_ndmc(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver,
                               device, save_model, dataset_name)

        elif model_type == "ndtf":
            device, model = build_model_ndtf(args)
            raw_data_train = get_data_ndtf("train", retrieval_solver, tokenizer, max_len)
            raw_data_val = get_data_ndtf("val", retrieval_solver, tokenizer, max_len)
            train_dataloader = process_data_ndmc(raw_data_train, batch_size, "train")
            val_dataloader = process_data_ndmc(raw_data_val, batch_size, "val")

            optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
            total_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            self.training_ndtf(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver,
                               device, save_model)

    def arg_phrase_test(self, args):
        models = [torch.load(args.pretrainings)]
        retrieval_solvers = [args.retrieval]
        model_types = [args.model_type]
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        max_len = args.maxlen
        batch_size = args.batchsize

        feats_train = []
        feats_val = []
        feats_test = []
        for model, model_type, retrieval_solver in zip(models, model_types, retrieval_solvers):
            if args.device == "gpu":
                device = torch.device("cuda")
                model.cuda(device=device)
            if args.device == "cpu":
                device = torch.device("cpu")
                model.cpu()
            model.eval()
            print("\n")
            if model_type == "dmc":
                print("val")
                raw_data_val = get_data_dmc("val", retrieval_solver, tokenizer, max_len)
                self.validation_dmc_rel(model, raw_data_val, batch_size, device)
                print("test")
                raw_data_test = get_data_dmc("test", retrieval_solver, tokenizer, max_len)
                self.validation_dmc_rel(model, raw_data_test, batch_size, device)
            if model_type == "ndmc":
                print("val")
                raw_data_val = get_data_ndmc("ndmc", "val", retrieval_solver, tokenizer, max_len)
                val_dataloader = process_data_ndmc(raw_data_val, batch_size, "val")
                self.validation_ndmc(model, val_dataloader, device)
                print("test")
                raw_data_test = get_data_ndmc("ndmc", "test", retrieval_solver, tokenizer, max_len)
                test_dataloader = process_data_ndmc(raw_data_test, batch_size, "test")
                self.validation_ndmc(model, test_dataloader, device)
            if model_type == "ndtf":
                print("val")
                raw_data_train = get_data_ndtf("val", retrieval_solver, tokenizer, max_len)
                train_dataloader = process_data_ndmc(raw_data_train, batch_size, "val")
                self.validation_ndtf(model, train_dataloader, device)
                print("test")
                raw_data_test = get_data_ndtf("test", retrieval_solver, tokenizer, max_len)
                test_dataloader = process_data_ndmc(raw_data_test, batch_size, "test")
                self.validation_ndtf(model, test_dataloader, device)

    def training_dmc_rel(self, model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size,
                         retrieval_solver,
                         device, save_model):
        # save the average loss for qa and det
        # avg_loss = np.zeros([2, epochs], dtype=np.float32)
        # lambda_weight = np.ones([2, epochs])

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            input_ids_list, att_mask_list, images_list, coords_list, labels_list, que_type_list = raw_data_train

            # Reset the total loss for this epoch.
            total_points = 0
            total_errors = 0
            total_points_det = 0
            total_errors_det = 0
            train_loss_list = []
            qa_loss_list = []
            det_loss_list = []
            pred_list = []
            type_list = []

            """
            T = 2
            if epoch_i < 2:
                lambda_weight[:, epoch_i] = 1
            else:
                w_1 = avg_loss[0, epoch_i - 1] / avg_loss[0, epoch_i - 2]
                w_2 = avg_loss[1, epoch_i - 1] / avg_loss[1, epoch_i - 2]
                lambda_weight[0, epoch_i] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, epoch_i] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            """
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
                b_input_images = [x for y, x in enumerate(images_list) if y in batch_ids]
                b_input_coords = [x for y, x in enumerate(coords_list) if y in batch_ids]
                b_labels = torch.tensor([x for y, x in enumerate(labels_list) if y in batch_ids]).to(device)
                b_types = torch.tensor([x for y, x in enumerate(que_type_list) if y in batch_ids]).to(device)
                outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords,
                                labels=b_labels)
                # outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, labels=b_labels)

                loss, qa_loss, det_loss, qa_logits, det_logits, det_labels = outputs

                # Move qa_logits and labels to CPU
                qa_logits = qa_logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                points, errors, pred = flat_accuracy(qa_logits, label_ids)
                total_points = total_points + points
                total_errors = total_errors + errors
                pred_list.extend(pred)
                type_list.extend(b_types)

                # Calculate the accuracy of relation detection
                points, errors = flat_accuracy_det(det_logits, det_labels)
                total_points_det = total_points_det + points
                total_errors_det = total_errors_det + errors

                # Perform a backward pass to calculate the gradients.
                # train_loss = [qa_loss, det_loss]
                # loss = sum([lambda_weight[i, epoch_i] * train_loss[i] for i in range(2)])
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
                acc_det = total_points_det / (total_points_det + total_errors_det)

                train_loss_list.append(loss.item())
                train_loss = np.mean(train_loss_list)

                qa_loss_list.append(qa_loss.item())
                qa_loss = np.mean(qa_loss_list)
                det_loss_list.append(det_loss.item())
                det_loss = np.mean(det_loss_list)
                pbar.set_description(
                    "qa_accuracy: {0:.4f} qa_loss: {1:.4f} det_accuracy: {2:.4f} det_loss: {3:.4f} total_loss: {4:.4f}".format(
                        train_acc, qa_loss, acc_det, det_loss, train_loss))

                """
                pbar.set_description(
                    "qa_accuracy: {0:.4f} qa_loss: {1:.4f} det_accuracy: {2:.4f} det_loss: {3:.4f} lambda_0: {4:.4f} lambda_1: {5:.4f} total_loss: {6:.4f}".format(
                        train_acc, qa_loss, acc_det, det_loss, lambda_weight[0][epoch_i], lambda_weight[1][epoch_i], train_loss))

            avg_loss[0][epoch_i] = np.mean(qa_loss_list)
            avg_loss[1][epoch_i] = np.mean(det_loss_list)
            """
            print('\n')
            print(
                'Accuracy -> what: {0:.4f} how: {1:.4f} which: {2:.4f}, where: {3:.4f} when: {4:.4f} who: {5:.4f}, why: {6:.4f} other: {7:.4f}'.
                    format(*get_acc_of_type(pred_list, type_list)))
            if save_model:
                torch.save(model, "checkpoints/dmc_dmc_roberta_" + str(epoch_i + 1) + ".pth")

            self.validation_dmc_rel(model, raw_data_val, batch_size, device)

        print("")
        print("Training complete!")

    def validation_dmc_rel(self, model, raw_data_val, batch_size, device):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        input_ids_list, att_mask_list, images_list, coords_list, labels_list, que_type_list = raw_data_val

        total_points = 0
        total_errors = 0
        total_points_det = 0
        total_errors_det = 0
        val_loss_list = []
        qa_loss_list = []
        det_loss_list = []
        final_res = []
        pred_list = []
        type_list = []

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Evaluate data for one epoch
        # sum_aux = 0
        # total_aux = 0

        dataset_ids = list(range(len(labels_list)))
        batched_ids = [dataset_ids[k:k + batch_size] for k in range(0, len(dataset_ids), batch_size)]
        pbar = tqdm(batched_ids)
        for batch_ids in pbar:
            # Unpack the inputs from our dataloader
            b_input_ids = torch.tensor([x for y, x in enumerate(input_ids_list) if y in batch_ids]).to(device)
            b_input_mask = torch.tensor([x for y, x in enumerate(att_mask_list) if y in batch_ids]).to(device)
            b_input_images = [x for y, x in enumerate(images_list) if y in batch_ids]
            b_input_coords = [x for y, x in enumerate(coords_list) if y in batch_ids]
            b_labels = torch.tensor([x for y, x in enumerate(labels_list) if y in batch_ids]).to(device)
            b_types = torch.tensor([x for y, x in enumerate(que_type_list) if y in batch_ids]).to(device)

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords,
                                labels=b_labels)
                # outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, labels=b_labels)

                loss, qa_loss, det_loss, qa_logits, det_logits, det_labels = outputs

                # Move qa_logits and labels to CPU
                qa_logits = qa_logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                points, errors, pred = flat_accuracy(qa_logits, label_ids)
                total_points = total_points + points
                total_errors = total_errors + errors
                pred_list.extend(pred)
                type_list.extend(b_types)

                # Calculate the accuracy of relation detection
                points, errors = flat_accuracy_det(det_logits, det_labels)
                total_points_det = total_points_det + points
                total_errors_det = total_errors_det + errors

            val_loss_list.append(loss.item())
            qa_loss_list.append(qa_loss.item())
            det_loss_list.append(det_loss.item())

        val_acc = total_points / (total_points + total_errors)
        val_acc_det = total_points_det / (total_points_det + total_errors_det)
        val_loss = np.mean(val_loss_list)
        val_qa_loss = np.mean(qa_loss_list)
        val_det_loss = np.mean(det_loss_list)

        print(
            "qa_accuracy: {0:.4f} qa_loss: {1:.4f} det_accuracy: {2:.4f} det_loss: {3:.4f} total_loss: {4:.4f}".format(
                val_acc, val_qa_loss, val_acc_det, val_det_loss, val_loss))
        print('\n')
        print(
            'Accuracy -> what: {0:.4f} how: {1:.4f} which: {2:.4f}, where: {3:.4f} when: {4:.4f} who: {5:.4f}, why: {6:.4f} other: {7:.4f}'.
                format(*get_acc_of_type(pred_list, type_list)))

        return final_res

    def training_ndmc(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver,
                      device,
                      save_model, dataset_name):
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Reset the total loss for this epoch.
            total_points = 0
            total_errors = 0
            train_loss_list = []
            pred_list = []
            type_list = []

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # For each batch of training data...
            pbar = tqdm(train_dataloader)
            for batch in pbar:
                model.train()
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_types = batch[3].to(device)

                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                loss, logits = outputs

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                points, errors, pred = flat_accuracy(logits, label_ids)
                total_points = total_points + points
                total_errors = total_errors + errors
                pred_list.extend(pred)
                type_list.extend(b_types)

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

                pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))
            print('\n')
            print(
                'Accuracy -> what: {0:.4f} how: {1:.4f} which: {2:.4f}, where: {3:.4f} when: {4:.4f} who: {5:.4f}, why: {6:.4f} other: {7:.4f}'.
                    format(*get_acc_of_type(pred_list, type_list)))
            if save_model:
                torch.save(model, "checkpoints/ndmc_" + dataset_name + "_roberta_" + retrieval_solver + "_e" + str(
                    epoch_i + 1) + ".pth")

            self.validation_ndmc(model, val_dataloader, device)

        print("")
        print("Training complete!")

    def validation_ndmc(self, model, val_dataloader, device):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        total_points = 0
        total_errors = 0
        val_loss_list = []
        final_res = []
        pred_list = []
        type_list = []

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Evaluate data for one epoch
        sum_aux = 0
        total_aux = 0

        for batch in tqdm(val_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_tyes = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss, logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            for l in logits:
                final_res.append(l)
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors, pred = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors
            pred_list.extend(pred)
            type_list.extend(b_tyes)

            val_loss_list.append(loss.item())

        val_acc = total_points / (total_points + total_errors)
        val_loss = np.mean(val_loss_list)

        print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
        print('\n')
        print(
            'Accuracy -> what: {0:.4f} how: {1:.4f} which: {2:.4f}, where: {3:.4f} when: {4:.4f} who: {5:.4f}, why: {6:.4f} other: {7:.4f}'.
                format(*get_acc_of_type(pred_list, type_list)))

        return final_res

    def training_ndtf(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver,
                      device,
                      save_model=False):
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Reset the total loss for this epoch.
            total_points = 0
            total_errors = 0
            train_loss_list = []

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # For each batch of training data...
            pbar = tqdm(train_dataloader)
            for batch in pbar:
                model.train()
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

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

                pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

            if save_model:
                torch.save(model, "checkpoints/ndtf_roberta_" + retrieval_solver + "_e" + str(epoch_i + 1) + ".pth")

            self.validation_ndtf(model, val_dataloader, device)

        print("")
        print("Training complete!")

    def validation_ndtf(self, model, val_dataloader, device):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        total_points = 0
        total_errors = 0
        val_loss_list = []
        final_res = []

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Evaluate data for one epoch
        sum_aux = 0
        total_aux = 0

        for batch in tqdm(val_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels,_ = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss, logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            for l in logits:
                final_res.append(l)
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors, _ = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            val_loss_list.append(loss.item())

        val_acc = total_points / (total_points + total_errors)
        val_loss = np.mean(val_loss_list)

        print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))

        return final_res


