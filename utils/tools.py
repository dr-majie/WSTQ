import json, math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from PIL import Image


def get_acc_of_spec_type(pred, type, flag):
    que_flag = type == flag  # questions belonging to this flag
    total_que = np.sum(que_flag)
    print(flag, total_que)
    total_acc_que = np.sum(que_flag * pred)
    # assert total_que != 0, 'total question {} should not be 0'.format(total_que)
    return total_acc_que / total_que


def get_acc_of_type(pred, type):
    pred = np.array(pred)
    type = np.array(type)
    assert pred.size == type.size, 'pred\'s shape must be eqal to type'

    # compute the accuracy of 'what'
    what_acc = get_acc_of_spec_type(pred, type, 0)
    # compute the accuracy of 'how'
    how_acc = get_acc_of_spec_type(pred, type, 1)
    which_acc = get_acc_of_spec_type(pred, type, 2)
    where_acc = get_acc_of_spec_type(pred, type, 3)
    when_acc = get_acc_of_spec_type(pred, type, 4)
    who_acc = get_acc_of_spec_type(pred, type, 5)
    why_acc = get_acc_of_spec_type(pred, type, 6)
    other_acc = get_acc_of_spec_type(pred, type, 7)

    return what_acc, how_acc, which_acc, where_acc, when_acc, who_acc, why_acc, other_acc


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # return np.sum(pred_flat == labels_flat) / len(labels_flat)
    return np.sum(pred_flat == labels_flat), np.sum(pred_flat != labels_flat), pred_flat == labels_flat


def flat_accuracy_det(preds, labels):
    total_case = preds.shape[-1]
    preds = torch.sigmoid(preds)
    pos_acc = ((preds > 0.5) * labels).sum()
    neg_acc = ((preds < 0.5) * (labels == 0)).sum()
    total_points = pos_acc + neg_acc
    total_errors = total_case - (pos_acc + neg_acc)
    return total_points, total_errors


def extend_coordinate(box, extension):
    ext_box = [0] * 4
    for i, coor in enumerate(box):
        if i < 2:
            diff = box[i] - extension
            ext_box[i] = diff if diff > 0 else 0
        else:
            ext_box[i] = box[i] + extension
    return ext_box


def get_relation_label(coords, extension):
    num_roi = len(coords)
    rel_label = np.zeros(num_roi * num_roi)
    ix = 0
    for box1 in coords:
        for box2 in coords:
            xmin1, ymin1, xmax1, ymax1 = extend_coordinate(box1, extension)
            xmin2, ymin2, xmax2, ymax2 = extend_coordinate(box2, extension)

            xx1 = np.max([xmin1, xmin2])
            yy1 = np.max([ymin1, ymin2])
            xx2 = np.min([xmax1, xmax2])
            yy2 = np.min([ymax1, ymax2])

            inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
            if inter_area > 0:
                rel_label[ix] = 1
            ix = ix + 1
    return rel_label

'''-------datasets-------'''


def get_que_type(que):
    que_list = que.split(" ")
    if 'what' in que_list:
        return 0
    # if que_list[0] in 'what':  # a
    #     return 0
    if 'how' in que_list:
        return 1
    if 'which' in que_list:
        return 2
    if 'where' in que_list:
        return 3
    if 'when' in que_list:
        return 4
    if 'who' in que_list:
        return 5
    if 'why' in que_list:
        return 6
    return 7


def get_choice_encoded(text, question, answer, max_len, tokenizer):
    if text != "":
        first_part = text
        second_part = question + " " + answer
        encoded = tokenizer.encode_plus(first_part, second_part, max_length=max_len, pad_to_max_length=True)
    else:
        encoded = tokenizer.encode_plus(question + " " + answer, max_length=max_len, pad_to_max_length=True)
    input_ids = encoded["input_ids"]
    att_mask = encoded["attention_mask"]
    return input_ids, att_mask


def get_data_dmc(split, retrieval_solver, tokenizer, max_len):
    input_ids_list = []
    att_mask_list = []
    images_list = []
    coords_list = []
    labels_list = []
    que_type_list = []
    cont = 0
    with open("jsons/tqa_dmc.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for i, doc in enumerate(tqdm(dataset)):
        # # debugging
        # if i > 10:
        #    break
        # debugging over
        question = doc["question"]
        que_type_list.append(get_que_type(question))
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        images_q = []
        coords_q = []
        for count_i in range(4):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
            images_q.append(doc["image_path"])
            coord = [c[:4] for c in doc["coords"]]
            coords_q.append(coord)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images_list.append(images_q)
        coords_list.append(coords_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, images_list, coords_list, labels_list, que_type_list]


def get_rois(img_path, vectors, device):
    image = Image.open(img_path)
    '''
    image_d = ImageDraw.ImageDraw(image)
    image_name = img_path.split('/')[4]
    for vector in vectors:
        image_d.rectangle(((vector[0], vector[1]), (vector[2], vector[3])), fill=None, outline='red', width=5)
    image.save(image_name)
    image1 = os.path.join(os.getcwd(), image_name)
    image1 = cv2.imread(image1)
    '''
    rois = []
    for vector in vectors:
        roi_image = image.crop(vector)
        roi_image = roi_image.resize((224, 224), Image.ANTIALIAS)
        roi_image = np.array(roi_image)
        roi_image = torch.tensor(roi_image).type(torch.FloatTensor).permute(2, 0, 1).cuda(device)
        rois.append(roi_image)
    rois = torch.stack(rois, dim=0)
    return rois


def get_data_ndmc(dataset_name, split, retrieval_solver, tokenizer, max_len):
    input_ids_list = []
    att_mask_list = []
    labels_list = []
    que_type_list = []

    with open("jsons/tqa_" + dataset_name + ".json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for i, doc in enumerate(tqdm(dataset)):
        # debuging model
        # if i > 10:
        #     break
        question = doc["question"]
        que_type_list.append(get_que_type(question))
        text = doc["paragraph_" + retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        if dataset_name == "ndmc":
            counter = 7
        if dataset_name == "dmc":
            counter = 4
        for count_i in range(counter):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, labels_list, que_type_list]


def process_data_ndmc(raw_data, batch_size, split):
    input_ids_list, att_mask_list, labels_list, que_type_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    labels = torch.tensor(labels_list)
    que_types = torch.tensor(que_type_list)

    if split == "train":
        data = TensorDataset(inputs, masks, labels, que_types)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, labels, que_types)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return dataloader


def get_txt_match_data(tokenizer, max_len):
    names = ['dmc', 'ndmc', 'ndtf']
    solvers = ['IR', 'NSP', 'NN']
    dataset = []
    input_ids_list = []
    att_mask_list = []
    labels_list = []

    for name in names:
        with open("jsons/tqa_" + name + ".json", "r", encoding="utf-8", errors="surrogatepass") as file:
            docs = json.load(file)
            dataset.extend(docs)
    total_doc = len(dataset) - 1

    for i, doc in enumerate(tqdm(dataset)):
        # debug model
        # if i > 20:
        #     break
        question = doc["question"]
        try:
            answers = list(doc["answers"].values())
            answers = ' '.join(ans for ans in answers)
        except:
            answers = doc['correct_answer']

        try:
            ir_txt = doc["paragraph_" + solvers[0]]
            nsp_txt = doc["paragraph_" + solvers[1]]
            nn_txt = doc["paragraph_" + solvers[2]]
        except:
            ir_txt = doc["sentence_" + solvers[0]]
            nsp_txt = doc["sentence_" + solvers[1]]
            nn_txt = doc["sentence_" + solvers[2]]

        ir_input_ids, ir_att_mask = get_choice_encoded(ir_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(ir_input_ids)
        att_mask_list.append(ir_att_mask)
        labels_list.append(1)

        nsp_input_ids, nsp_att_mask = get_choice_encoded(nsp_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(nsp_input_ids)
        att_mask_list.append(nsp_att_mask)
        labels_list.append(1)

        nn_input_ids, nn_att_mask = get_choice_encoded(nn_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(nn_input_ids)
        att_mask_list.append(nn_att_mask)
        labels_list.append(1)

        # build the negative paragraphs to the above question
        ix = (math.floor(total_doc / 2) + i) % total_doc
        try:
            neg_ir_txt = dataset[ix]["paragraph_" + solvers[0]]
            neg_nsp_txt = dataset[ix]["paragraph_" + solvers[1]]
            neg_nn_txt = dataset[ix]["paragraph_" + solvers[2]]
        except:
            neg_ir_txt = dataset[ix]["sentence_" + solvers[0]]
            neg_nsp_txt = dataset[ix]["sentence_" + solvers[1]]
            neg_nn_txt = dataset[ix]["sentence_" + solvers[2]]

        neg_ir_input_ids, neg_ir_att_mask = get_choice_encoded(neg_ir_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(neg_ir_input_ids)
        att_mask_list.append(neg_ir_att_mask)
        labels_list.append(0)

        neg_nsp_input_ids, neg_nsp_att_mask = get_choice_encoded(neg_nsp_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(neg_nsp_input_ids)
        att_mask_list.append(neg_nsp_att_mask)
        labels_list.append(0)

        neg_nn_input_ids, neg_nn_att_mask = get_choice_encoded(neg_nn_txt, question, answers, max_len, tokenizer)
        input_ids_list.append(neg_nn_input_ids)
        att_mask_list.append(neg_nn_att_mask)
        labels_list.append(0)

    return [input_ids_list, att_mask_list, labels_list]


def get_data_ndtf(split, retrieval_solver, tokenizer, max_len):
    input_ids_list = []
    att_mask_list = []
    labels_list = []
    que_type_list=[]
    with open("jsons/tqa_ndtf.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for i, doc in enumerate(tqdm(dataset)):
        # debuging model
        # if i > 20:
        #     break
        question = doc["question"]
        text = doc["sentence_" + retrieval_solver]
        encoded = tokenizer.encode_plus(text, question, max_length=max_len, pad_to_max_length=True)
        input_ids = encoded["input_ids"]
        att_mask = encoded["attention_mask"]
        label = 0
        if doc["correct_answer"] == "true":
            label = 1
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)
        labels_list.append(label)
        que_type_list.append(0)
    return [input_ids_list, att_mask_list, labels_list,que_type_list]