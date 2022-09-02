import torch
from tqdm import tqdm
import torch.nn as nn
from lib.utils.utils import get_lr
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score
import csv,os


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, train_mode, cuda):
    loss = 0
    val_loss = 0

    # GOO
    # Loss functions
    mse_loss = nn.MSELoss(reduce=False)  # not reducing in order to ignore outside cases
    loss_amp_factor = 10000  # multiplied to the loss to prevent underflow
    running_loss = []

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, faces, head, gaze_heatmap, gt_box = batch[0], batch[1], batch[2], batch[3], batch[4], \
                                                                 batch[7]

            images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            faces = torch.from_numpy(faces).type(torch.FloatTensor).cuda()
            head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            gaze_heatmap = torch.from_numpy(gaze_heatmap).type(torch.FloatTensor).cuda()
            targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

            # ----------------------#
            #   Clear gradient
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   Forward propagation
            # ----------------------#
            outputs = model_train(images, head, faces,train_mode)
            # GOO
            if train_mode==0:
                gaze_heatmap_pred = outputs[0]
                gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                # GOO Loss
                #l2 loss computed only for inside case
                l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap) * loss_amp_factor
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss)

            loss_value_all = 0
            num_pos_all = 0
            # ----------------------#
            #   Calculate the loss
            # ----------------------#
            outputs = outputs[1:]
            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            loss_value = loss_value_all / num_pos_all
            if train_mode==0:
                gt_box = gt_box.squeeze(1)
                box_energy_loss = compute_heatmap_loss_by_gtbox_predheatmap(gt_box, gaze_heatmap_pred)
                total_loss =  loss_value +l2_loss+ box_energy_loss
            if train_mode == 1:
                total_loss=loss_value
            total_loss.backward()
            optimizer.step()

            loss += total_loss.item()

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    loss_amp_factor = 10000  # multiplied to the loss to prevent underflow
    model_train.eval()
    test_loss = []
    total_error = []

    mse_loss = nn.MSELoss(reduce=False)  # not reducing in order to ignore outside cases

    all_gazepoints = []
    all_predmap = []
    all_gtmap = []
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets, faces, head, gaze_heatmap, eye_position, gaze = batch[0], batch[1], batch[2], batch[3], \
                                                                             batch[4], batch[5], batch[6]
            with torch.no_grad():
                val_images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                val_head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
                val_faces = torch.from_numpy(faces).type(torch.FloatTensor).cuda()
                val_gaze_heatmap = torch.from_numpy(gaze_heatmap).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

                outputs = model_train(val_images, val_head, val_faces, train_mode)
                # Loss
                # l2 loss computed only for inside case
                if train_mode==0:
                    val_gaze_heatmap_pred = outputs[0]
                    val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1)
                    l2_loss = mse_loss(val_gaze_heatmap_pred, val_gaze_heatmap) * loss_amp_factor
                    l2_loss = torch.mean(l2_loss, dim=1)
                    l2_loss = torch.mean(l2_loss, dim=1)
                    l2_loss = torch.mean(l2_loss)

                loss_value_all = 0
                num_pos_all = 0
                # ----------------------#
                #   Calculate the loss
                # ----------------------#
                outputs = outputs[1:]
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all
                if train_mode==0:
                    total_loss = l2_loss+loss_value  # + Xent_loss
                if train_mode == 1:
                    total_loss=loss_value
                val_loss+=total_loss
                test_loss.append(total_loss.item())

                # Obtaining eval metrics
                if train_mode==0:
                    final_output = val_gaze_heatmap_pred.cpu().data.numpy()
                    target = gaze
                    eye_position = eye_position

                    for f_point, gt_point, eye_point in \
                            zip(final_output, target, eye_position):
                        out_size = 64  # Size of heatmap for chong output
                        heatmap = np.copy(f_point)
                        f_point = f_point.reshape([out_size, out_size])

                        h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
                        f_point = np.array([w_index / out_size, h_index / out_size])
                        f_error = f_point - gt_point
                        f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

                        # angle
                        f_direction = f_point - eye_point
                        gt_direction = gt_point - eye_point

                        norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
                        norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

                        f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
                                    (norm_gt * norm_f + 1e-6)
                        f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
                        f_angle = np.arccos(f_cos_sim) * 180 / np.pi

                        # AUC calculation
                        heatmap = np.squeeze(heatmap)
                        heatmap = cv2.resize(heatmap, (5, 5))
                        gt_heatmap = np.zeros((5, 5))
                        x, y = list(map(int, gt_point * 5))
                        gt_heatmap[y, x] = 1.0

                        all_gazepoints.append(f_point)
                        all_predmap.append(heatmap)
                        all_gtmap.append(gt_heatmap)
                        # score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), heatmap.reshape([-1]))

                        total_error.append([f_dist, f_angle])
            pbar.set_postfix(**{'va_loss': val_loss / (iteration + 1)})
            pbar.update(1)
        if train_mode==0:
            l2, ang = np.mean(np.array(total_error), axis=0)
            all_gazepoints = np.vstack(all_gazepoints)
            all_predmap = np.stack(all_predmap).reshape([-1])
            all_gtmap = np.stack(all_gtmap).reshape([-1])
            auc = roc_auc_score(all_gtmap, all_predmap)
            print(auc, l2, ang)
            auc1 = round(auc, 4)
            l21 = round(l2, 4)
            ang = round(ang, 4)
            data_list=[auc1,l21,ang]
            f = open('logsSynth/performance.csv', 'a')
            writer=csv.writer(f)
            writer.writerow(data_list)
    if train_mode==0:
        torch.save(model.state_dict(),
                   os.path.abspath(os.path.join(os.getcwd(),'../..'))+'data/logsSynth/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
    if train_mode == 1:
        torch.save(model.state_dict(),
                   os.path.abspath(os.path.join(os.getcwd(),'../..'))+'data/logsReal/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                   epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im, onehot_im.size), np.reshape(heatmap, heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_heatmap_loss_by_gtbox_predheatmap(box, heatmap):
    '''
    Use ground truth box and predicted heatmap to compute the energy aggregation loss
    '''
    batch_size = heatmap.size()[0]
    power, total_power, total_eng_loss= 0., 0., 0.
    for i in range(batch_size):
        cur_box = box[i]
        cur_heatmap = heatmap[i]
        xmin, ymin, xmax, ymax = int(cur_box[0] / 224 * 64), int(cur_box[1] / 224 * 64), int(cur_box[2] / 224 * 64), int(cur_box[3] / 224 * 64)
        power = torch.sum(cur_heatmap[xmin: xmax + 1, ymin: ymax + 1])
        total_power = cur_heatmap.sum()
        eng_loss = 1 - (power / total_power)
        total_eng_loss = total_eng_loss + eng_loss
    return 10 * (total_eng_loss / batch_size)
