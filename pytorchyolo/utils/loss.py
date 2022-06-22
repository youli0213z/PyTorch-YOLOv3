import math

import torch
import torch.nn as nn

from .utils import to_cpu

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    #box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
        b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
        b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
        b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def compute_grid_offsets(self, grid_size, cuda=True):
    g = self.grid_size
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    stride = self.img_dim / self.grid_size
    # Calculate offsets for each grid
    grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
    grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
    self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
    self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
    self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

def _compute_loss(predictions, targets, model,ignore_thres=0.5):
    loss_f = nn.BCELoss()
    FloatTensor = torch.cuda.FloatTensor if targets.is_cuda else torch.FloatTensor
    # LongTensor = torch.cuda.LongTensor if targets.is_cuda else torch.LongTensor
    # ByteTensor = torch.cuda.ByteTensor if targets.is_cuda else torch.ByteTensor
    stride_list = [model.yolo_layers[i].stride for i in range(3)]
    anchors_list = [model.yolo_layers[i].anchors for i in range(3)]
    num_classes = model.yolo_layers[0].num_classes
    lcls, lbox, lobj = torch.zeros(1, device=targets.device), torch.zeros(1, device=targets.device), torch.zeros(1, device=targets.device)
    for layer_index in range(3):
        layer_predictions = predictions[layer_index]
        stride = stride_list[layer_index]
        anchors = anchors_list[layer_index]
        grid = layer_predictions.shape[3]
        num_samples = layer_predictions.size(0)
        x = torch.sigmoid(layer_predictions[..., 0])  #Center x
        y = torch.sigmoid(layer_predictions[..., 1])  # Center y
        w = layer_predictions[..., 2]  # Width
        h = layer_predictions[..., 3]  # Height
        pred_conf = torch.sigmoid(layer_predictions[..., 4])  # Conf
        pred_cls = torch.sigmoid(layer_predictions[..., 5:])  # Cls pred.
        grid_x = torch.arange(grid).repeat(grid, 1).view([1, 1, grid, grid]).type(FloatTensor)
        grid_y = torch.arange(grid).repeat(grid, 1).t().view([1, 1, grid, grid]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, -1, 1, 1))
        grid_size = layer_predictions.size(2)
        pred_boxes = FloatTensor(layer_predictions[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * stride,  # 还原到原始图中
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, num_classes),
            ),
            -1,
        )
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = _build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=scaled_anchors,
            ignore_thres=ignore_thres,
        )
        lbox +=  iou_scores
        loss_conf_obj = loss_f(pred_conf*obj_mask, tconf*obj_mask)
        loss_conf_noobj = loss_f(pred_conf*noobj_mask, tconf*noobj_mask)
        _obj_mask = obj_mask.unsqueeze(-1).repeat(1,1,1,1,num_classes)
        _noobj_mask = noobj_mask.unsqueeze(-1).repeat(1, 1, 1,1, num_classes)
        lobj += 1 * loss_conf_obj + 100 * loss_conf_noobj  # 有物体越接近1越好 没物体的越接近0越好
        lcls += loss_f(pred_cls*_obj_mask, tcls*_obj_mask)  # 分类损失
    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))



def compute_loss(predictions, targets, model):
    # Check which device was used
    device = targets.device

    # Add placeholder varables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    for i, yolo_layer in enumerate(model.yolo_layers):
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        anchors = yolo_layer.anchors / yolo_layer.stride
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        t = targets * gain
        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j]
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long()
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long()
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # Add correct anchor for each target to the list
        anch.append(anchors[a])
        # Add class for each target to the list
        tcls.append(c)

    return tcls, tbox, indices, anch

def bbox_wh_iou(wh1, wh2,eps = 1e-16):
    #wh2 = wh2.t()
    b = wh2.shape[0]
    w1, h1 = wh1[0].repeat(b), wh1[1].repeat(b)
    w2, h2 = wh2[:,0], wh2[:,1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 ) + w2 * h2 - inter_area + eps
    return inter_area / union_area

def _build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # batchsieze 4
    nA = pred_boxes.size(1) # 每个格子对应了多少个anchor
    nC = pred_cls.size(-1)  # 类别的数量
    nG = pred_boxes.size(2) # gridsize

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)  # obj，anchor包含物体, 即为1，默认为0 考虑前景
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) # noobj, anchor不包含物体, 则为1，默认为1 考虑背景
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0) # 类别掩膜，类别预测正确即为1，默认全为0
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0) # 预测框与真实框的iou得分
    tx = FloatTensor(nB, nA, nG, nG).fill_(0) # 真实框相对于网格的位置
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG #target中的xywh都是0-1的，可以得到其在当前gridsize上的xywh
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) #每一种规格的anchor跟每个标签上的框的IOU得分
    #print (ious.shape)
    best_ious, best_n = torch.max(ious,dim=0) # 得到其最高分以及哪种规格框和当前目标最相似
    # Separate target values
    b, target_labels = target[:, :2].long().t() # 真实框所对应的batch，以及每个框所代表的实际类别
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() #位置信息，向下取整了
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1 # 实际包含物体的设置成1
    noobj_mask[b, best_n, gj, gi] = 0 # 相反

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()): # IOU超过了指定的阈值就相当于有物体了
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gi # 根据真实框所在位置，得到其相当于网络的位置
    ty[b, best_n, gj, gi] = gy - gj
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1 #将真实框的标签转换为one-hot编码形式
    # Compute label correctness and iou at best anchor 计算预测的和真实一样的索引
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    #iou_scores[b, best_n, gj, gi]
    iou_scores = torch.mean(bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)) #与真实框想匹配的预测框之间的iou值

    tconf = obj_mask.float() # 真实框的置信度，也就是1
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
