"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    # S => split size
    # B => number of prediction boxes in one grid cell
    # C => number of classes
    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.C = C
        self.B = B

        # how much we should pay loss for no object and box coordinates
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, preds, target):
        # predictions are at shape (BATCH_SIZE, S*S*(C+B*5))
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # iou for box1 and box2 with the targets
        iou_b1 = intersection_over_union(
            preds[..., 21:25], target[..., 21:25]
        )  # (batch, S, S)
        iou_b2 = intersection_over_union(preds[..., 26:30], target[..., 21:25])  # (batch, S, S)

        ious = torch.cat(
            [iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0
        )  # (2, batch, S, S)

        # iou_maxes => the actual max value (batch, S, S)
        # best_box => index of max box
        iou_maxes, best_box = torch.max(
            ious, dim=0
        )  # across the number of boxes which is 2 here

        # target[..., 20] this is the objectness aka the probablity that cell contains object
        # then adding extra dimension for broadcasting during loss computations
        exists_box = target[..., 20].unsqueeze(3)

        ### BOX COORDINATES LOSS ###
        box_predictions = exists_box * (
            (best_box * preds[..., 26:30] + (1 - best_box) * preds[..., 21:25])
        )

        box_targets = exists_box * target[..., 21:25]

        # take a sqrt for width and heigh of all boxes
        # sign to keep the original sign of values
        sign = torch.sign(box_predictions[..., 2:4])

        box_predictions[..., 2:4] = sign * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        ### OBJECT LOSS ###
        # 20:21 => confidence score for box 1
        # 25:26 => confidence score for box 2
        pred_box = best_box * preds[..., 25:26] + (1 - best_box) * preds[..., 20:21] 
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        ### NO OBJECT LOSS ###
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * preds[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * preds[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        ### CLASS LOSS ###
        
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        ### FINAL LOSS ###

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        ) 

        return loss
