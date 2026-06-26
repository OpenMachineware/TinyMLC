#include "tinymlc.h"


static int max_int(int a, int b) {
    return (a > b) ? a : b;
}

static int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static int compute_iou_q7(const Box* a, const Box* b) {
    // All coordinates are Q7 fixed-point (1/128 precision)
    int x1 = max_int(a->x1, b->x1);
    int y1 = max_int(a->y1, b->y1);
    int x2 = min_int(a->x2, b->x2);
    int y2 = min_int(a->y2, b->y2);

    if (x2 <= x1 || y2 <= y1) {
        return 0;  // No overlap
    }

    int inter_w = x2 - x1;
    int inter_h = y2 - y1;
    int inter = inter_w * inter_h;

    int area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    int area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

    // iou = inter / (area_a + area_b - inter)
    // Return as Q7 fixed-point (multiply by 128 to avoid float)
    int denom = area_a + area_b - inter;
    if (denom == 0) return 0;

    int iou_q7 = (inter * 128) / denom;
    return iou_q7;
}

int tmlc_nms(Box* boxes, int num_boxes,
             int iou_threshold_q7, int max_output_size) {
    if (num_boxes == 0) return 0;

    // Mark all boxes as keep
    for (int i = 0; i < num_boxes; i++) {
        boxes[i].keep = 1;
    }

    // Sort by score descending (bubble sort)
    for (int i = 0; i < num_boxes - 1; i++) {
        for (int j = i + 1; j < num_boxes; j++) {
            if (boxes[j].score > boxes[i].score) {
                Box tmp = boxes[i];
                boxes[i] = boxes[j];
                boxes[j] = tmp;
            }
        }
    }

    int keep_count = 0;

    for (int i = 0; i < num_boxes && keep_count < max_output_size; i++) {
        if (!boxes[i].keep) continue;

        keep_count++;

        for (int j = i + 1; j < num_boxes; j++) {
            if (!boxes[j].keep) continue;

            int iou = compute_iou_q7(&boxes[i], &boxes[j]);

            if (iou > iou_threshold_q7) {
                boxes[j].keep = 0;
            }
        }
    }

    return keep_count;
}
