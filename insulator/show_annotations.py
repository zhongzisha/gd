import sys,os,glob,shutil
import numpy as np
from shapely import Polygon

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

def showBBox(self, anns, label_box=True):
    """
    show bounding box of annotations or predictions
    anns: loadAnns() annotations or predictions subject to coco results format
    label_box: show background of category labels or not
    """
    if len(anns) == 0:
        return 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    image2color = dict()
    for cat in self.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
    for ann in anns:
        c = image2color[ann['category_id']]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)
        # option for dash-line
        # ax.add_patch(Polygon(np_poly, linestyle='--', facecolor='none', edgecolor=c, linewidth=2))
        if label_box:
            label_bbox = dict(facecolor=c)
        else:
            label_bbox = None
        if 'score' in ann:
            ax.text(bbox_x, bbox_y, '%s: %.2f' % (self.loadCats(ann['category_id'])[0]['name'], ann['score']),
                    color='white', bbox=label_bbox)
        else:
            ax.text(bbox_x, bbox_y, '%s' % (self.loadCats(ann['category_id'])[0]['name']), color='white',
                    bbox=label_bbox)
    # option for filling bounding box
    # p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    # ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)