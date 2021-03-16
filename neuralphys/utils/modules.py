"""Module specific functions for data loader."""
import pdb
import numpy as np
from neuralphys.utils.config import _C as C
from neuralphys.utils.bbox import bb_intersection_over_union, display_img_with_bb
import matplotlib.pyplot as plt
import itertools

mapping = {'ball': [1, 0, 0, 0], 'bar': [0, 1, 0, 0], 'jar': [0, 0, 1, 0], 'stick': [0, 0, 0, 1]}
NUM_OBJS = 6    # temporary - replace to C.RIN.NUM_OBJS

dual_condition = True    # promised to be a temporary flag (dual_condition=False implies 'loose' data; whereas True implies 'constraint' data)


def check_overlap(img, bxA, bxB, thresh, t, i, j, display=False):
    # draw a border of (thresh/2) for both boxes
    bord = np.array([-1, -1, 1, 1]) * thresh/2
    boxA = bxA + bord
    boxB = bxB + bord
    boxA[boxA<0], boxA[boxA>128] = 0, 128
    boxB[boxB<0], boxB[boxB>128] = 0, 128

    if display:
        display_img_with_bb(np.transpose(img, (1, 2, 0)), boxA, boxB, 'a%d_%d_%d.png' % (t,i,j))

    # get iou
    iou = bb_intersection_over_union(boxA, boxB)
    if iou > 0:
        # touching object
        # print("t=%d, i=%d, j=%d, touch" % (t, i, j))
        return 1
    #print("t=%d, i=%d, j=%d, no-touch" % (t, i, j))
    return 0


def singular_module(imgs, boxes, valid, shapes, dynamics, thresh, input_size, mtype):
    """Return object-specific validity mask
        based on singular module type `mtype`.
    """
    nooverlap = np.zeros(NUM_OBJS, dtype=int)

    # object specific
    mbool = mapping[mtype]

    # if no relevant objects exist, return
    if mbool not in shapes.tolist():
        return nooverlap

    # get obj indices (which are obviously valid)
    curr_obj_indices = np.where((shapes == mbool).all(axis=1))[0].tolist()
    dynamic_indices = np.where(dynamics == 1)[0].tolist()
    dynamic_curr_obj_indices = list(set(curr_obj_indices) & set(dynamic_indices))
    valid_indices = np.where(valid == 1)[0].tolist()

    # for each dynamic object get whether condition is satisfied
    for i in dynamic_curr_obj_indices:
        # for all valid objects besides that at index i, check singular condition
        touch = 0
        for t in range(input_size):
            for j in valid_indices:
                if i == j: continue    # skip same object

                touch = check_overlap(imgs[t], boxes[t][i], boxes[t][j], thresh, t, i, j)

                if touch == 1: break    # overlap has happened for object i; no point checking objects j
            if touch == 1: break    # overlap has happened at this timestep t for object i; no point checking other times
        if touch == 0: nooverlap[i] = 1    # overlap has not happened with any object j at any timestep t

    # # testing - display images
    # for i, img in enumerate(imgs):
    #     plt.imshow(np.transpose(img, (1,2,0)))
    #     plt.savefig(str(i) + '.png')
    #     plt.close()
    # print("Current image displayed")
    # pdb.set_trace()

    return nooverlap


def dual_module(imgs, boxes, valid, shapes, dynamics, thresh, input_size, mtype1, mtype2):
    """Return object-specific validity mask
        based on dual module type `mtype1` and `mtype2`.
        If there is any case where 2 objects touch each other only
        (and no other object - either in the same frame or a different one)
        then that is positive for those 2 objects
    """
    nooverlap = np.zeros(NUM_OBJS, dtype=int)

    # object specific
    mbool1 = mapping[mtype1]
    mbool2 = mapping[mtype2]

    # if no relevant objects exist, return
    if mbool1 not in shapes.tolist() or mbool2 not in shapes.tolist():
        return nooverlap

    # get obj indices (which are obviously valid)
    dynamic_indices = np.where(dynamics == 1)[0].tolist()
    curr_obj_indices1 = np.where((shapes == mbool1).all(axis=1))[0].tolist()
    curr_obj_indices2 = np.where((shapes == mbool2).all(axis=1))[0].tolist()
    valid_indices = np.where(valid == 1)[0].tolist()

    valid_curr_obj_indices1 = list(set(curr_obj_indices1) & set(valid_indices))
    valid_curr_obj_indices2 = list(set(curr_obj_indices2) & set(valid_indices))

    # now take pairs of valid objects & remove duplicates (permutations)
    paired_list = list(itertools.product(valid_curr_obj_indices1, valid_curr_obj_indices2))
    paired_list = list(set(tuple(sorted(t)) for t in paired_list))

    # print("paired list: ", paired_list)

    # further filter indices based on IOU
    for [i, j] in paired_list:
        # print("checking pair: (%d, %d)" % (i, j))
        # for all valid objects besides that at index i, check dual condition
        touch = 0
        for t in range(input_size):
            if i == j or (i not in dynamic_indices and j not in dynamic_indices): continue    # skip if both are same object or if both are static
            touch = check_overlap(imgs[t], boxes[t][i], boxes[t][j], thresh, t, i, j)
            if touch == 1: break    # overlap has happened for these 2 objects; no need to check other times
        if touch == 0: continue

        if not dual_condition:
            nooverlap[i] = 1
            nooverlap[j] = 1
        else:
            # now objects (i and j) have touched at least once
            # check for no overlap between (i, k) and (j, k)
            # where k is a valid other object for all timesteps
            valid_other_indices = list(set(valid_indices) - set([i, j]))

            # print("cond1 done; checking cond2 for (%d, %d)" % (i, j))

            for obj_idx in [i, j]:
                touch = 0
                for k in valid_other_indices:
                    for t in range(input_size):
                        touch = check_overlap(imgs[t], boxes[t][obj_idx], boxes[t][k], thresh, t, obj_idx, k)
                        if touch == 1: break
                    if touch == 1: break
                if touch == 0: nooverlap[obj_idx] = 1

    return nooverlap


def module_mask(module_name, imgs, valid, shapes, dynamics, boxes, thresh, input_size):
    """Return boolean array of size (times, num_objs) ---> (1, 6)
        indicating that object at that timestep should be considered in loss or not
        based on module condition.
       e.g.,
        (1) 'ball' module:
            --if there is no ball, set all values as False and return; else
            --for each timestep, check whether ball is not close to other obj - if so, True, else False.
        (2) 'ball-ball' module:
            --if there is are not 2 ball, set all values as False and return; else
            --for each timestep, check whether ball is close to another ball - if so True, else False.
    """
    items = module_name.split('-')
    if len(items) == 1:
        check = singular_module(imgs, boxes, valid, shapes, dynamics, thresh, input_size, module_name)
    else:
        check = dual_module(imgs, boxes, valid, shapes, dynamics, thresh, input_size, items[0], items[1])
    return check
