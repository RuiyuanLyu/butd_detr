from utils.euler_util import bbox_to_corners, euler_iou3d, chamfer_distance
import pickle

with open('debug1.pkl','rb') as f:
    a = pickle.load(f)
out_bbox = a['out_bbox']
tgt_bbox = a['tgt_bbox']
out_corners = a['out_cor'].cuda()
tgt_corners = a['tgt_cor'].cuda()
# all_iou3d = euler_iou3d(out_corners, tgt_corners)
# print(all_iou3d)

N = out_corners.shape[0]
M = tgt_corners.shape[0]
for i in range(788,789):
    for j in range(19,20):
        pred_corners = out_corners[i:i+1]
        tar_corners = tgt_corners[j:j+1]
        print(i,j)
        print(out_bbox[i])
        print(tgt_bbox[j])
        iou3d = euler_iou3d(pred_corners, tar_corners)
        print(iou3d)
        print('=====================')