import os
import numpy as np
import cv2
from main_utils.flow_utils import read_16_bit_flow

left_t0 = '/media/faleotti/fe6638e6-0a75-42d0-8fb2-6d03546dec8e/ComputerVision/Dataset/KITTI/2015/training/image_2/000091_10.png'
left_t1 = left_t0.replace('_10', '_11')
flow = left_t0.replace('image_2','flow_occ')


for x in [left_t0, left_t1, flow]:
    assert os.path.exists(x)

left_t0 = cv2.cvtColor(cv2.imread(left_t0), cv2.COLOR_BGR2RGB)
left_t1 = cv2.cvtColor(cv2.imread(left_t1), cv2.COLOR_BGR2RGB)
flow = read_16_bit_flow(flow)

valid_mask = flow[:,:,2:3] > 0
left_t0_valid = np.where(valid_mask>0, left_t0, np.zeros_like(left_t0))

cv2.imwrite('results/pixel_with_flow.png', cv2.cvtColor(left_t0_valid.astype(np.uint8),cv2.COLOR_BGR2RGB))
cv2.imshow('valid points t0', left_t0_valid)
cv2.waitKey(0)
assert valid_mask[340,270,0] > 0 
assert valid_mask[276,879,0] > 0
k = 10

selected_points = np.zeros_like(left_t0)
selected_points[340-k:340+k,270-k:270+k,:] = left_t0[340-k:340+k,270-k:270+k,:]
selected_points[276-k:276+k,879-k:879+k,:] = left_t0[276-k:276+k,879-k:879+k,:]
selected_points[340,270,:] = [255,0,0]
selected_points[276,879,:] = [255,0,0]

flow = np.ceil(flow).astype(np.int32)
warped_p0 = [flow[340,270,1] + 340, flow[340,270,0] + 270]
warped_p1 = [flow[276,879,1] + 276, flow[276,879,0] + 879]

print('P0(x,y): ({},{}) -> ({},{})  flow: {} u, {} v'.format(270,340,warped_p0[1], warped_p0[0], flow[340,270,0],flow[340,270,1]))
print('P1(x,y): ({},{}) -> ({},{})  flow: {} u, {} v'.format(879,276,warped_p1[1], warped_p1[0], flow[276,879,0], flow[276,879,1]))

selected_points_t1 = np.zeros_like(left_t1)
selected_points_t1[warped_p0[0]-k:warped_p0[0]+k,warped_p0[1]-k:warped_p0[1]+k,:] = left_t1[warped_p0[0]-k:warped_p0[0]+k,warped_p0[1]-k:warped_p0[1]+k,:]
selected_points_t1[warped_p1[0]-k:warped_p1[0]+k,warped_p1[1]-k:warped_p1[1]+k,:] = left_t1[warped_p1[0]-k:warped_p1[0]+k,warped_p1[1]-k:warped_p1[1]+k,:]

cv2.imwrite('results/selected_points.png', cv2.cvtColor(selected_points,cv2.COLOR_RGB2BGR))
cv2.imwrite('results/selected_points_t1.png', cv2.cvtColor(selected_points_t1,cv2.COLOR_RGB2BGR))