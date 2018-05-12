from moviepy.editor import VideoFileClip
import numpy as np
v = VideoFileClip("VideoInfrared.avi")
print(v.size)
(im_width, im_height) = v.size
print np.array(v)
#print np.array(v).reshape((im_height, im_width, 3)).astype(np.uint8)