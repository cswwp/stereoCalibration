import numpy as np
import cv2

def rect_contains(r, pt):
	x = pt[0]
	y = pt[1]

	if x > r[0] and x < r[2] and y >r[1] and y<r[3]:
		return True
	else:
		return False


def draw_delaunay(img, subdiv, delaunay_color):
	triangleList = subdiv.getTriangleList();  # do Delaunay Triangulation
	size = img.shape
	print('fdsafdsafsa:', size)
	r = (0, 0, size[1], size[0])
	for t in triangleList:
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		#print(pt1, pst2, pt3)
		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
			cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
	cv2.imshow('img', img)
	cv2.waitKey(0)

def read_txt(txt_path):
	cors = [row.strip() for row in open(txt_path, 'r').readlines()]
	cors = [(float(cor.split(' ')[0]), float(cor.split(' ')[1])) for cor in cors]
	cors = [(int(cor[0]), int(cor[1])) for cor in cors]
	#cors = np.asarry(cors, dtype=np.uint8)
	return cors


def get3d(l_path, r_path):
	l_cors = read_txt(l_path)
	r_cors = read_txt(r_path)
	res = []
	for i in range(len(l_cors)):
		# l_x = l_cors[0]
		# r_x = r_cors[0]
		# dep = l_x - r_x
		res.append((l_cors[i][0], l_cors[i][1], abs(l_cors[i][0] - r_cors[i][0])))
	return res


if __name__ == '__main__':
	img_path = "/Users/parker/work/stereoDepth/20191213/person/remap_l/001475.jpg"
	txt_path = "/Users/parker/work/stereoDepth/20191213/person/remap_l/001475.txt"

	l_path = "/Users/parker/work/stereoDepth/20191213/person/remap_l/001475.txt"
	r_path = "/Users/parker/work/stereoDepth/20191213/person/remap_r/001475.txt"
	img = cv2.imread(img_path)
	print(get3d(l_path, r_path))


	# size = img.shape
	# rect = (0, 0, size[1], size[0])
	# subdiv = cv2.Subdiv2D(rect)
	#
	# keypoints = read_txt(txt_path)
	# #keypoints = [(a[0], a[1]) for a in utils.get_facial_landmarks(img)]
	# for p in keypoints:
	# 	subdiv.insert(p)
	# draw_delaunay(img, subdiv, (255, 255, 255))
