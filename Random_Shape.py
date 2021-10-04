import numpy as np
import cv2
from scipy.special import binom

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]


def get_bezier_curve(a, rad=0.2, edgy=0):
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T

    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


def create_particle(img_size=50):
    rad = 0.2
    edgy = 0.05

    a = get_random_points(n=5, scale=1)
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    x = [int(v*img_size*0.7) for v in x]
    y = [int(v*img_size*0.7) for v in y]

    random_curve = [[i, v] for i, v in zip(x, y)]
    arr = np.random.randint(255, 256, size=(img_size, img_size, 4), dtype=np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            if [i, j] in random_curve:
                arr[i][j] = (0, 0, 0, 255)
            else:
                arr[i][j] = (255, 255, 255, 255)


    kernel = np.ones((3,3), np.uint8)
    erode = cv2.erode(arr, kernel, iterations = 1)

    erode_gray = cv2.cvtColor(erode.copy(), cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(erode_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
       filledContours = cv2.drawContours(erode.copy(), contours, 1, (0, 0, 0, 255), thickness=-1)
    except:
        print(img_size, x, y)

    return filledContours


def inverse_img(img):
    img_height, img_width, _ = img.shape
    # if four corners are all dark, inverse the image
    if np.all(img[0, 0][0:2] < 100) and np.all(img[img_height-1, 0][0:2] < 100)\
        and np.all(img[0, img_width-1][0:2] < 100) and np.all(img[img_height-1, img_width-1][0:2] < 100):
        for h in range(img_height):
            for w in range(img_width):
                img[h, w] = (255 - img[h, w][0], 255 - img[h, w][1], 255 - img[h, w][2], 255)

    return img