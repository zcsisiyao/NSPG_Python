import numpy as np
import matplotlib.pyplot as plt

def pixel_location(x, y, h, w, ps):
    """
    Create patch Indexes.
    """

    x_location = list(range(x-ps, x+ps+1))
    y_location = list(range(y-ps, y+ps+1))
    if x-ps < 0:
        x_location = list(range(0, 1+2*ps))
    if y-ps < 0:
        y_location = list(range(0, 1+2*ps))
    if x+ps > h-1:
        x_location = list(range(h-2*ps-1, h))
    if y+ps > w-1:
        y_location = list(range(w-2*ps-1, w))
    return x_location, y_location

def find_centre(i, j, ws, ps, ds, image):
    """
    find the centre for all candidate neighbors.
    """
    i += 1
    j += 1
    h, w, _ = image.shape
    t = np.floor((ws - ps)/ds)
    if i - t*ds - ps > 0 and i + t * ds + ps < h + 1:
        x = list(range(int(i-t*ds), int(i+t*ds+1), ds))
    elif i-t*ds-ps < 1:
        temp1 = np.floor((i-ps-1)/ds)
        x = list(range(int(i-temp1*ds), int(i+t*ds+1), ds))
    elif i+t*ds+ps > h:
        temp2 = np.floor((h-i-ps)/ds)
        x = list(range(int(i-t*ds), int(i+temp2*ds+1), ds))
    else:
        x = i

    if j - t*ds - ps > 0 and j + t * ds + ps < w + 1:
        y = list(range(int(j-t*ds), int(j+t*ds+1), ds))
    elif j-t*ds-ps-1 < 1:
        temp1 = np.floor((j-ps-1)/ds)
        y = list(range(int(j-temp1*ds), int(j+t*ds+1), ds))
    elif j+t*ds+ps > w:
        temp2 = np.floor((w-j-ps)/ds)
        y = list(range(int(j-t*ds), int(j+temp2*ds+1), ds))
    else:
        y = j
    x = [i-1 for i in x]
    y = [i-1 for i in y]
    return x, y

def distancel2(patchx, patchy):
    """
    Calculate patch similarity by Euclidean distance.
    """
    _1, _2, c = patchx.shape
    dx = np.reshape(patchx, -1)
    dy = np.reshape(patchy, -1)
    dis_xy = np.linalg.norm(dx - dy, ord=2) ** 2
    return dis_xy / c

def local_position(i, ps, dp, h):
    """
    Move the patch position index to the difference image position index.
    """
    if i < ps:
        local_i = list(range(0, int(np.floor((i+ps)/dp) * dp), dp))
    elif i > h-ps-1:
        local_i = list(range(int(np.ceil((i-ps)/dp)*dp), int(np.floor(h/dp)*dp), dp))
    else:
        local_i = list(range(int(np.ceil((i-ps)/dp)*dp), int(np.floor((i+ps)/dp)*dp), dp))
    return local_i

def fuse_DI(fw, bw):
    """
    Fuse difference images.
    """
    fusion = (fw + bw) / 2
    return fusion

def NPSG(image1, image2, ws, ps, ds, dp, k):
    """
    Generate forward and reverse difference images.
    """
    h, w, _ = image1.shape
    f_x_distance = np.zeros((h, w))
    f_y_distance = np.zeros((h, w))
    for i in range(0, h, 3):
        for j in range(0, w, 3):
            x_location, y_location = pixel_location(i, j, h, w, ps=ps)
            patchx0 = image1[x_location[0]:x_location[-1]+1, y_location[0]:y_location[-1]+1, :]
            patchy0 = image2[x_location[0]:x_location[-1]+1, y_location[0]:y_location[-1]+1, :]
            centrex, centrey = find_centre(i, j, ws=ws, ps=ps, ds=ds, image=image1)
            disfort1 = np.zeros((len(centrex) * len(centrey), 1))
            disfort2 = np.zeros((len(centrex) * len(centrey), 1))
            t = 0
            for tx in range(len(centrex)):
                for ty in range(len(centrey)):
                    x = list(range(centrex[tx] - ps, centrex[tx] + ps + 1))
                    y = list(range(centrey[ty] - ps, centrey[ty] + ps + 1))
                    patcht1 = image1[x[0]:x[-1]+1, y[0]:y[-1]+1, :]
                    patcht2 = image2[x[0]:x[-1]+1, y[0]:y[-1]+1, :]
                    disfort1[t] = distancel2(patchx0, patcht1)
                    disfort2[t] = distancel2(patchy0, patcht2)
                    t += 1
            k = min(k, t)
            sorted_nums1 = sorted(enumerate(disfort1), key=lambda x111: x111[1])
            sorted_nums2 = sorted(enumerate(disfort2), key=lambda x222: x222[1])
            rankt1 = [i[0] for i in sorted_nums1]
            valuet1 = [i[1] for i in sorted_nums1]
            rankt2 = [i[0] for i in sorted_nums2]
            valuet2 = [i[1] for i in sorted_nums2]
            f_x_distance[i, j] = abs(np.mean(disfort1[rankt2[:k+1]]) - np.mean(valuet1[:k+1]))
            f_y_distance[i, j] = abs(np.mean(disfort2[rankt1[:k + 1]]) - np.mean(valuet2[:k + 1]))
    print("-------Finshed!!!!!!---------")
    DI_fw = np.zeros((h, w))
    DI_bw = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            local_i = local_position(i, ps=ps, dp=dp, h=h)
            local_j = local_position(j, ps=ps, dp=dp, h=w)
            DI_fw[i, j] = np.mean(np.mean(f_y_distance[local_i, local_j]))
            DI_bw[i, j] = np.mean(np.mean(f_x_distance[local_i, local_j]))
    return DI_fw, DI_bw

def main():
    """
    x: pre change image
    y: post change image
    gt: ground truth
    ws: window size (default: 50)
    ps: patch size 2 * ps +1 (default: 2)
    ds: (default: 5)
    dp: p_s <= dp <= 2 * p_s + 1 (default: 3)
    k: k neighbor (default: 35)
    """
    ws, ps, ds, dp, k = 50, 2, 5, 3, 35
    x, y, gt = heter()
    x, y = x.numpy(), y.numpy()
    fw, bw = NPSG(x, y, ws, ps, ds, dp, k)
    fusion = fuse_DI(fw, bw)
    plt.imshow(fusion, cmap='gray')
    plt.show()

if __name__=="__main__":
    main()

