# coding: utf-8

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
from PIL import Image
import cv2

import time


#
# 以下CUDAカーネル
#

mod = SourceModule("""

#define M_PI 3.141592654f

__global__ void get_diff(unsigned char *image, float *delta){

    int idx = threadIdx.x + blockDim.x* blockIdx.x;
    if(threadIdx.x < blockDim.x-4)
    {
        delta[idx] = (int(image[idx+4]) - int(image[idx]))/4.0;
    }
    else
    {
        delta[idx] = delta[idx-4];
    }
}

__global__ void get_moire_images(unsigned char *phase,
                      unsigned char *image,
                      float *delta,
                      int width, int height){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = width*blockIdx.x;

    int x;
    float s;
    int p;
    int page=0;

    for(int n=0;n<4;n++){
        if(threadIdx.x >= n){
            x = (threadIdx.x-n) % 4;
            s = delta[idx-x];
            p = image[idx-x] + x*s;
        }else{
            p = image[idy+n] - (n-threadIdx.x)*delta[idy+n];
        }
        if(p<0){
            p=0;
        }else if(p>255){
            p=255;
        }
        phase[page+idx] = p;
        page += width*height;
    }
}

__global__ void phaze_shift(unsigned char *phase,
                      unsigned char *output,
                      int width, int height){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float denominator = 0;
    float numerator = 0;
    int page = 0;

    for(int n=0;n<4;n++){
        denominator += phase[page+idx] * cos(2.0 * M_PI * n / 4.0);
        numerator += phase[page+idx] * sin(2.0 * M_PI * n / 4.0);
        page += width*height;
    }

    output[idx] = (int)((atan(numerator/denominator)/M_PI+0.5)*255);

}
""")

def GetMoireInputImage(ColorImage):
    """
    引数のColorImageをグレースケールに変換し、格子ピッチがN画素となるようにダウンサンプリングする。
    numpy.ndarray ColorImage : 処理したいカラー画像
    return : 格子ピッチに応じてダウンサンプリングされたグレースケール画像
    """

    grayImage = cv2.cvtColor(ColorImage, cv2.COLOR_BGR2GRAY)

    SamplingRate = 37 / 4
    #格子ピッチに応じてSamplingRateを決定


    #格子の1ピッチがN画素となるようにダウンサンプリング
    #SamplingImage = cv2.resize(grayImage, size, interpolation = cv2.INTER_LINEAR)
    SamplingImage = cv2.resize(grayImage, None, fx = 1 / SamplingRate, fy = 1 / SamplingRate,  interpolation = cv2.INTER_LINEAR)

    return SamplingImage

#
# 画像の読みこみ
#

img = cv2.imread('sample169-5.jpg')
image = GetMoireInputImage(img)

pil_output_image = Image.fromarray(image)
pil_output_image.save("input.png")

height, width = image.shape[:2]

print("height = ", height)
print("width  = ", width)

#
# CUDA周りの準備
#

# GPUメモリの確保
d_image = cuda.mem_alloc(image.nbytes)
d_delta = cuda.mem_alloc(image.nbytes*4)
d_phase = cuda.mem_alloc(image.nbytes*4)

# output画像の他に位相をずらした画像も確認したいので、ホストメモリを確保
phase = np.zeros((4,height,width), dtype=np.uint8)
output = np.zeros((height,width), dtype=np.uint8)

# 上で定義したCUDAのカーネル関数をPython側で呼び出す準備
get_diff = mod.get_function("get_diff")
get_moire_images = mod.get_function("get_moire_images")
phaze_shift = mod.get_function("phaze_shift")

# GPUのグリッド・ブロックサイズの設定
block   = (width,1,1)
grid    = (height,1,1)

# input画像をGPUメモリへ転送
cuda.memcpy_htod(d_image,image)

# deltaを求める
get_diff(d_image,
     d_delta,
     block=block, grid=grid)

# 間引き画像を求める
get_moire_images(d_phase,
      d_image,
      d_delta,
      np.int32(width),
      np.int32(height),
      block=block, grid=grid)

# 位相をずらした画像をGPUからホストメモリに転送
cuda.memcpy_dtoh(phase,d_phase)

# 位相シフト法
phaze_shift(d_phase,
      cuda.Out(output),
      np.int32(width),
      np.int32(height),
      block=block, grid=grid)
#
# 処理後の画像を保存
#
pil_output_image = Image.fromarray(image)
pil_output_image.save("input.png")
pil_output_image = Image.fromarray(phase[0,:,:].reshape(height,width))
pil_output_image.save("phase00.png")
pil_output_image = Image.fromarray(phase[1,:,:].reshape(height,width))
pil_output_image.save("phase01.png")
pil_output_image = Image.fromarray(phase[2,:,:].reshape(height,width))
pil_output_image.save("phase02.png")
pil_output_image = Image.fromarray(phase[3,:,:].reshape(height,width))
pil_output_image.save("phase03.png")
pil_output_image = Image.fromarray(np.uint8(output))
pil_output_image.save("output.png")
