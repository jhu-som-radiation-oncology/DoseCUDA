#ifndef CUDA_CLASSES_H
#define CUDA_CLASSES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <unistd.h>
#include <math_constants.h>
#include "./BaseClasses.h"


static inline void cuda_check(cudaError_t err)
{
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(err) cuda_check(err)


class CudaBeam : public BeamClass{

    public:

        __host__ CudaBeam(BeamClass * h_beam);

        __device__ void unitVectorToSource(const PointXYZ * point_xyz, PointXYZ * uvec);

};

class CudaDose : public DoseClass{

    public:

        __host__ CudaDose(DoseClass * h_dose);

        __device__ bool pointIJKWithinImage(const PointIJK * point_ijk);

        __device__ unsigned int pointIJKtoIndex(const PointIJK * point_ijk);

        __device__ void pointIJKtoXYZ(const PointIJK * point_ijk, PointXYZ * point_xyz, BeamClass * beam);

        __device__ void pointXYZtoIJK(const PointXYZ * point_xyz, PointIJK * point_ijk, BeamClass * beam);

        __device__ void pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head, BeamClass * beam);

        __device__ void pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img, BeamClass * beam);

        __device__ void pointXYZClosestCAXPoint(const PointXYZ * point_xyz, PointXYZ * point_cax, BeamClass * beam);

        __device__ float pointXYZDistanceToCAX(const PointXYZ * point_head_xyz);

        __device__ float pointXYZDistanceToSource(const PointXYZ * point_img_xyz, BeamClass * beam);

};

#endif