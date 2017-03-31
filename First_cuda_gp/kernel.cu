
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "book.h"
#include "gridcheck.h"

# define Section 12  // number of cooling sections
# define CoolSection 8
# define MoldSection 4

float ccml[Section + 1] = { 0.0,0.2,0.4,0.6,0.8,1.0925,2.27,4.29,5.831,9.6065,13.6090,19.87014,28.599 }; // The cooling sections
float H_Init[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections
float H_Init_Temp[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections
float Taim[CoolSection] = { 966.149841, 925.864746, 952.322083, 932.175537, 914.607117, 890.494263, 870.804443, 890.595825 };
float *Mean_TSurface;

void Calculation_MeanTemperature(int nx, int ny, int nz, float dy, float *ccml, float *T);
cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, int num_blocks, int num_threadsx, int num_threadsy);
__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd);
__device__ float Boundary_Condition(int j, float dx, float *ccml_zone, float *H_Init);

__global__ void addKernel(float *T_New, float *T_Last, float *ccml, float *H_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, bool disout)
{
	int i = threadIdx.x;
	int m = threadIdx.y;
	int j = blockIdx.x;
	int idx = j * nx * nz + m * nx + i;
	int ND = nx * nz;
	int D = nx;

	float pho, Ce, lamd; // physical parameters pho represents desity, Ce is specific heat and lamd is thermal conductivity
	float a, T_Up, T_Down, T_Right, T_Left, T_Forw, T_Back, h = 100.0, Tw = 30.0, Vcast = -0.02, T_Cast = 1558.0;

	if (disout) {
		Physicial_Parameters(T_Last[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(j, dy, ccml, H_Init);
		if (j == 0) //1
		{
			T_New[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}

	else
	{
		Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(j, dy, ccml, H_Init);
		if (j == 0) //1
		{
			T_Last[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}
}

int main()
{
	const int nx = 21, ny = 3000, nz = 21;   // nx is the number of grid in x direction, ny is the number of grid in y direction.
	int num_blocks = 1, num_threadsx = 1, num_threadsy = 1;// num_threadsz = 1; // block number(1D)  thread number in x and y dimension(2D)
	int tnpts = 10001;  // time step
	float T_Cast = 1558.0, Lx = 0.25, Ly = 28.599, Lz = 0.25, t_final = 2000.0, dx, dy, dz, tao;  // T_Cast is the casting temperature Lx and Ly is the thick and length of steel billets
	float *T_Init;

	T_Init = (float *)calloc(nx * ny * nz, sizeof(float));  // Initial condition
	Mean_TSurface = (float*)calloc(Section, sizeof(float));

	num_threadsx = nx;
	num_threadsy = nz;
	num_blocks = ny;

	for (int m = 0; m < nz; m++)
		for (int j = 0; j < ny; j++)
	       for (int i = 0; i < nx; i++)
			   T_Init[nx * ny * m + j * nx + i] = T_Cast;  // give the initial condition

	dx = Lx / (nx - 1);            // the grid size x
	dy = Ly / (ny - 1);            // the grid size y
	dz = Lz / (nz - 1);            // the grid size y
	tao = t_final / (tnpts - 1);   // the time step size
	//gridcheck(dx, dy, tao);

	printf("Casting Temperature = %f ", T_Cast);
	printf("\n");
	printf("The thick of steel billets(m) = %f ", Lx);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Ly);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Lz);
	printf("\n");
	printf("dx(m) = %f ", dx);
	printf("dy(m) = %f ", dy);
	printf("dz(m) = %f ", dz);
	printf("tao(s) = %f ", tao);
	printf("\n");
	printf("simulation time(s) = %f\n ", t_final);

	clock_t timestart = clock();
	cudaError_t cudaStatus = addWithCuda(T_Init, dx, dy, dz, tao, nx, ny, nz, tnpts, num_blocks, num_threadsx, num_threadsy);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t timeend = clock();

	printf("running time = %d(millisecond)", (timeend - timestart));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, int num_blocks, int num_threadsx, int num_threadsy)
{
	cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8;
	float *dev_ccml; // the point on GPU
	float *dev_T_New, *dev_T_Last, *dev_H_Init;    // the point on GPU
	float *dev_T_New0, *dev_T_Last0, *dev_H_Init0; // the point on GPU
	float *dev_T_New1, *dev_T_Last1, *dev_H_Init1; // the point on GPU
	float *dev_T_New2, *dev_T_Last2, *dev_H_Init2; // the point on GPU
	float *dev_T_New3, *dev_T_Last3, *dev_H_Init3; // the point on GPU
	float *dev_T_New4, *dev_T_Last4, *dev_H_Init4; // the point on GPU
	float *dev_T_New5, *dev_T_Last5, *dev_H_Init5; // the point on GPU
	float *dev_T_New6, *dev_T_Last6, *dev_H_Init6; // the point on GPU
	float *dev_T_New7, *dev_T_Last7, *dev_H_Init7; // the point on GPU
	float *dev_T_New8, *dev_T_Last8, *dev_H_Init8; // the point on GPU
	float *T_Result, *T_Result0, *T_Result1, *T_Result2, *T_Result3, *T_Result4, *T_Result5, *T_Result6, *T_Result7, *T_Result8;
	float *H_Init_Temp, *H_Init_Temp0, *H_Init_Temp1, *H_Init_Temp2, *H_Init_Temp3, *H_Init_Temp4, *H_Init_Temp5, *H_Init_Temp6, *H_Init_Temp7, *H_Init_Temp8;

	float *Delta_H_Init, *T_HoldLast, **Mean_TSurfaceElement, **Mean_TSurfaceElementOne, **JacobianMatrix;
	float dh = 1.0, arf1, arf2, step = -0.0001;
	const int Num_Iter = 10;      // The result can be obtained by every Num_Iter time step
	const int PrintLabel = 0;     //  If PrintLabel=1 print the information else will not print the information
	volatile bool dstOut = true;
	FILE *fp = NULL;

	Delta_H_Init = (float*)calloc(CoolSection, sizeof(float));

	T_HoldLast = (float*)calloc(nz * ny * nx, sizeof(float));

	JacobianMatrix = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		JacobianMatrix[i] = (float*)calloc(CoolSection, sizeof(float));

	Mean_TSurfaceElement = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		Mean_TSurfaceElement[i] = (float*)calloc(CoolSection, sizeof(float));

	Mean_TSurfaceElementOne = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		Mean_TSurfaceElementOne[i] = (float*)calloc(CoolSection, sizeof(float));

	cudaError_t cudaStatus;

	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	HANDLE_ERROR(cudaStreamCreate(&stream2));
	HANDLE_ERROR(cudaStreamCreate(&stream3));
	HANDLE_ERROR(cudaStreamCreate(&stream4));
	HANDLE_ERROR(cudaStreamCreate(&stream5));
	HANDLE_ERROR(cudaStreamCreate(&stream6));
	HANDLE_ERROR(cudaStreamCreate(&stream7));
	HANDLE_ERROR(cudaStreamCreate(&stream8));

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR(cudaSetDevice(0));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result, nx * ny * nz* sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last, nx * ny * nz *  sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result0, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp0, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New0, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last0, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init0, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result1, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp1, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New1, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last1, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init1, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result2, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp2, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New2, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last2, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init2, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result3, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp3, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New3, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last3, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init3, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result4, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp4, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New4, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last4, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init4, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result5, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp5, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New5, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last5, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init5, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result6, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp6, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New6, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last6, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init6, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result7, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp7, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New7, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last7, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init7, Section * sizeof(float)));

	HANDLE_ERROR(cudaHostAlloc((void**)&T_Result8, nx * ny * nz * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&H_Init_Temp8, Section * sizeof(float), cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New8, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last8, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init8, Section * sizeof(float)));

	HANDLE_ERROR(cudaMalloc((void**)&dev_ccml, (Section + 1) * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_T_Last, T_Init, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_ccml, ccml, (Section + 1) * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init, Section * sizeof(float), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(num_threadsx, num_threadsy);

	for (int i = 0; i < tnpts; i++)
	{
		if (i % Num_Iter == 0)
		{
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp0[temp] = H_Init[temp];
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp1[temp] = H_Init[temp];
			H_Init_Temp1[0 + MoldSection] = H_Init[0 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp2[temp] = H_Init[temp];
			H_Init_Temp2[1 + MoldSection] = H_Init[1 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp3[temp] = H_Init[temp];
			H_Init_Temp3[2 + MoldSection] = H_Init[2 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp4[temp] = H_Init[temp];
			H_Init_Temp4[3 + MoldSection] = H_Init[3 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp5[temp] = H_Init[temp];
			H_Init_Temp5[4 + MoldSection] = H_Init[4 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp6[temp] = H_Init[temp];
			H_Init_Temp6[5 + MoldSection] = H_Init[5 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp7[temp] = H_Init[temp];
			H_Init_Temp7[6 + MoldSection] = H_Init[6 + MoldSection] + dh;
			for (int temp = 0; temp < Section; temp++)
				H_Init_Temp8[temp] = H_Init[temp];
			H_Init_Temp8[7 + MoldSection] = H_Init[7 + MoldSection] + dh;

			HANDLE_ERROR(cudaMemcpy(T_HoldLast, dev_T_Last, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

			//  Section 0
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init0, H_Init_Temp0, Section * sizeof(float), cudaMemcpyHostToDevice, stream0));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last0, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream0));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream0 >> > (dev_T_New0, dev_T_Last0, dev_ccml, dev_H_Init0, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result0, dev_T_Last0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream0));

			//Section 1
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init1, H_Init_Temp1, Section * sizeof(float), cudaMemcpyHostToDevice, stream1));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last1, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream1));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream1 >> > (dev_T_New1, dev_T_Last1, dev_ccml, dev_H_Init1, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result1, dev_T_Last1, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream1));


			//Section 2
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init2, H_Init_Temp2, Section * sizeof(float), cudaMemcpyHostToDevice, stream2));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last2, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream2));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream2 >> > (dev_T_New2, dev_T_Last2, dev_ccml, dev_H_Init2, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result2, dev_T_Last2, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream2));

			//Section 3
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init3, H_Init_Temp3, Section * sizeof(float), cudaMemcpyHostToDevice, stream3));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last3, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream3));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream3 >> > (dev_T_New3, dev_T_Last3, dev_ccml, dev_H_Init3, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result3, dev_T_Last3, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream3));

			//Section 4
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init4, H_Init_Temp4, Section * sizeof(float), cudaMemcpyHostToDevice, stream4));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last4, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream4));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream4 >> > (dev_T_New4, dev_T_Last4, dev_ccml, dev_H_Init4, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result4, dev_T_Last4, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream4));

			//Section 5
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init5, H_Init_Temp5, Section * sizeof(float), cudaMemcpyHostToDevice, stream5));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last5, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream5));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream5 >> > (dev_T_New5, dev_T_Last5, dev_ccml, dev_H_Init5, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result5, dev_T_Last5, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream5));


			//Section 6
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init6, H_Init_Temp6, Section * sizeof(float), cudaMemcpyHostToDevice, stream6));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last6, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream6));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream6 >> > (dev_T_New6, dev_T_Last6, dev_ccml, dev_H_Init6, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result6, dev_T_Last6, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream6));


			//Section 7
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init7, H_Init_Temp7, Section * sizeof(float), cudaMemcpyHostToDevice, stream7));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last7, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream7));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream7 >> > (dev_T_New7, dev_T_Last7, dev_ccml, dev_H_Init7, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result7, dev_T_Last7, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream7));


			//Section 8
			HANDLE_ERROR(cudaMemcpyAsync(dev_H_Init8, H_Init_Temp8, Section * sizeof(float), cudaMemcpyHostToDevice, stream8));
			HANDLE_ERROR(cudaMemcpyAsync(dev_T_Last8, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice, stream8));
			for (int PNum = 0; PNum < Num_Iter; PNum++)
			{
				addKernel << <num_blocks, threadsPerBlock, 0, stream8 >> > (dev_T_New8, dev_T_Last8, dev_ccml, dev_H_Init8, dx, dy, dz, tao, nx, ny, nz, dstOut);
				dstOut = !dstOut;
			}
			HANDLE_ERROR(cudaMemcpyAsync(T_Result8, dev_T_Last8, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost, stream8));

			// stream synchronize
			HANDLE_ERROR(cudaStreamSynchronize(stream0));
			HANDLE_ERROR(cudaStreamSynchronize(stream1));
			HANDLE_ERROR(cudaStreamSynchronize(stream2));
			HANDLE_ERROR(cudaStreamSynchronize(stream3));
			HANDLE_ERROR(cudaStreamSynchronize(stream4));
			HANDLE_ERROR(cudaStreamSynchronize(stream5));
			HANDLE_ERROR(cudaStreamSynchronize(stream6));
			HANDLE_ERROR(cudaStreamSynchronize(stream7));
			HANDLE_ERROR(cudaStreamSynchronize(stream8));


			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result0);  // calculation the mean surface temperature of steel billets in every cooling sections
			for (int temp = 0; temp < CoolSection; temp++)
			{
				for (int column = 0; column < CoolSection; column++)
					Mean_TSurfaceElementOne[temp][column] = Mean_TSurface[column + MoldSection];
				//Mean_TSurfaceElementOne[CoolSection - 1][CoolSection - 1] = Mean_TSurface[column + MoldSection];
			}

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result1);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[0][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result2);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[1][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result3);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[2][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result4);  // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[3][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result5);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[4][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result6);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[5][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result7);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[6][column] = Mean_TSurface[column + MoldSection];

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result8);   // calculation the mean surface temperature of steel billets in every cooling sections
			for (int column = 0; column < CoolSection; column++)
				Mean_TSurfaceElement[7][column] = Mean_TSurface[column + MoldSection];


			if (PrintLabel == 1)
				printf("\nJacobianMatrix=\n");
			for (int row = 0; row < CoolSection; row++)
			{
				for (int column = 0; column < CoolSection; column++)
				{
					JacobianMatrix[row][column] = (Mean_TSurfaceElement[row][column] - Mean_TSurfaceElementOne[row][column]) / dh;
					if (PrintLabel == 1)
						printf("%f ", JacobianMatrix[row][column]);
				}
				if (PrintLabel == 1)
					printf("\n");
			}

			for (int temp = 0; temp < CoolSection; temp++)
				Delta_H_Init[temp] = 0.0;

			if (PrintLabel == 1)
				printf("\nDelta_H_Init=\n");
			for (int temp = 0; temp < CoolSection; temp++)
			{
				for (int column = 0; column < CoolSection; column++)
					Delta_H_Init[temp] += (Mean_TSurfaceElementOne[temp][column] - Taim[column]) * JacobianMatrix[temp][column];
				if (PrintLabel == 1)
					printf(" %f, ", Delta_H_Init[temp]);
			}

			arf1 = 0.0, arf2 = 0.0;
			for (int temp = 0; temp < CoolSection; temp++)
			{
				for (int column = 0; column < CoolSection; column++)
				{
					arf1 += (Mean_TSurfaceElementOne[0][temp] - Taim[temp]) * JacobianMatrix[temp][column] * Delta_H_Init[column];
					arf2 += JacobianMatrix[temp][column] * Delta_H_Init[column] * JacobianMatrix[temp][column] * Delta_H_Init[column];
				}
			}
			step = -arf1 / (arf2);

			for (int temp = 0; temp < CoolSection; temp++)
				H_Init[temp + MoldSection] += step *(Delta_H_Init[temp]);
		}

		for (int temp = 0; temp < Section; temp++)
			H_Init_Temp[temp] = H_Init[temp];
		HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init_Temp, Section * sizeof(float), cudaMemcpyHostToDevice));
		addKernel << <num_blocks, threadsPerBlock >> > (dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, dz, tao, nx, ny, nz, dstOut);
		dstOut = !dstOut;

		if (i % Num_Iter == 0)
		{
			HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_Last, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

			Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result); ;  // calculation the mean surface temperature of steel billets in every cooling sections

			if (PrintLabel == 1)
			{
				printf("time_step = %d  simulation time is %f\n", i, i*tao);
				printf("\nTSurface=\n");
				for (int temp = 0; temp < CoolSection; temp++)
					printf("%f, ", Mean_TSurface[temp + MoldSection]);

				printf("\nTSurface - Taim=\n");
				for (int temp = 0; temp < CoolSection; temp++)
					printf("%f, ", Mean_TSurface[temp + MoldSection] - Taim[temp]);
			}
		}
	}

	/*fp = fopen("D:\\Temperature3DGPU_Static.txt", "w");
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int m = 0; m < nz; m++)
				fprintf(fp, " %f", T_Result[nx * nz * j  + i * nz + m]);
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("D:\\SurfaceTemperature3DGPU_Static.txt", "w");
	for (int j = 0; j < ny; j++)
	{
		fprintf(fp, "%f", T_Result[nx * nz * j + 0 * nz + int((nx - 1) / 2)]);
		fprintf(fp, "\n");
	}
	fclose(fp);*/


	

	// Check for any errors launching the kernel
	HANDLE_ERROR(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.


Error:
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_H_Init);

	return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.

__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd)
{
	float Ts = 1462.0, Tl = 1518.0, lamds = 30, lamdl = 50, phos = 7000, phol = 7500, ce = 540.0, L = 265600.0, fs = 0.0;
	if (T<Ts)
	{
		fs = 0;
		*pho = phos;
		*lamd = lamds;
		*Ce = ce;
	}

	if (T >= Ts&&T <= Tl)
	{
		fs = (T - Ts) / (Tl - Ts);
		*pho = fs*phos + (1 - fs)*phol;
		*lamd = fs*lamds + (1 - fs)*lamdl;
		*Ce = ce + L / (Tl - Ts);
	}

	if (T>Tl)
	{
		fs = 1;
		*pho = phol;
		*lamd = lamdl;
		*Ce = ce;
	}

}

__device__ float Boundary_Condition(int j, float dy, float *ccml_zone, float *H_Init)
{
	float YLabel, h = 0.0;
	YLabel = j*dy;

	for (int i = 0; i < Section; i++)
	{
		if (YLabel >= *(ccml_zone + i) && YLabel <= *(ccml_zone + i + 1))
			h = *(H_Init + i);
	}
	return h;
}

void Calculation_MeanTemperature(int nx, int ny, int nz, float dy, float *ccml, float *T)
{
	float y;
	int count = 0;
	int i = 0;
	
	for (int i = 0; i < Section; i++)
	{
		Mean_TSurface[i] = 0.0;
		for (int j = 0; j < ny; j++)
		{
			y = j * dy;
			if (y > *(ccml + i) && y <= *(ccml + i + 1))
			{
				Mean_TSurface[i] = Mean_TSurface[i] + T[nx * nz * j + 0 * nz + int((nx - 1) / 2)];
				count++;
			}
		}
		Mean_TSurface[i] = Mean_TSurface[i] / float(count);
		count = 0;
	}
}