#include <stdio.h>
#include <stddef.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

int main (int argc, char *argv[]) {

	// Argpars
	int opt, n;
	float d;
	while ((opt = getopt(argc, argv, "n:d:")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'd':
				d = atof(optarg);
				break;
		}
	}

	// Parse init file with input values
	//FILE *fp = fopen("home/hpc2021/diffusion_opencl/test_data/init_100_100", "r");
	FILE *fp = fopen("init", "r");
	const int width, height;
	fscanf(fp, "%d %d", &width, &height);

	float *a = (float*) malloc(sizeof(float)*width*height);
	for (size_t i = 0; i < width*height; i++)
		a[i] = 0.;

	float **matrix = (float**) malloc(sizeof(float*)*height);
	for (size_t i = 0, j = 0; i < height; i++, j += width)
		matrix[i] = a + j;

	int row, col;
	float temp;
	while (fscanf(fp, "%d %d %f", &row, &col, &temp) == 3)
		matrix[row][col] = temp;
	


	for (size_t jx=0; jx<height; ++jx) {
		for (size_t ix=0; ix<width; ++ix) {
			printf(" %5.f ", a[jx*width+ ix]);
		}
		printf("\n");
	}

	free(matrix);

	// Boilerplate opencl begin
	cl_int error;

	cl_platform_id platform_id;
	cl_uint nmb_platforms;
	if ( clGetPlatformIDs(1, &platform_id, &nmb_platforms) != CL_SUCCESS ) {
		fprintf(stderr, "cannot get platform\n" );
		return 1;
	}

	cl_device_id device_id;
	cl_uint nmb_devices;
	if ( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nmb_devices) != CL_SUCCESS ) {
		fprintf(stderr, "cannot get device\n" );
		return 1;
	}

	cl_context context;
	cl_context_properties properties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) platform_id,
		0
	};
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create context\n");
		return 1;
	}

	cl_command_queue command_queue;
	command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create command queue\n");
		return 1;
	}
	
	// Parse kernel program
	char *opencl_program_src;
	{
		FILE *clfp = fopen("./diffusion.cl", "r");
		if ( clfp == NULL ) {
			fprintf(stderr, "could not load cl source code\n");
			return 1;
		}
		fseek(clfp, 0, SEEK_END);
		int clfsz = ftell(clfp);
		fseek(clfp, 0, SEEK_SET);
		opencl_program_src = (char*) malloc((clfsz+1)*sizeof(char));
		fread(opencl_program_src, sizeof(char), clfsz, clfp);
		opencl_program_src[clfsz] = 0;
		fclose(clfp);
	}

	cl_program program;
	size_t src_len = strlen(opencl_program_src);
	program = clCreateProgramWithSource(
			context, 1, (const char **) &opencl_program_src, (const size_t*) &src_len, &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create program\n");
		return 1;
	}

	free(opencl_program_src);

	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot build program. log:\n");

		size_t log_size = 0;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		char *log = malloc(log_size*sizeof(char));
		if ( log == NULL ) {
			fprintf(stderr, "could not allocate memory\n");
			return 1;
		}

		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		fprintf(stderr, "%s\n", log );

		free(log);

		return 1;
	}

	cl_kernel kernel = clCreateKernel(program, "diffusion", &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create kernel\n");
		return 1;
	}

	// create buffers (input and output)
	cl_mem input_buffer_a, output_buffer_c;
	input_buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE,
			width*height* sizeof(float), NULL, &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create buffer a\n");
		return 1;
	}
	output_buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE,
			width*height* sizeof(float), NULL, &error);
	if ( error != CL_SUCCESS ) {
		fprintf(stderr, "cannot create buffer c\n");
		return 1;
	}

	if ( clEnqueueWriteBuffer(command_queue,
				input_buffer_a, CL_TRUE, 0,width*height* sizeof(float), a, 0, NULL, NULL)
			!= CL_SUCCESS ) {
		fprintf(stderr, "cannot enqueue write of buffer a\n");
		return 1;
	}

	// create pointers to buffers to be able to swap (see below)
	cl_mem *ptrInput = &input_buffer_a;
	cl_mem *ptrOutput = &output_buffer_c;
	
	// Run computations n times in parallel
	for (int it = 0; it < n; it++) {

		// Set kernelargs
		clSetKernelArg(kernel, 0, sizeof(cl_mem), ptrInput);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), ptrOutput);
		clSetKernelArg(kernel, 2, sizeof(int), &width);
		clSetKernelArg(kernel, 3, sizeof(int), &height);
		clSetKernelArg(kernel, 4, sizeof(float), &d);

		const size_t global_sz[] = {height,width};
		if ( clEnqueueNDRangeKernel(command_queue, kernel,
					2, NULL, (const size_t *) &global_sz, NULL, 0, NULL, NULL)
				!= CL_SUCCESS ) {
			fprintf(stderr, "cannot enqueue kernel\n");
			return 1;
		}

		// swapping buffers, we want output buffer in iteration i as input buffer in iter i+1
		cl_mem *ptrTemp = ptrInput;
		ptrInput = ptrOutput;
		ptrOutput = ptrTemp;
	}

	// output array to read buffer to
	float *c = malloc(width*height* sizeof(float));
	if ( clEnqueueReadBuffer(command_queue,
				*ptrInput, CL_TRUE, 0,width*height* sizeof(float), c, 0, NULL, NULL)
			!= CL_SUCCESS ) {
		fprintf(stderr, "cannot enqueue read of buffer c\n");
		return 1;
	}


	if ( clFinish(command_queue) != CL_SUCCESS ) {
		fprintf(stderr, "cannot finish queue\n");
		return 1;
	}
	// average temp	
	float sum = 0;
	for (size_t jx=0; jx<height; ++jx) {
		for (size_t ix=0; ix<width; ++ix) {
			printf(" %5.f ", c[jx*width+ ix]);
			sum +=  c[jx*width+ ix];
		}
		printf("\n");
	}
	float avg_temp = sum/(height*width);
	printf("avg temp is %.2f\n",avg_temp);

/*
	// the absolute difference of each temperature and the average
	sum = 0.;
	float val;
	for (size_t jx=0; jx<height; ++jx) {
		for (size_t ix=0; ix<width; ++ix) {
			val = fabs(c[jx*width+ ix] - avg_temp);
			sum += val;
			printf(" %5.f ", val);
		}
		printf("\n");
	}
	avg_temp = sum/(width*height);
	printf("average temp of abs differences from avg is %.2f\n",avg_temp);
*/

	free(a);
	free(c);

	clReleaseMemObject(input_buffer_a);
	clReleaseMemObject(output_buffer_c);

	clReleaseProgram(program);
	clReleaseKernel(kernel);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
