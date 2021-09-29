#include <stdio.h>
#include <stddef.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <unistd.h>

float c = 0;
const char *opencl_program_src = \
  "__kernel void dot_prod_mul(__global const float * a, __global const float * b, __global float * c)\n" \
  "{\n" \
  "  int ix = get_global_id(0);\n" \
  "  c[ix] = a[ix] * b[ix];\n" \
  "}\n";


int
main(int argc, char *argv[])
{
	// Argparse
	int opt,n;
	float d;
    while ((opt = getopt(argc, argv, "t:l:")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'd':
				c = atof(optarg);
				break;
		}
    }
	
  cl_int error;
	// init platform
  cl_platform_id platform_id;
  cl_uint nmb_platforms;
  if ( clGetPlatformIDs(1, &platform_id, &nmb_platforms) != CL_SUCCESS ) {
    fprintf(stderr, "cannot get platform\n" );
    return 1;
  }

	// init device
  cl_device_id device_id;
  cl_uint nmb_devices;
  if ( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nmb_devices) != CL_SUCCESS ) {
    fprintf(stderr, "cannot get device\n" );
    return 1;
  }
	// init context
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
	// init command queue
  cl_command_queue command_queue;
  command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create command queue\n");
    return 1;
  }

  /*
  char *opencl_program_src;
  {
    FILE *clfp = fopen("./dotprod.cl", "r");
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
  */

  cl_program program;
  size_t src_len = strlen(opencl_program_src);
  program = clCreateProgramWithSource(
                context, 1, (const char **) &opencl_program_src, (const size_t*) &src_len, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create program\n");
    return 1;
  }

  // free(opencl_program_src);
  
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
  
  cl_kernel kernel = clCreateKernel(program, "dot_prod_mul", &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create kernel\n");
    return 1;
  }

  const int sz = 10000000;
  cl_mem input_buffer_a, input_buffer_b, output_buffer_c;
  input_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sz*sizeof(float), NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create buffer a\n");
    return 1;
  }
  input_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sz*sizeof(float), NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create buffer b\n");
    return 1;
  }
  output_buffer_c = clCreateBuffer(context, CL_MEM_READ_ONLY, sz*sizeof(float), NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create buffer c\n");
    return 1;
  }

  float *a = malloc(sz*sizeof(float));
  float *b = malloc(sz*sizeof(float));
  for ( int ix = 0; ix < sz; ++ix ) {
    a[ix] = ix;
    b[ix] = ix;
  }
  if ( clEnqueueWriteBuffer(command_queue,
           input_buffer_a, CL_TRUE, 0, sz*sizeof(float), a, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue write of buffer a\n");
    return 1;
  }
  if ( clEnqueueWriteBuffer(command_queue,
           input_buffer_b, CL_TRUE, 0, sz*sizeof(float), b, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue write of buffer b\n");
    return 1;
  }
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer_a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer_b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer_c);
  
  const size_t global_sz = (size_t) sz;
  if ( clEnqueueNDRangeKernel(command_queue, kernel,
           1, NULL, (const size_t*) &global_sz, NULL, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue kernel\n");
    return 1;
  }
  
  float *c = malloc(sz*sizeof(float));
  if ( clEnqueueReadBuffer(command_queue,
           output_buffer_c, CL_TRUE, 0, sz*sizeof(float), c, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue read of buffer c\n");
    return 1;
  }

  if ( clFinish(command_queue) != CL_SUCCESS ) {
    fprintf(stderr, "cannot finish queue\n");
    return 1;
  }


  double c_sum = 0;
  for (size_t ix = 0; ix < sz; ++ix)
    c_sum += c[ix];
  
  printf("Dot product equals %f\n", c_sum);
  

  free(a);
  free(b);
  free(c);

  clReleaseMemObject(input_buffer_a);
  clReleaseMemObject(input_buffer_b);
  clReleaseMemObject(output_buffer_c);

  clReleaseProgram(program);
  clReleaseKernel(kernel);

  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
