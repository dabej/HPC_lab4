#include <stdio.h>
#include <stddef.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

int
main()
{
	
	int n = 3;

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

  const int width = 3;
  const int height= 3;
	float d = 1./30.;

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

  float *a = malloc(width*height* sizeof(float));

// init a for testing  
	float a_stack[] = {0,0,0,0,1000000,0,0,0,0}; 
	for (int i = 0; i< width*height; i++)
		a[i] = a_stack[i];	

  if ( clEnqueueWriteBuffer(command_queue,
           input_buffer_a, CL_TRUE, 0,width*height* sizeof(float), a, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue write of buffer a\n");
    return 1;
  }

	cl_mem *ptrInput = &input_buffer_a;
	cl_mem *ptrOutput = &output_buffer_c;

	for (int it = 0; it < n; it++) {
  
  clSetKernelArg(kernel, 0, sizeof(cl_mem), ptrInput);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), ptrOutput);
  clSetKernelArg(kernel, 2, sizeof(int), &width);
  clSetKernelArg(kernel, 3, sizeof(int), &height);
  clSetKernelArg(kernel, 4, sizeof(float), &d);
  
	const size_t global_sz[] = {width,height};
  if ( clEnqueueNDRangeKernel(command_queue, kernel,
           2, NULL, (const size_t *) &global_sz, NULL, 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue kernel\n");
    return 1;
  }
		// swapping buffers
		cl_mem *ptrTemp = ptrInput;
		ptrInput = ptrOutput;
		ptrOutput = ptrTemp;
  }

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
	printf("avg temp is %f\n",avg_temp);
  

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
