__kernel
void
diffusion(
	__global const float *a,
	__global float *c,
	int width,
	int height,
	float d
	)
{

	int i = get_global_id(0);
	int j = get_global_id(1);

	float value = a[i + width * j];
	float up, down, left, right;

	if (i == 0)
		left = 0.;
	else 
		left = a[(i-1) + width * j];
		
	if (i == width-1)
		right = 0.;
	else
		right = a[(i+1) + width * j];
	
	if (j == 0)
		up = 0.;
	else
		up = a[i + width * (j-1)];
	
	if (j == height-1)
		down = 0.;
	else
		down = a[i + width * (j+1)];

	//printf("i,j are %d,%d --- up,down,left and right are %f,%f,%f,%f\n",i,j,up,down,left,right);
	value += d * ((up+down+left+right)/4 - value);
	c[i + width * j] = value;
} 
