__kernel void rgb_to_ycc(__global float *imgY, __global float *imgC1, __global float *imgC2, __global float *src, int m_x, int width, int y)
{
	// Get the work item's unique ID
    int idx = get_global_id(0);
	// Get rgb values
	const int r = src[idx].r;
	const int g = src[idx].g;
	const int b = src[idx].b;
	// Set YCC of pixel
    imgY[y*m_x + x] =  (0.299     * r) + (0.587     * g) + (0.114     * b)-128.0;
    imgC1[y*m_x + x] = -(0.168736  * r) - (0.331264  * g) + (0.5       * b);
    imgC2[y*m_x + x] =  (0.5       * r) - (0.418688  * g) - (0.081312  * b);
}
