typedef at::Half fp16;

template<typename F>
void wkv(int B, int C, 
            float *w, float *u, F *k, F *v, F *y, 
            float *aa, float *bb, float *pp, int *lens, int *numset);

void i8seq( int B, int N,  int M, 
                fp16 *x, uint8_t *w,
                fp16 *mx,
                fp16 *rx,
                fp16 *my,
                fp16 *ry,
                fp16 *y);