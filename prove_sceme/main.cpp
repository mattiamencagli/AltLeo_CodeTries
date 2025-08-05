#include<stdio.h>

int main(){

	float gaussian_kernel[5] = {0.0544886849820613861083984375f,
				    0.24420134723186492919921875f,
				    0.40261995792388916015625f,
				    0.24420134723186492919921875f,
			   	    0.0544886849820613861083984375f};
	float sum=0;

	for(int i=0; i<5; ++i)
		sum += gaussian_kernel[i];

	printf("sum : %g\n",sum);







	return 0;
}
