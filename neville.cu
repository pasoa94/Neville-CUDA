#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <helper_cuda.h>

#define SIZE_OF_GRID 512

double * neville_s(double *,double *, double *,int, int);//implimentazione seriale
double * load_k0(double *,double *, double *, int , int, int, int);//prepara e lancia il kernel0
__global__ void kernel0(double *, double *, double *, double *, int, int);
double * load_k1(double *,double *, double *, int , int, int, int);//prepara e lancia il kernel1
__global__ void kernel1(double *, double *, double *, double *, int, int);


int main (int argc, char *argv[]){
	double *px, *py;
	int N, Nx, ii, dimx, dimy;
	double *x, *y,*y1, *y2, PI, step;
	clock_t start;
	float cpu_time, gpu0_time, gpu1_time, err_medio;
	cudaEvent_t start0, stop0, start1, stop1;

	PI = 4*atan(1.0);
	Nx = 200000;
	x = (double *)malloc(Nx*sizeof(double));

	srand(123);
	for (ii = 0; ii < Nx; ii++)
		x[ii] = PI * rand() / (double) RAND_MAX;


	N = 32; // N-1 e' il grado del pol.
	px = (double *)malloc(N*sizeof(double));
	py = (double *)malloc(N*sizeof(double));

	// lookup table: sin() tra
	//  0 e PI
	step = 1.0 / (N-1);
	for (ii = 0; ii < N; ii++){
		px[ii] = ii*step*PI;
		py[ii] = sin(px[ii]);
	}


	//implementazione seriale
	start = clock();
	y = neville_s(px,py,x,N,Nx);
	start = clock() - start;
	cpu_time = start/(float)CLOCKS_PER_SEC;
	cpu_time *= 1000.0;//porto in millisecondi
	err_medio = 0;
	for(ii = 0; ii < Nx; ii++) err_medio += fabs(y[ii] - sin(x[ii]));

	printf("CPU time: %12.10f [ms]\n\terrore medio: %.15f\n", cpu_time,err_medio);


	//calcolo la dimensione della griglia
  if(Nx < 513){//griglia ad una dimensione
    dimx = Nx;
    dimy = 1;
  }
  else{//griglia a due dimensioni (max 512*512 punti)
    dimx = 512;
    dimy = (int)(Nx/512) + 1;
  }


	//implementazione kernel 0
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	cudaEventRecord( start0, 0 );

 	y1 = load_k0(px,py,x,N,Nx,dimx,dimy);

	cudaEventRecord( stop0, 0 );
	cudaEventSynchronize( stop0 );
	cudaEventElapsedTime( &gpu0_time, start0, stop0 );
	cudaEventDestroy(start0);
	cudaEventDestroy(stop0);

	err_medio = 0;
	for(ii = 0; ii < Nx; ii++) err_medio += fabs(y1[ii] - sin(x[ii]));
	printf("kernel0: % 12.3f [ms], speedup: %3.0f.\n",gpu0_time, cpu_time/gpu0_time);
	printf("\terrore medio: %.15f\n",err_medio);


	//implementazione kernel 1
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord( start1, 0 );

	y2 = load_k1(px,py,x,N,Nx,dimx,dimy);

	cudaEventRecord( stop1, 0 );
	cudaEventSynchronize( stop1 );
	cudaEventElapsedTime( &gpu1_time, start1, stop1 );
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	err_medio = 0;
	for(ii = 0; ii < Nx; ii++) err_medio += fabs(y2[ii] - sin(x[ii]));
	printf("kernel1: % 12.3f [ms], speedup: %3.0f.\n",gpu1_time, cpu_time/gpu1_time);
	printf("\terrore medio: %.15f\n",err_medio);


	free(px);
	free(py);
	free(x);
	free(y);
	free(y1);
	free(y2);
	return 0;
}

/*implementazione dell'algoritmo seriale*/
double * neville_s(double * px,double * py, double * x,int N, int Nx){
	double * y;//contiene f(x)
	double * s;//vettore utilizzato per la risoluzione dell'algoritmo
	int ii,jj,kk;//indici

	//allocazione memoria
	y = (double *)malloc(sizeof(double)*Nx);
	s = (double *)malloc(sizeof(double)*N);

	//implementazione del metodo
	for(ii = 0; ii<Nx; ii++){
		//copio i valori di py in s
		for(jj = 0; jj<N; jj++) s[jj] = py[jj];

		//algoritmo di Neville
		for(jj = 1; jj<=N-1; jj++){
			for(kk = 0; kk<=N-jj-1; kk++){
				s[kk]=((px[kk+jj] - x[ii])*s[kk] + (x[ii]-px[kk])*s[kk+1])/(px[kk+jj] - px[kk]);
			}
		}

		//in s[0] troviamo il risultato dell'interpolazione
		y[ii] = s[0];
	}

	free(s);

	return	y;
}

/*  ha il compito di preparare e passare gli elementi che il
    kernel 0 dovra' poi usare*/
double * load_k0(double *px,double * py,double *x,int N, int Nx,int dimx,int dimy){
	double * px_d, * py_d, * x_d, * y_d, * y;
	//int N_d, Nx_d;

	//allocazione memoria per device
	checkCudaErrors(cudaMalloc((void **) &px_d, sizeof(double)*N));
	checkCudaErrors(cudaMalloc((void **) &py_d, sizeof(double)*N));
	checkCudaErrors(cudaMalloc((void **) &x_d, sizeof(double)*Nx));
	checkCudaErrors(cudaMalloc((void **) &y_d, sizeof(double)*Nx));

	//copio i dati che mi servono nel device
	checkCudaErrors(cudaMemcpy(px_d, px, sizeof(double)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(py_d, py, sizeof(double)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(x_d, x, sizeof(double)*Nx, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(N_d, N, sizeof(int), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(Nx_d, Nx, sizeof(int), cudaMemcpyHostToDevice));

	//alloco il vettore che conterra' il risultato
	y = (double *)malloc(sizeof(double)*Nx);

	//definisco le dimensioni di griglia e blocco
	dim3 dimBlock(N, 1, 1);
	dim3 dimGrid(dimx, dimy, 1);

	//lancio il kernel0
	kernel0 <<< dimGrid, dimBlock >>> (px_d, py_d, x_d, y_d, N, Nx);

	//copia dei risultati ottenuti dal kernel
	checkCudaErrors( cudaMemcpy(y, y_d, sizeof(double)*Nx, cudaMemcpyDeviceToHost) );

	return y;
}

//implementazione del kernel 0
__global__ void kernel0(double *px, double *py, double *x, double *y, int N, int Nx){
  unsigned int t_index,b_index;
	double x_blk;//indica il valore del punto da interpolare del blocco
	__shared__ double s_x[32];
  __shared__ double s_y[32];
	int ii, jj;//indici generici

	//calcolo degli indici
	t_index = threadIdx.x;//indice del thread
	b_index = blockIdx.x + blockIdx.y * gridDim.x;//indice del blocco

	if(b_index < Nx){//filtro dei primi Nx blocchi
		x_blk = x[b_index];//ottengo il valore di x da interpolare del nos_ytro blocco

		//copio i valori di y in s
    s_x[t_index] = px[t_index];
		s_y[t_index] = py[t_index];

		//interpolazione sul thread 0
		if(t_index == 0){
			//algoritmo di Neville
      for(ii = 1; ii<N; ii++){
				for(jj = 0; jj<N-ii; jj++){
					s_y[jj]=(s_y[jj]*(s_x[jj+ii] - x_blk) + s_y[jj+1]*(x_blk-s_x[jj]))/(s_x[jj+ii] - s_x[jj]);
				}
			}
			y[b_index] = s_y[0];//copio il risultato
		}
	}
}

/*  ha il compito di preparare e passare gli elementi che il
    kernel 1 dovra' poi usare*/
double * load_k1(double *px,double * py,double *x,int N, int Nx,int dimx,int dimy){
	double * px_d, * py_d, * x_d, * y_d, * y;

	//allocazione memoria per device
	checkCudaErrors(cudaMalloc((void **) &px_d, sizeof(double)*N));
	checkCudaErrors(cudaMalloc((void **) &py_d, sizeof(double)*N));
	checkCudaErrors(cudaMalloc((void **) &x_d, sizeof(double)*Nx));
	checkCudaErrors(cudaMalloc((void **) &y_d, sizeof(double)*Nx));

	//copio i dati che mi servono nel device
	checkCudaErrors(cudaMemcpy(px_d, px, sizeof(double)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(py_d, py, sizeof(double)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(x_d, x, sizeof(double)*Nx, cudaMemcpyHostToDevice));

	//alloco il vettore che conterra' il risultato
	y = (double *)malloc(sizeof(double)*Nx);

	//definisco le dimensioni di griglia e blocco
	dim3 dimBlock(N, 1, 1);
	dim3 dimGrid(dimx, dimy, 1);

	//lancio il kernel1
	kernel1 <<< dimGrid, dimBlock >>> (px_d, py_d, x_d, y_d, N, Nx);

	//copia dei risultati ottenuti dal kernel
	checkCudaErrors( cudaMemcpy(y, y_d, sizeof(double)*Nx, cudaMemcpyDeviceToHost) );

	return y;
}

//implementazione del kernel 1
__global__ void kernel1(double *px, double *py, double *x, double *y, int N, int Nx){
  unsigned int t_index,b_index;
	double x_blk;//indica il valore del punto da interpolare del blocco
	__shared__ double s_x[32];
  __shared__ double s_y[32];
	int ii;//indici generici
	double cpy1, cpy2;//memorizzo i valori s che mi servono

	//calcolo degli indici
	t_index = threadIdx.x;//indice del thread
	b_index = blockIdx.x + blockIdx.y * gridDim.x;//indice del blocco

	if(b_index < Nx){//filtro dei primi Nx blocchi
		x_blk = x[b_index];//ottengo il valore di x da interpolare del nos_ytro blocco

		//copio i valori di y in s
    s_x[t_index] = px[t_index];
		s_y[t_index] = py[t_index];

		//applico l'algoritmo di Neville
		for(ii = 0; ii < N -1; ii++){
			//copio i valori che mi servono prima che altri thread me li modifichino
			cpy1 = s_y[t_index];
			//uso il modulo perche' il 32-esimo thread possa accedere a s_y[0]
			//tanto tale thread non serve a nulla e non fa nulla per il risultato
			//ad ogni giro il numero di thread "inutili" aumenta di 1
			cpy2 = s_y[(t_index + 1)%N];

			//prima di toccare il vettore mi assicuro che tutti i thread si servirano
			//copiati i valori necessari per l'elaboraione
			__syncthreads();

				//calcolo l's
				s_y[t_index] = ((s_x[(t_index+ii+1)%N]-x_blk)*cpy1 + (x_blk-s_x[t_index])*cpy2)/ (s_x[(t_index+ii+1)%N]-s_x[t_index]);
		}

		//copio il valore in y
		if(t_index == 0)
			y[b_index] = s_y[0];
	}
}
