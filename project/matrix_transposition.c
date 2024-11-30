#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
	#include <omp.h>
#endif
#define TRY 5
//check symmetric matrix
int checkSym(int n,double M[n][n]){
  int i,j;
    for (i = 0; i < n; i++) {
        for(j = i+1; j < n; j++) {
            if(M[i][j]!=M[j][i])
                return 0; // matrix is not simmetric
        }
    }
    return 1;//the matrix is simmetric
}
void matTranspose(int n,double M[n][n],double T[n][n]){
    int i,j;
    for (i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
                T[i][j] = M[j][i];
        }
    }
}

void matTransposeImp(int n,double M[n][n],double T[n][n]){
    int i,j,k,l,blocksize=64;
    #pragma simd
    for (i = 0; i < n; i += blocksize) {
      for (j = 0; j < n; j += blocksize) {
        for (l = j; l < j + blocksize && l < n;l++) {
          #pragma unroll(16)
          for (k = i; k < i + blocksize && k < n;k++) {
                  T[l][k] = M[k][l];
          }
        }
      }
    }
}

void matTransposeOMP(int n,double M[n][n],double T[n][n],int* n_threads){
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        {
        *n_threads = omp_get_num_threads();
        } 
        int i, j, l,k,blocksize=64;  
        #pragma omp for schedule(static) collapse(2)
        for (i = 0; i < n; i += blocksize) {
          for (j = 0; j < n; j += blocksize) {
            for (l = j; l < j + blocksize && l < n;l++) {
              for (k = i; k < i + blocksize && k < n;k++) {
                  T[l][k] = M[k][l];
              }
            }
          }
        }
    }
#endif
}


int main(int argc, char *argv[]){
    srand(time(NULL));

    int i,j,n_threads;
    //take dimension as input
    if(argc!=2){printf("inserisci dimensione array\n");return 0;}
    int n = atoi(argv[1]);
    double M[n][n];;  // Declare matrices M
    double T[n][n];;  // Declare matrices T
    double Time_Sequential[TRY]; //array with time for sequential
    double Time_Implicit[TRY];   //array with time for implicit
    double Time_Explicit[TRY];   //array with time for explicit
    int k;
    for(k=0;k<TRY+1;k++){
    
      // Initialize matrices M with random values
      for (i = 0; i < n; i++) {
          for(j = 0; j < n; j++) {
              M[i][j] = (rand()%10000)/100.0;
          }
      }
      
  
      //check for symmetric matrix
      if(checkSym(n,M)==1)
          printf("matrice simmetrica\n");
      else{
          //---------------------------------------------------------------------------    
          //SEQUENTIAL PART  
          double wt1=omp_get_wtime();
          matTranspose(n,M,T);
          double wt2=omp_get_wtime();
          
          //save time in array
          Time_Sequential[k] = wt2-wt1;
      
          
          //--------------------------------------------------------------------------
          //IMPLICIT PART
#ifdef _OPENMP          
          wt1=omp_get_wtime();
          matTransposeImp(n,M,T);
          wt2=omp_get_wtime();
          
          //save time in array
          Time_Implicit[k] = wt2-wt1;
#else
      printf("codice implicito   non compilato");
#endif
      
         
          //--------------------------------------------------------------------------
          //EXPLICIT PART
#ifdef _OPENMP
          wt1=omp_get_wtime();
          matTransposeOMP(n,M,T,&n_threads);
          wt2=omp_get_wtime();
          
          //save time in array
          Time_Explicit[k] = wt2-wt1;
#else
      printf("codice parallelo non compilato");
#endif
      
      }
    }
    
    //calculate of average  
    double avg_S = 0;
    double avg_I = 0;
    double avg_E = 0;
    //not consider the first value because is often an outlier
    for(k=1;k<TRY+1;k++){
      avg_S += Time_Sequential[k];
    }
    avg_S /= TRY;
    for(k=1;k<TRY+1;k++){
      avg_I += Time_Implicit[k];
    }
    avg_I /= TRY;
    for(k=1;k<TRY+1;k++){
      avg_E += Time_Explicit[k];
    }
    avg_E /= TRY;
    //print of result
    printf("             |  sequential time   |    implicit time    |     explicit time   |     speedup         |      efficiency\n");
    printf("-----------------------------------------------------------------------------------------------------------------------------\n");
    for(k=1;k<TRY+1;k++){
        printf("%d' attempt   |    ",k);
        printf( "%.6f sec    |     ", Time_Sequential[k]);
        printf( "%.6f sec    |     ", Time_Implicit[k]);
        printf( "%.6f sec    |     ", Time_Explicit[k]);
        printf( "%.6f        |      ", Time_Sequential[k]/Time_Explicit[k]);
        printf( "%.6f \n", Time_Sequential[k]/Time_Explicit[k] /n_threads*100);
        
    }
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("average      |    %.6f        |     %.6f        |     %.6f        |     %.6f        |      %.6f\n",avg_S,avg_I,avg_E,avg_S/avg_E,avg_S/avg_E/n_threads*100);
    
    
  
    
    /*
    //parte per controllare la correttezza 
    for (i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%.2f ",M[i][j]);
        }
        printf("\n");
    }
    printf("-------------------\n");
    for (i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%.2f ",T[i][j]);
        }
        printf("\n");
    }
    */
        

    return 0;
}
