// A simple program that computes the square root of a number
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <armadillo>
#include "Davidson.h"

using namespace arma;

int main ()
{
#ifdef DEBUG
    printf(" Debug mode\n");
#endif


    size_t N = 500;
    int n_roots = 2;
    
    Davidson my_davidson(N,n_roots);
    my_davidson.rand_init();

    arma_rng::set_seed(2);
    mat A = randn(N,N);
    A = A.t() + A;
    A = A - 1000*diagmat(randu(N));
    for(size_t i=0; i<N; i++) A(i,i) += i*2;
    
    my_davidson.set_max_iter(50);
    my_davidson.set_thresh(1e-6);
        
    // store diagonal
    my_davidson.H_diag() = A.diag();

    cout << my_davidson.iter() << endl;

    for(int i=0; i<my_davidson.max_iter(); i++)
    {
        // get sigma vector and save it to disk
        mat V = my_davidson.subspace_vecs();
    
        my_davidson.sigma() = A * V;
       
        if(i>30) my_davidson.turn_on_preconditioner();
        
        my_davidson.iterate();
        my_davidson.print_iteration();
        if(my_davidson.converged()) break;
        
        //if(my_davidson.subspace_size() > 20) my_davidson.restart(); 
            
    };
   
    if(N <= 5000)
    {
        mat U;
        vec e;

        eig_sym(e,U,A);
        cout << " exact  " << endl;
        for(int i=0; i<n_roots; i++) printf( "  %4i %16.12f \n", i, e(i));
    };
    return 0;
}

