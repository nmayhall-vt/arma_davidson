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


    size_t N = 10000;
    int n_roots = 1;
    
    Davidson my_davidson(N,n_roots,"scr");
    my_davidson.rand_init();

    mat A = randn(N,N);
    A = A.t() + A;
    A = A - 1000*diagmat(randu(N));
    
    my_davidson.set_max_iter(100);
        
    // store diagonal
    {
        vec d = diagvec(A);
        d.save(my_davidson.A_diag_file(), arma_binary);
    };


    cout << my_davidson.iter() << endl;

    for(int i=0; i<my_davidson.max_iter(); i++)
    {
        // get sigma vector and save it to disk
        mat V;
        V.load(my_davidson.subspace_file_curr(), arma_binary);
        mat sigma = A * V;
        
        sigma.save(my_davidson.sigma_file_curr());
    
        if(i>4) my_davidson.turn_on_preconditioner();
        
        my_davidson.iterate();
        my_davidson.print_iteration();
        if(my_davidson.converged()) break;
            
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

