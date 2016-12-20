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


    size_t N = 300;
    
    Davidson my_davidson(N,3,"scr");
    my_davidson.rand_init();

    mat A = randn(N,N);
    A = A.t() + A;
    A = A - 10*diagmat(randu(N));
    
    my_davidson.set_max_iter(100);

    cout << my_davidson.iter() << endl;

    for(int i=0; i<my_davidson.max_iter(); i++)
    {
        // get sigma vector and save it to disk
        mat V;
        V.load(my_davidson.subspace_file_curr(), arma_binary);
        mat sigma = A * V;
        sigma.save(my_davidson.sigma_file_curr());

        my_davidson.iterate();
        if(my_davidson.converged()) break;
            
    };
   
    {
        mat U;
        vec e;
        eig_sym(e,U,A);
        cout << " exact  " << endl;
        for(int i=0; i<10; i++) printf( "  %4i %16.12f \n", i, e(i));
    };
    return 0;
}

