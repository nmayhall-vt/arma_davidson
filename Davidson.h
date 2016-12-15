#ifndef DAVIDSON_H
#define DAVIDSON_H

#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <armadillo>

using namespace arma;
using namespace std;

class Davidson 
{
    /**  \brief
        Davidson solver  
        **/
        
    private:
        //mat _V;         ///< subspace vectors
        //mat _K;         ///< subspace vectors
        size_t _dim;    ///< dimension of CI space
        int _iter;      ///< current iteration   
        int _n_roots;   ///< number of roots sought
        double _thresh; ///< thresh
        int _max_iter;  ///< max iterations
        mat _sigma;     ///< sigma vectors
        string _sigma_file_curr; ///< sigma vector filename
        string _sigma_file_save; ///< sigma vector filename
        string _subspace_file_curr; ///< subspace vector filename
        string _subspace_file_save; ///< subspace vector filename
        string _scr_dir; 


    public:
        Davidson(const size_t& dim, const int& n_roots, const string& scr);
        
        // functions
      
        /// Initialize subspace vectors with random values and orthogonalize 
        void rand_init();

        /// Set scratch file directory 
        void set_scr_dir(const string& f){_scr_dir = f;};

        /// Set max number of iterations
        void set_max_iter(const int& m){_max_iter = m;};

        /// form matrix in krylov subspace
        void form_subspace_matrix();

        ///// Set sigma vector 
        //void set_sigma(mat s){_sigma = s;};


        // access
        
        /// get current subspace vectors file
        string& subspace_file_curr() {return _subspace_file_curr;}; 

        /// get current sigma vectors file
        string& sigma_file_curr() {return _sigma_file_curr;}; 
        
        /// get max iterations
        int max_iter() {return _max_iter;}; 
        
        /// get current iteration
        int iter() {return _iter;}; 

        /// get number of roots sought 
        int n_roots() {return _n_roots;}; 

        /// get dimension of CI space 
        size_t dim() {return _dim;}; 
};
#endif


