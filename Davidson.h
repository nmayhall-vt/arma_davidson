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
        mat _sig_curr;     ///< sigma vectors
        mat _sig_prev;     ///< sigma vectors
        mat _vec_curr;     ///< current subspace vectors
        mat _vec_prev;     ///< previous subspace vectors
        vec _Hd;            ///< vector of diagonal for preconditioner

        /*
        string _A_diag_file; ///< vector of diagonal for preconditioner filename
        string _sigma_file_curr; ///< sigma vector filename
        string _sigma_file_save; ///< sigma vector filename
        string _subspace_file_curr; ///< subspace vector filename
        string _subspace_file_save; ///< subspace vector filename
        string _scr_dir; 
        */

        size_t _subspace_size;  ///< Number of current subspace vectors
        vec _res_vals; ///< current residual values
        vec _ritz_vals; ///< current ritz values
        mat _ritz_vecs; ///< current ritz vectors
        int _do_preconditioner;
        void precondition(vec& Hd, vec& r, double& l);
        void precondition(vec& Hd, mat& R, vec& l);


    public:
        Davidson(const size_t& dim, const int& n_roots);
        
        // functions
      
        /// Initialize subspace vectors with random values and orthogonalize 
        void rand_init();

        /// Set convergence threshhold for norm of residual
        void set_thresh(const double&e){_thresh = e;};

        /// Set max number of iterations
        void set_max_iter(const int& m){_max_iter = m;};

        /// form matrix in krylov subspace and get updated values
        void iterate();

        /// turn-on preconditioning 
        void turn_on_preconditioner(){_do_preconditioner = 1;};

        /// turn-off preconditioning 
        void turn_off_preconditioner(){_do_preconditioner = 0;};

        ///// Set sigma vector 
        //void set_sigma(mat s){_sigma = s;};
        
        /// Print current iteration's info 
        void print_iteration();

        /// Collapse subspace 
        void restart(); 



        // access
        
        /// set diagonal of H 
        void set_H_diag(vec); 
        
        /// get diagonal of H 
        vec& H_diag() {return _Hd;}; 
        
        /// get sigma 
        mat& sigma() {return _sig_curr;}; 
        
        /// get current subspace vectors 
        mat& subspace_vecs() {return _vec_curr;}; 
        
        /// get max iterations
        int max_iter() {return _max_iter;}; 
        
        /// get current iteration
        int iter() {return _iter;}; 

        /// get number of roots sought 
        int n_roots() {return _n_roots;}; 

        /// get dimension of CI space 
        size_t dim() {return _dim;}; 

        /// get dimension of sub-space 
        size_t subspace_size() {return _subspace_size;}; 

        /// Check for convergence 
        int converged(); 
};
#endif


