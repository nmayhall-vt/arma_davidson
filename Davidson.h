#ifndef DAVIDSON_H
#define DAVIDSON_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <armadillo>
#include <sys/stat.h>

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
        
        vec _A_diag;
        mat _sigma_curr;
        mat _sigma_save;
        mat _subspace_curr;
        mat _subspace_save;
        
        string _A_diag_file; ///< vector of diagonal for preconditioner filename
        string _sigma_file_curr; ///< sigma vector filename
        string _sigma_file_save; ///< sigma vector filename
        string _subspace_file_curr; ///< subspace vector filename
        string _subspace_file_save; ///< subspace vector filename
        string _scr_dir; 
        size_t _subspace_size;  ///< Number of current subspace vectors
        vec _res_vals; ///< current residual values
        vec _ritz_vals; ///< current ritz values
        int _do_preconditioner;
        void precondition(vec& Hd, vec& r, double& l);
        void precondition(vec& Hd, mat& R, vec& l);
        mat _ritz_vecs;
        double _precond_thresh_switch;


    public:
        Davidson(const size_t& dim, const int& n_roots, const string& scr);
        //~Davidson();
        
        // functions
      
        /// Initialize subspace vectors with random values and orthogonalize 
        void rand_init();

        /// Set scratch file directory 
        void set_scr_dir(const string& f){_scr_dir = f;};

        /// Orth SS
        void orthogonalize_subspace();

        /// Set convergence threshhold for norm of residual
        void set_thresh(const double&e){_thresh = e;};

        /// Set convergence threshhold for deciding when to turn on preconditioner 
        void set_precond_thresh_switch(const double&e){_precond_thresh_switch = e;};

        /// Set diagonal of Hamiltonian 
        void set_A_diag(vec& m){_A_diag = m;};

        /// Set max number of iterations
        void set_max_iter(const int& m){_max_iter = m;};

        /// form matrix in krylov subspace and get updated values
        void iterate();

        /// turn-on preconditioning 
        void turn_on_preconditioner(){_do_preconditioner = 1;};

        /// turn-off preconditioning 
        void turn_off_preconditioner(){_do_preconditioner = 0;};

        ///// Set sigma vector 
        void set_sigma_curr(mat& s){_sigma_curr = s;};
        
        /// Print current iteration's info 
        void print_iteration();

        /// Restart subspace with current vector
        void restart();

        // access
        
        /// get number of subspace vectors 
        size_t subspace_size() {return _subspace_size;}; 
        
        /// get current subspace vectors file
        string& subspace_file_curr() {return _subspace_file_curr;}; 

        /// get current sigma vectors file
        string& sigma_file_curr() {return _sigma_file_curr;}; 
        
        /// get eigenvectors in a matrix 
        mat get_eigenvectors(); 
        
        /// get filename for vector of matrix diagonal 
        string& A_diag_file() {return _A_diag_file;}; 
        
        /// get max iterations
        int max_iter() {return _max_iter;}; 
        
        /// get current iteration
        int iter() {return _iter;}; 

        /// get number of roots sought 
        int n_roots() {return _n_roots;}; 

        /// get dimension of CI space 
        size_t dim() {return _dim;}; 

        /// Check for convergence 
        int converged(); 

        /// Check for convergence to within specific threshhold
        int converged(const double&); 

        /// Check for convergence 
        mat& get_subspace_curr(){return _subspace_curr;}; 

        /// Check for convergence 
        vec eigenvalues(){return _ritz_vals;}; 
};
#endif


