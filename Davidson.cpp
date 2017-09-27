#include "Davidson.h"
#include <sys/stat.h>


using namespace arma;
using namespace std;

Davidson::Davidson(const size_t& dim, const int& n_roots)
{/*{{{*/
    arma_rng::set_seed(2);  // set the seed to a random value
    _dim = dim;
    _iter = 0;
    _n_roots = n_roots;
    _thresh = 1E-6; ///< default value
    _max_iter = 100; ///< default value
    _do_preconditioner = 0;
    _res_vals = zeros(n_roots,1);
    _ritz_vals = zeros(n_roots,1);
    _subspace_size = 0;

};/*}}}*/

void Davidson::rand_init()
{/*{{{*/
    _vec_curr = randu(_dim, _n_roots);
    mat ovlp = _vec_curr.t() * _vec_curr;

    mat U,V;
    vec s;
    svd(U,s,V,ovlp);

    for (int i=0; i<s.n_elem; i++)
    {
        if( abs(s(i))>1e-6)
        {
            s(i) = 1/sqrt(s(i));
        }
        else(s(i)=1);
    };
    V = V*diagmat(s);
    _vec_curr = _vec_curr*V;
    
    _vec_curr = _vec_curr*U.t();
    _iter = 0;
};/*}}}*/

void Davidson::iterate()
{/*{{{*/
    /**
      Form v' H v. Here we assume we can store the full sigma vector in memory
      **/
   
    //  assume _sig_curr has already been updated
    
   
    // todo: clean this up to reduce memory
    mat _sig = join_rows(_sig_prev, _sig_curr);
    mat _vec = join_rows(_vec_prev, _vec_curr);
    
    mat T = _vec_curr.t() * _sig;

    T = .5*(T+T.t()); // get rid of any numerical noise breaking symmetry

    mat X;
    //vec l;

    eig_sym(_ritz_vals,X,T);
    
    X = X.cols(0,_n_roots-1);
    _ritz_vecs = X;
    _ritz_vals = _ritz_vals.subvec(0,_n_roots-1);
   
    mat V_new;

    for(int n=0; n<_n_roots; n++)
    {
        double l_n = _ritz_vals(n);
        vec v_n = _ritz_vecs.col(n);

        vec r_n = (_sig - l_n*_vec) * v_n;

        double b_n = norm(r_n);
        
        // append current eigenvalue of T to list of ritz_vals 
        _res_vals(n) = b_n;

        // do preconditioning
        if(_do_preconditioner)  precondition(_Hd, r_n, _ritz_vals(n));

        b_n = norm(r_n); 
        
       
        // check if this root is converged 
        //if(b_n > _thresh)
        {
            r_n = r_n - _vec*_vec.t()*r_n;
            if(V_new.n_cols > 0) r_n = r_n - V_new*V_new.t()*r_n;
            double b_n_p = norm(r_n);
            if(b_n_p / b_n > 1e-3)
            {
                r_n = r_n/b_n_p;
                V_new = join_rows(V_new,r_n);
            };
        };
    };
    
    //V_new = orth(V_new);

    _subspace_size += V_new.n_cols;

    _vec_curr = V_new;

    _iter += 1;
};/*}}}*/

void Davidson::print_iteration()
{/*{{{*/
    printf("  Iteration %4i ",_iter);
    printf("|");
    printf(" Vecs:%4li ",_subspace_size);
    printf("|");
    for(int r=0; r<_n_roots; r++) printf(" %16.8f ",_ritz_vals(r));
    printf("|");
    for(int r=0; r<_n_roots; r++) printf(" %6.1e ",_res_vals(r));
    printf("\n");
};/*}}}*/

int Davidson::converged()
{/*{{{*/
    //check all for convergence
    //
    // returns 0 if not converged
    // returns 1 if converged
    int done =1;
    for(int k=0; k<_res_vals.n_elem; k++)
    {
        if(abs(_res_vals(k)) > _thresh) done = 0; 
    };
    return done;
};/*}}}*/

void Davidson::precondition(vec& Hd, vec& r, double& l)
{/*{{{*/
    double denom;
    for(size_t i=0; i<r.n_elem; i++)
    {
        denom = l-Hd(i);
        if(abs(denom) < 1e-8)
        {
           r(i) = r(i) / 1e-8; 
        }
        else
        {
            r(i) = r(i) / denom;
        };
    };

    //r = M % r;
    //r = r/norm(r);
};/*}}}*/

void Davidson::precondition(vec& Hd, mat& R, vec& l)
{/*{{{*/
    double M;
    for(size_t i=0; i<R.n_cols; i++)
    {
        double Mi = 1;
        
        for(size_t j=0; j<Hd.n_elem; j++)
        {
            Mi = l(i) - Hd(j);
            if(abs(Mi) < 1e-8) 
            {
                R(j,i) = R(j,i) / 1e-8;
            }
            else
            {
                R(j,i) = R(j,i) / Mi;
            };
        };

    };
    //r = M % r;
    //r = r/norm(r);
};/*}}}*/

void Davidson::restart()
{/*{{{*/
    printf(" Restarting...\n");
   
    /*
    mat sig_prev, sig_curr;
    mat vec_prev, vec_curr;

    //  collect previous sigma vectors and recent additions 
    sig_prev.load(_sigma_file_save,arma_binary);

    //  collect previous subspace vectors and recent additions 
    vec_prev.load(_subspace_file_save,arma_binary);
    vec_curr.load(_subspace_file_curr,arma_binary);

    vec_prev = vec_prev * _ritz_vecs;
    sig_prev = sig_prev * _ritz_vecs;
    _ritz_vecs = eye(_n_roots, _n_roots);
    mat tmp = vec_prev.t() * sig_prev;
    _res_vals  = tmp.diag();

    cout << vec_prev.n_rows << " , " << vec_prev.n_cols << endl;
    //tmp = vec_prev.t() * sig_prev; 
    //cout.precision(8);
    //cout.setf(ios::fixed);
    //tmp.raw_print(cout, "A:");

    //cout << _ritz_vecs << endl;
    //vec_prev.save(_subspace_file_save,arma_binary);
*/
};/*}}}*/

/// set diagonal of H 
void Davidson::set_H_diag(vec in)
{/*{{{*/
    _Hd = in;
}; /*}}}*/
