#include "Davidson.h"
#include <sys/stat.h>


using namespace arma;
using namespace std;

Davidson::Davidson(const size_t& dim, const int& n_roots, const string& scr_dir)
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

    _scr_dir = scr_dir;
    _A_diag_file = _scr_dir + "/A_diag.mat";
    _sigma_file_curr = _scr_dir + "/sigma_curr.mat";
    _sigma_file_save = _scr_dir + "/sigma_save.mat";
    _subspace_file_curr = _scr_dir + "/subspace_curr.mat";
    _subspace_file_save = _scr_dir + "/subspace_save.mat";

    {
        struct stat sb;

        char tab2[1024];
        strcpy(tab2, _scr_dir.c_str());
        if (stat(tab2, &sb) == 0 && S_ISDIR(sb.st_mode))
        {
            printf("YES\n");
        }
        else
        {
            throw std::runtime_error("SCR DIR not present");
        }
    };
    mat tmp;
    tmp.save(_sigma_file_save);
    tmp.save(_sigma_file_curr);
    tmp.save(_subspace_file_save);
    tmp.save(_subspace_file_curr);
};/*}}}*/

void Davidson::rand_init()
{/*{{{*/
    mat v = randu(_dim, _n_roots);
    v.save(_subspace_file_curr, arma_binary);

    orthogonalize_subspace();

    _iter = 0;
};/*}}}*/

void Davidson::iterate()
{/*{{{*/
    /**
      Form v' H v. Here we assume we can store the full sigma vector in memory
      **/
    
    mat V,sigma;
    mat T;
    vec Hd; // matrix diagonal
    
    sigma.load(_sigma_file_save,arma_binary);
    Hd.load(A_diag_file(), arma_binary);
   
    //  collect previous sigma vectors and recent additions 
    {
        mat sigma_curr;
        sigma_curr.load(_sigma_file_curr,arma_binary);
        sigma = join_rows(sigma,sigma_curr);
    };
    sigma.save(_sigma_file_save, arma_binary);

    //  collect previous subspace vectors and recent additions 
    {
        V.load(_subspace_file_save,arma_binary);
        //cout << norm(eye(V.n_cols, V.n_cols)-V.t() * V) << endl;
        mat V_curr;
        V_curr.load(_subspace_file_curr,arma_binary);
        V = join_rows(V,V_curr);
        V.save(_subspace_file_save, arma_binary);
    };


    T = V.t() * sigma;// V'H*V

    /*
    mat S = V.t() * V;// V'V
    //cout << det(S) << endl;
    mat X, Xinv;

    {
        mat scr_m;    
        vec scr_v;    
        eig_sym(scr_v,scr_m,S);
        X = scr_m * diagmat(pow(scr_v,-0.5)) * scr_m.t();
        Xinv = scr_m * diagmat(pow(scr_v,0.5)) * scr_m.t();
    };
    */

    //T = .5*(T+T.t()); // get rid of any numerical noise breaking symmetry

    mat X;
    //vec l;

    eig_sym(_ritz_vals,X,T);
    X = X.cols(0,_n_roots-1);
    _ritz_vecs = X;
    _ritz_vals = _ritz_vals.subvec(0,_n_roots-1);
    //cout << _ritz_vals << endl;
    
    mat R = sigma * X - V * X * diagmat(_ritz_vals);
    mat V_new;

    for(int n=0; n<_n_roots; n++)
    {
        vec r_n = R.col(n);

        double b_n = norm(r_n);
        
        // append current eigenvalue of T to list of ritz_vals 
        _res_vals(n) = b_n;

        // do preconditioning
        if(_do_preconditioner)  precondition(Hd, r_n, _ritz_vals(n));

        b_n = norm(r_n); 
        
       
        // check if this root is converged 
        //if(b_n > _thresh)
        {
            r_n = r_n - V*V.t()*r_n;
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

    //V.save(_subspace_file_save, arma_binary);
    V_new.save(_subspace_file_curr, arma_binary);

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
    mat v,s;
    v.load(_subspace_file_save, arma_binary);
    s.load(_subspace_file_save, arma_binary);
    //cout << "v: " << v.n_rows << "," << v.n_cols << endl;
    //cout << "r: " << _ritz_vecs.n_rows << "," << _ritz_vecs.n_cols << endl;
    v = v * _ritz_vecs;
    s = s * _ritz_vecs;
    v.save(_subspace_file_curr, arma_binary);
    s.save(_sigma_file_curr, arma_binary);
    _ritz_vecs.eye();

    v.clear();
    s.clear();
    v.save(_subspace_file_save, arma_binary);
    s.save(_sigma_file_save, arma_binary);

    orthogonalize_subspace();

    _subspace_size = _n_roots;
};/*}}}*/

void Davidson::orthogonalize_subspace()
{/*{{{*/
    mat v;
    v.load(_subspace_file_curr, arma_binary);
   
     
    mat ovlp = v.t() * v;
    mat U,V;
    vec s;
    svd(U,s,V,ovlp);

    for (int i=0; i<s.n_elem; i++)
    {
        if( abs(s(i))>1e-6) s(i) = 1/sqrt(s(i));
        else(s(i)=1);
    };
    V = V*diagmat(s);
    v = v*V;
    
    ovlp = v.t() * v;
    if(norm(eye(ovlp.n_rows, ovlp.n_cols)-ovlp) > _thresh) throw std::runtime_error("problem in orthogonalize_subspace");

    v.save(_subspace_file_curr, arma_binary);
};/*}}}*/
