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
    _thresh = 1E-8; ///< default value
    _max_iter = 100; ///< default value

    _scr_dir = scr_dir;
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
    mat _V = randu(_dim, _n_roots);
    mat ovlp = _V.t() * _V;
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
    _V = _V*V;
    
    ovlp = _V.t() * _V;
    //cout << ovlp << endl;
    if(norm(eye(ovlp.n_rows, ovlp.n_cols)-ovlp) > _thresh) throw std::runtime_error("problem in rand_init");
    
    _V.save(_subspace_file_curr, arma_binary);
    _iter = 0;
};/*}}}*/

void Davidson::form_subspace_matrix()
{
    /**
      Form v' H v. Here we assume we can store the full sigma vector in memory
      **/
    
    mat V,sigma;
    mat T;
    
    sigma.load(_sigma_file_save,arma_binary);
   
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
        cout << norm(eye(V.n_cols, V.n_cols)-V.t() * V) << endl;
        mat V_curr;
        V_curr.load(_subspace_file_curr,arma_binary);
        V = join_rows(V,V_curr);
    };


    T = V.t() * sigma;// V'H*V

    T = .5*(T+T.t()); // get rid of any numerical noise breaking symmetry

    mat X;
    vec res_vals, ritz_vals;
    //vec l;

    eig_sym(ritz_vals,X,T);
    //cout << l << endl;

    mat V_new;
    for(int n=0; n<_n_roots; n++)
    {
        double l_n = ritz_vals(n);
        sigma -= l_n*V;
        mat r_n = sigma*X.col(n);

        double b_n = norm(r_n);
        //cout << b_n << endl;
        r_n = r_n/b_n;
        
        // append current eigenvalue of T to list of ritz_vals 
        res_vals.resize(res_vals.n_elem + 1);
        res_vals(res_vals.n_elem-1) = b_n;

        if(_iter > 4) {}; // do preconditioning
        
       
        // check if this root is converged 
        if(b_n > _thresh)
        {
            // this needs fixed
            //mat tmp = _V.join_cols(_V,r_n);
            //cout << _V << endl;
            //cout << join_rows(_V,r_n) << endl;
            
            /*
            mat Q,R;
            qr_econ(Q,R,join_rows(V,join_rows(V_new,r_n)));
            cout << "V" << endl;
            cout << V << endl;
            cout << "VR" << endl;
            cout << join_rows(V,r_n) << endl;
            cout << "V.t() * R" << endl;
            cout << V.t() * r_n << endl;
            cout << "Q" << endl;
            cout << Q  << endl<< endl; 
            V_new = join_rows(V_new,Q.col(Q.n_cols-1));
            V_new = join_rows(V_new,r_n);
            */
            for(int j=0; j<V.n_cols; j++) r_n = r_n - V.col(j)*dot(V.col(j),r_n);
            V_new = join_rows(V_new,r_n);
        }
        else
        {
            continue;
        };
    };
    
    V_new = orth(V_new);

    V.save(_subspace_file_save, arma_binary);
    V_new.save(_subspace_file_curr, arma_binary);

    printf("  Iteration %4i ",_iter);
    printf("|");
    printf(" Vecs:%3i : ",V.n_rows);
    printf("|");
    for(int r=0; r<_n_roots; r++) printf(" %12.8f ",ritz_vals(r));
    printf("|");
    for(int r=0; r<_n_roots; r++) printf(" %6.1e ",res_vals(r));
    printf("\n");


    //check all for convergence
    int done =1;
    for(int k=0; k<res_vals.n_elem; k++)
    {
        if(abs(res_vals(k)) > _thresh) done = 0; 
    };
    if(done == 1) return;
    _iter += 1;
};
