#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

// creates a vector of unsigned
arma::uvec unsign(const Rcpp::IntegerVector & x)
{
  arma::uvec result(x.size());
  for (unsigned i=0; i<result.size(); i++)
  {
    result[i] = (unsigned) x[i];
  }
  return result;
}

// creates a matrix of unsigned
arma::umat unsignMx(const Rcpp::IntegerMatrix & m)
{
  arma::umat result(m.nrow(), m.ncol());
  for (unsigned i=0; i<result.n_rows; i++)
  {
    for (unsigned j=0; j<result.n_cols; j++)
    {
      result(i,j) = (unsigned) m(i,j);
    }
  }
  return result;
}

// creates a random matrix for the initialization of z 
arma::umat randomIndices(const unsigned n, int k)
{
  Rcpp::NumericVector xR = Rcpp::runif(n, 0, k);
  arma::umat indices = arma::zeros<arma::umat>(n+1,k);
#pragma omp parallel for shared(indices)
  for (unsigned i=0; i<n; i++)
  {
    unsigned j = (unsigned)xR[i];
    indices(i,j) = 1;
  }
  return indices;
}

// returns a vector of samples from a gamma distr with parameters shape and rate specified as vectors
arma::rowvec rgamma(const arma::rowvec & shape, const arma::rowvec & rate)
{
  arma::rowvec result(shape.n_elem);
  for (unsigned i=0; i<shape.n_elem; i++)
  {
    result[i] = ::Rf_rgamma(shape[i], 1 / rate[i]);
  }
  return result;
}

// returns a vector with samples from a normal with mu and sd specified in vectors
arma::rowvec rnorm(const arma::rowvec & mean, const arma::rowvec & stddev)
{
  Rcpp::NumericVector xR = Rcpp::rnorm(mean.n_elem);
  arma::rowvec x(xR.begin(), xR.size(), false);
  return x % stddev + mean; // element-wise multiplication and addition
}

// 1d case  
// returns a matrix that on column[i] has the pdf of a normal distribution of mean[i], sd[i] in y[j] return size = (n_samples,k)
// we pass yunique and ymatch to avoid replicating calculations
// this automatically returns the log_density
arma::mat dnorm(const Rcpp::NumericVector & yunique, const arma::uvec & ymatch,
                const arma::rowvec & mean, const arma::rowvec & stddev)
{
  arma::vec prob(yunique.size());
  arma::mat probMx(ymatch.n_elem,mean.n_elem);
  for (unsigned i=0; i<mean.n_elem; i++)
  {
    for (int j=0; j<yunique.size(); j++)
    {
      prob[j] = ::Rf_dnorm4(yunique[j],mean[i],stddev[i],1); // the 1 is for the logscale = true
    }
    probMx.col(i) = prob.elem(ymatch);
  }
  return probMx;
}

// Multivariate normal random generation mean mu covariance matrix sigma
arma::vec rmvnrm(const arma::vec& mu, const arma::mat& sigma) {
  int ncols = sigma.n_cols;
  arma::vec Y = arma::randn(ncols);
  return mu + (Y.t() * arma::chol(sigma)).t();
}

// multivariate normal density mean mu cov matrix S, this returns a colvec of the evaluaions of the density in each y
// vectorized
arma::vec dmvnorm(const arma::mat& x, const arma::vec& mu,
                  const arma::mat& S, const bool log_p = false) {
  unsigned m = x.n_rows, n = x.n_cols;
  double det_S = arma::det(S);
  arma::mat S_inv = S.i();
  arma::vec result(n);
  arma::vec X(m);
  if ( log_p ) {
    double P = -1.0 * (x.n_rows/2.0) * M_LN_2PI - 0.5 * log(det_S);
    for (unsigned i = 0; i < n; ++i ) {
      X = x.col(i) - mu;
      result[i] = arma::as_scalar(P - 0.5 * X.t() * S_inv * X);
    }
    return result;
  }
  double P = 1.0 / sqrt(pow(M_2PI, m) * det_S);
  for (unsigned i = 0; i < n; ++i ) {
    X = x.col(i) - mu;
    result[i] = arma::as_scalar(P * exp(-0.5 * X.t() * S_inv * X));
  }
  return result;
}

// multivariate normal sample with mean mu and cov matrix S
arma::vec rmvnorm(const arma::vec& mu,const arma::mat& S) {
  unsigned m = S.n_cols;
  arma::mat result(1, m);
  arma::rowvec Mu = mu.t();
  for (unsigned j = 0; j < m; ++j ) {
    result(0, j) = R::rnorm(0.0, 1.0);
  }
  result = result * arma::chol(S);
  result.row(0) = result.row(0) + Mu;
  result.reshape(m,1);
  arma::vec res(result);
  return res;
}

// returns a matrix that on column[i] has the pdf of a normal distribution of mus[i], sigmas[i] in y[j] in logscales
arma::mat dnorm_field(const arma::mat & Y, const arma::mat & mus, const arma::cube & lambdas){
  arma::mat probMx(Y.n_cols,mus.n_cols);
  
  for (unsigned i=0; i<mus.n_cols; i++){
    
    probMx.col(i) = dmvnorm(Y,mus.col(i),lambdas.slice(i).i(),1); // the 1 is for logscale = true
  }
  
  return probMx;
}

// samples from the wishart distribution parametrization s.t. the expected value is df*S
arma::mat rwishart(unsigned int df, const arma::mat& S) {
  // Dimension of returned wishart
  unsigned int m = S.n_rows;
  
  // Z composition:
  // sqrt chisqs on diagonal
  // random normals below diagonal
  // misc above diagonal
  arma::mat Z(m,m);
  
  // Fill the diagonal
  for(unsigned int i = 0; i < m; i++) {
    Z(i,i) = sqrt(R::rchisq(df-i));
  }
  
  // Fill the lower matrix with random guesses
  for(unsigned int j = 0; j < m; j++) {  
    for(unsigned int i = j+1; i < m; i++) {    
      Z(i,j) = R::rnorm(0,1);
    }}
  
  // Lower triangle * chol decomp
  arma::mat C = arma::trimatl(Z).t() * arma::chol(S);
  
  // Return random wishart
  return C.t()*C;
}

// samples from the inverse wishart distribution
arma::mat rinvwish(unsigned int df, const arma::mat& Sig) {
  return rwishart(df,Sig.i()).i();
}

// the log sum of a vector of logs
// http://jblevins.org/log/log-sum-exp
double sum_logs(arma::vec log_prob)
{
  double suml = 0.0;
  double maxl = log_prob.max();
  for (unsigned i=0; i < log_prob.n_elem; i++)
  {
    if (arma::is_finite(log_prob(i)))
      suml += exp(log_prob(i) - maxl);
  }
  return log(suml) + maxl;
}

// updates the weight of the mixtures in the GMM 
arma::rowvec gibbsDirichlet(const arma::rowvec & nZ, const arma::rowvec & pr_lambda)
{
  arma::rowvec shape = nZ + pr_lambda;
  arma::rowvec sim = rgamma(shape, arma::ones<arma::rowvec>(shape.n_elem));
  return sim/arma::sum(sim);
}

// updates the means in the gibbs sampler 1D model
arma::rowvec gibbsMeans(const arma::rowvec & nZ, const arma::rowvec & sumY,
                        const arma::rowvec & pr_mu, const arma::rowvec & pr_mu_tau,
                        const arma::rowvec & sigma)
{
  arma::rowvec oldTau = arma::pow(sigma, -2);
  arma::rowvec newTau = pr_mu_tau + oldTau % nZ;
  arma::rowvec mean = (pr_mu_tau%pr_mu + oldTau%sumY) / newTau; // element-wise
  return rnorm(mean, arma::pow(newTau, -0.5));
}

// updates the means in the gibbs sampler nD model 
// to be vectorized 
arma::mat mdgibbsMeans(const arma::rowvec & nZ, const arma::mat & sumY,
                       const arma::mat & pr_mu, const arma::cube & pr_mu_sigma_inv,
                       const arma::cube & lambdas)
{
  arma::mat ret = arma::zeros(pr_mu.n_rows,pr_mu.n_cols);
  
  for (unsigned j = 0; j < pr_mu.n_cols; j++){
    if(nZ[j]==0){
      ret.col(j) = rmvnrm(pr_mu.col(j), pr_mu_sigma_inv.slice(j).i());
    }
    else{
      arma::mat Bj = (nZ(j)*lambdas.slice(j) + pr_mu_sigma_inv.slice(j)).i();
      arma::colvec ybarj = sumY.col(j)/nZ(j);
      arma::colvec bj = Bj*(nZ(j)*lambdas.slice(j)*ybarj + pr_mu_sigma_inv.slice(j)*pr_mu.col(j));
      ret.col(j) = rmvnrm(bj, Bj);
    }
  }
  return ret;
}

// updates the standard deviations in the gibbs sampler 1D model
arma::rowvec gibbsStdDev(const arma::rowvec & nZ, const arma::rowvec & sumY,
                         const arma::rowvec & sqDiff, const arma::rowvec & pr_sd_nu,
                         const arma::rowvec & pr_sd_SS, const arma::rowvec & mean)
{
  // avoid dividing by zero if one of the mixture components is empty
  arma::rowvec Ybar(sumY.n_elem);
  for (unsigned j=0; j < sumY.n_elem; j++)
  {
    if (nZ[j] == 0) Ybar[j] = 0;
    else Ybar[j] = sumY[j] / nZ[j];
  }
  arma::rowvec shape = (pr_sd_nu + nZ)/2;
  arma::rowvec rate = (pr_sd_SS + sqDiff + nZ%arma::square(Ybar - mean))/2;
  return arma::pow(rgamma(shape, rate), -0.5);
}


// otuer product of two vectors
arma::mat outer(const arma::colvec & a, const arma::colvec & b){
  arma::mat ret = arma::zeros(a.n_elem,b.n_elem);
  for(unsigned i = 0; i < a.n_elem; i++){
    for(unsigned j = 0; j < b.n_elem; j++){
      ret(i,j) = a(i)*b(j);
    }
  }
  return ret;
}

// updates the lambdaj in the gibbs sampler mD model
arma::cube gibbsLambda(const arma::rowvec & nZ, const arma::mat & Y, const arma::umat & z,
                               const arma::cube & pr_sigma_v0, const arma::rowvec & pr_sigma_n0, const arma::mat & mu)
{
  arma::cube ret = arma::zeros(pr_sigma_v0.n_rows,pr_sigma_v0.n_cols,pr_sigma_v0.n_slices);
  arma::rowvec np = nZ + pr_sigma_n0;
  arma::cube A = arma::zeros(pr_sigma_v0.n_rows,pr_sigma_v0.n_cols,pr_sigma_v0.n_slices);
  
  for(unsigned i = 0; i < Y.n_cols;i++){
    unsigned j = z.row(i).index_max();
    A.slice(j) = A.slice(j) + outer((Y.col(i)-mu.col(j)),(Y.col(i)-mu.col(j)));
  }
  
  for (unsigned j=0; j < nZ.n_elem; j++)
  {
    if(nZ(j)!=0){
      arma::mat Vp = (pr_sigma_v0.slice(j).i() + A.slice(j)).i();
      ret.slice(j) = rwishart(np(j), Vp);
    }
    else
      ret.slice(j) = rwishart(pr_sigma_n0(j),pr_sigma_v0.slice(j));
  }
  
  return ret;
}

// updates labels Z and count of allocations alloc
void classify(arma::umat & z, arma::umat & alloc, const arma::rowvec & lambda, const arma::mat & log_xfield)
{
  const Rcpp::NumericVector randU = Rcpp::runif(log_xfield.n_rows);
  
#pragma omp parallel for
  for (unsigned i=0; i < log_xfield.n_rows; i++)
  {
    // compute posterior probability for each label j
    arma::vec log_prob(z.n_cols);
    for (unsigned j=0; j < z.n_cols; j++)
    {
      log_prob(j) = log_xfield(i,j) + lambda(j);
    }
    double total_llike = sum_logs(log_prob);
    
    // update labels Z
    double cumProb = 0.0;
    z.row(i).zeros();
    for (unsigned j=0; j < log_prob.n_elem; j++)
    {
      cumProb += exp(log_prob[j] - total_llike);
      if (randU[i] < cumProb)
      {
        z(i,j) = 1;
        alloc(i,j) += 1;
        break;
      }
    }
  }
}

// updates nZ count of each cluster, sumY sum of each cluster value and the square difference between y and ybar
void updateStats(const arma::colvec & y, const arma::umat & z,
                 arma::rowvec & nZ, arma::rowvec & sumY, arma::rowvec & sqDiff)
{
  nZ.zeros();
  sumY.zeros();
  sqDiff.zeros();
  for (unsigned i=0; i < y.n_elem; i++)
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(i,j)==1)
      {
        nZ[j]++;
        sumY[j] += y[i];
      }
    }
  }
  arma::rowvec ybar = sumY/nZ;
  for (unsigned i=0; i < y.n_elem; i++)
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(i,j)==1)
      {
        sqDiff[j] += pow(y[i] - ybar[j],2);
      }
    }
  }
}

// updates nZ count of each cluster, sumY sum of each cluster value in the mD potts model 
void mdupdateStats(const arma::mat & Y, const arma::umat & z,
                   arma::rowvec & nZ, arma::mat & sumY)
{
  nZ.zeros();
  sumY.zeros();
  
  for (unsigned i=0; i < Y.n_cols; i++)
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(i,j)==1)
      {
        nZ[j]++;
        sumY.col(j) += Y.col(i);
      }
    }
  }
  
}


// monodimensional gaussian mixture model gibbs sampler
// inputs:
// inputs:
// yS a vector of data
// itS total number of iterations
// biS burn in iterations
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (vector)
//    mu.sd: the prior sd of the normal distr of the mean of each cluster (vector)
//    sigma: the shape parameter of the gamma for the sd prior (vector)
//    sigma.nu: the rate parameters of the gamma for the sd prior (vector)
//    lambda: the parameters of the dirichlet for the weights of the mixture (vector)
// [[Rcpp::export]]
SEXP gibbsGMM1d(SEXP yS, SEXP itS, SEXP biS, SEXP prS) {
  BEGIN_RCPP
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS);
  
  Rcpp::NumericVector yR(yS);       // creates Rcpp vector from SEXP
  Rcpp::NumericVector yunique = Rcpp::unique(yR);
  Rcpp::IntegerVector ymatchR = Rcpp::match(yR, yunique);
  // no easy conversion from IntegerVector to uvec
  arma::uvec ymatch = unsign(ymatchR - 1);
  arma::colvec y(yR.begin(), yR.size(), false); // reuses memory and avoids extra copy
  
  Rcpp::List prR(prS);
  unsigned k = Rcpp::as<unsigned>(prR["k"]);
  Rcpp::NumericVector prior_lambda = prR["lambda"];
  Rcpp::NumericVector prior_mu = prR["mu"];
  Rcpp::NumericVector prior_mu_sd = prR["mu.sd"];
  Rcpp::NumericVector prior_sd = prR["sigma"];
  Rcpp::NumericVector prior_sd_nu = prR["sigma.nu"];
  arma::rowvec pr_lambda(prior_lambda.begin(), prior_lambda.size(), false);
  arma::rowvec pr_mu(prior_mu.begin(), prior_mu.size(), false);
  arma::rowvec pr_mu_sd(prior_mu_sd.begin(), prior_mu_sd.size(), false);
  arma::rowvec pr_mu_tau = arma::pow(pr_mu_sd, -2);
  arma::rowvec pr_sd(prior_sd.begin(), prior_sd.size(), false);
  arma::rowvec pr_sd_nu(prior_sd_nu.begin(), prior_sd_nu.size(), false);
  arma::rowvec pr_sd_SS = pr_sd_nu % arma::square(pr_sd); // Schur product
  
  Rcpp::RNGScope scope;               // initialize random number generator
  arma::rowvec mu = rnorm(pr_mu,pr_mu_sd);
  arma::rowvec sd = arma::pow(rgamma(pr_sd_nu/2, pr_sd_SS/2), -0.5);
  arma::rowvec wt(k);
  for (unsigned j=0; j<k; j++)
  {
    wt[j] = unif_rand();
  }
  wt = wt/arma::sum(wt); // normalize to sum to 1
  arma::umat z = randomIndices(y.n_elem, k);
  
  arma::mat mu_save = arma::zeros(niter,k); // history of simulated values of mu
  arma::mat sd_save = arma::zeros(niter,k); // history of simulated values of sigma
  arma::mat wt_save = arma::zeros(niter,k); // history of simulated values of lambda
  arma::umat alloc  = arma::zeros<arma::umat>(y.n_elem, k);
  arma::rowvec nZ(k), sumY(k), sqDiff(k);
  
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    
    // update labels
    arma::mat alpha = dnorm(yunique, ymatch, mu, sd);
    classify(z, alloc, arma::log(wt), alpha);
    updateStats(y, z, nZ, sumY, sqDiff);
    
    // update means
    mu = gibbsMeans(nZ, sumY, pr_mu, pr_mu_tau, sd);
    mu_save.row(it) = mu;
    
    // update standard deviations
    sd = gibbsStdDev(nZ, sumY, sqDiff, pr_sd_nu, pr_sd_SS, mu);
    sd_save.row(it) = sd;
    
    // update mixture weights
    wt = gibbsDirichlet(nZ, pr_lambda);
    wt_save.row(it) = wt;
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,    // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,  // history of simulated values of mu
    Rcpp::Named("sigma") = sd_save,  // history of simulated values of sigma
    Rcpp::Named("z")     = z,        // final allocation matrix
    Rcpp::Named("w") = wt_save  // history of simulated values of lambda
  );
  END_RCPP
}


// multidimensional GMM gibbs sampler
// inputs:
// yS a matrix of data, samples on the cols, features on the rows
// itS total number of iterations
// biS burn in iterations
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (matrix)
//    mu.sigma: the covariance matrices of the normal distr of the mean of each cluster (3D array)
//    sigma.V0: the V= matrix for the wishart prior of the covariance matrices (3D array)
//    sigma.n0: the n0 parameters of the wishart prior for the covariance matrices (vector)
//    lambda: the parameters of the dirichlet for the weights of the mixture (vector)
// [[Rcpp::export]]
SEXP gibbsGMMmd(SEXP yS, SEXP itS, SEXP biS, SEXP prS) {
  BEGIN_RCPP
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS);
  
  //directly creating arma matrix/cube from SEXP
  arma::mat Y = Rcpp::as<arma::mat>(yS); 
  Rcpp::List prR(prS);
  
  // checks
  int d = Y.n_rows;
  if (prR.length() == 0)
  {
    throw std::invalid_argument("prior is empty");
  }
  int nvert = Y.n_cols;
  
  unsigned k = Rcpp::as<unsigned>(prR["k"]);
  Rcpp::NumericVector prior_lambda = prR["lambda"];
  arma::mat pr_mu = Rcpp::as<arma::mat>(prR["mu"]);
  arma::cube pr_mu_sigma = Rcpp::as<arma::cube>(prR["mu.sigma"]);
  arma::cube pr_sigma_v0 = Rcpp::as<arma::cube>(prR["sigma.V0"]);
  arma::rowvec pr_sigma_n0 = Rcpp::as<arma::rowvec>(prR["sigma.n0"]);
  arma::rowvec pr_lambda(prior_lambda.begin(), prior_lambda.size(), false); // avoid reusing memory
  
  // we initialize the mu and sigma from the corresponding distributions, 
  //we could pass it if needed or initialize it with the expected value 
  arma::mat mu(d,k);
  arma::cube sigma(d,d,k);
  for(unsigned i = 0; i < k; i++){
    mu.col(i) = rmvnorm(pr_mu.col(i),pr_mu_sigma.slice(i));
    sigma.slice(i) = rinvwish(pr_sigma_n0(i),pr_sigma_v0.slice(i));
  }
  
  // to avoid extra inversion of the sigma, pr_mu_sigma and pr_sigma
  arma::cube lambads = arma::zeros(d,d,k);
  arma::cube pr_mu_sigma_inv = arma::zeros(d,d,k); //B0^-1
  for (unsigned j = 0; j < pr_mu_sigma.n_slices; j++) {
    lambads.slice(j) = sigma.slice(j).i();
    pr_mu_sigma_inv.slice(j) = pr_mu_sigma.slice(j).i();
  }
  
  Rcpp::RNGScope scope;               // initialize random number generator
  //randomly initializing the weights vector, should I initialize them from the prior distribution? 
  arma::rowvec wt(k);
  for (unsigned j=0; j<k; j++)
  {
    wt[j] = unif_rand();
  }
  wt = wt/arma::sum(wt); // normalize to sum to 1
  arma::umat z = randomIndices(nvert, k);
  
  arma::cube mu_save = arma::zeros(d,k,niter);              // history of simulated values of mu
  arma::field<arma::cube> sigmas_save(niter);               // history of simulated values of sigma
  arma::mat wt_save = arma::zeros(niter,k);                 // history of simulated values of lambda
  arma::umat alloc = arma::zeros<arma::umat>(nvert, k); // number of allocations of each elem to each cluster 
  arma::rowvec nZ(k);                                       // number of element in each cluster
  arma::mat sumY(d,k);                                      // sum of the values in each cluster, used to compute the means per cluster
  mdupdateStats(Y, z, nZ, sumY);                            // updating the data structures following the random initialization
  
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    
    // update labels
    arma::mat alpha = dnorm_field(Y, mu, lambads);
    classify(z, alloc, arma::log(wt), alpha);
    mdupdateStats(Y, z, nZ, sumY);
    
    // update means
    mu = mdgibbsMeans(nZ, sumY, pr_mu, pr_mu_sigma_inv, lambads);
    mu_save.slice(it) = mu;
    
    // update standard deviations
    lambads = gibbsLambda(nZ, Y ,z, pr_sigma_v0, pr_sigma_n0, mu);
    arma::cube temp_sigma(d,d,k);
    for( unsigned int i = 0; i < k; i++){
      temp_sigma.slice(i) = lambads.slice(i).i();
    }
    sigmas_save(it) = temp_sigma;
    
    // update mixture weights
    wt = gibbsDirichlet(nZ, pr_lambda);
    wt_save.row(it) = wt;
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,    // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,  // history of simulated values of mu
    Rcpp::Named("sigma") = sigmas_save,  // history of simulated values of sigma
    Rcpp::Named("z")     = z,        // final allocation matrix
    Rcpp::Named("w") = wt_save  // history of simulated values of lambda
  );
  END_RCPP
}


// wrapper function for the gibbs sampler for the GMM
// this identifies wheter we are in the 1d or md case and calls the appropriate function
// [[Rcpp::export]]
SEXP GibbsGMM(SEXP yS, SEXP itS, SEXP biS, SEXP prS) {
  if(Rf_isMatrix(yS))
    return gibbsGMMmd(yS, itS, biS, prS);
  else if(Rf_isVector(yS))
    return gibbsGMM1d(yS, itS, biS, prS);
  else {
    Rcpp::Rcout << " the data must be either a vector or a matrix " << "\n";
    return Rcpp::List(); //return an empty list
  }
}

