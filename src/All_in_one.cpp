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

// the sufficient statistic of the Potts model: the number of identical pairs of neighbours
unsigned sum_ident(const arma::umat & z, const arma::umat & neigh, const std::vector<arma::uvec> & blocks)
{
  unsigned total = 0;
  const arma::uvec block = blocks[0];
#pragma omp parallel for reduction(+:total)
  for (unsigned i=0; i < block.n_elem; i++)
  {    
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(block(i),j) == 1)
      {
        unsigned sum_neigh = 0;
        for (unsigned k=0; k < neigh.n_cols; k++)
        {
          sum_neigh += z(neigh(block(i),k),j);
        }
        total += sum_neigh;
      }
    }
  }
  return total;
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

// updates the lambdaj in the gibbs sampler nD model (nZ , SSY, z, pr_sigma_v0, pr_sigma_n0)
arma::cube gibbsLambda(const arma::rowvec & nZ, const arma::cube & SSY,
                       const arma::umat & z, const arma::cube & pr_sigma_v0,
                       const arma::rowvec & pr_sigma_n0)
{
  // avoid dividing by zero if one of the mixture components is empty
  arma::cube ret = arma::zeros(pr_sigma_v0.n_rows,pr_sigma_v0.n_cols,pr_sigma_v0.n_slices);
  arma::rowvec np = nZ + pr_sigma_n0;
  
  for (unsigned j=0; j < nZ.n_elem; j++)
  {
    if(nZ(j)!=0){
      arma::mat Vp = (pr_sigma_v0.slice(j).i() + SSY.slice(j)).i();
      ret.slice(j) = rwishart(np(j), Vp);
    }
    else
      ret.slice(j) = rwishart(pr_sigma_n0(j),pr_sigma_v0.slice(j));
  }
  
  return ret;
}

// updates labels Z and count of allocations alloc
void gibbsLabels(const arma::umat & neigh, const std::vector<arma::uvec> & blocks,
                 arma::umat & z, arma::umat & alloc, const double beta,
                 const arma::mat & log_xfield)
{
  const Rcpp::NumericVector randU = Rcpp::runif(neigh.n_rows);
  
  // the blocks are conditionally independent
  for (unsigned b=0; b < blocks.size(); b++)
  {
    const arma::uvec block = blocks[b];
    // for each pixel in the block
#pragma omp parallel for
    for (unsigned i=0; i < block.size(); i++)
    {
      // compute posterior probability for each label j
      arma::vec log_prob(z.n_cols);
      for (unsigned j=0; j < z.n_cols; j++)
      {
        unsigned sum_neigh = 0;
        for (unsigned k=0; k < neigh.n_cols; k++)
        {
          sum_neigh += z(neigh(block[i],k),j);
        }
        log_prob[j] = log_xfield(block[i],j) + beta*sum_neigh;
      }
      double total_llike = sum_logs(log_prob);
      
      // update labels Z
      double cumProb = 0.0;
      z.row(block[i]).zeros();
      for (unsigned j=0; j < log_prob.n_elem; j++)
      {
        cumProb += exp(log_prob[j] - total_llike);
        if (randU[block[i]] < cumProb)
        {
          z(block[i],j) = 1;
          alloc(block[i],j) += 1;
          break;
        }
      }
    }
  }
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

// updates nZ count of each cluster, sumY sum of each cluster value in the mD potts model;
void mdupdateStats(const arma::mat & Y, const arma::umat & z, const arma::mat & mu,
                   arma::rowvec & nZ, arma::mat & sumY, arma::cube & SSY)
{
  nZ.zeros();
  sumY.zeros();
  SSY.zeros();
  
#pragma omp parallel for
  for (unsigned i=0; i < Y.n_cols; i++)
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(i,j)==1)
      {
        nZ[j]++;
        sumY.col(j) += Y.col(i);
        SSY.slice(j) += outer(Y.col(i) - mu.col(j),Y.col(i) - mu.col(j));
      }
    }
  }
  
}

// Computes the number of neighbouring pixels allocated to component j, for pixel i. 
void neighbj(arma::mat & ne, arma::uvec & e, const arma::umat & z, const arma::umat & neigh)
{
#pragma omp parallel for
  for (unsigned i=0; i < z.n_rows-1; i++) // since the last is the element put in place if the pixel is in the border
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      unsigned sum_neigh = 0;
      for (unsigned k=0; k < neigh.n_cols; k++)
      {
        sum_neigh += z(neigh(i,k),j);
      }
      ne(j,i) = (double)sum_neigh;
      if (z(i,j) == 1)
      {
        e[i] = j;
      }
    }
  }
}

// pseudo log-likelihood of the Potts model
double pseudolike(const arma::mat & ne, const arma::uvec & e, const double b, const unsigned n, const unsigned k)
{
  double num = 0.0;
  double denom = 0.0;
#pragma omp parallel for reduction(+:num,denom)
  for (unsigned i=0; i < n; i++)
  {
    num=num+ne(e[i],i);
    double tdenom=0.0;
    for (unsigned j=0; j < k; j++)
    {
      tdenom=tdenom+exp(b*ne(j,i));
    }
    denom=denom+log(tdenom);
  }
  return b*num-denom;
}

//random walk metropolis hastings step with bounds given in prior
double rwmh(const double mean, const double stddev, const double prior[2])
{
  double proposal = ::Rf_rnorm(mean, stddev);
  if (proposal < prior[0])
  {
    proposal = std::min(prior[1], prior[0] + prior[0] - proposal);
  }
  if (proposal > prior[1])
  {
    proposal = std::max(prior[0], prior[1] + prior[1] - proposal);
  }
  return proposal;
}

// updates inverse temperature using point pseudo-likelihood
// Tobias Ryde'n & D. M. Titterington (1998)
unsigned pseudoBeta(const arma::umat & neigh, const std::vector<arma::uvec> & blocks,
                    const arma::umat & z, double & beta, const double prior_beta[2],
                                                                                const double bw)
{
  // random walk proposal for B' ~ N(B, 0.01^2)
  double bprime = rwmh(beta, bw, prior_beta);
  
  arma::uvec e(z.n_rows-1); // allocation vector
  arma::mat ne = arma::zeros(z.n_cols, z.n_rows-1);  // counts of like neighbours
  neighbj(ne, e, z, neigh);
  
  double log_ratio = pseudolike(ne, e, bprime, z.n_rows-1, z.n_cols) - pseudolike(ne, e, beta, z.n_rows-1, z.n_cols);
  
  // accept/reject
  if (log(unif_rand()) < log_ratio)
  {// accept
    beta = bprime;
    return 1;
  }
  //reject
  return 0;
}

// Given the allocation matrix z it returns the allocation vecor vec to be salso compatible
void salsovec(const arma::umat& z, arma::rowvec & allocvec)
{
#pragma omp parallel for
  for (unsigned i=0; i < z.n_rows-1; i++) // since the last is the element put in place if the pixel is in the border
  {
    for (unsigned j=0; j < z.n_cols; j++)
    {
      if (z(i,j) == 1)
      {
        allocvec[i] = j;
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
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)
SEXP gibbsGMM1d(SEXP yS, SEXP prS, SEXP itS, SEXP biS,SEXP salitS) {
  BEGIN_RCPP
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  
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
  int nvert = y.n_elem;
  arma::mat mu_save = arma::zeros(niter,k); // history of simulated values of mu
  arma::mat sd_save = arma::zeros(niter,k); // history of simulated values of sigma
  arma::mat wt_save = arma::zeros(niter,k); // history of simulated values of lambda
  arma::umat alloc  = arma::zeros<arma::umat>(nvert, k);
  arma::rowvec nZ(k), sumY(k), sqDiff(k);
  arma::mat salout = arma::zeros(nsalso,nvert); // history of allocations salso compatible
  arma::rowvec allocvec = arma::zeros(nvert);
  
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
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
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
    Rcpp::Named("lambda") = wt_save,  // history of simulated values of lambda
    Rcpp::Named("salso_hist") = salout    // history of allocations salso compatible
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
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)
SEXP gibbsGMMmd(SEXP yS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS) {
  BEGIN_RCPP
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  
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
  arma::cube SSY = arma::zeros(d,d,k);                      // the matrix of sum (y-mu)(y-mu)^t for each cluster 
  mdupdateStats(Y, z, mu, nZ, sumY, SSY);                   // updating the data structures following the random initialization
  arma::mat salout = arma::zeros(nsalso,nvert);             // history of allocations salso compatible
  arma::rowvec allocvec(nvert);
  
  
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
    mdupdateStats(Y, z, mu, nZ, sumY, SSY);
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
    // update means
    mu = mdgibbsMeans(nZ, sumY, pr_mu, pr_mu_sigma_inv, lambads);
    mu_save.slice(it) = mu;
    
    // update standard deviations
    lambads = gibbsLambda(nZ, SSY, z, pr_sigma_v0, pr_sigma_n0);
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
    Rcpp::Named("lambda") = wt_save,  // history of simulated values of lambda
    Rcpp::Named("salso_hist") = salout   // history of allocations salso compatible
  );
  END_RCPP
}


// wrapper function for the gibbs sampler for the GMM
// this identifies wheter we are in the 1d or md case and calls the appropriate function
// [[Rcpp::export]]
SEXP GibbsGMM(SEXP yS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS) {
  if(Rf_isMatrix(yS))
    return gibbsGMMmd(yS, prS, itS, biS, salitS);
  else if(Rf_isVector(yS))
    return gibbsGMM1d(yS, prS, itS, biS, salitS);
  else {
    Rcpp::Rcout << " the data must be either a vector or a matrix " << "\n";
    return Rcpp::List(); //return an empty list
  }
}



// monodimensional Potts model gibbs sampler
// inputs:
// yS a vector of data
// betaS is the value of beta to use for the gibbs Distribution
// muS is a vector containing the starting point of the mu of the chain,
// sigmaS is a 3D array containing the starting values of the covariance matrices of the potts model, one per cluster
// nS is the neighbours object from BayesImageS
// bs is the blocks object from BayesImageS
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (vector)
//    mu.sd: the prior sd of the normal distr of the mean of each cluster (vector)
//    sigma the shape parameter of the gamma for the sd prior (vector)
//    sigma.nu the rate parameters of the gamma for the sd prior (vector)
// itS total number of iterations
// biS burn in iterations
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)

SEXP gibbsPotts1d(SEXP yS, SEXP betaS, SEXP muS, SEXP sdS, SEXP nS, SEXP bS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS) {
  BEGIN_RCPP
  Rcpp::NumericVector yR(yS), muR(muS), sdR(sdS); // creates Rcpp vector from SEXP
  Rcpp::IntegerMatrix nR(nS);             // creates Rcpp matrix from SEXP
  Rcpp::List bR(bS), prR(prS);
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  double beta = Rcpp::as<double>(betaS);
  
  Rcpp::NumericVector yunique = Rcpp::unique(yR);
  Rcpp::IntegerVector ymatchR = Rcpp::match(yR, yunique);
  // no easy conversion from IntegerVector to uvec
  arma::uvec ymatch = unsign(ymatchR) - 1;
  arma::umat neigh = unsignMx(nR) - 1;
  
  // block index vectors are not symmetric
  std::vector<arma::uvec> blocks;
  blocks.reserve(bR.length());
  for (int b=0; b<bR.length(); b++)
  {
    Rcpp::IntegerVector block = bR[b];
    arma::uvec ublock = unsign(block - 1);
    blocks.push_back(ublock);
  }
  
  if (prR.length() == 0)
  {
    throw std::invalid_argument("prior is empty");
  }
  int nvert = nR.nrow();
  if (nvert != yR.size())
  {
    throw std::invalid_argument("mismatch between observations and neighbourhood matrix");
  }
  int k = Rcpp::as<int>(prR["k"]);
  Rcpp::NumericVector prior_mu = prR["mu"];
  Rcpp::NumericVector prior_mu_sd = prR["mu.sd"];
  Rcpp::NumericVector prior_sd = prR["sigma"];
  Rcpp::NumericVector prior_sd_nu = prR["sigma.nu"];
  
  arma::colvec y(yR.begin(), yR.size(), false); // reuses memory and avoids extra copy
  arma::rowvec mu(muR.begin(), muR.size(), false);
  arma::rowvec sd(sdR.begin(), sdR.size(), false);
  arma::rowvec pr_mu(prior_mu.begin(), prior_mu.size(), false);
  arma::rowvec pr_mu_sd(prior_mu_sd.begin(), prior_mu_sd.size(), false);
  arma::rowvec pr_mu_tau = arma::pow(pr_mu_sd, -2);
  arma::rowvec pr_sd(prior_sd.begin(), prior_sd.size(), false);
  arma::rowvec pr_sd_nu(prior_sd_nu.begin(), prior_sd_nu.size(), false);
  arma::rowvec pr_sd_SS = pr_sd_nu % arma::square(pr_sd); // Schur product
  
  arma::umat z = randomIndices(nvert, k);
  arma::mat mu_save = arma::zeros(niter, k);   // history of simulated values of mu
  arma::mat sd_save = arma::zeros(niter, k);   // history of simulated values of sigma
  arma::vec sum_save = arma::zeros(niter);     // sum of identical neighbours
  arma::umat alloc = arma::zeros<arma::umat>(nR.nrow(), k);
  arma::rowvec nZ(k), sumY(k), sqDiff(k);
  arma::mat salout = arma::zeros(nsalso,nvert); // history of allocations salso compatible
  arma::rowvec allocvec = arma::zeros(nvert);
  
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    // check for interrupt every thousand iterations
    if (it % 1000 == 0) Rcpp::checkUserInterrupt();
    
    // update labels
    arma::mat alpha = dnorm(yunique, ymatch, mu, sd);
    gibbsLabels(neigh, blocks, z, alloc, beta, alpha);
    updateStats(y, z, nZ, sumY, sqDiff);
    sum_save(it) = sum_ident(z, neigh, blocks);
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
    // update means
    mu = gibbsMeans(nZ, sumY, pr_mu, pr_mu_tau, sd);
    mu_save.row(it) = mu;
    
    // update standard deviations
    sd = gibbsStdDev(nZ, sumY, sqDiff, pr_sd_nu, pr_sd_SS, mu);
    sd_save.row(it) = sd;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,         // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,       // sample of mu
    Rcpp::Named("sigma") = sd_save,       // sample of sigma
    Rcpp::Named("z")     = z,             // final sample from Gibbs distribution
    Rcpp::Named("sum") = sum_save,        // sum of identical neighbours
    Rcpp::Named("salso_hist") = salout    // history of allocations salso compatible
  );
  END_RCPP
}


// multidimensional Potts model gibbs sampler
// inputs:
// yS a matrix of data, samples on the cols, features on the rows
// betaS is the value of beta to use for the gibbs Distribution
// muS is a matrix containing the starting point of the mu of the chain, one col for cluster
// sigmaS is a 3D array containing the starting values of the covariance matrices of the potts model, one per cluster
// nS is the neighbours object from BayesImageS
// bs is the blocks object from BayesImageS
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (matrix)
//    mu.sigma the covariance matrices of the normal distr of the mean of each cluster (3D array)
//    sigma.V0 the V= matrix for the wishart prior of the covariance matrices (3D array)
//    sigma.n0 the n0 parameters of the wishart prior for the covariance matrices (vector)
// itS total number of iterations
// biS burn in iterations
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)

SEXP gibbsPottsmd(SEXP yS, SEXP betaS, SEXP muS, SEXP sigmaS, SEXP nS, SEXP bS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS) {
  BEGIN_RCPP
  //directly creating arma matrix/cube from SEXP
  arma::mat Y = Rcpp::as<arma::mat>(yS); 
  arma::mat mu = Rcpp::as<arma::mat>(muS);
  arma::cube sigma = Rcpp::as<arma::cube>(sigmaS);
  
  Rcpp::IntegerMatrix nR(nS);// creates Rcpp matrix from SEXP
  Rcpp::List bR(bS), prR(prS);
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  double beta = Rcpp::as<double>(betaS);
  
  // no easy conversion from IntegerVector to uvec
  arma::umat neigh = unsignMx(nR) - 1;
  
  // block index vectors are not symmetric
  std::vector<arma::uvec> blocks;
  blocks.reserve(bR.length());
  for (int b=0; b<bR.length(); b++)
  {
    Rcpp::IntegerVector block = bR[b];
    arma::uvec ublock = unsign(block - 1);
    blocks.push_back(ublock);
  }
  
  // checks
  int d = Y.n_rows;
  if (prR.length() == 0)
  {
    throw std::invalid_argument("prior is empty");
  }
  int nvert = nR.nrow();
  if (nvert != Y.n_cols)
  {
    throw std::invalid_argument("Mismatch between observations and neighbourhood matrix");
  }
  
  int k = Rcpp::as<int>(prR["k"]);
  arma::mat pr_mu = Rcpp::as<arma::mat>(prR["mu"]);
  arma::cube pr_mu_sigma = Rcpp::as<arma::cube>(prR["mu.sigma"]);
  arma::cube pr_sigma_v0 = Rcpp::as<arma::cube>(prR["sigma.V0"]);
  arma::rowvec pr_sigma_n0 = Rcpp::as<arma::rowvec>(prR["sigma.n0"]);
  
  // to avoid extra inversion of the sigma, pr_mu_sigma
  arma::cube lambads = arma::zeros(d,d,k);
  arma::cube pr_mu_sigma_inv = arma::zeros(d,d,k); //B0^-1
  for (unsigned j = 0; j < pr_mu_sigma.n_slices; j++) {
    lambads.slice(j) = sigma.slice(j).i();
    pr_mu_sigma_inv.slice(j) = pr_mu_sigma.slice(j).i();
  }
  
  arma::umat z = randomIndices(nvert, k);                   // randomly allocating the vertices
  arma::cube mu_save = arma::zeros(d,k,niter);              // history of simulated values of mu
  arma::field<arma::cube> sigmas_save(niter);               // history of simulated values of sigma
  arma::vec sum_save = arma::zeros(niter);                  // sum of identical neighbours
  arma::umat alloc = arma::zeros<arma::umat>(nR.nrow(), k); // number of allocations of each elem to each cluster 
  arma::rowvec nZ(k);                                       // number of element in each cluster
  arma::mat sumY(d,k);                                      // sum of the values in each cluster, used to compute the means per cluster
  arma::cube SSY = arma::zeros(d,d,k);                      // the matrix of sum (y-mu)(y-mu)^t for each cluster 
  mdupdateStats(Y, z, mu, nZ, sumY, SSY);                   // updating the data structures following the random initialization
  arma::mat salout = arma::zeros(nsalso,nvert);             // history of allocations salso compatible
  arma::rowvec allocvec(nvert);
  
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    // check for interrupt every thousand iterations
    if (it % 1000 == 0) Rcpp::checkUserInterrupt();
    
    // update labels
    arma::mat alpha = dnorm_field(Y, mu, lambads);
    gibbsLabels(neigh, blocks, z, alloc, beta, alpha);
    mdupdateStats(Y, z, mu, nZ, sumY, SSY);
    sum_save(it) = sum_ident(z, neigh, blocks);
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
    // update means
    mu = mdgibbsMeans(nZ, sumY, pr_mu, pr_mu_sigma_inv, lambads);
    mu_save.slice(it) = mu;
    
    // update standard deviations
    lambads = gibbsLambda(nZ, SSY, z, pr_sigma_v0, pr_sigma_n0);
    arma::cube temp_sigma(d,d,k);
    for( unsigned int i = 0; i < k; i++){
      temp_sigma.slice(i) = lambads.slice(i).i();
    }
    sigmas_save(it) = temp_sigma;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,                           // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,                         // history of simulated values of mu
    Rcpp::Named("sigmas") = sigmas_save,                    // history of simulated values of sigma
    Rcpp::Named("z")     = z,                               // final sample from Gibbs distribution
    Rcpp::Named("sum") = sum_save,                          // sum of identical neighbours
    Rcpp::Named("salso_hist") = salout                      // history of allocations salso compatible
  );
  END_RCPP
}


// wrapper function for the gibbs sampler for the potts model with fixed beta
// this identifies wheter we are in the 1d or md case and calls the appropriate function
// [[Rcpp::export]]
SEXP GibbsPotts(SEXP yS, SEXP betaS, SEXP muS, SEXP sigmaS, SEXP nS, SEXP bS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS) {
  if(Rf_isMatrix(yS))
    return gibbsPottsmd(yS, betaS, muS, sigmaS, nS, bS, prS, itS, biS, salitS);
  else if(Rf_isVector(yS))
    return gibbsPotts1d(yS, betaS, muS, sigmaS, nS, bS, prS, itS, biS, salitS);
  else {
    Rcpp::Rcout << " the data must be either a vector or a matrix " << "\n";
    return Rcpp::List(); //return an empty list
  }
}


// monodimensional Potts model gibbs sampler with beta sampled via Pseudolikelihood
// inputs:
// yS a vector of data
// nS is the neighbours object from BayesImageS
// bS is the blocks object from BayesImageS
// itS number of iterations to perform
// biS number of burn in iterations
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (vector)
//    mu.sd the prior sd of the normal distr of the mean of each cluster (vector)
//    sigma the shape parameter of the gamma for the sd prior (vector)
//    sigma.nu the rate parameters of the gamma for the sd prior (vector)
//    beta constaining the borders of the interval where the beta is uniformly distributed (vector)
// mhS is a list of options for the MH step when sampling beta,it includes:
//    bandwith the standard deviation of the normal distr used for the rwmh when updating beta
//    init optional value used to initalize the beta to start from
//    adaptive optional,if omitted the default is adaptive, if adaptive = NULL then the bandwidth is fixed
//    target optional, used to dedtermine how fast to change the bandwidth int the adaptive case (don't change if it is not strictly necessary)
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)
SEXP mcmcPotts1d(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS)
{
  BEGIN_RCPP
  Rcpp::NumericVector yR(yS);       // creates Rcpp vector from SEXP
  Rcpp::IntegerMatrix nR(nS);       // creates Rcpp matrix from SEXP
  Rcpp::List bR(bS), prR(prS), mhR(mhS);
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  
  double bw = Rcpp::as<double>(mhR["bandwidth"]);
  double init_beta = 0.0;
  if (mhR.containsElementNamed("init"))
  {
    init_beta = Rcpp::as<double>(mhR["init"]);
  }
  
  // whether to use the algorithm of Garthwaite, Fan & Sisson (2010)
  bool adaptGFS=true;
  double adaptTarget = 0.44; // see Roberts and Rosenthal (2001)
  if (mhR.containsElementNamed("adaptive"))
  {
    if (Rf_isNull(mhR["adaptive"]) || ISNA(mhR["adaptive"]))
    {
      adaptGFS = false;
    }
    else
    {
      std::string adapt_alg = Rcpp::as<std::string>(mhR["adaptive"]);
      adaptGFS = (adapt_alg.compare(0, adapt_alg.length(), "GFS", 0, adapt_alg.length()) == 0);
    }
    if (adaptGFS && mhR.containsElementNamed("target"))
    {
      adaptTarget = Rcpp::as<double>(mhR["target"]);
    }
  }
  
  if (prR.length() == 0)
  {
    throw std::invalid_argument("prior is empty");
  }
  int nvert = nR.nrow();
  if (nvert != yR.size())
  {
    throw std::invalid_argument("mismatch between observations and neighbourhood matrix");
  }
  
  int k = Rcpp::as<int>(prR["k"]);
  Rcpp::NumericVector prior_mu = prR["mu"];
  Rcpp::NumericVector prior_mu_sd = prR["mu.sd"];
  Rcpp::NumericVector prior_sd = prR["sigma"];
  Rcpp::NumericVector prior_sd_nu = prR["sigma.nu"];
  Rcpp::NumericVector prior_beta = prR["beta"];
  
  Rcpp::NumericVector yunique = Rcpp::unique(yR);
  Rcpp::IntegerVector ymatchR = Rcpp::match(yR, yunique);
  // no easy conversion from IntegerVector to uvec
  arma::uvec ymatch = unsign(ymatchR) - 1;
  arma::umat neigh = unsignMx(nR) - 1;
  
  // block index vectors are not symmetric
  std::vector<arma::uvec> blocks;
  blocks.reserve(bR.length());
  for (int b=0; b<bR.length(); b++)
  {
    Rcpp::IntegerVector block = bR[b];
    arma::uvec ublock = unsign(block - 1);
    blocks.push_back(ublock);
  }
  
  arma::colvec y(yR.begin(), yR.size(), false); // reuses memory and avoids extra copy
  arma::rowvec pr_mu(prior_mu.begin(), prior_mu.size(), false);
  arma::rowvec pr_mu_sd(prior_mu_sd.begin(), prior_mu_sd.size(), false);
  arma::rowvec pr_mu_tau = arma::pow(pr_mu_sd, -2);
  arma::rowvec pr_sd(prior_sd.begin(), prior_sd.size(), false);
  arma::rowvec pr_sd_nu(prior_sd_nu.begin(), prior_sd_nu.size(), false);
  arma::rowvec pr_sd_SS = pr_sd_nu % arma::square(pr_sd); // Schur product
  double pr_beta[2];
  pr_beta[0] = prior_beta[0];
  pr_beta[1] = prior_beta[prior_beta.size()-1];
  
  Rcpp::RNGScope scope;               // initialize random number generator
  arma::rowvec mu = rnorm(pr_mu,pr_mu_sd);
  arma::rowvec sd = arma::pow(rgamma(pr_sd_nu/2, pr_sd_SS/2), -0.5);
  arma::umat z = randomIndices(nvert, k);  
  double beta = init_beta; // use the supplied value only if it is within the prior support
  if (prior_beta[0] > init_beta) {
    beta = prior_beta[0];
  }
  
  arma::mat mu_save = arma::zeros(niter, k); // history of simulated values of mu
  arma::mat sd_save = arma::zeros(niter, k); // history of simulated values of sigma
  arma::vec beta_save = arma::zeros(niter);  // history of simulated values of beta
  arma::vec sum_save = arma::zeros(niter);   // sum of identical neighbours
  
  arma::umat alloc = arma::zeros<arma::umat>(nR.nrow(), k);
  arma::rowvec nZ(k), sumY(k), sqDiff(k);
  arma::mat salout = arma::zeros(nsalso,nvert); // history of allocations salso compatible
  arma::rowvec allocvec = arma::zeros(nvert);
  unsigned accept = 0, acc_old = 0;
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    // check for interrupt every thousand iterations
    if (it % 1000 == 0) Rcpp::checkUserInterrupt();
    
    // update labels
    arma::mat alpha = dnorm(yunique, ymatch, mu, sd);
    gibbsLabels(neigh, blocks, z, alloc, beta, alpha);
    updateStats(y, z, nZ, sumY, sqDiff);
    sum_save(it) = sum_ident(z, neigh, blocks);
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
    // update means
    mu = gibbsMeans(nZ, sumY, pr_mu, pr_mu_tau, sd);
    mu_save.row(it) = mu;
    
    // update standard deviations
    sd = gibbsStdDev(nZ, sumY, sqDiff, pr_sd_nu, pr_sd_SS, mu);
    sd_save.row(it) = sd;
    
    // update inverse temperature
    accept += pseudoBeta(neigh, blocks, z, beta, pr_beta, bw);
    
    // adaptive MCMC algorithm of Garthwaite, Fan & Sisson (2010)
    if (adaptGFS && accept > 10) {
      if (accept > acc_old) {
        bw = bw + bw/adaptTarget/it;
      }
      else {
        bw = bw - bw/(1-adaptTarget)/it;
      }
      acc_old = accept;
    }
    beta_save[it] = beta;
  }
  
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,                           // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,                         // history of simulated values of mu
    Rcpp::Named("sigma") = sd_save,                         // history of simulated values of sigma
    Rcpp::Named("beta")  = beta_save,                       // history of simulated values of beta
    Rcpp::Named("accept") = accept,                         // M-H acceptance
    Rcpp::Named("sum") = sum_save,                          // history of the sum of identical neighbours
    Rcpp::Named("z") = z,                                   // final sample from Gibbs distribution
    Rcpp::Named("salso_hist") = salout                      // history of allocations salso compatible
  );
  
  if (adaptGFS) {
    result["bandwidth"] = bw;                               // final M-H bandwidth produced by GFS algorithm
  }
  
  return result;
  
  END_RCPP
}


// multidimensional Potts model gibbs sampler with beta sampled via Pseudolikelihood
// inputs:
// yS a matrix of data, samples on the cols, features on the rows
// nS is the neighbours object from BayesImageS
// bS is the blocks object from BayesImageS
// itS number of iterations to perform
// biS number of burn in iterations
// prS is a list of priors for the model, it includes:
//    mu: the prior mean of the normal distr of the mean of each cluster (matrix)
//    mu.sigma the covariance matrices of the normal distr of the mean of each cluster (3D array)
//    sigma.V0 the V= matrix for the wishart prior of the covariance matrices (3D array)
//    sigma.n0 the n0 parameters of the wishart prior for the covariance matrices (vector)
//    beta constaining the borders of the interval where the beta is uniformly distributed (vector)
// mhS is a list of options for the MH step when sampling beta,it includes:
//    bandwith the standard deviation of the normal distr used for the rwmh when updating beta
//    init optional value used to initalize the beta to start from
//    adaptive optional,if omitted the default is adaptive, if adaptive = NULL then the bandwidth is fixed
//    target optional, used to dedtermine how fast to change the bandwidth int the adaptive case (don't change if it is not strictly necessary)
//salitS number of allocactions to save in a salso compatible manner (matrix each row is a iteration)
SEXP mcmcPottsmd(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS)
{
  BEGIN_RCPP
  //directly creating arma matrix from SEXP
  arma::mat Y = Rcpp::as<arma::mat>(yS);
  
  Rcpp::IntegerMatrix nR(nS);       // creates Rcpp matrix from SEXP
  Rcpp::List bR(bS), prR(prS), mhR(mhS);
  unsigned niter = Rcpp::as<unsigned>(itS), nburn = Rcpp::as<unsigned>(biS), nsalso = Rcpp::as<unsigned>(salitS);
  
  // no easy conversion from IntegerVector to uvec
  arma::umat neigh = unsignMx(nR) - 1;
  // block index vectors are not symmetric
  std::vector<arma::uvec> blocks;
  blocks.reserve(bR.length());
  for (int b=0; b<bR.length(); b++)
  {
    Rcpp::IntegerVector block = bR[b];
    arma::uvec ublock = unsign(block - 1);
    blocks.push_back(ublock);
  }
  
  double bw = Rcpp::as<double>(mhR["bandwidth"]);
  double init_beta = 0.0;
  if (mhR.containsElementNamed("init"))
  {
    init_beta = Rcpp::as<double>(mhR["init"]);
  }
  
  // whether to use the algorithm of Garthwaite, Fan & Sisson (2010)
  bool adaptGFS=true;
  double adaptTarget = 0.44; // see Roberts and Rosenthal (2001)
  if (mhR.containsElementNamed("adaptive"))
  {
    if (Rf_isNull(mhR["adaptive"]) || ISNA(mhR["adaptive"]))
    {
      adaptGFS = false;
    }
    else
    {
      std::string adapt_alg = Rcpp::as<std::string>(mhR["adaptive"]);
      adaptGFS = (adapt_alg.compare(0, adapt_alg.length(), "GFS", 0, adapt_alg.length()) == 0);
    }
    if (adaptGFS && mhR.containsElementNamed("target"))
    {
      adaptTarget = Rcpp::as<double>(mhR["target"]);
    }
  }
  
  // checks
  int d = Y.n_rows;
  if (prR.length() == 0)
  {
    throw std::invalid_argument("prior is empty");
  }
  int nvert = nR.nrow();
  if (nvert != Y.n_cols)
  {
    throw std::invalid_argument("mismatch between observations and neighbourhood matrix");
  }
  
  int k = Rcpp::as<int>(prR["k"]);
  arma::mat pr_mu = Rcpp::as<arma::mat>(prR["mu"]);
  arma::cube pr_mu_sigma = Rcpp::as<arma::cube>(prR["mu.sigma"]);
  arma::cube pr_sigma_v0 = Rcpp::as<arma::cube>(prR["sigma.V0"]);
  arma::rowvec pr_sigma_n0 = Rcpp::as<arma::rowvec>(prR["sigma.n0"]);
  Rcpp::NumericVector prior_beta = prR["beta"];
  
  double pr_beta[2];
  pr_beta[0] = prior_beta[0];
  pr_beta[1] = prior_beta[prior_beta.size()-1];
  
  Rcpp::RNGScope scope;               // initialize random number generator
  // we initialize the mu and sigma from the corresponding distributions, we could pass it if needed or initialize it with the expected value 
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
  
  double beta = init_beta; // use the supplied value only if it is within the prior support
  if (prior_beta[0] > init_beta) {
    beta = prior_beta[0];
  }
  arma::umat z = randomIndices(nvert, k);                   //randomly allocating the vertices
  arma::cube mu_save = arma::zeros(d,k,niter);              // history of simulated values of mu
  arma::field<arma::cube> sigmas_save(niter);               // history of simulated values of sigma
  arma::vec sum_save = arma::zeros(niter);                  // sum of identical neighbours
  arma::vec beta_save = arma::zeros(niter);                 // history of simulated values of beta
  arma::umat alloc = arma::zeros<arma::umat>(nR.nrow(), k); // number of allocations of each elem to each cluster 
  arma::rowvec nZ(k);                                       // number of element in each cluster
  arma::mat sumY(d,k);                                      // sum of the values in each cluster, used to compute the means per cluster
  arma::cube SSY = arma::zeros(d,d,k);                      // the matrix of sum (y-mu)(y-mu)^t for each cluster 
  mdupdateStats(Y, z, mu, nZ, sumY, SSY);                   // updating the data structures following the random initialization
  arma::mat salout = arma::zeros(nsalso,nvert);             // history of allocations salso compatible
  arma::rowvec allocvec(nvert);
  unsigned accept = 0, acc_old = 0;
  
  Rcpp::Rcout << niter << " iterations discarding the first " << nburn << " as burn-in." << "\n";
  bool display_progress = true ;
  Progress p(niter, display_progress) ;
  
  for (unsigned it=0; it<niter; it++){
    //status bar update
    p.increment();
    
    // reset labels after burn-in
    if (it == nburn) alloc.zeros();
    // check for interrupt every thousand iterations
    if (it % 1000 == 0) Rcpp::checkUserInterrupt();
    
    // update labels
    arma::mat alpha = dnorm_field(Y, mu, lambads);
    gibbsLabels(neigh, blocks, z, alloc, beta, alpha);
    mdupdateStats(Y, z, mu, nZ, sumY, SSY);
    sum_save(it) = sum_ident(z, neigh, blocks);
    if(it >= niter - nsalso){
      salsovec(z,allocvec);
      salout.row(it - niter + nsalso) = allocvec;
    }
    
    // update means
    mu = mdgibbsMeans(nZ, sumY, pr_mu, pr_mu_sigma_inv, lambads);
    mu_save.slice(it) = mu;
    
    /// update standard deviations
    lambads = gibbsLambda(nZ, SSY, z, pr_sigma_v0, pr_sigma_n0);
    arma::cube temp_sigma(d,d,k);
    for( unsigned int i = 0; i < k; i++){
      temp_sigma.slice(i) = lambads.slice(i).i();
    }
    sigmas_save(it) = temp_sigma;
    
    // update inverse temperature
    accept += pseudoBeta(neigh, blocks, z, beta, pr_beta, bw);
    
    // adaptive MCMC algorithm of Garthwaite, Fan & Sisson (2010)
    if (adaptGFS && accept > 10) {
      if (accept > acc_old) {
        bw = bw + bw/adaptTarget/it;
      }
      else {
        bw = bw - bw/(1-adaptTarget)/it;
      }
      acc_old = accept;
    }
    beta_save[it] = beta;
  }
  
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("alloc") = alloc,                           // count of allocations to each component
    Rcpp::Named("mu")    = mu_save,                         // history of simulated values of mu
    Rcpp::Named("sigma") = sigmas_save,                     // history of simulated values of sigma
    Rcpp::Named("beta")  = beta_save,                       // history of simulated values of beta
    Rcpp::Named("accept") = accept,                         // M-H acceptance
    Rcpp::Named("sum") = sum_save,                          // history of the sum of identical neighbours
    Rcpp::Named("z") = z,                                   // final sample from Gibbs distribution
    Rcpp::Named("salso_hist") = salout                      // history of allocations salso compatible
  );
  
  if (adaptGFS) {
    result["bandwidth"] = bw;           
    // final M-H bandwidth produced by GFS algorithm
  }
  
  return result;
  
  END_RCPP
}


// wrapper function for the gibbs sampler for the potts model with beta approximatted by pseudolikelihood 
// this identifies wheter we are in the 1d or md case and calls the appropriate function
// [[Rcpp::export]]
SEXP MCMCPotts(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS) {
  if(Rf_isMatrix(yS))
    return mcmcPottsmd(yS, nS, bS, itS, biS, prS, mhS, salitS);
  else if(Rf_isVector(yS))
    return mcmcPotts1d(yS, nS, bS, itS, biS, prS, mhS, salitS);
  else {
    Rcpp::Rcout << " the data must be either a vector or a matrix " << "\n";
    return Rcpp::List(); //return an empty list
  }
}


