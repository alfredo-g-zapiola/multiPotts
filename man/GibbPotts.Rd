% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{GibbPotts}
\alias{GibbPotts}
\title{wrapper function for the gibbs sampler for the potts model with fixed beta}
\usage{
GibbPotts(yS, betaS, muS, sigmaS, nS, bS, prS, itS, biS, salitS)
}
\arguments{
\item{yS}{a matrix (vector) of data, samples on the cols, features on the rows}

\item{betaS}{is the value of beta to use for the gibbs Distribution}

\item{muS}{is a matrix containing the starting point of the mu of the chain, one col for cluster}

\item{sigmaS}{is a 3D array containing the starting values of the covariance matrices of the potts model, one per cluster}

\item{nS}{is the neighbours object from BayesImageS}

\item{prS}{is a list of priors for the model, it includes:}

\item{itS}{total number of iterations}

\item{biS}{burn in iterations}

\item{salitS}{number of allocactions to save in a salso compatible manner (matrix each row is a iteration}

\item{bs}{is the blocks object from BayesImageS}

\item{mu:}{the prior mean of the normal distr of the mean of each cluster (matrix)}

\item{mu.sigma}{the covariance matrices of the normal distr of the mean of each cluster (3D array)}

\item{sigma.V0}{the V= matrix for the wishart prior of the covariance matrices (3D array)}

\item{sigma.n0}{the n0 parameters of the wishart prior for the covariance matrices (vector)}
}
\value{
List with the posterior samples of each random variable
}
\description{
this identifies whether we are in the 1d or md case and calls the appropriate function
}
