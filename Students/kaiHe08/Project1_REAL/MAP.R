
#GGD (Generalized Gaussian Distribution)

MAP <- function(im,var1,var2,beta1,beta2,p1,p2){
  n <- dim(im)[1]*dim(im)[2]
  alpha1 <- var1*sqrt(gamma(beta1)/gamma(3/beta1))
  alpha2 <- var2*sqrt(gamma(beta2)/gamma(3/beta2))
  
  H1 <- n*log(beta1)+n*p1-n*log(2*alpha1*gamma(1/beta1))-sum((abs(im)/alpha1)^beta1)
  H2 <- n*log(beta2)+n*p2-n*log(2*alpha2*gamma(1/beta2))-sum((abs(im)/alpha2)^beta2)
  
  LLT <-H1-H2
  
  return(LLT)
}