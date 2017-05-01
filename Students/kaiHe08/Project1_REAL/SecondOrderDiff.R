
library(imager)
library(ggplot2)

genMat<-function(x,y,wth,len,im)
{
  matr<-matrix(NA,nrow = wth+2,ncol = len+2)
  w<-wth+x-1
  l<-len+y-1
  matr[c(x:w),c(y:l)]<-im
  matr[-c(x:w),]<-0
  matr[,-c(y:l)]<-0
  return (mrt=matr)
}

difMat<-function(im,m1,m2,wth,len){
  im.dv<-2*im-m1-m2
  im.dv <-as.matrix(im.dv[c(2:(wth+1)),c(2:(len+1))])
  #im.dv <- array(im.dv,dim=nrow(im.dv)*ncol(im.dv))
  return (im.dv=im.dv)
}

SOD <- function(im)
{
  wth<-nrow(im)
  len<-ncol(im)
  x<-2;y<-2
  imo <- genMat(x,y,wth,len,im)
  
  x<-1
  im.h1<-genMat(x,y,wth,len,im)
  x<-3
  im.h2<-genMat(x,y,wth,len,im)
  
  im.dh <- difMat(imo,im.h1,im.h2,wth,len)
  
  x<-2;y<-1
  im.v1<-genMat(x,y,wth,len,im)
  y<-3
  im.v2<-genMat(x,y,wth,len,im)
  im.dv <- difMat(imo,im.v1,im.v2,wth,len)
  
  x<-3;y<-1
  im.d1<-genMat(x,y,wth,len,im)
  x<-1;y<-3
  im.d2<-genMat(x,y,wth,len,im)
  im.dd <- difMat(imo,im.d1,im.d2,wth,len)
  
  x<-1;y<-1
  im.m1<-genMat(x,y,wth,len,im)
  x<-3;y<-3
  im.m2<-genMat(x,y,wth,len,im)
  im.dm <- difMat(imo,im.m1,im.m2,wth,len)
  
  return (list(dh=im.dh,dv=im.dv,dd=im.dd,dm=im.dm))
}

CH3<-function(im){
  im.hsv<-RGBtoHSV(im)
  
  sod1<-SOD(im.hsv[,,1,1])
  sod2<-SOD(im.hsv[,,1,2])
  sod3<-SOD(im.hsv[,,1,3])
  
  return(list(sod1=sod1,sod2=sod2,sod3=sod3))
}

parInf<-function(im,size)
{
  im<-resize(im,size[1],size[2],1,3)
  im.ch3<-CH3(im)
  var<-sqrt(var(array(im.ch3$sod1$dh))/(size[1]*size[2]))
 # plot(density(im.ch3$sod1$dh),xlim=c(-30,30))
 # plot(density(im.ch3$sod1$dh),xlim=c(-30,30))
  
  return(list(var=var,mat=im.ch3$sod1$dv))
}

#GGD (Generalized Gaussian Distribution)

MAP <- function(im,var1=3,var2=0.2,beta1=1,beta2=2){
  n <- dim(im)[1]*dim(im)[2]
  
  alpha1<-var1*sqrt(gamma(1/beta1)/gamma(3/beta1))
  alpha2<-var2*sqrt(gamma(1/beta2)/gamma(3/beta2))
  
  H1 <- log(beta1)-log(2*alpha1*gamma(1/beta1))-sum((abs(im)/alpha1)^beta1)/n
  H2 <- log(beta2)-log(2*alpha2*gamma(1/beta2))-sum((abs(im)/alpha2)^beta2)/n
  
  LLT <-(H1-H2)
  #if (LLT>0){judg<-'Real'}else{judg<-'Synthetic'}
  return(LLT)
}


rslt1<-function(im,size){
  im1<-parInf(im,size)
  plot(density(im1$mat),xlim=c(-30,30),ylim=c(0,5))
  L1<-MAP(im1$mat)
  
  return(L1=L1)
  }
rslt<-function(im,size,col){
  im1<-parInf(im,size)
  lines(density(im1$mat),xlim=c('-30,30'),ylim=c(0,0.1),col=col)
  L1<-MAP(im1$mat)
  
  return(L1=L1)
}


size<-c(500,500)

r1<- load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_0565.jpg')
r2<- load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_2035.jpg')
r3<-load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_0557.jpg')
r4<-load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_0385.jpg')
r5<-load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_0370.jpg')
r6<-load.image('/Users/kaihe/Documents/Spring2017/Project1/IMG_4689.jpg')
r7<-load.image('/Users/kaihe/Documents/Spring2017/Project1/7.jpg')
#a1<-load.image('/Users/kaihe/Documents/Spring2017/Project1/tree8.jpg')
a2<- load.image('/Users/kaihe/Documents/Spring2017/Project1/tree2.jpg')
#a3 <- load.image('/Users/kaihe/Documents/Spring2017/Project1/tree.jpg')
a4 <- load.image('/Users/kaihe/Documents/Spring2017/Project1/tree3.jpg')
#a5 <- load.image('/Users/kaihe/Documents/Spring2017/Project1/Lone_House.jpg')
a6 <- load.image('/Users/kaihe/Documents/Spring2017/Project1/tree4.jpg')
a7 <- load.image('/Users/kaihe/Documents/Spring2017/Project1/ani2.jpg')

L1<-rslt1(r1,size)
L2<-rslt(r2,size,'yellow')
L3<-rslt1(r3,size)
L4<-rslt(r4,size,'blue')
L5<-rslt(r5,size,'purple')
L6<-rslt(r6,size,'blue')
L7<-rslt1(a2,size)
L8<-rslt(a4,size,'blue')
L9<-rslt(a6,size,'purple')
L10<-rslt(a7,size,'blue')



#hist(im1.ch3$)

