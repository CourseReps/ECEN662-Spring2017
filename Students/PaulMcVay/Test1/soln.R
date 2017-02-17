library(data.table)

data = as.data.table(read.csv('Data1Set3_PaulMcVay.csv'))

##data[, Zhat := as.integer(Y > 4.663831939)]
##data[, Zhat := as.integer(y > 0.25)]

data[, f1 := 1/(2*3.14159*(1/4)^(-.5))*exp(-.5*(4*Y0^2-2*3^(-0.5)*2*Y0*Y1+4*Y1^2))]
data[, f0 := 1/(2*3.14159)*exp(-0.5*(Y0^2 + Y1^2))]
data[, Zhat := as.integer(f1 > f0)]
write.csv(data, 'Data1Answer3.csv')
