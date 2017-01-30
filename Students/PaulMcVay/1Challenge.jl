using Distributions;
using Plots;
tppyplot()
using DataFrames;

d1 = Cauchy(-1)
d2 = Cauchy(1)

τ = -2:0.01:2
x = -20:0.01:20
τx = -20
τsave = []
fp = []
tp = []

for j in τ
    for i in x
        if log(pdf(d2, i)/pdf(d1, i)) >= j && τx == -20
            τx = i
        end
    end
    falsepositive = 1 - cdf(d1, τx)
    truepositive = 1 - cdf(d2, τx)
    push!(fp, falsepositive)
    push!(tp, truepositive)
    push!(τsave, τ)
    τx = -20
end

plot(fp, tp,linewidth=2,title="My Plot")
