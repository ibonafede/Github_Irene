
# Convergenza probabilistica al valore p dopo n prove

n <- 1000
p <- .3
successi <- rbinom(n,1,p)
s.cum <- cumsum(successi)
vett.p <- s.cum/1:n
plot(vett.p,ylim=c(0,1))
abline(p,0,col="red")

################################ Distribuzioni #################################

set.seed(50050)
# Distribuzione binomiale
repliche_esperimento <- 1000
numero_prove_indipendenti <- c(10, 100)
probabilità_singola_prova <- c(0.2, 0.5)
par(mfrow=c(2,2))
for(p in probabilità_singola_prova){
    for(n in numero_prove_indipendenti){
        binomiale <- rbinom(repliche_esperimento, n, p)
        hist(binomiale, freq=F, main=paste("Binomiale con n =",n,"e p = ",p))
        dens.norm <- dnorm(seq(0,max(binomiale),length=100),mean(binomiale),sd(binomiale))
        lines(seq(0,max(binomiale),length=100),dens.norm,col="red",lwd=2, lty=2)  
    }
}

# Distribuzione chi-quadro
repliche_esperimento <- 1000
gradi_di_libertà <- c(1,3,10,100)
par(mfrow=c(2,2))
for(gdl in gradi_di_libertà){
    chi_quadro <- rchisq(repliche_esperimento, gdl)
    hist(chi_quadro, freq=F, main=paste("Chi-quadro con gdl =",gdl))
    dens.norm <- dnorm(seq(0,max(chi_quadro),length=100),mean(chi_quadro),sd(chi_quadro))
    lines(seq(0,max(chi_quadro),length=100),dens.norm,col="red",lwd=2, lty=2)
}

# Distribuzione T di Student
repliche_esperimento <- 1000
gradi_di_libertà <- c(3,10,30,100)
par(mfrow=c(2,2))
for(gdl in gradi_di_libertà){
    t_student <- rt(repliche_esperimento, gdl)
    hist(t_student, freq=F, main=paste("T di Student con",gdl,"gdl"))
    dens.norm <- dnorm(seq(-4,4,length=100),mean(t_student),sd(t_student))
    lines(seq(-4,4,length=100),dens.norm,col="red",lwd=2, lty=2)
}

# Distribuzione F di Fisher
repliche_esperimento <- 1000
gradi_di_libertà1 <- c(5,50)
gradi_di_libertà2 <- c(5,50)
par(mfrow=c(2,2))
for(gdl1 in gradi_di_libertà1){
    for(gdl2 in gradi_di_libertà2){
        F_Fisher <- rf(repliche_esperimento, gdl1, gdl2)
        hist(F_Fisher, freq=F, main=paste("F di Fisher con",gdl1,"e",gdl2,"gdl"))
        dens.norm <- dnorm(seq(0,max(F_Fisher),length=100),mean(F_Fisher),sd(F_Fisher))
        lines(seq(0,max(F_Fisher),length=100),dens.norm,col="red",lwd=2, lty=2)
        }
}
