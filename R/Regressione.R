# OBIETTIVO: prevedere il traffico nel 10° mese usando le informazioni sul 
# cliente e sull'uso del servizio telefonico da parte del cliente nei 9 mesi precedenti


training <- read.table("Regressione_compagnia_telefonica/training_reg.dat", header=TRUE)
test <- read.table("Regressione_compagnia_telefonica/test_reg.dat", header=TRUE)

head(training)
dim(test)

#costruiamo la variabile target
y_tr <- training$q10.out.dur.peak + training$q10.out.dur.offpeak
y_ts <- test$q10.out.dur.peak + test$q10.out.dur.offpeak

# Un'alta percentuale di clienti ha Y=0:
sum( y_tr == 0)  
sum( y_ts== 0 ) 


tr <- training[ y_tr > 0, ]
ts <- test[ y_ts > 0 , ]
dim( tr )


data.tr <- cbind( tr[ , 3:100 ], y = y_tr[ y_tr > 0 ] )
data.ts <- cbind( ts[ , 3:100], y = y_ts[ y_ts > 0 ] )

str( data.tr )


data.tr$piano.tariff <- as.factor(data.tr$piano.tariff)
data.tr$zona.attivaz <- as.factor(data.tr$zona.attivaz)
data.tr$canale.attivaz <- as.factor(data.tr$canale.attivaz)

data.ts$piano.tariff <- as.factor(data.ts$piano.tariff)
data.ts$zona.attivaz <- as.factor(data.ts$zona.attivaz)
data.ts$canale.attivaz <- as.factor(data.ts$canale.attivaz)


# MODELLO DI REGRESSIONE LINEARE MULTIPLA (stepwise dei regressori)

regrlin <- lm( y ~ ., data = data.tr )
# rl.step <- step(regrlin, direction= "both") # impiega alcuni minuti
load("rlstep.RData")
summary( rl.step )		# R2 0.6; nr regressori = 56

sum(rl.step$fitted.values<0)

# stima dell'errore di previsione 
# decidiamo di sostituire le eventuali previsioni < 0 con il valore 0.5
Y.previsto.rl.step.ts <- predict( rl.step, data.ts )
sum( Y.previsto.rl.step.ts < 0 )
Y.previsto.rl.step.ts[ Y.previsto.rl.step.ts < 0 ] <- 0.5

# misure di performance basate sull'errore quadratico
MSE.rl.step.ts <- mean(( Y.previsto.rl.step.ts-data.ts$y )^2 )  
RMSE.rl.step.ts <- sqrt( MSE.rl.step.ts )		
SSE.rl.step.ts <- sum(( Y.previsto.rl.step.ts-data.ts$y )^2 )
RelativeSquaredError.rl.step.ts <- SSE.rl.step.ts/( var( data.ts$y )*( nrow( ts )-1 ))
R2.rl.step.ts<- 1 - RelativeSquaredError.rl.step.ts	 


#ALBERO DI REGRESSIONE

library(rpart)
library(rpart.plot)
set.seed(463)


?rpart		      
Tmax <- rpart( y ~ . , data = data.tr, control = rpart.control( minsplit = 2, cp = 0.0001 )) 
plot( Tmax ) 	   
text( Tmax )   		 


ris <- printcp( Tmax )

which.min(ris[,4]) 

albero <- prune( Tmax, cp=0.00429 ) # 23 foglie
albero <- prune( Tmax, cp = 0.018 ) # 7 foglie (J=nsplit+1)
par( mfrow = c( 1 , 1 ) ) 
plot( albero )
rpart.plot( albero )
text( albero )

albero



by(data.tr$y, data.tr$q09.out.dur.offpeak<3188.5, mean)
boxplot(data.tr$y~data.tr$q09.out.dur.offpeak<3188.5)


Y.previsto.albero.ts <- predict(albero, data.ts)

# errore quadratico
MSE.albero.ts <- mean((Y.previsto.albero.ts-data.ts$y)^2)
RMSE.albero.ts <- sqrt(MSE.albero.ts)		# 4828.32
SSE.albero.ts <- sum((Y.previsto.albero.ts-data.ts$y)^2)
RelativeSquaredError.albero.ts <- SSE.albero.ts/(var(data.ts$y)*(nrow(data.ts)-1))

# Il complemento a 1 di RSE si interpreta in modo equivalente a R2:
R2.albero.ts<- 1 - RelativeSquaredError.albero.ts   # 0.61

# errore assoluto
MAE.albero.ts <- mean(abs(Y.previsto.albero.ts-data.ts$y))  # 1769.565

