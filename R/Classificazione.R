 
# OBIETTIVO: prevedere la scelta del consumatore tra le 2 marche in funzione dei prezzi delle 2 marche,
# degli sconti applicati, della fedelt? alla marca, . ??? problema di classificazione (a 2 classi)
 
succo <- read.table("Classificazione_succo/succo.txt",header=T)
n <- nrow(succo)
n		# 1070
succo$negozio = as.factor(succo$negozio)

id.train <- as.matrix(read.table("id_train.txt"))
id.test <- as.matrix(read.table("id_test.txt"))
n.train <- length(id.train)
n.test<- length(id.test)



# MODELLO DI REGRESSIONE LOGISTICA
m1 <- glm( scelta ~ settimana + prezzoCH + prezzoMM + scontoCH + scontoMM+fedeleMM+negozio, data=succo[id.train,], family = binomial )

rlog <- update( m1, .~. -settimana , data = succo[id.train,])

# stima della performance classificatoria attesa per il modello rlog fuori campione
p.rlog.ts <- predict.glm(rlog, newdata = succo[id.test,], type="response")

# regola di classificazione di Bayes (o della massima probabilit? a posteriori):
Y.previsto.rlog.ts <- p.rlog.ts > 0.5
Y.previsto.rlog.ts <- as.integer(Y.previsto.rlog.ts)

mconf.rlog.ts <-  table(Y.previsto.rlog.ts, succo[id.test,1])
print(mconf.rlog.ts)

# tasso di errata classificazione
tasso.errore.rlog.ts <- 1 - sum( diag( mconf.rlog.ts ))/sum( mconf.rlog.ts )
1-tasso.errore.rlog.ts

# specificit? = capacit? di classificare correttamente i negativi, cio? i casi in cui Y=0)
fr.vn.rlog.ts <- mconf.rlog.ts[1,1]/(mconf.rlog.ts[1,1]+mconf.rlog.ts[2,1])   

# sensibilit? = capacit? di classificare correttamente i positivi, cio? i casi in cui Y=1)
fr.vp.rlog.ts <- mconf.rlog.ts[2,2]/(mconf.rlog.ts[1,2]+mconf.rlog.ts[2,2])

# curva ROC
install.packages("ROCR")
library(ROCR)
pred.obj <- prediction( p.rlog.ts,succo[id.test,1] )
sens_spec.rlog <- performance(pred.obj,"sens","spec")
plot(sens_spec.rlog,colorize=TRUE)



# ALBERO DI CLASSIFICAZIONE
# Nella operazione di pruning selettivo decidiamo di usare un "validation set". Dovremo destinare
# al ruolo di validation set una parte del training set.

set.seed(123)
permuta <- sample(as.vector(id.train),length(id.train))
id.train <- sort(permuta[1:600])
id.validation <- sort(permuta[601 :length(permuta)])  

install.packages("tree")
library(tree)

Tmax <- tree(scelta~settimana+prezzoCH+prezzoMM+scontoCH+scontoMM+fedeleMM+negozio, data=succo[id.train,],
             control=tree.control(nobs=length(id.train),minsize=2,mindev=0.0001))

plot(Tmax)
text(Tmax)

# il pruning selettivo di Tmax nel package tree si fa:
# con la funzione prune.tree se si dispone di un validation set
# con la funzione cv.tree se si vuole ricorrere alla cross-validation

?prune.tree
# misclass: numero totale di classificazioni errate compiute dal sottoalbero,
# deviance: entropia totale del sottoalbero; 
# Il default ? deviance

# risultati del  pruning selettivo e corrispondente rappresentazione grafica
ris <- prune.tree(Tmax, newdata=succo[id.validation,])
plot(ris) 	 

# prune.tree genera la sequenza finita di sottoalberi annidati di Tmax ottenuta minimizzando la
# funzione costo-complessit? al variare del parametro di penalizzazione

ris 

#  size : numero J di foglie per ciascuno dei sottoalberi della sequenza;
#  deviance: valore dell'entropia totale, calcolata sul validation set, per ciascuno dei sottoalberi della sequenza; 
#  k corrisponde al parametro di penalizzazione 


# cerchiamo, nella sequenza di sottoalberi di Tmax, quello migliore e isoliamolo, ricorrendo ancora
# alla function prune.tree

Jott <- ris$size[ris$dev==min(ris$dev)]
albero <- prune.tree(Tmax, best=Jott)
plot(albero)
text(albero)  # figura 5.15 (dx) di Azzalini e Scarpa

# Interpretazione del modello ad albero selezionato:
albero


# Ogni riga di questo output descrive un nodo. Con un asterisco sono evidenziate le foglie.
# Per ogni nodo, sono elencati 6 argomenti, ognuno dei quali riporta un'informazione relativa al 
# nodo (in analogia con l'output di rpart):

#  node) etichetta numericamente i nodi da sx (numeri dispari) a dx (numeri pari)
# Ogni riga ? indentata secondo il livello dell'albero in cui figura il nodo

# split ? la condizione logica che caratterizza le unit? comprese in quel nodo. 

# l'argomento n ? il numero di unit? contenute nel nodo

# l'argomento (yprob): rispettivamente ( ) e ( ), ovvero le frequenze relative delle due
# classi nel nodo
table(succo[id.train,]$scelta)/nrow(succo[id.train,])

# ricostruiamo il valore deviance per il nodo radice:
2*(-((371/600)*log(371/600)+(229/600)*log(229/600)))*600

# l'argomento yval: valore previsto di Y secondo la regola di classificazione di Bayes (o della
# massima probabilit? a posteriori)


# Nel grafico ad albero, le foglie sono "ETICHETTATE" con il corrispondente valore di yval.
T <- prune.tree(Tmax,best=10)
plot(T)
text(T)

ris.m <- prune.tree(Tmax,newdata=succo[id.validation,],method="misclass")
Jott.m <- min(ris.m$size[ris.m$dev==min(ris.m$dev)])
albero.m <- prune.tree(Tmax, best=Jott.m, method="misclass")
plot(albero.m)
text(albero.m)  





# stima della performance classificatoria per il modello albero fuori campione
Y.previsto.albero.ts.prob <- predict(albero,newdata=succo[id.test,], type="vector")
Y.previsto.albero.ts.class <- predict(albero,newdata=succo[id.test,], type="class")

# unit? sono classificate secondo la regola di Bayes
head(data.frame(Y.previsto.albero.ts.prob,Y.previsto.albero.ts.class))

mconf.albero.ts <-  table(Y.previsto.albero.ts.class, succo[id.test,1])
print(mconf.albero.ts)		# tabella 5.9 Azzalini e Scarpa

# tasso di errata classificazione
tasso.errore.albero.ts <- 1-sum(diag(mconf.albero.ts))/sum(mconf.albero.ts)

# specificit?
fr.vn.albero.ts<- mconf.albero.ts[1,1]/(mconf.albero.ts[1,1]+mconf.albero.ts[2,1])   

# sensibilit?
fr.vp.albero.ts<- mconf.albero.ts[2,2]/(mconf.albero.ts[1,2]+mconf.albero.ts[2,2])

# curve ROC
pred.obj <- prediction(Y.previsto.albero.ts.prob[,2], succo[id.test,1])
sens_spec.albero <- performance(pred.obj,"sens","spec")
plot(sens_spec.rlog,col="green")
par(new=TRUE)
plot(sens_spec.albero,col="red")


# E' preferibile il modello di regressione logistica.

