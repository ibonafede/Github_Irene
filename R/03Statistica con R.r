
###################### SCALE E TRASFORMAZIONI DI SCALE #########################

# Assegnazione delle variabili
x <- c(8, 10, 10, 8, 7); x
y <- c("a", "b", "b", "a", "c"); y

# Per conoscere la natura della variabile
class(x)
class(1:5)
class(y)
class(paste("Io ho", 30, "anni"))
class(cat("Io ho", 30, "anni\n"))
# Verifica se la variabile ?:
# quantitativa
is.numeric(x)  
is.numeric(y)
# quantitativa discreta
is.integer(x)
is.integer(1:5)
# qualitativa generica (indipendentemente dalla scala 
is.character(x)
is.character(y)
# qualitativa nominale
is.factor(y)
# qualitativa ordinale
is.ordered(y)

# Elenco delle modalit? della variabile
unique(x)
unique(y)
# Distribuzione di frequenza
table(x)
table(y)
# Creazione delle classi aperte a sinistra e chiuse a destra (right=T)
z <- rep(0:5,c(10,23,21,14,5,3)); z 
cut(z, breaks=c(0,2,4,6), right=T)
# "NA" poich? lo zero non ? incluso: per includerlo si aggiunge "include.lowest=T"
cut(z, breaks=c(0,2,4,6), right=T, include.lowest=T)
cut(z, breaks=c(0,2,4,6), right=T, include.lowest=T, labels=LETTERS[1:3])

# Trasformare una variabile:
# in nominale 
factor(x)
factor(x, labels=c("Gruppo1","Gruppo2","Gruppo3"))
# in ordinale
ordered(x)
ordered(x, labels=c("bassa","media","alta"))
ordered(ifelse(z<=3,"lieve","moderata"))
# in numerica 
y[y=="a"] <- 1; y
y[y=="b"] <- 2; y
y[y=="c"] <- 3; y
as.numeric(y)
y <- c("a", "b", "b", "a", "c")
ifelse(y=="a",1,ifelse(y=="b",2,3))

# Variabili con dati mancanti "NA" 
missing1 <- c(8, NA, 15, 10, 8, 7); missing1
# Nelle variabili qualitative "NA" fra virgolette significa la parola "NA" e non mancante:

missing2 <- as.factor(c("medio", "NA", NA, "alto", "basso")); missing2
# Indicatore dei dati mancanti
is.na(missing1)
which(is.na(missing1))
which(is.na(missing2))
# Visualizza i dati eccetto quelli mancanti
missing1[!is.na(missing1)]
missing2[!is.na(missing2)]
# Somma 
sum(missing1)

# Per sommare i valori in presenza di missing, questi vanno rimossi con "na.rm=T")
sum(na.rm=T,missing1)

################################# DATAFRAME ####################################

# Creazione di un dataframe.
# N.B. Non mettere "<-" bens? "=" per evitare di non caricare i file dopo averli salvati
dati <- data.frame(v1 = rep(c("M","F"),c(7,3)),
                   v2 = rep(c(1:3),c(3,5,2)),
                   v3 = rep(c(1:5),2),
                   v4 = c(10,11,7,7,7,10,10,10,10,11))  

# Informazioni sul dataframe:
# visualizzazione e modifica
View(dati)
fix(dati)
# struttura
str(dati)
# numero di osservazioni e variabili
dim(dati)
# nomi delle variabili
names(dati)           
# variabili
dati$v2
# Allego i dati al percorso di ricerca di R per fargli trovare le variabili richieste
attach(dati)
#attach permette di non mettere $
v2
# Per aggiungere una variabile
dati$v5 <- 1:10
v5
attach(dati)
v5
# Rimuovo i dati dal percorso di ricerca di R
detach(dati)
v2

# Estrazione dei dati di v3 per i maschi (condizione v1=="M") e poi per le femmine
v3[v1=="M"]
v3[v1=="F"]
# Tabelle di frequenza a doppia entrata
table(v1,v4)
# Tabelle di frequenza ad entrata multipla
ftable(dati,row.vars="v4",col.vars=c("v1","v2"))

# Salvataggio (in file .txt) e caricamento del dataframe
getwd()
write.table(dati,"C:/Users/ILJA' BARSANTI/Desktop/dati.txt")

# Verificare che il file dati.txt sia stato salvato sul desktop

# Rimuovere il dataframe "dati" dal workspace
rm(dati)

# Caricare il dataframe dal file precedemente salvato
dati <- read.table("C:/Users/ILJA' BARSANTI/Desktop/dati.txt", sep="",dec=".")
dati
# Per caricare i dati da un file di Excel
library(readxl)
dati_excel <- read_excel("C:/Users/ILJA' BARSANTI/Desktop/dati.xlsx", 
                         col_names=TRUE, na="", sheet="Foglio1")
# Se il file contiene tutte variabili quantitative si aggiunge col_types = "numeric":
dati_excel <- read_excel("C:/Users/ILJA' BARSANTI/Desktop/dati.xlsx", 
                         col_names=TRUE, na="", sheet="Foglio1", col_types = "numeric")
# Se invece il file contiene sia variabili quantitative che qualitative si aggiunge
# col_types = c("numeric", "text", ... ) a seconda dell'ordine delle variabili. 

############################### INDICI DESCRITTIVI #############################

# Creazione di un campione casuale senza reinserimento e poi con reiserimento
sample(1:10,10)
sample(1:10,10,replace=T)
# Settaggio del seme casuale per riottenere gli stessi valori in un secondo momento
set.seed(50137)
v <- c(sample(30:70,50,replace=T), sample(45:55,40,replace=T), sample(48:52,10,replace=T))
              
#Indici descrittivi di:
# tendenza centrale
mean(v)
median(v)
which.max(table(v))
# forma
library(labstatR)
skew(v)
# Curtosi di Pearson (rispetto a 3)
kurt(v)

# Variabilit?
min(v)
max(v)
max(v)-min(v)
quantile(v,0)
quantile(v,0.25)
quantile(v,0.5)
quantile(v,0.75)
quantile(v,1)
library(stats)
IQR(v)
# Dispersione
sd(v)  # divide per n-1
var(v) # divide per n-1
cv(v) 

############## INDICI DESCRITTIVI E DISTRIBUZIONI DI FREQUENZA  ################

# Descrittive sintetiche
summary(v)
summary(dati)
library(fBasics)
round(basicStats(v),2)
round(basicStats((dati)),2)
library(pastecs)
stat.desc(v)
stat.desc(dati)

# Tabelle di frequenza assoluta
v_new <- rep(c(1,8,0,5),c(10,7,12,15))
table(v_new)
# Tabelle di frequenza relativa
round(table(v_new)/length(v_new),2)
# Tabelle di frequenza cumulata relativa (funzione di ripatizione)
round(cumsum(table(v_new)/length(v_new)),2)

#################################### GRAFICI ###################################

# Grafici a torta
pie(table(v1))
pie(table(v1), main = "Grafico a torta", col=c("grey","yellow"), 
    clockwise=T, radius=1)
#clockwise mette in senso orario
## Diagrammi a barre
par(mfrow=c(1,2))
barplot(table(v_new)) 
# Per tener conto delle distanze fra i valori:
plot(table(v_new), type = "h", col = "red", lwd = 5)

# Boxplot
par(mfrow=c(1,1))
boxplot(v)
# Boxplot condizionati
par(mfrow=c(1,2))
boxplot(v4[v1=="M"])
boxplot(v4[v1=="F"])
#coda a destra(in alto)
par(mfrow=c(1,1))
boxplot(dati)

# Istogramma
hist(v)
hist(v,freq=T)
#default mette la frequenza
hist(v,freq=F)
#frequenza f frequenza rapportata alla dimensione della classe: densità,
#quando si guarda la moda si guarda la densità, la densità di ferquenza maggiore
# Istogramma
#breaks da 30 a 70, fa intervalli da 1, se volessi da 2seq(30:70,by=2)
hist(v, breaks=30:70)
#se c'è breaks mette freq=f di default
hist(v, breaks=c(min(v),48,49,50,51,55,60,max(v)))
# Si aggiunge la funzione di densit? empirica
v.dens <- density(v)
hist(v, breaks=30:70)
lines(v.dens,col="navy",lwd=2,lty=2)
# La curva di densit? per essere confrontata va aggiunta ad un istogramma con "freq=F" 
hist(v, breaks=30:70,freq=F)
lines(v.dens,col="navy",lwd=2,lty=2)
# Si aggiunge la densit? della Normale
x <- seq(min(v),max(v),length=100)
y <- dnorm(x,mean(v),sd(v))
lines(x,y,col="red",lwd=2,lty=2)

# Istogrammi condizionati al genere v1
par(mfrow=c(1,2))
hist(v4[v1=="M"], xlim=c(7,11), col="grey")
hist(v4[v1=="F"], xlim=c(7,11), col="grey")

# Grafici bivariati
plot(v2,v4)
plot(v1,v2)
# Condizionati
coplot(v2~v3|v1)
# Tutti assieme
pairs(dati)
