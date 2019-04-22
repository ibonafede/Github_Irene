
# Per permettere di inviare i comandi in blocco (dai pulsanti di Tinn-R) se non 
# dovesse farlo automaticamente
.trPaths <- paste(paste(Sys.getenv('APPDATA'), '\\Tinn-R\\tmp\\', sep=''), 
  c('', 'search.txt', 'objects.txt', 'file.r', 'selection.r', 'block.r', 
'lines.r'), sep='') 

################################# PACCHETTI ####################################

# Elenca tutti i pacchetti installati
# Controllare se ci sono i pacchetti "base", "graphics", "lattice", "stats", "utils". 
library()
# Installare uno o pi? pacchetti
install.packages(c("Rcmdr", "rgl", "readxl", "labstatR", "fBasics", "pastecs"))
# Disinstallare un pacchetto
# remove.packages("Rcmdr")
# Aggiornare i pacchetti
# update.packages()
# Caricamento di un pacchetto
library(base)
# Elenco delle informazioni e funzioni del pacchetto
library(help="base")

######################## FUNZIONI DEI PACCHETTI STANDARD #######################

# Assegnazione
a <- 3
x <- c(2,5,4,-8,0)
y <- c(1,6,5,4,1)
w <- c(3,1)
z <- c(x,y);z 

# Sequenze
1:10
seq(1,10)
seq(1,20,by=3)
seq(1,20,length=3)
seq(1,20,3)
letters[1:5]
LETTERS[1:27] # "NA" Not Available
# Lunghezza
length(x)
# Ripetizioni
rep(2,6)
rep(x,3)
rep(c(0,1,2,3),c(1,5,10,50))
# Ordinamento
x.ordinato <- sort(x,decreasing=T)
sort(x,T)
sort(x,F)
sort(x)

### Operatori matematici
# Addizione
x+3
1:10+3
x+y
# Sottrazione
y-a
y-x
# Moltiplicazione
x*a
# Divisione
x/a
x/y
# Modulo: quoziente della divisione
y%/%a
# Resto della divisione
y%%a
y%%x # NaN "Not a Number"
# Elevamento a potenza
a^3

### Operatori relazionali
# Disuguaglianze
a>7
x>a
x<y
z>=2
x<=1:5
# Uguale/diverso
a==10
y==x
a!=10
y!=1:5

# Operatori logici
# NOT
!TRUE
!FALSE
# AND 
T&T
T&F
1&0
# OR
T|T
T|F
1|0

### Funzioni di base
# Somma
sum(x)
sum(c(F,F,T,T,T))
sum(x,y)           
# Somma cumulata
cumsum(x)
# Prodotto
prod(x)
prod(2:5)
# Radice quadrata
sqrt(z)    
# Approssimazione
round(sqrt(2),3)
round(7/3,0)
# Troncamento alla parte intera
trunc(sqrt(2))
trunc(1/3)
# Minimo
min(x)
# Massimo
max(x)
# Valore assoluto
abs(x)
# Esponenziale
exp(3)
# Logaritmo
log(1000,10)
log(8,2)
log(2,exp(1))
log(2)

### Estrazione ed indicatori
# Restituzione degli elementi che soddisfano una condizione
x[2]
x[c(2,3)]
x[-3]
x[-c(1,3)]
x[c(T,F,F,F,T)]
x==5
x[x==5]
x[x>a]
x[y>a]
y[y>3&y<6]
# Restituzione dei primi elementi
library(utils)
head(x,2) 
# Restituzione degli ultimi elementi
tail(y,2)
# Restituzione delle posizioni degli elementi che soddisfano una condizione
which(x==0)
which(y>a)
# Restituzione della posizione del valore minimo
which.min(y) # Individua la posizione del 1? min senza valutare l'altro
# Restituzione della posizione del valore massimo
which.max(y)
# Restituzione delle posizioni degli elementi ordinati crescentemente
order(y)
order(y,decreasing=F)
order(y,decreasing=T)

###################### VETTORI, MATRICI, LISTE E ARRAY #########################

# Creazione di un vettore
num <- c(1,5,8)
cat <- c("Alto","Medio","Basso")
# Nomina degli elementi del vettore
names(num) <- cat
num
# Inizializzazione di un vettore numerico
zeri <- numeric(5)
numeric(0) # Vuoto
                               
# Creazione di una matrice
# Per colonna
matrix(1:12,3,4)
matrix(1:5,3,4)
# Per riga
matrix(1:6,3,2, byrow=T)
# Con dati sotto forma di tabella
matrice <- matrix(c(1,2,3,
                    4,5,6,
                    0,8,0),
                    3,3,byrow=T)
# Unendo per riga vettori di uguale dimensione
rbind(x,y)
# Unendo per colonna vettori di uguale dimensione
cbind(x,y)
# Immettendo i dati attraverso un foglio elettronico
library(utils)
dati <- data.frame()
fix(dati)
dati
# Si assegnando i nomi a righe e colonne 
rownames(matrice) <- letters[1:3]
colnames(matrice) <- c("Basso","Medio","Alto")
matrice
# Dimensione di una matrice
dim(matrice)
# Estrazione degli elementi 
matrice[2,2]
matrice[3,]
matrice[-1,]
matrice[,2]
matrice[,-3]
matrice[,"Alto"]
matrice["b",]
matrice[c(1,3),-2]
# Sostituzione di valori
matrice[1,1] <- 100; matrice
matrice[,3] <- c(11,22,33); matrice
# Trasformazione matrice-vettore
as.vector(matrice)
as.matrix(x)
# Matrice trasposta
t(matrice)
# Calcolo di funzioni in ogni riga ("1") di una matrice
apply(matrice,1,sum)
apply(matrice,1,prod)
# Calcolo di funzioni in ogni colonna ("2") di una matrice
apply(matrice,2,sum)
apply(matrice,2,prod)
        
# Creazione di una lista, insieme di elementi eterogenei
lista <- list(
         x1 = c(2,5,6,8),
         x2 = c(1,6,5,5,5,4),
         mat = matrix(1:9,3,3),
         x3 = LETTERS[1:5])
# Elenco dei nomi degli elementi della lista 
names(lista)
# Applicazione di una funzione a tutti gli elementi della lista
lapply(lista, length)
# Destrutturazione dei risultati dalla lista 
unlist(lapply(lista, length))
# Estrazione degli elementi da una lista
lista$x1
lista[1] 
# Aggiunta di nuovi elementi
lista$Continente <- c("America","Europa","Asia")
# Classe di un elemento
class(lista$x1)
class(lista$mat)
lapply(lista, class)
