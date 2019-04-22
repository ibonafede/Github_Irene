
# Per permettere di inviare i comandi in blocco (dai pulsanti di Tinn-R) se non 
# dovesse farlo automaticamente
.trPaths <- paste(paste(Sys.getenv('APPDATA'), '\\Tinn-R\\tmp\\', sep=''), 
  c('', 'search.txt', 'objects.txt', 'file.r', 'selection.r', 'block.r', 
'lines.r'), sep='') 

################################ WORKSPACE #####################################

# Data
date()

# Assegnazione
w <- c(2,5,4,1,6)
q <- c(5,2,2,1,0)
let <- LETTERS[1:5] 
matrice <- cbind(w,q,let); matrice

# Elenco degli oggetti creati
ls()
# Specifica le strutture degli oggetti creati 
ls.str()
# Cerca la directory di lavoro
getwd()
# Cambia la directory di lavoro
setwd("C:/Users/ILJA' BARSANTI/Desktop")
getwd()
# Salvataggio di tutti gli oggetti creati 
save.image("All.rda")
# Modifica i dati
fix(w)
# Rimozione di alcuni oggetti
w
rm(w)
w
# Rimozione totale
rm(list=ls())
ls()
# Caricamento degli oggetti salvati
load("all.rda")
ls()

# Help
help("library")
?library
# Ricerca di tutte le funzioni associate alla "stringa"
help.search("library")
??library

################################## GRAFICI #####################################



x <- seq(-2*pi, 2*pi, length = 300)
y <- sin(x)
plot(x,y)
plot(x,y,lwd=2)
plot(x,y, type="h")
plot(x,y, type="l", col="blue")
plot(x,sin(x), type="l", lwd=2, main="Titolo", sub="Didascalia", xlab="Asse delle x", 
     ylab="Asse delle y",col="dark red", xlim=c(-2*pi,2*pi), ylim=c(-1.2,1.2))

# Diverse tipologie di punti (pch) e dimensioni (cex)
plot(x,y, xlim=c(0, 11), ylim=c(0,8), type="n")
points(1:10, rep(7, 10), col=1:10, pch=1:10, cex=2)
text(1:10,rep(5, 10), labels=1:10, col=1:8, cex=seq(1,2.8,0.2))
points(1:10, rep(3, 10), col=1:10, pch=11:20, cex=seq(1,2.8,0.2))
text(1:10,rep(1, 10), labels=11:20, col=1:8, cex=seq(1,1.9,0.1))

# Rappresentazioni multiple di grafici
par(mfrow = c(2,2))
plot(x,exp(x))
plot(x,sin(x), pch=5, cex=1, axes=F, lwd=1.5)
plot(x,cos(x), col="orange", xlim=c(-2*pi,2*pi), type="h", lwd=0.5)
plot(x,tan(x), sub="Figura 3.1. Grafico della tangente", col="green", 
     xlim=c(-pi,pi), ylim=c(-10,10), type="l")
# Si ritorna alla modalit? ad una singola finestra grafica: il comando serve
# anche ad aprire una nuova finestra grafica
windows()

# Arricchire i grafici
plot(x,tan(x), sub="Figura 3.1. Grafico della tangente", col="green", 
     xlim=c(-2*pi,2*pi), ylim=c(-10,10), type="s", cex.axis=1, cex.lab=1.5)  
# Si aggiungono linee 
lines(c(3,-3), c(0,0), col="black")
abline(2,0, col="dark grey", lwd=2)
# Si aggiungono punti 
points(x, cos(x), pch="+", xlim=c(-2*pi,2*pi), col="blue")
# Si aggiunge testo 
# I primi 2 valori sono le coordinate dell'angolo in alto a sinistra
text(-1,9.7, "Testo aggiuntivo", col="black")
text(-1,6, expression(z=sqrt(x^2+y^2)), cex=2)
# Si aggiunge una freccia
arrows(-4,-10, 0,0, length = 0.2, angle = 20, code = 2, col = "red", lwd=2) 
# Si aggiunge la legenda
legend(3, 10, c("tan","cos"), col=c("green","blue"), lwd=c(2,1.5),       
       text.col="black", lty=c(2,1), bg='white') 

# Cercare le demo disponibili e lanciarle
demo(package = .packages(all.available = TRUE))
demo(graphics)
         
# Grafici 3D
library(graphics)  
x <- y <- seq(-pi, pi, length = 100)
# Matrice che calcola i valori di z come somma delle 2 funzioni
z <- outer(sin(2*x), cos(2*y), "+")
persp(x,y,z, phi=30, theta=45, d=2.5, col="orange")

# Grafici 3D ruotabili
library(rgl)
plot3d(cos(x),sin(y),exp(z))
demo(abundance) 
demo(lollipop3d)
f <- function(x,y) {sin(x)+cos(2*y)}
z <- f(x,y)
lollipop3d(rnorm(100),y,z,f,col.pt="red")
demo(hist3d)
# Chiudere tutte le precedenti finestre grafiche
hist3d(rnorm(3000),rnorm(3000),alpha=0.9,nclass=7,scale=30)
# Chiudere la precedente finestra grafica
hist3d(rnorm(3000),rnorm(3000),alpha=0.1,nclass=20,scale=200, col="blue")

################################## Cicli ######################################

# Condizioni
if(sum(x)>=0) print("somma positiva") 
ifelse(sum(x)>=0, "somma positiva", "somma negativa")

# Unire numeri e testo/stringa in un unico testo/stringa: "\n" serve per tornare a capo
stringa <- cat("Io ho", 30, "anni\n")
class(stringa)
# Stessa cosa ma letta come un'unica modalit? di una variabile qualitativa
nuova_stringa <- paste("Io ho", 30, "anni")
class(nuova_stringa )

# Creazione dei simboli -> ALT + (numeri del tastierino numerico):
126 = ~
123 = {
125 = }

# Stampa della parola "prova" x+y volte (flow chart 02)
x <- 2
y <- 5
k <- x+y
for(i in 1:k){
    print("prova") 
}

# Somma degli elementi in un vettore (flow chart 03)
v <- c(3,0,5,7,8,9)
k <- length(v)
somma <- 0
for(i in 1:k){
    somma <- somma+v[i] 
}
somma

# Stampa delle righe di una matrice
matrice <- matrix(1:20,4,5)
righe <- dim(matrice)[1]
for(i in 1:righe){
    cat("I seguenti valori corrispondono alla riga", i, "della matrice:\n")
    print(matrice[i,])
}

################################## Funzioni ####################################

# Funzione che stampa "prova" tante volte pari alla somma di due numeri:
# flowchart 2 con la condizione aggiuntiva che la somma sia intera e positiva.
stampa <- function(x,y){
          # x e y sono gli argomenti-input della funzione: i due numeri da sommare
          k <- x+y
          stopifnot(trunc(k)==k&k>=0) 
          for(i in 1:k){
              print("prova")
          }
}
# Applicazione e verifica
stampa(-1,2)
stampa(2/3,7/3)
stampa(2/3,2/3)

# Stessa funzione che invece di interrompersi in caso di errore, esplicita l'errore
stampa_new <- function(x,y){
              k <- x+y
              if(trunc(k)!=k) print("ERRORE: la somma non ? un numero intero")
              stopifnot(trunc(k)==k) 
              if(k<0) print("ERRORE: la somma ? negativa")   
              stopifnot(k>=0) 
              for(i in 1:k){
                  print("prova")
              }
}
# Applicazione e verifica
stampa_new(-1,2)
stampa_new(-1,-2)
stampa_new(2/3,2/3)

# Funzione che somma gli elementi di un vettore:
# flowchart 3 con la condizione aggiuntiva che il vettore sia numerico.
somma <- function(v){
         stopifnot(is.numeric(v))
         k <- length(v)
         somma <- 0
         for(i in 1:k){
             somma <- somma+v[i]
         }
         print(somma)
}
# Applicazione e verifica
somma(c(1,2,3.3))
somma(c(5,7,10,0,"a"))

# Funzione che stampa le righe di una matrice
stampa_righe <- function(m){
                # m ? l'argomento-input della funzione: la matrice
                stopifnot(is.matrix(m))
                k <- dim(m)[1]
                for(i in 1:k){
                    cat("I seguenti valori corrispondono alla riga", i, "della matrice:\n")
                    print(m[i,])
                }
}
# Applicazione e verifica
stampa_righe(matrix(1:20,4,5))
stampa_righe(c(1,2,3)) 
                
# Funzione che restituisce i risultati delle 4 operazioni fra due numeri
quattro_operazioni <- function(x,y){
                      # x e y sono gli argomenti-input della funzione: i due numeri 
                      print(x+y)
                      print(x-y)
                      print(x*y)
                      print(x/y)
                      }
# Applicazione e verifica
quattro_operazioni(100,5)
quattro_operazioni(7,0) 