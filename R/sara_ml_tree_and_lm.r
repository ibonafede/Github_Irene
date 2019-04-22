library(TH.data)
data(bodyfat)
myformula<-DEXfat~age+waistcirc+hipcirc+elbowbreadth+kneebreadth
bodyfat.glm<-glm(myformula,family=gaussian("log"),data=bodyfat)
summary(bodyfat.glm)
myformula<-DEXfat~age+waistcirc+hipcirc+kneebreadth
bodyfat.glm<-glm(myformula,family=gaussian("log"),data=bodyfat)
summary(bodyfat.glm)
rm(list=ls())
str(iris)
set.seed(1234)
#divido in train and test, creo 2 gruppi 1 e 2 e prendo le righe =1 e =2
ind<-sample(2,nrow(iris),replace = T,prob=c(0.7,0.3))
trainData<-iris[ind==1,]
testData<-iris[ind==2,]
ind
library(party)
#creo model
myFormula<-Species ~ Sepal.Length+Sepal.Width+ Petal.Length + Petal.Width
attach(bodyfat)
##########################
#random forest
##########################
iris_ctree<-ctree(myFormula,data=trainData)
save(iris_ctree, file = "my_model1.rda")
###################
#later...
#load("my_model1.rda")
## predict for the new `x`s in `newdf`
#predict(m1, newdata = newdf)
####################
table(predict(iris_ctree),trainData$Species)
plot(iris_ctree)

#per bodyfat
set.seed(1234)
ind<-sample(2,nrow(bodyfat),replace = T,prob=c(0.7,0.3))
trainData<-bodyfat[ind==1,]
testData<-bodyfat[ind==2,]
myFormula<-DEXfat~age+waistcirc+hipcirc+elbowbreadth+kneebreadth
library(rpart)
bodyfat_rpart<-rpart(myFormula,data=trainData,control=rpart.control(minsplit=10))
environment()
plot(bodyfat_rpart)
print(bodyfat_rpart$cptable)
opt(which.min(bodyfat_rpart$cptable[,"xerror"])) #seleziona
cp<-bodyfat_rpart$cptable(opt,"CP")
bodyfat_prune<-prune(bodyfat_rpart,cp=cp)
plot(body_fat_prune,use.n=T)
DEXfat_pred<-predict(bodyfat_prune,newdata=testData)
xlim<-range(bodyfat$DEXfat)
DEXfat_pred<-predict(bodyfat_prune,newdata=testData)
#########################
#grafici
#########################
#histogram
attach(mtcars)
h=hist(mpg,breaks=12,col="red",xlab='media_per_gallon',main='histogram of mpg')
head(mtcars)
xfit<-seq(min(mpg),max(mpg),length=40)
yfit<-dnorm(xfit,mean=mean(mpg),sd=sd(mpg))
yfit<-yfit*diff(h$mids[1:2]*length(mpg))
h$mids
lines(xfit,yfit,col="blue",lwd=2)
#density plot
d<-density(mpg)
plot(d,main="density_plot")
polygon(d,col="red",border="blue")
#dotchart
dotchart(mpg,labels=row.names(mtcars),cex=.7)
#barplot
mar<-table(gear)
barplot(mar)

#piechart
slices<-c(12,4,16,8)
lbls<-c('USA','UK','germania','Francia')

pie(slices,labels=lbls,col=c('black','red','yellow','orange'))
pie(slices,labels=lbls,col=rainbow(length(lbls)))
pct<-round(slices/sum(slices)*100)
lbls<-paste(lbls,pct,'%')
#piechart 3d
library(plotrix)
pie3D(slices,labels=lbls,explode=0.1,shade=0.1,theta=0.9,cex=0.5)
#boxplot
boxplot(mpg~cyl,data=mtcars)
#violet plot
library(vioplot)
x1<-mpg[cyl==4]
x2<-mpg[cyl==6]
x3<-mpg[cyl==8]
vioplot(x1,x2,x3,col="gold")
#plot
plot(wt,mpg,pch=19)
#draw more plots
par(mfrow=c(3,1))
hist(wt)
plot(wt,mpg)
par(fig=c(0,0.8,0.55,1))
hist(mpg)
hist(disp)
###################
#fourfoldplot
###################
library(datasets)
head(UCBAdmissions)
x<-aperm(UCBAdmissions,c(2,1,3))
dimnames(x)[[2]]<-c("si","no")
x
UCBAdmissions
names(dimnames(x))<-c("sesso","ammesso","dipartimento")
fourfoldplot(margin.table(x,c(1,2)))
class(x)
#################
#mosaic plot
#################
mosaicplot(x,shade=T)
#or
mosaicplot(Titanic)
x<-margin.table(HairEyeColor,c(1,2))
head(x)
assocplot(x)
library(vcd)
doubledecker(x)
#################
#assocplot
#vede la freq attesa del valore, sopra e nera, la freq osservata è> della prevista
#################
x<-margin.table(HairEyeColor,c(1,2))
head(x)
assocplot(x)
#####################
#cotabplot
###################
cotabplot(Titanic)
################
#agreementplot
################
agreementplot(SexualFun)
par(mfrow=c(1,1))
##################
#scatterplot3d
##################
library(scatterplot3d)
attach(iris)
scatterplot3d(Petal.Width,Sepal.Length,Sepal.Width)
head(iris)
library(rgl)
plot3d(iris)
##################
#levelplot
##################
library(lattice)
levelplot(Petal.Width~Sepal.Length*Sepal.Width,iris,cuts=9,col.regions=rainbow(10)[10:1])
help(levelplot)
filled.contour(volcano,color=terrain.colors,plot.axes=contour(volcano,add=T))
volcano
persp(volcano,theta=25,phi=30,expan=0.5,col="blue")
#persp3d(volcano,theta=25,phi=30,expan=0.5,col="grey")
################
#parcoord
################
library(MASS)
parcoord(iris[1:4],col=iris$Species)
iris[1:4]
head(iris)
################
#paralelplot
################
parallelplot(~iris[1:4]|Species,data=iris)
##############à
#qplot
##############
library(ggplot2)
qplot(Sepal.Length,Sepal.Width,data=iris,facets=Species~.)
