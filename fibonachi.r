fibonacci<- function(x){
  i<-0
  series<- numeric(x)
  for (j in 1:x){
    if (i<2){
      series[j]<- i
      i<- i+1
    } else {
      series[j]<- sum(series[1:(j-1)])
    }
  }
  return(series)
}

fibonacci(7)

