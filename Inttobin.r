binary<- function(x){
  i<- 0
  string<- numeric(32)
  while(x>0){
    string[32-i]<- x%%2
    x<- x%/%2
    i=i+1
  }
  first<- match(1,string)
  a<- as.character(string[first:32])
  n<- length(a)
  b<- ""
  for (i in 1:n){
    b<- paste(b,a[i],sep = "")
  }
  return(as.numeric(b))
}

binary(11)
