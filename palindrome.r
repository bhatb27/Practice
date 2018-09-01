pali<- function(x){
  x<- tolower(x)
  a<- strsplit(x,split = "")
  b<- rev.default(a)
  paste(a,sep = "")==paste(b,sep = "")
  if (paste(a,sep = "")==paste(b,sep = "")){
    print("Palindrome")
  } else {
    print("Not Palindrome")
  }
}

pali("malayalam")