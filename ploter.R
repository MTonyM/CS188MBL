rawjson <- rjson::fromJSON(file =
                             paste("data/","points_",
                                   ".json", sep=""))
rawpjson <- rjson::fromJSON(file =
                              paste("data/",folder,"/pointsp_",
                                    P,"_",PT,".json", sep=""))
    
Data <- as.data.frame(rawjson)
Datap <- as.data.frame(rawpjson)
    
p <- ggplot2::ggplot(Datap,
                     ggplot2::aes(x=Round,
                                  y=Points,
                                  group=categ,
                                  color=categ)
                     ) + ggplot2::geom_line(size=1.2)
    
png(paste("figure/","rewards",".png",sep=""))
    
plot(p)
    
dev.off()