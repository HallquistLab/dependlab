
#' Function to generate ggplot correlation heatmap
#'
#' @param df data.frame to plot
#' @param cormat Alternatively, pass a correlation matrix directory (not \code{df})
#' @param alphaSort Whether to sort variables in alphabetic order
#' @param base_size The base_size argument to the ggplot theme, controlling font sizes
#' @param tileTextSize The size of text inside the cells of the correlation matrix
#' @param title optional title added to ggplot
#'
#' @importFrom reshape2 melt
#' @importFrom ggplot2 ggplot geom_tile theme theme_bw scale_fill_gradient2 geom_text coord_cartesian
#' @importFrom ggplot2 scale_x_discrete scale_y_discrete waiver element_blank xlab ylab
#' @importFrom grid unit
#' @importFrom scales muted
#' @importFrom viridis scale_fill_viridis
#'
#' @examples
#'   library(dplyr)
#'   data(iris)
#' 
#'   cor_heatmap(iris) #Species is not numeric and will be dropped automatically
#' 
#' @export
#' 
cor_heatmap <- function(df, cormat=NULL, alphaSort=TRUE, base_size=14, tileTextSize=3.8, title=NULL, viridis = NULL) {
  #require(reshape2)
  #require(ggplot2)
  #require(scales)
  #require(grid)
  
  if (is.null(cormat)) {
    if (alphaSort) {
      df <- df[,sort(names(df))] #sort variables alphabetically
    }
    
    df <- df[,sapply(df, is.numeric)]
    
    cormat <- cor(df, use="pairwise.complete.obs")
  }
  
  molten.cormat <- melt(cormat)
  names(molten.cormat) <- c("V1", "V2", "corr")
  
  #only retain the lower triangle correlations
  cormat.lower <- molten.cormat[lower.tri(cormat),]
  cormat.upper <- molten.cormat[upper.tri(cormat),]
  
  #create a df with just the diagonal for plotting text
  cormat.ids <- subset(molten.cormat, V1 == V2)
  
  p <- ggplot(cormat.upper, aes(x=V1, y=V2, fill=corr)) + theme_bw(base_size=base_size) + geom_tile() #+ ggtitle(paste("EIFB Intercorrelations for month:", thisMonth))
  p <- p + geom_text(data=cormat.upper, aes(label=round(corr,2)), size=tileTextSize)
  p <- p + geom_text(data=cormat.ids, aes(x=V1, label=V1), colour="grey30", show.legend=FALSE, size=tileTextSize, hjust=0.13, vjust=0.5, angle=0)
  
  if(is.null(viridis)){
    p <- p + scale_fill_gradient2(name="Correlation\n", low=muted("blue"), high=muted("red"))
  } else {
    if(viridis %in% c("magma", "A", "inferno", "B", "plasma", "C")){
      p <- p + scale_fill_viridis() #default (viridis option)
    } else{
      p <- p + scale_fill_viridis(option = viridis)
      }
  }
  
  #p <- p + scale_fill_gradientn(name="Correlation", colours= c("white", "red"), limits=c(0,1))
  
  if(!is.null(title)){
	p <- p + ggtitle(title)
	}  


  #want matrix to run from upper left to lower right, so need to change limits of axes
  if(alphaSort){
    facLevels <- sort(as.character(unique(molten.cormat$V1)))
  } else{
    facLevels <- as.character(unique(molten.cormat$V1)) # retain ordering of input df
  }
  
  
  p <- p + scale_x_discrete(name="", limits=facLevels, labels=waiver()) + scale_y_discrete(name="", limits=rev(facLevels), labels=waiver()) + xlab("") + ylab("") +
      theme(legend.position=c(0.8, 0.8), plot.margin=unit(c(0.2,0.2,0.2,0.2), "lines"), axis.title.x=element_blank(), axis.title.y=element_blank()) +
      coord_cartesian(xlim=c(min(as.numeric(cormat.upper$V1)) - 0.5, max(as.numeric(cormat.upper$V1) + 3))) #pad a bit on the right
  #ylim=c(min(as.numeric(cormat.upper$V2)) - 1.5, max(as.numeric(cormat.upper$V2) + 3)))
  
  p <- p + theme(axis.ticks = element_blank(), axis.text.x=element_blank(), axis.text.y=element_blank())
  return(p)
}
