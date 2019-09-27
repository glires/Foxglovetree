#!/usr/bin/env Rscript
data <- read.table("data.tsv", header = TRUE)
individuals <- as.vector(data[, 1])
data <- as.matrix(data[, 2:ncol(data)])
rownames(data) <- individuals
# install.packages("RColorBrewer")
library("RColorBrewer")
pdf("heatmap.pdf")
heatmap(data, main = "", Rowv = TRUE, Colv =TRUE, distfun = dist, hclustfun = hclust, col = brewer.pal(9, "Blues"), cexRow = 0.08, cexCol = 0.08, mex = 1.2)
dev.off()
