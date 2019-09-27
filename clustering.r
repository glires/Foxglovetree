#!/usr/bin/env Rscript
data <- read.table("data.tsv", header = TRUE)
individuals <- as.vector(data[,1])
data <- data[, 2:ncol(data)]
rownames(data) <- individuals
rho <- cor(t(data), method = "spearman")
pdf("hclust.pdf")
par(ps = 14)
plot(hclust(dist(1 - rho), method = "ward.D2"), lwd = 2)
dev.off()
