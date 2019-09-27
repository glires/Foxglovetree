#!/usr/bin/env Rscript
data <- read.table("data.tsv", header = TRUE)
individuals <- as.vector(data[, 1])
data <- data[, 2:ncol(data)]
print(dim(data))
data <- as.matrix(data[, apply(data, 2, var, na.rm = TRUE) != 0])
	# avoiding "cannot rescale a constant/zero column to unit variance"
print(dim(data))
rownames(data) <- individuals
pca <- prcomp(data, scale = TRUE)
print(summary(pca))
print(pca$x[, 1:2])
PC1 <- c()
PC2 <- c()
group <- c()
for (i in grep(rownames(pca$x), pattern = "JPT")) { PC1 <- c(PC1, pca$x[i, 1]); PC2 <- c(PC2, pca$x[i, 2]); group <- c(group, "JPT") }
for (i in grep(rownames(pca$x), pattern = "CHB")) { PC1 <- c(PC1, pca$x[i, 1]); PC2 <- c(PC2, pca$x[i, 2]); group <- c(group, "CHB") }
for (i in grep(rownames(pca$x), pattern = "CEU")) { PC1 <- c(PC1, pca$x[i, 1]); PC2 <- c(PC2, pca$x[i, 2]); group <- c(group, "CEU") }
for (i in grep(rownames(pca$x), pattern = "YRI")) { PC1 <- c(PC1, pca$x[i, 1]); PC2 <- c(PC2, pca$x[i, 2]); group <- c(group, "YRI") }
data <- data.frame(PC1, PC2, group = group)
pdf("pca.pdf")
par(mex = 1.4)
plot(data[, 1], data[, 2], xlim = c(-34, 49), ylim = c(-27, 38), xlab = "PC1", ylab = "PC2", pch = 20, cex = 0.4, col = c("darkgreen", "blue", "red", "black")[unclass(data$group)])
axis(1, lwd=2, lwd.ticks=2)
axis(2, lwd=2, lwd.ticks=2)
box(lwd = 2)
dev.off()
