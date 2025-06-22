library(BalancedSampling)

# Read data
grd <- read.csv("sampleddata/USKSAT_OpenRefined_cleaned.csv")

# Remove the first column and the Ksat column
index <- grd[, 1]
ksat <- grd[, 2]
grd_data <- grd[, -c(1, 2)]

# Standardize features
grd_scaled <- as.data.frame(scale(grd_data))

# Define sample sizes
sample_levels <- c(1000, 5000, 10000)
num_samples_per_size <- 20

# Loop to generate sample sets for different sizes and repetitions
for (level in sample_levels) {
  for (t in 1:num_samples_per_size) {
    # Define sample file path
    sample_file <- sprintf("sampleddata/combined_samples_Ksat/BalancedSampling_sampled_data_%d_set_%d.csv", level, t)
    
    # Check if file exists
    if (file.exists(sample_file)) {
      cat(sprintf("File %s already exists, skipping...\n", sample_file))
      next
    }
    
    # Set parameters
    N <- nrow(grd_scaled)          # Population size
    n <- level                     # Sample size
    p <- rep(n/N, N)               # Inclusion probabilities
    
    # Combine inclusion probabilities and auxiliary variables using cbind
    X <- cbind(p, grd_scaled)
    
    # Perform balanced sampling using the cube method
    sample_indices <- cube(p, X)
    
    # Get the sampled data
    sampled_df <- grd[sample_indices, ]
    
    # Add index and ksat columns
    sampled_df$index <- index[sample_indices]
    sampled_df$ksat <- ksat[sample_indices]
    
    # Output the sampled data to a CSV file
    write.csv(sampled_df, sample_file, row.names = FALSE)
    cat(sprintf("Generated sample set %d for sample size %d\n", t, level))
  }
}