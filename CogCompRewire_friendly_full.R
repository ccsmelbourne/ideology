# -----------------------------------------------------------------------------
# ----------  Ideology, Communication, and Polarization Simulation ------------
# -----------------------------------------------------------------------------

# This script contains the code required to run the simulations and produce the  
# plots from Kashima et al. (2020), Ideology, Communication, and Polarization. 

# The key parameters that are varied in the simulations are: 

# (1) Agent *'Cognitive Type'*, which creates populations of agents with an:
## ideological filter and ideological ego-involvement (type = 1)
## ideological filter and non-ideological ego-involvement (type = 2)
## unbiased filter and ideological ego-involvement (type = 3) and
## unbiased filter and non-ideological ego-involvement (type = 4)

# and

# (2) Agent *'Relational Mobility'*, which constrains social dynamics to be:
## non-existant (relational_mobility = 1). 
### I.e., the Tensor Product model with random input at each timestep. 
## static; communication occurs in a fully connected network with consistent 
### agent influence overtime (relational_mobility = 2). 
### I.e., the Tensor Product + Social Influence models, with no network
### rewiring. 
## Dynamic, with high relational mobility (relational_mobility = 3)
### i.e., Tensor Product + Social Influence in a dynamic influence network, 
### where agents strengthen ties to those with similar ideologies and 
### sever ties to those they disagree with. 

# Set the 'desired_output' parameter below and run all. 

desired_output <- "figure_three"

# Options: 

# (1) "figure_one" - 
# Consequences of agent cognitive style (i.e., Tensor Product).
# Plots the dot product of each agents' output and the ideology a over a run. 
# Agent type varied (1:4) with relational_mobility set to 1

# (2) "figure_two" - 
# Trajectories of opinion dynamics with communication 
# (i.e., Tensor Product + Social Influence in static influence network). 
# Plots the cosine similarity of agents output and a over a run.
# Agent type varied (1:4) with relational_mobility set to 2

# (3) "figure_three" - 
# Trajectories of opinion dynamics with high relational mobility  
# (i.e., Tensor Product + Social Influence in dynamic influence network). 
# Plots the cosine similarity of agents output and a over a run.
# Agent type varied (1:4) with relational_mobility set to 3

# (4) "figure_s3" - 
# Variability in social network structure and opinion dynamics with and 
# without relational mobility for type 2 agents. 
# Plots the mean number of opinion clusters and standard deviation of 
# opinion similarities in simulated opinion dynamics. 
# Agent type set to 2, relational_mobility = 2 (no mobility) and 3 
# (high relational mobility). 

# (5) "figure_s4" - 
# Variability in social network structure and opinion dynamics with and 
# without relational mobility for type 4 agents.
# Again, plots the mean number of opinion clusters and standard deviation of 
# opinion similarities in simulated opinion dynamics.
# Agent type set to 4, relational_mobility = 2 and 3. 

if (!exists("desired_output")) { 
  stop("set desired_output!") # shout! 
}

# Set Up  ---------------------------------------------------------------------

# install.packages(c("tidyverse", "ggpubr", "igraph"))
 library(tidyverse)  # used to make figures  
 library(ggpubr)  # used to arrange figures
 library(igraph)  # used to compute social network statistics

##  Create Functions -----------------------------------------------------------

# to normalize vectors  
normv <- function(x) { 
  x / c(sqrt(x %*% x))
}

# to compute cosine similarity 
cosim <- function(x, y) {          
  normv(x) %*% normv(y)
}

# to create sigmoid function,
sigmoid <- function(s, x) {  # s = steepness 
  1 / (1 + exp(s * (-x)))
}

##  General -------------------------------------------------------------------

set.seed(999)
agent_N <- 100  # Number of agents 
T <- 50  # Number of timesteps in each run 

if (desired_output == "figure_s3" | desired_output == "figure_s4") {
  # The simulations described in the supplementary materials compare average
  # results for agents of a particular cognitive type (type 2 or 4) with and 
  # without relational mobility over 100 runs of each. Therefore:
  conditions <- 2
  simnum_N <- 100 
  out_op_dist <- list()  # for output 
  for (c in 1:conditions) {
    out_op_dist[[c]] <- array(0, dim = c(4, T + 1, simnum_N))  
  }
  names(out_op_dist) <- c("no_mobility", "mobility")
} else { 
  # simulations in the main body of the paper are used to illustrate opinion
  # dynamics across four cognitive types with varying relational mobility, so:
  conditions <- 4
  simnum_N <- 1
  OutX <- list() # for output
  for (c in 1:conditions) {
    OutX[[c]] <- array(0, dim = c(agent_N, 5, T + 1))  
  }
  names(OutX) <- c("type_1", "type_2", "type_3", "type_4")
  out_a <- matrix(0, 5, conditions * simnum_N) # for tracking ideology a
}

##  Sub Model 1. Tensor Product -----------------------------------------------

w_i <- .5 # Learning rate
# 0 < w_i < 1, where larger values = slower learning

op <-  10  # Opposition strength
# for ideology representation a and contrasting ideology b

##  Sub Model 2. Social Influence ---------------------------------------------

K <- 1 # number of iterations for Friedkin et al. grounding process each timestep

Cf <- diag(5)  # C matrix from original model; replaced with identity matrix here 
# as this is absorbed in the Tensor Product model. 

# Get going -------------------------------------------------------------------

types_run <- 1 

##  Simulations from main body of paper --------------------------------------

if (desired_output == "figure_one" | 
    desired_output == "figure_two" |
    desired_output == "figure_three") {
  while (types_run < 5) {
    
    for (simnum in 1:simnum_N) {
      
      # Set Agent Cognitive Type 
      type <- types_run
      
      # Set Relational Mobility
      if (desired_output == "figure_one") {  
        relational_mobility <- 1 # Friedkin et al. model is not included
      } 
      if (desired_output == "figure_two") { 
        relational_mobility <- 2  # Social influence via static network
      } 
      if (desired_output == "figure_three") {
        relational_mobility <- 3  # Social influence via dynamic network
      } 
      
      # Initialise sub model 1. Tensor Product 
      
      # Set up the ideology representation a and contrasting ideology b 
      a <- normv(runif(5, -1, 1))  # |a| = 5, normalized w values from -1 to 1
      out_a[, type] <- a  # NOTE - this only works if simnum_N = 1 
      b <- normv(-a + rnorm(5, 0, 1 / op))
      
      # Create the starting input matrix X0
      X0 <- matrix(0, agent_N, 5)  
      for (i in 1:agent_N) {
        X0[i, ] <- normv(runif(5, -1, 1))
      }
      
      # Set up to store output 
      OutX[[types_run]][, , 1] <- X0  # starting with t = 1
      X <- X0
      
      # Individualise learning 
      s0 <- matrix(0, agent_N, 5)  # self representation
      e0 <- matrix(0, agent_N, 5)  # learned ideology (of a with noise)
      eu <- matrix(0, agent_N, 5)  # individualised belief system 
      for (agent_i in 1:agent_N) {
        s0[agent_i,] <- normv(runif(5, -1, 1))       
        e0[agent_i,] <- normv(a + rnorm(5, 0, .1))  
        eu[agent_i,] <- normv(runif(5, -1, 1))
      }
      
      # Set agent cognitive style/type
      C <- array(0, dim=c(5, 5, agent_N)) # interpeter 
      M <- array(0, dim=c(5, 5, agent_N)) # memory 
      if (type == 1 | type == 2) {   
        ideological_filter <- TRUE   
      } else {                      
        ideological_filter <- FALSE
      }
      if (type == 1 | type == 3) {
        ideological_ego <- TRUE
      } else {
        ideological_ego <- FALSE
      }
      for (agent_i in 1:agent_N) {
        if(ideological_filter == TRUE) {  
          # get ideological filter 
          C[, , agent_i] <- normv(e0[agent_i, ]) %*% t(normv(e0[agent_i, ]))
        } else {  
          # otherwise unbiased filter
          C[, , agent_i] <- diag(5)                                    
        }
        if (ideological_ego == TRUE) {
          # get ideological ego-involvement
          M[, , agent_i] <- s0[agent_i, ] %*% t(e0[agent_i, ])              
        } else {
          # otherwise have non-ideological ego-involvement
          M[, , agent_i] <- s0[agent_i, ] %*% t(eu[agent_i, ])             
        } 
      }
      
      # Set up empty matrices:
      input <- matrix(0, agent_N, 5)  # input from each agent
      e <- matrix(0, agent_N, 5)  # interpretations of input received 
      s <- matrix(0, agent_N, 5)  # source memory 
      E <- matrix(0, 5, 5)  # episodic memory 
      output <- matrix(0, agent_N, 5)  # output 
      
      
      # Initialise sub model 2. Social Influence 
      
      # Set up the influence matrix, W
      W <- matrix(0, agent_N, agent_N)      
      W0 <- matrix(0, agent_N, agent_N)  # influence matrix at t = 1
      uW <- matrix(0, agent_N, agent_N)    
      St <- rnorm(agent_N, 0, 1)
      stubborn <- runif(agent_N, 0, 0.6) # agents influence on itself, wii, a trait 
      for (i in 1:agent_N) {
        for (j in 1:agent_N) {
          uW[, j] <- sigmoid(2, St[j] - St[i])
          W0[i, ] <- (( (1- stubborn[i]) / (sum(uW[i, ]) - uW[i, i])) * uW[i, ])
          W0[i, i] <- stubborn[i]  
        }
      }
      # Make sure influence matrix can be modified  
      W <- W0 
      Wb <- W0
      
      # Set up A 
      A <- matrix(0, agent_N, agent_N)
      for (i in 1:agent_N) {
        A[i, i] <- 1 - (diag(W0)[i])  # 1 - influence 
      }
      
      # Set up I
      I <- diag(agent_N) 
      
      # Add network dynamics 
      if (relational_mobility == 1) { 
        friedkin <- FALSE  
        change_ties <- FALSE  # or lack there of
        change_severe <- FALSE  
      } else {  
        friedkin <- TRUE
        if (relational_mobility == 2) {  # is this efficient? No.. but it works
          change_ties <- FALSE
          change_severe <- FALSE
        } else {  
          change_ties <- TRUE
          change_severe <- TRUE 
        }
      }
      
      # Ticks start (main)-----------------------------------------------------
      
      for (t in 1:T) {  # in each timestep
        
        for (agent_i in 1:agent_N) {  # each agent goes through
          
          # The Tensor Product Model 
          
          # input is taken from X
          input[agent_i, ] <- normv(X[agent_i, ])
          # interpeted via C
          e[agent_i, ] <- C[, , agent_i] %*% input[agent_i, ]  
          # source attributed via M
          s[agent_i, ] <- M[, , agent_i] %*% e[agent_i, ]  
          # episodic memory formed 
          E <- s[agent_i, ] %*% t(e[agent_i, ])  
          # LTM memory updated
          M[, , agent_i] <-  w_i * M[, , agent_i] +  (1 - w_i) * E 
          # output generated
          output[agent_i, ] <- 
            # intepretor filters product of self-opinion and memory 
            C[, , agent_i] %*% normv(as.vector(t(s0[agent_i, ]) %*%  M[, , agent_i]))   
          # output saved to feed in the Friedkin 
          X[agent_i, ] <- output[agent_i, ]
        }
        
        # Then, the Friedkin et al. model is repeated K times 
        
        if (friedkin == TRUE) {
          Xk <- X
          # Update output matrix via communication (grounding process) 
          for (k in 1:K) {
            Xk <- A %*% W %*% Xk %*% t(Cf) + 
              ((I - A) %*% X)
            X <- Xk
          }
        }
        
        # Store output from round, will act as input for next
        OutX[[types_run]][, , t + 1] <- X
        
        # Next, social network dynamics play out (if permitted )
        
        # Modify influence matrix W
        if (change_ties == TRUE) {
          if (change_severe == TRUE) {
            for (i in 1:agent_N) {
              for (j in 1:agent_N) {
                # callibrate influence at resonance and cut ties if dissonant
                if (cosim(X[i, ], X[j, ]) > 0) {
                  # boost!
                  Wb[i, j] <- sigmoid(2, cosim(X[i, ], X[j, ]))
                } else {
                  # bye!
                  Wb[i, j] <- 0
                }
              }  # end j 
              if (sum(Wb[i, ]) - Wb[i, i] == 0 ) {
                W[i, ] <- Wb[i, ]
              } else {
                W[i, ] <- (( (1- stubborn[i]) / (sum(Wb[i, ]) - Wb[i, i])) * Wb[i, ])
              }
              W[i, i] <- stubborn[i] # self-influence is trait, same over runs 
            } # end i 
            
            # If network is static, option to decrease influence if dissonant
            # (not included in the paper)
          } else {   
            if (cosim(X[i, ], X[j, ]) > 0) {
              Wb[i, j] <- sigmoid(2, cosim(X[i, ], X[j, ]))
            } else {
              Wb[i, j] <- sigmoid(-2, cosim(X[i, ], X[j, ]))
            }
          }
        }
        
        # When only incorperating the Tensor Product model, 
        if (friedkin == FALSE) {
          # create random input for the next time step. 
          for (i in 1:agent_N) {
            X[i, ] <- normv(runif(5, -1, 1))
          }
        }
      } # end timestep
    }  # end of runs for agent cognitive type 
    
    types_run <- types_run + 1
  }  
  # end simulations, ready to plot!  
}

##  Simulations from supplementary materials ----------------------------------

if (desired_output == "figure_s3" | desired_output == "figure_s4") {
  
  while (types_run < 3) {
    
    for (simnum in 1:simnum_N) {
      
      # Set Agent Cognitive Type 
      if (desired_output == "figure_s3") {
        type <- 2 
      } else { # figure_s4
        type <- 4 
      }
      
      # Set Relational Mobility
      if (types_run == 1) {
        relational_mobility <- 2  # run no rewiring
      } else {
        relational_mobility <- 3  # then rewiring 
      }
      
      # Initialise sub model 1. Tensor Product 
      
      # Set up the ideology representation a and contrasting ideology b 
      a <- normv(runif(5, -1, -1))  # |a| = 5, normalized w values from -1 to 1
      b <- normv(-a + rnorm(5, 0, 1 / op))
      
      # Create the starting input matrix X0
      X0 <- matrix(0, agent_N, 5)  
      for (i in 1:agent_N) {
        X0[i, ] <- normv(runif(5, -1, 1))
      }
      X <- X0
      
      # Individualise learning 
      s0 <- matrix(0, agent_N, 5)  # self representation
      e0 <- matrix(0, agent_N, 5)  # learned ideology (of a with noise)
      eu <- matrix(0, agent_N, 5)  # individualised belief system 
      for (agent_i in 1:agent_N) {
        s0[agent_i,] <- normv(runif(5, -1, 1))       
        e0[agent_i,] <- normv(a + rnorm(5, 0, .1))  
        eu[agent_i,] <- normv(runif(5, -1, 1))
      }
      
      # Set agent cognitive style/type
      C <- array(0, dim=c(5, 5, agent_N)) # interpeter 
      M <- array(0, dim=c(5, 5, agent_N)) # memory 
      if (type == 2) {
        ideological_filter <- TRUE
        ideological_ego <- FALSE
      }
      
      if (type == 4) {
        ideological_filter <- FALSE
        ideological_ego <- FALSE
      }
      for (agent_i in 1:agent_N) {
        if(ideological_filter == TRUE) {  
          # get ideological filter 
          C[, , agent_i] <- normv(e0[agent_i, ]) %*% t(normv(e0[agent_i, ]))
        } else {  
          # otherwise unbiased filter
          C[, , agent_i] <- diag(5)                                    
        }
        if (ideological_ego == TRUE) {
          # get ideological ego-involvement
          M[, , agent_i] <- s0[agent_i, ] %*% t(e0[agent_i, ])              
        } else {
          # otherwise have non-ideological ego-involvement
          M[, , agent_i] <- s0[agent_i, ] %*% t(eu[agent_i, ])             
        } 
      }
      
      # Set up empty matrices:
      input <- matrix(0, agent_N, 5)  # input from each agent
      e <- matrix(0, agent_N, 5)  # interpretations of input received 
      s <- matrix(0, agent_N, 5)  # source memory 
      E <- matrix(0, 5, 5)  # episodic memory 
      output <- matrix(0, agent_N, 5)  # output 
      
      # Initialise sub model 2. Social Influence 
      
      # Set up the influence matrix, W
      W <- matrix(0, agent_N, agent_N)      
      W0 <- matrix(0, agent_N, agent_N)  # influence matrix at t = 1
      uW <- matrix(0, agent_N, agent_N)    
      St <- rnorm(agent_N, 0, 1)
      stubborn <- runif(agent_N, 0, 0.6) # agents influence on itself, wii, a trait 
      for (i in 1:agent_N) {
        for (j in 1:agent_N) {
          uW[, j] <- sigmoid(2, St[j] - St[i])
          W0[i, ] <- (( (1- stubborn[i]) / (sum(uW[i, ]) - uW[i, i])) * uW[i, ])
          W0[i, i] <- stubborn[i]  
        }
      }
      # Make sure influence matrix can be modified  
      W <- W0 
      Wb <- W0
      
      # Set up A 
      A <- matrix(0, agent_N, agent_N)
      for (i in 1:agent_N) {
        A[i, i] <- 1 - (diag(W0)[i])  # 1 - influence 
      }
      
      # Set up I
      I <- diag(agent_N) 
      
      # Add network dynamics 
      if (relational_mobility == 2) {  
        change_ties <- FALSE
        change_severe <- FALSE
      } else {  
        change_ties <- TRUE
        change_severe <- TRUE 
      }
      
      # Describe initial opinion distribution
      simx <- matrix(0, agent_N, agent_N)  # to track similarity in input
      simxadj <- matrix(0, agent_N, agent_N)
      for (i in 1:agent_N) {
        for (j in 1:agent_N) {
          simx[i, j] <- cosim(X0[i, ], X0[j, ])
        }
      }
      
      # Converse cosine similarities to network ties
      for (i in 1:agent_N) {
        for (j in 1:agent_N) {
          if (simx[i, j] > 0) {   # ties based on similarity of ideologies a
            simxadj[i, j] <- 1   # no ties for neutral or oposing others 
          } else {
            simxadj[i, j] <- 0
          }
        }
      }
      
      simnet <- igraph::graph_from_adjacency_matrix(simxadj)
      op_community <- igraph::cluster_walktrap(simnet)
      # Saving: 
      # 1. average similarity of ideologies
      op_mean <- mean(simx) 
      # 2. and the standard deviation
      op_sd <- sd(simx) 
      # 3. number of network clusters
      op_cluster <- max(igraph::membership(op_community)) 
      # Starting with t = 1
      out_op_dist[[types_run]][, 1, simnum] <- c(type, op_mean, op_sd, op_cluster)
      
      
      # Ticks start (supp)-----------------------------------------------------
      
      for (t in 1:T) {  # in each timestep
        
        for (agent_i in 1:agent_N) {  # each agent goes through
          
          # The Tensor Product Model 
          
          # input is taken from X
          input[agent_i, ] <- normv(X[agent_i, ])
          # interpeted via C
          e[agent_i, ] <- C[, , agent_i] %*% input[agent_i, ]  
          # source attributed via M
          s[agent_i, ] <- M[, , agent_i] %*% e[agent_i, ]  
          # episodic memory formed 
          E <- s[agent_i, ] %*% t(e[agent_i, ])  
          # LTM memory updated
          M[, , agent_i] <-  w_i * M[, , agent_i] +  (1 - w_i) * E 
          # output generated
          output[agent_i, ] <- 
            # intepretor filters product of self-opinion and memory 
            C[, , agent_i] %*% normv(as.vector(t(s0[agent_i, ]) %*%  M[, , agent_i]))   
          # output saved to feed in the Friedkin 
          X[agent_i, ] <- output[agent_i, ]
        }
        
        # Then, the Friedkin et al. model as a group, repeated K times 
        
        Xk <- X
        # Update output matrix via communication (grounding process) 
        for (k in 1:K) {
          Xk <- A %*% W %*% Xk %*% t(Cf) + 
            ((I - A) %*% X)
          X <- Xk
        }
        
        # Social network dynamics play out (if permitted )
        
        # Modify influence matrix W
        if (change_ties == TRUE) {
          if (change_severe == TRUE) {
            for (i in 1:agent_N) {
              for (j in 1:agent_N) {
                # callibrate influence at resonance and cut ties if dissonant
                if (cosim(X[i, ], X[j, ]) > 0) {
                  # boost!
                  Wb[i, j] <- sigmoid(2, cosim(X[i, ], X[j, ]))
                } else {
                  # bye!
                  Wb[i, j] <- 0
                }
              }  # end j 
              if (sum(Wb[i, ]) - Wb[i, i] == 0 ) {
                W[i, ] <- Wb[i, ]
              } else {
                W[i, ] <- (( (1- stubborn[i]) / (sum(Wb[i, ]) - Wb[i, i])) * Wb[i, ])
              }
              W[i, i] <- stubborn[i] # self-influence is trait, same over runs 
            } # end i 
          } 
        }
        
        # Compute relevant statistics for opinion distribution
        # starting with cosine similarity between opinion vectors
        for (i in 1:agent_N) {
          for (j in 1:agent_N) {
            simx[i, j] <- cosim(X[i, ], X[j, ])
          }
        }
        
        # convert cosine similarities to network ties 
        for (i in 1:agent_N) {
          for (j in 1:agent_N) {
            if (simx[i, j] > 0) {   
              simxadj[i, j] <- 1   
            } else {
              simxadj[i, j] <- 0
            }
          }
        } # end of agents actions for the timestep
        
        # Get summary stats
        op_mean <- mean(simx)
        op_sd <- sd(simx)
        simnet <- igraph::graph_from_adjacency_matrix(simxadj)
        op_community <- igraph::cluster_walktrap(simnet)
        op_cluster <- max(igraph::membership(op_community)) 
        out_op_dist[[types_run]][, t + 1, simnum] <- c(type, op_mean, op_sd, op_cluster)
      }
    }
    types_run <- types_run + 1  
  }
} # end supplementary simulations 


# Plot  -----------------------------------------------------------------------

if (desired_output == "figure_one") {
  # get out dot product of each agent's output and a
  out <- matrix(0, agent_N * conditions, T + 3)
  hold <- list()
  for (c in 1:conditions) {
    a_2 <- out_a[, c] # get relevant a out 
    for (row in 1:agent_N) {  
      # save agent type in first col
      out[(agent_N * c) + (row - agent_N), 1] <- c 
      # label agents 
      out[(agent_N * c) + (row - agent_N), 2] <- row
    }
    hold <- OutX[[c]]
    for (timestep in 1:T + 1) {
      # get agents output at time t in relevant condition
      colvec <- t(hold[, , timestep])
      for (i in 1:agent_N) {
        out[(agent_N * c) + ( i - agent_N), 1 + timestep] <- (a_2 %*% colvec[, i])
      }
    } 
  }
  # make this plotable 
  out <- as.data.frame(out)
  colnames(out) <- c("agent_type", "agent_id", 1:51)
  out <- pivot_longer(out, 
                       cols = -c(agent_type, agent_id),
                       names_to = "time", 
                       values_to = "output_dot")
  out$time <- as.numeric(out$time)
  out$agent_type <- factor(out$agent_type)
  # plot 
  ggplot(out, aes(x = time, y = output_dot, color = factor(agent_id))) +
    geom_line() + 
    xlim(1,50) +
    theme_classic() + 
    theme(legend.position = "none") + 
    facet_wrap(~ agent_type)
}

if (desired_output == "figure_two" | desired_output == "figure_three") {
  time <- rep(0:T, conditions)
  out <- list()
  for (c in 1:conditions) {
    out[[c]] <- matrix(0, T + 1, agent_N + 1)
    out[[c]][, 1] <- c
    a_2 <- out_a[, c] 
    for (i in 1:agent_N) {
      out[[c]][1, i + 1] <- cosim(a_2, OutX[[c]][i, , 1])
      for (t in 1:T) {
        out[[c]][t+1, i + 1] <- cosim(a_2, OutX[[c]][i, , t+1])
      }
    }
    hold <- as.data.frame(out[[c]])
    hold <- cbind(time, hold)
    colnames(hold)[2] <- "agent_type"
    colnames(hold)[3:102] <- c(1:agent_N)
    hold <- pivot_longer(hold, 
                         cols = -c(time,agent_type),
                         names_to = "agent",
                         values_to = "opinion")
    out[[c]] <- hold 
  }
  out_all <- rbind(out[[1]], out[[2]], out[[3]], out[[4]])
  ggplot(out_all, aes( x = time, y = opinion, color = factor(agent))) + 
    ylim(-1,1) +
    geom_line() + 
    theme_classic() + 
    theme(legend.position = "none") + 
    facet_wrap(~ agent_type)
}

if (desired_output == "figure_s3" | desired_output == "figure_s4" ) {
  # plot mean opinion clusters per timestep across runs 
  # and sd of opinion similarities per timestep across runs
  out_op_dat <- list()
  for (c in 1:conditions) {
    out_op_dat[[c]] <- matrix(0, simnum_N * (T + 1), 5)
    for (simnum in 1:simnum_N) {
      for (t in 0:T) {
        out_op_dat[[c]][t + 1 + (simnum - 1) * (T + 1), 1] <- simnum
        out_op_dat[[c]][t + 1 + (simnum - 1) * (T + 1), 2] <- t + 1
        out_op_dat[[c]][t + 1 + (simnum - 1) * (T + 1), 3] <- out_op_dist[[c]][2, t + 1, simnum] # op_mean
        out_op_dat[[c]][t + 1 + (simnum - 1) * (T + 1), 4] <- out_op_dist[[c]][3, t + 1, simnum] # op_sd 
        out_op_dat[[c]][t + 1 + (simnum - 1) * (T + 1), 5] <- out_op_dist[[c]][4, t + 1, simnum] # clusters
      }
    }
    hold <- as.data.frame(out_op_dat[[c]])
    colnames(hold) <- c("simulation","time","opinion_mean","opinion_SD","cluster")
    mean_cluster <- stats::aggregate(hold$cluster, 
                                     list(hold$time), 
                                     FUN = "mean")
    if (c == 1) {
      mean_cluster$mobility <- "no_mobility"
      } else { 
      mean_cluster$mobility <- "high_relational_mobility"
    }
    colnames(mean_cluster)[1:2] <- c("time", "mean_clusters")
    # mean_sd 
    mean_sd <- stats::aggregate(hold$opinion_SD, 
                                list(hold$time), 
                                FUN = "mean")
    colnames(mean_sd) <- c("time", "mean_opinion_sd")
    hold <- cbind(mean_cluster, mean_sd)
    out_op_dat[[c]] <- hold
  }
 out_all <- rbind(out_op_dat[[1]], out_op_dat[[2]])
 out_all <- out_all[-c(4)] # drop duplicate col 
 # plot 
 cluster_plot <- 
   ggplot(out_all, aes(x = time, y = mean_clusters, color = factor(mobility))) + 
   geom_line() + 
   scale_colour_discrete(name  ="Agent Mobility") + 
   theme_classic() +
   theme(legend.position = "above") + 
   ylab("average number of clusters ") 
 sd_plot <- 
   ggplot(out_all, aes(x = time, y = mean_opinion_sd, color = factor(mobility))) + 
   geom_line() + 
   scale_colour_discrete(name  ="Agent Mobility") + 
   theme_classic() + 
   theme(legend.position = "above") + 
   ylab("average standard deviation of opinion") 
 
 ggarrange(sd_plot, cluster_plot, 
           labels = c("A", "B"))
}


# Q? --> ellep@student.unimelb.edu.au -----------------------------------------
# (aimed for reproducability and ease of understanding, not efficency)  
  
  
