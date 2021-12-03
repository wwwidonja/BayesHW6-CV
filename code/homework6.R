# libraries --------------------------------------------------------------------
library(cmdstanr)
library(ggplot2)
library(ggdist)
library(bayesplot)
library(posterior)
library(tidyverse)
library(mcmcse)
library(caret)
library(dplyr)
library(cowplot)
library(ggthemes)
library(RColorBrewer)
library(maps)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(rgeos)
library(scico)
require(gridExtra)
library(patchwork)
library(plyr)
library(reshape)
library(loo) # for WAIC and LOOIC calculations
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# data for map plot -----
# load the data

data <- read.csv("../data/happiness.csv")

data
d2 <- data %>% select(-year)
## get world data
averages <- aggregate(d2[,2:ncol(d2)], list(d2$country), mean)
xx <- dplyr::rename(averages, c("admin"=colnames(averages)[1],  "econ"="economy", "p_corr"="perceived_corruption"))
world <- ne_countries(scale = "medium", returnclass = "sf")

colnames(world)
world <- world %>%
  inner_join(xx, by="admin")

### Plot the grid ------------------



score <- ggplot(data = world) +
  geom_sf(aes(fill = score)) +
  scale_fill_viridis_c(option = "inferno", trans = "sqrt") +
  labs( x = "Longitude", y = "Latitude") +
  ggtitle("Happiness") + theme_fivethirtyeight() + theme(plot.title = element_text(size = 14, face = "bold"))+ theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))

GDP <- ggplot(data = world) +
  geom_sf(aes(fill = econ)) +
  scale_fill_viridis_c(option = "D", trans = "sqrt") +
  labs( x = "Longitude", y = "Latitude") +
  ggtitle("GDP") + theme_fivethirtyeight() + theme(plot.title = element_text(size = 14, face = "bold"))+theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))

corruption <- ggplot(data = world) +
  geom_sf(aes(fill = p_corr )) +
  scale_fill_scico(trans="sqrt") +
  labs( x = "Longitude", y = "Latitude") +
  ggtitle("Perc. Corruption") + theme_fivethirtyeight() + theme(plot.title = element_text(size = 14, face = "bold"))+theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))

plot_grid(score, GDP, corruption, nrow=3)

## get continent data for our models data ----------------
data <- read.csv("../data/happiness.csv")
world <- ne_countries(scale = "medium", returnclass = "sf")
d3 <- data
ws <- data.frame(world["continent", "admin", "subregion"])
ws <- data.frame(country=world$admin, continent=world$continent, region=world$subregion)
d3 <- d3 %>% inner_join(ws, by="country")
d3


## scaling/unscaling ----------------
mean_score <- mean(d3$score)
sd_score <- sd(d3$score)
mean_economy <- mean(d3$economy)
sd_economy <- sd(d3$economy)
mean_corr <- mean(d3$perceived_corruption)
sd_corr <- sd(d3$percieved_corruption)

d3 <- d3 %>% mutate(score_s=as.vector(scale(score)),
                        economy_s=as.vector(scale(economy)),
                    corruption_s=as.vector(scale(perceived_corruption)))

## onehot encoding ----------------
nd <- dummyVars("country ~ .", data=d3)
nd
final_df <- data.frame(predict(nd, newdata=d3))

## train/test split ----------------
train <- final_df %>% filter(year<=2018)
test <- final_df %>% filter (year > 2018)


# data wrangling and initializing stan data for 1st and 2nd model ---------
n_train <- nrow(train)
score_train <- train$score_s
economy_train <- train$economy_s
corruption_train <- train$corruption_s
n_test <- nrow(test)
score_test <- test$score_s
economy_test <- test$economy_s
corruption_test <- test$corruption_s

stan_data <- list(n_train=n_train,
                  score_train=score_train,
                  corruption_train=corruption_train,
                  economy_train=economy_train,
                  n_test=n_test,
                  score_test=score_test,
                  corruption_test=corruption_test,
                  economy_test=economy_test)

## loading models ------------
InteractionModel <- cmdstan_model("../models/model_1.stan")
NoInteractionModel <- cmdstan_model("../models/model_2.stan")
PolyInteractionModel <- cmdstan_model("../models/model_3.stan")
ContModel <- cmdstan_model("../models/model_4.stan")
OnlyContModel <- cmdstan_model("../models/model_4_nopoly.stan")
models <- c(NoInteractionModel, PolyInteractionModel)

# for loop -------------
df_mse_train <- data.frame(mse=numeric(), order=factor())
df_mse_test <- data.frame(mse=numeric(), order=factor())
fin_df <- data.frame(mse=numeric(), kind=factor(), model=factor())
kinds <- c('Linear', 'Poly (3)')
k <- 1
for (model in models){
  # fit
  fit <- model$sample(
    data = stan_data,
    parallel_chains = 4,
    seed = 1
  )
  mcmc_trace(fit$draws(c("b_corruption", "b_economy", "intercept", "sigma")))
  fit$summary(c("b_corruption", "b_economy", "intercept", "sigma"))
  df2 <- as_draws_df(fit$draws(c("mse_train", "mse_test")))
  df2 <- rename(df2,c('mse_train'='IN SAMPLE', 'mse_test'='OUT OF SAMPLE'))
  df2 <- df2%>% pivot_longer(cols=c('IN SAMPLE', 'OUT OF SAMPLE'), names_to =  "kind", values_to = "mse")
  #df2 <- df2 %>% select(-c('.chain', '.iteration', '.draw'))
  df2$model <- kinds[k]
  k <- k+1
  fin_df <- rbind(fin_df, df2)
}
fin_df
df2


# cont model -------------
conts_train <- data.matrix(train %>% dplyr:: select(starts_with("continent")))
conts_test <- data.matrix(test %>% dplyr:: select(starts_with("continent")))
#stan_data_2 <- stan_data

### comment this out (and above in) to use poly (don't forget to change model!)
stan_data2 <- list(n_train=n_train,
                   score_train=score_train,
                   n_test=n_test,
                   score_test=score_test)
stan_data2$num_continents <- ncol(conts_train)
stan_data2$conts_train <- conts_train
stan_data2$conts_test <- conts_test

# fit
fit <- OnlyContModel$sample(
  data = stan_data2,
  parallel_chains = 4,
  seed = 1
)
df2 <- as_draws_df(fit$draws(c("mse_train", "mse_test")))
df2 <- rename(df2,c('mse_train'='IN SAMPLE', 'mse_test'='OUT OF SAMPLE'))
df2 <- df2%>% pivot_longer(cols=c('IN SAMPLE', 'OUT OF SAMPLE'), names_to =  "kind", values_to = "mse")
#df2 <- df2 %>% select(-c('.chain', '.iteration', '.draw'))
df2$model <- "Continents"
fin_df <- rbind(fin_df, df2)

# region model -------------
conts_train <- data.matrix(train %>% dplyr:: select(starts_with("region")))
conts_test <- data.matrix(test %>% dplyr:: select(starts_with("region")))
#stan_data3 <- stan_data

##same as above.
stan_data3 <- list(n_train=n_train,
                   score_train=score_train,
                   n_test=n_test,
                   score_test=score_test)

stan_data3$num_continents <- ncol(conts_train)
stan_data3$conts_train <- conts_train
stan_data3$conts_test <- conts_test

# fit
fit <- OnlyContModel$sample(
  data = stan_data3,
  parallel_chains = 4,
  seed = 1
)
df2 <- as_draws_df(fit$draws(c("mse_train", "mse_test")))
df2 <- rename(df2,c('mse_train'='IN SAMPLE', 'mse_test'='OUT OF SAMPLE'))
df2 <- df2%>% pivot_longer(cols=c('IN SAMPLE', 'OUT OF SAMPLE'), names_to =  "kind", values_to = "mse")
#df2 <- df2 %>% select(-c('.chain', '.iteration', '.draw'))
df2$model <- "Subregions"
fin_df <- rbind(fin_df, df2)


stan_data4 <- stan_data


stan_data4$num_continents <- ncol(conts_train)
stan_data4$conts_train <- conts_train
stan_data4$conts_test <- conts_test

# fit
fit <- ContModel$sample(
  data = stan_data4,
  parallel_chains = 4,
  seed = 1
)
df2 <- as_draws_df(fit$draws(c("mse_train", "mse_test")))
df2 <- rename(df2,c('mse_train'='IN SAMPLE', 'mse_test'='OUT OF SAMPLE'))
df2 <- df2%>% pivot_longer(cols=c('IN SAMPLE', 'OUT OF SAMPLE'), names_to =  "kind", values_to = "mse")
#df2 <- df2 %>% select(-c('.chain', '.iteration', '.draw'))
df2$model <- "Poly - Reg"
fin_df <- rbind(fin_df, df2)




# draw the plot of models -------------
is <- fin_df[fin_df$kind == 'IN SAMPLE', ] 
p1 <- ggplot(data = is, aes(y = mse, x = reorder(model, -mse), fill=reorder(model, -mse))) +
  stat_eye(alpha = 0.75) +
  xlab("Model name") +
  ylab("MSE") + ggtitle('Out of sample MSE') + theme_fivethirtyeight() +
  theme(
    strip.text = element_text(
      size = 12, color = "red", face = "bold.italic"
    ), axis.title.x = element_text(color="black"), axis.title.y = element_text(color="black")
  ) + scale_fill_brewer(palette="Dark2") + theme(legend.position = "none")

ggplot(data = fin_df, aes(y = mse, x = reorder(model, -mse))) +
  stat_eye(fill = "red", alpha = 0.75) +
  xlab("Model name") +
  ylab("MSE") +
  facet_wrap(~ kind, ncol=1) + theme_fivethirtyeight() +
  theme(
  strip.text = element_text(
    size = 12, color = "red", face = "bold.italic"
  ), axis.title.x = element_text(color="black"), axis.title.y = element_text(color="black")
)

#### akaike -------------
# storages
log_lik <- list()
df_aic <- data.frame(AIC=c(), model=factor())


models <- c(InteractionModel, PolyInteractionModel)
datas <- list(stan_data, stan_data, stan_data2, stan_data3)
names <- c('Lin', 'Poly')

fit <- InteractionModel$sample(
  data = stan_data,
  parallel_chains = 4,
  seed = 1
)
log_lik[[1]] <- fit$draws(c("log_lik"))
df_ll <- as_draws_df(fit$draws(c("log_lik")))
df_aic <- data.frame(AIC=-2*rowSums(df_ll) + 2*2, model=as.factor('Lin'))


fit <- PolyInteractionModel$sample(
  data = stan_data,
  parallel_chains = 4,
  seed = 1
)

log_lik[[2]] <- fit$draws(c("log_lik"))
df_ll <- as_draws_df(fit$draws(c("log_lik")))
df_ll <- data.frame(df_ll %>% select(-.chain, -.iteration, -.draw))
df_aic <- data.frame(AIC=-2*rowSums(df_ll) + 2*2, model=as.factor('Poly'))



fit <- OnlyContModel$sample(
  data = stan_data2,
  parallel_chains = 4,
  seed = 1
)
log_lik[[3]] <- fit$draws(c("log_lik"))
df_ll <- as_draws_df(fit$draws(c("log_lik")))
df_ll <- data.frame(df_ll %>% select(-.chain, -.iteration, -.draw))
df_aic <- rbind(df_aic,
                data.frame(AIC=-2*rowSums(df_ll) + 2*((ncol(stan_data2$conts_train))), model=as.factor('Continents')))

fit <- OnlyContModel$sample(
  data = stan_data3,
  parallel_chains = 4,
  seed = 1
)
log_lik[[4]] <- fit$draws(c("log_lik"))
df_ll <- as_draws_df(fit$draws(c("log_lik")))
df_ll <- data.frame(df_ll %>% select(-.chain, -.iteration, -.draw))
df_aic <- rbind(df_aic,
                data.frame(AIC=-2*rowSums(df_ll) + 2*((ncol(stan_data3$conts_train))), model=as.factor('Regions')))


fit <- ContModel$sample(
  data = stan_data4,
  parallel_chains = 4,
  seed = 1
)
log_lik[[5]] <- fit$draws(c("log_lik"))
df_ll <- as_draws_df(fit$draws(c("log_lik")))
df_ll <- data.frame(df_ll %>% select(-.chain, -.iteration, -.draw))
df_aic <- rbind(df_aic,
                data.frame(AIC=-2*rowSums(df_ll) + 2*((2+ncol(stan_data3$conts_train))), model=as.factor('Regions')))

# WAIC
df_waic <- data.frame(WAIC=numeric(), SE=numeric(), model=factor())
mod_names <- c("Linear", "Poly (3)", "Continents", "Subregions", "Poly-Reg")
for (i in 1:4) {
  waic <- waic(log_lik[[i]])
  df_waic <- rbind(df_waic, data.frame(waic=waic$estimates[3,1],
                                       SE=waic$estimates[3,2],
                                       model=as.factor(mod_names[i])))
}

# plot the WAIC graph ----------------------
ggplot(data=df_waic, aes(x=model, y=waic)) +
  geom_point(shape=16, size=2) +
  geom_linerange(aes(ymin = (waic-SE), ymax = (waic+SE)), alpha=0.3) +
  xlab("Number of predictors") +
  ylab("WAIC")


# averaging
# calculate delta_waic
df_waic$delta_waic <- abs(df_waic$waic - min(df_waic$waic))

# calculate weights
df_waic$weight <- exp(-0.5 * df_waic$delta_waic) / sum(exp(-0.5 * df_waic$delta_waic))
df_waic$weight <- round(df_waic$weight, 2)

# plot
ggplot(data=df_waic, aes(x=model, y=weight)) +
  geom_bar(stat="identity", fill='red') +
  xlab("model") +
  ylab("Akaike weight") +
  theme_fivethirtyeight() +
  ylim(0, 1) + theme(
    strip.text = element_text(
      size = 12, color = "red", face = "bold.italic"
    ), axis.title.x = element_text(color="black"), axis.title.y = element_text(color="black"),
    legend.position = "none") + ggtitle('Akaike Weights2')

# LOOIC
df_loo <- data.frame(loo=numeric(), SE=numeric(), model=factor())

for (i in 1:5) {
  r_eff <- relative_eff(log_lik[[i]])
  loo <- loo(log_lik[[i]], r_eff=r_eff)
  df_loo <- rbind(df_loo, data.frame(loo=loo$estimates[3,1],
                                     SE=loo$estimates[3,2],
                                     model=as.factor(mod_names[i])))
}

# plot the MSE / LOO COMPARISON GRAPH ----------------------
p2 <- ggplot(data=df_loo, aes(x=reorder(model, -loo), y=loo, color=reorder(model, -loo))) +
  geom_point(size=3)+
  geom_errorbar(aes(ymin=loo-SE, ymax=loo+SE), width=.5, size=0.8,
                position=position_dodge(0.05)) +
  xlab("Model name") +
  ylab("LOOIC") + theme_fivethirtyeight() + theme(
    strip.text = element_text(
      size = 12, color = "red", face = "bold.italic"
    ), axis.title.x = element_text(color="black"), axis.title.y = element_text(color="black"),
    legend.position = "none") + ggtitle('LOOIC') + scale_color_brewer(palette="Dark2")

fin <- p1 / p2
fin

# averaging
# calculate delta_waic
df_loo$delta <- abs(df_loo$loo - min(df_loo$loo))

# calculate weights
df_loo$weight <- exp(-0.5 * df_loo$delta) / sum(exp(-0.5 * df_loo$delta))
df_loo$weight <- round(df_loo$weight, 2)

##df_loo -- TO EXTRACT LOOS -----------------
df_loo


#-------------- plot LOOIC/AKAIKE WEIGHTS
ggplot(data=df_loo, aes(x=model, y=weight)) +
  geom_bar(stat="identity", fill='red') +
  xlab("model") +
  ylab("LOOIC weight") +
  theme_fivethirtyeight() +
  ylim(0, 1) + theme(
    strip.text = element_text(
      size = 12, color = "red", face = "bold.italic"
    ), axis.title.x = element_text(color="black"), axis.title.y = element_text(color="black"),
    legend.position = "none") + ggtitle('LOOIC Weight')