# libs
library(tidyverse) # pipe, maps and tibble
library(lubridate) # manipulacao de datas
library(corrplot)  # correlation plot
library(ape)       # disk dendogram
library(RColorBrewer) # color palette

# recuperando contagem de palavras gravada no post anterior
beers <- readRDS("./content/post/data/beer_tm/beers.rds")

# define um "beer type"
beer.types <- beers %>%
  select( super.tipo, sub.tipo, tipo ) %>%
  distinct() %>%
  mutate(
    sub.tipo = case_when( is.na(sub.tipo) ~ super.tipo,
                          T ~sub.tipo )
  ) %>%
  arrange( super.tipo, sub.tipo ) %>%
  mutate( type.id = 1:nrow(.),
          super.tipo = as.factor(super.tipo),
          sub.tipo   = as.factor(sub.tipo) )


beer_wordc <- readRDS("./content/post/data/beer_tm/beer_wordc.rds")
glimpse(beer_wordc)

beers %>% 
  group_by(tipo) %>% 
  tally() %>% 
  filter(n>1) -> selected.types

beer_corr <- beer_wordc %>%
  filter(tipo %in% selected.types$tipo) %>%
  select(-super.tipo) %>%
  group_by(tipo) %>%
  mutate(proporcao = n / sum(n))  %>%
  subset(n > 1) %>%
  select(-n) %>% 
  spread(tipo, proporcao)

# replace NAs with 0 because an NA 
# is an observation of 0 words
beer_corr[is.na(beer_corr)] <- 0 

dmtx <- 1-cor(beer_corr[,-1], use = "pairwise.complete.obs") 

dmds <- cmdscale(dmtx,k=3)

dmds %>%
  as_tibble() %>%
  mutate( tipo = rownames(dmds) ) %>%
  inner_join(beer.types) -> st

library(plotly)
plot_ly(st, x=~V1, y=~V2, z=~V3, color=~super.tipo) %>%  
  add_markers()
