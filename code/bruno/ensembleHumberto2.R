humberto1  <- fread("./humberto.csv");
humberto2 <- fread("./humberto2.csv");
joao      <- fread("./v_007.csv");

hist(humberto1$CustomerInterest, nclass=200);
hist(humberto2$CustomerInterest, nclass=200);
hist(joao$CustomerInterest,      nclass=200);

################################################################################
# Rankings of basic solutions
################################################################################
r_humberto1 <- humberto1;
r_humberto2 <- humberto2;
r_joao <- joao;

r_humberto1 <- r_humberto1[with(r_humberto1, order(CustomerInterest)), ];
r_humberto2 <- r_humberto2[with(r_humberto2, order(CustomerInterest)), ];
r_joao      <- r_joao[with(r_joao, order(CustomerInterest)), ];

r_humberto1$CustomerInterest <- (1:nrow(r_humberto1))/nrow(r_humberto1);
r_humberto2$CustomerInterest <- (1:nrow(r_humberto2))/nrow(r_humberto2);
r_joao$CustomerInterest      <- (1:nrow(r_joao))/nrow(r_joao);

hist(r_humberto1$CustomerInterest, nclass=200);
hist(r_humberto2$CustomerInterest, nclass=200);
hist(r_joao$CustomerInterest,      nclass=200);
################################################################################

names(r_humberto1)[2] <- "previsaoHumberto1";
names(r_humberto2)[2] <- "previsaoHumberto2";
names(r_joao)[2] <- "previsaoJoao";

ensembleHumberto <- r_humberto1;
ensembleHumberto <- merge(r_humberto1, r_humberto2, by = "PredictionIdx", all.x = T);
names(ensembleHumberto);
hist(ensembleHumberto$previsaoHumberto1,nclass=200);
hist(ensembleHumberto$previsaoHumberto2,nclass=200);
ensembleHumberto$CustomerInterest <- 0.5 * (ensembleHumberto$previsaoHumberto1 + ensembleHumberto$previsaoHumberto2);
hist(ensembleHumberto$CustomerInterest,nclass=200);
ensembleHumberto$previsaoHumberto1 <- NULL;
ensembleHumberto$previsaoHumberto2 <- NULL;

hist(ensembleHumberto$CustomerInterest, nclass=200);

#### Ranking for the first ensemble ########
r_ensembleHumberto <- ensembleHumberto;
r_ensembleHumberto <- r_ensembleHumberto[with(r_ensembleHumberto, order(CustomerInterest)), ];
r_ensembleHumberto$CustomerInterest <- (1:nrow(r_ensembleHumberto))/nrow(r_ensembleHumberto);
hist(r_ensembleHumberto$CustomerInterest, nclass=200);

############################################
# second ensemble
############################################

hist(r_joao$previsaoJoao);
hist(r_ensembleHumberto$CustomerInterest);
names(r_ensembleHumberto)[2] <- "previsaoEnsemble1";
hist(r_ensembleHumberto$previsaoEnsemble1);

ensemble2 <- r_joao;
ensemble2 <- merge(ensemble2, r_ensembleHumberto, by = "PredictionIdx", all.x = T);
ensemble2$CustomerInterest <- 0.5 * (ensemble2$previsaoJoao + ensemble2$previsaoEnsemble1);
hist(ensemble2$previsaoJoao,nclass=200);
hist(ensemble2$previsaoEnsemble1,nclass=200);
hist(ensemble2$CustomerInterest,nclass=200);
ensemble2$previsaoJoao      <- NULL;
ensemble2$previsaoEnsemble1 <- NULL;

fwrite(ensemble2, "./dataMagician.csv");
