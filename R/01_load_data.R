library(skimr)
library(readr)
library(dplyr)
library(randomForest)
library(caret)

df_train <- read_csv("Data/train_final_v5.csv")
df_test <- read_csv("Data/test_final_v5.csv")

df_train$long_term %>% table() #0: 666,127   1: 25,312 
df_test$long_term %>% table()  #0: 320,208   1: 9,841 
df_train %>% colnames()
# df <- df_train %>% bind_rows(df_test)
df_v5 <- df_train %>% bind_rows(df_test)
df_v5$long_term %>% table()
df <- df_v5

#drug1 exp and drug1
df <- df %>% rename(drug1 = drug2)
df$drug1 %>% table()
#drug name one-hot code
df$drug1_Codeine     <- ifelse(df$drug1 == "Codeine", 1, 0)
df$drug1_Hydrocodone <- ifelse(df$drug1 == "Hydrocodone", 1, 0)
df$drug1_Oxycodone   <- ifelse(df$drug1 == "Oxycodone", 1, 0)
df$drug1_Tramadol    <- ifelse(df$drug1 == "Tramadol", 1, 0)
df$drug1_Other       <- ifelse(df$drug1 == "Other", 1, 0)

#columns to exclude
col_out <- c("drug1", "r_mme_v5_maxday_c", "mme_v5_maxday_c")

# Data preprocessing
df <- df %>% select(-all_of(col_out)) %>% mutate(across(where(is.character), as.factor))
df <- na.omit(df) #1,021,444 
rm(col_out,col_out2)
# skim(df)
colnames(df)

# split df into training and testing sets
set.seed(42)
train_index <- createDataPartition(df$long_term, p = 0.8, list = FALSE)
df_train <- df[train_index, ]
df_test <- df[-train_index, ]
rm(train_index)

# ----------------------------------------
# Common preprocessing: design matrix and target preparation
# ----------------------------------------
x_train <- model.matrix(long_term ~ . -1, data = df_train)
x_test <- model.matrix(long_term ~ . -1, data = df_test)
y_train <- as.numeric(as.character(df_train$long_term))
y_test <- as.numeric(as.character(df_test$long_term))
true_factor <- factor(y_test, levels = c(0, 1))
thresholds <- seq(0.01, 0.99, by = 0.01)

