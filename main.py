import PyGol as pygol
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('phishing_dataset.csv')
# remove serial no column
df = df.drop("Unnamed: 0", axis=1)


feature_cols=df.columns[:-1]
target_col=df.columns[-1]

# logic_rules=pygol.prepare_logic_rules(
#                 df,feature_cols, 
#                 file_name="generated/BK.pl",
#                 meta_information="generated/meta_df.info", 
#                 default_div=1, conditions={}
#             )

# examples=pygol.prepare_examples(
#             df,target_col, example_label="target", 
#             meta_information="generated/meta_df.info",
#             positive_example="generated/Pos.f", 
#             negative_example="generated/Neg.n"
#         )


# const=pygol.read_constants_meta_info(meta_information="generated/meta_df.info")

# P, N = pygol.bottom_clause_generation(
#             file="generated/BK.pl", 
#             constant_set = const, 
#             container = "dict",
#             positive_example="generated/Pos.f", 
#             negative_example="generated/Neg.n",
#             positive_file_dictionary="generated/Pos_BC",
#             negative_file_dictionary="generated/Neg_BC"
#         )


Train_P, Test_P, Train_N, Test_N=pygol.pygol_train_test_split(
                                    test_size=0.20, 
                                    positive_file_dictionary="generated/Neg_BC",
                                    negative_file_dictionary="generated/Pos_BC"
                                )
p = 50
n = 100


model= pygol.pygol_learn(
            Train_P, Train_N, file="generated/BK.pl",
            max_literals=2, exact_literals=True, 
            key_size=1, min_pos=p, max_neg=n
        )


metrics=pygol.evaluate_theory_prolog(model.hypothesis,"generated/BK.pl",Test_P, Test_N)


pickle.dump(model, open( f'models/{p}_{n}_{metrics.accuracy}_{metrics.precision}.pkl', "wb" ) )