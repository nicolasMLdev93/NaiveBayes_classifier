import pandas as pd

df = pd.read_csv('../data/spam_ham_dataset.csv')

def clean_dataset(df):
    # Eliminamos duplicados y datos nullos
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    id_column = df.columns[0]
    df.drop(columns=['label',id_column],inplace=True,axis=1)

    return df


clean_ds = clean_dataset(df)
# Crear un nuevo dataset con datos limpios
clean_ds.to_csv('clean_ds.csv',index=False)

