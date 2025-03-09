import biom
from biom.util import biom_open
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def preprocess_data():
    # get meta data
    names_meta_v = [9102, 9159, 9230, 9249]
    get_file_meta = lambda x: 'data/hospital/sample_information_from_prep_'+str(x)+'.tsv'
    hospital_meta = pd.concat([pd.read_csv(get_file_meta(i), sep='\t') for i in names_meta_v]).drop_duplicates()
    print('Data: Meta data loaded.')

    # merge biome tables
    table1 = biom.load_table("data/hospital/150/133520_all.biom")
    table2 = biom.load_table("data/hospital/150/134073_all.biom")
    table3 = biom.load_table("data/hospital/150/134769_all.biom")
    table4 = biom.load_table("data/hospital/150/134858_all.biom")
    merged_table = table1.merge(table2).merge(table3).merge(table4)

    with biom_open('data/input/merged_biom_table.biom', 'w') as f:
        merged_table.to_hdf5(f, 'created table')
    
    # merge meta data with biome data 
    hospital_meta = hospital_meta[['sample_name', 'sample_sarscov2_screening_result', 'study_sample_type']]

    # query to get relevant rows
    data = hospital_meta.query(
        "study_sample_type in ['stool', 'forehead', 'inside floor', 'nares'] & \
        sample_sarscov2_screening_result in ['not detected', 'positive']"
    ).reset_index(drop=True)

    # split data
    X = data.drop(columns=['study_sample_type', 'sample_sarscov2_screening_result'], axis=1)
    y = data[['study_sample_type', 'sample_sarscov2_screening_result']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


    # write txt file  with names of sample used for training
    with open("data/input/training_samples.txt", "w") as f:
        for s in X_train['sample_name']:
            f.write(f'{s}\n')
    
    # read the names of each sample into a array
    with open("data/input/training_samples.txt", "r") as f:
        samples_train = [s.strip() for s in f.readlines()]
    samples_train
    
    # write txt file  with names of sample used for testing
    with open("data/input/test_samples.txt", "w") as f:
        for s in X_test['sample_name']:
            f.write(f'{s}\n')
    
    # read the names of each sample into a array
    with open("data/input/test_samples.txt", "r") as f:
        samples_test = [s.strip() for s in f.readlines()]
    samples_test
    
    def check_covid_positive(row):
        return int(row =='positive')
            
    names_meta_v = [9102, 9159, 9230, 9249]
    get_file_meta = lambda x: 'data/hospital/sample_information_from_prep_'+str(x)+'.tsv'
    hospital_meta = pd.concat([pd.read_csv(get_file_meta(i), sep='\t') for i in names_meta_v]).drop_duplicates()
    hospital_meta['has_covid'] = hospital_meta['sample_sarscov2_screening_result'].apply(check_covid_positive)
    training_data = hospital_meta.loc[hospital_meta["sample_name"].isin(samples_train)]
    test_data = hospital_meta.loc[hospital_meta["sample_name"].isin(samples_test)]

    # save training metadata to tsv
    training_data.to_csv("data/input/training_metadata.tsv", sep="\t", index=False)
    test_data.to_csv("data/input/test_metadata.tsv", sep="\t", index=False)

    # get training metadata per sample
    training_metadata = pd.read_csv('data/input/training_metadata.tsv', sep="\t")
    training_metadata_inside_floor = training_metadata[training_metadata['study_sample_type'] == 'inside floor']
    training_metadata_forehead = training_metadata[training_metadata['study_sample_type'] == 'forehead']
    training_metadata_stool = training_metadata[training_metadata['study_sample_type'] == 'stool']
    training_metadata_nares = training_metadata[training_metadata['study_sample_type'] == 'nares']
    training_metadata_inside_floor.to_csv("data/input/training_metadata_inside_floor.tsv", sep="\t", index=False)
    training_metadata_forehead.to_csv("data/input/training_metadata_forehead.tsv", sep="\t", index=False)
    training_metadata_stool.to_csv("data/input/training_metadata_stool.tsv", sep="\t", index=False)
    training_metadata_nares.to_csv("data/input/training_metadata_nares.tsv", sep="\t", index=False)

    # get training metadata per sample
    test_metadata = pd.read_csv('data/input/test_metadata.tsv', sep="\t")
    test_metadata_inside_floor = test_metadata[test_metadata['study_sample_type'] == 'inside floor']
    test_metadata_forehead = test_metadata[test_metadata['study_sample_type'] == 'forehead']
    test_metadata_stool = test_metadata[test_metadata['study_sample_type'] == 'stool']
    test_metadata_nares = test_metadata[test_metadata['study_sample_type'] == 'nares']
    test_metadata_inside_floor.to_csv("data/input/test_metadata_inside_floor.tsv", sep="\t", index=False)
    test_metadata_forehead.to_csv("data/input/test_metadata_forehead.tsv", sep="\t", index=False)
    test_metadata_stool.to_csv("data/input/test_metadata_stool.tsv", sep="\t", index=False)
    test_metadata_nares.to_csv("data/input/test_metadata_nares.tsv", sep="\t", index=False)

    print('Data: AAM, DNABERT, DNABERT-2, and GROVER data preprocessing completed.')