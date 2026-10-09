import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold ,train_test_split

df = pd.read_csv(cluster_path+ '/cbica/home/innanis/clam_idh/dataset_csv/gbm_lgg_idh_labels_curated_data.csv')

x = df['slide_id']
y = df['label']

kf = StratifiedKFold(n_splits=10,random_state =1, shuffle = True)
kf.get_n_splits(x, y)

df_splitwise = pd.DataFrame()
count = 0
for train_index, test_index in kf.split(x,y):
    X_tr_va, X_test = df.iloc[train_index], df.iloc[test_index]
    X_train, X_val,  y_train, y_val = train_test_split(X_tr_va.slide_id,X_tr_va.label,test_size=0.1,shuffle=True,stratify=X_tr_va.label)
    df_splitwise = pd.DataFrame()
    df_splitwise['train'] = X_train.reset_index(drop=True)
    df_splitwise['train_label'] = y_train.reset_index(drop=True)
    df_splitwise['val'] = X_val.reset_index(drop=True)
    df_splitwise['val_label'] = y_val.reset_index(drop=True)
    df_splitwise['test'] = X_test.slide_id.reset_index(drop=True)
    df_splitwise['test_label'] = X_test.label.reset_index(drop=True)
    df_splitwise.to_csv(cluster_path+ '/cbica/home/innanis/clam_idh/splits/tcga_idh_10_folds/splits_'+str(count)+'.csv')
    count+=1

df = pd.concat([pd.read_csv('/home/shubham/clam_workstation/splits/tcga_gbm_lgg_survival_100/splits_'+str(i)+'.csv') for i in range(10)], ignore_index=True)


df = pd.read_csv(cluster_path + '/cbica/home/innanis/clam_resnet18/dataset_new/df_idh_label.csv')
df.case_id.nunique()
df_1 = df.drop_duplicates(subset=["case_id"], keep='first')
df_1.case_id.nunique()
x = df_1['case_id']
y = df_1['label']


kf = StratifiedKFold(n_splits=10,random_state =42, shuffle = True)
kf.get_n_splits(x, y)

count =0
for train_index, test_index in kf.split(x,y):
    X_tr_va, X_test = df_1.iloc[train_index], df_1.iloc[test_index]
    X_train, X_val,  y_train, y_val = train_test_split(X_tr_va.case_id,X_tr_va.label,test_size=0.1,shuffle=True,stratify=X_tr_va.label)
    train_slides , val_slides, test_slides = df[df['case_id'].isin(X_train)].slide_id, df[df['case_id'].isin(X_val)].slide_id, df[df['case_id'].isin(X_test['case_id'])].slide_id
    df_splitwise = pd.DataFrame()
    df_splitwise['train'] =train_slides.reset_index(drop=True)
    df_splitwise['val'] = val_slides.reset_index(drop=True)
    df_splitwise['test'] = test_slides.reset_index(drop=True)
    df_splitwise.to_csv(cluster_path+ '/cbica/home/innanis/clam_idh/splits/tcga_idh_10_folds_excluded_g2_oligos_blank_methylated/splits_'+str(count)+'.csv')
    count+=1


df_d = pd.DataFrame()
for m in range(10):
    df_splitwise = pd.read_csv(cluster_path+ '/cbica/home/innanis/clam_idh/splits/tcga_idh_10_folds_excluded_g2_oligos_blank_methylated/splits_'+str(m)+'.csv')
    train_slides = df_splitwise.train
    train_cases = df[df['slide_id'].isin(train_slides)].case_id.nunique()

    val_slides = df_splitwise.val.dropna()
    val_cases = df[df['slide_id'].isin(val_slides)].case_id.nunique()

    test_slides = df_splitwise.test.dropna()
    test_cases = df[df['slide_id'].isin(test_slides)].case_id.nunique()

    df_d = df_d.append({'split':m, 'train_slides':len(train_slides),'train_cases':train_cases, 'val_slides':len(val_slides),'val_cases':val_cases, \
            'test_slides':len(test_slides),'test_cases':test_cases},ignore_index=True)

df_d.to_csv(cluster_path+ '/cbica/home/innanis/clam_idh/splits/tcga_idh_10_folds_excluded_g2_oligos_blank_methylated/splits_summary.csv',index=False)


