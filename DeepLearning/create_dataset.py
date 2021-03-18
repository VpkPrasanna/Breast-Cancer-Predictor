import os
import pandas as pd

# benign = Not cancerous - 0
# maligant = cancerous - 1
bengin_Path="BCP_Image_Dataset/benign"
maligant_Path = "BCP_Image_Dataset/malignant"

benign_list = os.listdir(bengin_Path)
maligant_list = os.listdir(maligant_Path)

benign_list_new = [i for i in benign_list if "_mask" not in i]
maligant_list_new = [i for i in maligant_list if "_mask" not in i]

benign_values = [0] * len(benign_list_new)
maligant_values = [1] * len(maligant_list_new)

res = dict(zip(benign_list_new, benign_values)) 
res1 = dict(zip(maligant_list_new, maligant_values)) 

final_df1 = pd.DataFrame(res.items(),columns=['imageid','target'])
final_df2 = pd.DataFrame(res1.items(),columns=['imageid','target'])

final_df = pd.concat([final_df1,final_df2])
final_df.to_csv("DeepLearning/breast_tumor.csv",index=False)
print(len(final_df))


# for Masked Image 
benign_list_new_mask = [i for i in benign_list if "_mask"  in i]
maligant_list_new_mask = [i for i in maligant_list if "_mask"  in i]

benign_values_mask = [0] * len(benign_list_new_mask)
maligant_values_mask = [1] * len(maligant_list_new_mask)

res_mask = dict(zip(benign_list_new_mask, benign_values_mask)) 
res1_mask = dict(zip(maligant_list_new_mask, maligant_values_mask)) 

final_df1_mask = pd.DataFrame(res_mask.items(),columns=['imageid','target'])
final_df2_mask = pd.DataFrame(res1_mask.items(),columns=['imageid','target'])

final_df_mask = pd.concat([final_df1_mask,final_df2_mask])
final_df_mask.to_csv("DeepLearning/breast_tumor_mask.csv",index=False)
print(len(final_df_mask))