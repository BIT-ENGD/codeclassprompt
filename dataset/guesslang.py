from  torch.utils.data import Dataset,DataLoader

class lang_info(object):
    def __init__(self) -> None:
        pass

class  ds_guesslang(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data=list() 

    
    def __len__(self):
        return len(self.data)

    def __gettime__(self,idx):
        return self.data[idx] 

def GuessLang_load_data(dir,train,valid,test,extra):

    data=ds_guesslang()


    return data


# loader = DataLoader(
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0,
# )