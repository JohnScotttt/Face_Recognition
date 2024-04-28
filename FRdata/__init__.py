from . import FR11Ndata
    
def get_dataset(**kwargs):
    if kwargs['mode'] == "11N":
        return FR11Ndata.FRDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset mode {kwargs['mode']}")

def get_dataloader(**kwargs):
    if kwargs['mode'] == "11N":
        return FR11Ndata.FRDataloader(**kwargs)
    else:
        raise ValueError(f"Unknown dataloader mode {kwargs['mode']}")