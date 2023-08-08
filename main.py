'''
Date: 2023-06-20 15:35:06
FirstEditors: pystar360 pystar360@py-star.com
LastEditors: pystar360 pystar360@py-star.com
LastEditTime: 2023-08-08 09:49:23
FilePath: /organize_images_into_clusters/main.py
'''
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchreid
from PIL import Image
from torchreid.data.transforms import build_transforms
import torch
from torchreid.metrics import compute_distance_matrix
from communities.algorithms import louvain_method


class mydataset(Dataset):
    def __init__(self, root, transform_fun=None):
        super().__init__()
        self.root = root
        self.transform_fun = transform_fun
        self._init()
    
    def _init(self,):
        self.data_list = []
        for img_f in self.root.iterdir():
            if img_f.suffix in [".jpg", ".png"]:
                self.data_list.append(img_f)

    def __getitem__(self, index):
        img_f = self.data_list[index]
        img = Image.open(img_f)
        if self.transform_fun is not None:
            img = self.transform_fun(img)
        return str(img_f), img
    
    def __len__(self):
        return len(self.data_list)
    
    def get_datalist(self):
        return self.data_list


def build_model():
    # Please refer to https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.
    model = torchreid.models.build_model('osnet_x1_0', num_classes=751, pretrained=False)
    torchreid.utils.load_pretrained_weights(model, 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth')
    model.eval()
    return model


@torch.no_grad()
def get_feat(model, dataloader, device=torch.device("cpu")):
    model.to(device)
    outputs = []
    for img_f, data in dataloader:
        data = data.to(device)
        output = model(data)
        outputs.append(output)
    return torch.cat(outputs)


def bulid_data(root, batch_size=None):
    root = Path(root)
    _, transform_te = build_transforms(128, 64)
    dataset = mydataset(root, transform_te)
    data_list = dataset.get_datalist()
    if batch_size is None:
        batch_size = len(data_list)
    dataLoader = DataLoader(dataset, batch_size)
    return data_list, dataLoader


# def organize_folders(mask, data_list, root):
#     num = len(data_list)

#     res = { i: [] for i in range(num) }
#     for i in range(num):
#         tmp_mask = mask[i]
#         tmp_res = torch.where(tmp_mask==True)[0].numpy().tolist()
#         res[i] = tmp_res

#     for k, v in res.items():
#         if len(v) > 1:
#             save_path = root / str(k)
#             save_path.mkdir(exist_ok=True)

#             src_f = root / data_list[k].name
#             if src_f.exists():
#                 src_f.rename(save_path/src_f.name)
#             for i in v[1:]:
#                 src_f = root / data_list[i].name
#                 if src_f.exists():
#                     src_f.rename(save_path/src_f.name)

#     for tmp_root in root.iterdir():
#         if tmp_root.is_dir():
#             n = len(os.listdir(tmp_root))
#             if n < 2:
#                 for jpg_f in tmp_root.iterdir():
#                     jpg_f.rename(jpg_f.parent.parent / jpg_f.name)
#                 tmp_root.rmdir()


def organize_folders(mask, data_list, root):
    comm, _ = louvain_method(mask)
    for i, tmp_set in enumerate(comm):
        if len(tmp_set) > 1:
            save_root = root / str(i)
            save_root.mkdir(exist_ok=True)
            for idx in tmp_set:
                jpg_f = data_list[idx]
                jpg_f.rename(save_root/jpg_f.name)


if __name__ == "__main__":
    model = build_model()

    root = Path("img")
    data_list, dataLoader = bulid_data(root, batch_size=16)
    
    device = torch.device("cuda:0")
    output = get_feat(model, dataLoader, device) 
    dist_mat = compute_distance_matrix(output, output, "cosine")
    thre = 0.05
    mask = dist_mat < thre
    mask = mask.cpu()
    organize_folders(mask, data_list, root)

