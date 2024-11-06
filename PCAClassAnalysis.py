# prompt: find classes in coco dataset and then repeat for other datasets like fgvc and food101
import json
import os
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
pca = PCA(n_components=2)
class_names = {}
with open(os.path.join(".","train_class_names.json"),'r') as f:
    class_names = json.load(f)
Loss=torch.nn.CrossEntropyLoss()

tokens={}
model, preprocess = clip.load("ViT-B/32",device='cuda')
with torch.inference_mode(True):
    for key in class_names.keys():
        names=class_names[key]
        print("datasets: ",key)
        print("names: ",names)
        names=clip.tokenize(names).to('cuda')
        tokens.update({key:model.encode_text(names).cpu()})

fullpoints=torch.cat(tuple(list(tokens.values())),axis=0).to(torch.float)
optimumscore=fullpoints/torch.norm(fullpoints,dim=-1,keepdim=True)
optimumscore=optimumscore
optimumscore=optimumscore@optimumscore.T
##plot this as a confusion matrix
LossLabels=torch.arange(0,optimumscore.shape[0],device=optimumscore.device)
loss=Loss(optimumscore,LossLabels)

print("loss: ",loss)
plt.matshow(optimumscore.cpu().detach().numpy())
plt.title('Confusion Matrix of Original Classes, optimal score is '+str(loss.item()))
plt.savefig("confusion_matrix.png")

LossByBatchSize={}
#I want to show the minimum score by batch size by taking a random sample of the vectors...
for batchsize in [2,4,8,16,32,64,128,256,512]:
    LossLabels=torch.arange(0,batchsize,device=optimumscore.device)
    scores=[]
    for i in range(200):
        randomindices=torch.randperm(optimumscore.shape[0])[:batchsize]
        selection=fullpoints[randomindices]
        selection=selection/torch.norm(selection,dim=-1,keepdim=True)
        selection=selection@selection.T
        scores.append(Loss(selection,LossLabels).item())
    LossByBatchSize.update({batchsize:np.mean(scores)})


#plot the loss by batch size
#new plot
plt.figure()
plt.plot(list(LossByBatchSize.keys()),list(LossByBatchSize.values()))
plt.title('Minimum Expected Loss by Batch Size')
#use log scale on X axis
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.show()
plt.savefig("batchsize_loss.png")


X_pca = pca.fit_transform(fullpoints.detach().cpu().numpy())
optimumscore=fullpoints
#normalise the optimum score

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
for i, key in enumerate(tokens.keys()):
    points=pca.transform(tokens[key])
    ax.scatter(points[:,0],points[:,1], label=key, alpha=0.5)

ax.set_title('2D PCA of Text Embeddings for each class')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
plt.show() 

#save the pca plt
fig.savefig("PCA.png")

#next get the MSCOCO dataset and do the same thing, but iterate over all captions, and then encode them and use the same PCA to plot them

from COCODataModule import CustomCOCODatasetWithClasses
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import clip
import zipfile
from pySmartDL import SmartDL
import time
import torch

#load the dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
if not os.path.exists(os.path.join(".","data","annotations")):
            URLS=["http://images.cocodataset.org/zips/train2017.zip","http://images.cocodataset.org/annotations/annotations_trainval2017.zip"]
            
            for url in URLS:
                print("Downloading",url)
                obj=SmartDL(url,os.path.join(".","data",str(url).split('/')[-1]),progress_bar=False)
                obj.FileName=str(url).split('/')[-1]
                if not os.path.exists(obj.get_dest()):
                    obj.start(blocking=False)
                    print("obj Path ",obj.get_dest())
                while not obj.isFinished():
                    #print("Speed: %s" % obj.get_speed(human=True))
                    print("Eta: %s" % obj.get_eta(human=True))
                    time.sleep(5)
                if obj.isSuccessful():
                    print("Downloaded: %s" % obj.get_dest())
                path = obj.get_dest()
                if obj.FileName.startswith("annotations"):
                    print("Extracting annotations")
                    print("path:",path)

                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        try:
                            zip_ref.extractall(".","data")
                        except:
                            print("Error extracting annotations")
                            print("path:",path)
                            
                else:
                    print("Extracting images")
                    print("path:",path)
                    if obj.FileName.endswith(".zip"):
                        print("Extracting zip")
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            try:
                                zip_ref.extractall(".","data")
                            except:
                                print("Error extracting images")
                                print("path:",path)
                    print("Extracted: %s" % path)
        #now load the dataset
train_dataset = CustomCOCODatasetWithClasses(os.path.join(".","data","train2017"), os.path.join(".","data","annotations","captions_train2017.json"),os.path.join(".","data","annotations","instances_train2017.json"), preprocess)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=3, persistent_workers=True)
fig = plt.figure(figsize=(10, 7))

with torch.inference_mode():
    labels=np.array([])
    representationslist=[]
    X_pca_list=[]
    for i in range(5):
        for batch in train_loader:
            _, targets,captions = batch
            captions=captions.squeeze(1).to('cuda')
            representations = model.encode_text(captions)
            #do PCA on the representations
            X_pca = pca.transform(representations.cpu().numpy())
            
            #extend the labels
            # print(X_pca)
            representationslist.append(representations.cpu().detach().numpy())
            X_pca_list.append( X_pca)
            labels=np.concatenate((labels,targets),axis=0)
    X_pca_list=np.concatenate(X_pca_list,axis=0)
    for target in set(labels):
        mask=labels==target
        points=X_pca_list[mask]
        plt.scatter(points[:,0],points[:,1], label="Class "+str(target), alpha=0.5) 
plt.title('2D PCA of Text Embeddings for each class in COCO dataset')   
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
fig.savefig("PCA_COCO.png")


LossByBatchSizeCOCO={}
representations=torch.tensor(np.concatenate(representationslist,axis=0)).to(torch.float)
#I want to show the minimum score by batch size by taking a random sample of the vectors...

X_pca_list=torch.tensor(X_pca_list).to(torch.float)
for batchsize in [2,4,8,16,32,64,128,256,512]:
    LossLabels=torch.arange(0,batchsize,device=optimumscore.device)
    scores=[]
    for i in range(20000):
        randomindices=torch.randperm(representations.shape[0])[:batchsize]
        selection=representations[randomindices]
        selection=selection/torch.norm(selection,dim=-1,keepdim=True)
        selection=selection@selection.T
        scores.append(Loss(selection,LossLabels).item())
    LossByBatchSizeCOCO.update({batchsize:np.mean(scores)})


#plot the loss by batch size
#new plot
plt.figure()
plt.plot(list(LossByBatchSize.keys()),list(LossByBatchSize.values()),label="Original Classes")
plt.plot(list(LossByBatchSizeCOCO.keys()),list(LossByBatchSizeCOCO.values()),label="COCO Embeddings")
plt.title('Minimum Expected Loss by Batch Size')
#use log scale on X axis
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.savefig("batchsize_lossCOCO.png")

plt.figure()
plt.plot(list(LossByBatchSize.keys()),list(LossByBatchSize.values()),label="Original Classes")
plt.plot(list(LossByBatchSizeCOCO.keys()),list(LossByBatchSizeCOCO.values()),label="COCO Embeddings")
plt.title('Minimum Expected Loss by Batch Size')
#use log scale on X axis
plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("linearbatchsize_lossCOCO.png")