

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import os
from utils import cosine_lr
from utils import one_hot_embedding
from utils import accuracy,clamp,normalize
import torch.nn.functional as F
from clip import clip
import matplotlib.pyplot as plt
from models.prompters import TokenPrompter, NullPrompter
from torchattacks import AutoAttack
from utils import clip_img_preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
#get default dict for logging
from collections import defaultdict
import threading
import time
import queue
def multiGPU_CLIP(model, images, text_tokens):
   
    #images shape is (batch, 3, 224, 224)
    #text_tokens shape is (batch, 77)
    #the old shape was (C,77)
    #this is why we dont use labels, and use arange instead. 


    img_embed=model.encode_image(images)
    scale_text_embed=model.encode_text(text_tokens)
    img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
    scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
    logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
    #logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
    return logits_per_image#, logits_per_text, img_embed, scale_text_embed # the shape of output WAS (C,B) but now is (B,B) as we want.


ImageNet_MEAN = (0.485, 0.456, 0.406)
ImageNet_STD = (0.229, 0.224, 0.225)


class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                **args,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss=torch.nn.CrossEntropyLoss()
        self.args = args
        add_prompt_len = 0 if args.get("add_prompt","none") == 'none' else 1
        self.upper_limit, self.lower_limit = 1, 0
        self.model, _ = clip.load('ViT-B/32', device=self.device, jit=False,download_root=self.args.get("imagenet_root","./data"))
        self.model_ori, _ =  clip.load('ViT-B/32', device=self.device, jit=False,download_root=self.args.get("imagenet_root","./data"))
        #set the original model to eval mode
        self.model_ori.eval()
        self.model_ori.requires_grad_(False)
        self.model.requires_grad_(True)
    
        self.model_text, _= None, None
        self.prompter = NullPrompter()
        self.add_prompter = TokenPrompter(add_prompt_len)
        '''
        To be implemented: place into the token prompter the POS embedding takedn straight fom CLIP, might make the training much faster! , or even try initiialising from random noise properly! 
        (Note, they have several different prompters in the model.prompters.py file, you can use them as a reference)
        '''

        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.test_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.criterion_kl = nn.KLDivLoss(reduction="sum")
        self.versioncriteria=self.args.get("keys_of_interest","dataset labelType learning_rate batch_size train_eps train_numsteps train_eps train_stepsize attack_type prompt_size add_prompt_size optimizer freeze_text".split())
        self.version="_".join([str(self.args.get(key,"")) for key in self.versioncriteria])
        print("Version is: ",self.version)

        '''
        Dear Afra, heres where you put you transformer decoder to build your image! 
        
        i.e  self.model_clean_image_generator = TransformerDecoder()
        
        You probably also want to add a loss function here, and you can do that by adding it to the forward pass.

        self.YourCriterion = nn.CrossEntropyLoss() ? maybe MSE? but I suspect you actually might want DICE loss/ 
        
        '''
        if args.get("norm",'l_inf')=='l_inf':
            self.init_delta=self.init_uniform
            self.clamp=self.clamp_inf
            self.init_batch_delta=self.init_batch_uniform
            self.batch_clamp=self.clamp_batch_inf
        elif  args.get("norm",'l_inf')=='l_2':
            self.init_delta=self.init_normal
            self.init_batch_delta=self.init_batch_normal
            self.clamp=self.clamp_2
            self.batch_clamp=self.clamp_batch_2

        else:
            raise ValueError
        if args.get("labelType","image")=="image":
            self.make_labels=self.make_image_labels
            self.insert_eval_model_hook=self.insert_visual_model_ori_hook
        elif args.get("labelType","image")=="text":
            self.make_labels=self.make_text_labels
            self.insert_eval_model_hook=self.insert_text_model_hook
        elif args.get("labelType","image")=="Modimage":

            self.make_labels=self.make_Modimage_labels
            self.insert_eval_model_hook=self.insert_visual_model_hook
        else:
            raise ValueError
        if self.args.get("attack_type","pgd")=="pgd":
            self.attack=self.attack_pgd
        elif self.args.get("attack_type","pgd")=="CW":
            self.attack= self.attack_CW
        elif self.args.get("attack_type","pgd")=="text":
            self.attack= self.attack_text_pgd
        elif self.args.get("attack_type","pgd")=="autoattack":
            self.attack=self.autoattack
        elif self.args.get("attack_type","pgd")=="Noattack":
            self.attack=self.no_attack
        else:
            raise ValueError 
    
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
        
    def insert_text_model_hook(self):
        self.text_features={}
        def hook_fn(module, input, output):
            self.text_features[module]=output
        for layer in self.model.text:
            layer.register_forward_hook(hook_fn)
    def remove_text_model_hook(self):
        for layer in self.model.text:
            layer._forward_hooks.clear()
    def insert_visual_model_hook(self):
        self.visual_features={}
        def hook_fn(module, input, output):
            self.visual_features[module]=output
        for layer in self.model.visual:
            layer.register_forward_hook(hook_fn)
    def remove_visual_model_hook(self):
        for layer in self.model.visual:
            layer._forward_hooks.clear()
    def insert_visual_model_ori_hook(self):
        self.visual_features={}
        def hook_fn(module, input, output):
            self.visual_features[module]=output
        for layer in self.model_ori.visual:
            layer.register_forward_hook(hook_fn)
    def remove_visual_model_ori_hook(self):
        for layer in self.model_ori.visual:
            layer._forward_hooks.clear()
    def make_Modimage_labels(self,images,text):
        return self.model.encode_image(images)
    def make_image_labels(self,images,text):
        return self.model_ori.encode_image(images)
    def make_text_labels(self,images,text):
        return self.model.encode_text(text)
    def init_uniform(self, X,eps):
        delta=  torch.zeros_like(X,device=self.device,).uniform_(-eps, eps)
        delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
        delta.requires_grad = True
        return delta
    
    def init_normal(self, X,eps):
            delta=torch.zeros_like(X,device=self.device)
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            return delta
    
    def clamp_inf(self,d,alpha,g,eps):
        return torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
    
    def clamp_2(self,d,alpha,g,eps):
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=eps).view_as(d)
        return d
    
    @torch.enable_grad()
    def attack_text_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(text_tokens,epsilon)
        self.insert_eval_model_hook()
        self.make_labels(X,text_tokens) #do this with hooks 
        clean_features=self.text_features
        for _ in range(attack_iters):

            #step 1: modify text tokens
            #step 2: pass through CLIP model module that saves features,
            #step 3: Loss= cosine similarity of clean features to dirty features. 
            #step 4: now consider loss. 
            text_tokens+=delta
            


            img_embed=self.model.encode_image(X)
            #ensure self.model has text hooks 
            self.insert_eval_model_hook()
            scale_text_embed=self.make_labels(X,text_tokens)
            features=self.text_features
            #do Loss between each layer
            text_loss=torch.zeros((X.shape[0],X.shape[0]),device=self.device)
            for layer in features.keys():
                itemA=features[layer]
                itemB=clean_features[layer]
                itemA=itemA/itemA.norm(dim=-1, keepdim=True)
                itemB=itemB/itemB.norm(dim=-1, keepdim=True)
                similarities= itemA@itemB.T  # should be B,B in shape, 
                text_loss+=self.CETextLoss(similarities)
            self.log("text_loss",text_loss)

            #step 5: backpropagate, making noise closer to clean features
            text_loss.backward()
            #step 6: remove hooks and zero grad
            self.remove_text_model_hook()
            delta.grad.zero_()


            #step 7: now do attack as normal
            d = delta

            #I want to find a way to maximize the loss while minimizing text loss

            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
            logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
            # logits_per_text, img_embed, scale_text_embed


            loss = self.criterion(logits_per_text, torch.arange(X.size(0), device=self.device))
            loss.backward()
            self.log("attack_loss",loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return X,text_tokens+delta
    
    #insert function decorator to ensure this ALWAys has grad
    @torch.enable_grad()
    def attack_pgd(self,  X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        losses=[]
        
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            #prompted_images = self.prompter(normalize(delta + X ))
            #check prompted images has grad
            new_images = delta+X
            prompted_images = torch.div(torch.sub(new_images, self.mu_img), self.std_img) #normalize(new_images) but preserves grad
            img_embed=self.model.encode_image(prompted_images)
            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed=self.make_labels(X,text_tokens)
            scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            output = img_embed_norm @ scale_text_embed.t()
            loss = self.criterion(output, torch.arange(prompted_images.size(0), device=self.device))
            loss.backward()
            losses.append(loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
             
                        
        self.log("mean_attack_losses",sum(losses)/len(losses))
        self.log("max_attack_loss",max(losses))
        self.log("min_attack_loss",min(losses))
        return X+delta,text_tokens
    
    # @torch.enable_grad()
    # def attack_pgd_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
    #     delta=self.init_delta(X,epsilon)
    #     for _ in range(attack_iters):
    #         _images = normalize(X + delta)
    #         output= multiGPU_CLIP( self.model, _images, text_tokens)
    #         loss = self.criterion(output,  torch.arange(_images.size(0), device=self.device)) #edited from original paper to remove fixed target classes
    #         loss.backward()
    #         #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
    #         self.log("attack_loss",loss)
    #         grad = delta.grad.detach()
    #         d = delta[:, :, :, :]
    #         g = grad[:, :, :, :]
    #         x = X[:, :, :, :]
    #         d=self.clamp(d,alpha,g,epsilon)
    #         d = clamp(d, self.lower_limit - x, self.upper_limit - x)
    #         delta.data[:, :, :, :] = d
    #         delta.grad.zero_()

    #     return delta

    @torch.enable_grad()
    def attack_CW(self, X, target, text_tokens, alpha,attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)

        for _ in range(attack_iters):
            # output = model(normalize(X ))
            prompted_images = self.prompter(normalize(X + delta))
            # prompt_token = self.add_prompter()
           
            img_embed=self.model.encode_image(prompted_images)
            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed=self.make_labels(X,text_tokens)
            scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            output = img_embed_norm @ scale_text_embed.t()
            
            
            label_mask = one_hot_embedding(torch.arange(X.shape(0),device=X.device), output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

            loss.backward()
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            self.log("attack_loss",loss)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return X+delta, text_tokens
    
    @torch.enable_grad()
    def attack_CW_noprompt(self, X, target, text_tokens, alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        delta=self.init_delta(X,epsilon)
        loss=[]
        for _ in range(attack_iters):
            # output = model(normalize(X ))
            #_images = normalize(X + delta)
            # output, _ = model(_images, text_tokens)
            new_images = delta+X
            prompted_images = torch.div(torch.sub(new_images, self.mu_img), self.std_img) #normalize(new_images) but preserves grad
            img_embed=self.model.encode_image(prompted_images)
            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed=self.make_labels(X,text_tokens)
            scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            output = img_embed_norm @ scale_text_embed.t()
            label_mask = one_hot_embedding(torch.arange(X.shape[0],device=X.device), output.size(1))
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            # loss = criterion(output, target)
            loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))
            #Dear Afra, here is something you should probably log with self.log("attack_loss",loss)
            self.log("attack_loss",loss)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return X+delta, text_tokens
    @torch.enable_grad()
    def autoattack(self, images, target, text_tokens,  alpha, attack_iters, restarts=1, early_stop=True, epsilon=0):
        def model_fn(x):
            output_a = multiGPU_CLIP(self.model, self.prompter(clip_img_preprocessing(x)),text_tokens)
            return output_a.to(torch.float32)

        adversary = AutoAttack(model_fn, norm='Linf', eps=epsilon, version='standard')
        adv_samples = adversary.run_standard_evaluation(images, target, bs=100)
        delta_prompt = adv_samples - images
        delta_prompt = clamp(delta_prompt, self.lower_limit - images, self.upper_limit - images)
        return images+delta_prompt, text_tokens

    def no_attack(self, images, *args, **kwargs):
            return images

    def forward(self,input):
        #This inference steps of a foward pass of the model 
        return self.model(input)
    
    def on_train_epoch_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
        if self.args.get("attack_type","pgd")=="pgd":
            self.attack=self.attack_pgd
        elif self.args.get("attack_type","pgd")=="CW":
            self.attack= self.attack_CW
        elif self.args.get("attack_type","pgd")=="text":
            self.attack= self.attack_text_pgd
        elif self.args.get("attack_type","pgd")=="autoattack":
            self.attack=self.autoattack
        elif self.args.get("attack_type","pgd")=="Noattack":
            self.attack=self.no_attack
        else:
            raise ValueError 
        self.train_alpha = self.args.get("alpha",1)
        self.train_attack_iters = self.args.get("attack_iters",5)
        self.train_epsilon = self.args.get("train_eps",1)

    
    def training_step(self, batch, batch_idx):
        #The batch is collated for you, so just seperate it here and calculate loss. 
        #By default, PTL handles optimization and scheduling and logging steps. so All you have to focus on is functionality. Here's an example...
        images, target,text = batch #label shouldnt be used here! 
        #print(text.shape)
        text=text.squeeze(1) #B,77
        text_embed=self.make_labels(images,text) #B,512
        # ori_text_embed=self.model_ori.encode_text(text)
        text_embed= text_embed/ text_embed.norm(dim=-1, keepdim=True) #B,512
        # ori_text_embed= ori_text_embed/ ori_text_embed.norm(dim=-1, keepdim=True)
        # images = self.prompter(images) #does nothing - its a null prompter
        Dirtyimages,_=self.attack(images, target, text, self.train_alpha, self.train_attack_iters, epsilon=self.train_epsilon) #B,3,224,224
        '''
        Here's where you run the dirty image through your model... first through an encoder, then through a decoder.

        output = model(normalize(images))
        rebuilt_images = model_clean_image_generator(output)
        loss2 = self.YourCriterion(rebuilt_images, images)
        #and add your loss into the total loss. 
        '''
        Dirtyimages = torch.div(torch.sub(Dirtyimages, self.mu_img), self.std_img) #normalize(Dirtyimages) but preserves grad
        # prompted_Dirtyimages = self.prompter(normalize(Dirtyimages)) #does nothing - its a null prompter
        output_of_training_model_with_dirty_images= self.model.encode_image(Dirtyimages) #B,512
        output_of_training_model_with_dirty_images= output_of_training_model_with_dirty_images/ output_of_training_model_with_dirty_images.norm(dim=-1, keepdim=True)
        output_of_training_model_with_clean_images= self.model.encode_image(images)
        output_of_training_model_with_clean_images= output_of_training_model_with_clean_images/ output_of_training_model_with_clean_images.norm(dim=-1, keepdim=True)
        output_of_pretrained_model_with_dirty_images= self.model_ori.encode_image(Dirtyimages)
        output_of_pretrained_model_with_dirty_images= output_of_pretrained_model_with_dirty_images/ output_of_pretrained_model_with_dirty_images.norm(dim=-1, keepdim=True)
        output_of_pretrained_model_with_clean_images= self.model_ori.encode_image(images)
        output_of_pretrained_model_with_clean_images= output_of_pretrained_model_with_clean_images/ output_of_pretrained_model_with_clean_images.norm(dim=-1, keepdim=True)
        '''
        we would assume if the attack is successful, the model would be more confident in the wrong class, so we can do the following check:
        Loss_to_see_attack_success = self.CrossEntropy_loss(output_of_training_model_with_dirty_images, torch.arange(images.size(0), device=self.device))

        '''
        #This loss stops the divergence of the model from the pretrained model.
        loss_between_our_training_model_and_pretrained_on_dirty_images = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_pretrained_model_with_dirty_images, dim=1))
        loss_between_our_training_model_and_pretrained_on_clean_images = self.criterion_kl(F.log_softmax(output_of_training_model_with_clean_images, dim=1), F.softmax(output_of_pretrained_model_with_clean_images, dim=1))
        
        #This loss stops the divergence of the model from the clean images.
        loss_between_dirty_and_clean_images_on_training_model = self.criterion_kl(F.log_softmax(output_of_training_model_with_dirty_images, dim=1), F.softmax(output_of_training_model_with_clean_images, dim=1))
        
        #the final criterion is the loss of the model on the dirty images, towards the target.

        '''
        Dear Afra, something for you to try here, 

        I wonder whether balancing the losses using a scaling factor might help preserve overall performance
          (something to try by adding arguments to the demoparse.py file, then setting in the lightning module init.)
        
        '''
        logits_of_training_model_with_clean_images = output_of_training_model_with_clean_images @ text_embed.T

        logits_per_dirty_image = output_of_training_model_with_dirty_images @ text_embed.T
        loss_on_training_model_with_dirty_images = self.criterion(logits_per_dirty_image, torch.arange(images.size(0), device=self.device)) # the output of this is huge compared to others. 
        self.log("Loss on training model with clean images (no grad)",self.criterion(logits_of_training_model_with_clean_images, torch.arange(images.size(0), device=self.device)))
        self.log("Loss on training model with dirty images",loss_on_training_model_with_dirty_images)
        self.log("Loss between our training model and pretrained on clean images",loss_between_our_training_model_and_pretrained_on_clean_images )
        self.log("Loss on training model with dirty and clean images",loss_between_dirty_and_clean_images_on_training_model )
        self.log("Loss between our training model and pretrained on dirty images(no_grad)",loss_between_our_training_model_and_pretrained_on_dirty_images )

        loss=loss_on_training_model_with_dirty_images + loss_between_dirty_and_clean_images_on_training_model + loss_between_our_training_model_and_pretrained_on_clean_images #+ loss_between_our_training_model_and_pretrained_on_dirty_images
        
        #self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #if doing linear regression probes, you may want to have a line like 
        # self.results.append({"imfeatures":self.model(cleanimages), "dirtyfeatures":self.model(attackedImages),"classes":batch[2],"originalmodel":self.orimodel(cleanimages),"dirtyoriginalmodel":self.orimodel(attackedImages)})
        return loss
   
    def on_train_epoch_end(self):

        l2_norm_obj = sum(p.norm(2) for p in self.model.visual.parameters())
        l2_norm_ori = sum(p.norm(2) for p in self.model_ori.visual.parameters())
        ratio = abs(l2_norm_ori - l2_norm_obj) / float(l2_norm_ori)
        abs_l2 = abs(l2_norm_ori - l2_norm_obj)
        self.log('l2_norm_obj', l2_norm_obj, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('l2_norm_ori', l2_norm_ori, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ratio', ratio, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('abs_l2', abs_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def on_validation_epoch_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
        self.cleanresults=defaultdict(list)
        self.attackedresults=defaultdict(list)
        self.data_loader_count = len(self.trainer.datamodule.val_dataloader())

        if self.args.get("test_attack_type","pgd")=="pgd":
            self.testattack=self.attack_pgd
        elif self.args.get("test_attack_type","pgd")=="CW":
            self.testattack= self.attack_CW
        elif self.args.get("test_attack_type","pgd")=="text":
            self.testattack= self.attack_text_pgd
        elif self.args.get("test_attack_type","pgd")=="autoattack":
            self.testattack=self.autoattack
        elif self.args.get("test_attack_type","pgd")=="Noattack":
            self.testattack=self.no_attack
        else:
            raise ValueError 
        self.test_alpha = self.args.get("test_alpha",1)
        self.test_attack_iters = self.args.get("test_attack_iters",5)
        self.test_epsilon = self.args.get("test_eps",1)
    
    def validation_step(self, batch, batch_idx,  dataloader_idx=0, *args, **kwargs):
        images, target,text = batch
        #a is the image, b is the target
        #get the datamodule text list to lookup the text embeddings.s
        #print(text.shape)
        prompt_token = None
        text=text.squeeze(1)      
        # if target is None:
        #     print("No target in dataloader {}".format(dataloader_idx))
        
        img_embed=self.model.encode_image(images)
        scale_text_embed=self.make_labels(images,text)
        img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        output_prompt = img_embed_norm @ scale_text_embed_norm.t()
        # if batch_idx == 0:
        #     #save the first batch of images to disk
        #     #OR project into 2d using PCA and save that to disk
        #     #plot on graph. 
        #     #labels points by class 



        self.cleanresults[dataloader_idx].append({"logits":img_embed.detach(), "textlabels":target}) #using target like this is fine because each dataloader is tested and logged independently.
        loss = self.criterion(output_prompt, torch.arange(images.size(0), device=self.device))

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, torch.arange(images.shape[0],device=images.device), topk=(1,))
        self.log('val_clean_batch_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_clean_batch_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.args.get("CW",False):
        #     delta_prompt = self.attack_CW(
        #                             images, target, text,
        #                             self.args.get("test_stepsize",2), self.args.get("test_numsteps",20), epsilon=self.args.get("test_eps",1))
        # elif self.args.get("autoattack",False):#autoattack:
        #     def model_fn(x):
        #         output_a = multiGPU_CLIP(self.model, self.prompter(clip_img_preprocessing(x)),text)
        #         return output_a.to(torch.float32)

        #     adversary = AutoAttack(model_fn, norm='Linf', eps=self.args.get("test_eps",1), version='standard')
        #     adv_samples = adversary.run_standard_evaluation(images, target, bs=100)   ##is this correct? 
        #     delta_prompt = adv_samples - images
        #     delta_prompt = clamp(delta_prompt, self.lower_limit - images, self.upper_limit - images)
        # else:
        #     delta_prompt = self.attack_pgd(images, target, text,self.args.get("test_stepsize",2), self.args.get("test_numsteps",20), epsilon=self.args.get("test_eps",1))

        # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
        dirtyImages,dirtyText=self.testattack(images, target, text, self.test_alpha, self.test_attack_iters, epsilon=self.test_epsilon)

        img_embed=self.model.encode_image(clip_img_preprocessing(dirtyImages))
        #Add test for diff between dirty and clean images here.        
        scale_text_embed=self.make_labels(images,dirtyText)   #make labels out of whatevers clean from prior line. 
        img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        output_prompt_adv = img_embed_norm @ scale_text_embed_norm.t()

        #compare class probabilities between output_prompt and output_prompt_adv
        # movement= output_prompt_adv - output_prompt
        # # self.log("movement",movement)
        # # self.log("CleanSimilarity",output_prompt)
        # plot as heatmap using matplot and save as png
        # plot as scatter plot using matplot and save as png
        # fig=plt.figure()
        # plt.imshow(output_prompt.cpu().detach().numpy())
        # plt.colorbar()
        # plt.savefig("Cleanoutput_prompt_idx={}{}.png".format(batch_idx,dataloader_idx))
        # plt.close(fig)
        # fig=plt.figure()
        # plt.imshow(movement.cpu().detach().numpy())
        # plt.colorbar()
        # plt.savefig("movement_idx={}{}.png".format(batch_idx,dataloader_idx))
        # plt.close(fig)
        


        loss = self.criterion(output_prompt_adv, torch.arange(images.size(0),device=images.device)) #shoudl be torch arange(images.size(0), device=self.device)
        self.attackedresults[dataloader_idx].append({"logits":img_embed, "textlabels":target})
        
        #TODO: add logging for text loss here when we trial the attack

        acc1 = accuracy(output_prompt_adv, torch.arange(images.size(0),device=images.device), topk=(1,))
        self.log('val_dirty_batch_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dirty_batch_acc', acc1[0].item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        

        return loss
    
    def on_validation_epoch_end(self):

        #make linear probes here, and log the results.
        

        if not hasattr(self,"Cleanclassifier"):
            self.Cleanclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)
        if not hasattr(self,"Dirtyclassifier"):
            self.Dirtyclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)
        if not hasattr(self,"generalclassifier"):
            self.generalclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)
            #we've selected 100 based on where it plateaus in the first few runs. 
        for dataset_idx in range(self.data_loader_count):
            if len(self.cleanresults[dataset_idx]) == 0 or len(self.attackedresults[dataset_idx]) == 0:
                print("No results for dataset {}".format(dataset_idx))
                continue
            GoodLabels=torch.cat([val["textlabels"] for val in self.cleanresults[dataset_idx]],dim=0).cpu().numpy()
            GoodLogits=torch.nan_to_num(torch.cat([val["logits"] for val in self.cleanresults[dataset_idx]],dim=0)).cpu().numpy()
            BadLabels=torch.cat([val["textlabels"] for val in self.attackedresults[dataset_idx]],dim=0).cpu().numpy()
            BadLogits=torch.nan_to_num(torch.cat([val["logits"] for val in self.attackedresults[dataset_idx]],dim=0)).cpu().numpy()
            #check at least 2 classes are present in the dataset
            if len(np.unique(GoodLabels)) < 2 or len(np.unique(BadLabels)) < 2:
                print("Not enough classes to run linear probes on dataset {}".format(dataset_idx))
                #skip this dataset
                continue
            self.Dirtyclassifier.fit(BadLogits, BadLabels)
            self.Cleanclassifier.fit(GoodLogits, GoodLabels)
            self.log( "Clean Classifier on Dirty Features on dataset {}".format(dataset_idx),self.Cleanclassifier.score(BadLogits, BadLabels),prog_bar=False, logger=True, sync_dist=False)
            self.log( "Dirty Classifier on Clean Features on dataset {}".format(dataset_idx),self.Dirtyclassifier.score(GoodLogits, GoodLabels),prog_bar=False, logger=True, sync_dist=False)
            self.log( "Clean Classifier on Clean Features on dataset {}".format(dataset_idx),self.Cleanclassifier.score(GoodLogits, GoodLabels),prog_bar=False, logger=True, sync_dist=False)
            self.log( "Dirty Classifier on Dirty Features on dataset {}".format(dataset_idx),self.Dirtyclassifier.score(BadLogits, BadLabels),prog_bar=False, logger=True, sync_dist=False)
            self.generalclassifier.fit(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels]))
            self.log( "General Classifier on Dirty Features on dataset {}".format(dataset_idx),self.generalclassifier.score(BadLogits, BadLabels),prog_bar=False, logger=True, sync_dist=False)
            self.log( "General Classifier on Clean Features on dataset {}".format(dataset_idx),self.generalclassifier.score(GoodLogits, GoodLabels),prog_bar=False, logger=True, sync_dist=False)
            self.log( "General Classifier on All Features on dataset {}".format(dataset_idx),self.generalclassifier.score(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels])),prog_bar=False, logger=True, sync_dist=False)

        #this should give us PLENTY of data to write about! 
        
        #delete the results to save memory
        del self.cleanresults
        del self.attackedresults

         #You could log here the val_loss, or just print something. 
        
    def configure_optimizers(self):
        # pretty sure we probably want to use the same optimizer as the original paper: the adamw optimizer
        # https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        args={"lr":self.args.get("learning_rate",1e-5)}
        if self.args.get("optimizer","sgd") == "adamw":
            optimizer_fn=torch.optim.AdamW
            args.update({"betas":(0.9, 0.999),
                  "eps":1e-08,
                  "weight_decay":self.args.get("weight_decay",0.0001)})
        elif self.args.get("optimizer","sgd") == "sgd":
            optimizer_fn=torch.optim.SGD
            args.update({"momentum":self.args.get("momentum",0.9),
                  "weight_decay":self.args.get("weight_decay",0.0001)})

        elif self.args.get("optimizer","sgd") == "adam":
            optimizer_fn=torch.optim.Adam
            args.update({"betas":(0.9, 0.999),
                  "eps":1e-08,
                  "weight_decay":self.args.get("weight_decay",0.0001)})
        else:
            raise ValueError

        #note we've adjusted this to allow the text module to move too! 
        parameters = list(self.model.visual.parameters()) if self.args.get("freeze_text",True) else list(self.model.parameters())
        optimizer = optimizer_fn(parameters,
                                        **args)
        

        if self.args.get("last_num_ft",-1) != -1:
            optimizer = optimizer_fn(parameters[-self.args.last_num_ft:], # remember to add the parameters of your model decoder into this line!! 
                                        **args)
        #scheduler = cosine_lr(optimizer, self.args.get("learning_rate",1e-5), self.args.get("warmup",1000), self.args.get("total_steps",100000))
        return optimizer#([optimizer],[scheduler])


    def init_batch_uniform(self, X,eps):
        
        delta=  torch.stack([torch.zeros_like(X,device=self.device,).uniform_(-v, v) for v in eps])
        delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
        delta.requires_grad = True
        return delta
    
    def init_batch_normal(self, X,eps):
            delta=torch.zeros_like(X,device=self.device)
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            delta = clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True
            return delta
    @torch.enable_grad()
    @torch.inference_mode(False)
    def clamp_batch_inf(self,d,alpha,g,eps):
        return torch.clamp(d.clone() + alpha.clone().view(-1,1,1,1,1,1) * torch.sign(g.clone()), min=-eps.clone().view(1,-1,1,1,1,1)
                           , max=eps.clone().view(1,-1,1,1,1,1))
    @torch.enable_grad()
    @torch.inference_mode(False)
    def clamp_batch_2(self,d,alpha,g,eps):
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        d = (d + scaled_g * alpha.view(-1,1,1,1,1,1)).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=eps.view(1,-1,1,1,1,1)).view_as(d)
        return d
    
  
    #insert function decorator to ensure this ALWAys has grad
    @torch.enable_grad()
    @torch.inference_mode(False)
    def attack_batch_pgd(self,  X, target, text_tokens, alpha, attack_iters,epsilon, restarts=1, early_stop=True):
        delta=self.init_batch_delta(X,epsilon).unsqueeze(0).repeat(alpha.shape[0],1,1,1,1,1)#make epsilon stacks of delta and repeat for each alpha so we have shape alpha,epsilon,B,3,224,224
        # print("requires grad on delta? {} {}".format(delta.requires_grad,delta.retain_grad()))
        # losses=[]
        delta.retain_grad()
        return_dict={}
        X=X.clone().detach()
        text_tokens=text_tokens.clone().detach()
        with torch.no_grad():
            scale_text_embed=self.make_labels(X,text_tokens)
        
        scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        for iter_count in range(max(attack_iters)):
            new_images = torch.add(X, delta)
            prompted_images = torch.div(torch.sub(new_images, self.mu_img.clone()), self.std_img.clone()) #normalize(new_images) but preserves grad

            img_embed=self.model.encode_image(prompted_images.flatten(0,-4))
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            
            # print("requires grad on scale_text_embed? {} shape : {}".format(scale_text_embed.requires_grad,scale_text_embed.shape))

            output = img_embed @ scale_text_embed.t()
            output = output.view(alpha.size(0),epsilon.size(0),X.size(0),-1)
            loss = self.criterion(output.permute(-2,-1,0,1), torch.arange(X.shape[0], device=self.device).unsqueeze(-1).unsqueeze(-1).repeat(1,alpha.size(0),epsilon.size(0)))
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :,:,:]
            g = grad[:, :, :, :,:,:]
            x = X[:, :, :, :]
            d=self.batch_clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :,:,:] = d
            delta.grad.zero_()
            if iter_count+1 in attack_iters:
                return_dict.update({iter_count:(X+delta,text_tokens)})
        return return_dict
    
    @torch.enable_grad()
    @torch.inference_mode(False)
    def attack_batch_CW(self, X, target, text_tokens,alpha, attack_iters,epsilon, restarts=1, early_stop=True):
        delta=self.init_batch_delta(X,epsilon).unsqueeze(0).repeat(alpha.shape[0],1,1,1,1,1)#make epsilon stacks of delta and repeat for each alpha so we have shape alpha,epsilon,B,3,224,224
        delta.retain_grad()
        return_dict={}
        X=X.clone().detach()
        text_tokens=text_tokens.clone().detach()
        for iter_count in range(max(attack_iters)):
            new_images = torch.add(X, delta)
            prompted_images = torch.div(torch.sub(new_images, self.mu_img.clone()), self.std_img.clone())
            # prompt_token = self.add_prompter()
            img_embed=self.model.encode_image(prompted_images.flatten(0,-4))
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed=self.make_labels(X,text_tokens)
            scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            # print("requires grad on scale_text_embed? {} shape : {}".format(scale_text_embed.requires_grad,scale_text_embed.shape))

            output = img_embed @ scale_text_embed.t()
            output = output.view(alpha.size(0),epsilon.size(0),X.size(0),-1)
            label_mask = one_hot_embedding(torch.arange(X.shape(0),device=X.device), output.size(1)).unsqueeze(0).unsqueeze(0).repeat(alpha.size(0),epsilon.size(0),1,1)
            correct_logit = torch.sum(label_mask * output, dim=1)
            wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
            loss = - torch.sum(torch.sum(F.relu(correct_logit - wrong_logit + 50),dim=-1),dim=-1)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :,:,:]
            g = grad[:, :, :, :,:,:]
            x = X[:, :, :, :]
            d=self.batch_clamp(d,alpha,g,epsilon)
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :,:,:] = d
            delta.grad.zero_()
            if iter_count+1 in attack_iters:
                return_dict.update({iter_count:(X+delta,text_tokens)})
        return return_dict
    
    @torch.enable_grad()
    @torch.inference_mode(False)
    def attack_batch_text_pgd(self, X, target, text_tokens,alpha, attack_iters,epsilon, restarts=1, early_stop=True):

        delta=self.init_batch_delta(text_tokens,epsilon).unsqueeze(0).repeat(alpha.shape[0],1,1,1)#make epsilon stacks of delta and repeat for each alpha so we have shape alpha,epsilon,B,77
        #instead we should use the hidden shape after the clip token emb, and then return this to the original shape later. 
        self.insert_eval_model_hook() # TODO: insert hooks to save the features of whichever layer we want to use for the text loss
        self.make_labels(X,text_tokens).detach() #do this with hooks
        clean_features=self.test_text_features
        for _ in range(attack_iters):

            #step 1: modify text tokens
            #step 2: pass through CLIP model module that saves features,
            #step 3: Loss= cosine similarity of clean features to dirty features. 
            #step 4: now consider loss. 
            text_tokens+=delta
            


            img_embed=self.model.encode_image(X)
            #ensure self.model has text hooks 
            self.insert_eval_model_hook()
            scale_text_embed=self.make_labels(X,text_tokens)
            features=self.test_text_features
            #do Loss between each layer
            text_loss=torch.zeros((X.shape[0],X.shape[0]),device=self.device)
            for layer in features.keys():
                itemA=features[layer]
                itemB=clean_features[layer]
                itemA=itemA/itemA.norm(dim=-1, keepdim=True)
                itemB=itemB/itemB.norm(dim=-1, keepdim=True)
                similarities= itemA@itemB.T  # should be B,B in shape, 
                text_loss+=self.CETextLoss(similarities)
            self.log("text_loss",text_loss,prog_bar=False, logger=True, sync_dist=False)

            #step 5: backpropagate, making noise closer to clean features
            text_loss.backward()
            #step 6: remove hooks and zero grad
            self.remove_text_model_hook()
            delta.grad.zero_()


            #step 7: now do attack as normal
            d = delta

            #I want to find a way to maximize the loss while minimizing text loss

            img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
            scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
            logits_per_image = img_embed_norm @ scale_text_embed_norm.t()
            logits_per_text = scale_text_embed_norm @ img_embed_norm.t()
            # logits_per_text, img_embed, scale_text_embed


            loss = self.criterion(logits_per_text, torch.arange(X.size(0), device=self.device))
            loss.backward()
            self.log("attack_loss",loss,prog_bar=False, logger=True, sync_dist=False)
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]
            d=self.clamp(d.to(self.device),alpha.to(self.device),g.to(self.device),epsilon.to(self.device))
            d = clamp(d, self.lower_limit - x, self.upper_limit - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        return X,text_tokens+delta
    
    def on_test_start(self):
        self.mu_img = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(self.device)
        self.std_img = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(self.device)
        #to be thread safe we should create queues insead of lists.#
        self.test_epoch_end_called=False
        self.test_cleanresults=defaultdict(queue.Queue)
        self.test_attackedresults=defaultdict(queue.Queue)
        self.test_data_loader_count = len(self.trainer.datamodule.test_dataloader())
        if self.args.get("test_attack_type","pgd")=="pgd":
            self.testattack=self.attack_batch_pgd
        elif self.args.get("test_attack_type","pgd")=="CW":
            self.testattack= self.attack_CW
        elif self.args.get("test_attack_type","pgd")=="text":
            self.testattack= self.attack_text_pgd
        elif self.args.get("test_attack_type","pgd")=="autoattack":
            self.testattack=self.autoattack
        elif self.args.get("test_attack_type","pgd")=="Noattack":
            self.testattack=self.no_attack
        else:
            raise ValueError 
        #enable grad through our model to allow the attacks to work.
        self.model.eval()
        torch.set_grad_enabled(True)

        self.model_ori.eval()
        self.test_alphas = torch.tensor([1/255, 2/255, 4/255,8/255],device=self.device)
        self.test_epsilons = torch.tensor([1/255, 2/255, 4/255],device=self.device)
        self.test_numsteps = torch.tensor([5, 10],device=self.device)
        #instead of saving the results to memory, were going to save them to disk.
        #note : if using multiple nodes, this will need to be a shared file system, or a database... or revert to saving to memory, and praying you have enough!!
        self.save_result_worker_thread=threading.Thread(target=self.save_result_worker)
        self.save_result_worker_thread.start()
        #model is put in inference mode, we're going to change it back:
        self.model.train()
        torch.set_grad_enabled(True)
        

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx,  dataloader_idx=0, *args, **kwargs):
        images, target,text = batch
        text=text.clone().squeeze(1)
        images=images.clone()
        target=target.clone()
        img_embed=self.model.encode_image(images).detach()
        scale_text_embed=self.make_labels(images,text).detach()
        img_embed_norm = img_embed / img_embed.norm(dim=-1, keepdim=True)
        scale_text_embed_norm = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True)
        output_prompt = img_embed_norm @ scale_text_embed_norm.t()        

        self.test_cleanresults[dataloader_idx].put({"logits":img_embed.detach(), "textlabels":target}) #using target like this is fine because each dataloader is tested and logged independently.
        loss = self.criterion(output_prompt, torch.arange(images.size(0), device=self.device)).detach()

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, torch.arange(images.shape[0],device=images.device), topk=(1,))
        self.log('test_clean_batch_loss', loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_clean_batch_acc', acc1[0].item(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)


        return_dict = self.testattack(images, target, text, self.test_alphas, self.test_numsteps, self.test_epsilons)
        #this returns a dict of "steps" entris, each entry has a tensor of shape Alphas, Epsilons, B, 3, 224, 224
        for Attack_step, result_data in return_dict.items():
            
            attacked_images, attacked_text = result_data
            img_embed_dirty = self.model.encode_image(attacked_images.flatten(0,-4)).detach()
            scale_text_embed_dirty = self.make_labels(attacked_images.flatten(0,-4),attacked_text.flatten(0,-2)).detach()
            img_embed_norm_dirty = img_embed_dirty / img_embed_dirty.norm(dim=-1, keepdim=True)
            scale_text_embed_norm_dirty = scale_text_embed_dirty / scale_text_embed_dirty.norm(dim=-1, keepdim=True)
            output_prompt_adv = img_embed_norm_dirty @ scale_text_embed_norm_dirty.t()
            #at this point we have an array thats [alpha x epsilon x B, B]
            #we should reshape it to [alpha, epsilon, B, B]
            output_prompt_adv = output_prompt_adv.view(self.test_alphas.size(0),self.test_epsilons.size(0),images.size(0),-1)
            img_embed_dirty = img_embed_dirty.view(self.test_alphas.size(0),self.test_epsilons.size(0),images.size(0),-1)
              
            loss = self.test_criterion(output_prompt_adv.permute(-2,-1,0,1), torch.arange(images.size(0),device=images.device).unsqueeze(-1).unsqueeze(-1).repeat(1,self.test_alphas.size(0),self.test_epsilons.size(0))) #shoudl be torch arange(images.size(0), device=self.device)
            # print(loss.shape)
            loss=loss.permute(1,2,0).view(self.test_epsilons.size(0),self.test_alphas.size(0),images.size(0))
            for alpha in range(self.test_alphas.size(0)):
                for epsilon in range(self.test_epsilons.size(0)):
                        self.log(f'test_dirty_batch_loss_alpha_{self.test_alphas[alpha]}_epsilon_{self.test_epsilons[epsilon]}_numsteps_{Attack_step}', loss[epsilon,alpha].mean(), on_epoch=True, logger=True, sync_dist=True)
                        acc1 = accuracy(output_prompt_adv[alpha,epsilon], torch.arange(images.size(0),device=images.device), topk=(1,))
                        self.log(f'test_dirty_batch_acc_alpha_{self.test_alphas[alpha]}_epsilon_{self.test_epsilons[epsilon]}_numsteps_{Attack_step}', acc1[0].item(), on_epoch=True, logger=True, sync_dist=True)
                        self.test_attackedresults[dataloader_idx].put({"logits": img_embed_dirty[alpha,epsilon], "textlabels": target, "alpha": self.test_alphas[alpha].repeat(target.shape[0]), "epsilon": self.test_epsilons[epsilon].repeat(target.shape[0]), "step": torch.tensor(Attack_step).repeat(target.shape[0])})  
               
        return loss
                    
    
    
    def on_test_end(self):
        print("Test epoch end called")
        self.test_epoch_end_called=True
        if hasattr(self,"save_result_worker_thread"):
            self.save_result_worker_thread.join()

         
        #We need to modify the following code to sort by alpha, epsilon, step and then run the linear probes.
        #make a new dict with id as key as compound key of alpha, epsilon, step and then logits:labels as value


        if not hasattr(self,"Cleanclassifier"):
            self.Cleanclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)

        if not hasattr(self,"Dirtyclassifier"):
            self.Dirtyclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)
        if not hasattr(self,"generalclassifier"):
            self.generalclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=0, n_jobs=-1)
         
        #read in all files and begin processing them
        path=self.args.get("output_dir","./results")
        filenames=os.listdir(path)
        version=self.version


        for DataLoader_idx in range(self.test_data_loader_count):
            dirtyfilenames=filter(lambda x: x.startswith("dirtyresults_{}".format(version)),filenames)
            cleanfilenames=filter(lambda x: x.startswith("cleanresults_{}".format(version)),filenames)
            #split each name by _ and get the dataset index
            clean_files=list(filter(lambda x: int(list(x.split("_"))[-2]) == DataLoader_idx,cleanfilenames))
                                    
            dirty_files=list(filter(lambda x: int(list(x.split("_"))[-2]) == DataLoader_idx,dirtyfilenames))
        #                dirty_files=list(filter(lambda x: str(dataset_idx)+"_pt" in x,list(dirtyfilenames)))
            if len(clean_files) == 0 or len(dirty_files) == 0:
                print("No results for dataset {}".format(DataLoader_idx))
                print("Clean files: ",clean_files)
                print("Dirty files: ",dirty_files)
                print("Clean files: ",list(cleanfilenames))

                continue
            
            GoodLabels=[]
            GoodLogits=[]
            for file in clean_files:#
                if not os.path.exists(os.path.join(path,file)):
                    print("File {} does not exist".format(file))
                    continue

                with open(os.path.join(path,file), 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    GoodLabels.append(data["labels"])
                    GoodLogits.append(data["logits"])
                #delete the file

            GoodLabels=np.concatenate(GoodLabels) if len(GoodLabels) > 1 else GoodLabels[0]
            GoodLogits=np.concatenate(GoodLogits) if len(GoodLogits) > 1 else GoodLogits[0]
            self.Cleanclassifier.fit(GoodLogits, GoodLabels)
            #Log classifier weights and bias using self.logger.experiment.log
            #convert to dict and log

            # print("Clean Classifier Weights Dataset {}".format(DataLoader_idx),self.Cleanclassifier.coef_.tolist())
            # print("Clean Classifier Bias Dataset {}".format(DataLoader_idx),self.Cleanclassifier.intercept_.tolist())
            #save these to disk
            np.savez(os.path.join(path,"CleanClassifierWeights{}_{}.npz".format(self.version,DataLoader_idx)),weights=self.Cleanclassifier.coef_,bias=self.Cleanclassifier.intercept_)
            cleanscore=self.Cleanclassifier.score(GoodLogits, GoodLabels)
            BadLabels=[]
            BadLogits=[]
            alpha_eps_step_dict = defaultdict(list)
            for file in list(dirty_files):
                if not os.path.exists(os.path.join(path,file)):
                    print("File {} does not exist".format(file))
                    continue
                
                with open(os.path.join(path,file), 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    alphas=data["alphas"]
                    epsilons=data["epsilons"]
                    steps=data["numsteps"]
                    #stack the data
                    keys=np.stack([alphas,epsilons,steps],axis=1)
                    #shape is B,3
                    
                    unique_keys=np.unique(keys,axis=0)
                    for key in unique_keys:
                        key=tuple(key)
                        if file not in alpha_eps_step_dict[key]:
                            alpha_eps_step_dict[key].append(file)
                        #we do this so we can run one test at a time and not store all the data in memory
                #delete the file

            for key, val in alpha_eps_step_dict.items():
                BadLabels=[]
                BadLogits=[]
                a,e,s=key
                for file in val:
                    with open(os.path.join(path,file), 'rb') as f:
                        data = np.load(f, allow_pickle=True)
                        logits,labels=data["logits"],data["labels"]
                        alphas= data["alphas"]
                        epsilons= data["epsilons"]
                        steps= data["numsteps"]                       
                        mask= (alphas==a) & (epsilons==e) & (steps==s)
                        BadLabels.append(labels[mask])
                        BadLogits.append(logits[mask])
                        
                BadLabels=np.concatenate(BadLabels) if len(BadLabels) > 1 else BadLabels[0]
                BadLogits=np.concatenate(BadLogits) if len(BadLogits) > 1 else BadLogits[0]
                self.Dirtyclassifier.fit(BadLogits, BadLabels)
                #Log classifier weights and bias
                # self.logger.experiment.log("Dirty Classifier Weights Dataset {}".format(DataLoader_idx),self.Dirtyclassifier.coef_, sync_dist=True)
                # self.logger.experiment.log("Dirty Classifier Bias Dataset {}".format(DataLoader_idx), self.Dirtyclassifier.intercept_, sync_dist=True)
                self.generalclassifier.fit(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels]))
                # self.logger.experiment.log("General Classifier Weights Dataset {}".format(DataLoader_idx),self.generalclassifier.coef_, sync_dist=True)
                # self.logger.experiment.log("General Classifier Bias Dataset {}".format(DataLoader_idx), self.generalclassifier.intercept_, sync_dist=True)
                np.savez(os.path.join(path,"DirtyClassifierWeights{}_{}_{}_{}_{}.npz".format(self.version,DataLoader_idx,a,e,s)),weights=self.Dirtyclassifier.coef_,bias=self.Dirtyclassifier.intercept_)
                np.savez(os.path.join(path,"GeneralClassifierWeights{}_{}_{}_{}_{}.npz".format(self.version,DataLoader_idx,a,e,s)),weights=self.generalclassifier.coef_,bias=self.generalclassifier.intercept_)
                self.logger.experiment.log( "Test Clean Classifier on Dirty Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.Cleanclassifier.score(BadLogits, BadLabels), sync_dist=True)
                self.logger.experiment.log( "Test Dirty Classifier on Clean Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.Dirtyclassifier.score(GoodLogits, GoodLabels), sync_dist=True)
                self.logger.experiment.log( "Test Clean Classifier on Clean Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),cleanscore, sync_dist=True)
                self.logger.experiment.log( "Test Dirty Classifier on Dirty Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.Dirtyclassifier.score(BadLogits, BadLabels), sync_dist=True)
                self.logger.experiment.log( "Test General Classifier on Dirty Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.generalclassifier.score(BadLogits, BadLabels), sync_dist=True)
                self.logger.experiment.log( "Test General Classifier on Clean Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.generalclassifier.score(GoodLogits, GoodLabels), sync_dist=True)
                self.logger.experiment.log( "Test General Classifier on All Features on dataset {} alpha {} epsilon {} step {}".format(DataLoader_idx,key[0],key[1],key[2]),self.generalclassifier.score(np.concatenate([GoodLogits,BadLogits]), np.concatenate([GoodLabels,BadLabels])), sync_dist=True)

            #delete the files
            for file in list(dirty_files):
                print("Deleting file: ",os.path.join(path,file))
                os.remove(os.path.join(path,file))
            for file in list(clean_files):
                print("Deleting file: ",os.path.join(path,file))
                os.remove(os.path.join(path,file))


        # del self.test_cleanresults
        # del self.test_attackedresults
        if hasattr(self,"save_result_worker_thread"):
            self.save_result_worker_thread.join()
            del self.save_result_worker_thread

    def save_result_worker(self):
        #in this function, we will save the results to disk.
        #we will constantly check self.test_cleanresults and self.test_attackedresults
        #if there are results, we will save them with a filename thats unique to the run, and then has dataset_idx, alpha, epsilon, step 
        #inside the file we'll save the first 1000 pairs of vectors and labels.
        # the most efficient way to store tensors is to save them as numpy arrays, so we'll do that. 
        cleanidx=0
        dirtyidx=0
        path=os.path.join(self.args.get("output_dir","./results"))
        os.makedirs(path,exist_ok=True)
        #set version as a string of all the args
        version=self.version
        threshold=50
        EmptyCount=0
        while not self.test_epoch_end_called:

            time.sleep(120)
            clear=False
            for dataset_idx in range(self.test_data_loader_count):
                # print("Saving results for dataset {}".format(dataset_idx))
                filename="results_{}_{}_pt".format(version,dataset_idx)
                clear=True
                if not self.test_cleanresults[dataset_idx].empty():
                    clear=False
                    #take the first 1000 results and save them to disk.
                    #take first n results and save them to disk, remove them from the list
                    clean_results=[self.test_cleanresults[dataset_idx].get(False) for _ in range(min(self.test_cleanresults[dataset_idx].qsize(),threshold))]
                    #
                    clean_filename="clean"+filename+str(cleanidx)
                    cleanPath=os.path.join(path,clean_filename)
                    # print("Saving clean results {} to {}".format(len(clean_results),cleanPath))
                    logits=torch.cat([val["logits"] for val in clean_results],dim=0).cpu().numpy() if threshold > 1 else clean_results[0]["logits"].cpu().numpy()
                    labels=torch.cat([val["textlabels"] for val in clean_results],dim=0).cpu().numpy() if threshold > 1 else clean_results[0]["textlabels"].cpu().numpy()
                    np.savez(cleanPath,logits=logits,labels=labels)
                    # print("Saved clean results to {}".format(cleanPath))
                    cleanidx+=1
                if not self.test_attackedresults[dataset_idx].empty():
                    clear=False
                    dirty_filename="dirty"+filename+str(dirtyidx)
                    dirtyPath=os.path.join(path,dirty_filename)
                    dirty_results=[self.test_attackedresults[dataset_idx].get(False) for _ in range(min(self.test_attackedresults[dataset_idx].qsize(),threshold))]
                    # print("Saving dirty results {} to {}".format(len(dirty_results),dirtyPath))
                    logits=torch.cat([val["logits"] for val in dirty_results],dim=0).cpu().numpy() if threshold > 1 else dirty_results[0]["logits"].cpu().numpy()
                    labels=torch.cat([val["textlabels"] for val in dirty_results],dim=0).cpu().numpy() if threshold > 1 else dirty_results[0]["textlabels"].cpu().numpy()
                    alpha=torch.cat([val["alpha"] for val in dirty_results],dim=0).cpu().numpy() if threshold > 1 else dirty_results[0]["alpha"].cpu().numpy()
                    epsilons=torch.cat([val["epsilon"] for val in dirty_results],dim=0).cpu().numpy() if threshold > 1 else dirty_results[0]["epsilon"].cpu().numpy()
                    numsteps=torch.cat([val["step"] for val in dirty_results],dim=0).cpu().numpy() if threshold > 1 else dirty_results[0]["step"].cpu().numpy()

                    assert logits.shape[0] == labels.shape[0] == alpha.shape[0] == epsilons.shape[0] == numsteps.shape[0]
                    np.savez(dirtyPath,logits=logits,labels=labels,alphas=alpha,epsilons=epsilons,numsteps=numsteps)
                    # print("Saved dirty results to {}".format(dirtyPath))
                    dirtyidx+=1
            if clear:
                EmptyCount+=1
                time.sleep(300)

                # print("Saved results for dataset {}".format(dataset_idx))
            # if all([self.test_cleanresults[idx].empty() for idx in range(self.test_data_loader_count)]) and all([self.test_attackedresults[idx].empty() for idx in range(self.test_data_loader_count)]):
                #if test_epoch_end has been called, we can exit the loop
            # else:
            #     threshold= int(self.args.get("test_batch_size",8)/2)
        
        print("Exiting save results worker")
